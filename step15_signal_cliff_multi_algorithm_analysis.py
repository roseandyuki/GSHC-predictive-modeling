# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 15: Signal Cliff Multi-Algorithm Analysis ---
# 
# Purpose: Test the universality of the "Signal Cliff" phenomenon across 
# different machine learning algorithms
# 
# Scientific Question: Is the observed "Signal Cliff" a universal phenomenon 
# in low signal-to-noise environments, or is it specific to LASSO regularization?
# 
# Hypothesis: Different algorithms will show different cliff patterns:
# - LASSO: Sharp cliff due to L1 regularization
# - XGBoost: More gradual slope due to robustness to noise
# - Random Forest: Intermediate behavior
# - Logistic Regression (no regularization): Different cliff position
# 
# Method: 
# 1. Replicate the contamination experiment with multiple algorithms
# 2. For each contamination level (0%, 5%, 10%, ..., 50%), fit all models
# 3. Extract "signal strength" proxy for each algorithm
# 4. Compare signal decay curves across algorithms
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# --- Configuration ---
# =============================================================================

# Data file paths (same as previous steps)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

DATA_FILES = {
    'shhs1': os.path.join(SCRIPT_DIR, 'shhs1-dataset-0.21.0.csv'),
    'shhs2': os.path.join(SCRIPT_DIR, 'shhs2-dataset-0.21.0.csv')
}

# Variable mapping (corrected based on step14)
VAR_MAP = {
    'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1',
    'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all',
    'n3_percent': 'times34p', 'n1_percent': 'timest1p', 'n2_percent': 'timest2p',
    'rem_percent': 'timeremp', 'sleep_efficiency': 'slpeffp', 'waso': 'waso',
    'rdi': 'rdi4p', 'min_spo2': 'minsat', 'avg_spo2': 'avgsat',
    'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'
}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# Features and SDB variables
HIGH_QUALITY_FEATURES = [
    'bmi_v1', 'sbp_v1', 'dbp_v1', 'age_v1', 'gender', 'ess_v1',
    'arousal_index', 'n3_percent', 'n1_percent', 'n2_percent', 'rem_percent',
    'sleep_efficiency', 'waso', 'rdi', 'min_spo2', 'avg_spo2'
]

SDB_VARIABLES = ['rdi', 'min_spo2', 'avg_spo2', 'arousal_index', 'sleep_efficiency', 'waso']

# Contamination levels to test
CONTAMINATION_LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50]

# =============================================================================
# --- Algorithm Configurations ---
# =============================================================================

def get_algorithm_configs():
    """Define all algorithms to test"""
    return {
        'LASSO': {
            'model': LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 50), cv=5, penalty='l1', 
                solver='liblinear', scoring='roc_auc', 
                random_state=42, class_weight='balanced'
            ),
            'signal_extractor': 'coefficients',
            'color': '#1f77b4',
            'linestyle': '-'
        },
        'Logistic_NoReg': {
            'model': LogisticRegression(
                penalty=None, random_state=42,  # 修复：'none' -> None
                class_weight='balanced', max_iter=1000
            ),
            'signal_extractor': 'coefficients',
            'color': '#ff7f0e',
            'linestyle': '--'
        },
        'Random_Forest': {
            'model': RandomForestClassifier(
                n_estimators=100, random_state=42, 
                class_weight='balanced', n_jobs=-1
            ),
            'signal_extractor': 'feature_importance',
            'color': '#2ca02c',
            'linestyle': '-.'
        },
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=100, random_state=42, 
                eval_metric='logloss', n_jobs=-1
            ),
            'signal_extractor': 'shap_values',
            'color': '#d62728',
            'linestyle': ':'
        }
    }

# =============================================================================
# --- Utility Functions ---
# =============================================================================

def load_and_map_data(filepaths, id_col='nsrrid'):
    """Load and merge SHHS datasets"""
    try:
        df1 = pd.read_csv(filepaths['shhs1'], encoding='ISO-8859-1', low_memory=False)
        df2 = pd.read_csv(filepaths['shhs2'], encoding='ISO-8859-1', low_memory=False)
        merged_df = pd.merge(df1, df2, on=id_col, how='left', suffixes=('', '_dup'))
        return merged_df.rename(columns=RENAME_MAP)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")

def has_transitioned(row):
    """Check if participant transitioned to unhealthy state"""
    if pd.isna(row['bmi_v2']) or pd.isna(row['sbp_v2']) or pd.isna(row['dbp_v2']):
        return np.nan
    return 1 if any([row['bmi_v2'] >= 25, row['sbp_v2'] >= 120, row['dbp_v2'] >= 80]) else 0

def create_gshc(df):
    """Create Gold-Standard Healthy Cohort"""
    gshc_criteria = (
        (df['bmi_v1'] < 25) & 
        (df['sbp_v1'] < 120) & 
        (df['dbp_v1'] < 80)
    )
    return df[gshc_criteria].copy()

def extract_signal_strength(model, X, algorithm_config, feature_names):
    """Extract signal strength proxy for different algorithms"""
    extractor_type = algorithm_config['signal_extractor']
    
    if extractor_type == 'coefficients':
        # For linear models
        coeffs = model.coef_[0] if hasattr(model, 'coef_') else model.named_steps['lasso_cv'].coef_[0]
        signal_dict = dict(zip(feature_names, np.abs(coeffs)))
        
    elif extractor_type == 'feature_importance':
        # For tree-based models
        importances = model.feature_importances_
        signal_dict = dict(zip(feature_names, importances))
        
    elif extractor_type == 'shap_values':
        # For XGBoost using SHAP
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            signal_dict = dict(zip(feature_names, mean_abs_shap))
        except:
            # Fallback to feature importance if SHAP fails
            importances = model.feature_importances_
            signal_dict = dict(zip(feature_names, importances))
    
    return signal_dict

def run_contamination_experiment(gshc_df, full_cohort_df, contamination_level, algorithm_configs):
    """Run contamination experiment for all algorithms at given contamination level"""
    
    # Calculate number of contaminating samples
    n_gshc = len(gshc_df)
    n_contamination = int(n_gshc * contamination_level / 100)
    
    # Create contaminated cohort
    if n_contamination > 0:
        # Sample contaminating participants from full cohort (excluding GSHC)
        non_gshc = full_cohort_df[~full_cohort_df.index.isin(gshc_df.index)]
        contamination_sample = non_gshc.sample(n=n_contamination, random_state=42)
        
        # Combine GSHC with contamination
        contaminated_cohort = pd.concat([gshc_df, contamination_sample], ignore_index=True)
    else:
        contaminated_cohort = gshc_df.copy()
    
    # Prepare features and outcome
    X = contaminated_cohort[HIGH_QUALITY_FEATURES]
    y = contaminated_cohort['Y_Transition']
    
    # Results storage
    results = {}
    
    # Test each algorithm
    for alg_name, alg_config in algorithm_configs.items():
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('imputer', IterativeImputer(max_iter=10, random_state=42)),
                ('scaler', StandardScaler()),
                ('model', alg_config['model'])
            ])
            
            # Fit model
            pipeline.fit(X, y)
            
            # Extract signal strength
            X_processed = pipeline.named_steps['scaler'].transform(
                pipeline.named_steps['imputer'].transform(X)
            )
            signal_strengths = extract_signal_strength(
                pipeline.named_steps['model'], 
                X_processed, 
                alg_config, 
                HIGH_QUALITY_FEATURES
            )
            
            # Calculate AUC
            auc_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
            mean_auc = np.mean(auc_scores)
            
            results[alg_name] = {
                'signal_strengths': signal_strengths,
                'auc': mean_auc,
                'sdb_signal_sum': sum([signal_strengths.get(var, 0) for var in SDB_VARIABLES])
            }
            
        except Exception as e:
            print(f"Error with {alg_name} at {contamination_level}% contamination: {e}")
            results[alg_name] = {
                'signal_strengths': {var: 0 for var in HIGH_QUALITY_FEATURES},
                'auc': 0.5,
                'sdb_signal_sum': 0
            }
    
    return results

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_signal_cliff_comparison(all_results, algorithm_configs, output_dir):
    """Create the main signal cliff comparison plot"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Total SDB Signal Strength
    for alg_name, alg_config in algorithm_configs.items():
        contamination_levels = []
        sdb_signals = []

        for cont_level in CONTAMINATION_LEVELS:
            if cont_level in all_results and alg_name in all_results[cont_level]:
                contamination_levels.append(cont_level)
                sdb_signals.append(all_results[cont_level][alg_name]['sdb_signal_sum'])

        ax1.plot(contamination_levels, sdb_signals,
                color=alg_config['color'],
                linestyle=alg_config['linestyle'],
                marker='o', linewidth=2.5, markersize=6,
                label=alg_name.replace('_', ' '))

    ax1.set_xlabel('Contamination Level (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total SDB Signal Strength', fontsize=12, fontweight='bold')
    ax1.set_title('Signal Cliff Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 52)

    # Plot 2: Individual SDB Variables (LASSO vs XGBoost)
    sdb_vars_to_plot = ['rdi', 'min_spo2', 'arousal_index']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, sdb_var in enumerate(sdb_vars_to_plot):
        lasso_signals = []
        xgb_signals = []

        for cont_level in CONTAMINATION_LEVELS:
            if cont_level in all_results:
                lasso_sig = all_results[cont_level].get('LASSO', {}).get('signal_strengths', {}).get(sdb_var, 0)
                xgb_sig = all_results[cont_level].get('XGBoost', {}).get('signal_strengths', {}).get(sdb_var, 0)
                lasso_signals.append(lasso_sig)
                xgb_signals.append(xgb_sig)

        ax2.plot(CONTAMINATION_LEVELS, lasso_signals,
                color=colors[i], linestyle='-', marker='o',
                label=f'{sdb_var} (LASSO)', alpha=0.8)
        ax2.plot(CONTAMINATION_LEVELS, xgb_signals,
                color=colors[i], linestyle='--', marker='s',
                label=f'{sdb_var} (XGBoost)', alpha=0.8)

    ax2.set_xlabel('Contamination Level (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Individual Variable Signal Strength', fontsize=12, fontweight='bold')
    ax2.set_title('Key SDB Variables: LASSO vs XGBoost', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 52)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_cliff_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Signal cliff comparison plot saved")

def create_individual_sdb_analysis(all_results, algorithm_configs, output_dir):
    """Create detailed analysis of individual SDB variables"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, sdb_var in enumerate(SDB_VARIABLES):
        ax = axes[idx]

        for alg_name, alg_config in algorithm_configs.items():
            signals = []

            for cont_level in CONTAMINATION_LEVELS:
                if cont_level in all_results and alg_name in all_results[cont_level]:
                    signal = all_results[cont_level][alg_name]['signal_strengths'].get(sdb_var, 0)
                    signals.append(signal)
                else:
                    signals.append(0)

            ax.plot(CONTAMINATION_LEVELS, signals,
                   color=alg_config['color'],
                   linestyle=alg_config['linestyle'],
                   marker='o', linewidth=2, markersize=5,
                   label=alg_name.replace('_', ' '))

        ax.set_title(f'{sdb_var.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Contamination Level (%)', fontsize=10)
        ax.set_ylabel('Signal Strength', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 52)

        if idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=9)

    plt.suptitle('Individual SDB Variable Analysis Across Algorithms',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_sdb_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Individual SDB analysis plot saved")

def create_auc_comparison(all_results, algorithm_configs, output_dir):
    """Create AUC performance comparison"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for alg_name, alg_config in algorithm_configs.items():
        aucs = []

        for cont_level in CONTAMINATION_LEVELS:
            if cont_level in all_results and alg_name in all_results[cont_level]:
                auc = all_results[cont_level][alg_name]['auc']
                aucs.append(auc)
            else:
                aucs.append(0.5)

        ax.plot(CONTAMINATION_LEVELS, aucs,
               color=alg_config['color'],
               linestyle=alg_config['linestyle'],
               marker='o', linewidth=2.5, markersize=6,
               label=alg_name.replace('_', ' '))

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random Chance')
    ax.set_xlabel('Contamination Level (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation AUC', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs Contamination Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 52)
    ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ AUC comparison plot saved")

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Signal Cliff Multi-Algorithm Analysis ===")

    # Setup output directory
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_signal_cliff_analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {OUTPUT_DIR}")

    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    full_cohort_df = base_df.dropna(subset=['Y_Transition']).copy()
    gshc_df = create_gshc(full_cohort_df)

    print(f"Full cohort size: {len(full_cohort_df)}")
    print(f"GSHC size: {len(gshc_df)}")

    # Get algorithm configurations
    algorithm_configs = get_algorithm_configs()
    print(f"Testing algorithms: {list(algorithm_configs.keys())}")

    # Run experiments
    print(f"\n--- Running contamination experiments ---")
    all_results = {}

    for contamination_level in tqdm(CONTAMINATION_LEVELS, desc="Contamination levels"):
        print(f"\nTesting {contamination_level}% contamination...")
        results = run_contamination_experiment(
            gshc_df, full_cohort_df, contamination_level, algorithm_configs
        )
        all_results[contamination_level] = results

    # Save raw results
    results_file = os.path.join(OUTPUT_DIR, 'signal_cliff_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Raw results saved to: {results_file}")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    create_signal_cliff_comparison(all_results, algorithm_configs, OUTPUT_DIR)
    create_individual_sdb_analysis(all_results, algorithm_configs, OUTPUT_DIR)
    create_auc_comparison(all_results, algorithm_configs, OUTPUT_DIR)

    print("\n--- Analysis completed! ---")
    print(f"All results saved to: {OUTPUT_DIR}")

    # Print summary
    print("\n--- Summary of Key Findings ---")
    for contamination_level in [0, 15, 30]:
        print(f"\nAt {contamination_level}% contamination:")
        if contamination_level in all_results:
            for alg_name in algorithm_configs.keys():
                if alg_name in all_results[contamination_level]:
                    sdb_signal = all_results[contamination_level][alg_name]['sdb_signal_sum']
                    auc = all_results[contamination_level][alg_name]['auc']
                    print(f"  {alg_name}: SDB Signal = {sdb_signal:.4f}, AUC = {auc:.3f}")

    print("\n🎯 Signal Cliff Analysis Complete!")
    print("📊 Check the generated plots to see how different algorithms respond to contamination!")
    print("🔬 Key questions to explore:")
    print("   - Does LASSO show the sharpest cliff?")
    print("   - Is XGBoost more robust to contamination?")
    print("   - Where do the cliffs occur for each algorithm?")
    print("   - Which SDB variables are most sensitive to contamination?")
