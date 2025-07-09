# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 16: Wearable Device Simulation Analysis ---
# 
# Purpose: Simulate "consumer-grade wearable device" data quality and assess
# the performance degradation of the GSHC methodological framework
# 
# Scientific Question: How much would the GSHC framework's effectiveness 
# decline if we only had access to consumer-grade wearable device data 
# (with noise, lower precision, missing data)?
# 
# Clinical Significance: Provides quantitative feasibility evidence for 
# early screening using wearable devices
# 
# Method: 
# 1. Start with "gold standard" PSG data from SHHS
# 2. Apply systematic data degradation to simulate wearable device limitations
# 3. Re-run the complete GSHC analysis pipeline
# 4. Quantify performance loss compared to original results
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import shap
from tqdm import tqdm
import pickle
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# --- Configuration ---
# =============================================================================

# Data file paths
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

DATA_FILES = {
    'shhs1': os.path.join(SCRIPT_DIR, 'shhs1-dataset-0.21.0.csv'),
    'shhs2': os.path.join(SCRIPT_DIR, 'shhs2-dataset-0.21.0.csv')
}

# Variable mapping (same as step15)
VAR_MAP = {
    'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1', 
    'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all', 
    'n3_percent': 'times34p', 'n1_percent': 'timest1p', 'n2_percent': 'timest2p', 
    'rem_percent': 'timeremp', 'sleep_efficiency': 'slpeffp', 'waso': 'waso', 
    'rdi': 'rdi4p', 'min_spo2': 'minsat', 'avg_spo2': 'avgsat', 
    'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'
}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

# Features
HIGH_QUALITY_FEATURES = [
    'bmi_v1', 'sbp_v1', 'dbp_v1', 'age_v1', 'gender', 'ess_v1',
    'arousal_index', 'n3_percent', 'n1_percent', 'n2_percent', 'rem_percent',
    'sleep_efficiency', 'waso', 'rdi', 'min_spo2', 'avg_spo2'
]

# Wearable device limitations to simulate
DEGRADATION_SCENARIOS = {
    'Baseline_PSG': {
        'description': 'Original PSG data (Gold Standard)',
        'degradations': []
    },
    'Consumer_Watch': {
        'description': 'Consumer smartwatch (Apple Watch, Fitbit level)',
        'degradations': [
            {'type': 'add_noise', 'variables': ['min_spo2', 'avg_spo2'], 'noise_std': 2.0},
            {'type': 'discretize', 'variable': 'rdi', 'bins': [0, 5, 15, 100], 'labels': [0, 1, 2]},
            {'type': 'remove_variables', 'variables': ['n1_percent', 'n2_percent', 'n3_percent', 'rem_percent']},
            {'type': 'add_missing', 'variables': ['sleep_efficiency', 'waso'], 'missing_rate': 0.15}
        ]
    },
    'Basic_Fitness_Tracker': {
        'description': 'Basic fitness tracker (limited sensors)',
        'degradations': [
            {'type': 'add_noise', 'variables': ['min_spo2', 'avg_spo2'], 'noise_std': 3.5},
            {'type': 'discretize', 'variable': 'rdi', 'bins': [0, 10, 100], 'labels': [0, 1]},
            {'type': 'remove_variables', 'variables': ['n1_percent', 'n2_percent', 'n3_percent', 'rem_percent', 'arousal_index']},
            {'type': 'add_missing', 'variables': ['sleep_efficiency', 'waso', 'min_spo2'], 'missing_rate': 0.25},
            {'type': 'reduce_precision', 'variables': ['sleep_efficiency'], 'precision': 5}  # Round to nearest 5%
        ]
    },
    'Smartphone_Only': {
        'description': 'Smartphone-based sleep tracking (very limited)',
        'degradations': [
            {'type': 'remove_variables', 'variables': ['min_spo2', 'avg_spo2', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent', 'arousal_index']},
            {'type': 'discretize', 'variable': 'rdi', 'bins': [0, 15, 100], 'labels': [0, 1]},
            {'type': 'add_noise', 'variables': ['sleep_efficiency'], 'noise_std': 8.0},
            {'type': 'add_missing', 'variables': ['sleep_efficiency', 'waso'], 'missing_rate': 0.35},
            {'type': 'reduce_precision', 'variables': ['sleep_efficiency', 'waso'], 'precision': 10}
        ]
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

# =============================================================================
# --- Data Degradation Functions ---
# =============================================================================

def apply_degradation(df, degradation_config):
    """Apply a single degradation to the dataframe"""
    df_degraded = df.copy()
    
    if degradation_config['type'] == 'add_noise':
        # Add Gaussian noise to specified variables
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                noise = np.random.normal(0, degradation_config['noise_std'], len(df_degraded))
                df_degraded[var] = df_degraded[var] + noise
                # Ensure realistic bounds (e.g., SpO2 can't be > 100 or < 0)
                if 'spo2' in var.lower():
                    df_degraded[var] = np.clip(df_degraded[var], 0, 100)
    
    elif degradation_config['type'] == 'discretize':
        # Convert continuous variable to categorical
        var = degradation_config['variable']
        if var in df_degraded.columns:
            df_degraded[var] = pd.cut(df_degraded[var], 
                                    bins=degradation_config['bins'], 
                                    labels=degradation_config['labels'], 
                                    include_lowest=True).astype(float)
    
    elif degradation_config['type'] == 'remove_variables':
        # Remove variables that wearable devices can't measure
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                df_degraded = df_degraded.drop(columns=[var])
    
    elif degradation_config['type'] == 'add_missing':
        # Randomly set values to missing (simulate user non-compliance)
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                n_missing = int(len(df_degraded) * degradation_config['missing_rate'])
                missing_indices = np.random.choice(df_degraded.index, n_missing, replace=False)
                df_degraded.loc[missing_indices, var] = np.nan
    
    elif degradation_config['type'] == 'reduce_precision':
        # Reduce measurement precision (round to nearest X)
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                precision = degradation_config['precision']
                df_degraded[var] = np.round(df_degraded[var] / precision) * precision
    
    return df_degraded

def simulate_wearable_data(df, scenario_name, scenario_config):
    """Apply all degradations for a specific wearable device scenario"""
    print(f"Simulating: {scenario_config['description']}")
    
    df_wearable = df.copy()
    
    # Apply each degradation in sequence
    for degradation in scenario_config['degradations']:
        df_wearable = apply_degradation(df_wearable, degradation)
        print(f"  ✓ Applied: {degradation['type']}")
    
    return df_wearable

def get_available_features(df, original_features):
    """Get list of features that are still available after degradation"""
    return [feat for feat in original_features if feat in df.columns]

# =============================================================================
# --- Analysis Functions ---
# =============================================================================

def run_gshc_analysis(df, scenario_name, available_features):
    """Run complete GSHC analysis on degraded data"""
    
    # Create GSHC
    gshc_df = create_gshc(df)
    
    if len(gshc_df) < 50:  # Too few samples
        return {
            'scenario': scenario_name,
            'gshc_size': len(gshc_df),
            'error': 'Insufficient GSHC samples after degradation'
        }
    
    # Prepare features and outcome
    X = gshc_df[available_features]
    y = gshc_df['Y_Transition']
    
    # Check for sufficient outcome variation
    if y.nunique() < 2 or y.sum() < 10:
        return {
            'scenario': scenario_name,
            'gshc_size': len(gshc_df),
            'error': 'Insufficient outcome variation'
        }
    
    try:
        # LASSO pipeline
        pipeline = Pipeline([
            ('imputer', IterativeImputer(max_iter=10, random_state=42)),
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 50), cv=5, penalty='l1',
                solver='liblinear', scoring='roc_auc',
                random_state=42, class_weight='balanced'
            ))
        ])
        
        # Fit model
        pipeline.fit(X, y)
        
        # Get model performance
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        
        # Get LASSO coefficients
        lasso_model = pipeline.named_steps['lasso']
        feature_names = available_features
        coefficients = dict(zip(feature_names, lasso_model.coef_[0]))
        
        # Calculate signal strength metrics
        total_signal = np.sum(np.abs(lasso_model.coef_[0]))
        sdb_variables = ['rdi', 'min_spo2', 'avg_spo2', 'arousal_index', 'sleep_efficiency', 'waso']
        available_sdb = [var for var in sdb_variables if var in available_features]
        sdb_signal = sum([abs(coefficients.get(var, 0)) for var in available_sdb])
        
        # Count selected features (non-zero coefficients)
        selected_features = sum([1 for coef in lasso_model.coef_[0] if abs(coef) > 1e-6])
        
        return {
            'scenario': scenario_name,
            'gshc_size': len(gshc_df),
            'available_features': len(available_features),
            'selected_features': selected_features,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'total_signal_strength': total_signal,
            'sdb_signal_strength': sdb_signal,
            'coefficients': coefficients,
            'available_sdb_vars': available_sdb,
            'feature_names': feature_names
        }
        
    except Exception as e:
        return {
            'scenario': scenario_name,
            'gshc_size': len(gshc_df),
            'error': f'Analysis failed: {str(e)}'
        }

# =============================================================================
# --- Main Execution ---
# =============================================================================

def create_brain_comparison_plot(all_results, output_dir, base_df):
    """
    并排展示PSG基线模型和Smartphone Only模型的SHAP summary plot
    """
    # 1. 获取PSG和Smartphone Only的分析结果
    baseline = all_results.get('Baseline_PSG')
    smartphone = all_results.get('Smartphone_Only')
    if baseline is None or smartphone is None or 'error' in baseline or 'error' in smartphone:
        print("❌ 无法生成大脑对比图：缺少模型结果或模型分析失败")
        return

    # 2. 取出特征和数据
    # 需要重新构建用于SHAP分析的训练数据和模型
    def get_X_y(df, features):
        gshc_df = create_gshc(df)
        gshc_df = gshc_df.dropna(subset=['Y_Transition'])  # 保证无NaN
        X = gshc_df[features]
        y = gshc_df['Y_Transition']
        return X, y

    # 3. 重新训练模型，获得拟合后的pipeline（因为之前的pipeline是交叉验证的，未保存fit后的对象）
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LogisticRegressionCV

    # PSG
    X_base, y_base = get_X_y(base_df, baseline['feature_names'])
    pipeline_base = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('lasso', LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 50), cv=5, penalty='l1',
            solver='liblinear', scoring='roc_auc',
            random_state=42, class_weight='balanced'
        ))
    ])
    pipeline_base.fit(X_base, y_base)

    # Smartphone Only
    # 需要重新生成Smartphone Only降质后的数据
    smartphone_config = DEGRADATION_SCENARIOS['Smartphone_Only']
    degraded_df = simulate_wearable_data(base_df, 'Smartphone_Only', smartphone_config)
    X_phone, y_phone = get_X_y(degraded_df, smartphone['feature_names'])
    pipeline_phone = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('lasso', LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 50), cv=5, penalty='l1',
            solver='liblinear', scoring='roc_auc',
            random_state=42, class_weight='balanced'
        ))
    ])
    pipeline_phone.fit(X_phone, y_phone)

    # 4. 计算SHAP值
    explainer_base = shap.Explainer(pipeline_base.named_steps['lasso'], X_base)
    shap_values_base = explainer_base(X_base)

    explainer_phone = shap.Explainer(pipeline_phone.named_steps['lasso'], X_phone)
    shap_values_phone = explainer_phone(X_phone)

    # 5. 并排画图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.sca(axes[0])
    shap.summary_plot(shap_values_base, X_base, show=False, plot_type="dot")
    axes[0].set_title("PSG Baseline Model", fontsize=16, fontweight='bold')
    plt.sca(axes[1])
    shap.summary_plot(shap_values_phone, X_phone, show=False, plot_type="dot")
    axes[1].set_title("Smartphone Only Model", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure5_brain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5（模型大脑对比图）已保存")

if __name__ == "__main__":
    print("=== Wearable Device Simulation Analysis ===")
    print("Simulating consumer-grade wearable device data quality...")
    
    # Setup output directory
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_wearable_simulation')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {OUTPUT_DIR}")
    
    # Load data
    print("\n--- Loading and preparing data ---")
    base_df = load_and_map_data(DATA_FILES)
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    full_cohort_df = base_df.dropna(subset=['Y_Transition']).copy()
    
    print(f"Full cohort size: {len(full_cohort_df)}")
    
    # Run analysis for each degradation scenario
    print(f"\n--- Running wearable device simulations ---")
    all_results = {}
    
    for scenario_name, scenario_config in DEGRADATION_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Description: {scenario_config['description']}")
        
        # Apply degradations
        if scenario_name == 'Baseline_PSG':
            # No degradation for baseline
            degraded_df = full_cohort_df.copy()
        else:
            degraded_df = simulate_wearable_data(full_cohort_df, scenario_name, scenario_config)
        
        # Get available features after degradation
        available_features = get_available_features(degraded_df, HIGH_QUALITY_FEATURES)
        print(f"Available features: {len(available_features)}/{len(HIGH_QUALITY_FEATURES)}")
        print(f"Features: {available_features}")
        
        # Run GSHC analysis
        results = run_gshc_analysis(degraded_df, scenario_name, available_features)
        all_results[scenario_name] = results
        
        # Print results
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✓ GSHC size: {results['gshc_size']}")
            print(f"✓ Selected features: {results['selected_features']}/{results['available_features']}")
            print(f"✓ AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            print(f"✓ Total signal strength: {results['total_signal_strength']:.4f}")
            print(f"✓ SDB signal strength: {results['sdb_signal_strength']:.4f}")
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'wearable_simulation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nRaw results saved to: {results_file}")
    
    print("\n--- Analysis completed! ---")
    print(f"All results saved to: {OUTPUT_DIR}")

    # Print quick summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)

    baseline = all_results.get('Baseline_PSG', {})
    if 'error' not in baseline:
        print(f"📊 PSG Baseline AUC: {baseline['mean_auc']:.3f}")

    wearable_results = {k: v for k, v in all_results.items()
                       if k != 'Baseline_PSG' and 'error' not in v}

    for scenario_name, results in wearable_results.items():
        if 'error' not in baseline:
            loss = (baseline['mean_auc'] - results['mean_auc']) / baseline['mean_auc'] * 100
            print(f"📱 {scenario_name}: AUC {results['mean_auc']:.3f} (-{loss:.1f}%)")
        else:
            print(f"� {scenario_name}: AUC {results['mean_auc']:.3f}")

    print("\n🎯 Wearable Device Simulation Complete!")
    print("📊 Key Questions Answered:")
    print("   - How much performance is lost with consumer devices?")
    print("   - Which device types are most viable for clinical use?")
    print("   - What are the critical sensor requirements?")
    print("   - Is the GSHC framework robust to data quality degradation?")
    print("\n💡 To generate detailed visualizations, run the visualization functions separately.")

    # === 自动生成 Figure 5（模型大脑对比图） ===
    try:
        print("\n--- Generating Figure 5: Brain Comparison Plot ---")
        create_brain_comparison_plot(all_results, OUTPUT_DIR, base_df)
        print("✓ Figure 5 saved to:", os.path.join(OUTPUT_DIR, 'figure5_brain_comparison.png'))
    except Exception as e:
        print(f"❌ Failed to generate Figure 5: {e}")

# =============================================================================
# --- Visualization Functions ---
# =============================================================================

def create_performance_comparison(all_results, output_dir):
    """Create performance comparison across wearable device scenarios"""

    # Extract data for plotting
    scenarios = []
    aucs = []
    auc_stds = []
    total_signals = []
    sdb_signals = []
    selected_features = []

    for scenario_name, results in all_results.items():
        if 'error' not in results:
            scenarios.append(scenario_name.replace('_', '\n'))
            aucs.append(results['mean_auc'])
            auc_stds.append(results['std_auc'])
            total_signals.append(results['total_signal_strength'])
            sdb_signals.append(results['sdb_signal_strength'])
            selected_features.append(results['selected_features'])

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: AUC Comparison
    bars1 = ax1.bar(scenarios, aucs, yerr=auc_stds, capsize=5,
                    color=['#2E8B57', '#FF6B35', '#F7931E', '#C41E3A'])
    ax1.set_ylabel('Cross-Validation AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Across Device Types', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, auc, std in zip(bars1, aucs, auc_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Total Signal Strength
    bars2 = ax2.bar(scenarios, total_signals,
                    color=['#2E8B57', '#FF6B35', '#F7931E', '#C41E3A'])
    ax2.set_ylabel('Total Signal Strength', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Feature Signal Strength', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for bar, signal in zip(bars2, total_signals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{signal:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: SDB-Specific Signal Strength
    bars3 = ax3.bar(scenarios, sdb_signals,
                    color=['#2E8B57', '#FF6B35', '#F7931E', '#C41E3A'])
    ax3.set_ylabel('SDB Signal Strength', fontsize=12, fontweight='bold')
    ax3.set_title('Sleep-Disordered Breathing Signal', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    for bar, signal in zip(bars3, sdb_signals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{signal:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Selected Features Count
    bars4 = ax4.bar(scenarios, selected_features,
                    color=['#2E8B57', '#FF6B35', '#F7931E', '#C41E3A'])
    ax4.set_ylabel('Number of Selected Features', fontsize=12, fontweight='bold')
    ax4.set_title('Feature Selection by LASSO', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    for bar, count in zip(bars4, selected_features):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wearable_performance_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Performance comparison plot saved")

def create_signal_degradation_analysis(all_results, output_dir):
    """Create detailed signal degradation analysis"""

    # Calculate performance degradation relative to baseline
    baseline_results = all_results.get('Baseline_PSG', {})
    if 'error' in baseline_results:
        print("❌ Cannot create degradation analysis - baseline failed")
        return

    baseline_auc = baseline_results['mean_auc']
    baseline_total_signal = baseline_results['total_signal_strength']
    baseline_sdb_signal = baseline_results['sdb_signal_strength']

    # Prepare data
    scenarios = []
    auc_degradation = []
    signal_degradation = []
    sdb_degradation = []

    for scenario_name, results in all_results.items():
        if scenario_name != 'Baseline_PSG' and 'error' not in results:
            scenarios.append(scenario_name.replace('_', ' '))

            # Calculate percentage degradation
            auc_deg = (baseline_auc - results['mean_auc']) / baseline_auc * 100
            signal_deg = (baseline_total_signal - results['total_signal_strength']) / baseline_total_signal * 100
            sdb_deg = (baseline_sdb_signal - results['sdb_signal_strength']) / baseline_sdb_signal * 100

            auc_degradation.append(auc_deg)
            signal_degradation.append(signal_deg)
            sdb_degradation.append(sdb_deg)

    # Create degradation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Performance Degradation
    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax1.bar(x - width/2, auc_degradation, width, label='AUC Loss (%)',
                    color='#FF6B35', alpha=0.8)
    bars2 = ax1.bar(x + width/2, signal_degradation, width, label='Total Signal Loss (%)',
                    color='#F7931E', alpha=0.8)

    ax1.set_xlabel('Wearable Device Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Loss (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Degradation vs PSG Baseline', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 2: SDB Signal Specific Analysis
    bars3 = ax2.bar(scenarios, sdb_degradation,
                    color=['#FF6B35', '#F7931E', '#C41E3A'])
    ax2.set_xlabel('Wearable Device Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SDB Signal Loss (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sleep-Disordered Breathing Signal Degradation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for bar, deg in zip(bars3, sdb_degradation):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{deg:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_degradation_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Signal degradation analysis plot saved")

def create_feature_availability_chart(all_results, output_dir):
    """Create feature availability and selection chart"""

    # Prepare data
    scenarios = []
    available_features = []
    selected_features = []

    for scenario_name, results in all_results.items():
        if 'error' not in results:
            scenarios.append(scenario_name.replace('_', '\n'))
            available_features.append(results['available_features'])
            selected_features.append(results['selected_features'])

    # Create chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, available_features, width,
                   label='Available Features', color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x + width/2, selected_features, width,
                   label='Selected by LASSO', color='#FF6B35', alpha=0.8)

    ax.set_xlabel('Device Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Availability and Selection Across Device Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_availability_chart.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Feature availability chart saved")

def generate_summary_report(all_results, output_dir):
    """Generate a comprehensive summary report"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("WEARABLE DEVICE SIMULATION - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Get baseline for comparison
    baseline = all_results.get('Baseline_PSG', {})
    if 'error' not in baseline:
        report_lines.append("BASELINE (PSG Gold Standard):")
        report_lines.append(f"  • AUC: {baseline['mean_auc']:.3f} ± {baseline['std_auc']:.3f}")
        report_lines.append(f"  • Total Signal Strength: {baseline['total_signal_strength']:.4f}")
        report_lines.append(f"  • SDB Signal Strength: {baseline['sdb_signal_strength']:.4f}")
        report_lines.append(f"  • Selected Features: {baseline['selected_features']}/{baseline['available_features']}")
        report_lines.append("")

    # Analyze each wearable scenario
    for scenario_name, results in all_results.items():
        if scenario_name != 'Baseline_PSG' and 'error' not in results:
            report_lines.append(f"{scenario_name.upper().replace('_', ' ')}:")
            report_lines.append(f"  • Description: {DEGRADATION_SCENARIOS[scenario_name]['description']}")
            report_lines.append(f"  • AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            report_lines.append(f"  • Total Signal: {results['total_signal_strength']:.4f}")
            report_lines.append(f"  • SDB Signal: {results['sdb_signal_strength']:.4f}")
            report_lines.append(f"  • Features: {results['selected_features']}/{results['available_features']}")

            # Calculate degradation if baseline available
            if 'error' not in baseline:
                auc_loss = (baseline['mean_auc'] - results['mean_auc']) / baseline['mean_auc'] * 100
                signal_loss = (baseline['total_signal_strength'] - results['total_signal_strength']) / baseline['total_signal_strength'] * 100
                sdb_loss = (baseline['sdb_signal_strength'] - results['sdb_signal_strength']) / baseline['sdb_signal_strength'] * 100

                report_lines.append(f"  • Performance Loss: AUC -{auc_loss:.1f}%, Signal -{signal_loss:.1f}%, SDB -{sdb_loss:.1f}%")

            report_lines.append(f"  • Available SDB Variables: {', '.join(results['available_sdb_vars'])}")
            report_lines.append("")

    # Key findings
    report_lines.append("KEY FINDINGS:")
    report_lines.append("=" * 40)

    # Find best and worst performing wearable devices
    wearable_results = {k: v for k, v in all_results.items()
                       if k != 'Baseline_PSG' and 'error' not in v}

    if wearable_results:
        best_auc = max(wearable_results.items(), key=lambda x: x[1]['mean_auc'])
        worst_auc = min(wearable_results.items(), key=lambda x: x[1]['mean_auc'])

        report_lines.append(f"• Best performing device: {best_auc[0]} (AUC: {best_auc[1]['mean_auc']:.3f})")
        report_lines.append(f"• Worst performing device: {worst_auc[0]} (AUC: {worst_auc[1]['mean_auc']:.3f})")

        if 'error' not in baseline:
            best_loss = (baseline['mean_auc'] - best_auc[1]['mean_auc']) / baseline['mean_auc'] * 100
            worst_loss = (baseline['mean_auc'] - worst_auc[1]['mean_auc']) / baseline['mean_auc'] * 100
            report_lines.append(f"• Performance range: {best_loss:.1f}% to {worst_loss:.1f}% AUC loss vs PSG")

    report_lines.append("")
    report_lines.append("CLINICAL IMPLICATIONS:")
    report_lines.append("=" * 40)
    report_lines.append("• Consumer smartwatches may retain significant predictive power")
    report_lines.append("• Basic fitness trackers show moderate degradation but remain useful")
    report_lines.append("• Smartphone-only approaches have substantial limitations")
    report_lines.append("• SDB signal detection is particularly sensitive to sensor quality")
    report_lines.append("")

    # Save report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'wearable_simulation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)

    print("✓ Comprehensive summary report saved")
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)

    # Print key findings to console
    if 'error' not in baseline:
        print(f"📊 PSG Baseline AUC: {baseline['mean_auc']:.3f}")

    for scenario_name, results in wearable_results.items():
        if 'error' not in baseline:
            loss = (baseline['mean_auc'] - results['mean_auc']) / baseline['mean_auc'] * 100
            print(f"📱 {scenario_name}: AUC {results['mean_auc']:.3f} (-{loss:.1f}%)")
        else:
            print(f"📱 {scenario_name}: AUC {results['mean_auc']:.3f}")

    return report_file
