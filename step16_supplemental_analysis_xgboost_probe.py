# -*- coding: utf-8 -*-
# =============================================================================
# --- Supplemental Analysis: The Non-Linear Probe ---
#
# Objective:
# To verify if the "AUC Deceptiveness" phenomenon, observed with a linear
# LASSO probe, also holds true when using a powerful, non-linear model like
# XGBoost. This supplemental analysis serves as a robustness check to confirm
# that the decoupling of performance and mechanistic fidelity is a general
# issue related to data quality degradation, not an artifact of the linear model.
#
# Method:
# 1. Train a baseline XGBoost model on the full, gold-standard PSG data.
# 2. Train another XGBoost model on the severely degraded "Smartphone Only" data.
# 3. Generate a side-by-side SHAP summary plot to compare the "internal logic"
#    or "model brain" of these two models.
#
# Expected Outcome:
# A visual confirmation that while AUC might be stable, the feature importance
# and decision-making logic of the XGBoost model are fundamentally rewired,
# thus proving the generality of our main finding.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score

# --- Global Settings ---
warnings.filterwarnings('ignore')
plt.style.use('default')

# --- Part 1: Configuration & Data Loading (Adapted from previous scripts) ---
print("--- Part 1: Initializing configuration and loading data... ---")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

# --- Create a dedicated output directory for this analysis ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_supplemental_xgboost')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_DIR}")

DATA_FILES = {
    'shhs1': os.path.join(SCRIPT_DIR, 'shhs1-dataset-0.21.0.csv'),
    'shhs2': os.path.join(SCRIPT_DIR, 'shhs2-dataset-0.21.0.csv')
}

VAR_MAP = {
    'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1',
    'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all', 'n3_percent': 'times34p',
    'n1_percent': 'timest1p', 'n2_percent': 'timest2p', 'rem_percent': 'timeremp',
    'sleep_efficiency': 'slpeffp', 'waso': 'waso', 'rdi': 'rdi4p', 'min_spo2': 'minsat',
    'avg_spo2': 'avgsat', 'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'
}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

HIGH_QUALITY_FEATURES = [
    'bmi_v1', 'sbp_v1', 'dbp_v1', 'age_v1', 'gender', 'ess_v1', 'arousal_index',
    'n3_percent', 'n1_percent', 'n2_percent', 'rem_percent', 'sleep_efficiency',
    'waso', 'rdi', 'min_spo2', 'avg_spo2'
]

# --- Part 2: Utility & Data Degradation Functions (Adapted from step16) ---
print("--- Part 2: Defining utility and data degradation functions... ---")

def load_and_map_data(filepaths, id_col='nsrrid'):
    try:
        df1 = pd.read_csv(filepaths['shhs1'], encoding='ISO-8859-1', low_memory=False)
        df2 = pd.read_csv(filepaths['shhs2'], encoding='ISO-8859-1', low_memory=False)
        merged_df = pd.merge(df1, df2, on=id_col, how='left', suffixes=('', '_dup'))
        return merged_df.rename(columns=RENAME_MAP)
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")

def create_gshc(df):
    gshc_criteria = (df['bmi_v1'] < 25) & (df['sbp_v1'] < 120) & (df['dbp_v1'] < 80)
    return df[gshc_criteria].copy()

def has_transitioned(row):
    if pd.isna(row['bmi_v2']) or pd.isna(row['sbp_v2']) or pd.isna(row['dbp_v2']):
        return np.nan
    return 1 if any([row['bmi_v2'] >= 25, row['sbp_v2'] >= 120, row['dbp_v2'] >= 80]) else 0

# We only need the 'Smartphone_Only' degradation scenario for this analysis
SMARTPHONE_DEGRADATION = [
    {'type': 'remove_variables', 'variables': ['min_spo2', 'avg_spo2', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent', 'arousal_index']},
    {'type': 'discretize', 'variable': 'rdi', 'bins': [0, 15, 100], 'labels': [0, 1]},
    {'type': 'add_noise', 'variables': ['sleep_efficiency'], 'noise_std': 8.0},
    {'type': 'add_missing', 'variables': ['sleep_efficiency', 'waso'], 'missing_rate': 0.35},
    {'type': 'reduce_precision', 'variables': ['sleep_efficiency', 'waso'], 'precision': 10}
]

def apply_degradation(df, degradation_config):
    df_degraded = df.copy()
    dtype = degradation_config['type']
    if dtype == 'remove_variables':
        vars_to_remove = [v for v in degradation_config['variables'] if v in df_degraded.columns]
        df_degraded = df_degraded.drop(columns=vars_to_remove)
    # Add other degradation types from your step16 if needed, following the pattern above.
    # For this specific analysis, 'remove_variables' is the most critical part.
    # The following lines are copied from your step16 script to ensure full simulation fidelity.
    elif dtype == 'add_noise':
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                noise = np.random.normal(0, degradation_config['noise_std'], len(df_degraded))
                df_degraded[var] += noise
    elif dtype == 'discretize':
        var = degradation_config['variable']
        if var in df_degraded.columns:
            df_degraded[var] = pd.cut(df_degraded[var], bins=degradation_config['bins'], labels=degradation_config['labels'], include_lowest=True).astype(float)
    elif dtype == 'add_missing':
         for var in degradation_config['variables']:
            if var in df_degraded.columns:
                n_missing = int(len(df_degraded) * degradation_config['missing_rate'])
                missing_indices = np.random.choice(df_degraded.index, n_missing, replace=False)
                df_degraded.loc[missing_indices, var] = np.nan
    elif dtype == 'reduce_precision':
        for var in degradation_config['variables']:
            if var in df_degraded.columns:
                precision = degradation_config['precision']
                df_degraded[var] = (df_degraded[var] / precision).round() * precision
    return df_degraded

def simulate_smartphone_data(df):
    df_simulated = df.copy()
    for degradation in SMARTPHONE_DEGRADATION:
        df_simulated = apply_degradation(df_simulated, degradation)
    return df_simulated

def get_available_features(df, original_features):
    return [feat for feat in original_features if feat in df.columns]

# --- Part 3: Core Analysis and Visualization ---
print("--- Part 3: Running comparative XGBoost analysis... ---")

def run_comparative_shap_analysis(base_df):
    """The main function to run the analysis and generate the comparison plot."""
    
    # --- 1. Prepare Data ---
    base_df['Y_Transition'] = base_df.apply(has_transitioned, axis=1)
    gshc_df_full = create_gshc(base_df)
    gshc_df_full = gshc_df_full.dropna(subset=['Y_Transition'])
    gshc_df_full['Y_Transition'] = gshc_df_full['Y_Transition'].astype(int)

    # --- 2. Baseline PSG Model ---
    print("  -> Training Baseline XGBoost Model (PSG)...")
    X_psg = gshc_df_full[HIGH_QUALITY_FEATURES]
    y_psg = gshc_df_full['Y_Transition']

    # Define the XGBoost pipeline (adapted from your step8 script)
    scale_pos_weight = y_psg.value_counts()[0] / y_psg.value_counts()[1]
    pipeline_psg = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
            random_state=42, scale_pos_weight=scale_pos_weight
        ))
    ])
    pipeline_psg.fit(X_psg, y_psg)
    # Calculate AUC for PSG model
    y_psg_pred = pipeline_psg.predict_proba(X_psg)[:, 1]
    auc_psg = roc_auc_score(y_psg, y_psg_pred)
    print(f"    [AUC] PSG Baseline Model: {auc_psg:.3f}")

    # --- 3. Smartphone Only Model ---
    print("  -> Training Smartphone Only XGBoost Model...")
    df_phone_simulated = simulate_smartphone_data(gshc_df_full)
    phone_features = get_available_features(df_phone_simulated, HIGH_QUALITY_FEATURES)
    X_phone = df_phone_simulated[phone_features]
    y_phone = df_phone_simulated['Y_Transition']
    # Re-use the same pipeline structure
    scale_pos_weight_phone = y_phone.value_counts()[0] / y_phone.value_counts()[1]
    pipeline_phone = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
            random_state=42, scale_pos_weight=scale_pos_weight_phone
        ))
    ])
    pipeline_phone.fit(X_phone, y_phone)
    # Calculate AUC for Smartphone model
    y_phone_pred = pipeline_phone.predict_proba(X_phone)[:, 1]
    auc_phone = roc_auc_score(y_phone, y_phone_pred)
    print(f"    [AUC] Smartphone Only Model: {auc_phone:.3f}")

    # --- 4. SHAP Value Calculation ---
    print("  -> Calculating SHAP values for both models...")
    # Pre-process data for SHAP Explainer
    X_psg_processed = pipeline_psg.named_steps['scaler'].transform(
        pipeline_psg.named_steps['imputer'].transform(X_psg)
    )
    X_psg_processed_df = pd.DataFrame(X_psg_processed, columns=X_psg.columns)

    X_phone_processed = pipeline_phone.named_steps['scaler'].transform(
        pipeline_phone.named_steps['imputer'].transform(X_phone)
    )
    X_phone_processed_df = pd.DataFrame(X_phone_processed, columns=X_phone.columns)

    # Create explainers and get SHAP values
    explainer_psg = shap.Explainer(pipeline_psg.named_steps['xgb'], X_psg_processed_df)
    shap_values_psg = explainer_psg(X_psg_processed_df)

    explainer_phone = shap.Explainer(pipeline_phone.named_steps['xgb'], X_phone_processed_df)
    shap_values_phone = explainer_phone(X_phone_processed_df)

    # --- 5. Generate and Save the Comparison Plot ---
    print("  -> Generating the final comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot for PSG Model
    plt.sca(axes[0])
    shap.summary_plot(shap_values_psg, X_psg_processed_df, show=False, plot_type="dot")
    axes[0].set_title("PSG Baseline Model (XGBoost)", fontsize=16, fontweight='normal', pad=10)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Plot for Smartphone Model
    plt.sca(axes[1])
    shap.summary_plot(shap_values_phone, X_phone_processed_df, show=False, plot_type="dot")
    axes[1].set_title("Smartphone Only Model (XGBoost)", fontsize=16, fontweight='normal', pad=10)
    axes[1].tick_params(axis='both', labelsize=12)
    
    # Reduce overall layout padding
    plt.tight_layout(pad=1.5)
    
    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'figure_supplemental_xgboost_brain_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Success! Comparison plot saved to: {output_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("================================================================")
    print("--- Running Supplemental Analysis: XGBoost as Non-Linear Probe ---")
    print("================================================================")
    
    # Load base data
    df_base = load_and_map_data(DATA_FILES)
    
    # Run the main analysis function
    run_comparative_shap_analysis(df_base)
    
    print("\n★★★ Supplemental analysis complete. ★★★")