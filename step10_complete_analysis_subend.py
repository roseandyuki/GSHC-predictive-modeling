# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 10: Complete Analysis Pipeline (Subend) - V3 ---
#
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

# --- Utility Functions (NRI & IDI) ---
def calculate_nri(y_true, y_pred_new, y_pred_base, threshold=0.5):
    pred_new = (y_pred_new >= threshold).astype(int)
    pred_base = (y_pred_base >= threshold).astype(int)
    events = y_true == 1
    event_up, event_down = np.sum((pred_new > pred_base) & events), np.sum((pred_new < pred_base) & events)
    n_events = np.sum(events)
    nonevents = y_true == 0
    nonevent_up, nonevent_down = np.sum((pred_new > pred_base) & nonevents), np.sum((pred_new < pred_base) & nonevents)
    n_nonevents = np.sum(nonevents)
    if n_events == 0 or n_nonevents == 0: return 0
    return ((event_up - event_down) / n_events) + ((nonevent_down - nonevent_up) / n_nonevents)

def calculate_idi(y_true, y_pred_new, y_pred_base):
    events = y_true == 1
    nonevents = y_true == 0
    if np.sum(events) == 0 or np.sum(nonevents) == 0: return 0
    new_event_mean, new_nonevent_mean = np.mean(y_pred_new[events]), np.mean(y_pred_new[nonevents])
    base_event_mean, base_nonevent_mean = np.mean(y_pred_base[events]), np.mean(y_pred_base[nonevents])
    return (new_event_mean - new_nonevent_mean) - (base_event_mean - base_nonevent_mean)

# --- Bootstrap Confidence Interval Function for AUC ---
def bootstrap_auc_ci(X_data, y_data, model_pipeline, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """
    Calculate Bootstrap 95% Confidence Interval for AUC

    Parameters:
    -----------
    X_data : DataFrame
        Feature data
    y_data : Series
        Target data
    model_pipeline : sklearn Pipeline
        Fitted model pipeline
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    tuple : (lower_bound, upper_bound, bootstrap_aucs)
    """
    np.random.seed(random_state)
    bootstrap_aucs = []
    n_samples = len(X_data)

    for i in range(n_bootstrap):
        # Bootstrap sampling with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_data.iloc[bootstrap_indices]
        y_bootstrap = y_data.iloc[bootstrap_indices]

        # Skip if bootstrap sample doesn't have both classes
        if len(np.unique(y_bootstrap)) < 2:
            continue

        try:
            # Get predictions from the fitted model
            y_pred_bootstrap = model_pipeline.predict_proba(X_bootstrap)[:, 1]
            auc_bootstrap = roc_auc_score(y_bootstrap, y_pred_bootstrap)
            bootstrap_aucs.append(auc_bootstrap)
        except Exception:
            continue

    if len(bootstrap_aucs) == 0:
        return np.nan, np.nan, []

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_aucs, lower_percentile)
    upper_bound = np.percentile(bootstrap_aucs, upper_percentile)

    return lower_bound, upper_bound, bootstrap_aucs

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# --- Part 1: Data Foundation and Final Cohort Confirmation ---
# =============================================================================
print("--- Part 1: Building analysis foundation... ---")
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All results will be saved to: {OUTPUT_DIR}")

DATA_FILES = {
    'shhs1': os.path.join(SCRIPT_DIR, 'shhs1-dataset-0.21.0.csv'),
    'shhs2': os.path.join(SCRIPT_DIR, 'shhs2-dataset-0.21.0.csv')
}
VAR_MAP = {'bmi_v1': 'bmi_s1', 'sbp_v1': 'systbp', 'dbp_v1': 'diasbp', 'age_v1': 'age_s1', 'gender': 'gender', 'ess_v1': 'ess_s1', 'arousal_index': 'ai_all', 'n3_percent': 'times34p', 'n1_percent': 'timest1p', 'n2_percent': 'timest2p', 'rem_percent': 'timeremp', 'sleep_efficiency': 'slpeffp', 'waso': 'waso', 'rdi': 'rdi4p', 'min_spo2': 'minsat', 'avg_spo2': 'avgsat', 'bmi_v2': 'bmi_s2', 'sbp_v2': 'avg23bps_s2', 'dbp_v2': 'avg23bpd_s2'}
RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

def load_and_map_data(filepaths, id_col='nsrrid'):
    try:
        df1 = pd.read_csv(filepaths['shhs1'], encoding='ISO-8859-1', low_memory=False)
        df2 = pd.read_csv(filepaths['shhs2'], encoding='ISO-8859-1', low_memory=False)
        merged_df = pd.merge(df1, df2, on=id_col, how='left', suffixes=('', '_dup'))
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]
        merged_df.rename(columns=RENAME_MAP, inplace=True)
        return merged_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Critical data file missing - {e}. Please ensure all data files are in the script directory.")

base_df = load_and_map_data(DATA_FILES)

def is_healthy_v1(row):
    return row['bmi_v1'] < 25 and row['sbp_v1'] < 120 and row['dbp_v1'] < 80

def has_transitioned(row):
    if pd.isna(row['bmi_v2']) or pd.isna(row['sbp_v2']) or pd.isna(row['dbp_v2']): return np.nan
    return 1 if any([row['bmi_v2'] >= 25, row['sbp_v2'] >= 120, row['dbp_v2'] >= 80]) else 0

healthy_cohort = base_df[base_df.apply(is_healthy_v1, axis=1)].copy()
healthy_cohort['Y_Transition'] = healthy_cohort.apply(has_transitioned, axis=1)

HIGH_QUALITY_FEATURES = ['bmi_v1', 'sbp_v1', 'dbp_v1', 'age_v1', 'gender', 'ess_v1', 'arousal_index', 'n3_percent', 'n1_percent', 'n2_percent', 'rem_percent', 'sleep_efficiency', 'waso', 'rdi', 'min_spo2', 'avg_spo2']
final_df = healthy_cohort[['Y_Transition'] + HIGH_QUALITY_FEATURES + ['bmi_v2', 'sbp_v2', 'dbp_v2']].dropna(subset=['Y_Transition'])
final_df['Y_Transition'] = final_df['Y_Transition'].astype(int)

X = final_df[HIGH_QUALITY_FEATURES]
y = final_df['Y_Transition']
print(f"Final sample size for predictive analysis: n={len(final_df)}")

# =============================================================================
# --- Part 1.5: Supplementary Causal Inference Analysis (PSM) ---
# =============================================================================
print("\n--- Part 1.5: Performing Supplementary Causal Inference using PSM... ---")
# ... (This part was logically correct and remains unchanged)
causal_df = base_df.copy()
causal_df['outcome'] = causal_df.apply(has_transitioned, axis=1)
causal_df['treatment'] = ((causal_df['rdi'] >= 5) & (causal_df['min_spo2'] < 94)).astype(int)
confounders = ['age_v1', 'gender', 'bmi_v1', 'sbp_v1']
causal_vars = ['outcome', 'treatment'] + confounders
causal_df_clean = causal_df[causal_vars].dropna().reset_index(drop=True)
X_ps = causal_df_clean[confounders]
y_ps = causal_df_clean['treatment']
lr = LogisticRegression(solver='liblinear')
lr.fit(X_ps, y_ps)
causal_df_clean['propensity_score'] = lr.predict_proba(X_ps)[:, 1]
treatment_group = causal_df_clean[causal_df_clean['treatment'] == 1]
control_group = causal_df_clean[causal_df_clean['treatment'] == 0]
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control_group[['propensity_score']])
distances, indices = nn.kneighbors(treatment_group[['propensity_score']])
matched_control_group = control_group.iloc[indices.flatten()]
matched_df = pd.concat([treatment_group.reset_index(drop=True), matched_control_group.reset_index(drop=True)], axis=0)
matched_treatment_outcome = matched_df[matched_df['treatment'] == 1]['outcome']
matched_control_outcome = matched_df[matched_df['treatment'] == 0]['outcome']
ate = matched_treatment_outcome.mean() - matched_control_outcome.mean()
p_value = ttest_ind(matched_treatment_outcome, matched_control_outcome).pvalue
print(f"Estimated Average Treatment Effect (ATE): {ate:.4f} (p-value: {p_value:.4f})")


# =============================================================================
# --- Part 2: Main Predictive Analysis - Feature Selection & EDA ---
# =============================================================================
print("\n--- Part 2: Performing main predictive analysis... ---")
pipeline_lasso = Pipeline([('imputer', IterativeImputer(max_iter=10, random_state=42)), ('scaler', StandardScaler()), ('lasso_cv', LogisticRegressionCV(Cs=np.logspace(-4, 4, 100), cv=10, penalty='l1', solver='liblinear', scoring='roc_auc', random_state=42, class_weight='balanced'))])
pipeline_lasso.fit(X, y)
lasso_model_for_path = pipeline_lasso.named_steps['lasso_cv']
selected_coeffs = lasso_model_for_path.coef_[0]
lasso_6_features = X.columns[selected_coeffs != 0].tolist()
print(f"LASSO finally selected features ({len(lasso_6_features)}): {lasso_6_features}")

# =============================================================================
# --- Part 2.5: Endpoint Deconstruction Analysis Setup ---
# =============================================================================
print("\n--- Part 2.5: Setting up for Endpoint Deconstruction Analysis... ---")

# --- Create a clean dataset for predicting ONLY Hypertension ---
# Outcome (y_htn): 1 if developed HTN only, 0 if remained healthy.
# We must exclude those who became overweight-only or both.
is_hypertensive_v2 = (final_df['sbp_v2'] >= 120) | (final_df['dbp_v2'] >= 80)
is_overweight_v2 = final_df['bmi_v2'] >= 25

# Define the three groups based on outcome
healthy_group = final_df[final_df['Y_Transition'] == 0]
htn_only_group = final_df[is_hypertensive_v2 & ~is_overweight_v2]
ow_only_group = final_df[~is_hypertensive_v2 & is_overweight_v2]

# Create the final dataset for Hypertension-Only prediction
df_for_htn_analysis = pd.concat([htn_only_group, healthy_group])
X_htn_endpoint = df_for_htn_analysis[HIGH_QUALITY_FEATURES]
y_htn_endpoint = df_for_htn_analysis['Y_Transition']
print(f"Sample size for Hypertension-Only Endpoint analysis: n={len(df_for_htn_analysis)}")

# --- Create a clean dataset for predicting ONLY Overweight ---
# Outcome (y_ow): 1 if developed Overweight only, 0 if remained healthy.
df_for_ow_analysis = pd.concat([ow_only_group, healthy_group])
X_ow_endpoint = df_for_ow_analysis[HIGH_QUALITY_FEATURES]
y_ow_endpoint = df_for_ow_analysis['Y_Transition']
print(f"Sample size for Overweight-Only Endpoint analysis: n={len(df_for_ow_analysis)}")

# =============================================================================
# --- Part 3: Main Predictive Analysis - Repeated Cross-Validation ---
# =============================================================================
print("\n--- Part 3: Evaluating model performance using 10-repeats CV... ---")

baseline_features = ['bmi_v1', 'sbp_v1', 'dbp_v1', 'age_v1', 'gender']
models_to_cv = {"Baseline Model": baseline_features, "Full Model (16 features)": HIGH_QUALITY_FEATURES, "LASSO Model (6 features)": lasso_6_features}

# --- ENHANCED FUNCTION WITH BOOTSTRAP CI FOR SUBGROUPS ---
def evaluate_model_cv(X_data, y_data, analysis_name):
    print(f"\n--- Evaluating: {analysis_name} (n={len(X_data)}) ---")

    # --- SAFETY CHECK ---
    if len(X_data) == 0:
        print("Warning: Input data is empty. Skipping evaluation for this group.")
        return None, 0, 0

    n_splits, n_repeats = (5, 5) if len(X_data) < 100 else (10, 10) # Reduced repeats for speed if needed
    if n_splits == 5: print(f"Small sample size, using {n_repeats} repeats of {n_splits}-fold CV.")

    cv_strategy = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = {}

    # Check if this is a subgroup analysis (for Bootstrap CI)
    is_subgroup = "Subgroup" in analysis_name

    for name, features in models_to_cv.items():
        # Ensure features exist in the dataframe to prevent KeyErrors
        available_features = [f for f in features if f in X_data.columns]
        pipeline_model = Pipeline([('imputer', IterativeImputer(max_iter=10, random_state=42)), ('scaler', StandardScaler()), ('model', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))])
        scores = cross_val_score(pipeline_model, X_data[available_features], y_data, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        results[name] = scores

        # Calculate mean and 95% Confidence Interval from the CV scores
        mean_auc = scores.mean()
        cv_lower_bound = np.percentile(scores, 2.5)
        cv_upper_bound = np.percentile(scores, 97.5)

        # For subgroup analyses, also calculate Bootstrap CI
        if is_subgroup:
            print(f"{name}: Mean AUC = {mean_auc:.4f} (95% CV-CI: {cv_lower_bound:.4f} - {cv_upper_bound:.4f})")

            # Fit the model on full data for Bootstrap CI calculation
            pipeline_model.fit(X_data[available_features], y_data)
            boot_lower, boot_upper, boot_aucs = bootstrap_auc_ci(X_data[available_features], y_data, pipeline_model, n_bootstrap=1000)

            if not np.isnan(boot_lower):
                print(f"    Bootstrap 95% CI: {boot_lower:.4f} - {boot_upper:.4f} (n_bootstrap={len(boot_aucs)})")
            else:
                print(f"    Bootstrap CI: Could not calculate (insufficient valid samples)")
        else:
            # For main analysis, just show CV-based CI
            print(f"{name}: Mean AUC = {mean_auc:.4f} (95% CI: {cv_lower_bound:.4f} - {cv_upper_bound:.4f})")

    p_lasso_vs_full = ttest_rel(results["LASSO Model (6 features)"], results["Full Model (16 features)"]).pvalue
    print(f"Paired t-test (LASSO vs Full Model): p-value = {p_lasso_vs_full:.4f}")
    p_lasso_vs_base = ttest_rel(results["LASSO Model (6 features)"], results["Baseline Model"]).pvalue
    print(f"Paired t-test (LASSO vs Baseline Model): p-value = {p_lasso_vs_base:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'cv_results_{analysis_name.replace(" ", "_")}.csv'))

    return results_df, n_splits, n_repeats

# Subgroup Analyses Definitions
is_hypertensive_v2 = (final_df['sbp_v2'] >= 120) | (final_df['dbp_v2'] >= 80)
is_overweight_v2 = final_df['bmi_v2'] >= 25
persistently_healthy = final_df[y == 0]
hypertension_only = final_df[(y == 1) & is_hypertensive_v2 & ~is_overweight_v2]
overweight_only = final_df[(y == 1) & ~is_hypertensive_v2 & is_overweight_v2]

X_htn = pd.concat([hypertension_only, persistently_healthy])[HIGH_QUALITY_FEATURES]
y_htn = pd.concat([hypertension_only, persistently_healthy])['Y_Transition']
X_ow = pd.concat([overweight_only, persistently_healthy])[HIGH_QUALITY_FEATURES]
y_ow = pd.concat([overweight_only, persistently_healthy])['Y_Transition']

# --- Now, call the evaluation function ---
cv_results_main, n_splits_main, n_repeats_main = evaluate_model_cv(X, y, "Main Analysis")
cv_results_htn, n_splits_htn, n_repeats_htn = evaluate_model_cv(X_htn, y_htn, "Hypertension Subgroup")
cv_results_ow, n_splits_ow, n_repeats_ow = evaluate_model_cv(X_ow, y_ow, "Overweight Subgroup")

# --- Run Endpoint Deconstruction CV ---
cv_results_htn_endpoint, n_splits_htn_e, n_repeats_htn_e = evaluate_model_cv(X_htn_endpoint, y_htn_endpoint, "Hypertension Only Endpoint")
cv_results_ow_endpoint, n_splits_ow_e, n_repeats_ow_e = evaluate_model_cv(X_ow_endpoint, y_ow_endpoint, "Overweight Only Endpoint")

# =============================================================================
# --- Part 4: Figure Generation & Robustness Checks ---
# =============================================================================
print("\n--- Part 4: Generating final publication-quality figures... ---")

def plot_forest_cv(cv_data, n_repeats, n_splits, title, filename):
    if cv_data is None: # Safety check if evaluation was skipped
        print(f"Skipping forest plot for '{title}' as no data was generated.")
        return
        
    plt.figure(figsize=(10, 5))
    means, stds = cv_data.mean(), cv_data.std()
    plt.errorbar(x=means, y=cv_data.columns, xerr=stds, fmt='o', color='black', ecolor='gray', elinewidth=3, capsize=5)
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.xlabel(f"AUC (from {n_repeats}-repeats of {n_splits}-fold CV)", fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.grid(axis='x')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated and saved forest plot: {filename}")

plot_forest_cv(cv_results_main, n_repeats_main, n_splits_main, "Model Performance Comparison (Main Analysis)", "forest_plot_main_analysis.png")
plot_forest_cv(cv_results_htn, n_repeats_htn, n_splits_htn, "Model Performance Comparison (Hypertension Subgroup)", "forest_plot_hypertension_subgroup.png")
plot_forest_cv(cv_results_ow, n_repeats_ow, n_splits_ow, "Model Performance Comparison (Overweight Subgroup)", "forest_plot_overweight_subgroup.png")

# =============================================================================
# --- Part 4 (Continued): Robustness Checks & Model Interpretability ---
# =============================================================================

print("\n--- Robustness Check: Training and Interpreting XGBoost Model... ---")
xgb_pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
        random_state=42, scale_pos_weight=y.value_counts()[0] / y.value_counts()[1]
    ))
])
cv_strategy_xgb = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
xgb_scores = cross_val_score(xgb_pipeline, X[lasso_6_features], y, cv=cv_strategy_xgb, scoring='roc_auc', n_jobs=-1)
print(f"XGBoost (6 features) Mean AUC: {xgb_scores.mean():.4f} (± {xgb_scores.std():.4f})")

# Fit the pipeline to get processed data for SHAP
xgb_pipeline.fit(X[lasso_6_features], y)
X_processed_for_shap = pd.DataFrame(
    xgb_pipeline.named_steps['scaler'].transform(
        xgb_pipeline.named_steps['imputer'].transform(X[lasso_6_features])
    ),
    columns=lasso_6_features
)
explainer_xgb = shap.Explainer(xgb_pipeline.named_steps['xgb'], X_processed_for_shap)
shap_values_xgb = explainer_xgb(X_processed_for_shap)

plt.figure()
shap.summary_plot(shap_values_xgb, X_processed_for_shap, show=False, plot_size=(10, 6))
plt.title("SHAP Summary for XGBoost Model (6 LASSO Features)", pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_xgboost.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated and saved SHAP summary plot for XGBoost.")


print("\n--- Generating SHAP plot for the final LASSO model... ---")
final_lasso_pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
])
final_lasso_pipeline.fit(X[lasso_6_features], y)
processed_X_df = pd.DataFrame(
    final_lasso_pipeline.named_steps['scaler'].transform(
        final_lasso_pipeline.named_steps['imputer'].transform(X[lasso_6_features])
    ),
    columns=lasso_6_features
)
explainer = shap.LinearExplainer(final_lasso_pipeline.named_steps['model'], processed_X_df)
shap_values = explainer.shap_values(processed_X_df)

plt.figure()
shap.summary_plot(shap_values, processed_X_df, show=False, plot_size=(10, 6))
plt.title("SHAP Summary for Final LASSO Model", pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated and saved SHAP summary plot for LASSO model.")

# =============================================================================
print("\n--- Part 4.5: Extracting Final Model Specification (for TRIPOD Checklist)... ---")

# Extract the fitted logistic regression model from the pipeline
final_model = final_lasso_pipeline.named_steps['model']

# Get the coefficients and intercept
coefficients = final_model.coef_[0]
intercept = final_model.intercept_[0]

# Create a clear DataFrame for display
coeff_df = pd.DataFrame({
    'Feature': lasso_6_features,
    'Coefficient (beta)': coefficients
})
# Add the intercept to the dataframe
intercept_row = pd.DataFrame([{'Feature': 'Intercept', 'Coefficient (beta)': intercept}])
coeff_df = pd.concat([intercept_row, coeff_df], ignore_index=True)

print("\n--- ★★★ Final LASSO Model Coefficients (TRIPOD Item 22) ★★★ ---")
print(coeff_df.to_string(index=False))
print("\nNote: These coefficients apply to Z-score standardized data.")
print("For prediction: logit(p) = intercept + β₁×(standardized_feature₁) + β₂×(standardized_feature₂) + ...")

# =============================================================================
# --- Plot Endpoint Deconstruction Results ---
plot_forest_cv(cv_results_htn_endpoint, n_repeats_htn_e, n_splits_htn_e, "Model Performance (Hypertension Only Endpoint)", "forest_plot_htn_endpoint.png")
plot_forest_cv(cv_results_ow_endpoint, n_repeats_ow_e, n_splits_ow_e, "Model Performance (Overweight Only Endpoint)", "forest_plot_ow_endpoint.png")


# =============================================================================
# --- Part 5: Clinical Utility Analysis (NRI, IDI, OR with Bootstrap) ---
# =============================================================================
print("\n--- Part 5: Performing Clinical Utility Analysis with Bootstrap Validation... ---")

# --- Fit models and get predictions once for the whole dataset ---
pipeline_baseline = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', random_state=42))
])
pipeline_baseline.fit(X[baseline_features], y)
y_pred_baseline = pipeline_baseline.predict_proba(X[baseline_features])[:, 1]

# The final_lasso_pipeline is already fitted, just predict
y_pred_lasso = final_lasso_pipeline.predict_proba(X[lasso_6_features])[:, 1]

# --- Point Estimates ---
print(f"Point Estimate for NRI: {calculate_nri(y, y_pred_lasso, y_pred_baseline):.4f}")
idi_point_estimate = calculate_idi(y, y_pred_lasso, y_pred_baseline)
print(f"Point Estimate for IDI: {idi_point_estimate:.4f}")

# --- Bootstrap Validation ---
print("Performing Bootstrap Validation (may take a moment)...")
n_bootstraps = 1000
idi_bootstraps, or_bootstraps = [], []
y_np = y.to_numpy()

# Impute data once for OR calculation to use in the loop
X_imputed_for_or = pd.DataFrame(
    IterativeImputer(max_iter=10, random_state=42).fit_transform(X),
    columns=X.columns,
    index=X.index  # <-- GEMINI'S FIX: Explicitly assign the original index from X
)
X_imputed_for_or['high_risk_sdb'] = ((X_imputed_for_or['rdi'] >= 5) & (X_imputed_for_or['min_spo2'] < 94)).astype(int)
or_vars = ['high_risk_sdb', 'age_v1', 'gender', 'bmi_v1', 'sbp_v1']

for i in range(n_bootstraps):
    indices = resample(np.arange(len(y_np)), random_state=i)
    if len(np.unique(y_np[indices])) < 2: continue
    
    # IDI Bootstrap
    idi_bootstraps.append(calculate_idi(y_np[indices], y_pred_lasso[indices], y_pred_baseline[indices]))
    
    # OR Bootstrap
    X_boot = X_imputed_for_or.iloc[indices]
    y_boot = y_np[indices]
    X_boot_final = sm.add_constant(X_boot[or_vars])
    try:
        model_boot = sm.Logit(y_boot, X_boot_final).fit(disp=0)
        or_bootstraps.append(np.exp(model_boot.params['high_risk_sdb']))
    except Exception:
        continue

# --- Final Results Presentation ---
print("\n--- ★★★ Final Clinical Utility Results ★★★ ---")
idi_ci = np.percentile(idi_bootstraps, [2.5, 97.5])
idi_p_value = np.mean(np.array(idi_bootstraps) <= 0) * 2 # Two-tailed p-value
print("\n--- Integrated Discrimination Improvement (IDI) ---")
print(f"IDI Point Estimate: {idi_point_estimate:.4f}")
print(f"95% Bootstrap CI: [{idi_ci[0]:.4f}, {idi_ci[1]:.4f}], p-value: {idi_p_value:.4f}")
if idi_p_value < 0.05: print("(Statistically significant improvement in discrimination)")

or_ci_boot = np.percentile(or_bootstraps, [2.5, 97.5])
or_point_estimate = np.exp(sm.Logit(y, sm.add_constant(X_imputed_for_or[or_vars])).fit(disp=0).params['high_risk_sdb'])
print("\n--- Subclinical Threshold Odds Ratio (OR) ---")
print(f"Adjusted OR Point Estimate: {or_point_estimate:.2f}")
print(f"95% Bootstrap CI: [{or_ci_boot[0]:.2f}, {or_ci_boot[1]:.2f}]")
if or_ci_boot[0] > 1: print("(Statistically significant increased risk)")
else: print("(Trend towards increased risk, but not statistically significant)")

# =============================================================================
# --- Part 6: Advanced Clinical Utility Validation Plots (Calibration & DCA) ---
# =============================================================================
print("\n--- Part 6: Generating Advanced Clinical Validation Plots... ---")

y_pred_xgboost = xgb_pipeline.predict_proba(X[lasso_6_features])[:, 1]

# --- Calibration Curve ---
from sklearn.calibration import calibration_curve
fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for model_preds, model_name, style, color in [(y_pred_lasso, "LASSO Model", "s-", "blue"), (y_pred_xgboost, "XGBoost Model", "o--", "green")]:
    frac_pos, mean_pred = calibration_curve(y, model_preds, n_bins=10, strategy='uniform')
    ax1.plot(mean_pred, frac_pos, style, label=model_name, color=color)
ax1.set(xlabel="Mean predicted probability", ylabel="Fraction of positives (Observed probability)", title="Comparative Calibration Curve")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.hist(y_pred_lasso, bins=20, histtype='step', color='blue', alpha=0.5, lw=2)
ax2.set_ylabel("Prediction Distribution", color='blue', alpha=0.6)
ax2.tick_params(axis='y', labelcolor='blue', colors='blue')
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated and saved Comparative Calibration Curve.")

# --- Decision Curve Analysis (DCA) ---
def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    net_benefit_model = []
    n = len(y_true)
    for t in thresholds:
        tp = np.sum((y_pred_proba >= t) & (y_true == 1))
        fp = np.sum((y_pred_proba >= t) & (y_true == 0))
        net_benefit_model.append((tp / n) - (fp / n) * (t / (1 - t)))
    return np.array(net_benefit_model)

thresholds = np.linspace(0.01, 0.99, 100)
net_benefit_model = calculate_net_benefit(y, y_pred_lasso, thresholds)
net_benefit_all = (y.mean()) - (1 - y.mean()) * (thresholds / (1 - thresholds))

plt.figure(figsize=(10, 8))
plt.plot(thresholds, net_benefit_model, label="LASSO Model", color='blue')
plt.plot(thresholds, net_benefit_all, label="Treat All", color='gray', linestyle='--')
plt.plot(thresholds, np.zeros_like(thresholds), label="Treat None", color='black', linestyle=':')
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA)")
plt.ylim(-0.1, 0.6)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated and saved Decision Curve Analysis plot.")

print("\n--- ★★★ ALL ANALYSES AND REPORTS ARE COMPLETE ★★★ ---")

# =============================================================================
print("\n--- Part 7: Generating Baseline Characteristics (Table 1)... ---")

from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd

table1_df = final_df.copy()

group0 = table1_df[table1_df['Y_Transition'] == 0] # 保持健康组
group1 = table1_df[table1_df['Y_Transition'] == 1] # 发生转变组

# 创建一个空的列表来存储每一行的结果
table1_rows = []

# 定义变量的显示名称，让表格更美观
variable_labels = {
    'age_v1': '年龄 (岁)',
    'gender': '性别 (男, %)',
    'bmi_v1': '体重指数 (kg/m²)',
    'sbp_v1': '收缩压 (mmHg)',
    'dbp_v1': '舒张压 (mmHg)',
    'ess_v1': 'Epworth嗜睡量表',
    'sleep_efficiency': '睡眠效率 (%)',
    'waso': '入睡后清醒时间 (分钟)',
    'n1_percent': 'N1期睡眠占比 (%)',
    'n2_percent': 'N2期睡眠占比 (%)',
    'n3_percent': 'N3期睡眠占比 (%)',
    'rem_percent': 'REM期睡眠占比 (%)',
    'arousal_index': '微觉醒指数 (次/小时)',
    'rdi': '呼吸紊乱指数 (次/小时)',
    'avg_spo2': '平均血氧饱和度 (%)',
    'min_spo2': '最低血氧饱和度 (%)'
}

# 按照模板的顺序对变量进行排序
ordered_features = [
    'age_v1', 'gender', 'bmi_v1', 'sbp_v1', 'dbp_v1', 'ess_v1',
    'sleep_efficiency', 'waso', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent',
    'arousal_index', 'rdi', 'avg_spo2', 'min_spo2'
]


# 循环处理每一个特征
for feature in ordered_features:
    row = {'特征 (Characteristic)': variable_labels.get(feature, feature)}

    # 处理连续变量
    if feature != 'gender':
        mean_total, std_total = table1_df[feature].mean(), table1_df[feature].std()
        row['总体 (n=447)'] = f"{mean_total:.2f} ({std_total:.2f})"
        mean0, std0 = group0[feature].mean(), group0[feature].std()
        row['保持健康组 (Y=0)'] = f"{mean0:.2f} ({std0:.2f})"
        mean1, std1 = group1[feature].mean(), group1[feature].std()
        row['发生转变组 (Y=1)'] = f"{mean1:.2f} ({std1:.2f})"
        _, p_val = ttest_ind(group0[feature].dropna(), group1[feature].dropna())
        row['p-value'] = f"{p_val:.3f}"
        if p_val < 0.001:
            row['p-value'] = "<0.001"
    # 处理分类变量 (性别)
    else:
        # ** ROBUST FIX: Auto-detect gender coding and handle properly **
        # 首先检查性别变量的唯一值
        unique_values = sorted(table1_df[feature].dropna().unique())
        print(f"DEBUG: Gender unique values: {unique_values}")

        # 自动检测编码方式并确定哪个值代表男性
        if set(unique_values) == {0, 1}:
            # 0/1 编码：假设1=男性，0=女性
            male_code = 1
        elif set(unique_values) == {1, 2}:
            # 1/2 编码：假设1=男性，2=女性
            male_code = 1
        else:
            # 其他编码：使用较小的值作为男性（通常的惯例）
            male_code = min(unique_values)

        print(f"DEBUG: Using {male_code} as male code")

        # 总体
        n_total_male = (table1_df[feature] == male_code).sum()
        n_total = len(table1_df[feature].dropna())
        pct_total_male = (n_total_male / n_total) * 100
        row['总体 (n=447)'] = f"{int(n_total_male)} ({pct_total_male:.1f}%)"

        # 保持健康组
        n0_male = (group0[feature] == male_code).sum()
        n0_total = len(group0[feature].dropna())
        pct0_male = (n0_male / n0_total) * 100
        row['保持健康组 (Y=0)'] = f"{int(n0_male)} ({pct0_male:.1f}%)"

        # 发生转变组
        n1_male = (group1[feature] == male_code).sum()
        n1_total = len(group1[feature].dropna())
        pct1_male = (n1_male / n1_total) * 100
        row['发生转变组 (Y=1)'] = f"{int(n1_male)} ({pct1_male:.1f}%)"

        # P-value (chi-squared test)
        contingency_table = pd.crosstab(table1_df[feature], table1_df['Y_Transition'])
        _, p_val, _, _ = chi2_contingency(contingency_table)
        row['p-value'] = f"{p_val:.3f}"

    table1_rows.append(row)

# 创建最终的DataFrame
table1_final_df = pd.DataFrame(table1_rows)

# 打印最终结果
print("\n--- ★★★ Baseline Characteristics (Table 1) Data (v2) ★★★ ---")
print(f"保持健康组 (Y=0): n={len(group0)}")
print(f"发生转变组 (Y=1): n={len(group1)}")
print(table1_final_df.to_string())
# =============================================================================
