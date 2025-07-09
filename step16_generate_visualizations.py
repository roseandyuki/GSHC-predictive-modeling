# -*- coding: utf-8 -*-
# =============================================================================
# --- Step 16B: Generate Wearable Device Simulation Visualizations ---
# 
# Purpose: Generate detailed visualizations and reports for the wearable 
# device simulation analysis
# 
# Usage: Run this after step16_wearable_device_simulation.py has completed
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

plt.style.use('default')

# =============================================================================
# --- Configuration ---
# =============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = '.'

INPUT_DIR = os.path.join(SCRIPT_DIR, 'output_wearable_simulation')
OUTPUT_DIR = INPUT_DIR  # Save in same directory

# Degradation scenarios (for report generation)
DEGRADATION_SCENARIOS = {
    'Baseline_PSG': {
        'description': 'Original PSG data (Gold Standard)',
    },
    'Consumer_Watch': {
        'description': 'Consumer smartwatch (Apple Watch, Fitbit level)',
    },
    'Basic_Fitness_Tracker': {
        'description': 'Basic fitness tracker (limited sensors)',
    },
    'Smartphone_Only': {
        'description': 'Smartphone-based sleep tracking (very limited)',
    }
}

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
        report_lines.append(f"  - AUC: {baseline['mean_auc']:.3f} ± {baseline['std_auc']:.3f}")
        report_lines.append(f"  - Total Signal Strength: {baseline['total_signal_strength']:.4f}")
        report_lines.append(f"  - SDB Signal Strength: {baseline['sdb_signal_strength']:.4f}")
        report_lines.append(f"  - Selected Features: {baseline['selected_features']}/{baseline['available_features']}")
        report_lines.append("")
    
    # Analyze each wearable scenario
    for scenario_name, results in all_results.items():
        if scenario_name != 'Baseline_PSG' and 'error' not in results:
            report_lines.append(f"{scenario_name.upper().replace('_', ' ')}:")
            report_lines.append(f"  - Description: {DEGRADATION_SCENARIOS[scenario_name]['description']}")
            report_lines.append(f"  - AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
            report_lines.append(f"  - Total Signal: {results['total_signal_strength']:.4f}")
            report_lines.append(f"  - SDB Signal: {results['sdb_signal_strength']:.4f}")
            report_lines.append(f"  - Features: {results['selected_features']}/{results['available_features']}")

            # Calculate degradation if baseline available
            if 'error' not in baseline:
                auc_loss = (baseline['mean_auc'] - results['mean_auc']) / baseline['mean_auc'] * 100
                signal_loss = (baseline['total_signal_strength'] - results['total_signal_strength']) / baseline['total_signal_strength'] * 100
                sdb_loss = (baseline['sdb_signal_strength'] - results['sdb_signal_strength']) / baseline['sdb_signal_strength'] * 100

                report_lines.append(f"  - Performance Loss: AUC -{auc_loss:.1f}%, Signal -{signal_loss:.1f}%, SDB -{sdb_loss:.1f}%")

            report_lines.append(f"  - Available SDB Variables: {', '.join(results['available_sdb_vars'])}")
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
        
        report_lines.append(f"- Best performing device: {best_auc[0]} (AUC: {best_auc[1]['mean_auc']:.3f})")
        report_lines.append(f"- Worst performing device: {worst_auc[0]} (AUC: {worst_auc[1]['mean_auc']:.3f})")

        if 'error' not in baseline:
            best_loss = (baseline['mean_auc'] - best_auc[1]['mean_auc']) / baseline['mean_auc'] * 100
            worst_loss = (baseline['mean_auc'] - worst_auc[1]['mean_auc']) / baseline['mean_auc'] * 100
            report_lines.append(f"- Performance range: {best_loss:.1f}% to {worst_loss:.1f}% AUC loss vs PSG")
    
    report_lines.append("")
    report_lines.append("CLINICAL IMPLICATIONS:")
    report_lines.append("=" * 40)
    report_lines.append("- Consumer smartwatches may retain significant predictive power")
    report_lines.append("- Basic fitness trackers show moderate degradation but remain useful")
    report_lines.append("- Smartphone-only approaches have substantial limitations")
    report_lines.append("- SDB signal detection is particularly sensitive to sensor quality")
    report_lines.append("")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'wearable_simulation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ Comprehensive summary report saved")
    return report_file

# =============================================================================
# --- Main Execution ---
# =============================================================================

if __name__ == "__main__":
    print("=== Wearable Device Simulation - Visualization Generator ===")
    
    # Load results
    results_file = os.path.join(INPUT_DIR, 'wearable_simulation_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"❌ Error: Results file not found at {results_file}")
        print("Please run step16_wearable_device_simulation.py first!")
        exit(1)
    
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Found results for {len(all_results)} scenarios")
    
    # Generate all visualizations
    print("\n--- Generating visualizations ---")
    create_performance_comparison(all_results, OUTPUT_DIR)
    create_signal_degradation_analysis(all_results, OUTPUT_DIR)
    create_feature_availability_chart(all_results, OUTPUT_DIR)
    
    # Generate comprehensive report
    print("\n--- Generating comprehensive report ---")
    report_file = generate_summary_report(all_results, OUTPUT_DIR)
    
    print(f"\n✅ All visualizations and report generated!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Detailed report: {report_file}")
    
    print("\n🎯 Generated Files:")
    print("   📊 wearable_performance_comparison.png")
    print("   📉 signal_degradation_analysis.png") 
    print("   📋 feature_availability_chart.png")
    print("   📄 wearable_simulation_report.txt")
