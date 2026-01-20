"""
EHS Instrument Validity Experiment

This experiment tests the ability of EHS (Entner-Hoyer-Spirtes) criteria to correctly
identify valid vs invalid instruments under realistic confounded structures.

Scenarios:
1. Valid Instrument: Z → X, U → X, U → Y, X → Y (Z satisfies exclusion)
2. Invalid Instrument: Z → X, Z → Y (exclusion violated)
3. Weak Instrument: Z → X weak (low power to detect relevance)
4. Confounder vs Instrument: Both Z (instrument) and W (confounder) present

Tests:
- Test (i): Y ⊥ Z | X — exogeneity (should NOT reject for valid instrument)
- Test (ii): Y ⊥̸ Z — outcome relevance (should reject, Z affects Y through X)
- Test (i.a): X ⊥̸ Z — instrument relevance (should reject, Z affects X)

Run: python experiments/ehs_instrument_validity_experiment.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import create_ci_engine
from causal_grounding.confounded_instrument_dgp import (
    ConfoundedInstrumentDGP,
    generate_confounded_instrument_data,
    compute_ground_truth_effects,
    create_scenario_1_valid_instrument,
    create_scenario_2_exclusion_violated,
    create_scenario_3_weak_instrument,
    create_scenario_4_confounder_vs_instrument,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'ehs_instrument_validity'

# Experiment parameters
N_REPLICATIONS = 200  # Number of datasets per condition
SAMPLE_SIZES = [500, 1000, 2000]
ALPHA = 0.05  # Significance level

# Plot styling
COLORS = {'CMI': '#1f77b4', 'LOCO': '#ff7f0e'}

# Scenario parameters
INSTRUMENT_STRENGTHS = [0.1, 0.3, 0.5, 1.0]  # alpha_z values
DIRECT_EFFECTS = [0.0, 0.2, 0.5]  # beta_z values (0 = valid, >0 = invalid)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_ehs_test_on_data(
    engine,
    data: pd.DataFrame,
    instrument: str,
    treatment: str,
    outcome: str,
    other_covariates: List[str]
) -> Dict[str, Any]:
    """
    Run EHS criteria tests on a single dataset.

    Returns:
        Dict with test results for all three EHS criteria.
    """
    start = time.time()

    result = engine.score_ehs_criteria(
        data=data,
        z_a=instrument,
        z_b=other_covariates,
        treatment=treatment,
        outcome=outcome,
        use_permutation_test=True
    )

    elapsed = time.time() - start
    result['runtime'] = elapsed

    return result


def run_scenario(
    engine,
    scenario_num: int,
    dgp_factory,
    n_samples: int,
    n_replications: int,
    method_name: str,
    extra_fields: Dict[str, Any],
    desc_suffix: str = ""
) -> List[Dict]:
    """
    Generic scenario runner for scenarios 1-3.

    Args:
        engine: CI test engine (CMI or LOCO)
        scenario_num: Scenario number (1, 2, or 3)
        dgp_factory: Callable that returns a DGP instance
        n_samples: Number of samples per dataset
        n_replications: Number of replications
        method_name: Name of the method ('CMI' or 'LOCO')
        extra_fields: Additional fields to include in results
        desc_suffix: Suffix for tqdm description
    """
    SCENARIO_NAMES = {1: 'Valid Instrument', 2: 'Exclusion Violated', 3: 'Weak Instrument'}

    results = []
    dgp = dgp_factory()
    ground_truth = compute_ground_truth_effects(dgp)

    for rep in tqdm(range(n_replications), desc=f"  {method_name} S{scenario_num}{desc_suffix}", leave=False):
        data = generate_confounded_instrument_data(dgp, n_samples, seed=rep)

        try:
            ehs_result = run_ehs_test_on_data(
                engine, data,
                instrument='Z',
                treatment='X',
                outcome='Y',
                other_covariates=[]
            )

            results.append({
                'scenario': scenario_num,
                'scenario_name': SCENARIO_NAMES[scenario_num],
                'method': method_name,
                'n_samples': n_samples,
                'replication': rep,
                'ground_truth_ate': ground_truth['ate'],
                **extra_fields,
                **{f'ehs_{k}': v for k, v in ehs_result.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S{scenario_num} rep {rep}: {e}")

    return results


def run_scenario_1_valid_instrument(engine, n_samples: int, alpha_z: float,
                                     n_replications: int, method_name: str) -> List[Dict]:
    """Run Scenario 1: Valid Instrument tests."""
    return run_scenario(
        engine, 1,
        lambda: create_scenario_1_valid_instrument(alpha_z=alpha_z),
        n_samples, n_replications, method_name,
        {'alpha_z': alpha_z, 'beta_z': 0.0, 'is_valid_instrument': True},
        f" (alpha_z={alpha_z})"
    )


def run_scenario_2_exclusion_violated(engine, n_samples: int, beta_z: float,
                                       n_replications: int, method_name: str) -> List[Dict]:
    """Run Scenario 2: Invalid Instrument (exclusion violated) tests."""
    return run_scenario(
        engine, 2,
        lambda: create_scenario_2_exclusion_violated(beta_z=beta_z),
        n_samples, n_replications, method_name,
        {'alpha_z': 1.0, 'beta_z': beta_z, 'is_valid_instrument': beta_z == 0.0},
        f" (beta_z={beta_z})"
    )


def run_scenario_3_weak_instrument(engine, n_samples: int, alpha_z: float,
                                    n_replications: int, method_name: str) -> List[Dict]:
    """Run Scenario 3: Weak Instrument tests."""
    return run_scenario(
        engine, 3,
        lambda: create_scenario_3_weak_instrument(alpha_z=alpha_z),
        n_samples, n_replications, method_name,
        {'alpha_z': alpha_z, 'beta_z': 0.0, 'is_valid_instrument': True, 'is_weak_instrument': alpha_z < 0.3},
        f" (alpha_z={alpha_z})"
    )


def run_scenario_4_confounder_vs_instrument(
    engine,
    n_samples: int,
    n_replications: int,
    method_name: str
) -> List[Dict]:
    """Run Scenario 4: Discriminate Confounder vs Instrument."""
    results = []

    dgp = create_scenario_4_confounder_vs_instrument()
    ground_truth = compute_ground_truth_effects(dgp)

    for rep in tqdm(range(n_replications), desc=f"  {method_name} S4", leave=False):
        data = generate_confounded_instrument_data(dgp, n_samples, seed=rep)

        # Test Z as candidate instrument (should pass EHS)
        try:
            ehs_result_z = run_ehs_test_on_data(
                engine, data,
                instrument='Z',
                treatment='X',
                outcome='Y',
                other_covariates=['W']
            )

            results.append({
                'scenario': 4,
                'scenario_name': 'Confounder vs Instrument',
                'candidate_type': 'Instrument',
                'candidate_var': 'Z',
                'method': method_name,
                'n_samples': n_samples,
                'replication': rep,
                'is_valid_instrument': True,  # Z is valid instrument
                'ground_truth_ate': ground_truth['ate'],
                **{f'ehs_{k}': v for k, v in ehs_result_z.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S4 (Z) rep {rep}: {e}")

        # Test W as candidate instrument (should fail EHS - W is confounder)
        try:
            ehs_result_w = run_ehs_test_on_data(
                engine, data,
                instrument='W',
                treatment='X',
                outcome='Y',
                other_covariates=['Z']
            )

            results.append({
                'scenario': 4,
                'scenario_name': 'Confounder vs Instrument',
                'candidate_type': 'Confounder',
                'candidate_var': 'W',
                'method': method_name,
                'n_samples': n_samples,
                'replication': rep,
                'is_valid_instrument': False,  # W is NOT a valid instrument
                'ground_truth_ate': ground_truth['ate'],
                **{f'ehs_{k}': v for k, v in ehs_result_w.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S4 (W) rep {rep}: {e}")

    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_ehs_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy of EHS classification."""

    def accuracy_metrics(group):
        # True label: is_valid_instrument
        # Predicted: ehs_passes_full_ehs or ehs_passes_ehs

        true_valid = group['is_valid_instrument']
        pred_valid = group['ehs_passes_full_ehs']

        tp = ((true_valid == True) & (pred_valid == True)).sum()
        tn = ((true_valid == False) & (pred_valid == False)).sum()
        fp = ((true_valid == False) & (pred_valid == True)).sum()
        fn = ((true_valid == True) & (pred_valid == False)).sum()

        total = len(group)
        accuracy = (tp + tn) / total if total > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        return pd.Series({
            'n': total,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        })

    return results_df.groupby(['scenario', 'method', 'n_samples']).apply(accuracy_metrics).reset_index()


def compute_test_performance(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute individual test performance (Type I/II error)."""

    metrics = []

    for (scenario, method, n_samples), group in results_df.groupby(['scenario', 'method', 'n_samples']):
        # Test (i) performance: Should NOT reject for valid instruments
        valid_group = group[group['is_valid_instrument'] == True]
        if len(valid_group) > 0:
            # For valid instruments, rejecting test (i) is a false positive
            test_i_fpr = valid_group['ehs_test_i_reject'].mean()
        else:
            test_i_fpr = np.nan

        # Test (i) for invalid instruments (should reject)
        invalid_group = group[group['is_valid_instrument'] == False]
        if len(invalid_group) > 0:
            # For invalid instruments, rejecting test (i) is a true positive
            test_i_tpr = invalid_group['ehs_test_i_reject'].mean()
        else:
            test_i_tpr = np.nan

        # Test (i.a) - instrument relevance
        if 'is_weak_instrument' in group.columns:
            weak_group = group[group.get('is_weak_instrument', False) == True]
            strong_group = group[group.get('is_weak_instrument', False) == False]
            test_ia_weak_power = weak_group['ehs_test_ia_reject'].mean() if len(weak_group) > 0 else np.nan
            test_ia_strong_power = strong_group['ehs_test_ia_reject'].mean() if len(strong_group) > 0 else np.nan
        else:
            test_ia_weak_power = np.nan
            test_ia_strong_power = valid_group['ehs_test_ia_reject'].mean() if len(valid_group) > 0 else np.nan

        metrics.append({
            'scenario': scenario,
            'method': method,
            'n_samples': n_samples,
            'test_i_false_positive_rate': test_i_fpr,
            'test_i_true_positive_rate': test_i_tpr,
            'test_ia_power_weak': test_ia_weak_power,
            'test_ia_power_strong': test_ia_strong_power,
            'mean_runtime': group['ehs_runtime'].mean()
        })

    return pd.DataFrame(metrics)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_by_method(ax, df: pd.DataFrame, groupby_col: str, value_col: str,
                   marker: str = 'o', linewidth: float = 2.5, markersize: int = 8):
    """Plot grouped data by method with consistent CMI vs LOCO styling."""
    for method in ['CMI', 'LOCO']:
        if method in df['method'].values:
            data = df[df['method'] == method].groupby(groupby_col)[value_col].mean()
            ax.plot(data.index, data.values, marker=marker, label=method,
                   linewidth=linewidth, markersize=markersize, color=COLORS.get(method))


def plot_scenario_1_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot Scenario 1: Valid instrument detection by strength - multiple plots."""
    s1 = results_df[results_df['scenario'] == 1]
    if len(s1) == 0:
        print("  No Scenario 1 results to plot")
        return

    # Plot 1: All three EHS tests power curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    plot_by_method(ax1, s1, 'alpha_z', 'ehs_test_i_reject')
    ax1.axhline(y=ALPHA, color='red', linestyle='--', linewidth=2, label=f'Nominal α={ALPHA}')
    ax1.axhspan(0.03, 0.07, color='green', alpha=0.1, label='Acceptable range')
    ax1.set_xlabel('Instrument Strength (α_z)', fontsize=12)
    ax1.set_ylabel('Rejection Rate', fontsize=12)
    ax1.set_title('Test (i): Exogeneity [Y ⊥ Z | X]\n(Should NOT reject)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.25)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    plot_by_method(ax2, s1, 'alpha_z', 'ehs_test_ia_reject')
    ax2.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='80% Power')
    ax2.axhline(y=ALPHA, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Instrument Strength (α_z)', fontsize=12)
    ax2.set_ylabel('Power (Rejection Rate)', fontsize=12)
    ax2.set_title('Test (i.a): Instrument Relevance [X ⊥̸ Z]\n(Power curve)', fontsize=11)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    plot_by_method(ax3, s1, 'alpha_z', 'ehs_passes_full_ehs', marker='s')
    ax3.set_xlabel('Instrument Strength (α_z)', fontsize=12)
    ax3.set_ylabel('Pass Rate', fontsize=12)
    ax3.set_title('Full EHS Pass Rate\n(All 3 tests pass)', fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Scenario 1: Valid Instrument - CMI vs LOCO Power Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path1 = output_dir / 'scenario1_valid_instrument_cmi_vs_loco_power_curves.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1.name}")

    # Plot 2: Test (ii) outcome relevance power curve
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_by_method(ax, s1, 'alpha_z', 'ehs_test_ii_reject', markersize=10)
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='80% Power')
    ax.axhline(y=ALPHA, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label=f'α={ALPHA}')
    ax.set_xlabel('Instrument Strength (α_z)', fontsize=13)
    ax.set_ylabel('Power (Rejection Rate)', fontsize=13)
    ax.set_title('Test (ii): Outcome Relevance [Y ⊥̸ Z] - CMI vs LOCO\n(Detecting indirect effect Z → X → Y)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = output_dir / 'scenario1_valid_instrument_cmi_vs_loco_outcome_relevance_power.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2.name}")


def plot_scenario_2_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot Scenario 2: Invalid instrument (exclusion violated) detection."""
    s2 = results_df[results_df['scenario'] == 2]
    if len(s2) == 0:
        print("  No Scenario 2 results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Test (i) rejection rate by direct effect (should increase)
    ax1 = axes[0]
    plot_by_method(ax1, s2, 'beta_z', 'ehs_test_i_reject', markersize=10)
    ax1.axhline(y=ALPHA, color='red', linestyle='--', linewidth=2, label=f'Type I error (α={ALPHA})')
    ax1.set_xlabel('Direct Effect of Z on Y (β_z)', fontsize=12)
    ax1.set_ylabel('Test (i) Rejection Rate', fontsize=12)
    ax1.set_title('Exogeneity Test Performance\n(Should reject when β_z > 0)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 0.5)
    ax1.grid(True, alpha=0.3)
    ax1.annotate('β_z = 0: Type I error\nβ_z > 0: Power to detect', xy=(0.25, 0.38), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 2: Correct rejection of invalid instruments
    ax2 = axes[1]
    for method in ['CMI', 'LOCO']:
        if method in s2['method'].values:
            data = s2[s2['method'] == method].groupby('beta_z')['ehs_passes_full_ehs'].mean()
            ax2.plot(data.index, 1 - data.values, marker='s', label=method,
                    linewidth=2.5, markersize=10, color=COLORS.get(method, None))
    ax2.set_xlabel('Direct Effect of Z on Y (β_z)', fontsize=12)
    ax2.set_ylabel('Correct Rejection Rate', fontsize=12)
    ax2.set_title('Invalid Instrument Rejection Rate\n(Higher = better detection)', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Scenario 2: Exclusion Violation Detection - CMI vs LOCO', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'scenario2_exclusion_violation_cmi_vs_loco_detection_power.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


def plot_scenario_4_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot Scenario 4: Confounder vs Instrument discrimination."""
    s4 = results_df[results_df['scenario'] == 4]
    if len(s4) == 0:
        print("  No Scenario 4 results to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Test (i) rejection by candidate type
    ax1 = axes[0]
    test_i_data = s4.groupby(['method', 'candidate_type'])['ehs_test_i_reject'].mean().unstack()
    x = np.arange(len(test_i_data.index))
    width = 0.35
    ax1.bar(x - width/2, test_i_data['Confounder'], width, label='Confounder (W)', color='#d62728')
    ax1.bar(x + width/2, test_i_data['Instrument'], width, label='Instrument (Z)', color='#2ca02c')
    ax1.axhline(y=ALPHA, color='black', linestyle='--', linewidth=1.5, label=f'α={ALPHA}')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Test (i) Rejection Rate', fontsize=12)
    ax1.set_title('Exogeneity Test [Y ⊥ Z | X]\n(Confounder FAIL, Instrument PASS)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_i_data.index)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Full EHS pass rate by candidate type
    ax2 = axes[1]
    pass_data = s4.groupby(['method', 'candidate_type'])['ehs_passes_full_ehs'].mean().unstack()
    ax2.bar(x - width/2, pass_data['Confounder'], width, label='Confounder (W)', color='#d62728')
    ax2.bar(x + width/2, pass_data['Instrument'], width, label='Instrument (Z)', color='#2ca02c')
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Full EHS Pass Rate', fontsize=12)
    ax2.set_title('Full EHS Pass Rate\n(Instrument PASS, Confounder FAIL)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pass_data.index)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Discrimination accuracy
    ax3 = axes[2]
    accuracy = {}
    for method in s4['method'].unique():
        method_df = s4[s4['method'] == method]
        inst_test_i_pass = 1 - method_df[method_df['candidate_type'] == 'Instrument']['ehs_test_i_reject'].mean()
        conf_test_i_fail = method_df[method_df['candidate_type'] == 'Confounder']['ehs_test_i_reject'].mean()
        accuracy[method] = (inst_test_i_pass + conf_test_i_fail) / 2
    bars = ax3.bar(accuracy.keys(), accuracy.values(), color=[COLORS.get(m, 'gray') for m in accuracy.keys()])
    ax3.set_xlabel('Method', fontsize=12)
    ax3.set_ylabel('Discrimination Accuracy', fontsize=12)
    ax3.set_title('Confounder vs Instrument Discrimination\n(Higher = better separation)', fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, accuracy.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('Scenario 4: Confounder vs Instrument Discrimination - CMI vs LOCO', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'scenario4_confounder_vs_instrument_cmi_vs_loco_discrimination.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


def plot_method_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Plot overall method comparison - comprehensive summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Runtime comparison
    ax1 = axes[0, 0]
    runtime_data = results_df.groupby('method')['ehs_runtime'].agg(['mean', 'std'])
    bars = ax1.bar(runtime_data.index, runtime_data['mean'],
                   yerr=runtime_data['std'], capsize=5, color=[COLORS.get(m, 'gray') for m in runtime_data.index])
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Mean Runtime per EHS Evaluation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, runtime_data['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=11)

    # 2. Type I error (test i under valid instruments - Scenario 1)
    ax2 = axes[0, 1]
    s1 = results_df[results_df['scenario'] == 1]
    if len(s1) > 0:
        type1_data = s1.groupby('method')['ehs_test_i_reject'].mean()
        bars = ax2.bar(type1_data.index, type1_data.values, color=[COLORS.get(m, 'gray') for m in type1_data.index])
        ax2.axhline(y=ALPHA, color='red', linestyle='--', linewidth=2, label=f'Nominal α={ALPHA}')
        ax2.axhspan(0.03, 0.07, color='green', alpha=0.2, label='Acceptable range')
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Type I Error Rate', fontsize=12)
        ax2.set_title('Test (i) Type I Error\n(Valid Instruments - Scenario 1)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 0.15)
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Confounder detection (Scenario 4)
    ax3 = axes[1, 0]
    s4 = results_df[results_df['scenario'] == 4]
    if len(s4) > 0:
        conf_detect = s4[s4['candidate_type'] == 'Confounder'].groupby('method')['ehs_test_i_reject'].mean()
        bars = ax3.bar(conf_detect.index, conf_detect.values, color=[COLORS.get(m, 'gray') for m in conf_detect.index])
        ax3.set_xlabel('Method', fontsize=12)
        ax3.set_ylabel('Confounder Detection Rate', fontsize=12)
        ax3.set_title('Confounder Detection via Test (i)\n(Scenario 4 - Higher = Better)', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, conf_detect.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Instrument relevance power (Scenario 1, alpha_z=1.0)
    ax4 = axes[1, 1]
    if len(s1) > 0:
        s1_strong = s1[s1['alpha_z'] == s1['alpha_z'].max()]
        power_data = s1_strong.groupby('method')['ehs_test_ia_reject'].mean()
        bars = ax4.bar(power_data.index, power_data.values, color=[COLORS.get(m, 'gray') for m in power_data.index])
        ax4.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='80% Power target')
        ax4.set_xlabel('Method', fontsize=12)
        ax4.set_ylabel('Power', fontsize=12)
        ax4.set_title('Test (i.a) Power at Max Instrument Strength\n(Instrument Relevance Detection)', fontsize=13, fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, power_data.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle('CMI vs LOCO: Overall EHS Criteria Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'cmi_vs_loco_overall_ehs_comparison_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


def plot_all_power_curves_combined(results_df: pd.DataFrame, output_dir: Path):
    """Plot all EHS test power curves in a single figure."""
    s1 = results_df[results_df['scenario'] == 1]
    if len(s1) == 0:
        print("  No Scenario 1 data for combined power curves")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    tests = [
        ('ehs_test_i_reject', 'Test (i): Exogeneity [Y ⊥ Z | X]', 'Type I Error (should be ~5%)', 0.25, True),
        ('ehs_test_ii_reject', 'Test (ii): Outcome Relevance [Y ⊥̸ Z]', 'Power curve', 1.05, False),
        ('ehs_test_ia_reject', 'Test (i.a): Instrument Relevance [X ⊥̸ Z]', 'Power curve', 1.05, False),
        ('ehs_passes_full_ehs', 'Full EHS: All 3 Tests Pass', 'Valid instrument detection rate', 1.05, False)
    ]

    for idx, (col, title, subtitle, ylim, is_type1) in enumerate(tests):
        ax = axes[idx // 2, idx % 2]
        plot_by_method(ax, s1, 'alpha_z', col)

        if is_type1:
            ax.axhline(y=ALPHA, color='red', linestyle='--', linewidth=2, label=f'α={ALPHA}')
            ax.axhspan(0.03, 0.07, color='green', alpha=0.1)
        else:
            ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='80% Power')
            ax.axhline(y=ALPHA, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Instrument Strength (α_z)', fontsize=11)
        ax.set_ylabel('Rate', fontsize=11)
        ax.set_title(f'{title}\n({subtitle})', fontsize=11)
        ax.legend(loc='best')
        ax.set_ylim(0, ylim)
        ax.grid(True, alpha=0.3)

    plt.suptitle('CMI vs LOCO: All EHS Test Power Curves (Scenario 1 - Valid Instruments)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'cmi_vs_loco_all_ehs_tests_power_curves_combined.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the full EHS instrument validity experiment."""
    print("=" * 70)
    print("EHS INSTRUMENT VALIDITY EXPERIMENT")
    print("=" * 70)

    # Create engines
    print("\nCreating CI engines...")
    cmi_engine = create_ci_engine('cmi', n_permutations=200, random_seed=42)
    loco_engine = create_ci_engine('loco', function_class='gbm',
                                    n_estimators=50, max_depth=2, random_state=42)

    engines = [
        ('CMI', cmi_engine),
        ('LOCO', loco_engine)
    ]

    print(f"  CMI: {type(cmi_engine).__name__}")
    print(f"  LOCO: {type(loco_engine).__name__}")

    all_results = []

    # Scenario 1: Valid Instrument with varying strength
    print("\n" + "-" * 70)
    print("SCENARIO 1: Valid Instrument (varying instrument strength)")
    print("-" * 70)
    for method_name, engine in engines:
        for n_samples in SAMPLE_SIZES[:1]:  # Use first sample size for quick run
            for alpha_z in INSTRUMENT_STRENGTHS:
                results = run_scenario_1_valid_instrument(
                    engine, n_samples, alpha_z, N_REPLICATIONS, method_name
                )
                all_results.extend(results)

    # Scenario 2: Invalid Instrument (exclusion violated)
    print("\n" + "-" * 70)
    print("SCENARIO 2: Invalid Instrument (exclusion violated)")
    print("-" * 70)
    for method_name, engine in engines:
        for n_samples in SAMPLE_SIZES[:1]:
            for beta_z in DIRECT_EFFECTS:
                results = run_scenario_2_exclusion_violated(
                    engine, n_samples, beta_z, N_REPLICATIONS, method_name
                )
                all_results.extend(results)

    # Scenario 3: Weak Instrument
    print("\n" + "-" * 70)
    print("SCENARIO 3: Weak Instrument")
    print("-" * 70)
    for method_name, engine in engines:
        for n_samples in SAMPLE_SIZES[:1]:
            for alpha_z in INSTRUMENT_STRENGTHS:
                results = run_scenario_3_weak_instrument(
                    engine, n_samples, alpha_z, N_REPLICATIONS, method_name
                )
                all_results.extend(results)

    # Scenario 4: Confounder vs Instrument
    print("\n" + "-" * 70)
    print("SCENARIO 4: Confounder vs Instrument Discrimination")
    print("-" * 70)
    for method_name, engine in engines:
        for n_samples in SAMPLE_SIZES[:1]:
            results = run_scenario_4_confounder_vs_instrument(
                engine, n_samples, N_REPLICATIONS, method_name
            )
            all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    print(f"\nTotal results: {len(results_df)} rows")

    return results_df


def main():
    """Run experiment and generate all outputs."""

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results_df = run_experiment()

    # Save raw results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save scenario-specific results
    for scenario in [1, 2, 3, 4]:
        scenario_df = results_df[results_df['scenario'] == scenario]
        if len(scenario_df) > 0:
            scenario_names = {
                1: 'scenario_1_valid_instrument_results.csv',
                2: 'scenario_2_exclusion_violation_results.csv',
                3: 'scenario_3_weak_instrument_results.csv',
                4: 'scenario_4_confounder_vs_instrument_results.csv'
            }
            path = OUTPUT_DIR / scenario_names[scenario]
            scenario_df.to_csv(path, index=False)
            print(f"  Saved: {path}")

    # Compute and save accuracy summary
    accuracy_df = compute_ehs_accuracy(results_df)
    accuracy_df.to_csv(OUTPUT_DIR / 'ehs_test_accuracy_summary.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'ehs_test_accuracy_summary.csv'}")

    # Compute and save test performance
    test_perf_df = compute_test_performance(results_df)
    test_perf_df.to_csv(OUTPUT_DIR / 'loco_vs_cmi_ehs_comparison.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'loco_vs_cmi_ehs_comparison.csv'}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_scenario_1_results(results_df, OUTPUT_DIR)
    plot_scenario_2_results(results_df, OUTPUT_DIR)
    plot_scenario_4_results(results_df, OUTPUT_DIR)
    plot_method_comparison(results_df, OUTPUT_DIR)
    plot_all_power_curves_combined(results_df, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Scenario 1 summary
    s1 = results_df[results_df['scenario'] == 1]
    if len(s1) > 0:
        print("\nScenario 1: Valid Instrument")
        print("-" * 40)
        s1_summary = s1.groupby(['method', 'alpha_z']).agg({
            'ehs_test_i_reject': 'mean',
            'ehs_test_ia_reject': 'mean',
            'ehs_passes_full_ehs': 'mean'
        }).round(3)
        print(s1_summary.to_string())

    # Scenario 2 summary
    s2 = results_df[results_df['scenario'] == 2]
    if len(s2) > 0:
        print("\nScenario 2: Invalid Instrument (Exclusion Violated)")
        print("-" * 40)
        s2_summary = s2.groupby(['method', 'beta_z']).agg({
            'ehs_test_i_reject': 'mean',
            'ehs_passes_full_ehs': 'mean'
        }).round(3)
        print(s2_summary.to_string())

    # Scenario 4 summary
    s4 = results_df[results_df['scenario'] == 4]
    if len(s4) > 0:
        print("\nScenario 4: Confounder vs Instrument")
        print("-" * 40)
        s4_summary = s4.groupby(['method', 'candidate_type']).agg({
            'ehs_test_i_reject': 'mean',
            'ehs_passes_full_ehs': 'mean'
        }).round(3)
        print(s4_summary.to_string())

    # Overall method comparison
    print("\nOverall Runtime Comparison:")
    print("-" * 40)
    runtime_summary = results_df.groupby('method')['ehs_runtime'].agg(['mean', 'std']).round(3)
    print(runtime_summary.to_string())

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")

    return results_df


if __name__ == "__main__":
    main()
