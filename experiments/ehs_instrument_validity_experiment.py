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
from scipy import stats
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
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

OUTPUT_DIR = Path(__file__).parent / 'experiment_results' / 'ehs_instrument_validity'
FIGURES_DIR = OUTPUT_DIR

# Experiment parameters
N_REPLICATIONS = 200  # Number of datasets per condition
SAMPLE_SIZES = [500, 1000, 2000]
ALPHA = 0.05  # Significance level

# Scenario parameters
INSTRUMENT_STRENGTHS = [0.1, 0.3, 0.5, 1.0]  # alpha_z values
DIRECT_EFFECTS = [0.0, 0.2, 0.5]  # beta_z values (0 = valid, >0 = invalid)
CONFOUNDING_STRENGTHS = [0.5, 1.0]

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


def run_scenario_1_valid_instrument(
    engine,
    n_samples: int,
    alpha_z: float,
    n_replications: int,
    method_name: str
) -> List[Dict]:
    """Run Scenario 1: Valid Instrument tests."""
    results = []

    dgp = create_scenario_1_valid_instrument(alpha_z=alpha_z)
    ground_truth = compute_ground_truth_effects(dgp)

    for rep in tqdm(range(n_replications), desc=f"  {method_name} S1 (alpha_z={alpha_z})", leave=False):
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
                'scenario': 1,
                'scenario_name': 'Valid Instrument',
                'method': method_name,
                'n_samples': n_samples,
                'alpha_z': alpha_z,
                'beta_z': 0.0,
                'replication': rep,
                'is_valid_instrument': True,  # Ground truth
                'ground_truth_ate': ground_truth['ate'],
                **{f'ehs_{k}': v for k, v in ehs_result.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S1 rep {rep}: {e}")

    return results


def run_scenario_2_exclusion_violated(
    engine,
    n_samples: int,
    beta_z: float,
    n_replications: int,
    method_name: str
) -> List[Dict]:
    """Run Scenario 2: Invalid Instrument (exclusion violated) tests."""
    results = []

    dgp = create_scenario_2_exclusion_violated(beta_z=beta_z)
    ground_truth = compute_ground_truth_effects(dgp)

    for rep in tqdm(range(n_replications), desc=f"  {method_name} S2 (beta_z={beta_z})", leave=False):
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
                'scenario': 2,
                'scenario_name': 'Exclusion Violated',
                'method': method_name,
                'n_samples': n_samples,
                'alpha_z': 1.0,  # Default strength
                'beta_z': beta_z,
                'replication': rep,
                'is_valid_instrument': beta_z == 0.0,  # Ground truth
                'ground_truth_ate': ground_truth['ate'],
                **{f'ehs_{k}': v for k, v in ehs_result.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S2 rep {rep}: {e}")

    return results


def run_scenario_3_weak_instrument(
    engine,
    n_samples: int,
    alpha_z: float,
    n_replications: int,
    method_name: str
) -> List[Dict]:
    """Run Scenario 3: Weak Instrument tests."""
    results = []

    dgp = create_scenario_3_weak_instrument(alpha_z=alpha_z)
    ground_truth = compute_ground_truth_effects(dgp)

    for rep in tqdm(range(n_replications), desc=f"  {method_name} S3 (alpha_z={alpha_z})", leave=False):
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
                'scenario': 3,
                'scenario_name': 'Weak Instrument',
                'method': method_name,
                'n_samples': n_samples,
                'alpha_z': alpha_z,
                'beta_z': 0.0,
                'replication': rep,
                'is_valid_instrument': True,  # Valid but weak
                'is_weak_instrument': alpha_z < 0.3,  # Ground truth
                'ground_truth_ate': ground_truth['ate'],
                **{f'ehs_{k}': v for k, v in ehs_result.items() if k != 'z_a'}
            })
        except Exception as e:
            print(f"  Error in S3 rep {rep}: {e}")

    return results


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

def plot_scenario_1_results(results_df: pd.DataFrame, output_path: Path):
    """Plot Scenario 1: Valid instrument detection by strength."""
    s1 = results_df[results_df['scenario'] == 1]
    if len(s1) == 0:
        print("  No Scenario 1 results to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Test (i) rejection rate by instrument strength (should be low)
    ax1 = axes[0]
    for method in s1['method'].unique():
        method_data = s1[s1['method'] == method].groupby('alpha_z')['ehs_test_i_reject'].mean()
        ax1.plot(method_data.index, method_data.values, marker='o', label=method, linewidth=2)
    ax1.axhline(y=ALPHA, color='red', linestyle='--', label=f'Nominal alpha={ALPHA}')
    ax1.set_xlabel('Instrument Strength (alpha_z)')
    ax1.set_ylabel('Test (i) Rejection Rate')
    ax1.set_title('Exogeneity Test (Should NOT Reject for Valid)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test (i.a) rejection rate by instrument strength (power)
    ax2 = axes[1]
    for method in s1['method'].unique():
        method_data = s1[s1['method'] == method].groupby('alpha_z')['ehs_test_ia_reject'].mean()
        ax2.plot(method_data.index, method_data.values, marker='o', label=method, linewidth=2)
    ax2.axhline(y=0.8, color='green', linestyle='--', label='80% Power')
    ax2.set_xlabel('Instrument Strength (alpha_z)')
    ax2.set_ylabel('Test (i.a) Rejection Rate')
    ax2.set_title('Instrument Relevance Test (Power)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Full EHS pass rate
    ax3 = axes[2]
    for method in s1['method'].unique():
        method_data = s1[s1['method'] == method].groupby('alpha_z')['ehs_passes_full_ehs'].mean()
        ax3.plot(method_data.index, method_data.values, marker='o', label=method, linewidth=2)
    ax3.set_xlabel('Instrument Strength (alpha_z)')
    ax3.set_ylabel('Pass Rate')
    ax3.set_title('Full EHS Pass Rate (Valid Instruments)')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Scenario 1: Valid Instrument Detection', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scenario_2_results(results_df: pd.DataFrame, output_path: Path):
    """Plot Scenario 2: Invalid instrument (exclusion violated) detection."""
    s2 = results_df[results_df['scenario'] == 2]
    if len(s2) == 0:
        print("  No Scenario 2 results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Test (i) rejection rate by direct effect (should increase)
    ax1 = axes[0]
    for method in s2['method'].unique():
        method_data = s2[s2['method'] == method].groupby('beta_z')['ehs_test_i_reject'].mean()
        ax1.plot(method_data.index, method_data.values, marker='o', label=method, linewidth=2)
    ax1.axhline(y=ALPHA, color='red', linestyle='--', label=f'Nominal alpha={ALPHA}')
    ax1.set_xlabel('Direct Effect on Y (beta_z)')
    ax1.set_ylabel('Test (i) Rejection Rate')
    ax1.set_title('Exogeneity Test (Should Reject When beta_z > 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Full EHS rejection rate (correctly rejecting invalid instruments)
    ax2 = axes[1]
    # Invalid instruments should fail EHS (passes_full_ehs = False)
    for method in s2['method'].unique():
        method_data = s2[s2['method'] == method].groupby('beta_z')['ehs_passes_full_ehs'].mean()
        # For invalid instruments (beta_z > 0), we want LOW pass rate
        ax2.plot(method_data.index, 1 - method_data.values, marker='o', label=method, linewidth=2)
    ax2.set_xlabel('Direct Effect on Y (beta_z)')
    ax2.set_ylabel('Correct Rejection Rate')
    ax2.set_title('Invalid Instrument Rejection Rate')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Scenario 2: Invalid Instrument Detection (Exclusion Violated)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scenario_4_results(results_df: pd.DataFrame, output_path: Path):
    """Plot Scenario 4: Confounder vs Instrument discrimination."""
    s4 = results_df[results_df['scenario'] == 4]
    if len(s4) == 0:
        print("  No Scenario 4 results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: EHS pass rate by candidate type
    ax1 = axes[0]
    pass_rates = s4.groupby(['method', 'candidate_type'])['ehs_passes_full_ehs'].mean().unstack()
    pass_rates.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('EHS Pass Rate')
    ax1.set_title('EHS Pass Rate by Candidate Type')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(title='Candidate Type')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Test (i) rejection rate comparison
    ax2 = axes[1]
    test_i_rates = s4.groupby(['method', 'candidate_type'])['ehs_test_i_reject'].mean().unstack()
    test_i_rates.plot(kind='bar', ax=ax2, width=0.7)
    ax2.axhline(y=ALPHA, color='red', linestyle='--', label=f'alpha={ALPHA}')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Test (i) Rejection Rate')
    ax2.set_title('Exogeneity Test: Instrument Should Pass, Confounder Should Fail')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(title='Candidate Type')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Scenario 4: Confounder vs Instrument Discrimination', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_method_comparison(results_df: pd.DataFrame, output_path: Path):
    """Plot overall method comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compute accuracy for valid vs invalid classification
    results_with_validity = results_df[results_df['is_valid_instrument'].notna()].copy()

    # Plot 1: Overall accuracy by method and sample size
    ax1 = axes[0]
    accuracy = results_with_validity.groupby(['method', 'n_samples']).apply(
        lambda g: ((g['is_valid_instrument'] == g['ehs_passes_full_ehs']).sum()) / len(g)
    ).unstack(level=0)
    accuracy.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Instrument Classification Accuracy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(title='Method')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Runtime comparison
    ax2 = axes[1]
    runtime = results_df.groupby(['method', 'n_samples'])['ehs_runtime'].mean().unstack(level=0)
    runtime.plot(kind='bar', ax=ax2, width=0.7)
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Mean Runtime per EHS Evaluation')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(title='Method')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('LOCO vs CMI: EHS Criteria Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


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

    plot_scenario_1_results(results_df, OUTPUT_DIR / 'ehs_test_power_by_effect_size.png')
    plot_scenario_2_results(results_df, OUTPUT_DIR / 'ehs_test_type1_error_calibration.png')
    plot_scenario_4_results(results_df, OUTPUT_DIR / 'valid_vs_invalid_instrument_separation.png')
    plot_method_comparison(results_df, OUTPUT_DIR / 'loco_vs_cmi_ehs_overall_comparison.png')

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
