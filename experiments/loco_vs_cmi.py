"""
LOCO vs CMI Comparison on Synthetic Data

Compares statistical properties of LOCO and CMI conditional independence tests:
- Type I error (null true scenario)
- Statistical power (null false scenarios)
- Runtime performance
- P-value calibration

Run: python experiments/loco_vs_cmi_synthetic_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import create_ci_engine


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'

# Experiment parameters
N_REPLICATIONS = 100  # Number of datasets per condition
SAMPLE_SIZES = [200, 500, 1000]
N_COVARIATES_LIST = [2, 5]
EFFECT_SIZES = [0.0, 0.3, 0.7, 1.2]  # 0 = null true
ALPHA = 0.05  # Significance level

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_ci_test_data(
    n_samples: int,
    n_covariates: int,
    effect_size: float,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic data for CI testing.

    Tests: Y _||_ X | W

    Args:
        n_samples: Number of samples
        n_covariates: Number of conditioning covariates
        effect_size: Effect of X on Y (0 = null true, >0 = null false)
        seed: Random seed

    Returns:
        DataFrame with X, Y, W1, W2, ..., Wk
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate conditioning covariates W (categorical)
    W = {}
    for i in range(n_covariates):
        W[f'W{i+1}'] = np.random.choice([0, 1, 2], size=n_samples)

    # Generate X (binary)
    X = np.random.binomial(1, 0.5, n_samples)

    # Generate Y depending on W and possibly X
    logit = -0.5
    for w_col in W.values():
        logit = logit + 0.3 * (w_col == 1) + 0.2 * (w_col == 2)

    # Add X effect (0 for null true, >0 for null false)
    logit = logit + effect_size * X

    prob_Y = 1 / (1 + np.exp(-logit))
    Y = np.random.binomial(1, prob_Y)

    # Build dataframe
    df = pd.DataFrame({'X': X, 'Y': Y})
    for name, vals in W.items():
        df[name] = vals

    return df


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_test(engine, df, conditioning_cols):
    """Run a single CI test and return results with timing."""
    start = time.time()
    result = engine.test_conditional_independence(df, 'X', 'Y', conditioning_cols)
    elapsed = time.time() - start
    return {
        'p_value': result['p_value'],
        'reject': result['reject_independence'],
        'cmi': result['cmi'],
        'runtime': elapsed
    }


def run_experiment():
    """Run the full comparison experiment."""
    print("=" * 60)
    print("EXPERIMENT 1: LOCO vs CMI on Synthetic Data")
    print("=" * 60)

    # Create engines
    print("\nCreating CI engines...")
    cmi_engine = create_ci_engine('cmi', n_permutations=200, random_seed=42)
    loco_engine = create_ci_engine('loco', function_class='gbm',
                                    n_estimators=50, max_depth=2, random_state=42)

    print(f"  CMI: {type(cmi_engine).__name__}")
    print(f"  LOCO: {type(loco_engine).__name__}")

    # Run experiments
    print(f"\nRunning experiments...")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  N covariates: {N_COVARIATES_LIST}")
    print(f"  Effect sizes: {EFFECT_SIZES}")
    print(f"  Replications: {N_REPLICATIONS}")

    results = []
    total_conditions = len(SAMPLE_SIZES) * len(N_COVARIATES_LIST) * len(EFFECT_SIZES)
    condition_idx = 0

    for n_samples in SAMPLE_SIZES:
        for n_cov in N_COVARIATES_LIST:
            for effect in EFFECT_SIZES:
                condition_idx += 1
                print(f"\n[{condition_idx}/{total_conditions}] n={n_samples}, k={n_cov}, effect={effect}")

                for rep in tqdm(range(N_REPLICATIONS), desc="  Replications", leave=False):
                    # Generate data
                    df = generate_ci_test_data(n_samples, n_cov, effect, seed=rep*1000+n_samples)
                    conditioning_cols = [f'W{i+1}' for i in range(n_cov)]

                    # Run CMI test
                    try:
                        cmi_result = run_single_test(cmi_engine, df, conditioning_cols)
                        results.append({
                            'method': 'CMI',
                            'n_samples': n_samples,
                            'n_covariates': n_cov,
                            'effect_size': effect,
                            'replication': rep,
                            **cmi_result
                        })
                    except Exception as e:
                        print(f"  CMI error: {e}")

                    # Run LOCO test
                    try:
                        loco_result = run_single_test(loco_engine, df, conditioning_cols)
                        results.append({
                            'method': 'LOCO',
                            'n_samples': n_samples,
                            'n_covariates': n_cov,
                            'effect_size': effect,
                            'replication': rep,
                            **loco_result
                        })
                    except Exception as e:
                        print(f"  LOCO error: {e}")

    results_df = pd.DataFrame(results)
    print(f"\nTotal results: {len(results_df)} rows")

    return results_df


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_type1_error(results_df):
    """Analyze Type I error rates."""
    null_true = results_df[results_df['effect_size'] == 0]

    type1 = null_true.groupby(['method', 'n_samples', 'n_covariates'])['reject'].mean().reset_index()
    type1.columns = ['method', 'n_samples', 'n_covariates', 'type1_error']

    return type1


def analyze_power(results_df):
    """Analyze statistical power."""
    power = results_df.groupby(['method', 'n_samples', 'n_covariates', 'effect_size'])['reject'].mean().reset_index()
    power.columns = ['method', 'n_samples', 'n_covariates', 'effect_size', 'power']

    return power


def analyze_runtime(results_df):
    """Analyze runtime performance."""
    runtime = results_df.groupby(['method', 'n_samples', 'n_covariates'])['runtime'].agg(['mean', 'std']).reset_index()
    runtime.columns = ['method', 'n_samples', 'n_covariates', 'mean_runtime', 'std_runtime']

    return runtime


def plot_type1_error(type1_df, output_path):
    """Plot Type I error comparison."""
    fig, axes = plt.subplots(1, len(N_COVARIATES_LIST), figsize=(12, 5))

    colors = {'CMI': '#1f77b4', 'LOCO': '#ff7f0e'}

    for idx, n_cov in enumerate(N_COVARIATES_LIST):
        ax = axes[idx]
        subset = type1_df[type1_df['n_covariates'] == n_cov]

        for method in ['CMI', 'LOCO']:
            method_data = subset[subset['method'] == method]
            ax.plot(method_data['n_samples'], method_data['type1_error'],
                    marker='o', label=method, linewidth=2, markersize=8,
                    color=colors[method])

        ax.axhline(y=ALPHA, color='red', linestyle='--', label=f'Nominal α={ALPHA}', alpha=0.7)
        ax.axhspan(0.03, 0.07, color='green', alpha=0.1, label='Acceptable range')

        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('Type I Error Rate', fontsize=12)
        ax.set_title(f'k = {n_cov} Covariates', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 0.20)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Type I Error Rate: LOCO vs CMI', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_power_curves(power_df, output_path):
    """Plot power curves."""
    fig, axes = plt.subplots(len(N_COVARIATES_LIST), len(SAMPLE_SIZES),
                             figsize=(4*len(SAMPLE_SIZES), 4*len(N_COVARIATES_LIST)))

    colors = {'CMI': '#1f77b4', 'LOCO': '#ff7f0e'}

    for i, n_cov in enumerate(N_COVARIATES_LIST):
        for j, n_samples in enumerate(SAMPLE_SIZES):
            ax = axes[i, j] if len(N_COVARIATES_LIST) > 1 else axes[j]
            subset = power_df[(power_df['n_samples'] == n_samples) &
                              (power_df['n_covariates'] == n_cov)]

            for method in ['CMI', 'LOCO']:
                method_data = subset[subset['method'] == method]
                ax.plot(method_data['effect_size'], method_data['power'],
                        marker='o', label=method, linewidth=2, markersize=8,
                        color=colors[method])

            ax.axhline(y=ALPHA, color='red', linestyle='--', alpha=0.5, label=f'α={ALPHA}')
            ax.axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='80% power')

            ax.set_xlabel('Effect Size', fontsize=11)
            ax.set_ylabel('Power', fontsize=11)
            ax.set_title(f'n={n_samples}, k={n_cov}', fontsize=12)
            ax.legend(loc='lower right', fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Statistical Power: LOCO vs CMI', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_runtime_comparison(runtime_df, output_path):
    """Plot runtime comparison."""
    fig, axes = plt.subplots(1, len(N_COVARIATES_LIST), figsize=(12, 5))

    colors = {'CMI': '#1f77b4', 'LOCO': '#ff7f0e'}

    for idx, n_cov in enumerate(N_COVARIATES_LIST):
        ax = axes[idx]
        subset = runtime_df[runtime_df['n_covariates'] == n_cov]

        x = np.arange(len(SAMPLE_SIZES))
        width = 0.35

        cmi_data = subset[subset['method'] == 'CMI']
        loco_data = subset[subset['method'] == 'LOCO']

        ax.bar(x - width/2, cmi_data['mean_runtime'], width, label='CMI',
               yerr=cmi_data['std_runtime'], capsize=5, color=colors['CMI'])
        ax.bar(x + width/2, loco_data['mean_runtime'], width, label='LOCO',
               yerr=loco_data['std_runtime'], capsize=5, color=colors['LOCO'])

        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title(f'k = {n_cov} Covariates', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(SAMPLE_SIZES)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Runtime Comparison: LOCO vs CMI', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pvalue_calibration(results_df, output_path):
    """Plot p-value calibration under null."""
    null_true = results_df[results_df['effect_size'] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'CMI': '#1f77b4', 'LOCO': '#ff7f0e'}

    for idx, method in enumerate(['CMI', 'LOCO']):
        ax = axes[idx]
        method_null = null_true[null_true['method'] == method]

        ax.hist(method_null['p_value'], bins=20, density=True, alpha=0.7,
                edgecolor='black', color=colors[method])
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform (ideal)')

        ax.set_xlabel('P-value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{method}', fontsize=14)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

        # KS test for uniformity
        ks_stat, ks_pval = stats.kstest(method_null['p_value'], 'uniform')
        ax.text(0.95, 0.95, f'KS p-val: {ks_pval:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('P-value Distribution Under Null (Should be Uniform)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_summary_table(type1_df, power_df, runtime_df):
    """Create summary statistics table."""
    summary = []

    for method in ['CMI', 'LOCO']:
        method_type1 = type1_df[type1_df['method'] == method]['type1_error'].mean()
        method_runtime = runtime_df[runtime_df['method'] == method]['mean_runtime'].mean()

        # Power at effect_size = 0.7, n = 500
        power_subset = power_df[(power_df['method'] == method) &
                                (power_df['effect_size'] == 0.7) &
                                (power_df['n_samples'] == 500)]
        method_power = power_subset['power'].mean() if len(power_subset) > 0 else np.nan

        summary.append({
            'Method': method,
            'Mean Type I Error': f"{method_type1:.3f}",
            'Power @ effect=0.7, n=500': f"{method_power:.3f}",
            'Mean Runtime (s)': f"{method_runtime:.3f}"
        })

    return pd.DataFrame(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run experiment and generate all outputs."""

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results_df = run_experiment()

    # Save raw results
    results_path = OUTPUT_DIR / 'loco_vs_cmi_raw_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved raw results: {results_path}")

    # Analyze
    print("\nAnalyzing results...")
    type1_df = analyze_type1_error(results_df)
    power_df = analyze_power(results_df)
    runtime_df = analyze_runtime(results_df)

    # Save analysis
    type1_df.to_csv(OUTPUT_DIR / 'loco_vs_cmi_type1_error.csv', index=False)
    power_df.to_csv(OUTPUT_DIR / 'loco_vs_cmi_power.csv', index=False)
    runtime_df.to_csv(OUTPUT_DIR / 'loco_vs_cmi_runtime.csv', index=False)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_type1_error(type1_df, FIGURES_DIR / 'loco_vs_cmi_type1_error.png')
    plot_power_curves(power_df, FIGURES_DIR / 'loco_vs_cmi_power_curves.png')
    plot_runtime_comparison(runtime_df, FIGURES_DIR / 'loco_vs_cmi_runtime.png')
    plot_pvalue_calibration(results_df, FIGURES_DIR / 'loco_vs_cmi_pvalue_calibration.png')

    # Summary table
    summary_df = create_summary_table(type1_df, power_df, runtime_df)
    summary_df.to_csv(OUTPUT_DIR / 'loco_vs_cmi_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Print Type I error details
    print("\n\nType I Error by Condition:")
    print(type1_df.pivot(index=['n_samples', 'n_covariates'], columns='method', values='type1_error').to_string())

    # Print runtime comparison
    print("\n\nMean Runtime by Condition:")
    print(runtime_df.pivot(index=['n_samples', 'n_covariates'], columns='method', values='mean_runtime').round(3).to_string())

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
