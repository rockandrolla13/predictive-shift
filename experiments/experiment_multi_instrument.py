"""
Experiment 1: Multi-Instrument Aggregation Analysis

This experiment evaluates the effectiveness of multi-instrument bounds aggregation
compared to single-best instrument selection.

Key Questions:
1. Does using multiple instruments tighten bounds compared to single-best?
2. How does instrument agreement affect bound quality?
3. What is the optimal number of instruments (k) to use?

This experiment leverages the newly implemented:
- select_top_k_instruments() from covariate_scoring.py
- aggregate_across_instruments() from transfer.py
- compute_instrument_agreement() from transfer.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CITestEngine,
    rank_covariates,
    select_top_k_instruments,
    select_best_instrument,
    aggregate_across_instruments,
    aggregate_with_weights,
    compute_instrument_agreement,
    BinarySyntheticDGP,
    generate_random_dgp,
    simulate_observational,
    simulate_rct,
    compute_true_cate,
    solve_all_bounds_binary_lp,
    bounds_to_dataframe,
    compute_coverage_rate,
    compute_informativeness,
)

warnings.filterwarnings('ignore')

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

CONFIG = {
    'n_samples': 1000,
    'n_covariates': 5,
    'n_dgps': 10,  # Number of DGPs to test
    'random_seed': 42,
    'epsilon': 0.1,
    'k_values': [1, 2, 3, 4, 5],  # Number of instruments to try
    'n_permutations': 100,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, output_path, close=True):
    """Save figure with consistent settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format='png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    if close:
        plt.close(fig)


def compute_bounds_for_instrument(
    data: pd.DataFrame,
    instrument: str,
    other_covariates: List[str],
    treatment: str,
    outcome: str,
    epsilon: float
) -> Dict[Tuple, Tuple]:
    """
    Compute CATE bounds using a specific instrument.

    Returns dict mapping stratum -> (lower, upper) bounds.
    """
    # Create a simple training data structure with required F column
    data_copy = data.copy()
    if 'F' not in data_copy.columns:
        data_copy['F'] = 'on'  # Add regime indicator

    training_data = {'site_1': data_copy}

    # All covariates for stratification
    covariates = [instrument] + other_covariates

    try:
        bounds = solve_all_bounds_binary_lp(
            training_data,
            covariates=covariates,
            treatment=treatment,
            outcome=outcome,
            epsilon=epsilon,
            regime_col='F'
        )
        return bounds
    except Exception as e:
        print(f"  Warning: Failed to compute bounds for {instrument}: {e}")
        return {}


def run_single_dgp_experiment(
    dgp_id: int,
    n_samples: int,
    n_covariates: int,
    epsilon: float,
    k_values: List[int],
    n_permutations: int,
    seed: int
) -> Dict[str, Any]:
    """
    Run multi-instrument experiment for a single DGP.

    Returns results dict with bounds comparisons.
    """
    np.random.seed(seed + dgp_id)

    # Generate random DGP
    dgp = generate_random_dgp(n_covariates=n_covariates, seed=seed + dgp_id)

    # Simulate observational data
    obs_data = simulate_observational(dgp, n_samples)

    # Simulate RCT for ground truth
    rct_data = simulate_rct(dgp, n_samples // 2)

    # Compute true CATEs
    true_cates = {}
    for x_tuple in obs_data.groupby([f'X{i}' for i in range(n_covariates)]).groups.keys():
        if not isinstance(x_tuple, tuple):
            x_tuple = (x_tuple,)
        x_array = np.array([x_tuple])
        true_cate = compute_true_cate(dgp, x_array)
        true_cates[x_tuple] = true_cate[0]

    # Identify covariates
    covariate_cols = [f'X{i}' for i in range(n_covariates)]

    # Score covariates using CI engine
    ci_engine = CITestEngine(n_permutations=n_permutations, random_seed=seed)

    try:
        scores_df = rank_covariates(
            obs_data, covariate_cols,
            treatment='A', outcome='Y',
            ci_engine=ci_engine
        )
    except Exception as e:
        return {'error': f'Covariate scoring failed: {e}', 'dgp_id': dgp_id}

    # Results storage
    results = {
        'dgp_id': dgp_id,
        'n_samples': n_samples,
        'n_covariates': n_covariates,
        'epsilon': epsilon,
        'n_true_cate_strata': len(true_cates),
        'covariate_scores': scores_df.to_dict('records'),
        'bounds_by_k': {},
        'coverage_by_k': {},
        'width_by_k': {},
        'instrument_agreement': None,
    }

    # Compute bounds for different k values
    all_instrument_bounds = {}

    for k in k_values:
        # Select top-k instruments
        top_k = select_top_k_instruments(scores_df, k=k)

        if len(top_k) == 0:
            continue

        # Compute bounds for each selected instrument
        k_instrument_bounds = {}
        for inst in top_k:
            other_covs = [c for c in covariate_cols if c != inst]
            bounds = compute_bounds_for_instrument(
                obs_data, inst, other_covs,
                'A', 'Y', epsilon
            )
            if bounds:
                k_instrument_bounds[inst] = bounds
                all_instrument_bounds[inst] = bounds

        if len(k_instrument_bounds) == 0:
            continue

        # Aggregate across instruments
        if len(k_instrument_bounds) == 1:
            aggregated = list(k_instrument_bounds.values())[0]
        else:
            aggregated = aggregate_across_instruments(
                k_instrument_bounds, method='intersection'
            )

        # Compute metrics
        if len(aggregated) > 0:
            # Coverage
            coverage = 0
            total = 0
            for z, (lower, upper) in aggregated.items():
                if z in true_cates:
                    total += 1
                    if lower <= true_cates[z] <= upper:
                        coverage += 1

            coverage_rate = coverage / total if total > 0 else np.nan

            # Width
            widths = [upper - lower for lower, upper in aggregated.values()]
            mean_width = np.mean(widths) if widths else np.nan

            results['bounds_by_k'][k] = {
                'n_instruments': len(k_instrument_bounds),
                'instruments_used': list(k_instrument_bounds.keys()),
                'n_strata': len(aggregated),
                'bounds': {str(k): v for k, v in aggregated.items()},
            }
            results['coverage_by_k'][k] = coverage_rate
            results['width_by_k'][k] = mean_width

    # Compute instrument agreement (for visualization)
    if len(all_instrument_bounds) >= 2:
        try:
            agreement_df = compute_instrument_agreement(all_instrument_bounds)
            results['instrument_agreement'] = agreement_df.to_dict('records')
        except Exception as e:
            print(f"  Warning: Could not compute agreement: {e}")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_multi_instrument_experiment(
    output_dir: str = 'results/multi_instrument_experiment',
    n_dgps: int = 10,
    n_samples: int = 1000,
    n_covariates: int = 5,
    epsilon: float = 0.1,
    k_values: List[int] = [1, 2, 3, 4, 5],
    n_permutations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the full multi-instrument aggregation experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1: MULTI-INSTRUMENT AGGREGATION ANALYSIS")
    print("=" * 70)
    print(f"  DGPs: {n_dgps}")
    print(f"  Samples per DGP: {n_samples}")
    print(f"  Covariates: {n_covariates}")
    print(f"  Epsilon: {epsilon}")
    print(f"  K values to test: {k_values}")
    print()

    all_results = []

    for dgp_id in range(n_dgps):
        print(f"[{dgp_id + 1}/{n_dgps}] Running DGP {dgp_id}...")

        result = run_single_dgp_experiment(
            dgp_id=dgp_id,
            n_samples=n_samples,
            n_covariates=n_covariates,
            epsilon=epsilon,
            k_values=k_values,
            n_permutations=n_permutations,
            seed=seed
        )

        if 'error' not in result:
            all_results.append(result)

            # Print summary
            coverages = result.get('coverage_by_k', {})
            widths = result.get('width_by_k', {})
            if coverages:
                cov_str = ", ".join([f"k={k}:{v:.2f}" for k, v in coverages.items()])
                print(f"    Coverage: {cov_str}")
        else:
            print(f"    Error: {result['error']}")

    # Aggregate results
    summary = {
        'n_successful_dgps': len(all_results),
        'config': {
            'n_dgps': n_dgps,
            'n_samples': n_samples,
            'n_covariates': n_covariates,
            'epsilon': epsilon,
            'k_values': k_values,
            'n_permutations': n_permutations,
            'seed': seed
        },
        'individual_results': all_results,
        'aggregate_metrics': {}
    }

    # Compute aggregate metrics
    for k in k_values:
        coverages = [r['coverage_by_k'].get(k) for r in all_results if k in r.get('coverage_by_k', {})]
        widths = [r['width_by_k'].get(k) for r in all_results if k in r.get('width_by_k', {})]

        coverages = [c for c in coverages if c is not None and not np.isnan(c)]
        widths = [w for w in widths if w is not None and not np.isnan(w)]

        if coverages:
            summary['aggregate_metrics'][k] = {
                'mean_coverage': np.mean(coverages),
                'std_coverage': np.std(coverages),
                'mean_width': np.mean(widths) if widths else np.nan,
                'std_width': np.std(widths) if widths else np.nan,
                'n_dgps': len(coverages)
            }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_path / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    return summary


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_multi_instrument_visualizations(
    results: Dict[str, Any],
    output_dir: str = 'results/multi_instrument_experiment'
) -> List[str]:
    """
    Generate visualizations for multi-instrument experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    print("\nGenerating visualizations...")

    aggregate_metrics = results.get('aggregate_metrics', {})

    if not aggregate_metrics:
        print("  No aggregate metrics to visualize")
        return generated_files

    # 1. Coverage vs Number of Instruments
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(aggregate_metrics.keys())
    mean_coverages = [aggregate_metrics[k]['mean_coverage'] for k in k_values]
    std_coverages = [aggregate_metrics[k]['std_coverage'] for k in k_values]

    ax.errorbar(k_values, mean_coverages, yerr=std_coverages,
                fmt='o-', markersize=10, capsize=5, capthick=2,
                color='steelblue', linewidth=2)

    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% Target')

    ax.set_xlabel('Number of Instruments (k)', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('CATE Coverage vs Number of Instruments', fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    filename = 'coverage_vs_k.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 2. Bound Width vs Number of Instruments
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_widths = [aggregate_metrics[k]['mean_width'] for k in k_values]
    std_widths = [aggregate_metrics[k]['std_width'] for k in k_values]

    # Filter out NaN values
    valid_k = [k for k, w in zip(k_values, mean_widths) if not np.isnan(w)]
    valid_widths = [w for w in mean_widths if not np.isnan(w)]
    valid_stds = [s for k, s in zip(k_values, std_widths) if k in valid_k]

    if valid_widths:
        ax.errorbar(valid_k, valid_widths, yerr=valid_stds,
                    fmt='s-', markersize=10, capsize=5, capthick=2,
                    color='darkorange', linewidth=2)

        ax.set_xlabel('Number of Instruments (k)', fontsize=12)
        ax.set_ylabel('Mean Bound Width', fontsize=12)
        ax.set_title('Bound Width vs Number of Instruments', fontsize=14)
        ax.set_xticks(valid_k)
        ax.grid(True, alpha=0.3)

        # Add annotation showing tightening
        if len(valid_widths) >= 2:
            reduction = (valid_widths[0] - valid_widths[-1]) / valid_widths[0] * 100
            ax.annotate(f'{reduction:.1f}% narrower\nwith k={valid_k[-1]} vs k={valid_k[0]}',
                        xy=(valid_k[-1], valid_widths[-1]),
                        xytext=(valid_k[-1] - 0.5, valid_widths[-1] + 0.05),
                        fontsize=10, color='darkgreen',
                        arrowprops=dict(arrowstyle='->', color='darkgreen'))

    filename = 'width_vs_k.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 3. Coverage-Width Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for i, k in enumerate(k_values):
        metrics = aggregate_metrics.get(k, {})
        if 'mean_coverage' in metrics and 'mean_width' in metrics:
            cov = metrics['mean_coverage']
            width = metrics['mean_width']
            if not np.isnan(cov) and not np.isnan(width):
                ax.scatter(width, cov, s=200, c=[colors[i]], edgecolors='black',
                          linewidth=2, label=f'k={k}', zorder=5)

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Coverage')
    ax.set_xlabel('Mean Bound Width', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('Coverage-Width Tradeoff by Number of Instruments', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    filename = 'coverage_width_tradeoff.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 4. Per-DGP Improvement Analysis
    individual_results = results.get('individual_results', [])

    if len(individual_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Coverage improvement (k=1 to k=max)
        max_k = max(k_values)
        improvements_coverage = []
        improvements_width = []

        for r in individual_results:
            cov_k1 = r['coverage_by_k'].get(1)
            cov_kmax = r['coverage_by_k'].get(max_k)
            width_k1 = r['width_by_k'].get(1)
            width_kmax = r['width_by_k'].get(max_k)

            if cov_k1 is not None and cov_kmax is not None:
                if not np.isnan(cov_k1) and not np.isnan(cov_kmax):
                    improvements_coverage.append(cov_kmax - cov_k1)

            if width_k1 is not None and width_kmax is not None:
                if not np.isnan(width_k1) and not np.isnan(width_kmax):
                    improvements_width.append(width_k1 - width_kmax)

        # Coverage improvement histogram
        if improvements_coverage:
            ax1 = axes[0]
            ax1.hist(improvements_coverage, bins=15, edgecolor='black', alpha=0.7,
                    color='steelblue')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax1.axvline(x=np.mean(improvements_coverage), color='green', linestyle='-',
                       linewidth=2, label=f'Mean: {np.mean(improvements_coverage):.3f}')
            ax1.set_xlabel(f'Coverage Improvement (k={max_k} - k=1)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Coverage Improvement Distribution', fontsize=14)
            ax1.legend()

        # Width reduction histogram
        if improvements_width:
            ax2 = axes[1]
            ax2.hist(improvements_width, bins=15, edgecolor='black', alpha=0.7,
                    color='darkorange')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.axvline(x=np.mean(improvements_width), color='green', linestyle='-',
                       linewidth=2, label=f'Mean: {np.mean(improvements_width):.3f}')
            ax2.set_xlabel(f'Width Reduction (k=1 - k={max_k})', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Width Reduction Distribution', fontsize=14)
            ax2.legend()

        plt.tight_layout()
        filename = 'improvement_distributions.png'
        save_figure(fig, output_path / filename)
        generated_files.append(filename)

    # 5. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table_data = []
    headers = ['k', 'Coverage', 'Std', 'Width', 'Std', 'N']

    for k in k_values:
        metrics = aggregate_metrics.get(k, {})
        row = [
            str(k),
            f"{metrics.get('mean_coverage', np.nan):.3f}",
            f"{metrics.get('std_coverage', np.nan):.3f}",
            f"{metrics.get('mean_width', np.nan):.3f}",
            f"{metrics.get('std_width', np.nan):.3f}",
            str(metrics.get('n_dgps', 0))
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Color header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best coverage row
    best_k_idx = np.argmax([aggregate_metrics.get(k, {}).get('mean_coverage', 0) for k in k_values])
    for j in range(len(headers)):
        table[(best_k_idx + 1, j)].set_facecolor('#E2EFDA')

    ax.set_title('Summary: Multi-Instrument Aggregation Results', fontsize=14, pad=20)

    filename = 'summary_table.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    print(f"\nGenerated {len(generated_files)} visualizations:")
    for f in generated_files:
        print(f"  - {f}")

    return generated_files


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the multi-instrument experiment."""
    output_dir = 'results/multi_instrument_experiment'

    # Run experiment
    results = run_multi_instrument_experiment(
        output_dir=output_dir,
        n_dgps=CONFIG['n_dgps'],
        n_samples=CONFIG['n_samples'],
        n_covariates=CONFIG['n_covariates'],
        epsilon=CONFIG['epsilon'],
        k_values=CONFIG['k_values'],
        n_permutations=CONFIG['n_permutations'],
        seed=CONFIG['random_seed']
    )

    # Generate visualizations
    generate_multi_instrument_visualizations(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    agg = results.get('aggregate_metrics', {})
    for k in sorted(agg.keys()):
        metrics = agg[k]
        print(f"  k={k}: Coverage={metrics.get('mean_coverage', np.nan):.3f} ± {metrics.get('std_coverage', np.nan):.3f}, "
              f"Width={metrics.get('mean_width', np.nan):.3f} ± {metrics.get('std_width', np.nan):.3f}")

    # Key finding
    if len(agg) >= 2:
        k_vals = sorted(agg.keys())
        cov_k1 = agg.get(k_vals[0], {}).get('mean_coverage', 0)
        cov_kmax = agg.get(k_vals[-1], {}).get('mean_coverage', 0)
        width_k1 = agg.get(k_vals[0], {}).get('mean_width', 1)
        width_kmax = agg.get(k_vals[-1], {}).get('mean_width', 1)

        print(f"\n  KEY FINDING:")
        print(f"    Using k={k_vals[-1]} instruments vs k={k_vals[0]}:")
        print(f"    - Coverage: {cov_k1:.3f} -> {cov_kmax:.3f} ({(cov_kmax-cov_k1)*100:+.1f}%)")
        if width_k1 > 0:
            print(f"    - Width: {width_k1:.3f} -> {width_kmax:.3f} ({(1-width_kmax/width_k1)*100:.1f}% tighter)")

    print(f"\n  Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
