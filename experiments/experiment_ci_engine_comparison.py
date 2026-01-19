"""
Experiment 2: CI Engine Comparison (CMI vs L1-Regression)

This experiment compares two approaches for conditional independence testing:
1. CMI (Conditional Mutual Information) - Our original approach
2. L1-Regression - Ricardo's approach using logistic regression with L1 penalty

Key Questions:
1. Do CMI and L1-Regression identify the same instruments?
2. Which method produces better CATE bounds?
3. What are the runtime differences?
4. Under what conditions does each method excel?

This experiment leverages the newly implemented:
- CITestEngine (CMI-based) from ci_tests.py
- L1RegressionCIEngine from ci_tests_l1.py
- rank_covariates() with both engines
"""

import sys
import json
import time
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
    L1RegressionCIEngine,
    create_ci_engine,
    rank_covariates,
    select_best_instrument,
    BinarySyntheticDGP,
    generate_random_dgp,
    simulate_observational,
    simulate_rct,
    compute_true_cate,
    solve_all_bounds_binary_lp,
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
    'n_samples_list': [200, 500, 1000, 2000],  # Different sample sizes
    'n_covariates': 5,
    'n_dgps': 10,  # Number of DGPs per sample size
    'random_seed': 42,
    'epsilon': 0.1,
    'n_permutations_cmi': 100,  # For CMI engine
    'alpha': 0.05,  # Significance level
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


def compute_ranking_agreement(scores_cmi: pd.DataFrame, scores_l1: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute agreement metrics between CMI and L1 rankings.

    Returns dict with:
    - top1_agreement: Whether both methods select the same best instrument
    - top3_agreement: Jaccard similarity of top 3
    - rank_correlation: Spearman correlation of rankings
    - ehs_agreement: Proportion of covariates where both agree on EHS pass/fail
    """
    if len(scores_cmi) == 0 or len(scores_l1) == 0:
        return {
            'top1_agreement': np.nan,
            'top3_agreement': np.nan,
            'rank_correlation': np.nan,
            'ehs_agreement': np.nan
        }

    # Get covariate names
    cmi_names = scores_cmi['z_a'].tolist()
    l1_names = scores_l1['z_a'].tolist()

    # Top-1 agreement
    top1_cmi = scores_cmi.iloc[0]['z_a'] if len(scores_cmi) > 0 else None
    top1_l1 = scores_l1.iloc[0]['z_a'] if len(scores_l1) > 0 else None
    top1_agreement = top1_cmi == top1_l1

    # Top-3 agreement (Jaccard)
    top3_cmi = set(scores_cmi.head(3)['z_a'].tolist())
    top3_l1 = set(scores_l1.head(3)['z_a'].tolist())
    if len(top3_cmi) > 0 or len(top3_l1) > 0:
        jaccard = len(top3_cmi & top3_l1) / len(top3_cmi | top3_l1)
    else:
        jaccard = np.nan

    # Rank correlation
    try:
        from scipy.stats import spearmanr

        # Align rankings
        common_covs = set(cmi_names) & set(l1_names)
        if len(common_covs) >= 3:
            cmi_ranks = {row['z_a']: i for i, row in scores_cmi.iterrows()}
            l1_ranks = {row['z_a']: i for i, row in scores_l1.iterrows()}

            ranks1 = [cmi_ranks[c] for c in common_covs]
            ranks2 = [l1_ranks[c] for c in common_covs]

            corr, _ = spearmanr(ranks1, ranks2)
        else:
            corr = np.nan
    except:
        corr = np.nan

    # EHS agreement
    ehs_agreement = np.nan
    if 'passes_ehs' in scores_cmi.columns and 'passes_ehs' in scores_l1.columns:
        cmi_ehs = dict(zip(scores_cmi['z_a'], scores_cmi['passes_ehs']))
        l1_ehs = dict(zip(scores_l1['z_a'], scores_l1['passes_ehs']))

        common = set(cmi_ehs.keys()) & set(l1_ehs.keys())
        if len(common) > 0:
            agreements = sum(1 for c in common if cmi_ehs[c] == l1_ehs[c])
            ehs_agreement = agreements / len(common)

    return {
        'top1_agreement': top1_agreement,
        'top3_agreement': jaccard,
        'rank_correlation': corr,
        'ehs_agreement': ehs_agreement
    }


def run_single_comparison(
    dgp_id: int,
    n_samples: int,
    n_covariates: int,
    epsilon: float,
    n_permutations: int,
    seed: int
) -> Dict[str, Any]:
    """
    Run CI engine comparison for a single DGP.

    Returns results dict with timing and agreement metrics.
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

    results = {
        'dgp_id': dgp_id,
        'n_samples': n_samples,
        'n_covariates': n_covariates,
    }

    # ==========================================
    # CMI-based CI Testing
    # ==========================================
    try:
        start_time = time.time()

        cmi_engine = CITestEngine(n_permutations=n_permutations, random_seed=seed)
        scores_cmi = rank_covariates(
            obs_data, covariate_cols,
            treatment='A', outcome='Y',
            ci_engine=cmi_engine,
            use_permutation_test=True
        )

        cmi_time = time.time() - start_time

        results['cmi_runtime'] = cmi_time
        results['cmi_scores'] = scores_cmi.to_dict('records')
        results['cmi_best_instrument'] = scores_cmi.iloc[0]['z_a'] if len(scores_cmi) > 0 else None
        results['cmi_n_passing_ehs'] = scores_cmi['passes_ehs'].sum() if 'passes_ehs' in scores_cmi.columns else 0

    except Exception as e:
        results['cmi_error'] = str(e)
        scores_cmi = pd.DataFrame()

    # ==========================================
    # L1-Regression CI Testing
    # ==========================================
    try:
        start_time = time.time()

        l1_engine = L1RegressionCIEngine(alpha=CONFIG['alpha'])
        scores_l1 = rank_covariates(
            obs_data, covariate_cols,
            treatment='A', outcome='Y',
            ci_engine=l1_engine,
            use_permutation_test=False  # L1 doesn't use permutation tests
        )

        l1_time = time.time() - start_time

        results['l1_runtime'] = l1_time
        results['l1_scores'] = scores_l1.to_dict('records')
        results['l1_best_instrument'] = scores_l1.iloc[0]['z_a'] if len(scores_l1) > 0 else None
        results['l1_n_passing_ehs'] = scores_l1['passes_ehs'].sum() if 'passes_ehs' in scores_l1.columns else 0

    except Exception as e:
        results['l1_error'] = str(e)
        scores_l1 = pd.DataFrame()

    # ==========================================
    # Agreement Metrics
    # ==========================================
    if not scores_cmi.empty and not scores_l1.empty:
        agreement = compute_ranking_agreement(scores_cmi, scores_l1)
        results['agreement'] = agreement

    # ==========================================
    # Bounds Quality Comparison (using best instrument from each)
    # ==========================================
    def compute_bounds_and_coverage(best_inst, method_name):
        """Compute bounds using the best instrument and evaluate coverage."""
        if best_inst is None:
            return {}

        try:
            # Add regime column
            obs_data_copy = obs_data.copy()
            if 'F' not in obs_data_copy.columns:
                obs_data_copy['F'] = 'on'

            training_data = {'site_1': obs_data_copy}
            other_covs = [c for c in covariate_cols if c != best_inst]

            bounds = solve_all_bounds_binary_lp(
                training_data,
                covariates=[best_inst] + other_covs,
                treatment='A',
                outcome='Y',
                epsilon=epsilon,
                regime_col='F'
            )

            if len(bounds) == 0:
                return {}

            # Compute coverage
            covered = 0
            total = 0
            widths = []

            for z, (lower, upper) in bounds.items():
                widths.append(upper - lower)
                if z in true_cates:
                    total += 1
                    if lower <= true_cates[z] <= upper:
                        covered += 1

            return {
                f'{method_name}_coverage': covered / total if total > 0 else np.nan,
                f'{method_name}_mean_width': np.mean(widths) if widths else np.nan,
                f'{method_name}_n_strata': len(bounds)
            }
        except Exception as e:
            return {f'{method_name}_error': str(e)}

    cmi_bounds = compute_bounds_and_coverage(results.get('cmi_best_instrument'), 'cmi')
    l1_bounds = compute_bounds_and_coverage(results.get('l1_best_instrument'), 'l1')

    results.update(cmi_bounds)
    results.update(l1_bounds)

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_ci_engine_comparison_experiment(
    output_dir: str = 'results/ci_engine_comparison',
    n_samples_list: List[int] = [200, 500, 1000, 2000],
    n_dgps: int = 10,
    n_covariates: int = 5,
    epsilon: float = 0.1,
    n_permutations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the full CI engine comparison experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2: CI ENGINE COMPARISON (CMI vs L1-REGRESSION)")
    print("=" * 70)
    print(f"  Sample sizes: {n_samples_list}")
    print(f"  DGPs per sample size: {n_dgps}")
    print(f"  Covariates: {n_covariates}")
    print(f"  Epsilon: {epsilon}")
    print(f"  CMI permutations: {n_permutations}")
    print()

    all_results = []
    total = len(n_samples_list) * n_dgps
    current = 0

    for n_samples in n_samples_list:
        print(f"\n--- Sample size: {n_samples} ---")

        for dgp_id in range(n_dgps):
            current += 1
            print(f"[{current}/{total}] n={n_samples}, DGP {dgp_id}...")

            result = run_single_comparison(
                dgp_id=dgp_id,
                n_samples=n_samples,
                n_covariates=n_covariates,
                epsilon=epsilon,
                n_permutations=n_permutations,
                seed=seed
            )

            all_results.append(result)

            # Print brief summary
            if 'cmi_runtime' in result and 'l1_runtime' in result:
                speedup = result['cmi_runtime'] / max(result['l1_runtime'], 0.001)
                agreement = result.get('agreement', {}).get('top1_agreement', 'N/A')
                print(f"    CMI: {result['cmi_runtime']:.2f}s, L1: {result['l1_runtime']:.2f}s "
                      f"(L1 {speedup:.1f}x faster), Top-1 agree: {agreement}")

    # Aggregate results
    summary = {
        'config': {
            'n_samples_list': n_samples_list,
            'n_dgps': n_dgps,
            'n_covariates': n_covariates,
            'epsilon': epsilon,
            'n_permutations': n_permutations,
            'seed': seed
        },
        'individual_results': all_results,
        'aggregate_by_n': {}
    }

    # Compute aggregate metrics by sample size
    for n_samples in n_samples_list:
        n_results = [r for r in all_results if r['n_samples'] == n_samples]

        if len(n_results) == 0:
            continue

        # Runtime
        cmi_times = [r.get('cmi_runtime') for r in n_results if 'cmi_runtime' in r]
        l1_times = [r.get('l1_runtime') for r in n_results if 'l1_runtime' in r]

        # Agreement
        top1_agreements = [r.get('agreement', {}).get('top1_agreement') for r in n_results]
        top1_agreements = [a for a in top1_agreements if a is not None]

        top3_agreements = [r.get('agreement', {}).get('top3_agreement') for r in n_results]
        top3_agreements = [a for a in top3_agreements if a is not None and not np.isnan(a)]

        rank_corrs = [r.get('agreement', {}).get('rank_correlation') for r in n_results]
        rank_corrs = [c for c in rank_corrs if c is not None and not np.isnan(c)]

        # Coverage
        cmi_coverages = [r.get('cmi_coverage') for r in n_results if 'cmi_coverage' in r]
        cmi_coverages = [c for c in cmi_coverages if c is not None and not np.isnan(c)]

        l1_coverages = [r.get('l1_coverage') for r in n_results if 'l1_coverage' in r]
        l1_coverages = [c for c in l1_coverages if c is not None and not np.isnan(c)]

        # Width
        cmi_widths = [r.get('cmi_mean_width') for r in n_results if 'cmi_mean_width' in r]
        cmi_widths = [w for w in cmi_widths if w is not None and not np.isnan(w)]

        l1_widths = [r.get('l1_mean_width') for r in n_results if 'l1_mean_width' in r]
        l1_widths = [w for w in l1_widths if w is not None and not np.isnan(w)]

        summary['aggregate_by_n'][n_samples] = {
            'n_runs': len(n_results),
            'cmi_mean_runtime': np.mean(cmi_times) if cmi_times else np.nan,
            'cmi_std_runtime': np.std(cmi_times) if cmi_times else np.nan,
            'l1_mean_runtime': np.mean(l1_times) if l1_times else np.nan,
            'l1_std_runtime': np.std(l1_times) if l1_times else np.nan,
            'speedup_factor': np.mean(cmi_times) / np.mean(l1_times) if cmi_times and l1_times and np.mean(l1_times) > 0 else np.nan,
            'top1_agreement_rate': np.mean(top1_agreements) if top1_agreements else np.nan,
            'top3_jaccard': np.mean(top3_agreements) if top3_agreements else np.nan,
            'rank_correlation': np.mean(rank_corrs) if rank_corrs else np.nan,
            'cmi_mean_coverage': np.mean(cmi_coverages) if cmi_coverages else np.nan,
            'l1_mean_coverage': np.mean(l1_coverages) if l1_coverages else np.nan,
            'cmi_mean_width': np.mean(cmi_widths) if cmi_widths else np.nan,
            'l1_mean_width': np.mean(l1_widths) if l1_widths else np.nan,
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

def generate_ci_comparison_visualizations(
    results: Dict[str, Any],
    output_dir: str = 'results/ci_engine_comparison'
) -> List[str]:
    """
    Generate visualizations for CI engine comparison experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    print("\nGenerating visualizations...")

    aggregate = results.get('aggregate_by_n', {})

    if not aggregate:
        print("  No aggregate metrics to visualize")
        return generated_files

    n_values = sorted(aggregate.keys())

    # 1. Runtime Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    cmi_times = [aggregate[n]['cmi_mean_runtime'] for n in n_values]
    cmi_stds = [aggregate[n]['cmi_std_runtime'] for n in n_values]
    l1_times = [aggregate[n]['l1_mean_runtime'] for n in n_values]
    l1_stds = [aggregate[n]['l1_std_runtime'] for n in n_values]

    x = np.arange(len(n_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, cmi_times, width, yerr=cmi_stds, label='CMI',
                   color='steelblue', capsize=5)
    bars2 = ax.bar(x + width/2, l1_times, width, yerr=l1_stds, label='L1-Regression',
                   color='darkorange', capsize=5)

    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('CI Engine Runtime Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, n in enumerate(n_values):
        speedup = aggregate[n].get('speedup_factor', np.nan)
        if not np.isnan(speedup):
            ax.annotate(f'{speedup:.1f}x', xy=(i, max(cmi_times[i], l1_times[i])),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, color='darkgreen')

    filename = 'runtime_comparison.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 2. Agreement Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Top-1 agreement
    ax1 = axes[0]
    top1_rates = [aggregate[n].get('top1_agreement_rate', np.nan) for n in n_values]
    ax1.bar(n_values, top1_rates, color='mediumseagreen', edgecolor='black')
    ax1.set_xlabel('Sample Size (n)', fontsize=11)
    ax1.set_ylabel('Top-1 Agreement Rate', fontsize=11)
    ax1.set_title('Best Instrument Agreement', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.legend()

    # Top-3 Jaccard
    ax2 = axes[1]
    jaccard_scores = [aggregate[n].get('top3_jaccard', np.nan) for n in n_values]
    ax2.bar(n_values, jaccard_scores, color='mediumpurple', edgecolor='black')
    ax2.set_xlabel('Sample Size (n)', fontsize=11)
    ax2.set_ylabel('Top-3 Jaccard Similarity', fontsize=11)
    ax2.set_title('Top-3 Instrument Overlap', fontsize=12)
    ax2.set_ylim(0, 1.05)

    # Rank correlation
    ax3 = axes[2]
    rank_corrs = [aggregate[n].get('rank_correlation', np.nan) for n in n_values]
    colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in rank_corrs]
    ax3.bar(n_values, rank_corrs, color=colors, edgecolor='black')
    ax3.set_xlabel('Sample Size (n)', fontsize=11)
    ax3.set_ylabel('Spearman Correlation', fontsize=11)
    ax3.set_title('Ranking Correlation', fontsize=12)
    ax3.set_ylim(-0.2, 1.05)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    plt.tight_layout()
    filename = 'agreement_metrics.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 3. Coverage Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    cmi_coverages = [aggregate[n].get('cmi_mean_coverage', np.nan) for n in n_values]
    l1_coverages = [aggregate[n].get('l1_mean_coverage', np.nan) for n in n_values]

    x = np.arange(len(n_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, cmi_coverages, width, label='CMI', color='steelblue')
    bars2 = ax.bar(x + width/2, l1_coverages, width, label='L1-Regression', color='darkorange')

    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% Target')

    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('CATE Coverage by CI Engine', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    filename = 'coverage_comparison.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 4. Width Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    cmi_widths = [aggregate[n].get('cmi_mean_width', np.nan) for n in n_values]
    l1_widths = [aggregate[n].get('l1_mean_width', np.nan) for n in n_values]

    ax.plot(n_values, cmi_widths, 'o-', markersize=10, linewidth=2, label='CMI',
            color='steelblue')
    ax.plot(n_values, l1_widths, 's-', markersize=10, linewidth=2, label='L1-Regression',
            color='darkorange')

    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Mean Bound Width', fontsize=12)
    ax.set_title('Bound Width by CI Engine', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    filename = 'width_comparison.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 5. Comprehensive Summary Plot
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Runtime (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(n_values, cmi_times, 'o-', markersize=8, linewidth=2, label='CMI', color='steelblue')
    ax1.semilogy(n_values, l1_times, 's-', markersize=8, linewidth=2, label='L1', color='darkorange')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Runtime (seconds, log scale)')
    ax1.set_title('A. Runtime Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Agreement
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(n_values, top1_rates, 'o-', markersize=8, linewidth=2, label='Top-1', color='mediumseagreen')
    ax2.plot(n_values, jaccard_scores, 's-', markersize=8, linewidth=2, label='Top-3 Jaccard', color='mediumpurple')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Agreement Rate')
    ax2.set_title('B. Method Agreement')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Coverage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(n_values, cmi_coverages, 'o-', markersize=8, linewidth=2, label='CMI', color='steelblue')
    ax3.plot(n_values, l1_coverages, 's-', markersize=8, linewidth=2, label='L1', color='darkorange')
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Coverage Rate')
    ax3.set_title('C. CATE Coverage')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Width
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_values, cmi_widths, 'o-', markersize=8, linewidth=2, label='CMI', color='steelblue')
    ax4.plot(n_values, l1_widths, 's-', markersize=8, linewidth=2, label='L1', color='darkorange')
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Mean Bound Width')
    ax4.set_title('D. Bound Informativeness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('CI Engine Comparison: CMI vs L1-Regression', fontsize=16, y=1.02)

    filename = 'comprehensive_summary.png'
    save_figure(fig, output_path / filename)
    generated_files.append(filename)

    # 6. Summary Table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    table_data = []
    headers = ['n', 'CMI Time', 'L1 Time', 'Speedup', 'Top-1 Agree', 'CMI Cov', 'L1 Cov', 'CMI Width', 'L1 Width']

    for n in n_values:
        m = aggregate.get(n, {})
        row = [
            str(n),
            f"{m.get('cmi_mean_runtime', np.nan):.2f}s",
            f"{m.get('l1_mean_runtime', np.nan):.2f}s",
            f"{m.get('speedup_factor', np.nan):.1f}x",
            f"{m.get('top1_agreement_rate', np.nan):.2%}",
            f"{m.get('cmi_mean_coverage', np.nan):.3f}",
            f"{m.get('l1_mean_coverage', np.nan):.3f}",
            f"{m.get('cmi_mean_width', np.nan):.3f}",
            f"{m.get('l1_mean_width', np.nan):.3f}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    # Color header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary: CMI vs L1-Regression CI Engine Comparison', fontsize=14, pad=20)

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
    """Run the CI engine comparison experiment."""
    output_dir = 'results/ci_engine_comparison'

    # Run experiment
    results = run_ci_engine_comparison_experiment(
        output_dir=output_dir,
        n_samples_list=CONFIG['n_samples_list'],
        n_dgps=CONFIG['n_dgps'],
        n_covariates=CONFIG['n_covariates'],
        epsilon=CONFIG['epsilon'],
        n_permutations=CONFIG['n_permutations_cmi'],
        seed=CONFIG['random_seed']
    )

    # Generate visualizations
    generate_ci_comparison_visualizations(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    agg = results.get('aggregate_by_n', {})
    for n in sorted(agg.keys()):
        m = agg[n]
        print(f"\n  n={n}:")
        print(f"    Runtime: CMI={m.get('cmi_mean_runtime', np.nan):.2f}s, L1={m.get('l1_mean_runtime', np.nan):.2f}s "
              f"(L1 {m.get('speedup_factor', np.nan):.1f}x faster)")
        print(f"    Agreement: Top-1={m.get('top1_agreement_rate', np.nan):.1%}, "
              f"Top-3 Jaccard={m.get('top3_jaccard', np.nan):.2f}")
        print(f"    Coverage: CMI={m.get('cmi_mean_coverage', np.nan):.3f}, "
              f"L1={m.get('l1_mean_coverage', np.nan):.3f}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    if agg:
        all_speedups = [agg[n].get('speedup_factor', 0) for n in agg]
        avg_speedup = np.mean([s for s in all_speedups if s > 0])
        all_agreements = [agg[n].get('top1_agreement_rate', 0) for n in agg]
        avg_agreement = np.mean([a for a in all_agreements if not np.isnan(a)])

        print(f"    - L1-Regression is on average {avg_speedup:.1f}x faster than CMI")
        print(f"    - Methods agree on best instrument {avg_agreement:.1%} of the time")

    print(f"\n  Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
