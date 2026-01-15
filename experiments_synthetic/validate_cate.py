"""
CATE Coverage Validation for Synthetic Data

Since synthetic data has known ground truth CATEs, we can precisely validate
whether our bounds achieve the desired coverage properties.

Key questions addressed:
1. Do bounds contain true CATEs at the nominal rate?
2. How does coverage vary with confounding strength?
3. How does coverage vary with epsilon?
4. Is there systematic under/over-coverage for certain strata?

Usage:
    python experiments_synthetic/validate_cate.py --beta 0.3 --epsilon 0.1
    python experiments_synthetic/validate_cate.py --sweep-beta
    python experiments_synthetic/validate_cate.py --sweep-epsilon
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_synthetic.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
)

from experiments_synthetic.run_experiment import (
    run_synthetic_experiment,
    run_synthetic_grid,
    BETAS,
    EPSILONS,
)


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_single_experiment(
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    beta: float = 0.3,
    treatment_effect: float = 0.2,
    treatment_z_interaction: float = 0.1,
    epsilon: float = 0.1,
    n_permutations: int = 100,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run validation on a single configuration.
    
    Returns detailed coverage metrics and diagnostics.
    """
    # Run experiment
    result = run_synthetic_experiment(
        n_sites=n_sites,
        n_per_site=n_per_site,
        n_z_values=n_z_values,
        beta=beta,
        treatment_effect=treatment_effect,
        treatment_z_interaction=treatment_z_interaction,
        epsilon=epsilon,
        n_permutations=n_permutations,
        random_seed=random_seed,
        verbose=False
    )
    
    if 'error' in result:
        return {'error': result['error']}
    
    # Extract per-stratum results
    per_stratum = result['per_stratum_results']
    
    # Compute detailed metrics
    coverages = [r['is_covered'] for r in per_stratum]
    widths = [r['width'] for r in per_stratum]
    true_cates = [r['true_cate'] for r in per_stratum]
    bound_midpoints = [(r['bound_lower'] + r['bound_upper']) / 2 for r in per_stratum]
    
    # Coverage rate with confidence interval (Wilson score)
    n = len(coverages)
    k = sum(coverages)
    if n > 0:
        p_hat = k / n
        z = 1.96  # 95% CI
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
        coverage_ci = (center - margin, center + margin)
    else:
        p_hat = np.nan
        coverage_ci = (np.nan, np.nan)
    
    # Bias analysis
    biases = [bound_midpoints[i] - true_cates[i] for i in range(len(true_cates))]
    mean_bias = np.mean(biases) if biases else np.nan
    
    # Width analysis
    mean_width = np.mean(widths) if widths else np.nan
    width_to_effect_ratio = mean_width / abs(treatment_effect) if treatment_effect != 0 else np.nan
    
    return {
        'config': result['config'],
        'metrics': result['metrics'],
        'validation': {
            'coverage_rate': p_hat,
            'coverage_ci_lower': coverage_ci[0],
            'coverage_ci_upper': coverage_ci[1],
            'n_strata': n,
            'n_covered': k,
            'mean_bias': mean_bias,
            'mean_width': mean_width,
            'width_to_effect_ratio': width_to_effect_ratio,
        },
        'per_stratum': per_stratum,
        'true_cates': result['true_cates']
    }


def validate_coverage_sweep(
    betas: Optional[List[float]] = None,
    epsilons: Optional[List[float]] = None,
    n_replications: int = 10,
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    treatment_effect: float = 0.2,
    treatment_z_interaction: float = 0.1,
    n_permutations: int = 100,
    base_seed: int = 42,
    output_dir: str = 'validation_results',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run coverage validation across multiple configurations and replications.
    
    This provides a robust estimate of coverage by running multiple replications
    with different random seeds.
    """
    betas = betas or BETAS
    epsilons = epsilons or [0.1]
    
    results = []
    total = len(betas) * len(epsilons) * n_replications
    current = 0
    
    for beta in betas:
        for epsilon in epsilons:
            for rep in range(n_replications):
                current += 1
                seed = base_seed + rep
                
                if verbose:
                    print(f"[{current}/{total}] beta={beta}, epsilon={epsilon}, rep={rep}")
                
                try:
                    val_result = validate_single_experiment(
                        n_sites=n_sites,
                        n_per_site=n_per_site,
                        n_z_values=n_z_values,
                        beta=beta,
                        treatment_effect=treatment_effect,
                        treatment_z_interaction=treatment_z_interaction,
                        epsilon=epsilon,
                        n_permutations=n_permutations,
                        random_seed=seed
                    )
                    
                    if 'error' not in val_result:
                        row = {
                            'beta': beta,
                            'epsilon': epsilon,
                            'replication': rep,
                            'seed': seed,
                            **val_result['metrics'],
                            **val_result['validation']
                        }
                        results.append(row)
                    else:
                        if verbose:
                            print(f"  Error: {val_result['error']}")
                            
                except Exception as e:
                    if verbose:
                        print(f"  Exception: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'coverage_validation_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    return results_df


def compute_coverage_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated coverage statistics from validation results.
    """
    # Group by beta and epsilon
    grouped = results_df.groupby(['beta', 'epsilon']).agg({
        'coverage_rate': ['mean', 'std', 'count'],
        'ate_covered': ['mean', 'std'],
        'mean_width': ['mean', 'std'],
        'mean_bias': ['mean', 'std'],
    }).reset_index()
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Compute CI for coverage rate
    grouped['coverage_se'] = grouped['coverage_rate_std'] / np.sqrt(grouped['coverage_rate_count'])
    grouped['coverage_ci_lower'] = grouped['coverage_rate_mean'] - 1.96 * grouped['coverage_se']
    grouped['coverage_ci_upper'] = grouped['coverage_rate_mean'] + 1.96 * grouped['coverage_se']
    
    return grouped


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def save_figure(fig, output_path, close=True):
    """Save figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    if close:
        plt.close(fig)


def plot_coverage_by_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_by_beta.png'
) -> None:
    """Plot coverage rate with error bars by beta."""
    stats_df = compute_coverage_statistics(results_df)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CATE coverage
    ax = axes[0]
    ax.errorbar(
        stats_df['beta'],
        stats_df['coverage_rate_mean'],
        yerr=1.96 * stats_df['coverage_se'],
        fmt='o-', color='steelblue', markersize=8, capsize=5, linewidth=2
    )
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% target')
    ax.fill_between(stats_df['beta'], 0.9, 1.0, alpha=0.1, color='green')
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('CATE Coverage Rate')
    ax.set_ylim(0, 1.1)
    ax.set_title('CATE Coverage vs Confounding')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ATE coverage
    ax = axes[1]
    ax.errorbar(
        stats_df['beta'],
        stats_df['ate_covered_mean'],
        yerr=1.96 * stats_df['ate_covered_std'] / np.sqrt(stats_df['coverage_rate_count']),
        fmt='s-', color='darkorange', markersize=8, capsize=5, linewidth=2
    )
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% target')
    ax.fill_between(stats_df['beta'], 0.9, 1.0, alpha=0.1, color='green')
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('ATE Coverage Rate')
    ax.set_ylim(0, 1.1)
    ax.set_title('ATE Coverage vs Confounding')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_width_vs_coverage(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_vs_coverage.png'
) -> None:
    """Scatter plot of width vs coverage colored by beta."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(
        results_df['mean_width'],
        results_df['coverage_rate'],
        c=results_df['beta'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Reference line
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% target')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'Confounding Strength ($\beta$)')
    
    ax.set_xlabel('Mean Bound Width')
    ax.set_ylabel('CATE Coverage Rate')
    ax.set_title('Coverage vs Width Trade-off')
    ax.legend()
    
    save_figure(fig, Path(output_dir) / filename)


def plot_per_stratum_coverage(
    result: Dict[str, Any],
    output_dir: str,
    filename: str = 'per_stratum_coverage.png'
) -> None:
    """Plot coverage details for each stratum."""
    per_stratum = result['per_stratum']
    
    n_strata = len(per_stratum)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * n_strata + 2))
    
    # Forest plot
    ax = axes[0]
    for i, row in enumerate(per_stratum):
        color = 'green' if row['is_covered'] else 'red'
        ax.hlines(y=i, xmin=row['bound_lower'], xmax=row['bound_upper'],
                  color=color, linewidth=3, alpha=0.7)
        ax.plot(row['true_cate'], i, '*', color='black', markersize=12)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(range(n_strata))
    ax.set_yticklabels([f"Z={r['z']}" for r in per_stratum])
    ax.set_xlabel('CATE')
    ax.set_title('Bounds vs True CATE')
    ax.invert_yaxis()
    
    # Coverage bar
    ax = axes[1]
    colors = ['green' if r['is_covered'] else 'red' for r in per_stratum]
    ax.barh(range(n_strata), [1 if r['is_covered'] else 0 for r in per_stratum],
            color=colors, alpha=0.7)
    ax.set_yticks(range(n_strata))
    ax.set_yticklabels([f"Z={r['z']}" for r in per_stratum])
    ax.set_xlabel('Covered')
    ax.set_title('Coverage by Stratum')
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_bias_analysis(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'bias_analysis.png'
) -> None:
    """Plot bias analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bias by beta
    ax = axes[0]
    bias_by_beta = results_df.groupby('beta')['mean_bias'].agg(['mean', 'std'])
    ax.errorbar(
        bias_by_beta.index,
        bias_by_beta['mean'],
        yerr=bias_by_beta['std'],
        fmt='o-', color='steelblue', markersize=8, capsize=5
    )
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Mean Bias (Midpoint - True)')
    ax.set_title('Bias vs Confounding')
    ax.grid(True, alpha=0.3)
    
    # Bias distribution
    ax = axes[1]
    ax.hist(results_df['mean_bias'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='Zero bias')
    ax.axvline(x=results_df['mean_bias'].mean(), color='blue', linestyle=':',
               label=f"Mean: {results_df['mean_bias'].mean():.3f}")
    ax.set_xlabel('Mean Bias')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bias')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def generate_validation_report(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None
) -> None:
    """Generate comprehensive validation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating validation plots...")
    
    if 'beta' in results_df.columns and len(results_df['beta'].unique()) > 1:
        plot_coverage_by_beta(results_df, output_dir)
        plot_bias_analysis(results_df, output_dir)
    
    if 'mean_width' in results_df.columns and 'coverage_rate' in results_df.columns:
        plot_width_vs_coverage(results_df, output_dir)
    
    if single_result is not None and 'per_stratum' in single_result:
        plot_per_stratum_coverage(single_result, output_dir)
    
    # Compute statistics
    stats_df = compute_coverage_statistics(results_df)
    
    # Save statistics
    stats_path = output_path / 'coverage_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")
    
    # Generate markdown report
    md_lines = [
        '# CATE Coverage Validation Report',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '\n## Summary Statistics\n',
        f'- Total experiments: {len(results_df)}',
        f"- Mean CATE coverage: {results_df['coverage_rate'].mean():.2%}",
        f"- Mean ATE coverage: {results_df['ate_covered'].mean():.2%}",
        f"- Mean bound width: {results_df['mean_width'].mean():.3f}",
        f"- Mean bias: {results_df['mean_bias'].mean():.4f}",
    ]
    
    if len(stats_df) > 1:
        md_lines.extend([
            '\n## Coverage by Configuration\n',
            '| Beta | Coverage | 95% CI | Width |',
            '|------|----------|--------|-------|'
        ])
        for _, row in stats_df.iterrows():
            ci = f"[{row['coverage_ci_lower']:.2f}, {row['coverage_ci_upper']:.2f}]"
            md_lines.append(
                f"| {row['beta']:.2f} | {row['coverage_rate_mean']:.2%} | {ci} | {row['mean_width_mean']:.3f} |"
            )
    
    # Key findings
    overall_coverage = results_df['coverage_rate'].mean()
    md_lines.extend([
        '\n## Key Findings\n',
        f"- Overall CATE coverage rate: **{overall_coverage:.1%}**",
    ])
    
    if overall_coverage >= 0.95:
        md_lines.append("- ✓ Coverage meets 95% target")
    else:
        md_lines.append(f"- ✗ Coverage below 95% target (deficit: {0.95 - overall_coverage:.1%})")
    
    md_path = output_path / 'VALIDATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved report to {md_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate CATE coverage on synthetic data'
    )
    
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Confounding strength')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Naturalness tolerance')
    parser.add_argument('--n-sites', type=int, default=10,
                        help='Number of training sites')
    parser.add_argument('--n-per-site', type=int, default=500,
                        help='Samples per site')
    parser.add_argument('--n-z-values', type=int, default=3,
                        help='Number of Z values')
    parser.add_argument('--n-replications', type=int, default=10,
                        help='Number of replications for sweep')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='validation_results',
                        help='Output directory')
    parser.add_argument('--sweep-beta', action='store_true',
                        help='Sweep over beta values')
    parser.add_argument('--sweep-epsilon', action='store_true',
                        help='Sweep over epsilon values')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation with fewer replications')
    
    args = parser.parse_args()
    
    n_reps = 3 if args.quick else args.n_replications
    
    if args.sweep_beta:
        print("=" * 60)
        print("CATE COVERAGE VALIDATION - BETA SWEEP")
        print("=" * 60)
        
        results_df = validate_coverage_sweep(
            betas=BETAS,
            epsilons=[args.epsilon],
            n_replications=n_reps,
            n_sites=args.n_sites,
            n_per_site=args.n_per_site,
            n_z_values=args.n_z_values,
            base_seed=args.seed,
            output_dir=args.output
        )
        
        generate_validation_report(results_df, args.output)
        
    elif args.sweep_epsilon:
        print("=" * 60)
        print("CATE COVERAGE VALIDATION - EPSILON SWEEP")
        print("=" * 60)
        
        results_df = validate_coverage_sweep(
            betas=[args.beta],
            epsilons=EPSILONS,
            n_replications=n_reps,
            n_sites=args.n_sites,
            n_per_site=args.n_per_site,
            n_z_values=args.n_z_values,
            base_seed=args.seed,
            output_dir=args.output
        )
        
        generate_validation_report(results_df, args.output)
        
    else:
        print("=" * 60)
        print("CATE COVERAGE VALIDATION - SINGLE")
        print("=" * 60)
        
        result = validate_single_experiment(
            n_sites=args.n_sites,
            n_per_site=args.n_per_site,
            n_z_values=args.n_z_values,
            beta=args.beta,
            epsilon=args.epsilon,
            random_seed=args.seed
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        # Print results
        val = result['validation']
        print(f"\nConfiguration:")
        print(f"  Beta: {args.beta}")
        print(f"  Epsilon: {args.epsilon}")
        print(f"  Sites: {args.n_sites}")
        print(f"  Z values: {args.n_z_values}")
        
        print(f"\nCoverage Results:")
        print(f"  CATE coverage: {val['coverage_rate']:.2%} ({val['n_covered']}/{val['n_strata']})")
        print(f"  95% CI: [{val['coverage_ci_lower']:.2%}, {val['coverage_ci_upper']:.2%}]")
        print(f"  ATE covered: {result['metrics']['ate_covered']}")
        
        print(f"\nBound Quality:")
        print(f"  Mean width: {val['mean_width']:.3f}")
        print(f"  Mean bias: {val['mean_bias']:.4f}")
        
        print(f"\nPer-stratum results:")
        for row in result['per_stratum']:
            status = "✓" if row['is_covered'] else "✗"
            print(f"  Z={row['z']}: True={row['true_cate']:.3f}, "
                  f"Bounds=[{row['bound_lower']:.3f}, {row['bound_upper']:.3f}] {status}")
        
        # Generate plots for single result
        print(f"\nGenerating plots in {args.output}...")
        single_df = pd.DataFrame([{
            'beta': args.beta,
            'epsilon': args.epsilon,
            **result['metrics'],
            **result['validation']
        }])
        generate_validation_report(single_df, args.output, result)


if __name__ == '__main__':
    main()
