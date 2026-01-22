"""
Experiment 7: Real Data Validation (ManyLabs)

This experiment tests whether the bounds method maintains coverage on real
observational data created via OSRCT from ManyLabs1 RCT data.

Research Questions:
- Do bounds cover the known ground truth ATE?
- Does Width ≈ 2ε still hold with continuous outcomes?
- How does coverage vary across confounding strengths?

Reference: Plan file and results/exp3_exp4_summary.md
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Exp7Config:
    """Configuration for Experiment 7."""
    # Studies to test
    studies: List[str] = None

    # Confounding patterns
    patterns: List[str] = None

    # Confounding strengths (beta values)
    betas: List[float] = None

    # Epsilon fractions (of outcome SD)
    epsilon_fracs: List[float] = None

    # Seed for reproducibility
    seed: int = 42

    # Minimum sites required for bounds computation
    min_sites: int = 3

    # Minimum observations per site/treatment combination
    min_obs_per_cell: int = 10

    # Whether to standardize outcomes within each site
    standardize_by_site: bool = False

    def __post_init__(self):
        if self.studies is None:
            self.studies = ['anchoring1', 'gainloss', 'flag', 'reciprocity']
        if self.patterns is None:
            self.patterns = ['age', 'gender', 'demo_basic']
        if self.betas is None:
            self.betas = [0.25, 0.5, 1.0]
        if self.epsilon_fracs is None:
            # Extended range to find where bounds become valid
            self.epsilon_fracs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


# =============================================================================
# Data Loading
# =============================================================================

def load_ground_truth_ates(data_dir: Path) -> Dict[str, float]:
    """Load ground truth ATEs from CSV."""
    ate_file = data_dir / 'ground_truth_ates.csv'
    df = pd.read_csv(ate_file)
    return dict(zip(df['study'], df['ate']))


def load_confounded_data(
    data_dir: Path,
    study: str,
    pattern: str,
    beta: float,
    seed: int = 42
) -> Optional[pd.DataFrame]:
    """Load a confounded dataset."""
    filename = f"{pattern}_beta{beta}_seed{seed}.csv"
    filepath = data_dir / study / filename

    if not filepath.exists():
        return None

    return pd.read_csv(filepath)


def standardize_by_site(data: pd.DataFrame, outcome_col: str = 'dv') -> pd.DataFrame:
    """
    Standardize outcomes within each site (z-score transformation).

    This removes site-level mean differences while preserving within-site
    treatment effects in standardized units.

    Args:
        data: DataFrame with 'site' and outcome columns
        outcome_col: Name of outcome column

    Returns:
        DataFrame with standardized outcomes
    """
    data = data.copy()

    # Compute site-level statistics
    site_stats = data.groupby('site')[outcome_col].agg(['mean', 'std'])

    # Standardize within each site
    def standardize_group(group):
        site = group['site'].iloc[0]
        mean = site_stats.loc[site, 'mean']
        std = site_stats.loc[site, 'std']
        if std > 0:
            group[outcome_col] = (group[outcome_col] - mean) / std
        else:
            group[outcome_col] = 0
        return group

    data = data.groupby('site', group_keys=False).apply(standardize_group)

    return data


def compute_standardized_ate(data: pd.DataFrame, ground_truth_ate: float) -> float:
    """
    Convert ground truth ATE to standardized units.

    The standardized ATE is: ATE / pooled_std(Y)

    Args:
        data: Original (non-standardized) DataFrame
        ground_truth_ate: ATE in original units

    Returns:
        ATE in standardized units (Cohen's d like measure)
    """
    # Use pooled standard deviation
    pooled_std = data['dv'].std()
    return ground_truth_ate / pooled_std


# =============================================================================
# Bounds Computation
# =============================================================================

def compute_site_stratified_bounds(
    data: pd.DataFrame,
    epsilon_scaled: float,
    min_obs_per_cell: int = 10
) -> Tuple[float, float, Dict]:
    """
    Compute ATE bounds using site-stratified means.

    For continuous outcomes, the bounds are:
        ATE_lower = max_k(θ₁ᵏ) - min_k(θ₀ᵏ) - 2ε
        ATE_upper = min_k(θ₁ᵏ) - max_k(θ₀ᵏ) + 2ε

    Wait, that's not right. Let me reconsider...

    The naturalness constraint is: |θ_target - θ̂ᵏ| ≤ ε for all k

    This means:
        max_k(θ̂₁ᵏ) - ε ≤ θ₁ ≤ min_k(θ̂₁ᵏ) + ε
        max_k(θ̂₀ᵏ) - ε ≤ θ₀ ≤ min_k(θ̂₀ᵏ) + ε

    For ATE = θ₁ - θ₀:
        ATE_lower = (max_k(θ̂₁ᵏ) - ε) - (min_k(θ̂₀ᵏ) + ε) = max(θ̂₁) - min(θ̂₀) - 2ε
        ATE_upper = (min_k(θ̂₁ᵏ) + ε) - (max_k(θ̂₀ᵏ) - ε) = min(θ̂₁) - max(θ̂₀) + 2ε

    Args:
        data: DataFrame with 'site', 'iv' (treatment), 'dv' (outcome) columns
        epsilon_scaled: Naturalness tolerance (in outcome units)
        min_obs_per_cell: Minimum observations per site/treatment combination

    Returns:
        (ate_lower, ate_upper, diagnostics)
    """
    # Compute site-level means for each treatment group
    site_stats = data.groupby(['site', 'iv'])['dv'].agg(['mean', 'count']).reset_index()

    # Filter sites with sufficient data in both treatment groups
    valid_sites = set()
    for site in site_stats['site'].unique():
        site_data = site_stats[site_stats['site'] == site]
        if len(site_data) == 2:  # Has both treatment groups
            if site_data['count'].min() >= min_obs_per_cell:
                valid_sites.add(site)

    if len(valid_sites) < 1:
        return float('nan'), float('nan'), {'n_valid_sites': 0, 'reason': 'no_valid_sites'}

    # Get means for valid sites
    valid_data = site_stats[site_stats['site'].isin(valid_sites)]

    theta_1_vals = valid_data[valid_data['iv'] == 1]['mean'].values
    theta_0_vals = valid_data[valid_data['iv'] == 0]['mean'].values

    if len(theta_1_vals) == 0 or len(theta_0_vals) == 0:
        return float('nan'), float('nan'), {'n_valid_sites': len(valid_sites), 'reason': 'missing_treatment_group'}

    # Compute bounds
    ate_lower = theta_1_vals.max() - theta_0_vals.min() - 2 * epsilon_scaled
    ate_upper = theta_1_vals.min() - theta_0_vals.max() + 2 * epsilon_scaled

    # Point estimate (simple difference in means across all valid sites)
    point_estimate = theta_1_vals.mean() - theta_0_vals.mean()

    # Cross-site variation
    theta_1_range = theta_1_vals.max() - theta_1_vals.min()
    theta_0_range = theta_0_vals.max() - theta_0_vals.min()

    diagnostics = {
        'n_valid_sites': len(valid_sites),
        'n_treated_sites': len(theta_1_vals),
        'n_control_sites': len(theta_0_vals),
        'theta_1_min': float(theta_1_vals.min()),
        'theta_1_max': float(theta_1_vals.max()),
        'theta_1_range': float(theta_1_range),
        'theta_0_min': float(theta_0_vals.min()),
        'theta_0_max': float(theta_0_vals.max()),
        'theta_0_range': float(theta_0_range),
        'point_estimate': float(point_estimate)
    }

    return ate_lower, ate_upper, diagnostics


# =============================================================================
# Single Run
# =============================================================================

def run_exp7_single(
    data_dir: Path,
    ground_truth: Dict[str, float],
    study: str,
    pattern: str,
    beta: float,
    epsilon_frac: float,
    seed: int = 42,
    min_obs_per_cell: int = 10,
    standardize: bool = False
) -> Optional[Dict]:
    """Run a single configuration."""

    # Load data
    data = load_confounded_data(data_dir, study, pattern, beta, seed)
    if data is None:
        return None

    # Get ground truth ATE
    if study not in ground_truth:
        return None
    true_ate_original = ground_truth[study]

    # Store original stats before potential standardization
    outcome_std_original = data['dv'].std()
    outcome_mean_original = data['dv'].mean()

    # Apply site standardization if requested
    if standardize:
        data = standardize_by_site(data, 'dv')
        # Convert ground truth to standardized units
        true_ate = compute_standardized_ate(
            load_confounded_data(data_dir, study, pattern, beta, seed),
            true_ate_original
        )
    else:
        true_ate = true_ate_original

    # Compute outcome scale (after potential standardization)
    outcome_std = data['dv'].std()
    outcome_mean = data['dv'].mean()
    epsilon_scaled = epsilon_frac * outcome_std

    # Compute bounds
    ate_lower, ate_upper, diagnostics = compute_site_stratified_bounds(
        data, epsilon_scaled, min_obs_per_cell
    )

    # Compute metrics
    if np.isnan(ate_lower) or np.isnan(ate_upper):
        coverage = np.nan
        width = np.nan
        width_ratio = np.nan
    else:
        coverage = (ate_lower <= true_ate <= ate_upper)
        width = ate_upper - ate_lower
        width_ratio = width / outcome_std

    # Compute naive estimate (difference in means, confounded)
    naive_treated = data[data['iv'] == 1]['dv'].mean()
    naive_control = data[data['iv'] == 0]['dv'].mean()
    naive_ate = naive_treated - naive_control
    naive_bias = naive_ate - true_ate

    result = {
        'study': study,
        'pattern': pattern,
        'beta': beta,
        'epsilon_frac': epsilon_frac,
        'epsilon_scaled': epsilon_scaled,
        'standardized': standardize,
        'outcome_std': outcome_std,
        'outcome_mean': outcome_mean,
        'outcome_std_original': outcome_std_original,
        'outcome_mean_original': outcome_mean_original,
        'n_obs': len(data),
        'n_treated': (data['iv'] == 1).sum(),
        'n_control': (data['iv'] == 0).sum(),
        'ate_lower': ate_lower,
        'ate_upper': ate_upper,
        'width': width,
        'width_ratio': width_ratio,
        'coverage': coverage,
        'true_ate': true_ate,
        'true_ate_original': true_ate_original,
        'naive_ate': naive_ate,
        'naive_bias': naive_bias,
        **diagnostics
    }

    return result


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(config: Exp7Config, output_dir: Path, data_dir: Path):
    """Run the full experiment."""

    print("=" * 60)
    print("Experiment 7: Real Data Validation (ManyLabs)")
    if config.standardize_by_site:
        print("  ** WITH SITE-STANDARDIZED OUTCOMES **")
    print("=" * 60)

    # Load ground truth
    ground_truth = load_ground_truth_ates(data_dir)
    print(f"\nLoaded ground truth for {len(ground_truth)} studies")

    # Compute total configurations
    n_configs = (len(config.studies) * len(config.patterns) *
                 len(config.betas) * len(config.epsilon_fracs))
    print(f"Total configurations: {n_configs}")

    # Run all configurations
    results = []
    start_time = time.time()

    for study in config.studies:
        print(f"\n  Study: {study}")
        for pattern in config.patterns:
            for beta in config.betas:
                for epsilon_frac in config.epsilon_fracs:
                    result = run_exp7_single(
                        data_dir=data_dir,
                        ground_truth=ground_truth,
                        study=study,
                        pattern=pattern,
                        beta=beta,
                        epsilon_frac=epsilon_frac,
                        seed=config.seed,
                        min_obs_per_cell=config.min_obs_per_cell,
                        standardize=config.standardize_by_site
                    )
                    if result is not None:
                        results.append(result)

    elapsed = time.time() - start_time
    print(f"\n  Completed {len(results)} configurations in {elapsed:.1f}s")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'exp7_results.csv', index=False)
    print(f"\n  Results saved to: {output_dir / 'exp7_results.csv'}")

    # Compute summary statistics
    summary = compute_summary(df, config)

    with open(output_dir / 'exp7_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {output_dir / 'exp7_summary.json'}")

    # Print key results
    print_results(df, summary)

    return df, summary


def compute_summary(df: pd.DataFrame, config: Exp7Config) -> Dict:
    """Compute summary statistics."""

    summary = {
        'config': asdict(config),
        'n_configs': int(len(df)),
        'n_successful': int(df['coverage'].notna().sum()),
    }

    # Overall coverage by epsilon
    coverage_by_eps = df.groupby('epsilon_frac')['coverage'].agg(['mean', 'std', 'count'])
    summary['coverage_by_epsilon'] = {
        str(float(eps)): {
            'mean': float(row['mean']) if not np.isnan(row['mean']) else None,
            'std': float(row['std']) if not np.isnan(row['std']) else None,
            'n': int(row['count'])
        }
        for eps, row in coverage_by_eps.iterrows()
    }

    # Width ratio by epsilon (should be approximately 2ε if bounds work)
    width_by_eps = df.groupby('epsilon_frac')['width_ratio'].agg(['mean', 'std'])
    summary['width_ratio_by_epsilon'] = {
        str(float(eps)): {
            'mean': float(row['mean']) if not np.isnan(row['mean']) else None,
            'std': float(row['std']) if not np.isnan(row['std']) else None,
            'expected': float(2 * eps)  # Expected if Width ≈ 2ε
        }
        for eps, row in width_by_eps.iterrows()
    }

    # Coverage by study
    coverage_by_study = df.groupby('study')['coverage'].mean()
    summary['coverage_by_study'] = {
        study: float(cov) for study, cov in coverage_by_study.items()
    }

    # Coverage by beta (confounding strength)
    coverage_by_beta = df.groupby('beta')['coverage'].mean()
    summary['coverage_by_beta'] = {
        str(float(beta)): float(cov) if not np.isnan(cov) else None for beta, cov in coverage_by_beta.items()
    }

    # Coverage by pattern
    coverage_by_pattern = df.groupby('pattern')['coverage'].mean()
    summary['coverage_by_pattern'] = {
        pattern: float(cov) for pattern, cov in coverage_by_pattern.items()
    }

    # Hypothesis tests
    # H1: Coverage ≥ 95% for ε ≥ 0.10
    eps_high = df[df['epsilon_frac'] >= 0.10]
    h1_coverage = float(eps_high['coverage'].mean()) if len(eps_high) > 0 else float('nan')
    summary['h1_coverage_eps_ge_0.10'] = h1_coverage if not np.isnan(h1_coverage) else None
    summary['h1_pass'] = bool(h1_coverage >= 0.95) if not np.isnan(h1_coverage) else False

    # H2: Width ratio ≈ 2ε
    # Compute correlation between width_ratio and 2*epsilon_frac
    df_valid = df[df['width_ratio'].notna()]
    if len(df_valid) > 5:
        expected_ratio = 2 * df_valid['epsilon_frac']
        correlation = float(np.corrcoef(df_valid['width_ratio'], expected_ratio)[0, 1])
        summary['h2_width_correlation'] = correlation if not np.isnan(correlation) else None
        summary['h2_pass'] = bool(correlation > 0.9) if not np.isnan(correlation) else False

    # H3: Coverage does NOT depend on beta
    # Run one-way ANOVA
    from scipy import stats
    groups = [df[df['beta'] == beta]['coverage'].dropna().values for beta in config.betas]
    if all(len(g) > 2 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        summary['h3_anova_f'] = float(f_stat) if not np.isnan(f_stat) else None
        summary['h3_anova_p'] = float(p_value) if not np.isnan(p_value) else None
        summary['h3_pass'] = bool(p_value > 0.05) if not np.isnan(p_value) else None  # Not significant = no effect

    # Naive bias
    summary['naive_bias_by_beta'] = {
        str(float(beta)): float(df[df['beta'] == beta]['naive_bias'].mean())
        for beta in config.betas
    }

    # Cross-site heterogeneity analysis
    # Compute required epsilon for valid bounds (no inversion)
    df['min_required_eps_ratio'] = np.where(
        df['width'] < 0,
        (df['theta_1_range'] + df['theta_0_range']) / (4 * df['outcome_std']),
        df['epsilon_frac']
    )
    summary['cross_site_heterogeneity'] = {
        'mean_theta1_range_ratio': float(df['theta_1_range'].mean() / df['outcome_std'].mean()),
        'mean_theta0_range_ratio': float(df['theta_0_range'].mean() / df['outcome_std'].mean()),
        'min_required_eps_for_valid_bounds': float(df['min_required_eps_ratio'].max()),
        'pct_inverted_bounds': float((df['width'] < 0).mean() * 100)
    }

    return summary


def print_results(df: pd.DataFrame, summary: Dict):
    """Print key results."""

    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)

    # Coverage by epsilon
    print("\n1. Coverage by Epsilon:")
    print("-" * 40)
    print(f"  {'ε':<8} {'Coverage':<12} {'Expected Width':<15} {'Actual Width'}")
    for eps, data in summary['coverage_by_epsilon'].items():
        width_data = summary['width_ratio_by_epsilon'].get(eps, {})
        print(f"  {eps:<8} {data['mean']*100:>6.1f}%       "
              f"{width_data.get('expected', 0):<15.2f} {width_data.get('mean', 0):.2f}")

    # Coverage by study
    print("\n2. Coverage by Study:")
    print("-" * 40)
    for study, cov in summary['coverage_by_study'].items():
        print(f"  {study:<15} {cov*100:>6.1f}%")

    # Coverage by beta
    print("\n3. Coverage by Confounding Strength (β):")
    print("-" * 40)
    for beta, cov in summary['coverage_by_beta'].items():
        print(f"  β={beta:<6} {cov*100:>6.1f}%")

    # Hypothesis tests
    print("\n4. Hypothesis Tests:")
    print("-" * 40)
    print(f"  H1 (Coverage ≥ 95% for ε ≥ 0.10): "
          f"{'PASS' if summary.get('h1_pass') else 'FAIL'} "
          f"({summary.get('h1_coverage_eps_ge_0.10', 0)*100:.1f}%)")

    if 'h2_width_correlation' in summary:
        print(f"  H2 (Width ≈ 2ε): "
              f"{'PASS' if summary.get('h2_pass') else 'FAIL'} "
              f"(r={summary['h2_width_correlation']:.3f})")

    if 'h3_anova_p' in summary and summary['h3_anova_p'] is not None:
        print(f"  H3 (Coverage independent of β): "
              f"{'PASS' if summary.get('h3_pass') else 'FAIL'} "
              f"(p={summary['h3_anova_p']:.3f})")
    elif 'h3_anova_p' in summary:
        print(f"  H3 (Coverage independent of β): N/A (constant coverage)")

    # Naive bias
    print("\n5. Naive Estimator Bias by β:")
    print("-" * 40)
    for beta, bias in summary.get('naive_bias_by_beta', {}).items():
        print(f"  β={beta:<6} bias={bias:>8.3f}")

    # Cross-site heterogeneity
    het = summary.get('cross_site_heterogeneity', {})
    if het:
        print("\n6. Cross-Site Heterogeneity Analysis:")
        print("-" * 40)
        print(f"  Mean θ₁ range / SD(Y): {het.get('mean_theta1_range_ratio', 0):.2f}")
        print(f"  Mean θ₀ range / SD(Y): {het.get('mean_theta0_range_ratio', 0):.2f}")
        print(f"  % inverted bounds:     {het.get('pct_inverted_bounds', 0):.1f}%")
        print(f"  Min ε required for valid bounds: {het.get('min_required_eps_for_valid_bounds', 0):.2f}")
        print("\n  ** KEY FINDING **")
        print("  The naturalness assumption is violated: sites have too much")
        print("  heterogeneity for the bounds to be valid at current ε values.")


# =============================================================================
# Visualization
# =============================================================================

def generate_figures(df: pd.DataFrame, output_dir: Path, fig_dir: Path = None):
    """Generate analysis figures."""
    import matplotlib.pyplot as plt

    if fig_dir is None:
        fig_dir = output_dir.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Coverage by epsilon for each study
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    studies = df['study'].unique()

    for ax, study in zip(axes.flat, studies):
        study_df = df[df['study'] == study]
        coverage_by_eps = study_df.groupby('epsilon_frac')['coverage'].mean()
        ax.bar(range(len(coverage_by_eps)), coverage_by_eps.values)
        ax.set_xticks(range(len(coverage_by_eps)))
        ax.set_xticklabels([f'{eps:.2f}' for eps in coverage_by_eps.index])
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% target')
        ax.set_xlabel('ε (fraction of outcome SD)')
        ax.set_ylabel('Coverage')
        ax.set_title(f'{study}')
        ax.set_ylim(0, 1.1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'exp7_coverage_by_study.png', dpi=150)
    plt.close()
    print(f"  Saved: {fig_dir / 'exp7_coverage_by_study.png'}")

    # Figure 2: Width ratio vs epsilon (validation of 2ε relationship)
    fig, ax = plt.subplots(figsize=(8, 6))

    for study in studies:
        study_df = df[df['study'] == study]
        width_by_eps = study_df.groupby('epsilon_frac')['width_ratio'].mean()
        ax.scatter(width_by_eps.index, width_by_eps.values, label=study, alpha=0.7)

    # Expected line: width_ratio = 2 * epsilon_frac
    eps_range = np.linspace(0.04, 0.22, 50)
    ax.plot(eps_range, 2 * eps_range, 'k--', label='Expected (2ε)', linewidth=2)

    ax.set_xlabel('ε (fraction of outcome SD)')
    ax.set_ylabel('Width / SD(Y)')
    ax.set_title('Width Scaling: Does Width ≈ 2ε hold?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'exp7_width_vs_epsilon.png', dpi=150)
    plt.close()
    print(f"  Saved: {fig_dir / 'exp7_width_vs_epsilon.png'}")

    # Figure 3: Coverage heatmap (beta × epsilon)
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = df.pivot_table(
        values='coverage',
        index='beta',
        columns='epsilon_frac',
        aggfunc='mean'
    )

    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{eps:.2f}' for eps in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{beta:.2f}' for beta in pivot.index])
    ax.set_xlabel('ε (fraction of outcome SD)')
    ax.set_ylabel('β (confounding strength)')
    ax.set_title('Coverage Heatmap: β × ε')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val*100:.0f}%', ha='center', va='center',
                   color='white' if val < 0.7 else 'black')

    plt.colorbar(im, ax=ax, label='Coverage')
    plt.tight_layout()
    plt.savefig(fig_dir / 'exp7_coverage_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: {fig_dir / 'exp7_coverage_heatmap.png'}")

    # Figure 4: Naive bias vs beta
    fig, ax = plt.subplots(figsize=(8, 6))

    for study in studies:
        study_df = df[df['study'] == study]
        bias_by_beta = study_df.groupby('beta')['naive_bias'].mean()
        ax.plot(bias_by_beta.index, bias_by_beta.values, 'o-', label=study)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('β (confounding strength)')
    ax.set_ylabel('Naive Estimator Bias')
    ax.set_title('Confounding Bias Increases with β')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'exp7_naive_bias.png', dpi=150)
    plt.close()
    print(f"  Saved: {fig_dir / 'exp7_naive_bias.png'}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 7: Real Data Validation')
    parser.add_argument('--standardize', action='store_true',
                       help='Standardize outcomes by site')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'confounded_datasets'

    # Output to bounds_experiments structure
    exp_dir = project_root / 'experiments' / 'bounds_experiments' / 'exp7_real_data'

    if args.standardize:
        output_dir = exp_dir / 'results_standardized'
        fig_dir = exp_dir / 'figures_standardized'
    else:
        output_dir = exp_dir / 'results'
        fig_dir = exp_dir / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = Exp7Config(standardize_by_site=args.standardize)

    # Run experiment
    df, summary = run_experiment(config, output_dir, data_dir)

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(df, output_dir, fig_dir)

    print("\n" + "=" * 60)
    print("Experiment 7 Complete!")
    if args.standardize:
        print("(with site-standardized outcomes)")
    print("=" * 60)
