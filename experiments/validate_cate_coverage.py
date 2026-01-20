"""
Validate CATE bounds coverage against ground truth computed from RCT.

This addresses the question: Do our CATE bounds actually contain the true conditional effects?

Key insight: We compute "ground truth" CATE(z) directly from RCT data where treatment is randomized,
so E[Y|X=1,Z=z] - E[Y|X=0,Z=z] is an unbiased estimate of the true CATE.

Usage:
    python experiments/validate_cate_coverage.py --study anchoring1 --beta 0.3
    python experiments/validate_cate_coverage.py --mode grid
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CausalGroundingEstimator,
    create_train_target_split,
    load_rct_data,
    discretize_covariates
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
# CONSTANTS
# =============================================================================

STUDIES = [
    'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4',
    'gamblerfallacy', 'sunkfallacy', 'gainloss', 'quote'
]

BETAS = [0.1, 0.25, 0.5, 0.75, 1.0]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig, output_path, close=True):
    """Save figure as high-quality PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format='png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")
    if close:
        plt.close(fig)


def find_osrct_file(study, pattern, beta, data_dir):
    """Find OSRCT data file."""
    data_path = Path(data_dir)
    study_dir = data_path / study

    if not study_dir.exists():
        return None

    # Try multiple naming patterns
    patterns_to_try = [
        f"{pattern}_beta{beta}_seed42.csv",
        f"{study}_{pattern}_beta_{beta}.pkl",
        f"{study}_{pattern}_beta_{beta}.csv",
    ]

    for file_pattern in patterns_to_try:
        candidate = study_dir / file_pattern
        if candidate.exists():
            return candidate

    # Fallback: any file with matching pattern and beta
    for f in study_dir.glob(f"*{pattern}*beta*{beta}*"):
        return f

    return None


# =============================================================================
# CATE GROUND TRUTH COMPUTATION
# =============================================================================

def compute_cate_ground_truth_with_ci(
    rct_data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list,
    min_samples: int = 20,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute CATE ground truth with confidence intervals from RCT data.

    For each covariate stratum z:
    - CATE(z) = E[Y|X=1,Z=z] - E[Y|X=0,Z=z]
    - SE = sqrt(var(Y|X=1,Z=z)/n1 + var(Y|X=0,Z=z)/n0)
    - CI using t-distribution

    Args:
        rct_data: RCT DataFrame with randomized treatment
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of covariate names to stratify by
        min_samples: Minimum samples per treatment arm per stratum
        confidence: Confidence level for CI

    Returns:
        DataFrame with columns: z, cate, se, ci_lower, ci_upper, n_treated, n_control
    """
    # Discretize covariates
    data = discretize_covariates(rct_data.copy())

    # Build list of discretized covariate column names
    discretized_cols = []
    for cov in covariates:
        cat_col = f"{cov}_cat"
        if cat_col in data.columns:
            discretized_cols.append(cat_col)
        elif cov in data.columns:
            discretized_cols.append(cov)

    # Filter to columns that exist
    discretized_cols = [c for c in discretized_cols if c in data.columns]

    if len(discretized_cols) == 0:
        return pd.DataFrame()

    results = []

    try:
        grouped = data.groupby(discretized_cols, observed=True)

        for z_values, group in grouped:
            # Ensure z_values is a tuple
            if not isinstance(z_values, tuple):
                z_values = (z_values,)

            # Split by treatment
            treated = group[group[treatment] == 1][outcome].dropna()
            control = group[group[treatment] == 0][outcome].dropna()

            n_treated = len(treated)
            n_control = len(control)

            # Check minimum sample size
            if n_treated >= min_samples and n_control >= min_samples:
                # Compute means
                y1_mean = treated.mean()
                y0_mean = control.mean()
                cate = y1_mean - y0_mean

                # Compute standard errors
                se1 = treated.std(ddof=1) / np.sqrt(n_treated)
                se0 = control.std(ddof=1) / np.sqrt(n_control)
                se = np.sqrt(se1**2 + se0**2)

                # Compute CI using t-distribution
                df = n_treated + n_control - 2
                t_crit = stats.t.ppf((1 + confidence) / 2, df)
                ci_lower = cate - t_crit * se
                ci_upper = cate + t_crit * se

                results.append({
                    'z': z_values,
                    'z_str': str(z_values),
                    'cate': cate,
                    'se': se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_treated': n_treated,
                    'n_control': n_control,
                    'y1_mean': y1_mean,
                    'y0_mean': y0_mean
                })

    except Exception as e:
        print(f"Warning: Error computing CATE ground truth: {e}")

    return pd.DataFrame(results)


# =============================================================================
# COVERAGE EVALUATION
# =============================================================================

def evaluate_cate_coverage(bounds_df: pd.DataFrame, cate_truth_df: pd.DataFrame) -> dict:
    """
    Evaluate coverage of CATE bounds against ground truth.

    Args:
        bounds_df: DataFrame with estimated bounds (cate_lower, cate_upper)
        cate_truth_df: DataFrame with ground truth CATEs and CIs

    Returns:
        Dict with coverage metrics and per-stratum results
    """
    if len(bounds_df) == 0 or len(cate_truth_df) == 0:
        return {
            'n_strata_evaluated': 0,
            'n_covered': 0,
            'coverage_rate': np.nan,
            'n_covered_with_uncertainty': 0,
            'per_stratum_results': []
        }

    per_stratum_results = []
    n_covered = 0
    n_covered_with_uncertainty = 0
    n_evaluated = 0

    # Match strata between bounds and truth
    for _, truth_row in cate_truth_df.iterrows():
        z_truth = truth_row['z']
        z_str = truth_row['z_str']
        true_cate = truth_row['cate']
        cate_ci_lower = truth_row['ci_lower']
        cate_ci_upper = truth_row['ci_upper']

        # Find matching bounds row
        matched = False
        for _, bounds_row in bounds_df.iterrows():
            # Get z from bounds
            z_bounds = bounds_row.get('z', None)

            if z_bounds is not None:
                # Convert to tuple for comparison
                if not isinstance(z_bounds, tuple):
                    if hasattr(z_bounds, '__iter__') and not isinstance(z_bounds, str):
                        z_bounds = tuple(z_bounds)
                    else:
                        z_bounds = (z_bounds,)

                if z_bounds == z_truth or str(z_bounds) == z_str:
                    matched = True
                    bound_lower = bounds_row['cate_lower']
                    bound_upper = bounds_row['cate_upper']
                    bound_width = bounds_row.get('width', bound_upper - bound_lower)

                    # Check if true CATE is covered by bounds
                    is_covered = bound_lower <= true_cate <= bound_upper

                    # Check if CI overlaps with bounds
                    ci_overlaps = not (cate_ci_upper < bound_lower or cate_ci_lower > bound_upper)

                    if is_covered:
                        n_covered += 1
                    if ci_overlaps:
                        n_covered_with_uncertainty += 1
                    n_evaluated += 1

                    per_stratum_results.append({
                        'z': z_str,
                        'true_cate': true_cate,
                        'cate_ci_lower': cate_ci_lower,
                        'cate_ci_upper': cate_ci_upper,
                        'bound_lower': bound_lower,
                        'bound_upper': bound_upper,
                        'bound_width': bound_width,
                        'is_covered': is_covered,
                        'ci_overlaps_bounds': ci_overlaps
                    })
                    break

        if not matched:
            # Record that this stratum had no matching bounds
            per_stratum_results.append({
                'z': z_str,
                'true_cate': true_cate,
                'cate_ci_lower': cate_ci_lower,
                'cate_ci_upper': cate_ci_upper,
                'bound_lower': np.nan,
                'bound_upper': np.nan,
                'bound_width': np.nan,
                'is_covered': np.nan,
                'ci_overlaps_bounds': np.nan
            })

    coverage_rate = n_covered / n_evaluated if n_evaluated > 0 else np.nan

    return {
        'n_strata_evaluated': n_evaluated,
        'n_covered': n_covered,
        'coverage_rate': coverage_rate,
        'n_covered_with_uncertainty': n_covered_with_uncertainty,
        'coverage_with_uncertainty': n_covered_with_uncertainty / n_evaluated if n_evaluated > 0 else np.nan,
        'per_stratum_results': per_stratum_results
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_cate_coverage_details(
    bounds_df: pd.DataFrame,
    cate_truth_df: pd.DataFrame,
    output_dir: str,
    study: str = '',
    beta: float = 0.0
):
    """Create detailed CATE coverage visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get coverage evaluation
    coverage_results = evaluate_cate_coverage(bounds_df, cate_truth_df)
    per_stratum = coverage_results['per_stratum_results']

    if len(per_stratum) == 0:
        print("  No strata to plot")
        return

    per_stratum_df = pd.DataFrame(per_stratum)
    valid_df = per_stratum_df.dropna(subset=['bound_lower', 'bound_upper'])

    if len(valid_df) == 0:
        print("  No valid strata with bounds to plot")
        return

    # Limit to first 20 for readability
    valid_df = valid_df.head(20)

    # Plot 1: Forest plot with true CATEs and CIs
    n_strata = len(valid_df)
    fig_height = max(6, n_strata * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = range(n_strata)

    for i, (_, row) in enumerate(valid_df.iterrows()):
        # Draw bound interval
        color = 'green' if row['is_covered'] else 'red'
        ax.hlines(y=i, xmin=row['bound_lower'], xmax=row['bound_upper'],
                  color=color, linewidth=3, alpha=0.6, label='_nolegend_')

        # Draw true CATE with CI as error bar
        ax.errorbar(row['true_cate'], i,
                    xerr=[[row['true_cate'] - row['cate_ci_lower']],
                          [row['cate_ci_upper'] - row['true_cate']]],
                    fmt='o', color='black', markersize=8, capsize=4,
                    label='_nolegend_')

    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(valid_df['z'].tolist())
    ax.set_xlabel('CATE')
    ax.set_ylabel('Stratum (Z)')

    title = 'CATE Bounds vs Ground Truth'
    if study:
        title += f' ({study}'
        if beta:
            title += f', β={beta}'
        title += ')'
    ax.set_title(title)

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Covered'),
        Patch(facecolor='red', alpha=0.6, label='Not Covered'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=8, label='True CATE ± CI')
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.invert_yaxis()
    suffix = f"_{study}_{beta}" if study else ""
    save_figure(fig, output_path / f'cate_forest_detailed{suffix}.png')

    # Plot 2: Coverage by stratum (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if c else 'red' for c in valid_df['is_covered']]
    bars = ax.bar(range(len(valid_df)), valid_df['is_covered'].astype(float),
                  color=colors, alpha=0.7)

    ax.set_xticks(range(len(valid_df)))
    ax.set_xticklabels(valid_df['z'].tolist(), rotation=45, ha='right')
    ax.set_xlabel('Stratum (Z)')
    ax.set_ylabel('Covered (1) or Not (0)')
    ax.set_ylim(-0.1, 1.1)

    coverage_rate = valid_df['is_covered'].mean()
    ax.axhline(y=coverage_rate, color='blue', linestyle='--',
               label=f'Coverage Rate: {coverage_rate:.2f}')
    ax.legend(loc='best')

    title = 'CATE Coverage by Stratum'
    if study:
        title += f' ({study}, β={beta})'
    ax.set_title(title)

    save_figure(fig, output_path / f'cate_coverage_by_stratum{suffix}.png')

    # Plot 3: Bound width vs CATE uncertainty
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_df['cate_se'] = (valid_df['cate_ci_upper'] - valid_df['cate_ci_lower']) / (2 * 1.96)
    colors = ['green' if c else 'red' for c in valid_df['is_covered']]

    ax.scatter(valid_df['bound_width'], valid_df['cate_se'],
               c=colors, s=80, alpha=0.7)

    ax.set_xlabel('Bound Width')
    ax.set_ylabel('CATE Standard Error')

    title = 'Bound Width vs CATE Uncertainty'
    if study:
        title += f' ({study}, β={beta})'
    ax.set_title(title)

    # Legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Covered'),
        Patch(facecolor='red', alpha=0.7, label='Not Covered')
    ]
    ax.legend(handles=legend_elements, loc='best')

    save_figure(fig, output_path / f'width_vs_uncertainty{suffix}.png')


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def run_cate_validation(
    study: str,
    pattern: str,
    beta: float,
    epsilon: float = 0.1,
    target_site: str = 'mturk',
    data_dir: str = 'confounded_datasets',
    rct_path: str = 'ManyLabs1/pre-process/Manylabs1_data.pkl',
    n_permutations: int = 100,
    min_samples: int = 20
) -> dict:
    """
    Run full CATE validation for one dataset.

    Returns dict with bounds, ground truth, and coverage metrics.
    """
    results = {
        'config': {
            'study': study,
            'pattern': pattern,
            'beta': beta,
            'epsilon': epsilon,
            'min_samples': min_samples
        }
    }

    # Find and load OSRCT data
    osrct_path = find_osrct_file(study, pattern, beta, data_dir)
    if osrct_path is None:
        results['error'] = f"OSRCT data not found for {study}/{pattern}/beta={beta}"
        return results

    try:
        if str(osrct_path).endswith('.pkl'):
            osrct_data = pd.read_pickle(osrct_path)
        else:
            osrct_data = pd.read_csv(osrct_path)
    except Exception as e:
        results['error'] = f"Failed to load OSRCT: {e}"
        return results

    # Load RCT data
    try:
        rct_data = load_rct_data(study, rct_path)
    except Exception as e:
        results['error'] = f"Failed to load RCT: {e}"
        return results

    # Create train/target split
    try:
        training_data, target_data = create_train_target_split(
            osrct_data, rct_data, target_site=target_site
        )
    except Exception as e:
        results['error'] = f"Failed to create split: {e}"
        return results

    # Fit estimator and get bounds
    try:
        estimator = CausalGroundingEstimator(
            epsilon=epsilon,
            n_permutations=n_permutations,
            random_seed=42,
            verbose=False
        )
        estimator.fit(training_data, treatment='iv', outcome='dv')
        bounds_df = estimator.predict_bounds()
        results['bounds'] = bounds_df.to_dict('records')
        results['n_bounds_strata'] = len(bounds_df)
    except Exception as e:
        results['error'] = f"Failed to fit estimator: {e}"
        return results

    # Compute CATE ground truth from RCT
    try:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']
        cate_truth_df = compute_cate_ground_truth_with_ci(
            rct_data,
            treatment='iv',
            outcome='dv',
            covariates=covariates,
            min_samples=min_samples
        )
        results['cate_truth'] = cate_truth_df.to_dict('records')
        results['n_truth_strata'] = len(cate_truth_df)
    except Exception as e:
        results['error'] = f"Failed to compute CATE truth: {e}"
        return results

    # Evaluate coverage
    try:
        coverage_results = evaluate_cate_coverage(bounds_df, cate_truth_df)
        results['coverage'] = {
            'n_strata_evaluated': coverage_results['n_strata_evaluated'],
            'n_covered': coverage_results['n_covered'],
            'coverage_rate': coverage_results['coverage_rate'],
            'n_covered_with_uncertainty': coverage_results['n_covered_with_uncertainty'],
            'coverage_with_uncertainty': coverage_results['coverage_with_uncertainty']
        }
        results['per_stratum'] = coverage_results['per_stratum_results']
    except Exception as e:
        results['error'] = f"Failed to evaluate coverage: {e}"
        return results

    return results


def run_cate_validation_grid(
    studies: list = None,
    betas: list = None,
    pattern: str = 'age',
    epsilon: float = 0.1,
    output_dir: str = 'results/cate_validation',
    **kwargs
) -> pd.DataFrame:
    """
    Run CATE validation across grid of studies and betas.

    Returns DataFrame with summary results.
    """
    studies = studies or STUDIES[:4]
    betas = betas or BETAS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_list = []
    total = len(studies) * len(betas)
    current = 0

    for study in studies:
        for beta in betas:
            current += 1
            print(f"[{current}/{total}] {study}/beta={beta}")

            try:
                result = run_cate_validation(
                    study=study,
                    pattern=pattern,
                    beta=beta,
                    epsilon=epsilon,
                    **kwargs
                )

                if 'error' not in result:
                    row = {
                        'study': study,
                        'pattern': pattern,
                        'beta': beta,
                        'epsilon': epsilon,
                        'n_bounds_strata': result.get('n_bounds_strata', 0),
                        'n_truth_strata': result.get('n_truth_strata', 0),
                        **result.get('coverage', {})
                    }
                    results_list.append(row)
                else:
                    print(f"  Error: {result['error']}")

            except Exception as e:
                print(f"  Exception: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'cate_validation_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    return results_df


def generate_cate_validation_report(results_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive CATE validation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(results_df) == 0:
        print("No results to report")
        return

    # 1. Summary statistics table (LaTeX)
    summary_stats = {
        'Mean Coverage Rate': results_df['coverage_rate'].mean(),
        'Std Coverage Rate': results_df['coverage_rate'].std(),
        'Min Coverage Rate': results_df['coverage_rate'].min(),
        'Max Coverage Rate': results_df['coverage_rate'].max(),
        'Mean Strata Evaluated': results_df['n_strata_evaluated'].mean(),
        'Total Experiments': len(results_df)
    }

    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{CATE Coverage Validation Summary}',
        r'\label{tab:cate_validation}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Metric} & \textbf{Value} \\',
        r'\midrule'
    ]

    for metric, value in summary_stats.items():
        if isinstance(value, float):
            latex_lines.append(f'{metric} & {value:.3f} \\\\')
        else:
            latex_lines.append(f'{metric} & {value} \\\\')

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])

    tex_path = output_path / 'cate_validation_summary.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"  Saved: {tex_path}")

    # 2. Coverage heatmap by study and beta
    if 'study' in results_df.columns and 'beta' in results_df.columns:
        pivot = results_df.pivot_table(
            values='coverage_rate',
            index='study',
            columns='beta',
            aggfunc='mean'
        )

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                        vmin=0, vmax=1, ax=ax)
            ax.set_xlabel(r'Confounding Strength ($\beta$)')
            ax.set_ylabel('Study')
            ax.set_title('CATE Coverage Rate by Study and Confounding Strength')
            save_figure(fig, output_path / 'cate_coverage_heatmap.png')

    # 3. Coverage distribution histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results_df['coverage_rate'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=results_df['coverage_rate'].mean(), color='red', linestyle='--',
               label=f"Mean: {results_df['coverage_rate'].mean():.2f}")
    ax.axvline(x=0.95, color='green', linestyle='--', label='95% Target')
    ax.set_xlabel('Coverage Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of CATE Coverage Rates')
    ax.legend(loc='best')
    save_figure(fig, output_path / 'cate_coverage_distribution.png')

    # 4. Markdown report
    md_lines = [
        '# CATE Coverage Validation Report',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '\n## Summary Statistics\n',
        '| Metric | Value |',
        '|--------|-------|'
    ]

    for metric, value in summary_stats.items():
        if isinstance(value, float):
            md_lines.append(f'| {metric} | {value:.3f} |')
        else:
            md_lines.append(f'| {metric} | {value} |')

    md_lines.extend([
        '\n## Key Findings\n',
        f'- Average CATE coverage rate: **{summary_stats["Mean Coverage Rate"]:.1%}**',
        f'- Coverage ranges from {summary_stats["Min Coverage Rate"]:.1%} to {summary_stats["Max Coverage Rate"]:.1%}',
        f'- Average of {summary_stats["Mean Strata Evaluated"]:.1f} strata evaluated per experiment',
        '\n## Interpretation\n',
        'Coverage rate indicates the proportion of covariate strata where the true CATE ',
        '(computed from RCT data) falls within our estimated bounds.',
        '\nA coverage rate of ~95% would indicate well-calibrated bounds that are ',
        'neither too narrow (undercoverage) nor too wide (overcoverage).'
    ])

    md_path = output_path / 'cate_validation_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved: {md_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate CATE bounds coverage against ground truth'
    )

    parser.add_argument(
        '--mode', type=str, choices=['single', 'grid'], default='single',
        help='Run mode: single validation or full grid'
    )
    parser.add_argument(
        '--study', type=str, default='anchoring1',
        help='Study name for single mode'
    )
    parser.add_argument(
        '--pattern', type=str, default='age',
        help='Confounding pattern'
    )
    parser.add_argument(
        '--beta', type=float, default=0.1,
        help='Confounding strength'
    )
    parser.add_argument(
        '--epsilon', type=float, default=0.1,
        help='Naturalness tolerance'
    )
    parser.add_argument(
        '--min-samples', type=int, default=20,
        help='Minimum samples per stratum for ground truth'
    )
    parser.add_argument(
        '--output', type=str, default='results/cate_validation',
        help='Output directory'
    )
    parser.add_argument(
        '--target-site', type=str, default='mturk',
        help='Target site'
    )
    parser.add_argument(
        '--data-dir', type=str, default='confounded_datasets',
        help='Data directory'
    )
    parser.add_argument(
        '--rct-path', type=str, default='ManyLabs1/pre-process/Manylabs1_data.pkl',
        help='Path to RCT data'
    )
    parser.add_argument(
        '--n-permutations', type=int, default=100,
        help='Number of permutations for CI tests'
    )

    args = parser.parse_args()

    if args.mode == 'single':
        print("=" * 60)
        print("CATE COVERAGE VALIDATION - SINGLE")
        print("=" * 60)

        result = run_cate_validation(
            study=args.study,
            pattern=args.pattern,
            beta=args.beta,
            epsilon=args.epsilon,
            target_site=args.target_site,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            n_permutations=args.n_permutations,
            min_samples=args.min_samples
        )

        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Print results
        print(f"\nStudy: {args.study}, Pattern: {args.pattern}, Beta: {args.beta}")
        print(f"Bounds strata: {result.get('n_bounds_strata', 'N/A')}")
        print(f"Truth strata: {result.get('n_truth_strata', 'N/A')}")

        coverage = result.get('coverage', {})
        print(f"\nCoverage Results:")
        print(f"  Strata evaluated: {coverage.get('n_strata_evaluated', 'N/A')}")
        print(f"  Strata covered: {coverage.get('n_covered', 'N/A')}")
        print(f"  Coverage rate: {coverage.get('coverage_rate', 'N/A'):.2%}" if coverage.get('coverage_rate') else "  Coverage rate: N/A")

        # Generate plots
        if 'bounds' in result and 'cate_truth' in result:
            bounds_df = pd.DataFrame(result['bounds'])
            cate_truth_df = pd.DataFrame(result['cate_truth'])

            print(f"\nGenerating plots in {args.output}...")
            plot_cate_coverage_details(
                bounds_df, cate_truth_df, args.output,
                study=args.study, beta=args.beta
            )

    elif args.mode == 'grid':
        print("=" * 60)
        print("CATE COVERAGE VALIDATION - GRID")
        print("=" * 60)

        results_df = run_cate_validation_grid(
            pattern=args.pattern,
            epsilon=args.epsilon,
            output_dir=args.output,
            target_site=args.target_site,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            n_permutations=args.n_permutations,
            min_samples=args.min_samples
        )

        if len(results_df) > 0:
            print("\nGenerating report...")
            generate_cate_validation_report(results_df, args.output)
            print("\nDone!")
        else:
            print("No valid results to report.")


if __name__ == '__main__':
    main()
