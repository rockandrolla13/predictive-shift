"""
Compare causal grounding bounds to baseline causal inference methods.

Evaluates:
1. Coverage: Does the true ATE fall within bounds?
2. Width: How informative are the bounds?
3. Bias: How do baseline point estimates compare?
4. Which methods fail under confounding?

Usage:
    python experiments/compare_to_baselines.py --study anchoring1 --beta 0.3
    python experiments/compare_to_baselines.py --mode grid --output comparison_results/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CausalGroundingEstimator,
    create_train_target_split,
    load_rct_data
)

# Try to import baseline methods
try:
    from causal_methods import (
        estimate_naive,
        estimate_ipw,
        estimate_or,
        estimate_aipw,
        estimate_psm
    )
    BASELINE_METHODS = {
        'naive': estimate_naive,
        'ipw': estimate_ipw,
        'or': estimate_or,
        'aipw': estimate_aipw,
        'psm': estimate_psm
    }
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINE_METHODS = {}
    BASELINES_AVAILABLE = False
    print("Warning: causal_methods not available. Baseline comparisons disabled.")


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

PATTERNS = ['age', 'gender', 'polideo']

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


def load_ground_truth_ate(study, gt_path='confounded_datasets/ground_truth_ates.csv'):
    """Load ground truth ATE for a study."""
    try:
        gt_df = pd.read_csv(gt_path)
        for col in ['study', 'effect', 'name']:
            if col in gt_df.columns:
                row = gt_df[gt_df[col] == study]
                if len(row) > 0:
                    for ate_col in ['ate', 'true_ate', 'ATE', 'effect_size']:
                        if ate_col in gt_df.columns:
                            return row[ate_col].values[0]
        return np.nan
    except Exception:
        return np.nan


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
# COMPARISON FUNCTIONS
# =============================================================================

def run_baseline_comparison(
    study: str,
    pattern: str,
    beta: float,
    epsilon: float = 0.1,
    target_site: str = 'mturk',
    data_dir: str = 'confounded_datasets',
    rct_path: str = 'ManyLabs1/pre-process/Manylabs1_data.pkl',
    n_permutations: int = 100
) -> dict:
    """
    Run grounding and all baselines on one dataset.

    Returns dict with grounding bounds, baseline estimates, and metrics.
    """
    results = {
        'config': {
            'study': study,
            'pattern': pattern,
            'beta': beta,
            'epsilon': epsilon,
            'target_site': target_site
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

    # Load ground truth ATE
    true_ate = load_ground_truth_ate(study)
    results['true_ate'] = true_ate

    # Run CausalGroundingEstimator
    try:
        estimator = CausalGroundingEstimator(
            epsilon=epsilon,
            n_permutations=n_permutations,
            random_seed=42,
            verbose=False
        )
        estimator.fit(training_data, treatment='iv', outcome='dv')
        bounds_df = estimator.predict_bounds()

        mean_lower = bounds_df['cate_lower'].mean()
        mean_upper = bounds_df['cate_upper'].mean()
        width = mean_upper - mean_lower
        midpoint = (mean_lower + mean_upper) / 2

        ate_covered = False
        if not np.isnan(true_ate):
            ate_covered = mean_lower <= true_ate <= mean_upper

        results['grounding'] = {
            'mean_lower': mean_lower,
            'mean_upper': mean_upper,
            'width': width,
            'midpoint': midpoint,
            'ate_covered': ate_covered,
            'n_strata': len(bounds_df)
        }
    except Exception as e:
        results['grounding'] = {'error': str(e)}

    # Run baseline methods on target data (F='idle' only, which is confounded)
    results['baselines'] = {}

    if BASELINES_AVAILABLE:
        # Use only idle data (confounded observational data)
        if 'F' in target_data.columns:
            baseline_data = target_data[target_data['F'] == 'idle'].copy()
        else:
            baseline_data = target_data.copy()

        # Get covariates
        potential_covs = ['resp_age', 'resp_gender', 'resp_polideo']
        covariates = [c for c in potential_covs if c in baseline_data.columns]

        for method_name, method_func in BASELINE_METHODS.items():
            try:
                if method_name == 'naive':
                    estimate = method_func(baseline_data, 'iv', 'dv')
                else:
                    estimate = method_func(baseline_data, 'iv', 'dv', covariates)

                # Extract estimate value
                if isinstance(estimate, dict):
                    est_val = estimate.get('estimate', estimate.get('ate', np.nan))
                else:
                    est_val = float(estimate)

                bias = est_val - true_ate if not np.isnan(true_ate) else np.nan
                abs_bias = abs(bias) if not np.isnan(bias) else np.nan

                # Check if within grounding bounds
                within_bounds = False
                if 'grounding' in results and 'error' not in results['grounding']:
                    within_bounds = (results['grounding']['mean_lower'] <= est_val <=
                                     results['grounding']['mean_upper'])

                results['baselines'][method_name] = {
                    'estimate': est_val,
                    'bias': bias,
                    'abs_bias': abs_bias,
                    'within_bounds': within_bounds
                }
            except Exception as e:
                results['baselines'][method_name] = {'error': str(e)}

    return results


def compute_comparison_metrics(results: dict) -> dict:
    """
    Compute comparison metrics from run_baseline_comparison results.

    Returns flattened dict suitable for DataFrame row.
    """
    metrics = {
        'study': results['config']['study'],
        'pattern': results['config']['pattern'],
        'beta': results['config']['beta'],
        'epsilon': results['config']['epsilon'],
        'true_ate': results.get('true_ate', np.nan)
    }

    # Grounding metrics
    grounding = results.get('grounding', {})
    if 'error' not in grounding:
        metrics['grounding_lower'] = grounding.get('mean_lower', np.nan)
        metrics['grounding_upper'] = grounding.get('mean_upper', np.nan)
        metrics['grounding_width'] = grounding.get('width', np.nan)
        metrics['grounding_midpoint'] = grounding.get('midpoint', np.nan)
        metrics['grounding_coverage'] = grounding.get('ate_covered', False)

        # Grounding midpoint bias
        if not np.isnan(results.get('true_ate', np.nan)):
            metrics['grounding_midpoint_bias'] = grounding.get('midpoint', np.nan) - results['true_ate']
            metrics['grounding_midpoint_abs_bias'] = abs(metrics['grounding_midpoint_bias'])
        else:
            metrics['grounding_midpoint_bias'] = np.nan
            metrics['grounding_midpoint_abs_bias'] = np.nan
    else:
        metrics['grounding_error'] = grounding.get('error')

    # Baseline metrics
    baselines = results.get('baselines', {})
    best_abs_bias = np.inf
    best_method = None

    for method_name, baseline_result in baselines.items():
        if 'error' not in baseline_result:
            metrics[f'{method_name}_estimate'] = baseline_result.get('estimate', np.nan)
            metrics[f'{method_name}_bias'] = baseline_result.get('bias', np.nan)
            metrics[f'{method_name}_abs_bias'] = baseline_result.get('abs_bias', np.nan)
            metrics[f'{method_name}_within_bounds'] = baseline_result.get('within_bounds', False)

            abs_bias = baseline_result.get('abs_bias', np.inf)
            if not np.isnan(abs_bias) and abs_bias < best_abs_bias:
                best_abs_bias = abs_bias
                best_method = method_name
        else:
            metrics[f'{method_name}_error'] = baseline_result.get('error')

    metrics['best_baseline'] = best_method
    metrics['best_baseline_abs_bias'] = best_abs_bias if best_abs_bias != np.inf else np.nan

    # Does grounding beat best baseline?
    grounding_beats = False
    if metrics.get('grounding_coverage', False) and best_method is not None:
        # Grounding wins if it covers and has width < 2 * best baseline bias
        width = metrics.get('grounding_width', np.inf)
        if width < 2 * best_abs_bias:
            grounding_beats = True
    metrics['grounding_beats_best'] = grounding_beats

    return metrics


def run_comparison_grid(
    studies: list = None,
    patterns: list = None,
    betas: list = None,
    epsilon: float = 0.1,
    output_dir: str = 'comparison_results',
    **kwargs
) -> pd.DataFrame:
    """
    Run comparison across grid of studies, patterns, betas.

    Returns DataFrame with all comparison metrics.
    """
    studies = studies or STUDIES[:4]
    patterns = patterns or ['age']
    betas = betas or BETAS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_list = []
    total = len(studies) * len(patterns) * len(betas)
    current = 0

    for study in studies:
        for pattern in patterns:
            for beta in betas:
                current += 1
                print(f"[{current}/{total}] {study}/{pattern}/beta={beta}")

                try:
                    result = run_baseline_comparison(
                        study=study,
                        pattern=pattern,
                        beta=beta,
                        epsilon=epsilon,
                        **kwargs
                    )

                    if 'error' not in result:
                        metrics = compute_comparison_metrics(result)
                        results_list.append(metrics)
                    else:
                        print(f"  Error: {result['error']}")

                except Exception as e:
                    print(f"  Exception: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'comparison_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    return results_df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_bias_comparison(comparison_df: pd.DataFrame, output_dir: str):
    """Create bias comparison plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Boxplot of absolute bias by method
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect bias data for each method
    methods = ['naive', 'ipw', 'or', 'aipw', 'psm', 'grounding_midpoint']
    bias_data = []
    method_labels = []

    for method in methods:
        col = f'{method}_abs_bias' if method != 'grounding_midpoint' else 'grounding_midpoint_abs_bias'
        if col in comparison_df.columns:
            data = comparison_df[col].dropna().tolist()
            if data:
                bias_data.append(data)
                method_labels.append(method.upper() if method != 'grounding_midpoint' else 'Grounding\nMidpoint')

    if bias_data:
        bp = ax.boxplot(bias_data, patch_artist=True, widths=0.6)
        colors = plt.cm.Set2(np.linspace(0, 1, len(bias_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(method_labels)
        ax.set_xlabel('Method')
        ax.set_ylabel('Absolute Bias')
        ax.set_title('Absolute Bias Comparison Across Methods')

        save_figure(fig, output_path / 'bias_comparison_boxplot.png')

    # Plot 2: Heatmap of success rate by beta
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute success rates by beta
    success_cols = {
        'Grounding': 'grounding_coverage',
        'NAIVE': 'naive_within_bounds',
        'IPW': 'ipw_within_bounds',
        'OR': 'or_within_bounds',
        'AIPW': 'aipw_within_bounds',
        'PSM': 'psm_within_bounds'
    }

    if 'beta' in comparison_df.columns:
        betas = sorted(comparison_df['beta'].unique())
        success_matrix = []
        method_names = []

        for method_name, col in success_cols.items():
            if col in comparison_df.columns:
                rates = [comparison_df[comparison_df['beta'] == b][col].mean() for b in betas]
                success_matrix.append(rates)
                method_names.append(method_name)

        if success_matrix:
            success_df = pd.DataFrame(
                success_matrix,
                index=method_names,
                columns=[f'β={b}' for b in betas]
            )

            sns.heatmap(success_df, annot=True, fmt='.2f', cmap='RdYlGn',
                        vmin=0, vmax=1, ax=ax)
            ax.set_title('Success Rate by Method and Confounding Strength')

            save_figure(fig, output_path / 'method_success_heatmap.png')

    # Plot 3: Scatter of grounding width vs best baseline bias
    if 'grounding_width' in comparison_df.columns and 'best_baseline_abs_bias' in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))

        valid = comparison_df.dropna(subset=['grounding_width', 'best_baseline_abs_bias'])

        if len(valid) > 0:
            colors = valid['grounding_coverage'].map({True: 'green', False: 'red'})
            ax.scatter(valid['grounding_width'], valid['best_baseline_abs_bias'],
                       c=colors, alpha=0.7, s=50)

            ax.set_xlabel('Grounding Bound Width')
            ax.set_ylabel('Best Baseline Absolute Bias')
            ax.set_title('Grounding Width vs Best Baseline Bias')

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='True ATE Covered'),
                Patch(facecolor='red', label='True ATE Not Covered')
            ]
            ax.legend(handles=legend_elements, loc='best')

            save_figure(fig, output_path / 'width_vs_bias_scatter.png')


def plot_method_ranking(comparison_df: pd.DataFrame, output_dir: str):
    """Plot method ranking across conditions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute mean absolute bias by method
    methods = ['naive', 'ipw', 'or', 'aipw', 'psm', 'grounding_midpoint']
    mean_biases = []
    method_labels = []

    for method in methods:
        col = f'{method}_abs_bias' if method != 'grounding_midpoint' else 'grounding_midpoint_abs_bias'
        if col in comparison_df.columns:
            mean_bias = comparison_df[col].mean()
            if not np.isnan(mean_bias):
                mean_biases.append(mean_bias)
                method_labels.append(method.upper() if method != 'grounding_midpoint' else 'GROUNDING')

    if mean_biases:
        # Sort by bias
        sorted_indices = np.argsort(mean_biases)
        sorted_biases = [mean_biases[i] for i in sorted_indices]
        sorted_labels = [method_labels[i] for i in sorted_indices]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_biases)))
        bars = ax.barh(range(len(sorted_biases)), sorted_biases, color=colors)

        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel('Mean Absolute Bias')
        ax.set_title('Method Ranking by Mean Absolute Bias (Lower is Better)')

        # Add value labels
        for bar, bias in zip(bars, sorted_biases):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{bias:.2f}', va='center', fontsize=9)

        save_figure(fig, output_path / 'method_ranking.png')


def generate_comparison_table(comparison_df: pd.DataFrame, output_dir: str):
    """Generate LaTeX comparison table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics for each method
    methods = ['naive', 'ipw', 'or', 'aipw', 'psm', 'grounding_midpoint']
    summary_rows = []

    for method in methods:
        bias_col = f'{method}_abs_bias' if method != 'grounding_midpoint' else 'grounding_midpoint_abs_bias'
        est_col = f'{method}_estimate' if method != 'grounding_midpoint' else 'grounding_midpoint'

        if bias_col in comparison_df.columns:
            abs_bias_mean = comparison_df[bias_col].mean()
            abs_bias_std = comparison_df[bias_col].std()

            # Coverage/success rate
            if method == 'grounding_midpoint':
                success_col = 'grounding_coverage'
            else:
                success_col = f'{method}_within_bounds'

            if success_col in comparison_df.columns:
                success_rate = comparison_df[success_col].mean()
            else:
                success_rate = np.nan

            # Estimate std (stability)
            if est_col in comparison_df.columns:
                est_std = comparison_df[est_col].std()
            else:
                est_std = np.nan

            summary_rows.append({
                'Method': method.upper() if method != 'grounding_midpoint' else 'GROUNDING',
                'Mean Abs Bias': abs_bias_mean,
                'Std Abs Bias': abs_bias_std,
                'Success Rate': success_rate,
                'Estimate Std': est_std
            })

    summary_df = pd.DataFrame(summary_rows)

    # Generate LaTeX
    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Comparison of Causal Inference Methods}',
        r'\label{tab:method_comparison}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Method} & \textbf{Mean Abs Bias} & \textbf{Std Bias} & \textbf{Success Rate} & \textbf{Est. Std} \\',
        r'\midrule'
    ]

    for _, row in summary_df.iterrows():
        cells = [
            row['Method'],
            f"{row['Mean Abs Bias']:.3f}" if not np.isnan(row['Mean Abs Bias']) else '--',
            f"{row['Std Abs Bias']:.3f}" if not np.isnan(row['Std Abs Bias']) else '--',
            f"{row['Success Rate']:.2f}" if not np.isnan(row['Success Rate']) else '--',
            f"{row['Estimate Std']:.3f}" if not np.isnan(row['Estimate Std']) else '--'
        ]
        latex_lines.append(' & '.join(cells) + r' \\')

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])

    latex_str = '\n'.join(latex_lines)

    # Write to file
    tex_path = output_path / 'comparison_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_str)
    print(f"  Saved: {tex_path}")

    return latex_str


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare causal grounding to baseline methods'
    )

    parser.add_argument(
        '--mode', type=str, choices=['single', 'grid'], default='single',
        help='Run mode: single comparison or full grid'
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
        '--studies', type=str, default=None,
        help='Comma-separated list of studies for grid mode'
    )
    parser.add_argument(
        '--output', type=str, default='comparison_results',
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
        print("BASELINE COMPARISON - SINGLE")
        print("=" * 60)

        result = run_baseline_comparison(
            study=args.study,
            pattern=args.pattern,
            beta=args.beta,
            epsilon=args.epsilon,
            target_site=args.target_site,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            n_permutations=args.n_permutations
        )

        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Print results
        print(f"\nStudy: {args.study}, Pattern: {args.pattern}, Beta: {args.beta}")
        print(f"True ATE: {result.get('true_ate', 'N/A')}")

        print("\nGrounding Results:")
        grounding = result.get('grounding', {})
        if 'error' not in grounding:
            print(f"  Bounds: [{grounding.get('mean_lower', 'N/A'):.3f}, {grounding.get('mean_upper', 'N/A'):.3f}]")
            print(f"  Width: {grounding.get('width', 'N/A'):.3f}")
            print(f"  Midpoint: {grounding.get('midpoint', 'N/A'):.3f}")
            print(f"  ATE Covered: {grounding.get('ate_covered', 'N/A')}")
        else:
            print(f"  Error: {grounding.get('error')}")

        print("\nBaseline Results:")
        for method, baseline in result.get('baselines', {}).items():
            if 'error' not in baseline:
                print(f"  {method.upper()}:")
                print(f"    Estimate: {baseline.get('estimate', 'N/A'):.3f}")
                print(f"    Bias: {baseline.get('bias', 'N/A'):.3f}")
                print(f"    Within Bounds: {baseline.get('within_bounds', 'N/A')}")
            else:
                print(f"  {method.upper()}: Error - {baseline.get('error')}")

        # Compute metrics
        metrics = compute_comparison_metrics(result)
        print(f"\nBest Baseline: {metrics.get('best_baseline', 'N/A')}")
        print(f"Grounding Beats Best: {metrics.get('grounding_beats_best', 'N/A')}")

    elif args.mode == 'grid':
        print("=" * 60)
        print("BASELINE COMPARISON - GRID")
        print("=" * 60)

        studies = args.studies.split(',') if args.studies else None

        comparison_df = run_comparison_grid(
            studies=studies,
            patterns=[args.pattern],
            betas=BETAS,
            epsilon=args.epsilon,
            output_dir=args.output,
            target_site=args.target_site,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            n_permutations=args.n_permutations
        )

        if len(comparison_df) > 0:
            print("\nGenerating plots...")
            plot_bias_comparison(comparison_df, args.output)
            plot_method_ranking(comparison_df, args.output)
            generate_comparison_table(comparison_df, args.output)
            print("\nDone!")
        else:
            print("No valid results to plot.")


if __name__ == '__main__':
    main()
