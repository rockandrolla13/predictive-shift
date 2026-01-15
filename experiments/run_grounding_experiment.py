"""
Enhanced Causal Grounding Experiment Script

Runs experiments on OSRCT benchmark data and generates publication-quality outputs.

Output formats:
- Individual PNG files (300 DPI) for each figure
- LaTeX table files (.tex) for each table
- JSON results for programmatic access

Usage:
    python experiments/run_grounding_experiment.py --study anchoring1 --beta 0.3
    python experiments/run_grounding_experiment.py --mode full-grid
    python experiments/run_grounding_experiment.py --mode figures --results-dir results/
    python experiments/run_grounding_experiment.py --mode epsilon-sensitivity --study anchoring1
    python experiments/run_grounding_experiment.py --grid --binary-only  # Only binary outcome studies
    python experiments/run_grounding_experiment.py --all --binary-only --report  # Run + generate report
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import argparse
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns

# Add parent directory to path for importing causal_grounding
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CausalGroundingEstimator,
    create_train_target_split,
    load_rct_data,
    discretize_covariates
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

FIGURE_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

plt.rcParams.update(FIGURE_CONFIG)


# =============================================================================
# EXPERIMENT CONSTANTS
# =============================================================================

STUDIES = [
    'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4',
    'gamblerfallacy', 'sunkfallacy', 'gainloss', 'quote',
    'allowforbid', 'reciprocity', 'scaleframe', 'contact',
    'imaginedcontact', 'flagprime', 'moneypriming'
]

# Studies with binary (0/1) outcomes
# Based on ManyLabs1 data analysis:
#   - allowedforbidden -> allowforbid
#   - gainloss (values 1/2, represents binary choice)
#   - reciprocity (strict 0/1)
#   - scales -> scaleframe (strict 0/1)
BINARY_OUTCOME_STUDIES = [
    'allowforbid',
    'gainloss',
    'reciprocity',
    'scaleframe'
]

PATTERNS = [
    'age',
    'gender',
    'polideo',
    'demo_basic',
    'demo_full'
]

BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

EPSILONS = [0.05, 0.1, 0.2, 0.3, 0.5]

BASELINE_METHODS = [
    'naive',
    'ipw',
    'or',
    'aipw',
    'psm',
    'causal_forest'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig, output_path, close=True):
    """
    Save a matplotlib figure with consistent settings.

    Args:
        fig: matplotlib Figure object
        output_path: Path to save the figure
        close: Whether to close the figure after saving
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        format='png',
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )

    print(f"  Saved: {output_path}")

    if close:
        plt.close(fig)


def load_ground_truth_ate(study: str, gt_path: str = 'ground_truth/rct_ates.csv') -> float:
    """
    Load ground truth ATE for a study from CSV file.

    Args:
        study: Study name (e.g., 'anchoring1')
        gt_path: Path to ground truth CSV file

    Returns:
        ATE value or np.nan if not found
    """
    try:
        gt_df = pd.read_csv(gt_path)

        # Try different possible column names for study identifier
        study_col = None
        for col in ['study', 'effect', 'name']:
            if col in gt_df.columns:
                study_col = col
                break

        if study_col is None:
            return np.nan

        # Find matching row
        row = gt_df[gt_df[study_col] == study]

        if len(row) == 0:
            return np.nan

        # Try different possible column names for ATE
        for col in ['ate', 'true_ate', 'ATE', 'effect_size']:
            if col in gt_df.columns:
                return row[col].values[0]

        return np.nan

    except Exception as e:
        print(f"Warning: Could not load ground truth for {study}: {e}")
        return np.nan


def compute_cate_ground_truth(
    rct_data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    min_samples: int = 20
) -> Dict[Tuple, float]:
    """
    Compute ground truth CATE from RCT data by stratifying on covariates.

    Args:
        rct_data: RCT DataFrame
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of covariate names to stratify by
        min_samples: Minimum samples per arm per stratum

    Returns:
        Dict mapping z-tuple to CATE value
    """
    # Discretize covariates
    data = discretize_covariates(rct_data.copy())

    # Build list of discretized covariate column names
    discretized_cols = []
    for cov in covariates:
        # Check for _cat suffix version first
        cat_col = f"{cov}_cat"
        if cat_col in data.columns:
            discretized_cols.append(cat_col)
        elif cov in data.columns:
            discretized_cols.append(cov)

    # Filter to columns that exist
    discretized_cols = [c for c in discretized_cols if c in data.columns]

    if len(discretized_cols) == 0:
        return {}

    # Group by discretized covariates
    cate_dict = {}

    try:
        grouped = data.groupby(discretized_cols, observed=True)

        for z_values, group in grouped:
            # Ensure z_values is a tuple
            if not isinstance(z_values, tuple):
                z_values = (z_values,)

            # Check treatment arm sample sizes
            treated = group[group[treatment] == 1]
            control = group[group[treatment] == 0]

            if len(treated) >= min_samples and len(control) >= min_samples:
                # Compute CATE as difference in means
                y1_mean = treated[outcome].mean()
                y0_mean = control[outcome].mean()
                cate = y1_mean - y0_mean

                cate_dict[tuple(z_values)] = cate

    except Exception as e:
        print(f"Warning: Error computing CATE ground truth: {e}")

    return cate_dict


def run_baseline_methods(
    data: pd.DataFrame,
    treatment: str,
    outcome: str
) -> Dict[str, Dict[str, Any]]:
    """
    Run baseline causal inference methods for comparison.

    Args:
        data: DataFrame with treatment, outcome, and covariates
        treatment: Treatment column name
        outcome: Outcome column name

    Returns:
        Dict mapping method name to results dict with 'estimate', 'se', etc.
    """
    results = {}

    try:
        from causal_methods import (
            estimate_naive,
            estimate_ipw,
            estimate_or,
            estimate_aipw,
            estimate_psm
        )

        # Define covariates to use
        potential_covariates = [
            'resp_age', 'resp_gender', 'resp_polideo',
            'resp_age_cat', 'resp_polideo_cat'
        ]

        # Filter to covariates present in data
        covariates = [c for c in potential_covariates if c in data.columns]

        if len(covariates) == 0:
            print("Warning: No covariates found for baseline methods")
            return {}

        # Run each method
        try:
            results['naive'] = estimate_naive(data, treatment, outcome)
        except Exception as e:
            print(f"Warning: Naive estimation failed: {e}")

        try:
            results['ipw'] = estimate_ipw(data, treatment, outcome, covariates)
        except Exception as e:
            print(f"Warning: IPW estimation failed: {e}")

        try:
            results['or'] = estimate_or(data, treatment, outcome, covariates)
        except Exception as e:
            print(f"Warning: OR estimation failed: {e}")

        try:
            results['aipw'] = estimate_aipw(data, treatment, outcome, covariates)
        except Exception as e:
            print(f"Warning: AIPW estimation failed: {e}")

        try:
            results['psm'] = estimate_psm(data, treatment, outcome, covariates)
        except Exception as e:
            print(f"Warning: PSM estimation failed: {e}")

    except ImportError:
        # causal_methods module not available
        pass

    return results


# =============================================================================
# EXPERIMENT RUNNER FUNCTIONS
# =============================================================================

def find_dataset_file(
    study: str,
    pattern: str,
    beta: float,
    data_dir: str = 'confounded_datasets'
) -> Optional[Path]:
    """
    Find the dataset file for a given study/pattern/beta combination.

    Args:
        study: Study name
        pattern: Confounding pattern
        beta: Confounding strength
        data_dir: Base directory for datasets

    Returns:
        Path to dataset file or None if not found
    """
    data_path = Path(data_dir)

    # Try different filename patterns (pkl first, then csv)
    patterns_to_try = [
        f"{study}_{pattern}_beta_{beta}.pkl",
        f"{pattern}_beta{beta}_seed42.pkl",
        f"{pattern}_beta{beta:.1f}_seed42.pkl",
        f"{pattern}_beta{beta:.2f}_seed42.pkl",
        f"{study}_{pattern}_beta_{beta}.csv",
        f"{pattern}_beta{beta}_seed42.csv",
        f"{pattern}_beta{beta:.1f}_seed42.csv",
        f"{pattern}_beta{beta:.2f}_seed42.csv",
    ]

    for filename in patterns_to_try:
        filepath = data_path / study / filename
        if filepath.exists():
            return filepath

    return None


def run_single_experiment(
    study: str,
    pattern: str,
    beta: float,
    epsilon: float,
    target_site: str = 'mturk',
    n_permutations: int = 500,
    data_dir: str = 'confounded_datasets',
    rct_path: str = 'ManyLabs1/pre-process/Manylabs1_data.pkl',
    random_seed: int = 42,
    compute_baselines: bool = False
) -> Dict[str, Any]:
    """
    Run one complete grounding experiment and return comprehensive results.

    Args:
        study: Study name (e.g., 'anchoring1')
        pattern: Confounding pattern (e.g., 'age', 'gender', 'polideo')
        beta: Confounding strength (0.0 to 1.0)
        epsilon: Naturalness tolerance for LP constraints
        target_site: Target site to hold out (default: 'mturk')
        n_permutations: Number of permutations for CI tests
        data_dir: Base directory for OSRCT datasets
        rct_path: Path to RCT data pickle file
        random_seed: Random seed for reproducibility
        compute_baselines: Whether to run baseline methods on target data

    Returns:
        Dict with keys:
            - 'config': dict of input parameters
            - 'bounds': list of bound records
            - 'metrics': dict with true_ate, mean_lower, mean_upper, etc.
            - 'covariate_scores': DataFrame as dict (if available)
            - 'best_instrument': best instrumental covariate
            - 'baselines': baseline method results (if computed)
            - 'true_cates': dict of ground truth CATEs with string keys
            - 'error': error message (only if experiment failed)
    """
    # Store configuration
    config = {
        'study': study,
        'pattern': pattern,
        'beta': beta,
        'epsilon': epsilon,
        'target_site': target_site,
        'n_permutations': n_permutations,
        'data_dir': data_dir,
        'rct_path': rct_path,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }

    # Construct OSRCT data path
    osrct_path = Path(data_dir) / study / f"{study}_{pattern}_beta_{beta}.pkl"

    # Check if path exists, try alternative patterns
    if not osrct_path.exists():
        osrct_path = find_dataset_file(study, pattern, beta, data_dir)

    if osrct_path is None or not osrct_path.exists():
        return {
            'config': config,
            'error': f"Dataset not found for {study}/{pattern}/beta={beta}"
        }

    try:
        # Load OSRCT data
        if str(osrct_path).endswith('.pkl'):
            osrct_data = pd.read_pickle(osrct_path)
        else:
            osrct_data = pd.read_csv(osrct_path)

        # Load RCT data
        rct_data = load_rct_data(study, rct_path)

        # Create train/target split
        training_data, target_data = create_train_target_split(
            osrct_data, rct_data, target_site=target_site
        )

        # Initialize estimator
        estimator = CausalGroundingEstimator(
            epsilon=epsilon,
            transfer_method='conservative',
            n_permutations=n_permutations,
            random_seed=random_seed,
            verbose=False
        )

        # Fit estimator
        estimator.fit(training_data, treatment='iv', outcome='dv')

        # Get bounds DataFrame
        bounds_df = estimator.predict_bounds()

        # Load ground truth ATE
        true_ate = load_ground_truth_ate(study, f'{data_dir}/ground_truth_ates.csv')

        # Compute ground truth CATEs
        true_cates = compute_cate_ground_truth(
            rct_data,
            treatment='iv',
            outcome='dv',
            covariates=['resp_age', 'resp_gender', 'resp_polideo']
        )

        # Calculate metrics
        mean_lower = bounds_df['cate_lower'].mean()
        mean_upper = bounds_df['cate_upper'].mean()
        mean_width = bounds_df['width'].mean()
        median_width = bounds_df['width'].median()

        # Check if true ATE falls within bounds
        ate_covered = False
        if not np.isnan(true_ate):
            ate_covered = mean_lower <= true_ate <= mean_upper

        # Compute CATE coverage
        cate_coverage = []
        for z_tuple, true_cate in true_cates.items():
            # Find matching bounds row
            for _, row in bounds_df.iterrows():
                # Extract z values from row
                row_z = row.get('z', None)
                if row_z is not None and tuple(row_z) == z_tuple:
                    if row['cate_lower'] <= true_cate <= row['cate_upper']:
                        cate_coverage.append(1.0)
                    else:
                        cate_coverage.append(0.0)
                    break

        cate_coverage_rate = np.mean(cate_coverage) if len(cate_coverage) > 0 else np.nan

        # Build metrics dict
        metrics = {
            'true_ate': true_ate,
            'mean_lower': mean_lower,
            'mean_upper': mean_upper,
            'mean_width': mean_width,
            'median_width': median_width,
            'ate_covered': ate_covered,
            'cate_coverage_rate': cate_coverage_rate,
            'n_strata': len(bounds_df),
            'n_training_sites': len(training_data)
        }

        # Get covariate scores if available
        covariate_scores = None
        if hasattr(estimator, 'covariate_scores_') and estimator.covariate_scores_ is not None:
            if isinstance(estimator.covariate_scores_, pd.DataFrame):
                covariate_scores = estimator.covariate_scores_.to_dict()
            else:
                covariate_scores = estimator.covariate_scores_

        # Get best instrument
        best_instrument = getattr(estimator, 'best_instrument_', None)

        # Run baseline methods if requested
        baselines = {}
        if compute_baselines:
            baselines = run_baseline_methods(target_data, 'iv', 'dv')

        # Convert true_cates keys to strings for JSON serialization
        true_cates_str = {str(k): v for k, v in true_cates.items()}

        return {
            'config': config,
            'bounds': bounds_df.to_dict('records'),
            'metrics': metrics,
            'covariate_scores': covariate_scores,
            'best_instrument': best_instrument,
            'baselines': baselines,
            'true_cates': true_cates_str
        }

    except Exception as e:
        return {
            'config': config,
            'error': str(e)
        }


def run_full_grid(
    studies: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    betas: Optional[List[float]] = None,
    epsilon: float = 0.1,
    output_dir: str = 'results',
    **kwargs
) -> pd.DataFrame:
    """
    Run experiments across a grid of studies, patterns, and betas.

    Args:
        studies: List of studies (default: first 5 from STUDIES)
        patterns: List of confounding patterns (default: ['age'])
        betas: List of beta values (default: BETAS)
        epsilon: Naturalness tolerance
        output_dir: Directory for output files
        **kwargs: Additional arguments passed to run_single_experiment
                  (target_site, n_permutations, data_dir, rct_path, random_seed)

    Returns:
        DataFrame with experiment results
    """
    # Set defaults
    if studies is None:
        studies = STUDIES[:5]
    if patterns is None:
        patterns = ['age']
    if betas is None:
        betas = BETAS

    # Initialize results list
    results = []

    # Calculate total experiments
    total = len(studies) * len(patterns) * len(betas)
    current = 0

    # Triple nested loop
    for study in studies:
        for pattern in patterns:
            for beta in betas:
                current += 1
                print(f"[{current}/{total}] Running {study}/{pattern}/beta={beta}")

                try:
                    # Run single experiment with kwargs
                    result = run_single_experiment(
                        study=study,
                        pattern=pattern,
                        beta=beta,
                        epsilon=epsilon,
                        **kwargs
                    )

                    # If no error, extract metrics and add to results
                    if 'error' not in result:
                        row = {
                            'study': study,
                            'pattern': pattern,
                            'beta': beta,
                            'epsilon': epsilon,
                            **result['metrics']
                        }
                        results.append(row)
                    else:
                        print(f"  Error: {result['error']}")

                except Exception as e:
                    print(f"  Exception: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save to CSV
    csv_path = output_path / f'grid_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    return results_df


def run_experiment_sweep(
    studies: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    betas: Optional[List[float]] = None,
    epsilons: Optional[List[float]] = None,
    output_dir: str = 'results',
    verbose: bool = True,
    return_full_results: bool = False
) -> pd.DataFrame:
    """
    Run experiments across multiple configurations (legacy interface).

    Args:
        studies: List of studies (default: all)
        patterns: List of patterns (default: all)
        betas: List of beta values (default: all)
        epsilons: List of epsilon values (default: [0.1])
        output_dir: Directory for output files
        verbose: Print progress
        return_full_results: If True, return tuple of (DataFrame, list of full result dicts)

    Returns:
        DataFrame with all results (or tuple if return_full_results=True)
    """
    studies = studies or STUDIES
    patterns = patterns or PATTERNS
    betas = betas or BETAS
    epsilons = epsilons or [0.1]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    full_results = []
    total = len(studies) * len(patterns) * len(betas) * len(epsilons)
    current = 0

    for study in studies:
        for pattern in patterns:
            for beta in betas:
                for epsilon in epsilons:
                    current += 1
                    if verbose:
                        print(f"\n[{current}/{total}] {study}/{pattern}/beta={beta}/eps={epsilon}")

                    result = run_single_experiment(
                        study=study,
                        pattern=pattern,
                        beta=beta,
                        epsilon=epsilon
                    )

                    full_results.append(result)

                    # Flatten result for DataFrame
                    flat_result = {
                        'study': study,
                        'pattern': pattern,
                        'beta': beta,
                        'epsilon': epsilon,
                        'success': 'error' not in result
                    }
                    if 'metrics' in result:
                        flat_result.update(result['metrics'])
                    if 'error' in result:
                        flat_result['error'] = result['error']

                    all_results.append(flat_result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'experiment_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Save detailed JSON (includes bounds and true_cates)
    json_path = output_path / f'experiment_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Saved detailed results to {json_path}")

    if return_full_results:
        return results_df, full_results
    return results_df


def run_epsilon_sensitivity(
    study: str,
    pattern: str,
    beta: float,
    epsilons: Optional[List[float]] = None,
    **kwargs
) -> Dict[float, Dict]:
    """
    Run experiments varying epsilon to analyze sensitivity.

    Args:
        study: Study name (e.g., 'anchoring1')
        pattern: Confounding pattern (e.g., 'age')
        beta: Confounding strength
        epsilons: List of epsilon values to test (default: EPSILONS)
        **kwargs: Additional arguments passed to run_single_experiment

    Returns:
        Dict mapping epsilon -> result dict
    """
    if epsilons is None:
        epsilons = EPSILONS

    results_by_epsilon = {}

    for eps in epsilons:
        print(f"  Running ε={eps}...")
        result = run_single_experiment(
            study=study,
            pattern=pattern,
            beta=beta,
            epsilon=eps,
            **kwargs
        )
        results_by_epsilon[eps] = result

    return results_by_epsilon


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: str,
    float_format: str = '%.3f',
    column_format: Optional[str] = None,
    bold_max: bool = False,
    bold_min: bool = False
) -> str:
    """
    Generate a standalone LaTeX table file with consistent formatting.

    Args:
        df: DataFrame to convert to LaTeX table
        caption: Table caption
        label: LaTeX label for referencing
        output_path: Path to save the .tex file
        float_format: Format string for floating point numbers
        column_format: LaTeX column format (e.g., 'lccc'). Auto-generated if None.
        bold_max: Bold the maximum value in each numeric column
        bold_min: Bold the minimum value in each numeric column

    Returns:
        LaTeX table as string
    """
    n_cols = len(df.columns)

    # Auto-generate column format if not provided
    if column_format is None:
        column_format = 'l' + 'c' * (n_cols - 1)

    # Build LaTeX string
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{{column_format}}}',
        r'\toprule'
    ]

    # Header row
    header_cells = [f'\\textbf{{{col}}}' for col in df.columns]
    lines.append(' & '.join(header_cells) + r' \\')
    lines.append(r'\midrule')

    # Find max/min for each numeric column if needed
    col_max = {}
    col_min = {}
    if bold_max or bold_min:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_max[col] = df[col].max()
                col_min[col] = df[col].min()

    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]

            # Handle different types
            if pd.isna(val):
                cell = '--'
            elif isinstance(val, bool) or (isinstance(val, (np.bool_, np.integer)) and val in [0, 1, True, False]):
                cell = r'\checkmark' if val else ''
            elif isinstance(val, float):
                cell = float_format % val
            else:
                cell = str(val)

            # Apply bold formatting for max/min
            if col in col_max and pd.api.types.is_numeric_dtype(df[col]) and not pd.isna(val):
                if bold_max and val == col_max[col]:
                    cell = f'\\textbf{{{cell}}}'
                elif bold_min and val == col_min[col]:
                    cell = f'\\textbf{{{cell}}}'

            cells.append(cell)

        lines.append(' & '.join(cells) + r' \\')

    # Footer
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])

    latex_str = '\n'.join(lines)

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"Saved LaTeX table to {output_path}")
    return latex_str


def generate_latex_results_table(
    results_df: pd.DataFrame,
    output_path: str,
    caption: str,
    label: str
) -> str:
    """
    Generate a LaTeX table for experiment results.

    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the .tex file
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table as string
    """
    # Select and rename columns
    cols = ['study', 'beta', 'ate_covered', 'mean_width', 'cate_coverage_rate', 'n_strata']
    available_cols = [c for c in cols if c in results_df.columns]

    df = results_df[available_cols].copy()

    # Rename columns for LaTeX
    rename_map = {
        'study': 'Study',
        'beta': r'$\beta$',
        'ate_covered': 'ATE Cov.',
        'mean_width': 'Width',
        'cate_coverage_rate': 'CATE Cov.',
        'n_strata': r'$|Z|$'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return generate_latex_table(df, caption, label, output_path, float_format='%.2f')


def generate_latex_covariate_table(
    covariate_scores: pd.DataFrame,
    output_path: str,
    caption: str,
    label: str
) -> str:
    """
    Generate a LaTeX table for covariate scores.

    Args:
        covariate_scores: DataFrame with covariate scoring results
        output_path: Path to save the .tex file
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table as string
    """
    df = covariate_scores.copy()

    # Rename columns to Title Case
    df.columns = [col.replace('_', ' ').title() for col in df.columns]

    return generate_latex_table(df, caption, label, output_path, bold_max=True)


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_coverage_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_heatmap.png'
) -> None:
    """
    Plot ATE coverage as a heatmap by study and beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    # Pivot table
    pivot = results_df.pivot_table(
        values='ate_covered',
        index='study',
        columns='beta',
        aggfunc='mean'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Coverage Rate'}
    )

    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Study')
    ax.set_title('ATE Coverage by Study and Confounding Strength')

    save_figure(fig, Path(output_dir) / filename)


def plot_width_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_heatmap.png'
) -> None:
    """
    Plot mean bound width as a heatmap by study and beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    pivot = results_df.pivot_table(
        values='mean_width',
        index='study',
        columns='beta',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Mean Width'}
    )

    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Study')
    ax.set_title('Mean CATE Bound Width by Study and Confounding Strength')

    save_figure(fig, Path(output_dir) / filename)


def plot_cate_coverage_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'cate_coverage_heatmap.png'
) -> None:
    """
    Plot CATE coverage rate as a heatmap by study and beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    pivot = results_df.pivot_table(
        values='cate_coverage_rate',
        index='study',
        columns='beta',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'CATE Coverage Rate'}
    )

    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Study')
    ax.set_title('CATE Coverage Rate by Study and Confounding Strength')

    save_figure(fig, Path(output_dir) / filename)


def plot_width_boxplot_by_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_boxplot_beta.png'
) -> None:
    """
    Plot boxplot of bound widths grouped by beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by beta
    betas = sorted(results_df['beta'].unique())
    data = [results_df[results_df['beta'] == b]['mean_width'].dropna().tolist() for b in betas]

    # Boxplot
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)

    # Color boxes using gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(betas)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('Distribution of Bound Widths by Confounding Strength')

    save_figure(fig, Path(output_dir) / filename)


def plot_width_boxplot_by_study(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_boxplot_study.png'
) -> None:
    """
    Plot boxplot of bound widths grouped by study.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by study
    studies = sorted(results_df['study'].unique())
    data = [results_df[results_df['study'] == s]['mean_width'].dropna().tolist() for s in studies]

    # Boxplot
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)

    # Color boxes using gradient
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(studies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(studies, rotation=45, ha='right')
    ax.set_xlabel('Study')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('Distribution of Bound Widths by Study')

    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_coverage_boxplot_by_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_boxplot_beta.png'
) -> None:
    """
    Plot boxplot of ATE coverage grouped by beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by beta
    betas = sorted(results_df['beta'].unique())
    data = [results_df[results_df['beta'] == b]['ate_covered'].astype(float).dropna().tolist()
            for b in betas]

    # Boxplot
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)

    # Color boxes
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(betas)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Reference line at 95%
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% target')

    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('ATE Coverage')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_title('ATE Coverage by Confounding Strength')

    save_figure(fig, Path(output_dir) / filename)


def plot_forest_bounds(
    bounds_df: pd.DataFrame,
    true_ate: float,
    true_cates: Dict[str, float],
    output_dir: str,
    filename: str,
    study: str = '',
    beta: float = 0.0
) -> None:
    """
    Plot forest-style bound visualization for each stratum.

    Args:
        bounds_df: DataFrame with bound estimates
        true_ate: Ground truth ATE
        true_cates: Dict mapping z-tuple strings to true CATE values
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
        beta: Beta value for title
    """
    # Limit to first 20 strata for readability
    df = bounds_df.head(20).copy()
    n_strata = len(df)

    # Figure height scales with number of strata
    fig_height = max(6, n_strata * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = range(n_strata)

    for i, (_, row) in enumerate(df.iterrows()):
        lower = row['cate_lower']
        upper = row['cate_upper']
        midpoint = (lower + upper) / 2

        # Horizontal line for bounds
        ax.hlines(y=i, xmin=lower, xmax=upper, color='steelblue', linewidth=2)

        # Vertical caps at ends
        cap_height = 0.2
        ax.vlines(x=lower, ymin=i - cap_height, ymax=i + cap_height, color='steelblue', linewidth=2)
        ax.vlines(x=upper, ymin=i - cap_height, ymax=i + cap_height, color='steelblue', linewidth=2)

        # Dot at midpoint
        ax.plot(midpoint, i, 'o', color='steelblue', markersize=6)

        # Check if true CATE available for this stratum
        z_val = row.get('z', None)
        if z_val is not None:
            z_str = str(tuple(z_val)) if hasattr(z_val, '__iter__') else str(z_val)
            if z_str in true_cates:
                ax.plot(true_cates[z_str], i, '*', color='red', markersize=12)

    # Reference lines
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    if not np.isnan(true_ate):
        ax.axvline(x=true_ate, color='red', linestyle='--', linewidth=1.5, label=f'True ATE = {true_ate:.1f}')

    # Y-axis labels
    y_labels = [f"Z={row.get('z', i)}" for i, (_, row) in enumerate(df.iterrows())]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('CATE')
    ax.set_ylabel('Stratum')

    title = 'Estimated CATE Bounds by Stratum'
    if study:
        title += f' ({study}'
        if beta:
            title += f', β={beta}'
        title += ')'
    ax.set_title(title)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Estimated Bounds'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='True CATE'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='True ATE')
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.invert_yaxis()  # Top stratum at top
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_baseline_comparison_bar(
    results: Dict[str, Any],
    output_dir: str,
    filename: str,
    study: str = ''
) -> None:
    """
    Plot bar chart comparing grounding bounds to baseline methods.

    Args:
        results: Experiment results dict with 'metrics' and 'baselines'
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = results.get('metrics', {})
    baselines = results.get('baselines', {})
    true_ate = metrics.get('true_ate', np.nan)

    # Build data
    methods = ['Grounding']
    estimates = [(metrics.get('mean_lower', 0) + metrics.get('mean_upper', 0)) / 2]
    errors_low = [estimates[0] - metrics.get('mean_lower', 0)]
    errors_high = [metrics.get('mean_upper', 0) - estimates[0]]
    colors = ['steelblue']

    grounding_midpoint = estimates[0]

    for method, res in baselines.items():
        if 'estimate' in res:
            methods.append(method.upper())
            est = res['estimate']
            estimates.append(est)
            errors_low.append(0)
            errors_high.append(0)

            # Color based on distance to true ATE
            if not np.isnan(true_ate):
                grounding_dist = abs(grounding_midpoint - true_ate)
                baseline_dist = abs(est - true_ate)
                if baseline_dist < grounding_dist:
                    colors.append('green')
                elif baseline_dist > grounding_dist * 2:
                    colors.append('red')
                else:
                    colors.append('orange')
            else:
                colors.append('gray')

    x = np.arange(len(methods))
    bars = ax.bar(x, estimates, color=colors, width=0.6)

    # Error bars for grounding
    ax.errorbar(0, estimates[0], yerr=[[errors_low[0]], [errors_high[0]]],
                fmt='none', color='black', capsize=5, capthick=2)

    # True ATE reference line
    if not np.isnan(true_ate):
        ax.axhline(y=true_ate, color='red', linestyle='--', linewidth=1.5, label=f'True ATE = {true_ate:.1f}')

    # Value labels
    for bar, est in zip(bars, estimates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(estimates),
                f'{est:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Estimated ATE')
    ax.legend(loc='best')

    title = 'Comparison with Baseline Methods'
    if study:
        title += f' ({study})'
    ax.set_title(title)

    save_figure(fig, Path(output_dir) / filename)


def plot_epsilon_sensitivity_coverage(
    results_by_epsilon: Dict[float, Dict],
    output_dir: str,
    filename: str,
    study: str = ''
) -> None:
    """
    Plot coverage vs epsilon sensitivity analysis.

    Args:
        results_by_epsilon: Dict mapping epsilon to results dict
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    epsilons = sorted(results_by_epsilon.keys())
    coverages = [results_by_epsilon[e].get('metrics', {}).get('ate_covered', np.nan)
                 for e in epsilons]

    # Convert booleans to float
    coverages = [float(c) if isinstance(c, bool) else c for c in coverages]

    ax.plot(epsilons, coverages, 'o-', color='steelblue', markersize=8,
            markerfacecolor='white', markeredgecolor='steelblue', markeredgewidth=2)

    # Reference line at 95%
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% target')

    ax.set_xlabel(r'Epsilon ($\epsilon$)')
    ax.set_ylabel('ATE Coverage')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    title = 'Coverage Sensitivity to Epsilon'
    if study:
        title += f' ({study})'
    ax.set_title(title)

    save_figure(fig, Path(output_dir) / filename)


def plot_epsilon_sensitivity_width(
    results_by_epsilon: Dict[float, Dict],
    output_dir: str,
    filename: str,
    study: str = ''
) -> None:
    """
    Plot bound width vs epsilon sensitivity analysis.

    Args:
        results_by_epsilon: Dict mapping epsilon to results dict
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    epsilons = sorted(results_by_epsilon.keys())
    widths = [results_by_epsilon[e].get('metrics', {}).get('mean_width', np.nan)
              for e in epsilons]

    ax.plot(epsilons, widths, 's-', color='darkorange', markersize=8,
            markerfacecolor='white', markeredgecolor='darkorange', markeredgewidth=2)

    ax.set_xlabel(r'Epsilon ($\epsilon$)')
    ax.set_ylabel('Mean Bound Width')
    ax.grid(True, alpha=0.3)

    title = 'Bound Width Sensitivity to Epsilon'
    if study:
        title += f' ({study})'
    ax.set_title(title)

    save_figure(fig, Path(output_dir) / filename)


def plot_coverage_vs_width_scatter(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_vs_width_scatter.png'
) -> None:
    """
    Plot scatter of coverage vs width colored by beta.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert coverage to float
    coverage = results_df['ate_covered'].astype(float)
    width = results_df['mean_width']
    beta = results_df['beta']

    scatter = ax.scatter(width, coverage, c=beta, cmap='viridis', alpha=0.7, s=50)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'Confounding Strength ($\beta$)')

    # Reference line at 95%
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Mean Bound Width')
    ax.set_ylabel('ATE Coverage')
    ax.set_title('Coverage vs Bound Width Trade-off')

    save_figure(fig, Path(output_dir) / filename)


def plot_coverage_by_pattern(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_by_pattern.png'
) -> None:
    """
    Plot bar chart of coverage by confounding pattern.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'pattern' not in results_df.columns:
        print("Warning: 'pattern' column not found, skipping coverage_by_pattern plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean coverage by pattern
    coverage_by_pattern = results_df.groupby('pattern')['ate_covered'].mean()
    patterns = coverage_by_pattern.index.tolist()
    coverages = coverage_by_pattern.values

    # Colors from Set2 colormap
    colors = plt.cm.Set2(np.linspace(0, 1, len(patterns)))

    bars = ax.bar(patterns, coverages, color=colors, width=0.6)

    # Reference line at 95%
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% target')

    # Value labels on bars
    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{cov:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Confounding Pattern')
    ax.set_ylabel('Mean ATE Coverage')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    ax.set_title('ATE Coverage by Confounding Pattern')

    save_figure(fig, Path(output_dir) / filename)


def plot_width_by_site(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_by_site.png'
) -> None:
    """
    Plot mean bound width by site (study) with error bars.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'study' not in results_df.columns or 'mean_width' not in results_df.columns:
        print("Warning: Required columns not found, skipping width_by_site plot")
        return

    # Filter to successful experiments
    df = results_df[results_df['mean_width'].notna()].copy()
    if len(df) == 0:
        print("Warning: No valid data for width_by_site plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate by study
    width_stats = df.groupby('study')['mean_width'].agg(['mean', 'std', 'count'])
    width_stats = width_stats.sort_values('mean', ascending=True)

    studies = width_stats.index.tolist()
    means = width_stats['mean'].values
    stds = width_stats['std'].fillna(0).values
    counts = width_stats['count'].values

    # Standard error
    sems = stds / np.sqrt(counts)

    # Bar positions
    x = np.arange(len(studies))

    # Colors using gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(studies)))

    # Bar chart with error bars
    bars = ax.bar(x, means, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, capthick=1.5)

    # Value labels on bars
    for i, (bar, mean, count) in enumerate(zip(bars, means, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sems[i] + 0.01,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                f'n={int(count)}', ha='center', va='bottom', fontsize=7, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(studies, rotation=45, ha='right')
    ax.set_xlabel('Site (Study)')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('CATE Bound Width by Site')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_width_by_site_and_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_by_site_and_beta.png'
) -> None:
    """
    Plot bound width by site (study) grouped by beta values.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'study' not in results_df.columns or 'mean_width' not in results_df.columns:
        print("Warning: Required columns not found, skipping width_by_site_and_beta plot")
        return

    # Filter to successful experiments
    df = results_df[results_df['mean_width'].notna()].copy()
    if len(df) == 0:
        print("Warning: No valid data for width_by_site_and_beta plot")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Get unique studies and betas
    studies = sorted(df['study'].unique())
    betas = sorted(df['beta'].unique())
    n_betas = len(betas)

    # Bar width and positions
    bar_width = 0.8 / n_betas
    x = np.arange(len(studies))

    # Colors for each beta
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_betas))

    # Plot bars for each beta
    for i, beta in enumerate(betas):
        beta_data = df[df['beta'] == beta]
        means = []
        for study in studies:
            study_data = beta_data[beta_data['study'] == study]
            if len(study_data) > 0:
                means.append(study_data['mean_width'].mean())
            else:
                means.append(0)

        offset = (i - n_betas / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, label=f'β={beta}',
                      color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(studies, rotation=45, ha='right')
    ax.set_xlabel('Site (Study)')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('CATE Bound Width by Site and Confounding Strength')
    ax.legend(title='Confounding', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_cate_bounds_forest(
    result: Dict[str, Any],
    output_dir: str,
    filename: str = 'cate_bounds_forest.png',
    study: str = '',
    beta: float = 0.0
) -> None:
    """
    Plot forest-style visualization showing true CATE with estimated bounds.

    Args:
        result: Single experiment result dict with 'bounds' and 'true_cates'
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
        beta: Beta value for title
    """
    if 'bounds' not in result:
        print("Warning: No bounds data available for forest plot")
        return

    bounds_df = pd.DataFrame(result['bounds'])
    true_cates = result.get('true_cates', {})
    true_ate = result.get('metrics', {}).get('true_ate', np.nan)

    if len(bounds_df) == 0:
        print("Warning: Empty bounds data")
        return

    # Match bounds with true CATEs
    plot_data = []
    for _, row in bounds_df.iterrows():
        z_val = row.get('z', None)
        if z_val is not None:
            z_str = str(tuple(z_val)) if hasattr(z_val, '__iter__') else str(z_val)
            if z_str in true_cates:
                plot_data.append({
                    'z': z_str,
                    'true_cate': true_cates[z_str],
                    'lower': row['cate_lower'],
                    'upper': row['cate_upper'],
                    'covered': row['cate_lower'] <= true_cates[z_str] <= row['cate_upper']
                })

    if len(plot_data) == 0:
        # Fall back to using bounds without true CATEs
        for i, row in bounds_df.head(20).iterrows():
            z_val = row.get('z', i)
            plot_data.append({
                'z': str(z_val),
                'true_cate': None,
                'lower': row['cate_lower'],
                'upper': row['cate_upper'],
                'covered': True  # Unknown
            })

    # Sort by true CATE if available
    if plot_data[0]['true_cate'] is not None:
        plot_data = sorted(plot_data, key=lambda x: x['true_cate'])

    n_strata = len(plot_data)
    fig_height = max(5, n_strata * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_strata)

    for i, row in enumerate(plot_data):
        lower = row['lower']
        upper = row['upper']
        true_cate = row['true_cate']
        is_covered = row['covered']

        # Color based on coverage
        color = '#2ecc71' if is_covered else '#e74c3c'

        # Draw bounds
        ax.hlines(y=i, xmin=lower, xmax=upper, color=color, linewidth=3, alpha=0.7)
        cap_height = 0.25
        ax.vlines(x=lower, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)
        ax.vlines(x=upper, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)

        # True CATE marker
        if true_cate is not None:
            ax.plot(true_cate, i, 'D', color='black', markersize=8, zorder=5)

    # Reference lines
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    if not np.isnan(true_ate):
        ax.axvline(x=true_ate, color='#3498db', linestyle=':', linewidth=2,
                   label=f'True ATE = {true_ate:.2f}')

    # Y-axis labels
    y_labels = [row['z'][:20] for row in plot_data]  # Truncate long labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel('Treatment Effect', fontsize=11)
    ax.set_ylabel('Stratum', fontsize=11)

    title = 'CATE Bounds vs True Values'
    if study:
        title += f' ({study}'
        if beta:
            title += f', β={beta}'
        title += ')'
    ax.set_title(title, fontsize=12)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=3, label='Covered'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Not Covered'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=8, label='True CATE'),
    ]
    if not np.isnan(true_ate):
        legend_elements.append(Line2D([0], [0], color='#3498db', linestyle=':', linewidth=2, label='True ATE'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_cate_bounds_grid(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str = 'cate_bounds_grid.png'
) -> None:
    """
    Plot grid of CATE bounds vs true values for multiple experiments.

    Args:
        results: List of experiment result dicts
        output_dir: Directory to save the figure
        filename: Output filename
    """
    # Filter to results with bounds and true_cates
    valid_results = []
    for r in results:
        if 'error' not in r and 'bounds' in r:
            bounds_df = pd.DataFrame(r['bounds'])
            true_cates = r.get('true_cates', {})
            if len(bounds_df) > 0 and len(true_cates) > 0:
                valid_results.append(r)

    if len(valid_results) == 0:
        print("Warning: No valid results for CATE bounds grid")
        return

    # Limit to reasonable number
    valid_results = valid_results[:9]
    n_results = len(valid_results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, result in enumerate(valid_results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        bounds_df = pd.DataFrame(result['bounds'])
        true_cates = result.get('true_cates', {})

        # Build plot data
        plot_data = []
        for _, brow in bounds_df.iterrows():
            z_val = brow.get('z', None)
            if z_val is not None:
                z_str = str(tuple(z_val)) if hasattr(z_val, '__iter__') else str(z_val)
                if z_str in true_cates:
                    plot_data.append({
                        'true_cate': true_cates[z_str],
                        'lower': brow['cate_lower'],
                        'upper': brow['cate_upper'],
                        'covered': brow['cate_lower'] <= true_cates[z_str] <= brow['cate_upper']
                    })

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        plot_data = sorted(plot_data, key=lambda x: x['true_cate'])
        n_strata = len(plot_data)
        y_positions = np.arange(n_strata)

        for i, prow in enumerate(plot_data):
            color = '#2ecc71' if prow['covered'] else '#e74c3c'
            ax.hlines(y=i, xmin=prow['lower'], xmax=prow['upper'], color=color, linewidth=2, alpha=0.7)
            ax.vlines(x=prow['lower'], ymin=i - 0.2, ymax=i + 0.2, color=color, linewidth=1.5)
            ax.vlines(x=prow['upper'], ymin=i - 0.2, ymax=i + 0.2, color=color, linewidth=1.5)
            ax.plot(prow['true_cate'], i, 'D', color='black', markersize=5, zorder=5)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        config = result.get('config', {})
        study = config.get('study', '?')
        beta = config.get('beta', '?')
        coverage = result.get('metrics', {}).get('cate_coverage_rate', 0)
        ax.set_title(f'{study} β={beta}\n(Cov: {coverage:.0%})', fontsize=10)
        ax.set_xlabel('Effect', fontsize=9)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

    # Hide unused subplots
    for idx in range(n_results, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=2, label='Covered'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='Not Covered'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=5, label='True CATE'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, Path(output_dir) / filename)


def plot_covariate_scores_bar(
    covariate_scores: pd.DataFrame,
    output_dir: str,
    filename: str,
    study: str = ''
) -> None:
    """
    Plot horizontal bar chart of covariate scores.

    Args:
        covariate_scores: DataFrame with covariate scoring results
        output_dir: Directory to save the figure
        filename: Output filename
        study: Study name for title
    """
    if covariate_scores is None or len(covariate_scores) == 0:
        print("Warning: No covariate scores to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get score column
    score_col = 'score' if 'score' in covariate_scores.columns else covariate_scores.columns[-1]
    name_col = 'z_a' if 'z_a' in covariate_scores.columns else covariate_scores.index.name or 'covariate'

    # Sort by score ascending (best at top when reading)
    if name_col in covariate_scores.columns:
        df = covariate_scores.sort_values(score_col, ascending=True)
        names = df[name_col].tolist()
    else:
        df = covariate_scores.sort_values(score_col, ascending=True)
        names = df.index.tolist()

    scores = df[score_col].values

    # Colors using RdYlGn gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores)))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors)

    # Reference line at x=0
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Score')
    ax.set_ylabel('Covariate')

    title = 'Covariate Scores (EHS Criteria)'
    if study:
        title += f' ({study})'
    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


# =============================================================================
# FIGURE GENERATION HELPERS
# =============================================================================

def generate_all_figures(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None,
    results_by_epsilon: Optional[Dict[float, Dict]] = None,
    all_results: Optional[List[Dict[str, Any]]] = None,
    study: str = '',
    beta: float = 0.0,
    verbose: bool = True
) -> List[str]:
    """
    Generate all figures for experiment results.

    Args:
        results_df: DataFrame with experiment results (from run_full_grid or run_experiment_sweep)
        output_dir: Directory to save all figures
        single_result: Optional single experiment result dict (for forest/baseline plots)
        results_by_epsilon: Optional dict mapping epsilon to results (for sensitivity plots)
        all_results: Optional list of all experiment result dicts (for CATE bounds grid)
        study: Study name for titles
        beta: Beta value for titles
        verbose: Print progress messages

    Returns:
        List of generated figure paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if verbose:
        print(f"\nGenerating figures in {output_dir}...")

    # Grid-level plots (require results_df with multiple experiments)
    if results_df is not None and len(results_df) > 0:
        # Heatmaps
        try:
            if 'ate_covered' in results_df.columns and 'beta' in results_df.columns:
                plot_coverage_heatmap(results_df, output_dir)
                generated_files.append('coverage_heatmap.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate coverage heatmap: {e}")

        try:
            if 'mean_width' in results_df.columns and 'beta' in results_df.columns:
                plot_width_heatmap(results_df, output_dir)
                generated_files.append('width_heatmap.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width heatmap: {e}")

        try:
            if 'cate_coverage_rate' in results_df.columns and 'beta' in results_df.columns:
                plot_cate_coverage_heatmap(results_df, output_dir)
                generated_files.append('cate_coverage_heatmap.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate CATE coverage heatmap: {e}")

        # Boxplots
        try:
            if 'mean_width' in results_df.columns and 'beta' in results_df.columns:
                plot_width_boxplot_by_beta(results_df, output_dir)
                generated_files.append('width_boxplot_beta.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width boxplot by beta: {e}")

        try:
            if 'mean_width' in results_df.columns and 'study' in results_df.columns:
                plot_width_boxplot_by_study(results_df, output_dir)
                generated_files.append('width_boxplot_study.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width boxplot by study: {e}")

        try:
            if 'ate_covered' in results_df.columns and 'beta' in results_df.columns:
                plot_coverage_boxplot_by_beta(results_df, output_dir)
                generated_files.append('coverage_boxplot_beta.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate coverage boxplot: {e}")

        # Scatter plot
        try:
            if 'ate_covered' in results_df.columns and 'mean_width' in results_df.columns:
                plot_coverage_vs_width_scatter(results_df, output_dir)
                generated_files.append('coverage_vs_width_scatter.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate coverage vs width scatter: {e}")

        # Coverage by pattern
        try:
            if 'pattern' in results_df.columns and 'ate_covered' in results_df.columns:
                plot_coverage_by_pattern(results_df, output_dir)
                generated_files.append('coverage_by_pattern.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate coverage by pattern: {e}")

        # Width by site
        try:
            if 'study' in results_df.columns and 'mean_width' in results_df.columns:
                plot_width_by_site(results_df, output_dir)
                generated_files.append('width_by_site.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width by site: {e}")

        # Width by site and beta
        try:
            if 'study' in results_df.columns and 'mean_width' in results_df.columns and 'beta' in results_df.columns:
                plot_width_by_site_and_beta(results_df, output_dir)
                generated_files.append('width_by_site_and_beta.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width by site and beta: {e}")

    # Single experiment plots (require single_result)
    if single_result is not None and 'error' not in single_result:
        # Forest bounds plot
        try:
            if 'bounds' in single_result:
                bounds_df = pd.DataFrame(single_result['bounds'])
                true_ate = single_result.get('metrics', {}).get('true_ate', np.nan)
                true_cates = single_result.get('true_cates', {})

                filename = f'forest_bounds_{study}_{beta}.png' if study else 'forest_bounds.png'
                plot_forest_bounds(
                    bounds_df, true_ate, true_cates,
                    output_dir, filename,
                    study=study, beta=beta
                )
                generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate forest bounds plot: {e}")

        # Baseline comparison
        try:
            if single_result.get('baselines'):
                filename = f'baseline_comparison_{study}_{beta}.png' if study else 'baseline_comparison.png'
                plot_baseline_comparison_bar(
                    single_result, output_dir, filename, study=study
                )
                generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate baseline comparison: {e}")

        # Covariate scores
        try:
            if single_result.get('covariate_scores'):
                cov_scores = single_result['covariate_scores']
                if isinstance(cov_scores, dict):
                    cov_scores = pd.DataFrame(cov_scores)

                filename = f'covariate_scores_{study}_{beta}.png' if study else 'covariate_scores.png'
                plot_covariate_scores_bar(
                    cov_scores, output_dir, filename, study=study
                )
                generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate covariate scores plot: {e}")

        # CATE bounds forest plot
        try:
            filename = f'cate_bounds_forest_{study}_{beta}.png' if study else 'cate_bounds_forest.png'
            plot_cate_bounds_forest(
                single_result, output_dir, filename,
                study=study, beta=beta
            )
            generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate CATE bounds forest: {e}")

    # CATE bounds grid if multiple results provided
    if all_results is not None and len(all_results) > 0:
        valid_results = [r for r in all_results if 'error' not in r and 'bounds' in r]
        if len(valid_results) > 0:
            try:
                plot_cate_bounds_grid(valid_results, output_dir)
                generated_files.append('cate_bounds_grid.png')
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate CATE bounds grid: {e}")

    # Epsilon sensitivity plots (require results_by_epsilon)
    if results_by_epsilon is not None and len(results_by_epsilon) > 0:
        try:
            filename = f'epsilon_sensitivity_coverage_{study}.png' if study else 'epsilon_sensitivity_coverage.png'
            plot_epsilon_sensitivity_coverage(
                results_by_epsilon, output_dir, filename, study=study
            )
            generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate epsilon coverage plot: {e}")

        try:
            filename = f'epsilon_sensitivity_width_{study}.png' if study else 'epsilon_sensitivity_width.png'
            plot_epsilon_sensitivity_width(
                results_by_epsilon, output_dir, filename, study=study
            )
            generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate epsilon width plot: {e}")

    if verbose:
        print(f"\nGenerated {len(generated_files)} figures:")
        for f in generated_files:
            print(f"  - {f}")

    return generated_files


def generate_all_tables(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None,
    study: str = '',
    verbose: bool = True
) -> List[str]:
    """
    Generate all LaTeX tables for experiment results.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save all tables
        single_result: Optional single experiment result dict
        study: Study name for captions
        verbose: Print progress messages

    Returns:
        List of generated table paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if verbose:
        print(f"\nGenerating LaTeX tables in {output_dir}...")

    # Results summary table
    if results_df is not None and len(results_df) > 0:
        try:
            caption = f'Experiment Results{" for " + study if study else ""}'
            filename = f'results_table_{study}.tex' if study else 'results_table.tex'
            filepath = output_path / filename

            generate_latex_results_table(
                results_df, str(filepath),
                caption=caption,
                label=f'tab:results_{study}' if study else 'tab:results'
            )
            generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate results table: {e}")

    # Covariate scores table
    if single_result is not None and 'error' not in single_result:
        try:
            if single_result.get('covariate_scores'):
                cov_scores = single_result['covariate_scores']
                if isinstance(cov_scores, dict):
                    cov_scores = pd.DataFrame(cov_scores)

                caption = f'Covariate Scores (EHS Criteria){" for " + study if study else ""}'
                filename = f'covariate_table_{study}.tex' if study else 'covariate_table.tex'
                filepath = output_path / filename

                generate_latex_covariate_table(
                    cov_scores, str(filepath),
                    caption=caption,
                    label=f'tab:covariates_{study}' if study else 'tab:covariates'
                )
                generated_files.append(filename)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate covariate table: {e}")

    if verbose:
        print(f"\nGenerated {len(generated_files)} tables:")
        for f in generated_files:
            print(f"  - {f}")

    return generated_files


def generate_full_report(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None,
    results_by_epsilon: Optional[Dict[float, Dict]] = None,
    all_results: Optional[List[Dict[str, Any]]] = None,
    study: str = '',
    beta: float = 0.0,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Generate complete report with all figures and tables.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save all outputs
        single_result: Optional single experiment result dict
        results_by_epsilon: Optional dict mapping epsilon to results
        all_results: Optional list of all experiment result dicts
        study: Study name for titles/captions
        beta: Beta value for titles
        verbose: Print progress messages

    Returns:
        Dict with 'figures' and 'tables' lists of generated files
    """
    output_path = Path(output_dir)

    # Create subdirectories
    figures_dir = output_path / 'figures'
    tables_dir = output_path / 'tables'

    figures = generate_all_figures(
        results_df=results_df,
        output_dir=str(figures_dir),
        single_result=single_result,
        results_by_epsilon=results_by_epsilon,
        all_results=all_results,
        study=study,
        beta=beta,
        verbose=verbose
    )

    tables = generate_all_tables(
        results_df=results_df,
        output_dir=str(tables_dir),
        single_result=single_result,
        study=study,
        verbose=verbose
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"REPORT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Figures: {len(figures)} in {figures_dir}")
        print(f"Tables:  {len(tables)} in {tables_dir}")

    return {
        'figures': figures,
        'tables': tables
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run causal grounding experiments'
    )

    parser.add_argument('--study', type=str, help='Study name')
    parser.add_argument('--pattern', type=str, default='age', help='Confounding pattern')
    parser.add_argument('--beta', type=float, default=0.3, help='Confounding strength')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Naturalness tolerance')
    parser.add_argument('--target', type=str, default='mturk', help='Target site')
    parser.add_argument('--n-permutations', type=int, default=500, help='Number of permutations')
    parser.add_argument('--data-dir', type=str, default='confounded_datasets', help='Data directory')
    parser.add_argument('--rct-path', type=str, default='ManyLabs1/pre-process/Manylabs1_data.pkl',
                        help='Path to RCT data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--baselines', action='store_true', help='Compute baseline methods')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--grid', action='store_true', help='Run grid of experiments')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--binary-only', action='store_true',
                        help='Filter to studies with binary outcomes only '
                             f'({", ".join(BINARY_OUTCOME_STUDIES)})')
    parser.add_argument('--report', action='store_true',
                        help='Generate full report with figures and tables after running experiments')

    args = parser.parse_args()

    # Determine which studies to run
    if args.binary_only:
        studies_to_run = BINARY_OUTCOME_STUDIES
        print(f"Filtering to binary outcome studies: {studies_to_run}")
    else:
        studies_to_run = None  # Use defaults in each function

    if args.all:
        # Run full sweep
        results_df, full_results = run_experiment_sweep(
            studies=studies_to_run,
            output_dir=args.output,
            verbose=True,
            return_full_results=True
        )
        # Generate report if requested
        if args.report and results_df is not None and len(results_df) > 0:
            print("\nGenerating report...")
            generate_full_report(
                results_df=results_df,
                output_dir=args.output,
                all_results=full_results,
                verbose=True
            )
    elif args.grid:
        # Run grid experiment
        results_df = run_full_grid(
            studies=studies_to_run,
            epsilon=args.epsilon,
            output_dir=args.output,
            target_site=args.target,
            n_permutations=args.n_permutations,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            random_seed=args.seed
        )
        # Generate report if requested
        if args.report and results_df is not None and len(results_df) > 0:
            print("\nGenerating report...")
            generate_full_report(
                results_df=results_df,
                output_dir=args.output,
                verbose=True
            )
    elif args.study:
        # Run single experiment
        result = run_single_experiment(
            study=args.study,
            pattern=args.pattern,
            beta=args.beta,
            epsilon=args.epsilon,
            target_site=args.target,
            n_permutations=args.n_permutations,
            data_dir=args.data_dir,
            rct_path=args.rct_path,
            random_seed=args.seed,
            compute_baselines=args.baselines
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)

        if 'error' not in result:
            config = result['config']
            metrics = result['metrics']

            print(f"Study: {config['study']}")
            print(f"Pattern: {config['pattern']}, Beta: {config['beta']}")
            print(f"Epsilon: {config['epsilon']}")
            print(f"Training sites: {metrics['n_training_sites']}")
            print(f"Z-values (strata): {metrics['n_strata']}")
            print(f"\nBounds:")
            print(f"  Mean lower: {metrics['mean_lower']:.2f}")
            print(f"  Mean upper: {metrics['mean_upper']:.2f}")
            print(f"  Mean width: {metrics['mean_width']:.2f}")
            print(f"  Median width: {metrics['median_width']:.2f}")

            if not np.isnan(metrics['true_ate']):
                print(f"\nGround Truth:")
                print(f"  True ATE: {metrics['true_ate']:.2f}")
                print(f"  ATE covered: {metrics['ate_covered']}")
                if not np.isnan(metrics['cate_coverage_rate']):
                    print(f"  CATE coverage rate: {metrics['cate_coverage_rate']:.1%}")

            print(f"\nBest instrument: {result['best_instrument']}")

            if result['baselines']:
                print("\nBaseline Methods:")
                for method, res in result['baselines'].items():
                    if 'estimate' in res:
                        print(f"  {method}: {res['estimate']:.2f}")
        else:
            print(f"Error: {result['error']}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
