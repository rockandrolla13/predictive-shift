"""
Synthetic Data Experiment Runner

Runs causal grounding experiments on synthetic data with known ground truth.

Usage:
    python experiments_synthetic/run_experiment.py --beta 0.3 --n-sites 10
    python experiments_synthetic/run_experiment.py --grid
    python experiments_synthetic/run_experiment.py --all --report
"""

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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CausalGroundingEstimator,
)

from experiments_synthetic.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_multi_site_data,
    compute_true_cate,
    create_confounding_sweep_configs,
    create_heterogeneity_sweep_configs,
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

BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
EPSILONS = [0.05, 0.1, 0.2, 0.3, 0.5]
N_SITES_OPTIONS = [5, 10, 20, 50]
N_Z_VALUES_OPTIONS = [2, 3, 5, 10]
TREATMENT_EFFECTS = [0.0, 0.1, 0.2, 0.3, 0.5]
INTERACTIONS = [0.0, 0.05, 0.1, 0.15, 0.2]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig, output_path, close=True):
    """Save a matplotlib figure with consistent settings."""
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


def adapt_training_data_columns(
    training_data: Dict[str, pd.DataFrame],
    treatment_col: str = 'X',
    outcome_col: str = 'Y',
    covariate_cols: List[str] = ['Z']
) -> Dict[str, pd.DataFrame]:
    """
    Adapt synthetic data column names for CausalGroundingEstimator.
    
    The estimator expects:
    - treatment='iv', outcome='dv'
    - covariates with '_cat' suffix
    
    Args:
        training_data: Dict of site DataFrames
        treatment_col: Current treatment column name
        outcome_col: Current outcome column name
        covariate_cols: Current covariate column names
    
    Returns:
        Adapted training data dict
    """
    adapted = {}
    
    for site_id, df in training_data.items():
        df_new = df.copy()
        
        # Rename treatment and outcome
        df_new['iv'] = df_new[treatment_col]
        df_new['dv'] = df_new[outcome_col]
        
        # Create discretized covariate columns
        for cov in covariate_cols:
            df_new[f'{cov}_cat'] = df_new[cov]
        
        adapted[site_id] = df_new
    
    return adapted


# =============================================================================
# EXPERIMENT RUNNER FUNCTIONS
# =============================================================================

def run_synthetic_experiment(
    config: Optional[SyntheticDataConfig] = None,
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    beta: float = 0.3,
    treatment_effect: float = 0.2,
    treatment_z_interaction: float = 0.1,
    epsilon: float = 0.1,
    n_permutations: int = 100,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run one complete experiment on synthetic data.
    
    Args:
        config: Full SyntheticDataConfig (takes precedence over individual params)
        n_sites: Number of training sites
        n_per_site: Samples per site
        n_z_values: Number of discrete covariate values
        beta: Confounding strength
        treatment_effect: Main treatment effect
        treatment_z_interaction: CATE heterogeneity
        epsilon: Naturalness tolerance
        n_permutations: Permutations for CI tests
        random_seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with config, bounds, metrics, and ground truth
    """
    # Build config if not provided
    if config is None:
        config = SyntheticDataConfig(
            n_z_values=n_z_values,
            beta=beta,
            treatment_effect=treatment_effect,
            treatment_z_interaction=treatment_z_interaction,
            n_per_site=n_per_site,
            seed=random_seed
        )
    
    # Store experiment configuration
    exp_config = {
        'n_sites': n_sites,
        'n_per_site': config.n_per_site,
        'n_z_values': config.n_z_values,
        'beta': config.beta,
        'treatment_effect': config.treatment_effect,
        'treatment_z_interaction': config.treatment_z_interaction,
        'epsilon': epsilon,
        'n_permutations': n_permutations,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print(f"Running experiment: beta={config.beta}, epsilon={epsilon}")
    
    try:
        # Generate synthetic data
        generator = SyntheticDataGenerator(config, n_sites=n_sites)
        training_data = generator.generate_training_data()
        
        # Get ground truth
        true_cates = generator.get_true_cates()
        true_ate = generator.get_true_ate()
        
        # Adapt data for estimator
        adapted_data = adapt_training_data_columns(training_data)
        
        # Initialize and fit estimator
        # Disable discretization since we already have discrete covariates
        estimator = CausalGroundingEstimator(
            epsilon=epsilon,
            transfer_method='conservative',
            n_permutations=n_permutations,
            discretize=False,  # Already discrete
            random_seed=random_seed,
            verbose=False
        )
        
        # Fit with Z_cat as covariate
        estimator.fit(
            adapted_data, 
            treatment='iv', 
            outcome='dv',
            covariates=['Z_cat']
        )
        
        # Get bounds
        bounds_df = estimator.predict_bounds()
        
        # Calculate metrics
        mean_lower = bounds_df['cate_lower'].mean()
        mean_upper = bounds_df['cate_upper'].mean()
        mean_width = bounds_df['width'].mean()
        median_width = bounds_df['width'].median()
        
        # Check ATE coverage
        ate_covered = mean_lower <= true_ate <= mean_upper
        
        # Compute CATE coverage
        cate_coverage = []
        bounds_by_z = {}
        
        for _, row in bounds_df.iterrows():
            z_col = 'Z_cat' if 'Z_cat' in row else row.get('z', None)
            if z_col is not None and z_col in row:
                z_val = (row[z_col],) if not isinstance(row[z_col], tuple) else row[z_col]
            elif 'z' in row:
                z_val = row['z']
                if not isinstance(z_val, tuple):
                    z_val = (z_val,)
            else:
                continue
            
            bounds_by_z[z_val] = (row['cate_lower'], row['cate_upper'])
            
            if z_val in true_cates:
                true_cate = true_cates[z_val]
                if row['cate_lower'] <= true_cate <= row['cate_upper']:
                    cate_coverage.append(1.0)
                else:
                    cate_coverage.append(0.0)
        
        cate_coverage_rate = np.mean(cate_coverage) if cate_coverage else np.nan
        
        # Per-stratum coverage details
        per_stratum_results = []
        for z_val, true_cate in true_cates.items():
            if z_val in bounds_by_z:
                lower, upper = bounds_by_z[z_val]
                is_covered = lower <= true_cate <= upper
                per_stratum_results.append({
                    'z': z_val,
                    'true_cate': true_cate,
                    'bound_lower': lower,
                    'bound_upper': upper,
                    'width': upper - lower,
                    'is_covered': is_covered
                })
        
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
        
        # Get covariate scores
        covariate_scores = None
        if hasattr(estimator, 'covariate_scores_') and estimator.covariate_scores_ is not None:
            covariate_scores = estimator.covariate_scores_
            if isinstance(covariate_scores, pd.DataFrame):
                covariate_scores = covariate_scores.to_dict('records')
        
        best_instrument = getattr(estimator, 'best_instrument_', None)
        
        # Convert true_cates keys to strings for JSON
        true_cates_str = {str(k): v for k, v in true_cates.items()}
        
        return {
            'config': exp_config,
            'bounds': bounds_df.to_dict('records'),
            'metrics': metrics,
            'true_cates': true_cates_str,
            'per_stratum_results': per_stratum_results,
            'covariate_scores': covariate_scores,
            'best_instrument': best_instrument
        }
        
    except Exception as e:
        return {
            'config': exp_config,
            'error': str(e)
        }


def run_synthetic_grid(
    betas: Optional[List[float]] = None,
    epsilons: Optional[List[float]] = None,
    n_sites_list: Optional[List[int]] = None,
    n_z_values: int = 3,
    treatment_effect: float = 0.2,
    treatment_z_interaction: float = 0.1,
    n_per_site: int = 500,
    n_permutations: int = 100,
    random_seed: int = 42,
    output_dir: str = 'results_synthetic',
    verbose: bool = True,
    return_full_results: bool = False
) -> pd.DataFrame:
    """
    Run experiments across a grid of parameters.
    
    Args:
        betas: Confounding strengths to test
        epsilons: Naturalness tolerances to test
        n_sites_list: Number of sites to test
        n_z_values: Number of Z values
        treatment_effect: Main effect
        treatment_z_interaction: Heterogeneity
        n_per_site: Samples per site
        n_permutations: Permutations for CI tests
        random_seed: Random seed
        output_dir: Output directory
        verbose: Print progress
        return_full_results: If True, return tuple of (DataFrame, list of full result dicts)
    
    Returns:
        DataFrame with all results (or tuple if return_full_results=True)
    """
    # Set defaults
    if betas is None:
        betas = BETAS
    if epsilons is None:
        epsilons = [0.1]
    if n_sites_list is None:
        n_sites_list = [10]
    
    # Initialize results
    results = []
    full_results = []
    
    # Calculate total experiments
    total = len(betas) * len(epsilons) * len(n_sites_list)
    current = 0
    
    for n_sites in n_sites_list:
        for beta in betas:
            for epsilon in epsilons:
                current += 1
                if verbose:
                    print(f"[{current}/{total}] n_sites={n_sites}, beta={beta}, epsilon={epsilon}")
                
                try:
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
                    
                    full_results.append(result)
                    
                    if 'error' not in result:
                        row = {
                            'n_sites': n_sites,
                            'beta': beta,
                            'epsilon': epsilon,
                            'n_z_values': n_z_values,
                            'treatment_effect': treatment_effect,
                            'treatment_z_interaction': treatment_z_interaction,
                            **result['metrics']
                        }
                        results.append(row)
                    else:
                        if verbose:
                            print(f"  Error: {result['error']}")
                
                except Exception as e:
                    if verbose:
                        print(f"  Exception: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'synthetic_grid_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Save detailed JSON (including per_stratum_results)
    json_path = output_path / f'synthetic_grid_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Saved detailed results to {json_path}")
    
    if return_full_results:
        return results_df, full_results
    return results_df


def run_confounding_sweep(
    betas: Optional[List[float]] = None,
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    epsilon: float = 0.1,
    n_permutations: int = 100,
    random_seed: int = 42,
    output_dir: str = 'results_synthetic',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run sweep over confounding strengths.
    
    Args:
        betas: Confounding strengths to test
        n_sites: Number of sites
        n_per_site: Samples per site
        n_z_values: Number of Z values
        epsilon: Naturalness tolerance
        n_permutations: CI test permutations
        random_seed: Random seed
        output_dir: Output directory
        verbose: Print progress
    
    Returns:
        DataFrame with results
    """
    return run_synthetic_grid(
        betas=betas or BETAS,
        epsilons=[epsilon],
        n_sites_list=[n_sites],
        n_z_values=n_z_values,
        n_per_site=n_per_site,
        n_permutations=n_permutations,
        random_seed=random_seed,
        output_dir=output_dir,
        verbose=verbose
    )


def run_epsilon_sweep(
    epsilons: Optional[List[float]] = None,
    beta: float = 0.3,
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    n_permutations: int = 100,
    random_seed: int = 42,
    output_dir: str = 'results_synthetic',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run sweep over epsilon (naturalness tolerance).
    """
    return run_synthetic_grid(
        betas=[beta],
        epsilons=epsilons or EPSILONS,
        n_sites_list=[n_sites],
        n_z_values=n_z_values,
        n_per_site=n_per_site,
        n_permutations=n_permutations,
        random_seed=random_seed,
        output_dir=output_dir,
        verbose=verbose
    )


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_coverage_vs_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'coverage_vs_beta.png'
) -> None:
    """Plot ATE and CATE coverage vs confounding strength."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ATE coverage
    ax = axes[0]
    coverage_by_beta = results_df.groupby('beta')['ate_covered'].mean()
    ax.plot(coverage_by_beta.index, coverage_by_beta.values, 
            'o-', color='steelblue', markersize=8, linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% target')
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('ATE Coverage Rate')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('ATE Coverage vs Confounding')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # CATE coverage
    ax = axes[1]
    cate_coverage_by_beta = results_df.groupby('beta')['cate_coverage_rate'].mean()
    ax.plot(cate_coverage_by_beta.index, cate_coverage_by_beta.values,
            's-', color='darkorange', markersize=8, linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% target')
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('CATE Coverage Rate')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('CATE Coverage vs Confounding')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_width_vs_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_vs_beta.png'
) -> None:
    """Plot bound width vs confounding strength."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    width_by_beta = results_df.groupby('beta')['mean_width'].agg(['mean', 'std'])
    
    ax.errorbar(
        width_by_beta.index, 
        width_by_beta['mean'],
        yerr=width_by_beta['std'],
        fmt='o-', 
        color='steelblue', 
        markersize=8, 
        linewidth=2,
        capsize=5
    )
    
    ax.set_xlabel(r'Confounding Strength ($\beta$)')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('Bound Width vs Confounding Strength')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, Path(output_dir) / filename)


def plot_forest_bounds_synthetic(
    result: Dict[str, Any],
    output_dir: str,
    filename: str = 'forest_bounds.png'
) -> None:
    """Plot forest-style bound visualization."""
    if 'per_stratum_results' not in result or not result['per_stratum_results']:
        print("No per-stratum results to plot")
        return
    
    per_stratum = result['per_stratum_results']
    n_strata = len(per_stratum)
    
    fig_height = max(4, n_strata * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y_positions = range(n_strata)
    
    for i, row in enumerate(per_stratum):
        lower = row['bound_lower']
        upper = row['bound_upper']
        true_cate = row['true_cate']
        is_covered = row['is_covered']
        
        color = 'green' if is_covered else 'red'
        
        # Draw bound interval
        ax.hlines(y=i, xmin=lower, xmax=upper, color=color, linewidth=3, alpha=0.7)
        
        # Caps
        cap_height = 0.2
        ax.vlines(x=lower, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)
        ax.vlines(x=upper, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)
        
        # True CATE marker
        ax.plot(true_cate, i, '*', color='black', markersize=12)
    
    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # True ATE
    true_ate = result['metrics']['true_ate']
    ax.axvline(x=true_ate, color='blue', linestyle=':', linewidth=1.5, 
               label=f'True ATE = {true_ate:.3f}')
    
    # Y-axis labels
    y_labels = [f"Z={row['z']}" for row in per_stratum]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('CATE')
    ax.set_ylabel('Stratum')
    ax.set_title(f"CATE Bounds (β={result['config']['beta']}, ε={result['config']['epsilon']})")
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Covered', alpha=0.7),
        Line2D([0], [0], color='red', linewidth=3, label='Not Covered', alpha=0.7),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
               markersize=12, label='True CATE'),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=1.5, label='True ATE')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_coverage_heatmap_synthetic(
    results_df: pd.DataFrame,
    output_dir: str,
    x_col: str = 'beta',
    y_col: str = 'epsilon',
    filename: str = 'coverage_heatmap.png'
) -> None:
    """Plot coverage as heatmap by two parameters."""
    if x_col not in results_df.columns or y_col not in results_df.columns:
        print(f"Columns {x_col} or {y_col} not found in results")
        return
    
    pivot = results_df.pivot_table(
        values='ate_covered',
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print("No data to plot heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'ATE Coverage Rate'}
    )
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title('ATE Coverage by Parameters')
    
    save_figure(fig, Path(output_dir) / filename)


def plot_width_by_n_sites(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_by_n_sites.png'
) -> None:
    """
    Plot bound width vs number of training sites.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'n_sites' not in results_df.columns or 'mean_width' not in results_df.columns:
        print("Warning: Required columns not found, skipping width_by_n_sites plot")
        return
    
    # Filter to valid data
    df = results_df[results_df['mean_width'].notna()].copy()
    if len(df) == 0:
        print("Warning: No valid data for width_by_n_sites plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate by n_sites
    width_stats = df.groupby('n_sites')['mean_width'].agg(['mean', 'std', 'count'])
    
    n_sites_vals = width_stats.index.tolist()
    means = width_stats['mean'].values
    stds = width_stats['std'].fillna(0).values
    counts = width_stats['count'].values
    
    # Standard error
    sems = stds / np.sqrt(counts)
    
    # Line plot with error bars
    ax.errorbar(n_sites_vals, means, yerr=sems, 
                fmt='o-', color='steelblue', markersize=10,
                linewidth=2, capsize=6, capthick=2,
                markerfacecolor='white', markeredgewidth=2)
    
    # Value labels
    for x, y, sem in zip(n_sites_vals, means, sems):
        ax.annotate(f'{y:.3f}', (x, y + sem + 0.01),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Number of Training Sites')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('CATE Bound Width vs Number of Training Sites')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show integers only
    ax.set_xticks(n_sites_vals)
    
    save_figure(fig, Path(output_dir) / filename)


def plot_width_by_n_sites_and_beta(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_by_n_sites_and_beta.png'
) -> None:
    """
    Plot bound width vs number of sites, with separate lines for each beta.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'n_sites' not in results_df.columns or 'mean_width' not in results_df.columns:
        print("Warning: Required columns not found, skipping width_by_n_sites_and_beta plot")
        return
    
    # Filter to valid data
    df = results_df[results_df['mean_width'].notna()].copy()
    if len(df) == 0:
        print("Warning: No valid data for width_by_n_sites_and_beta plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique values
    n_sites_vals = sorted(df['n_sites'].unique())
    betas = sorted(df['beta'].unique())
    
    # Colors for each beta
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(betas)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for i, beta in enumerate(betas):
        beta_data = df[df['beta'] == beta]
        
        means = []
        sems = []
        valid_n_sites = []
        
        for n_sites in n_sites_vals:
            subset = beta_data[beta_data['n_sites'] == n_sites]
            if len(subset) > 0:
                mean_val = subset['mean_width'].mean()
                std_val = subset['mean_width'].std()
                count = len(subset)
                means.append(mean_val)
                sems.append(std_val / np.sqrt(count) if count > 1 else 0)
                valid_n_sites.append(n_sites)
        
        if len(valid_n_sites) > 0:
            marker = markers[i % len(markers)]
            ax.errorbar(valid_n_sites, means, yerr=sems,
                        fmt=f'{marker}-', color=colors[i], markersize=8,
                        linewidth=2, capsize=4, capthick=1.5,
                        label=f'β={beta}', markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('Number of Training Sites')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('CATE Bound Width by Number of Sites and Confounding Strength')
    ax.legend(title='Confounding', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show integers
    ax.set_xticks(n_sites_vals)
    
    plt.tight_layout()
    save_figure(fig, Path(output_dir) / filename)


def plot_width_by_site_boxplot(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = 'width_by_site_boxplot.png'
) -> None:
    """
    Plot boxplot of bound widths grouped by number of sites.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'n_sites' not in results_df.columns or 'mean_width' not in results_df.columns:
        print("Warning: Required columns not found, skipping width_by_site_boxplot plot")
        return
    
    # Filter to valid data
    df = results_df[results_df['mean_width'].notna()].copy()
    if len(df) == 0:
        print("Warning: No valid data for width_by_site_boxplot plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique n_sites values
    n_sites_vals = sorted(df['n_sites'].unique())
    data = [df[df['n_sites'] == n]['mean_width'].dropna().tolist() for n in n_sites_vals]
    
    # Filter out empty lists
    valid_data = [(n, d) for n, d in zip(n_sites_vals, data) if len(d) > 0]
    if len(valid_data) == 0:
        print("Warning: No valid data for boxplot")
        return
    
    n_sites_vals, data = zip(*valid_data)
    
    # Boxplot
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    
    # Color boxes using gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_sites_vals)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels([str(int(n)) for n in n_sites_vals])
    ax.set_xlabel('Number of Training Sites')
    ax.set_ylabel('Mean Bound Width')
    ax.set_title('Distribution of Bound Widths by Number of Training Sites')
    ax.grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, Path(output_dir) / filename)


def plot_cate_bounds_forest(
    result: Dict[str, Any],
    output_dir: str,
    filename: str = 'cate_bounds_forest.png'
) -> None:
    """
    Plot forest-style visualization showing true CATE with bounds for each stratum.
    
    Args:
        result: Single experiment result dict with 'per_stratum_results'
        output_dir: Directory to save the figure
        filename: Output filename
    """
    if 'per_stratum_results' not in result or not result['per_stratum_results']:
        print("Warning: No per-stratum results available for forest plot")
        return
    
    per_stratum = result['per_stratum_results']
    n_strata = len(per_stratum)
    
    if n_strata == 0:
        print("Warning: Empty per-stratum results")
        return
    
    # Sort by true CATE for better visualization
    per_stratum = sorted(per_stratum, key=lambda x: x['true_cate'])
    
    fig_height = max(5, n_strata * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y_positions = np.arange(n_strata)
    
    for i, row in enumerate(per_stratum):
        true_cate = row['true_cate']
        lower = row['bound_lower']
        upper = row['bound_upper']
        is_covered = row['is_covered']
        
        # Color based on coverage
        color = '#2ecc71' if is_covered else '#e74c3c'  # Green if covered, red if not
        
        # Draw horizontal line for bounds
        ax.hlines(y=i, xmin=lower, xmax=upper, color=color, linewidth=3, alpha=0.7)
        
        # Vertical caps at ends
        cap_height = 0.25
        ax.vlines(x=lower, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)
        ax.vlines(x=upper, ymin=i - cap_height, ymax=i + cap_height, color=color, linewidth=2)
        
        # True CATE marker (diamond)
        ax.plot(true_cate, i, 'D', color='black', markersize=8, zorder=5)
    
    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # True ATE line
    true_ate = result['metrics']['true_ate']
    ax.axvline(x=true_ate, color='#3498db', linestyle=':', linewidth=2, 
               label=f'True ATE = {true_ate:.3f}')
    
    # Y-axis labels
    y_labels = [f"Z={row['z']}" for row in per_stratum]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    
    ax.set_xlabel('Treatment Effect', fontsize=11)
    ax.set_ylabel('Stratum', fontsize=11)
    
    config = result.get('config', {})
    beta = config.get('beta', '?')
    epsilon = config.get('epsilon', '?')
    coverage_rate = result['metrics'].get('cate_coverage_rate', 0)
    ax.set_title(f'CATE Bounds vs True Values (β={beta}, ε={epsilon})\n'
                 f'Coverage: {coverage_rate:.0%}', fontsize=12)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=3, label='Covered (True CATE in bounds)'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Not Covered'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black', 
               markersize=8, label='True CATE'),
        Line2D([0], [0], color='#3498db', linestyle=':', linewidth=2, label='True ATE')
    ]
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
        results: List of experiment result dicts with 'per_stratum_results'
        output_dir: Directory to save the figure
        filename: Output filename
    """
    # Filter to results with per_stratum_results
    valid_results = [r for r in results if 'per_stratum_results' in r and r['per_stratum_results']]
    
    if len(valid_results) == 0:
        print("Warning: No valid results for CATE bounds grid")
        return
    
    # Determine grid size
    n_results = len(valid_results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    
    for idx, result in enumerate(valid_results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        per_stratum = result['per_stratum_results']
        per_stratum = sorted(per_stratum, key=lambda x: x['true_cate'])
        
        n_strata = len(per_stratum)
        y_positions = np.arange(n_strata)
        
        for i, stratum in enumerate(per_stratum):
            true_cate = stratum['true_cate']
            lower = stratum['bound_lower']
            upper = stratum['bound_upper']
            is_covered = stratum['is_covered']
            
            color = '#2ecc71' if is_covered else '#e74c3c'
            
            # Bounds
            ax.hlines(y=i, xmin=lower, xmax=upper, color=color, linewidth=2, alpha=0.7)
            ax.vlines(x=lower, ymin=i - 0.2, ymax=i + 0.2, color=color, linewidth=1.5)
            ax.vlines(x=upper, ymin=i - 0.2, ymax=i + 0.2, color=color, linewidth=1.5)
            
            # True CATE
            ax.plot(true_cate, i, 'D', color='black', markersize=6, zorder=5)
        
        # Reference lines
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        true_ate = result['metrics']['true_ate']
        ax.axvline(x=true_ate, color='#3498db', linestyle=':', linewidth=1.5)
        
        # Labels
        config = result.get('config', {})
        beta = config.get('beta', '?')
        coverage = result['metrics'].get('cate_coverage_rate', 0)
        ax.set_title(f'β={beta} (Cov: {coverage:.0%})', fontsize=10)
        ax.set_xlabel('Effect', fontsize=9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Z{i}" for i in range(n_strata)], fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_results, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall legend
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=2, label='Covered'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='Not Covered'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black', 
               markersize=6, label='True CATE'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, Path(output_dir) / filename)


def generate_all_figures_synthetic(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None,
    all_results: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True
) -> List[str]:
    """Generate all figures for synthetic experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated = []
    
    if verbose:
        print(f"\nGenerating figures in {output_dir}...")
    
    # Coverage vs beta
    if 'beta' in results_df.columns and len(results_df['beta'].unique()) > 1:
        try:
            plot_coverage_vs_beta(results_df, output_dir)
            generated.append('coverage_vs_beta.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate coverage vs beta: {e}")
    
    # Width vs beta
    if 'beta' in results_df.columns and 'mean_width' in results_df.columns:
        try:
            plot_width_vs_beta(results_df, output_dir)
            generated.append('width_vs_beta.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width vs beta: {e}")
    
    # Heatmap if multiple parameters
    if ('beta' in results_df.columns and 'epsilon' in results_df.columns and 
        len(results_df['epsilon'].unique()) > 1):
        try:
            plot_coverage_heatmap_synthetic(results_df, output_dir)
            generated.append('coverage_heatmap.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate heatmap: {e}")
    
    # Width by n_sites
    if 'n_sites' in results_df.columns and 'mean_width' in results_df.columns:
        try:
            plot_width_by_n_sites(results_df, output_dir)
            generated.append('width_by_n_sites.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate width by n_sites: {e}")
        
        # Width by n_sites and beta
        if 'beta' in results_df.columns and len(results_df['n_sites'].unique()) > 1:
            try:
                plot_width_by_n_sites_and_beta(results_df, output_dir)
                generated.append('width_by_n_sites_and_beta.png')
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate width by n_sites and beta: {e}")
        
        # Width by site boxplot
        if len(results_df['n_sites'].unique()) > 1:
            try:
                plot_width_by_site_boxplot(results_df, output_dir)
                generated.append('width_by_site_boxplot.png')
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate width by site boxplot: {e}")
    
    # Forest bounds if single result provided
    if single_result is not None and 'error' not in single_result:
        try:
            plot_forest_bounds_synthetic(single_result, output_dir)
            generated.append('forest_bounds.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate forest bounds: {e}")
        
        # CATE bounds forest plot
        try:
            plot_cate_bounds_forest(single_result, output_dir)
            generated.append('cate_bounds_forest.png')
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate CATE bounds forest: {e}")
    
    # CATE bounds grid if multiple results provided
    if all_results is not None and len(all_results) > 0:
        valid_results = [r for r in all_results if 'error' not in r and 'per_stratum_results' in r]
        if len(valid_results) > 0:
            try:
                plot_cate_bounds_grid(valid_results, output_dir)
                generated.append('cate_bounds_grid.png')
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate CATE bounds grid: {e}")
    
    if verbose:
        print(f"\nGenerated {len(generated)} figures")
    
    return generated


def generate_latex_table_synthetic(
    results_df: pd.DataFrame,
    output_path: str,
    caption: str = 'Synthetic Data Experiment Results',
    label: str = 'tab:synthetic_results'
) -> str:
    """Generate LaTeX table for synthetic results."""
    cols = ['beta', 'ate_covered', 'cate_coverage_rate', 'mean_width', 'n_strata']
    available_cols = [c for c in cols if c in results_df.columns]
    
    df = results_df[available_cols].copy()
    
    # Rename columns
    rename_map = {
        'beta': r'$\beta$',
        'ate_covered': 'ATE Cov.',
        'cate_coverage_rate': 'CATE Cov.',
        'mean_width': 'Width',
        'n_strata': r'$|Z|$'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Group by beta and aggregate
    if r'$\beta$' in df.columns:
        df = df.groupby(r'$\beta$').mean().reset_index()
    
    # Generate LaTeX
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{' + 'l' + 'c' * (len(df.columns) - 1) + '}',
        r'\toprule'
    ]
    
    # Header
    header_cells = [f'\\textbf{{{col}}}' for col in df.columns]
    lines.append(' & '.join(header_cells) + r' \\')
    lines.append(r'\midrule')
    
    # Rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                cells.append('--')
            elif isinstance(val, bool):
                cells.append(r'\checkmark' if val else '')
            elif isinstance(val, float):
                cells.append(f'{val:.2f}')
            else:
                cells.append(str(val))
        lines.append(' & '.join(cells) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])
    
    latex_str = '\n'.join(lines)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX table to {output_path}")
    return latex_str


def generate_report_synthetic(
    results_df: pd.DataFrame,
    output_dir: str,
    single_result: Optional[Dict[str, Any]] = None,
    all_results: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """Generate complete report for synthetic experiments."""
    output_path = Path(output_dir)
    figures_dir = output_path / 'figures'
    tables_dir = output_path / 'tables'
    
    figures = generate_all_figures_synthetic(
        results_df, str(figures_dir), single_result, all_results, verbose
    )
    
    tables = []
    try:
        generate_latex_table_synthetic(
            results_df,
            str(tables_dir / 'results_table.tex')
        )
        tables.append('results_table.tex')
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate table: {e}")
    
    # Generate markdown summary
    md_lines = [
        '# Synthetic Data Experiment Report',
        f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '\n## Summary Statistics\n',
        f'- Total experiments: {len(results_df)}',
        f"- Mean ATE coverage: {results_df['ate_covered'].mean():.2%}",
        f"- Mean CATE coverage: {results_df['cate_coverage_rate'].mean():.2%}",
        f"- Mean bound width: {results_df['mean_width'].mean():.3f}",
    ]
    
    if 'beta' in results_df.columns:
        md_lines.extend([
            '\n## Coverage by Confounding Strength\n',
            '| Beta | ATE Cov. | CATE Cov. | Width |',
            '|------|----------|-----------|-------|'
        ])
        for beta in sorted(results_df['beta'].unique()):
            subset = results_df[results_df['beta'] == beta]
            md_lines.append(
                f"| {beta:.2f} | {subset['ate_covered'].mean():.2%} | "
                f"{subset['cate_coverage_rate'].mean():.2%} | {subset['mean_width'].mean():.3f} |"
            )
    
    md_path = output_path / 'EXPERIMENT_REPORT.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved report to {md_path}")
    
    if verbose:
        print(f"\n{'='*60}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Figures: {len(figures)} in {figures_dir}")
        print(f"Tables: {len(tables)} in {tables_dir}")
    
    return {'figures': figures, 'tables': tables}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run causal grounding experiments on synthetic data'
    )
    
    parser.add_argument('--n-sites', type=int, default=10, 
                        help='Number of training sites')
    parser.add_argument('--n-per-site', type=int, default=500,
                        help='Samples per site')
    parser.add_argument('--n-z-values', type=int, default=3,
                        help='Number of discrete covariate values')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Confounding strength')
    parser.add_argument('--treatment-effect', type=float, default=0.2,
                        help='Main treatment effect')
    parser.add_argument('--interaction', type=float, default=0.1,
                        help='Treatment-covariate interaction')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Naturalness tolerance')
    parser.add_argument('--n-permutations', type=int, default=100,
                        help='Number of permutations for CI tests')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results_synthetic',
                        help='Output directory')
    parser.add_argument('--grid', action='store_true',
                        help='Run grid of experiments')
    parser.add_argument('--all', action='store_true',
                        help='Run comprehensive sweep')
    parser.add_argument('--report', action='store_true',
                        help='Generate report after experiments')
    
    args = parser.parse_args()
    
    if args.all:
        print("=" * 60)
        print("RUNNING COMPREHENSIVE SYNTHETIC EXPERIMENTS")
        print("=" * 60)
        
        results_df, full_results = run_synthetic_grid(
            betas=BETAS,
            epsilons=[0.1],
            n_sites_list=[10],
            n_z_values=args.n_z_values,
            treatment_effect=args.treatment_effect,
            treatment_z_interaction=args.interaction,
            n_per_site=args.n_per_site,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
            output_dir=args.output,
            return_full_results=True
        )
        
        if args.report and len(results_df) > 0:
            print("\nGenerating report...")
            generate_report_synthetic(results_df, args.output, all_results=full_results)
            
    elif args.grid:
        print("=" * 60)
        print("RUNNING GRID EXPERIMENTS")
        print("=" * 60)
        
        results_df, full_results = run_synthetic_grid(
            betas=BETAS,
            epsilons=[args.epsilon],
            n_sites_list=[args.n_sites],
            n_z_values=args.n_z_values,
            treatment_effect=args.treatment_effect,
            treatment_z_interaction=args.interaction,
            n_per_site=args.n_per_site,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
            output_dir=args.output,
            return_full_results=True
        )
        
        if args.report and len(results_df) > 0:
            print("\nGenerating report...")
            generate_report_synthetic(results_df, args.output, all_results=full_results)
    else:
        # Single experiment
        result = run_synthetic_experiment(
            n_sites=args.n_sites,
            n_per_site=args.n_per_site,
            n_z_values=args.n_z_values,
            beta=args.beta,
            treatment_effect=args.treatment_effect,
            treatment_z_interaction=args.interaction,
            epsilon=args.epsilon,
            n_permutations=args.n_permutations,
            random_seed=args.seed
        )
        
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        
        if 'error' not in result:
            config = result['config']
            metrics = result['metrics']
            
            print(f"\nConfiguration:")
            print(f"  Sites: {config['n_sites']}")
            print(f"  Samples per site: {config['n_per_site']}")
            print(f"  Z values: {config['n_z_values']}")
            print(f"  Beta (confounding): {config['beta']}")
            print(f"  Treatment effect: {config['treatment_effect']}")
            print(f"  Interaction: {config['treatment_z_interaction']}")
            print(f"  Epsilon: {config['epsilon']}")
            
            print(f"\nBounds:")
            print(f"  Mean lower: {metrics['mean_lower']:.3f}")
            print(f"  Mean upper: {metrics['mean_upper']:.3f}")
            print(f"  Mean width: {metrics['mean_width']:.3f}")
            
            print(f"\nGround Truth & Coverage:")
            print(f"  True ATE: {metrics['true_ate']:.3f}")
            print(f"  ATE covered: {metrics['ate_covered']}")
            print(f"  CATE coverage rate: {metrics['cate_coverage_rate']:.1%}")
            
            print(f"\nPer-stratum Results:")
            for row in result['per_stratum_results']:
                status = "✓" if row['is_covered'] else "✗"
                print(f"  Z={row['z']}: True={row['true_cate']:.3f}, "
                      f"Bounds=[{row['bound_lower']:.3f}, {row['bound_upper']:.3f}] {status}")
            
            if args.report:
                # Create single-row DataFrame for report
                results_df = pd.DataFrame([{
                    'beta': config['beta'],
                    'epsilon': config['epsilon'],
                    'n_sites': config['n_sites'],
                    **metrics
                }])
                generate_report_synthetic(results_df, args.output, single_result=result, all_results=[result])
        else:
            print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()
