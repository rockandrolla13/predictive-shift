"""
Visualization Module for Causal Grounding

This module provides plotting functions for visualizing:
- CATE bounds (error bars, forest plots, comparisons)
- Coverage analysis (by stratum, heatmaps)
- CI test diagnostics (EHS scores, CMI distributions)
- Method comparisons (summary charts, runtime)

All plots use matplotlib and follow a consistent style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Union, Any, Tuple

# Style configuration
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#bcbd22',
    'info': '#17becf',
    'gray': '#7f7f7f',
}

STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
}


def _apply_style():
    """Apply consistent matplotlib style."""
    for key, value in STYLE.items():
        plt.rcParams[key] = value


# =============================================================================
# BOUNDS VISUALIZATION
# =============================================================================

def plot_cate_bounds(
    bounds_df: pd.DataFrame,
    true_cate: Optional[float] = None,
    title: Optional[str] = None,
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    stratum_col: Optional[str] = None,
    sort_by: str = 'lower',
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot CATE bounds as error bars for each stratum.

    Args:
        bounds_df: DataFrame with lower and upper bound columns
        true_cate: Optional true CATE value to show as reference line
        title: Plot title
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound
        stratum_col: Column name for stratum labels (uses index if None)
        sort_by: Sort strata by 'lower', 'upper', 'width', or 'none'
        figsize: Figure size tuple
        ax: Optional axes to plot on

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    df = bounds_df.copy()

    # Get stratum labels
    if stratum_col and stratum_col in df.columns:
        labels = df[stratum_col].astype(str).values
    else:
        labels = [str(i) for i in range(len(df))]

    # Sort if requested
    if sort_by != 'none':
        if sort_by == 'lower':
            order = np.argsort(df[lower_col].values)
        elif sort_by == 'upper':
            order = np.argsort(df[upper_col].values)
        elif sort_by == 'width':
            order = np.argsort((df[upper_col] - df[lower_col]).values)
        else:
            order = np.arange(len(df))
        df = df.iloc[order].reset_index(drop=True)
        labels = [labels[i] for i in order]

    lower = df[lower_col].values
    upper = df[upper_col].values
    midpoints = (lower + upper) / 2
    errors = np.vstack([midpoints - lower, upper - midpoints])

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(df))

    # Plot error bars
    ax.errorbar(
        x, midpoints, yerr=errors,
        fmt='o', color=COLORS['primary'],
        capsize=3, capthick=1, markersize=4
    )

    # Add true CATE line if provided
    if true_cate is not None:
        ax.axhline(y=true_cate, color=COLORS['danger'], linestyle='--',
                   linewidth=2, label=f'True CATE = {true_cate:.2f}')
        ax.legend()

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Stratum')
    ax.set_ylabel('CATE Bounds')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('CATE Bounds by Stratum')

    plt.tight_layout()
    return fig


def plot_bounds_forest(
    bounds_df: pd.DataFrame,
    top_k: int = 10,
    sort_by: str = 'lower',
    true_cate: Optional[float] = None,
    title: Optional[str] = None,
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    stratum_col: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create forest plot style visualization of bounds.

    Args:
        bounds_df: DataFrame with bounds
        top_k: Number of top strata to show
        sort_by: Sort by 'lower', 'upper', or 'width'
        true_cate: Optional true CATE reference line
        title: Plot title
        lower_col, upper_col, stratum_col: Column names
        figsize: Figure size
        ax: Optional axes

    Returns:
        matplotlib Figure
    """
    _apply_style()

    df = bounds_df.copy()

    # Sort and select top k
    if sort_by == 'lower':
        df = df.sort_values(lower_col, ascending=False)
    elif sort_by == 'upper':
        df = df.sort_values(upper_col, ascending=False)
    elif sort_by == 'width':
        df['_width'] = df[upper_col] - df[lower_col]
        df = df.sort_values('_width', ascending=True)

    df = df.head(top_k).reset_index(drop=True)

    # Get labels
    if stratum_col and stratum_col in df.columns:
        labels = df[stratum_col].astype(str).values
    else:
        labels = [f'Stratum {i}' for i in range(len(df))]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(df))
    lower = df[lower_col].values
    upper = df[upper_col].values
    midpoints = (lower + upper) / 2

    # Plot horizontal lines for bounds
    for i in range(len(df)):
        ax.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]],
                color=COLORS['primary'], linewidth=2)
        ax.plot(midpoints[i], y_pos[i], 'o', color=COLORS['primary'], markersize=6)

    # Add true CATE line
    if true_cate is not None:
        ax.axvline(x=true_cate, color=COLORS['danger'], linestyle='--',
                   linewidth=2, label=f'True CATE = {true_cate:.2f}')
        ax.legend(loc='lower right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('CATE')
    ax.set_ylabel('Stratum')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Forest Plot: Top {top_k} Strata')

    ax.invert_yaxis()  # Highest at top
    plt.tight_layout()
    return fig


def plot_bounds_comparison(
    bounds_dict: Dict[str, pd.DataFrame],
    labels: Optional[List[str]] = None,
    true_cate: Optional[float] = None,
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Compare bounds from multiple methods side by side.

    Args:
        bounds_dict: Dict mapping method name to bounds DataFrame
        labels: Optional list of method labels
        true_cate: Optional true CATE reference
        lower_col, upper_col: Column names
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    methods = list(bounds_dict.keys())
    n_methods = len(methods)
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
              COLORS['warning'], COLORS['info']][:n_methods]

    fig, ax = plt.subplots(figsize=figsize)

    # Assume all bounds have same number of strata
    n_strata = len(list(bounds_dict.values())[0])
    x = np.arange(n_strata)
    width = 0.8 / n_methods

    for i, (method, bounds_df) in enumerate(bounds_dict.items()):
        lower = bounds_df[lower_col].values
        upper = bounds_df[upper_col].values
        midpoints = (lower + upper) / 2
        errors = np.vstack([midpoints - lower, upper - midpoints])

        offset = (i - n_methods/2 + 0.5) * width
        ax.errorbar(
            x + offset, midpoints, yerr=errors,
            fmt='o', color=colors[i], capsize=2, capthick=1,
            markersize=3, label=labels[i] if labels else method
        )

    if true_cate is not None:
        ax.axhline(y=true_cate, color=COLORS['danger'], linestyle='--',
                   linewidth=2, label=f'True = {true_cate:.2f}')

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i}' for i in range(n_strata)], rotation=45)
    ax.set_xlabel('Stratum')
    ax.set_ylabel('CATE Bounds')
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Bounds Comparison Across Methods')

    plt.tight_layout()
    return fig


# =============================================================================
# COVERAGE VISUALIZATION
# =============================================================================

def plot_coverage_by_stratum(
    bounds_df: pd.DataFrame,
    ground_truth: Union[float, np.ndarray],
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    stratum_col: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot coverage status (hit/miss) for each stratum.

    Args:
        bounds_df: DataFrame with bounds
        ground_truth: True CATE value(s)
        lower_col, upper_col, stratum_col: Column names
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    df = bounds_df.copy()
    lower = df[lower_col].values
    upper = df[upper_col].values

    if isinstance(ground_truth, (int, float)):
        truth = np.full(len(df), ground_truth)
    else:
        truth = np.asarray(ground_truth)

    covered = (lower <= truth) & (truth <= upper)

    # Get labels
    if stratum_col and stratum_col in df.columns:
        labels = df[stratum_col].astype(str).values
    else:
        labels = [f'S{i}' for i in range(len(df))]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    colors_list = [COLORS['success'] if c else COLORS['danger'] for c in covered]

    # Plot bounds as bars
    widths = upper - lower
    ax.bar(x, widths, bottom=lower, color=colors_list, alpha=0.7, edgecolor='black')

    # Add truth line
    if isinstance(ground_truth, (int, float)):
        ax.axhline(y=ground_truth, color='black', linestyle='--',
                   linewidth=2, label=f'Truth = {ground_truth:.2f}')
    else:
        ax.scatter(x, truth, color='black', marker='*', s=50, zorder=5, label='Truth')

    # Legend
    covered_patch = mpatches.Patch(color=COLORS['success'], label='Covered', alpha=0.7)
    missed_patch = mpatches.Patch(color=COLORS['danger'], label='Missed', alpha=0.7)
    ax.legend(handles=[covered_patch, missed_patch], loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Stratum')
    ax.set_ylabel('CATE')

    coverage_rate = np.mean(covered)
    if title:
        ax.set_title(f'{title} (Coverage: {coverage_rate:.1%})')
    else:
        ax.set_title(f'Coverage by Stratum (Rate: {coverage_rate:.1%})')

    plt.tight_layout()
    return fig


def plot_coverage_heatmap(
    coverage_matrix: np.ndarray,
    method_names: List[str],
    stratum_names: List[str],
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot coverage heatmap across methods and strata.

    Args:
        coverage_matrix: 2D array (methods x strata) of coverage indicators
        method_names: List of method names
        stratum_names: List of stratum names
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(stratum_names)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(stratum_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(method_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coverage')

    # Add text annotations
    for i in range(len(method_names)):
        for j in range(len(stratum_names)):
            text = 'Y' if coverage_matrix[i, j] > 0.5 else 'N'
            color = 'white' if coverage_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Coverage Heatmap: Methods x Strata')

    plt.tight_layout()
    return fig


# =============================================================================
# WIDTH / INFORMATIVENESS
# =============================================================================

def plot_width_distribution(
    bounds_df: pd.DataFrame,
    bins: int = 20,
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of bound widths.

    Args:
        bounds_df: DataFrame with bounds
        bins: Number of histogram bins
        lower_col, upper_col: Column names
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    widths = bounds_df[upper_col] - bounds_df[lower_col]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(widths, bins=bins, color=COLORS['primary'], alpha=0.7, edgecolor='black')

    # Add statistics
    mean_w = widths.mean()
    median_w = widths.median()
    ax.axvline(mean_w, color=COLORS['danger'], linestyle='--',
               label=f'Mean: {mean_w:.2f}')
    ax.axvline(median_w, color=COLORS['warning'], linestyle=':',
               label=f'Median: {median_w:.2f}')

    ax.set_xlabel('Bound Width')
    ax.set_ylabel('Frequency')
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Distribution of Bound Widths')

    plt.tight_layout()
    return fig


def plot_width_vs_sample_size(
    bounds_df: pd.DataFrame,
    sample_sizes: np.ndarray,
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of bound width vs sample size.

    Args:
        bounds_df: DataFrame with bounds
        sample_sizes: Array of sample sizes per stratum
        lower_col, upper_col: Column names
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    widths = bounds_df[upper_col] - bounds_df[lower_col]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(sample_sizes, widths, alpha=0.6, color=COLORS['primary'])

    # Add trend line
    z = np.polyfit(sample_sizes, widths, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample_sizes.min(), sample_sizes.max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['danger'], linestyle='--',
            label='Trend')

    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Bound Width')
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Bound Width vs Sample Size')

    plt.tight_layout()
    return fig


# =============================================================================
# CI TEST DIAGNOSTICS
# =============================================================================

def plot_ehs_scores(
    scores_df: pd.DataFrame,
    highlight_best: bool = True,
    score_col: str = 'score',
    covariate_col: str = 'z_a',
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart of EHS scores for covariates.

    Args:
        scores_df: DataFrame with EHS scores
        highlight_best: Highlight the best scoring covariate
        score_col: Column name for score
        covariate_col: Column name for covariate identifier
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    df = scores_df.sort_values(score_col, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    scores = df[score_col].values

    if covariate_col in df.columns:
        labels = df[covariate_col].astype(str).values
    else:
        labels = df.index.astype(str).values

    colors_list = [COLORS['success'] if i == 0 and highlight_best else COLORS['primary']
                   for i in range(len(df))]

    ax.barh(x, scores, color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel('EHS Score')
    ax.set_ylabel('Covariate')
    ax.invert_yaxis()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('EHS Scores by Covariate (Top 10)')

    plt.tight_layout()
    return fig


def plot_cmi_distribution(
    null_dist: np.ndarray,
    observed: float,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot CMI permutation test null distribution with observed value.

    Args:
        null_dist: Array of CMI values under null hypothesis
        observed: Observed CMI value
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(null_dist, bins=30, color=COLORS['gray'], alpha=0.7,
            edgecolor='black', label='Null Distribution')

    ax.axvline(observed, color=COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Observed = {observed:.4f}')

    p_value = (np.sum(null_dist >= observed) + 1) / (len(null_dist) + 1)
    ax.text(0.95, 0.95, f'p = {p_value:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('CMI Value')
    ax.set_ylabel('Frequency')
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('CMI Permutation Test')

    plt.tight_layout()
    return fig


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_method_comparison_summary(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Summary comparison plot showing multiple metrics across methods.

    Args:
        results_dict: Dict mapping method name to dict of metrics
        metrics: List of metric names to plot (uses all if None)
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
        # Filter to numeric metrics
        metrics = [m for m in metrics if isinstance(
            list(results_dict.values())[0].get(m), (int, float, np.floating)
        )]

    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
                   COLORS['warning'], COLORS['info']][:len(methods)]

    for idx, metric in enumerate(metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        values = [results_dict[m].get(metric, 0) for m in methods]
        x = np.arange(len(methods))

        ax.bar(x, values, color=colors_list, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(metric)

    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, y=1.02)
    else:
        fig.suptitle('Method Comparison Summary', y=1.02)

    plt.tight_layout()
    return fig


def plot_runtime_comparison(
    timing_results: Dict[str, float],
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing runtime across methods.

    Args:
        timing_results: Dict mapping method name to runtime (seconds)
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    methods = list(timing_results.keys())
    times = list(timing_results.values())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(methods))
    colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

    ax.barh(x, times, color=colors_list, alpha=0.7, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Runtime (seconds)')
    ax.invert_yaxis()

    # Add value labels
    for i, t in enumerate(times):
        ax.text(t + max(times)*0.01, i, f'{t:.2f}s', va='center')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Runtime Comparison')

    plt.tight_layout()
    return fig


def plot_agreement_matrix(
    decisions1: np.ndarray,
    decisions2: np.ndarray,
    labels: Optional[List[str]] = None,
    method1_name: str = 'Method 1',
    method2_name: str = 'Method 2',
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot agreement matrix between two methods' decisions.

    Args:
        decisions1: Boolean array of decisions from method 1
        decisions2: Boolean array of decisions from method 2
        labels: Optional labels for items
        method1_name, method2_name: Names for methods
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _apply_style()

    # Compute agreement
    both_true = np.sum(decisions1 & decisions2)
    both_false = np.sum(~decisions1 & ~decisions2)
    m1_only = np.sum(decisions1 & ~decisions2)
    m2_only = np.sum(~decisions1 & decisions2)

    matrix = np.array([[both_true, m1_only], [m2_only, both_false]])

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Reject', 'Accept'])
    ax.set_yticklabels(['Reject', 'Accept'])
    ax.set_xlabel(method2_name)
    ax.set_ylabel(method1_name)

    # Add text
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black',
                    fontsize=14)

    agreement = (both_true + both_false) / len(decisions1)
    if title:
        ax.set_title(f'{title} (Agreement: {agreement:.1%})')
    else:
        ax.set_title(f'Decision Agreement (Rate: {agreement:.1%})')

    plt.tight_layout()
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 150,
    bbox_inches: str = 'tight'
) -> None:
    """Save figure to file."""
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def create_multi_panel_figure(
    n_panels: int,
    figsize: Optional[Tuple[float, float]] = None,
    n_cols: int = 2
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a multi-panel figure.

    Args:
        n_panels: Number of panels
        figsize: Figure size (auto-computed if None)
        n_cols: Number of columns

    Returns:
        Tuple of (Figure, list of Axes)
    """
    n_rows = (n_panels + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:n_panels]


# Module test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 20

    bounds = pd.DataFrame({
        'stratum': [f'S{i}' for i in range(n)],
        'lower': 1000 + np.random.uniform(-200, 100, n),
        'upper': 1600 + np.random.uniform(-100, 200, n)
    })

    true_ate = 1550

    # Test plots
    print("Creating test plots...")

    fig1 = plot_cate_bounds(bounds, true_cate=true_ate, stratum_col='stratum')
    plt.savefig('/tmp/test_cate_bounds.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  - Created CATE bounds plot")

    fig2 = plot_bounds_forest(bounds, top_k=10, true_cate=true_ate, stratum_col='stratum')
    plt.savefig('/tmp/test_forest.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  - Created forest plot")

    fig3 = plot_coverage_by_stratum(bounds, true_ate, stratum_col='stratum')
    plt.savefig('/tmp/test_coverage.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  - Created coverage plot")

    fig4 = plot_width_distribution(bounds)
    plt.savefig('/tmp/test_width_dist.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  - Created width distribution plot")

    print("\nTest plots saved to /tmp/")
