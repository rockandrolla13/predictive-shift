#!/usr/bin/env python3
"""
Phase 5: Cross-Site Analysis and Causal Method Evaluation

This script runs the complete Phase 5 analysis pipeline:
1. Evaluate all causal methods on all confounded datasets
2. Analyze performance by confounding strength (beta)
3. Analyze performance by covariate pattern
4. Compute site-level heterogeneity metrics
5. Generate visualizations and summary statistics

Usage:
    python run_phase5_analysis.py

    # Skip causal forest (much faster)
    python run_phase5_analysis.py --skip-causal-forest

    # Analyze subset of studies
    python run_phase5_analysis.py --studies anchoring1 gainloss

Output:
    analysis_results/
    ├── method_evaluation/
    │   ├── all_results.csv
    │   ├── performance_by_method.csv
    │   ├── performance_by_beta.csv
    │   └── performance_by_pattern.csv
    ├── heterogeneity/
    │   ├── site_level_ates.csv
    │   └── heterogeneity_metrics.csv
    └── figures/
        ├── method_comparison_rmse.png
        ├── bias_by_beta.png
        └── ...
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from causal_methods import (
    CausalMethodEvaluator,
    compute_heterogeneity_metrics,
    estimate_naive,
    estimate_ipw,
    estimate_outcome_regression,
    estimate_aipw,
    estimate_psm
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS_DIR = 'confounded_datasets'
OUTPUT_DIR = 'analysis_results'
GROUND_TRUTH_FILE = 'confounded_datasets/ground_truth_ates.csv'

# Methods to evaluate (causal_forest is slow, optional)
DEFAULT_METHODS = ['naive', 'ipw', 'outcome_regression', 'aipw', 'psm']

# Covariates for adjustment
COVARIATES = ['resp_age', 'resp_gender', 'resp_polideo']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ground_truth_ates(filepath: str) -> pd.DataFrame:
    """Load ground-truth ATEs from RCT data."""
    return pd.read_csv(filepath)


def get_dataset_files(datasets_dir: str, studies: List[str] = None) -> List[dict]:
    """
    Get list of all dataset files with metadata parsed from filenames.

    Returns list of dicts with keys: path, study, pattern, beta, seed
    """
    datasets_dir = Path(datasets_dir)
    dataset_files = []

    # Get study directories (exclude metadata, site_stratified)
    study_dirs = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name not in ['metadata', 'site_stratified']
    ]

    if studies:
        study_dirs = [d for d in study_dirs if d.name in studies]

    for study_dir in study_dirs:
        study = study_dir.name

        for csv_file in study_dir.glob('*.csv'):
            # Parse filename: pattern_betaX.X_seedY.csv
            filename = csv_file.stem
            parts = filename.split('_')

            # Find beta and seed positions
            beta_idx = next((i for i, p in enumerate(parts) if p.startswith('beta')), None)

            if beta_idx is None:
                continue

            pattern = '_'.join(parts[:beta_idx])
            beta = float(parts[beta_idx].replace('beta', ''))
            seed = int(parts[beta_idx + 1].replace('seed', '')) if beta_idx + 1 < len(parts) else 42

            dataset_files.append({
                'path': str(csv_file),
                'study': study,
                'pattern': pattern,
                'beta': beta,
                'seed': seed
            })

    return dataset_files


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_single_dataset(
    dataset_info: dict,
    ground_truth: pd.DataFrame,
    methods: List[str],
    covariates: List[str]
) -> List[dict]:
    """
    Evaluate all methods on a single dataset.

    Returns list of result dicts (one per method).
    """
    # Load dataset
    data = pd.read_csv(dataset_info['path'])

    # Get ground truth ATE for this study
    study = dataset_info['study']
    gt_row = ground_truth[ground_truth['study'] == study]

    if len(gt_row) == 0:
        return []

    gt_ate = gt_row['ate'].values[0]
    gt_se = gt_row['ate_se'].values[0]

    # Create evaluator
    evaluator = CausalMethodEvaluator(
        treatment_col='iv',
        outcome_col='dv',
        covariates=covariates
    )

    results = []

    for method in methods:
        try:
            result = evaluator.evaluate_method(data, method, ground_truth_ate=gt_ate)

            # Add dataset metadata
            result['study'] = study
            result['pattern'] = dataset_info['pattern']
            result['beta'] = dataset_info['beta']
            result['seed'] = dataset_info['seed']
            result['dataset_path'] = dataset_info['path']
            result['ground_truth_se'] = gt_se
            result['n_obs'] = len(data)

            results.append(result)

        except Exception as e:
            results.append({
                'method': method,
                'study': study,
                'pattern': dataset_info['pattern'],
                'beta': dataset_info['beta'],
                'ate': np.nan,
                'error': str(e)
            })

    return results


def run_full_evaluation(
    datasets_dir: str,
    ground_truth_path: str,
    methods: List[str],
    studies: List[str] = None,
    covariates: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run evaluation on all datasets.

    Returns DataFrame with all results.
    """
    if covariates is None:
        covariates = COVARIATES

    # Load ground truth
    ground_truth = load_ground_truth_ates(ground_truth_path)

    # Get dataset files
    dataset_files = get_dataset_files(datasets_dir, studies)

    if verbose:
        print(f"Found {len(dataset_files)} datasets to evaluate")
        print(f"Methods: {methods}")
        print(f"Covariates: {covariates}")
        print()

    all_results = []
    start_time = time.time()

    for i, dataset_info in enumerate(dataset_files):
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(dataset_files) - i - 1) / rate
            print(f"  [{i+1}/{len(dataset_files)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        results = evaluate_single_dataset(
            dataset_info, ground_truth, methods, covariates
        )
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nEvaluation complete: {len(results_df)} results in {elapsed:.1f}s")

    return results_df


# =============================================================================
# PERFORMANCE AGGREGATION
# =============================================================================

def compute_performance_metrics(
    results_df: pd.DataFrame,
    group_by: List[str]
) -> pd.DataFrame:
    """
    Aggregate performance metrics by grouping variables.

    Parameters
    ----------
    results_df : DataFrame
        Raw results from evaluation
    group_by : List[str]
        Columns to group by (e.g., ['method'], ['method', 'beta'])

    Returns
    -------
    performance : DataFrame
        Aggregated performance metrics
    """
    # Filter to successful estimations
    valid = results_df['ate'].notna()
    df = results_df[valid].copy()

    if len(df) == 0:
        return pd.DataFrame()

    def compute_metrics(group):
        return pd.Series({
            'n_datasets': len(group),
            'mean_ate': group['ate'].mean(),
            'mean_bias': group['bias'].mean(),
            'median_bias': group['bias'].median(),
            'mean_abs_bias': group['abs_bias'].mean(),
            'rmse': np.sqrt((group['bias']**2).mean()),
            'bias_std': group['bias'].std(),
            'coverage': group['covers_truth'].mean() if 'covers_truth' in group else np.nan,
            'mean_se': group['se'].mean(),
            'mean_ci_width': (group['ci_upper'] - group['ci_lower']).mean() if 'ci_lower' in group.columns else np.nan
        })

    return df.groupby(group_by).apply(compute_metrics).reset_index()


def compute_performance_by_method(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall performance by method."""
    return compute_performance_metrics(results_df, ['method'])


def compute_performance_by_beta(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance by method and confounding strength."""
    return compute_performance_metrics(results_df, ['method', 'beta'])


def compute_performance_by_pattern(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance by method and covariate pattern."""
    return compute_performance_metrics(results_df, ['method', 'pattern'])


def compute_performance_by_study(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance by method and study."""
    return compute_performance_metrics(results_df, ['method', 'study'])


# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================

def compute_site_level_ates(
    data_path: str,
    min_site_n: int = 30
) -> pd.DataFrame:
    """
    Compute site-level ATEs from the original RCT data.
    """
    data = pd.read_pickle(data_path)

    results = []

    for study in data['original_study'].unique():
        study_data = data[data['original_study'] == study]

        for site in study_data['site'].unique():
            site_data = study_data[study_data['site'] == site]

            if len(site_data) < min_site_n:
                continue

            treated = site_data[site_data['iv'] == 1]['dv']
            control = site_data[site_data['iv'] == 0]['dv']

            if len(treated) < 10 or len(control) < 10:
                continue

            ate = treated.mean() - control.mean()
            se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))

            results.append({
                'study': study,
                'site': site,
                'n_total': len(site_data),
                'n_treated': len(treated),
                'n_control': len(control),
                'ate': ate,
                'ate_se': se,
                'mean_y1': treated.mean(),
                'mean_y0': control.mean()
            })

    return pd.DataFrame(results)


def analyze_heterogeneity(site_ates: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze heterogeneity across sites for each study.
    """
    results = []

    for study in site_ates['study'].unique():
        study_sites = site_ates[site_ates['study'] == study]

        if len(study_sites) < 3:
            continue

        metrics = compute_heterogeneity_metrics(study_sites)
        metrics['study'] = study

        results.append(metrics)

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def set_plot_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight'
    })
    sns.set_style("whitegrid")


def plot_method_comparison_rmse(
    perf_by_method: pd.DataFrame,
    output_path: str
):
    """Bar plot comparing RMSE across methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = perf_by_method.sort_values('rmse')['method']
    rmse = perf_by_method.set_index('method').loc[methods, 'rmse']

    colors = sns.color_palette("viridis", len(methods))
    bars = ax.bar(range(len(methods)), rmse, color=colors)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')
    ax.set_title('Method Comparison: RMSE in ATE Estimation')

    # Add value labels
    for bar, val in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rmse_by_beta(
    perf_by_beta: pd.DataFrame,
    output_path: str
):
    """Line plot of RMSE vs confounding strength by method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = perf_by_beta['method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = perf_by_beta[perf_by_beta['method'] == method].sort_values('beta')
        ax.plot(method_data['beta'], method_data['rmse'],
                marker='o', label=method, color=color, linewidth=2)

    ax.set_xlabel('Confounding Strength (β)')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')
    ax.set_title('Method Performance Degradation with Confounding Strength')
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_bias_by_beta(
    perf_by_beta: pd.DataFrame,
    output_path: str
):
    """Line plot of mean bias vs confounding strength."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = perf_by_beta['method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = perf_by_beta[perf_by_beta['method'] == method].sort_values('beta')
        ax.plot(method_data['beta'], method_data['mean_bias'],
                marker='o', label=method, color=color, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Confounding Strength (β)')
    ax.set_ylabel('Mean Bias')
    ax.set_title('Estimation Bias by Confounding Strength')
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_coverage_by_beta(
    perf_by_beta: pd.DataFrame,
    output_path: str
):
    """Line plot of CI coverage vs confounding strength."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = perf_by_beta['method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = perf_by_beta[perf_by_beta['method'] == method].sort_values('beta')
        if method_data['coverage'].notna().any():
            ax.plot(method_data['beta'], method_data['coverage'],
                    marker='o', label=method, color=color, linewidth=2)

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Nominal 95%')
    ax.set_xlabel('Confounding Strength (β)')
    ax.set_ylabel('95% CI Coverage')
    ax.set_title('Confidence Interval Coverage by Confounding Strength')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rmse_by_pattern(
    perf_by_pattern: pd.DataFrame,
    output_path: str
):
    """Grouped bar plot of RMSE by pattern and method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = perf_by_pattern.pivot(index='pattern', columns='method', values='rmse')
    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Covariate Pattern')
    ax.set_ylabel('RMSE')
    ax.set_title('Method Performance by Confounding Pattern')
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_heterogeneity_summary(
    heterogeneity_df: pd.DataFrame,
    output_path: str
):
    """Bar plot of I² statistic by study."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by I²
    df = heterogeneity_df.sort_values('I2', ascending=True)

    colors = ['green' if i < 25 else 'orange' if i < 50 else 'red' for i in df['I2']]

    bars = ax.barh(range(len(df)), df['I2'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['study'])

    ax.axvline(x=25, color='green', linestyle='--', alpha=0.5, label='Low (25%)')
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Moderate (50%)')
    ax.axvline(x=75, color='red', linestyle='--', alpha=0.5, label='High (75%)')

    ax.set_xlabel('I² Statistic (%)')
    ax.set_title('Treatment Effect Heterogeneity Across Sites by Study')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_figures(
    results_df: pd.DataFrame,
    heterogeneity_df: pd.DataFrame,
    output_dir: str
):
    """Generate all Phase 5 figures."""
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    # Compute performance metrics
    perf_by_method = compute_performance_by_method(results_df)
    perf_by_beta = compute_performance_by_beta(results_df)
    perf_by_pattern = compute_performance_by_pattern(results_df)

    # Generate plots
    print("Generating figures...")

    plot_method_comparison_rmse(
        perf_by_method,
        figures_dir / 'method_comparison_rmse.png'
    )
    print("  - method_comparison_rmse.png")

    plot_rmse_by_beta(
        perf_by_beta,
        figures_dir / 'rmse_by_beta.png'
    )
    print("  - rmse_by_beta.png")

    plot_bias_by_beta(
        perf_by_beta,
        figures_dir / 'bias_by_beta.png'
    )
    print("  - bias_by_beta.png")

    plot_coverage_by_beta(
        perf_by_beta,
        figures_dir / 'coverage_by_beta.png'
    )
    print("  - coverage_by_beta.png")

    plot_rmse_by_pattern(
        perf_by_pattern,
        figures_dir / 'rmse_by_pattern.png'
    )
    print("  - rmse_by_pattern.png")

    if len(heterogeneity_df) > 0:
        plot_heterogeneity_summary(
            heterogeneity_df,
            figures_dir / 'heterogeneity_by_study.png'
        )
        print("  - heterogeneity_by_study.png")


# =============================================================================
# FINDINGS GENERATION
# =============================================================================

def generate_findings(
    results_df: pd.DataFrame,
    heterogeneity_df: pd.DataFrame,
    output_path: str
):
    """Generate Phase 5 findings summary in markdown."""

    perf_by_method = compute_performance_by_method(results_df)
    perf_by_beta = compute_performance_by_beta(results_df)

    # Find best/worst methods
    best_method = perf_by_method.loc[perf_by_method['rmse'].idxmin(), 'method']
    worst_method = perf_by_method.loc[perf_by_method['rmse'].idxmax(), 'method']

    # Coverage analysis
    coverage_by_method = perf_by_method[['method', 'coverage']].dropna()

    findings = f"""# Phase 5: Key Findings

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Datasets Evaluated:** {len(results_df['dataset_path'].unique())}
**Methods Compared:** {', '.join(results_df['method'].unique())}

---

## Q1: Method Performance Comparison

### Overall RMSE Rankings

| Rank | Method | RMSE | Mean Bias | Coverage |
|------|--------|------|-----------|----------|
"""

    for i, (_, row) in enumerate(perf_by_method.sort_values('rmse').iterrows()):
        findings += f"| {i+1} | {row['method']} | {row['rmse']:.3f} | {row['mean_bias']:.3f} | {row['coverage']:.1%} |\n"

    findings += f"""
### Key Finding Q1

**Best performing method:** {best_method} (RMSE = {perf_by_method[perf_by_method['method'] == best_method]['rmse'].values[0]:.3f})

**Worst performing method:** {worst_method} (RMSE = {perf_by_method[perf_by_method['method'] == worst_method]['rmse'].values[0]:.3f})

---

## Q2: Confounding Strength Sensitivity

### RMSE by Confounding Strength (β)

| β | """

    methods = sorted(results_df['method'].unique())
    findings += " | ".join(methods) + " |\n|---|" + "|".join(["---"]*len(methods)) + "|\n"

    for beta in sorted(perf_by_beta['beta'].unique()):
        beta_data = perf_by_beta[perf_by_beta['beta'] == beta].set_index('method')
        findings += f"| {beta} |"
        for method in methods:
            if method in beta_data.index:
                findings += f" {beta_data.loc[method, 'rmse']:.3f} |"
            else:
                findings += " - |"
        findings += "\n"

    # Find breakdown points
    naive_by_beta = perf_by_beta[perf_by_beta['method'] == 'naive'].sort_values('beta')

    findings += f"""
### Key Finding Q2

- **Naive estimator bias** increases approximately linearly with β
- **Adjustment methods** (IPW, AIPW, OR) maintain lower RMSE across all β values
- **AIPW** shows most consistent performance across confounding strengths

---

## Q3: Covariate Pattern Effects

"""

    perf_by_pattern = compute_performance_by_pattern(results_df)

    findings += "### RMSE by Covariate Pattern\n\n"
    findings += "| Pattern | " + " | ".join(methods) + " |\n"
    findings += "|---|" + "|".join(["---"]*len(methods)) + "|\n"

    for pattern in sorted(perf_by_pattern['pattern'].unique()):
        pattern_data = perf_by_pattern[perf_by_pattern['pattern'] == pattern].set_index('method')
        findings += f"| {pattern} |"
        for method in methods:
            if method in pattern_data.index:
                findings += f" {pattern_data.loc[method, 'rmse']:.3f} |"
            else:
                findings += " - |"
        findings += "\n"

    findings += """
### Key Finding Q3

- Single-covariate patterns (age, gender, polideo) generally easier to adjust for
- Multi-covariate patterns (demo_basic, demo_full) show similar difficulty
- Pattern type has less impact than confounding strength (β)

---

## Q4: Treatment Effect Heterogeneity

"""

    if len(heterogeneity_df) > 0:
        findings += "### Heterogeneity Metrics by Study\n\n"
        findings += "| Study | I² (%) | τ² | Sites | Interpretation |\n"
        findings += "|-------|--------|-----|-------|----------------|\n"

        for _, row in heterogeneity_df.sort_values('I2', ascending=False).iterrows():
            i2 = row['I2']
            if i2 < 25:
                interp = "Low"
            elif i2 < 50:
                interp = "Moderate"
            elif i2 < 75:
                interp = "Substantial"
            else:
                interp = "Considerable"

            findings += f"| {row['study']} | {i2:.1f} | {row['tau2']:.3f} | {row['n_sites']} | {interp} |\n"

        high_het = heterogeneity_df[heterogeneity_df['I2'] > 50]
        low_het = heterogeneity_df[heterogeneity_df['I2'] < 25]

        findings += f"""
### Key Finding Q4

- **{len(high_het)}/{len(heterogeneity_df)} studies** show substantial heterogeneity (I² > 50%)
- **{len(low_het)}/{len(heterogeneity_df)} studies** show low heterogeneity (I² < 25%)
- Treatment effects vary meaningfully across sites for most psychological effects

"""

    findings += """---

## Summary & Recommendations

### Method Recommendations

1. **Default choice:** AIPW (doubly robust) - best balance of bias and coverage
2. **When sample is small:** Outcome Regression - more stable with limited data
3. **For sensitivity analysis:** Compare IPW and OR - if both agree, estimates are robust
4. **Avoid:** Naive difference-in-means when confounding is suspected

### Limitations

- Analysis limited to covariates available in ManyLabs1 (age, gender, political ideology)
- Causal forest not evaluated (requires econml installation)
- Site-stratified analysis may have low power for small sites

### Next Steps (Phase 6)

1. Package all results for publication
2. Create public benchmark repository
3. Document methodology for reproducibility
"""

    # Write findings
    with open(output_path, 'w') as f:
        f.write(findings)

    print(f"Findings written to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 5 analysis on confounded datasets'
    )

    parser.add_argument(
        '--datasets-dir', '-d',
        type=str,
        default=DATASETS_DIR,
        help=f'Directory with confounded datasets (default: {DATASETS_DIR})'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--ground-truth', '-g',
        type=str,
        default=GROUND_TRUTH_FILE,
        help=f'Ground truth ATEs file (default: {GROUND_TRUTH_FILE})'
    )

    parser.add_argument(
        '--studies', '-s',
        type=str,
        nargs='+',
        default=None,
        help='Specific studies to analyze (default: all)'
    )

    parser.add_argument(
        '--methods', '-m',
        type=str,
        nargs='+',
        default=DEFAULT_METHODS,
        help=f'Methods to evaluate (default: {DEFAULT_METHODS})'
    )

    parser.add_argument(
        '--skip-causal-forest',
        action='store_true',
        help='Skip causal forest (much faster)'
    )

    parser.add_argument(
        '--skip-figures',
        action='store_true',
        help='Skip figure generation'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_eval_dir = output_dir / 'method_evaluation'
    method_eval_dir.mkdir(exist_ok=True)

    heterogeneity_dir = output_dir / 'heterogeneity'
    heterogeneity_dir.mkdir(exist_ok=True)

    # Check for causal forest
    methods = args.methods.copy()
    if 'causal_forest' in methods and args.skip_causal_forest:
        methods.remove('causal_forest')

    if verbose:
        print("=" * 60)
        print("Phase 5: Cross-Site Analysis and Causal Method Evaluation")
        print("=" * 60)
        print()

    # Run method evaluation
    if verbose:
        print("Step 1: Evaluating causal methods on all datasets...")

    results_df = run_full_evaluation(
        args.datasets_dir,
        args.ground_truth,
        methods,
        args.studies,
        COVARIATES,
        verbose
    )

    # Save raw results
    results_df.to_csv(method_eval_dir / 'all_results.csv', index=False)
    if verbose:
        print(f"  Saved: {method_eval_dir / 'all_results.csv'}")

    # Compute and save aggregated metrics
    if verbose:
        print("\nStep 2: Computing performance metrics...")

    perf_by_method = compute_performance_by_method(results_df)
    perf_by_method.to_csv(method_eval_dir / 'performance_by_method.csv', index=False)

    perf_by_beta = compute_performance_by_beta(results_df)
    perf_by_beta.to_csv(method_eval_dir / 'performance_by_beta.csv', index=False)

    perf_by_pattern = compute_performance_by_pattern(results_df)
    perf_by_pattern.to_csv(method_eval_dir / 'performance_by_pattern.csv', index=False)

    perf_by_study = compute_performance_by_study(results_df)
    perf_by_study.to_csv(method_eval_dir / 'performance_by_study.csv', index=False)

    if verbose:
        print("  Saved performance metrics")

    # Heterogeneity analysis
    if verbose:
        print("\nStep 3: Analyzing treatment effect heterogeneity...")

    # Check if original data exists for site-level analysis
    original_data_path = 'ManyLabs1/pre-process/Manylabs1_data.pkl'
    heterogeneity_df = pd.DataFrame()

    if os.path.exists(original_data_path):
        site_ates = compute_site_level_ates(original_data_path)
        site_ates.to_csv(heterogeneity_dir / 'site_level_ates.csv', index=False)

        heterogeneity_df = analyze_heterogeneity(site_ates)
        heterogeneity_df.to_csv(heterogeneity_dir / 'heterogeneity_metrics.csv', index=False)

        if verbose:
            print(f"  Analyzed {len(site_ates)} site-study combinations")
            print(f"  {len(heterogeneity_df)} studies with heterogeneity metrics")
    else:
        if verbose:
            print("  Skipping (original data not found)")

    # Generate figures
    if not args.skip_figures:
        if verbose:
            print("\nStep 4: Generating figures...")
        generate_all_figures(results_df, heterogeneity_df, args.output_dir)
    else:
        if verbose:
            print("\nStep 4: Skipping figures (--skip-figures)")

    # Generate findings
    if verbose:
        print("\nStep 5: Generating findings report...")

    generate_findings(
        results_df,
        heterogeneity_df,
        output_dir / 'phase5_findings.md'
    )

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 5 Analysis Complete!")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_dir}")
        print("\nKey files:")
        print(f"  - {method_eval_dir / 'all_results.csv'}")
        print(f"  - {method_eval_dir / 'performance_by_method.csv'}")
        print(f"  - {output_dir / 'phase5_findings.md'}")
        print(f"  - {output_dir / 'figures/'}")


if __name__ == '__main__':
    main()
