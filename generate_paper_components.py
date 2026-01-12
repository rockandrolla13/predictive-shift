#!/usr/bin/env python3
"""
Phase 6.3: Generate Research Paper Components

This script generates:
1. Publication-quality figures (PDF/PNG at 300 DPI)
2. LaTeX tables for paper
3. Supplementary materials

Usage:
    python generate_paper_components.py --output-dir paper_components
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent
BENCHMARK_DIR = PROJECT_ROOT / "osrct_benchmark_v1.0"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Figure size (single column: 3.5", double column: 7")
        'figure.figsize': (7, 5),
    })

    # Color palette for methods
    return {
        'naive': '#E74C3C',        # Red
        'ipw': '#3498DB',          # Blue
        'outcome_regression': '#2ECC71',  # Green
        'aipw': '#9B59B6',         # Purple
        'psm': '#F39C12',          # Orange
        'causal_forest': '#1ABC9C' # Teal
    }


def load_analysis_results(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all analysis result files."""
    results = {}

    method_eval_dir = analysis_dir / "method_evaluation"
    if method_eval_dir.exists():
        for csv_file in method_eval_dir.glob("*.csv"):
            results[csv_file.stem] = pd.read_csv(csv_file)

    return results


def load_ground_truth(benchmark_dir: Path) -> pd.DataFrame:
    """Load ground truth ATEs."""
    gt_path = benchmark_dir / "ground_truth" / "rct_ates.csv"
    if gt_path.exists():
        return pd.read_csv(gt_path)
    return None


def figure_1_method_comparison(results: Dict, colors: Dict, output_dir: Path):
    """
    Figure 1: Method Performance Comparison
    Bar chart showing RMSE for each method.
    """
    print("  Generating Figure 1: Method Comparison...")

    if 'performance_by_method' not in results:
        print("    Skipping: performance_by_method not found")
        return

    df = results['performance_by_method']

    # Sort by RMSE
    df_sorted = df.sort_values('rmse')

    fig, ax = plt.subplots(figsize=(6, 4))

    methods = df_sorted['method'].values
    rmse = df_sorted['rmse'].values
    bar_colors = [colors.get(m, '#888888') for m in methods]

    bars = ax.barh(methods, rmse, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, rmse):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)

    ax.set_xlabel('Root Mean Squared Error (RMSE)')
    ax.set_title('Causal Method Performance Comparison')
    ax.set_xlim(0, max(rmse) * 1.15)

    # Clean up method names for display
    ax.set_yticklabels([m.replace('_', ' ').title() for m in methods])

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(output_dir / 'fig1_method_comparison.pdf')
    fig.savefig(output_dir / 'fig1_method_comparison.png', dpi=300)
    plt.close(fig)


def figure_2_rmse_by_beta(results: Dict, colors: Dict, output_dir: Path):
    """
    Figure 2: RMSE by Confounding Strength
    Line plot showing how RMSE changes with beta for each method.
    """
    print("  Generating Figure 2: RMSE by Confounding Strength...")

    if 'performance_by_beta' not in results:
        print("    Skipping: performance_by_beta not found")
        return

    df = results['performance_by_beta']

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = df['method'].unique()

    for method in methods:
        method_data = df[df['method'] == method].sort_values('beta')
        ax.plot(method_data['beta'], method_data['rmse'],
                marker='o', label=method.replace('_', ' ').title(),
                color=colors.get(method, '#888888'),
                linewidth=2, markersize=6)

    ax.set_xlabel('Confounding Strength (β)')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')
    ax.set_title('Method Performance Across Confounding Strengths')

    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_rmse_by_beta.pdf')
    fig.savefig(output_dir / 'fig2_rmse_by_beta.png', dpi=300)
    plt.close(fig)


def figure_3_bias_by_beta(results: Dict, colors: Dict, output_dir: Path):
    """
    Figure 3: Bias by Confounding Strength
    Shows how bias evolves with confounding for each method.
    """
    print("  Generating Figure 3: Bias by Confounding Strength...")

    if 'performance_by_beta' not in results:
        print("    Skipping: performance_by_beta not found")
        return

    df = results['performance_by_beta']

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = df['method'].unique()

    for method in methods:
        method_data = df[df['method'] == method].sort_values('beta')
        ax.plot(method_data['beta'], method_data['mean_bias'],
                marker='s', label=method.replace('_', ' ').title(),
                color=colors.get(method, '#888888'),
                linewidth=2, markersize=6)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Confounding Strength (β)')
    ax.set_ylabel('Mean Bias')
    ax.set_title('Estimation Bias Across Confounding Strengths')

    ax.legend(loc='best', framealpha=0.9)
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_bias_by_beta.pdf')
    fig.savefig(output_dir / 'fig3_bias_by_beta.png', dpi=300)
    plt.close(fig)


def figure_4_heterogeneity(ground_truth: pd.DataFrame, output_dir: Path):
    """
    Figure 4: Treatment Effect Heterogeneity Across Studies
    Forest plot style showing ATEs with confidence intervals.
    """
    print("  Generating Figure 4: Treatment Effect Heterogeneity...")

    if ground_truth is None:
        print("    Skipping: ground truth not found")
        return

    # Normalize ATEs for visualization (use standardized effects)
    gt = ground_truth.copy()
    gt['ate_std'] = gt['ate'] / gt['std_y1'].replace(0, 1)
    gt['se_std'] = gt['ate_se'] / gt['std_y1'].replace(0, 1)

    # Sort by effect size
    gt = gt.sort_values('ate_std')

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(gt))
    studies = gt['study'].values
    ates = gt['ate_std'].values
    ses = gt['se_std'].values

    # Plot points with error bars
    ax.errorbar(ates, y_pos, xerr=1.96*ses,
                fmt='o', color='#3498DB', capsize=3,
                markersize=6, linewidth=1.5)

    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies)
    ax.set_xlabel('Standardized Treatment Effect (Cohen\'s d)')
    ax.set_title('Treatment Effects Across ManyLabs1 Studies')

    # Add effect size interpretation
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small effect')
    ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Medium effect')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_heterogeneity.pdf')
    fig.savefig(output_dir / 'fig4_heterogeneity.png', dpi=300)
    plt.close(fig)


def figure_5_pattern_comparison(results: Dict, colors: Dict, output_dir: Path):
    """
    Figure 5: Performance by Covariate Pattern
    Grouped bar chart comparing methods across patterns.
    """
    print("  Generating Figure 5: Pattern Comparison...")

    if 'performance_by_pattern' not in results:
        print("    Skipping: performance_by_pattern not found")
        return

    df = results['performance_by_pattern']

    methods = df['method'].unique()
    patterns = df['pattern'].unique()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(patterns))
    width = 0.15
    n_methods = len(methods)

    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        rmse_vals = [method_data[method_data['pattern'] == p]['rmse'].values[0]
                     if len(method_data[method_data['pattern'] == p]) > 0 else 0
                     for p in patterns]

        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, rmse_vals, width,
               label=method.replace('_', ' ').title(),
               color=colors.get(method, '#888888'),
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Covariate Pattern')
    ax.set_ylabel('RMSE')
    ax.set_title('Method Performance by Confounding Pattern')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in patterns])
    ax.legend(loc='upper right', ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_pattern_comparison.pdf')
    fig.savefig(output_dir / 'fig5_pattern_comparison.png', dpi=300)
    plt.close(fig)


def generate_latex_tables(results: Dict, ground_truth: pd.DataFrame, output_dir: Path):
    """Generate LaTeX tables for the paper."""
    print("\nGenerating LaTeX tables...")

    # Table 1: Study Summary
    table1_study_summary(ground_truth, output_dir)

    # Table 2: Method Performance
    table2_method_performance(results, output_dir)

    # Table 3: Performance by Beta
    table3_performance_by_beta(results, output_dir)

    # Table 4: Confounding Patterns
    table4_confounding_patterns(output_dir)


def table1_study_summary(ground_truth: pd.DataFrame, output_dir: Path):
    """Table 1: Summary of ManyLabs1 Studies."""
    print("  Generating Table 1: Study Summary...")

    if ground_truth is None:
        return

    latex = r"""\begin{table}[htbp]
\centering
\caption{Summary of ManyLabs1 Studies Included in OSRCT Benchmark}
\label{tab:studies}
\begin{tabular}{lrrrrr}
\toprule
Study & N & Treated & Control & ATE & SE \\
\midrule
"""

    for _, row in ground_truth.iterrows():
        study = row['study'].replace('_', r'\_')
        n_total = int(row['n_total'])
        n_treated = int(row['n_treated'])
        n_control = int(row['n_control'])
        ate = row['ate']
        se = row['ate_se']

        # Format ATE based on magnitude
        if abs(ate) > 100:
            ate_str = f"{ate:.0f}"
            se_str = f"{se:.0f}"
        elif abs(ate) > 1:
            ate_str = f"{ate:.2f}"
            se_str = f"{se:.2f}"
        else:
            ate_str = f"{ate:.3f}"
            se_str = f"{se:.3f}"

        latex += f"{study} & {n_total} & {n_treated} & {n_control} & {ate_str} & {se_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'table1_study_summary.tex', 'w') as f:
        f.write(latex)


def table2_method_performance(results: Dict, output_dir: Path):
    """Table 2: Overall Method Performance."""
    print("  Generating Table 2: Method Performance...")

    if 'performance_by_method' not in results:
        return

    df = results['performance_by_method'].sort_values('rmse')

    latex = r"""\begin{table}[htbp]
\centering
\caption{Causal Method Performance on OSRCT Benchmark}
\label{tab:method_performance}
\begin{tabular}{lccccc}
\toprule
Method & RMSE & Bias & Abs. Bias & Coverage & CI Width \\
\midrule
"""

    for _, row in df.iterrows():
        method = row['method'].replace('_', ' ').title()
        rmse = f"{row['rmse']:.2f}"
        bias = f"{row['mean_bias']:.2f}"
        abs_bias = f"{row['mean_abs_bias']:.2f}"

        # Handle optional columns
        coverage = f"{row.get('coverage', 0)*100:.1f}\\%" if 'coverage' in row else "N/A"
        ci_width = f"{row.get('mean_ci_width', 0):.2f}" if 'mean_ci_width' in row else "N/A"

        latex += f"{method} & {rmse} & {bias} & {abs_bias} & {coverage} & {ci_width} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: RMSE = Root Mean Squared Error. Coverage = proportion of 95\% CIs containing true ATE.
\end{tablenotes}
\end{table}
"""

    with open(output_dir / 'table2_method_performance.tex', 'w') as f:
        f.write(latex)


def table3_performance_by_beta(results: Dict, output_dir: Path):
    """Table 3: Method Performance by Confounding Strength."""
    print("  Generating Table 3: Performance by Beta...")

    if 'performance_by_beta' not in results:
        return

    df = results['performance_by_beta']

    # Pivot to get methods as rows, betas as columns
    pivot = df.pivot(index='method', columns='beta', values='rmse')

    latex = r"""\begin{table}[htbp]
\centering
\caption{Method RMSE by Confounding Strength ($\beta$)}
\label{tab:performance_by_beta}
\begin{tabular}{l"""

    betas = sorted(pivot.columns)
    latex += "c" * len(betas)
    latex += "}\n\\toprule\nMethod"

    for beta in betas:
        latex += f" & $\\beta$={beta}"
    latex += " \\\\\n\\midrule\n"

    for method in pivot.index:
        method_name = method.replace('_', ' ').title()
        latex += method_name
        for beta in betas:
            val = pivot.loc[method, beta]
            latex += f" & {val:.1f}"
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'table3_performance_by_beta.tex', 'w') as f:
        f.write(latex)


def table4_confounding_patterns(output_dir: Path):
    """Table 4: Confounding Pattern Definitions."""
    print("  Generating Table 4: Confounding Patterns...")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Confounding Patterns in OSRCT Benchmark}
\label{tab:patterns}
\begin{tabular}{llp{6cm}}
\toprule
Pattern & Type & Covariates \\
\midrule
age & Single-covariate & Participant age (continuous) \\
gender & Single-covariate & Participant gender (binary) \\
polideo & Single-covariate & Political ideology (ordinal, 0-6) \\
demo\_basic & Multi-covariate & Age + Gender \\
demo\_full & Multi-covariate & Age + Gender + Political ideology \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'table4_confounding_patterns.tex', 'w') as f:
        f.write(latex)


def generate_supplementary_materials(results: Dict, ground_truth: pd.DataFrame, output_dir: Path):
    """Generate supplementary materials document."""
    print("\nGenerating supplementary materials...")

    supp_content = f"""# OSRCT Benchmark - Supplementary Materials

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## S1. Complete Study Results

### S1.1 Ground Truth ATEs by Study

"""

    if ground_truth is not None:
        supp_content += "| Study | N | True ATE | SE | 95% CI |\n"
        supp_content += "|-------|---|----------|----|---------|\n"

        for _, row in ground_truth.iterrows():
            ci_lower = row['ate_ci_lower']
            ci_upper = row['ate_ci_upper']
            supp_content += f"| {row['study']} | {int(row['n_total'])} | {row['ate']:.3f} | {row['ate_se']:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] |\n"

    supp_content += """

### S1.2 Method Performance by Study

"""

    if 'performance_by_study' in results:
        df = results['performance_by_study']
        supp_content += "| Study | Method | RMSE | Bias |\n"
        supp_content += "|-------|--------|------|------|\n"

        for study in df['study'].unique():
            study_data = df[df['study'] == study].sort_values('rmse')
            for _, row in study_data.iterrows():
                supp_content += f"| {study} | {row['method']} | {row['rmse']:.2f} | {row['mean_bias']:.2f} |\n"

    supp_content += """

---

## S2. Complete Performance Tables

### S2.1 Full Results by Beta

"""

    if 'performance_by_beta' in results:
        df = results['performance_by_beta']
        supp_content += "| Beta | Method | RMSE | Bias | Abs Bias |\n"
        supp_content += "|------|--------|------|------|----------|\n"

        for beta in sorted(df['beta'].unique()):
            beta_data = df[df['beta'] == beta].sort_values('rmse')
            for _, row in beta_data.iterrows():
                supp_content += f"| {beta} | {row['method']} | {row['rmse']:.2f} | {row['mean_bias']:.2f} | {row['mean_abs_bias']:.2f} |\n"

    supp_content += """

### S2.2 Full Results by Pattern

"""

    if 'performance_by_pattern' in results:
        df = results['performance_by_pattern']
        supp_content += "| Pattern | Method | RMSE | Bias |\n"
        supp_content += "|---------|--------|------|------|\n"

        for pattern in df['pattern'].unique():
            pattern_data = df[df['pattern'] == pattern].sort_values('rmse')
            for _, row in pattern_data.iterrows():
                supp_content += f"| {pattern} | {row['method']} | {row['rmse']:.2f} | {row['mean_bias']:.2f} |\n"

    supp_content += """

---

## S3. Dataset Generation Details

### S3.1 OSRCT Algorithm

The OSRCT (Observational Sampling from RCT) algorithm creates confounded observational
data from randomized experiments by sampling units based on their covariates:

```
For each unit i in RCT:
    1. Compute selection probability: p_i = σ(β₀ + Σⱼ βⱼ·Cⱼᵢ)
    2. Sample preferred treatment: t_s ~ Bernoulli(p_i)
    3. If actual treatment T_i equals t_s: include unit
       Else: exclude unit
```

### S3.2 Experimental Grid

- **Studies:** 15 (anchoring1-4, gainloss, sunk, flag, quote, etc.)
- **Beta values:** 7 (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
- **Patterns:** 5 (age, gender, polideo, demo_basic, demo_full)
- **Total datasets:** 15 × 7 × 5 = 525

### S3.3 Random Seeds

All datasets generated with seed=42 for reproducibility.

---

## S4. Reproducibility

### S4.1 Software Versions

```
Python: 3.10
numpy: 1.26.4
pandas: 2.3.3
scipy: 1.11.4
scikit-learn: 1.7.2
matplotlib: 3.8.2
seaborn: 0.13.0
pyreadstat: 1.3.2
```

### S4.2 Reproduction Steps

```bash
# Clone repository
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
make all
```

---

## S5. Data Access

### S5.1 OSF Links

- ManyLabs1 Data: https://osf.io/wx7ck/
- OSRCT Benchmark: [To be added upon publication]

### S5.2 Citation

```bibtex
@misc{osrct_benchmark_2025,
  title={OSRCT Benchmark: Semi-Synthetic Datasets for Causal Inference Evaluation},
  author={[Authors]},
  year={2025}
}
```
"""

    with open(output_dir / 'supplementary_materials.md', 'w') as f:
        f.write(supp_content)

    print("  Created supplementary_materials.md")


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper components for OSRCT Benchmark'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='paper_components',
        help='Output directory for paper components'
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OSRCT Benchmark - Paper Components Generation")
    print("=" * 60)

    # Setup style
    colors = setup_publication_style()

    # Load data
    print("\nLoading data...")
    results = load_analysis_results(ANALYSIS_DIR)
    print(f"  Loaded {len(results)} result files")

    ground_truth = load_ground_truth(BENCHMARK_DIR)
    if ground_truth is not None:
        print(f"  Loaded ground truth for {len(ground_truth)} studies")

    # Generate figures
    print("\nGenerating figures...")
    figure_1_method_comparison(results, colors, figures_dir)
    figure_2_rmse_by_beta(results, colors, figures_dir)
    figure_3_bias_by_beta(results, colors, figures_dir)
    figure_4_heterogeneity(ground_truth, figures_dir)
    figure_5_pattern_comparison(results, colors, figures_dir)

    # Generate tables
    generate_latex_tables(results, ground_truth, tables_dir)

    # Generate supplementary materials
    generate_supplementary_materials(results, ground_truth, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("PAPER COMPONENTS COMPLETE")
    print("=" * 60)

    n_figures = len(list(figures_dir.glob("*.pdf")))
    n_tables = len(list(tables_dir.glob("*.tex")))

    print(f"\nOutput directory: {output_dir}")
    print(f"Figures generated: {n_figures} (PDF + PNG)")
    print(f"LaTeX tables: {n_tables}")
    print(f"Supplementary: supplementary_materials.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
