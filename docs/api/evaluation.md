# Evaluation Module API Reference

Evaluation utilities and performance metrics for the OSRCT Benchmark.

---

## Performance Metrics

### Standard Metrics

The benchmark uses the following metrics to evaluate causal inference methods:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Bias** | `ATE_est - ATE_true` | Systematic error (positive = overestimate) |
| **Absolute Bias** | `|ATE_est - ATE_true|` | Magnitude of bias |
| **RMSE** | `sqrt(mean((ATE_est - ATE_true)²))` | Root mean squared error |
| **Coverage** | `mean(CI contains ATE_true)` | 95% CI coverage rate |
| **CI Width** | `CI_upper - CI_lower` | Confidence interval width |

### Computing Metrics

```python
import numpy as np
import pandas as pd

def compute_performance_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate performance metrics from evaluation results.

    Parameters
    ----------
    results_df : DataFrame
        Must have columns: 'ate', 'ground_truth_ate', 'ci_lower', 'ci_upper'

    Returns
    -------
    metrics : dict
        Performance metrics
    """
    bias = results_df['ate'] - results_df['ground_truth_ate']

    return {
        'mean_bias': bias.mean(),
        'median_bias': bias.median(),
        'abs_bias': bias.abs().mean(),
        'rmse': np.sqrt((bias ** 2).mean()),
        'coverage': (
            (results_df['ci_lower'] <= results_df['ground_truth_ate']) &
            (results_df['ground_truth_ate'] <= results_df['ci_upper'])
        ).mean(),
        'ci_width': (results_df['ci_upper'] - results_df['ci_lower']).mean()
    }
```

---

## Evaluation Workflow

### Single Dataset Evaluation

```python
from causal_methods import CausalMethodEvaluator
import pandas as pd

# Load data and ground truth
data = pd.read_csv('dataset.csv')
ground_truth = pd.read_csv('ground_truth/rct_ates.csv')
true_ate = ground_truth[ground_truth['study'] == 'anchoring1']['ate'].values[0]

# Evaluate all methods
evaluator = CausalMethodEvaluator()
results = evaluator.evaluate_all(data, ground_truth_ate=true_ate)

# View results
print(results[['method', 'ate', 'se', 'bias', 'covers_truth']])
```

### Full Benchmark Evaluation

```python
import os
from pathlib import Path

def run_full_benchmark(
    datasets_dir: str,
    ground_truth_path: str,
    covariates: list = None,
    skip_causal_forest: bool = True
) -> pd.DataFrame:
    """
    Run evaluation on all benchmark datasets.

    Parameters
    ----------
    datasets_dir : str
        Path to confounded_datasets/by_study/
    ground_truth_path : str
        Path to rct_ates.csv
    covariates : list
        Covariates for adjustment methods
    skip_causal_forest : bool
        Whether to skip slow causal forest method

    Returns
    -------
    results : DataFrame
        Results for all methods on all datasets
    """
    # Load ground truth
    gt = pd.read_csv(ground_truth_path)
    gt_dict = dict(zip(gt['study'], gt['ate']))

    # Initialize evaluator
    evaluator = CausalMethodEvaluator(covariates=covariates)

    all_results = []
    datasets_path = Path(datasets_dir)

    for study_dir in datasets_path.iterdir():
        if not study_dir.is_dir():
            continue

        study = study_dir.name
        true_ate = gt_dict.get(study)

        if true_ate is None:
            continue

        for csv_file in study_dir.glob('*.csv'):
            # Parse filename
            parts = csv_file.stem.split('_')
            pattern = parts[1]
            beta = float(parts[2].replace('beta', ''))

            # Load and evaluate
            data = pd.read_csv(csv_file)
            results = evaluator.evaluate_all(
                data,
                ground_truth_ate=true_ate,
                skip_causal_forest=skip_causal_forest
            )

            # Add metadata
            results['study'] = study
            results['pattern'] = pattern
            results['beta'] = beta
            results['dataset'] = csv_file.name

            all_results.append(results)

    return pd.concat(all_results, ignore_index=True)

# Run benchmark
results = run_full_benchmark(
    'osrct_benchmark_v1.0/confounded_datasets/by_study',
    'osrct_benchmark_v1.0/ground_truth/rct_ates.csv'
)

# Aggregate by method
summary = results.groupby('method').agg({
    'bias': 'mean',
    'ate': lambda x: np.sqrt((x ** 2).mean()),  # RMSE proxy
    'covers_truth': 'mean'
}).rename(columns={'ate': 'rmse', 'covers_truth': 'coverage'})

print(summary)
```

---

## Stratified Analysis

### By Confounding Strength (Beta)

```python
def analyze_by_beta(results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze method performance by confounding strength.
    """
    return results.groupby(['method', 'beta']).agg({
        'bias': 'mean',
        'abs_bias': lambda x: x.abs().mean(),
        'covers_truth': 'mean'
    }).reset_index()

by_beta = analyze_by_beta(results)

# Pivot for easier comparison
pivot = by_beta.pivot(index='beta', columns='method', values='bias')
print(pivot)
```

### By Confounding Pattern

```python
def analyze_by_pattern(results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze method performance by confounding pattern.
    """
    return results.groupby(['method', 'pattern']).agg({
        'bias': 'mean',
        'abs_bias': lambda x: x.abs().mean(),
        'covers_truth': 'mean'
    }).reset_index()

by_pattern = analyze_by_pattern(results)
```

### By Study

```python
def analyze_by_study(results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze method performance by study.
    """
    return results.groupby(['method', 'study']).agg({
        'bias': 'mean',
        'abs_bias': lambda x: x.abs().mean(),
        'covers_truth': 'mean'
    }).reset_index()

by_study = analyze_by_study(results)
```

---

## Visualization Functions

### RMSE Comparison Plot

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rmse_comparison(results: pd.DataFrame, save_path: str = None):
    """
    Create bar plot comparing RMSE across methods.
    """
    rmse_by_method = results.groupby('method').apply(
        lambda x: np.sqrt(((x['ate'] - x['ground_truth_ate'])**2).mean())
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_by_method.plot(kind='barh', ax=ax)
    ax.set_xlabel('RMSE')
    ax.set_title('Method Performance (RMSE)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig

plot_rmse_comparison(results, 'rmse_comparison.png')
```

### Bias by Beta Plot

```python
def plot_bias_by_beta(results: pd.DataFrame, save_path: str = None):
    """
    Create line plot showing bias across confounding strengths.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in results['method'].unique():
        method_data = results[results['method'] == method]
        by_beta = method_data.groupby('beta')['bias'].mean()
        ax.plot(by_beta.index, by_beta.values, marker='o', label=method)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Beta (Confounding Strength)')
    ax.set_ylabel('Mean Bias')
    ax.set_title('Bias by Confounding Strength')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig
```

### Coverage Plot

```python
def plot_coverage(results: pd.DataFrame, save_path: str = None):
    """
    Create coverage plot with 95% nominal line.
    """
    coverage_by_method = results.groupby('method')['covers_truth'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    coverage_by_method.plot(kind='barh', ax=ax)
    ax.axvline(x=0.95, color='r', linestyle='--', label='Nominal (95%)')
    ax.set_xlabel('Coverage Rate')
    ax.set_title('95% CI Coverage by Method')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig
```

---

## Heterogeneity Analysis

### Cross-Site Heterogeneity

```python
from causal_methods import compute_heterogeneity_metrics

def analyze_heterogeneity(
    results: pd.DataFrame,
    site_col: str = 'site'
) -> pd.DataFrame:
    """
    Analyze treatment effect heterogeneity across sites.
    """
    heterogeneity_results = []

    for method in results['method'].unique():
        method_data = results[results['method'] == method]

        # Get site-level estimates
        site_ates = method_data.groupby(site_col).agg({
            'ate': 'mean',
            'se': lambda x: np.sqrt((x**2).mean())  # Combined SE
        }).reset_index()
        site_ates.columns = [site_col, 'ate', 'ate_se']

        # Compute heterogeneity metrics
        metrics = compute_heterogeneity_metrics(site_ates)
        metrics['method'] = method
        heterogeneity_results.append(metrics)

    return pd.DataFrame(heterogeneity_results)

het_results = analyze_heterogeneity(results, site_col='study')
print(het_results[['method', 'I2', 'tau', 'pooled_ate_re']])
```

---

## Model Diagnostics

### Propensity Score Diagnostics

```python
def diagnose_propensity_scores(results: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and summarize propensity score diagnostics.
    """
    ipw_results = results[results['method'].isin(['ipw', 'aipw', 'psm'])]

    diagnostics = ipw_results.groupby('method').agg({
        'propensity_mean': 'mean',
        'propensity_std': 'mean',
        'propensity_min': 'min',
        'propensity_max': 'max'
    })

    return diagnostics
```

### Overlap Assessment

```python
def assess_overlap(data: pd.DataFrame, covariates: list) -> Dict[str, float]:
    """
    Assess covariate overlap between treatment groups.
    """
    from sklearn.linear_model import LogisticRegression

    X = data[covariates].values
    T = data['iv'].values

    # Fit propensity model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, T)
    e = model.predict_proba(X)[:, 1]

    return {
        'propensity_mean': e.mean(),
        'propensity_std': e.std(),
        'propensity_min': e.min(),
        'propensity_max': e.max(),
        'overlap_region': ((e > 0.1) & (e < 0.9)).mean(),
        'extreme_weights_pct': ((e < 0.05) | (e > 0.95)).mean()
    }
```

---

## Saving Results

### Export to CSV

```python
# Save full results
results.to_csv('benchmark_results.csv', index=False)

# Save summary by method
summary = results.groupby('method').agg({
    'bias': ['mean', 'std'],
    'covers_truth': 'mean'
})
summary.columns = ['_'.join(col) for col in summary.columns]
summary.to_csv('method_summary.csv')
```

### Export to LaTeX

```python
def results_to_latex(results: pd.DataFrame, caption: str = '') -> str:
    """
    Export results summary to LaTeX table.
    """
    summary = results.groupby('method').agg({
        'bias': 'mean',
        'se': 'mean',
        'covers_truth': 'mean'
    }).round(3)

    return summary.to_latex(caption=caption)

latex_table = results_to_latex(results, caption='Method Performance Summary')
print(latex_table)
```
