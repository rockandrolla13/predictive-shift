# Tutorials

Step-by-step guides for using the OSRCT Benchmark.

---

## Tutorial 1: Understanding OSRCT Data

### What is OSRCT?

OSRCT (Observational Sampling from RCTs) creates realistic confounded observational data from randomized controlled trials. The key insight is that by selectively sampling units based on their covariates, we can introduce confounding while preserving the known ground-truth treatment effect.

### Loading a Dataset

```python
import pandas as pd

# Load a confounded dataset
data = pd.read_csv(
    'osrct_benchmark_v1.0/confounded_datasets/by_study/anchoring1/anchoring1_demo_full_beta1.0.csv'
)

# Examine structure
print(f"Shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print(f"\nTreatment distribution:\n{data['iv'].value_counts()}")
```

### Understanding the Filename

Format: `{study}_{pattern}_beta{strength}.csv`

- **study**: ManyLabs1 study name (anchoring1, flag, sunk, etc.)
- **pattern**: Which covariates induce confounding
  - `age`: Only resp_age
  - `gender`: Only resp_gender
  - `polideo`: Only resp_polideo
  - `demo_basic`: resp_age + resp_gender
  - `demo_full`: resp_age + resp_gender + resp_polideo
- **strength**: Beta coefficient (0.1 to 2.0)

### Comparing to Ground Truth

```python
# Load ground truth ATEs from original RCTs
ground_truth = pd.read_csv('osrct_benchmark_v1.0/ground_truth/rct_ates.csv')
print(ground_truth[['study', 'ate', 'ate_se', 'n_total']])

# Get true ATE for anchoring1
true_ate = ground_truth[ground_truth['study'] == 'anchoring1']['ate'].values[0]
print(f"\nTrue ATE: {true_ate:.2f}")

# Naive estimate from confounded data (biased!)
treated = data[data['iv'] == 1]['dv'].mean()
control = data[data['iv'] == 0]['dv'].mean()
naive_ate = treated - control

print(f"Naive ATE: {naive_ate:.2f}")
print(f"Bias: {naive_ate - true_ate:.2f}")
```

---

## Tutorial 2: Evaluating Causal Methods

### Setup

```python
import sys
sys.path.append('osrct_benchmark_v1.0/code')

from causal_methods import (
    estimate_naive,
    estimate_ipw,
    estimate_outcome_regression,
    estimate_aipw,
    estimate_psm,
    CausalMethodEvaluator
)
```

### Running Individual Methods

```python
# Define covariates used for confounding (and adjustment)
covariates = ['resp_age', 'resp_gender', 'resp_polideo']

# Naive (no adjustment)
naive = estimate_naive(data, 'iv', 'dv')
print(f"Naive: {naive['ate']:.2f} (SE: {naive['se']:.2f})")

# IPW
ipw = estimate_ipw(data, 'iv', 'dv', covariates)
print(f"IPW: {ipw['ate']:.2f} (SE: {ipw['se']:.2f})")

# Outcome Regression
or_result = estimate_outcome_regression(data, 'iv', 'dv', covariates)
print(f"OR: {or_result['ate']:.2f} (SE: {or_result['se']:.2f})")

# AIPW (Doubly Robust)
aipw = estimate_aipw(data, 'iv', 'dv', covariates)
print(f"AIPW: {aipw['ate']:.2f} (SE: {aipw['se']:.2f})")

# PSM
psm = estimate_psm(data, 'iv', 'dv', covariates)
print(f"PSM: {psm['ate']:.2f} (SE: {psm['se']:.2f})")
```

### Using the Unified Evaluator

```python
# Create evaluator
evaluator = CausalMethodEvaluator(
    treatment_col='iv',
    outcome_col='dv',
    covariates=['resp_age', 'resp_gender', 'resp_polideo']
)

# Evaluate all methods
results = evaluator.evaluate_all(
    data,
    ground_truth_ate=true_ate,
    skip_causal_forest=True  # Requires econml
)

# Display results
print(results[['method', 'ate', 'se', 'bias', 'covers_truth']])
```

### Interpreting Results

| Column | Interpretation |
|--------|----------------|
| `ate` | Estimated treatment effect |
| `se` | Standard error |
| `bias` | ate - ground_truth (positive = overestimate) |
| `covers_truth` | True if 95% CI contains true ATE |

A good method should have:
- Low absolute bias
- Reasonable SE (not too wide, not too narrow)
- ~95% coverage rate across datasets

---

## Tutorial 3: Running the Full Benchmark

### Evaluating Across All Datasets

```python
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Setup
datasets_dir = Path('osrct_benchmark_v1.0/confounded_datasets/by_study')
ground_truth = pd.read_csv('osrct_benchmark_v1.0/ground_truth/rct_ates.csv')
gt_dict = dict(zip(ground_truth['study'], ground_truth['ate']))

evaluator = CausalMethodEvaluator(
    covariates=['resp_age', 'resp_gender', 'resp_polideo']
)

# Run benchmark
all_results = []

for study_dir in sorted(datasets_dir.iterdir()):
    if not study_dir.is_dir():
        continue

    study = study_dir.name
    true_ate = gt_dict.get(study)

    for csv_file in sorted(study_dir.glob('*.csv')):
        # Parse filename
        parts = csv_file.stem.split('_')
        pattern = parts[1] if len(parts) > 1 else 'unknown'
        beta = float(parts[-1].replace('beta', '')) if 'beta' in parts[-1] else 0

        print(f"Processing: {csv_file.name}")

        # Load and evaluate
        data = pd.read_csv(csv_file)
        results = evaluator.evaluate_all(
            data, ground_truth_ate=true_ate, skip_causal_forest=True
        )

        # Add metadata
        results['study'] = study
        results['pattern'] = pattern
        results['beta'] = beta

        all_results.append(results)

# Combine results
benchmark_results = pd.concat(all_results, ignore_index=True)
print(f"\nTotal evaluations: {len(benchmark_results)}")
```

### Aggregating Results

```python
# Overall performance by method
overall = benchmark_results.groupby('method').agg({
    'bias': ['mean', 'std'],
    'covers_truth': 'mean'
})
overall.columns = ['mean_bias', 'std_bias', 'coverage']
overall['rmse'] = benchmark_results.groupby('method').apply(
    lambda x: np.sqrt(((x['ate'] - x['ground_truth_ate'])**2).mean())
)
print("\n=== Overall Performance ===")
print(overall.sort_values('rmse'))

# Performance by beta
by_beta = benchmark_results.groupby(['method', 'beta']).agg({
    'bias': 'mean',
    'covers_truth': 'mean'
}).reset_index()
print("\n=== Performance by Confounding Strength ===")
print(by_beta.pivot(index='beta', columns='method', values='bias'))
```

---

## Tutorial 4: Generating Custom Confounded Data

### Basic Generation

```python
from osrct import OSRCTSampler, load_manylabs1_data

# Load original RCT data
rct_data = load_manylabs1_data(
    'ManyLabs1/pre-process/Manylabs1_data.pkl',
    study_filter='anchoring1'
)
print(f"Original RCT size: {len(rct_data)}")

# Create sampler with specific confounding
sampler = OSRCTSampler(
    biasing_covariates=['resp_age', 'resp_gender'],
    biasing_coefficients={'resp_age': 0.5, 'resp_gender': 0.8},
    intercept=0.0,
    random_seed=42
)

# Generate confounded sample
obs_data, selection_probs = sampler.sample(rct_data, treatment_col='iv')
```

### Varying Confounding Strength

```python
# Generate datasets with different confounding strengths
beta_values = [0.1, 0.5, 1.0, 2.0]

for beta in beta_values:
    sampler = OSRCTSampler(
        biasing_covariates=['resp_age'],
        biasing_coefficients={'resp_age': beta},
        random_seed=42
    )

    obs_data, _ = sampler.sample(rct_data, treatment_col='iv', verbose=False)

    # Compute naive ATE
    treated = obs_data[obs_data['iv'] == 1]['dv'].mean()
    control = obs_data[obs_data['iv'] == 0]['dv'].mean()
    naive_ate = treated - control

    print(f"Beta={beta}: Naive ATE = {naive_ate:.2f}, Bias = {naive_ate - true_ate:.2f}")
```

### Evaluating Confounding

```python
from osrct import evaluate_osrct_sample

metrics = evaluate_osrct_sample(
    rct_data, obs_data,
    treatment_col='iv',
    outcome_col='dv',
    covariates=['resp_age', 'resp_gender']
)

print(f"True ATE (RCT): {metrics['rct_ate']:.2f}")
print(f"Naive ATE (Obs): {metrics['obs_ate_naive']:.2f}")
print(f"Confounding Bias: {metrics['confounding_bias']:.2f}")
print(f"Sample Retention: {metrics['sample_retention_rate']:.1%}")
print(f"Treatment Rate - RCT: {metrics['rct_treatment_rate']:.1%}")
print(f"Treatment Rate - Obs: {metrics['obs_treatment_rate']:.1%}")
```

---

## Tutorial 5: Implementing Your Own Method

### Method Interface

All methods must follow this interface:

```python
def your_method(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list
) -> dict:
    """
    Your causal inference method.

    Parameters
    ----------
    data : DataFrame
        Observational data with treatment, outcome, and covariates
    treatment_col : str
        Name of treatment column (binary 0/1)
    outcome_col : str
        Name of outcome column (continuous)
    covariates : list
        List of covariate column names

    Returns
    -------
    dict with keys:
        'method': str - Method name
        'ate': float - Point estimate
        'se': float - Standard error
        'ci_lower': float - 95% CI lower bound
        'ci_upper': float - 95% CI upper bound
    """
    # Your implementation here
    pass
```

### Example: Simple Regression Adjustment

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def estimate_simple_regression(data, treatment_col, outcome_col, covariates):
    """Simple regression adjustment estimator."""

    # Prepare data
    X = data[[treatment_col] + covariates].values
    y = data[outcome_col].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # ATE is the treatment coefficient
    ate = model.coef_[0]

    # Bootstrap SE
    n = len(data)
    bootstrap_ates = []
    for _ in range(200):
        idx = np.random.choice(n, n, replace=True)
        X_b, y_b = X[idx], y[idx]
        model_b = LinearRegression().fit(X_b, y_b)
        bootstrap_ates.append(model_b.coef_[0])

    se = np.std(bootstrap_ates)

    return {
        'method': 'simple_regression',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se
    }

# Test it
result = estimate_simple_regression(data, 'iv', 'dv', ['resp_age', 'resp_gender'])
print(f"Simple Regression ATE: {result['ate']:.2f}")
```

### Benchmarking Your Method

```python
# Evaluate on all datasets
your_results = []

for study_dir in datasets_dir.iterdir():
    for csv_file in study_dir.glob('*.csv'):
        data = pd.read_csv(csv_file)
        study = study_dir.name
        true_ate = gt_dict.get(study)

        result = estimate_simple_regression(
            data, 'iv', 'dv', ['resp_age', 'resp_gender', 'resp_polideo']
        )
        result['study'] = study
        result['true_ate'] = true_ate
        result['bias'] = result['ate'] - true_ate
        result['covers'] = result['ci_lower'] <= true_ate <= result['ci_upper']

        your_results.append(result)

your_df = pd.DataFrame(your_results)

# Compute metrics
rmse = np.sqrt((your_df['bias']**2).mean())
coverage = your_df['covers'].mean()

print(f"Your Method - RMSE: {rmse:.2f}, Coverage: {coverage:.1%}")
```

### Submitting to Leaderboard

See [LEADERBOARD.md](../osrct_benchmark_v1.0/LEADERBOARD.md) for submission instructions.

---

## Tutorial 6: Heterogeneity Analysis

### Site-Level Analysis

```python
from causal_methods import compute_heterogeneity_metrics

# Compute site-level ATEs
site_results = []
for study in benchmark_results['study'].unique():
    study_data = benchmark_results[
        (benchmark_results['study'] == study) &
        (benchmark_results['method'] == 'aipw')
    ]

    if len(study_data) > 0:
        site_results.append({
            'site': study,
            'ate': study_data['ate'].mean(),
            'ate_se': study_data['se'].mean()
        })

site_df = pd.DataFrame(site_results)

# Compute heterogeneity metrics
het_metrics = compute_heterogeneity_metrics(site_df)

print(f"Number of sites: {het_metrics['n_sites']}")
print(f"Pooled ATE (FE): {het_metrics['pooled_ate_fe']:.2f}")
print(f"Pooled ATE (RE): {het_metrics['pooled_ate_re']:.2f}")
print(f"I² statistic: {het_metrics['I2']:.1f}%")
print(f"τ (between-site SD): {het_metrics['tau']:.2f}")
print(f"Prediction interval: [{het_metrics['prediction_interval_lower']:.2f}, {het_metrics['prediction_interval_upper']:.2f}]")
```

### Visualizing Heterogeneity

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# Forest plot
sites = site_df['site'].values
ates = site_df['ate'].values
ses = site_df['ate_se'].values

y_pos = range(len(sites))
ax.errorbar(ates, y_pos, xerr=1.96*ses, fmt='o', capsize=3)
ax.axvline(x=het_metrics['pooled_ate_re'], color='r', linestyle='--',
           label=f"Pooled RE: {het_metrics['pooled_ate_re']:.2f}")
ax.set_yticks(y_pos)
ax.set_yticklabels(sites)
ax.set_xlabel('Treatment Effect')
ax.set_title(f"Forest Plot (I² = {het_metrics['I2']:.1f}%)")
ax.legend()

plt.tight_layout()
plt.savefig('forest_plot.png', dpi=150)
```

---

## Next Steps

- Explore the [API Reference](api/index.md) for complete function documentation
- Check [LEADERBOARD.md](../osrct_benchmark_v1.0/LEADERBOARD.md) to compare methods
- Review [exploration notebooks](../exploration_notebooks/) for detailed analyses
