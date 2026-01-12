# Quick Start Guide

Get started with the OSRCT Benchmark in 5 minutes.

---

## 1. Load a Confounded Dataset

```python
import pandas as pd

# Load a pre-generated confounded dataset
data = pd.read_csv(
    'osrct_benchmark_v1.0/confounded_datasets/by_study/anchoring1/anchoring1_demo_full_beta1.0.csv'
)

print(f"Dataset: {len(data)} observations")
print(f"Treatment rate: {data['iv'].mean():.1%}")
print(f"Columns: {list(data.columns)}")
```

**Dataset naming convention**: `{study}_{pattern}_beta{strength}.csv`
- `study`: One of 15 ManyLabs1 studies
- `pattern`: Confounding pattern (age, gender, polideo, demo_basic, demo_full)
- `strength`: Beta value (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

---

## 2. Get Ground Truth ATE

```python
# Load ground truth
ground_truth = pd.read_csv('osrct_benchmark_v1.0/ground_truth/rct_ates.csv')

# Get true ATE for anchoring1 study
true_ate = ground_truth[ground_truth['study'] == 'anchoring1']['ate'].values[0]
print(f"True ATE: {true_ate:.2f}")
```

---

## 3. Estimate Treatment Effects

```python
import sys
sys.path.append('osrct_benchmark_v1.0/code')
from causal_methods import estimate_naive, estimate_ipw, estimate_aipw

# Naive estimate (biased due to confounding)
naive = estimate_naive(data, treatment_col='iv', outcome_col='dv')
print(f"Naive ATE: {naive['ate']:.2f} (Bias: {naive['ate'] - true_ate:.2f})")

# IPW estimate (adjusts for confounding)
covariates = ['resp_age', 'resp_gender', 'resp_polideo']
ipw = estimate_ipw(data, 'iv', 'dv', covariates)
print(f"IPW ATE: {ipw['ate']:.2f} (Bias: {ipw['ate'] - true_ate:.2f})")

# AIPW estimate (doubly robust)
aipw = estimate_aipw(data, 'iv', 'dv', covariates)
print(f"AIPW ATE: {aipw['ate']:.2f} (Bias: {aipw['ate'] - true_ate:.2f})")
```

---

## 4. Evaluate All Methods

```python
from causal_methods import CausalMethodEvaluator

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
    skip_causal_forest=True  # Set False if econml installed
)

# Display results
print(results[['method', 'ate', 'se', 'bias', 'covers_truth']])
```

---

## 5. Generate Your Own Confounded Data

```python
from osrct import OSRCTSampler, load_manylabs1_data

# Load original RCT data
rct_data = load_manylabs1_data(
    'ManyLabs1/pre-process/Manylabs1_data.pkl',
    study_filter='anchoring1'
)

# Create sampler with custom confounding
sampler = OSRCTSampler(
    biasing_covariates=['resp_age', 'resp_gender'],
    biasing_coefficients={'resp_age': 0.5, 'resp_gender': 0.8},
    random_seed=42
)

# Generate confounded sample
obs_data, probs = sampler.sample(rct_data, treatment_col='iv')
```

---

## 6. Benchmark Your Method

```python
def your_custom_method(data, treatment_col, outcome_col, covariates):
    """
    Implement your causal inference method here.

    Returns
    -------
    dict with keys: 'ate', 'se', 'ci_lower', 'ci_upper'
    """
    # Your implementation
    ate = ...
    se = ...
    return {
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se
    }

# Evaluate on multiple datasets
import os
from pathlib import Path

results = []
datasets_dir = Path('osrct_benchmark_v1.0/confounded_datasets/by_study')

for study_dir in datasets_dir.iterdir():
    for csv_file in study_dir.glob('*.csv'):
        data = pd.read_csv(csv_file)
        study = study_dir.name
        true_ate = ground_truth[ground_truth['study'] == study]['ate'].values[0]

        result = your_custom_method(data, 'iv', 'dv', covariates)
        result['study'] = study
        result['dataset'] = csv_file.name
        result['bias'] = result['ate'] - true_ate
        results.append(result)

results_df = pd.DataFrame(results)
print(f"RMSE: {(results_df['bias']**2).mean()**0.5:.2f}")
```

---

## Next Steps

- **[API Reference](api/index.md)**: Complete function documentation
- **[Tutorials](tutorials.md)**: In-depth guides
- **[Notebooks](../osrct_benchmark_v1.0/notebooks/)**: Interactive examples
- **[LEADERBOARD.md](../osrct_benchmark_v1.0/LEADERBOARD.md)**: Submit your method
