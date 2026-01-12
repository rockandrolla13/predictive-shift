# Continuous Treatments API Reference

`osrct_continuous.py` - Extension for continuous and multi-valued treatments

---

## Overview

The standard OSRCT algorithm handles binary treatments. This module extends the framework to:

1. **Continuous treatments** (e.g., dosage, duration, intensity)
2. **Multi-valued treatments** (e.g., K treatment arms)
3. **Dose-response estimation** methods

---

## Classes

### GPSSampler

Generalized Propensity Score Sampler for continuous treatments.

```python
class GPSSampler(
    biasing_covariates: List[str],
    biasing_coefficients: Optional[Dict[str, float]] = None,
    treatment_noise_scale: float = 1.0,
    sampling_method: str = 'importance',
    standardize: bool = True,
    random_seed: Optional[int] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `biasing_covariates` | `List[str]` | Required | Covariates influencing treatment |
| `biasing_coefficients` | `Dict[str, float]` | `None` | Coefficients for GPS model |
| `treatment_noise_scale` | `float` | `1.0` | Noise scale (confounding strength) |
| `sampling_method` | `str` | `'importance'` | `'importance'`, `'rejection'`, or `'threshold'` |
| `standardize` | `bool` | `True` | Standardize covariates |
| `random_seed` | `int` | `None` | Random seed |

#### Methods

##### `sample()`

Generate confounded sample for continuous treatment.

```python
def sample(
    self,
    rct_data: pd.DataFrame,
    treatment_col: str,
    confounding_strength: float = 1.0,
    target_size: Optional[int] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rct_data` | `DataFrame` | Required | RCT data with continuous treatment |
| `treatment_col` | `str` | Required | Continuous treatment column |
| `confounding_strength` | `float` | `1.0` | Controls confounding intensity |
| `target_size` | `int` | `None` | Target sample size (default: 50%) |
| `verbose` | `bool` | `True` | Print statistics |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `observational_sample` | `DataFrame` | Confounded data with `_sampling_weight` column |
| `sampling_weights` | `ndarray` | Weights used for sampling |

**Example:**

```python
from osrct_continuous import GPSSampler

# Create sampler
sampler = GPSSampler(
    biasing_covariates=['age', 'income'],
    confounding_strength=2.0,
    random_seed=42
)

# Generate confounded sample
obs_data, weights = sampler.sample(
    rct_data,
    treatment_col='dosage',
    confounding_strength=2.0
)
```

##### `compute_gps()`

Compute Generalized Propensity Score f(T|X).

```python
def compute_gps(
    self,
    data: pd.DataFrame,
    treatment_col: str
) -> np.ndarray
```

**Returns:** GPS values (densities) for each unit

---

### MultiValuedTreatmentSampler

OSRCT Sampler for treatments with K > 2 categories.

```python
class MultiValuedTreatmentSampler(
    biasing_covariates: List[str],
    biasing_coefficients: Optional[Dict[str, Dict[int, float]]] = None,
    standardize: bool = True,
    random_seed: Optional[int] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `biasing_covariates` | `List[str]` | Required | Covariates for biasing |
| `biasing_coefficients` | `Dict` | `None` | Nested dict: `{cov: {level: coef}}` |
| `standardize` | `bool` | `True` | Standardize covariates |
| `random_seed` | `int` | `None` | Random seed |

#### Methods

##### `sample()`

```python
def sample(
    self,
    rct_data: pd.DataFrame,
    treatment_col: str,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]
```

**Algorithm:**
1. Compute P(T=k|X) for each treatment level using softmax
2. Sample preferred treatment from this distribution
3. Keep units where actual treatment matches preferred

**Example:**

```python
from osrct_continuous import MultiValuedTreatmentSampler

# 4-arm trial
sampler = MultiValuedTreatmentSampler(
    biasing_covariates=['age', 'gender'],
    random_seed=42
)

obs_data, probs = sampler.sample(
    rct_data,
    treatment_col='treatment_arm'  # Values: 1, 2, 3, 4
)
```

---

## Estimation Functions

### estimate_gps_weighting()

GPS-based weighting for dose-response estimation.

```python
def estimate_gps_weighting(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str],
    n_bins: int = 5,
    trim_threshold: float = 0.01
) -> Dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | Required | Continuous treatment |
| `outcome_col` | `str` | Required | Outcome column |
| `covariates` | `List[str]` | Required | Covariates for GPS |
| `n_bins` | `int` | `5` | Number of treatment bins |
| `trim_threshold` | `float` | `0.01` | GPS trimming threshold |

**Returns:**

```python
{
    'method': 'gps_weighting',
    'dose_response': [
        {'treatment_level': float, 'expected_outcome': float, 'n_units': int},
        ...
    ],
    'gps_mean': float,
    'gps_std': float,
    'weight_mean': float,
    'weight_std': float,
    'n_used': int
}
```

**Example:**

```python
from osrct_continuous import estimate_gps_weighting

result = estimate_gps_weighting(
    obs_data,
    treatment_col='dosage',
    outcome_col='response',
    covariates=['age', 'weight'],
    n_bins=10
)

# Plot dose-response
for point in result['dose_response']:
    print(f"Dosage {point['treatment_level']:.1f}: E[Y] = {point['expected_outcome']:.2f}")
```

---

### estimate_dose_response_kernel()

Kernel-based dose-response curve estimation.

```python
def estimate_dose_response_kernel(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str],
    treatment_grid: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None
) -> Dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | Required | Continuous treatment |
| `outcome_col` | `str` | Required | Outcome |
| `covariates` | `List[str]` | Required | Covariates for GPS |
| `treatment_grid` | `ndarray` | `None` | Treatment values for evaluation |
| `bandwidth` | `float` | `None` | Kernel bandwidth (Silverman's rule if None) |

**Returns:**

```python
{
    'method': 'kernel_dose_response',
    'dose_response': [
        {'treatment_level': float, 'expected_outcome': float, 'effective_n': float},
        ...
    ],
    'bandwidth': float,
    'n_used': int
}
```

---

### evaluate_continuous_treatment_sample()

Evaluate GPS sampling quality.

```python
def evaluate_continuous_treatment_sample(
    rct_data: pd.DataFrame,
    obs_data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str]
) -> Dict
```

**Returns:**

```python
{
    'n_rct': int,
    'n_obs': int,
    'retention_rate': float,
    'treatment_mean_rct': float,
    'treatment_mean_obs': float,
    'treatment_std_rct': float,
    'treatment_std_obs': float,
    'confounding': {
        'covariate_name': {
            'rct_correlation': float,
            'obs_correlation': float,
            'change': float
        }
    },
    'outcome_mean_rct': float,
    'outcome_mean_obs': float
}
```

---

## Mathematical Background

### Generalized Propensity Score

For continuous treatment T, the GPS is the conditional density:

```
r(t, x) = f(T=t | X=x)
```

Under the assumption:

```
Y(t) ⊥ T | r(T, X)   (weak unconfoundedness)
```

The GPS can be used to adjust for confounding.

### GPS Model

We model the treatment as:

```
T | X ~ N(X'β, σ²)
```

Then:

```
r(t, x) = (1/σ√2π) exp(-(t - x'β)² / 2σ²)
```

### Dose-Response Estimation

The dose-response function is:

```
μ(t) = E[Y | T = t]
```

Under confounding, naive estimation is biased. GPS methods estimate:

```
μ(t) = E[E[Y | T, r(T,X)] | T = t]
```

---

## Usage Examples

### Example 1: Continuous Treatment Confounding

```python
import numpy as np
import pandas as pd
from osrct_continuous import GPSSampler, evaluate_continuous_treatment_sample

# Simulate RCT with continuous treatment (e.g., drug dosage)
np.random.seed(42)
n = 1000

age = np.random.normal(50, 10, n)
weight = np.random.normal(70, 15, n)
dosage = np.random.uniform(10, 100, n)  # Randomly assigned in RCT

# True dose-response: outcome = 5 + 0.3*dosage + 0.2*age + noise
outcome = 5 + 0.3 * dosage + 0.2 * age + np.random.normal(0, 5, n)

rct_data = pd.DataFrame({
    'age': age, 'weight': weight, 'dosage': dosage, 'outcome': outcome
})

# Create confounded sample (older patients get higher dosage)
sampler = GPSSampler(
    biasing_covariates=['age', 'weight'],
    random_seed=42
)

obs_data, _ = sampler.sample(
    rct_data,
    treatment_col='dosage',
    confounding_strength=2.0
)

# Check confounding
metrics = evaluate_continuous_treatment_sample(
    rct_data, obs_data, 'dosage', 'outcome', ['age', 'weight']
)
print(f"Age-Dosage correlation: RCT={metrics['confounding']['age']['rct_correlation']:.3f}, "
      f"Obs={metrics['confounding']['age']['obs_correlation']:.3f}")
```

### Example 2: Multi-Valued Treatment

```python
from osrct_continuous import MultiValuedTreatmentSampler

# 3-arm trial (placebo, low dose, high dose)
treatment_arms = np.random.choice([0, 1, 2], n, p=[0.33, 0.34, 0.33])

rct_data['treatment_arm'] = treatment_arms

sampler = MultiValuedTreatmentSampler(
    biasing_covariates=['age'],
    random_seed=42
)

obs_data, probs = sampler.sample(rct_data, 'treatment_arm')
```

### Example 3: Dose-Response Estimation

```python
from osrct_continuous import estimate_gps_weighting, estimate_dose_response_kernel

# GPS weighting
result_gps = estimate_gps_weighting(
    obs_data, 'dosage', 'outcome', ['age', 'weight'],
    n_bins=10
)

# Kernel method
result_kernel = estimate_dose_response_kernel(
    obs_data, 'dosage', 'outcome', ['age', 'weight'],
    treatment_grid=np.linspace(10, 100, 20)
)

# Compare methods
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
gps_x = [p['treatment_level'] for p in result_gps['dose_response']]
gps_y = [p['expected_outcome'] for p in result_gps['dose_response']]
kernel_x = [p['treatment_level'] for p in result_kernel['dose_response']]
kernel_y = [p['expected_outcome'] for p in result_kernel['dose_response']]

ax.plot(gps_x, gps_y, 'o-', label='GPS Weighting')
ax.plot(kernel_x, kernel_y, 's-', label='Kernel')
ax.set_xlabel('Dosage')
ax.set_ylabel('E[Outcome]')
ax.legend()
plt.savefig('dose_response.png')
```

---

## References

1. Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. *Applied Bayesian modeling and causal inference from incomplete-data perspectives*, 226164, 73-84.

2. Imai, K., & Van Dyk, D. A. (2004). Causal inference with general treatment regimes. *Journal of the American Statistical Association*, 99(467), 854-866.

3. Gentzel, M., Garant, D., & Jensen, D. (2021). The case for evaluating causal models using interventional measures and empirical data. *NeurIPS 2021*.

4. Kennedy, E. H., Ma, Z., McHugh, M. D., & Small, D. S. (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *Journal of the Royal Statistical Society: Series B*, 79(4), 1229-1245.
