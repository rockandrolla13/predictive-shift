# Causal Methods Module API Reference

`causal_methods.py` - Causal inference method implementations

---

## Estimator Functions

### estimate_naive()

Naive Difference-in-Means Estimator.

```python
def estimate_naive(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None  # unused
) -> Dict[str, Any]
```

**Formula:**

```
ATE = E[Y|T=1] - E[Y|T=0]
```

**Assumption:** No confounding (violated in OSRCT data)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column (binary 0/1) |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `None` | Unused (for interface consistency) |

**Returns:**

```python
{
    'method': 'naive',
    'ate': float,        # Point estimate
    'se': float,         # Standard error
    'ci_lower': float,   # 95% CI lower bound
    'ci_upper': float,   # 95% CI upper bound
    'n_treated': int,    # Number of treated units
    'n_control': int     # Number of control units
}
```

**Example:**

```python
from causal_methods import estimate_naive

result = estimate_naive(data, 'iv', 'dv')
print(f"Naive ATE: {result['ate']:.2f} (SE: {result['se']:.2f})")
```

---

### estimate_ipw()

Inverse Probability Weighting (Horvitz-Thompson) Estimator.

```python
def estimate_ipw(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    trim_threshold: float = 0.01,
    normalize_weights: bool = True
) -> Dict[str, Any]
```

**Formula:**

```
ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
```

Where `e(X) = P(T=1|X)` is the propensity score.

**Assumption:** Correct propensity score model specification

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Covariates for propensity score model |
| `trim_threshold` | `float` | `0.01` | Trim propensity scores outside [threshold, 1-threshold] |
| `normalize_weights` | `bool` | `True` | Whether to normalize weights (Hajek estimator) |

**Returns:**

```python
{
    'method': 'ipw',
    'ate': float,
    'se': float,           # Bootstrap SE
    'ci_lower': float,
    'ci_upper': float,
    'propensity_mean': float,
    'propensity_std': float,
    'propensity_min': float,
    'propensity_max': float,
    'n_used': int
}
```

**Example:**

```python
from causal_methods import estimate_ipw

result = estimate_ipw(
    data, 'iv', 'dv',
    covariates=['resp_age', 'resp_gender', 'resp_polideo'],
    trim_threshold=0.05
)
print(f"IPW ATE: {result['ate']:.2f}")
print(f"Propensity score range: [{result['propensity_min']:.3f}, {result['propensity_max']:.3f}]")
```

---

### estimate_outcome_regression()

Outcome Regression (G-computation) Estimator.

```python
def estimate_outcome_regression(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    include_interactions: bool = False
) -> Dict[str, Any]
```

**Formula:**

```
Fit: E[Y|T,X] = α + βT + γX (+ δ(T*X) if interactions)
ATE = E[μ(1,X) - μ(0,X)]
```

**Assumption:** Correct outcome model specification

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Covariates for outcome model |
| `include_interactions` | `bool` | `False` | Whether to include treatment-covariate interactions |

**Returns:**

```python
{
    'method': 'outcome_regression',
    'ate': float,
    'se': float,         # Bootstrap SE
    'ci_lower': float,
    'ci_upper': float,
    'r_squared': float,  # Model fit
    'n_used': int
}
```

**Example:**

```python
from causal_methods import estimate_outcome_regression

result = estimate_outcome_regression(
    data, 'iv', 'dv',
    covariates=['resp_age', 'resp_gender'],
    include_interactions=True
)
print(f"OR ATE: {result['ate']:.2f}, R²: {result['r_squared']:.3f}")
```

---

### estimate_aipw()

Augmented Inverse Probability Weighting (Doubly Robust) Estimator.

```python
def estimate_aipw(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    trim_threshold: float = 0.01
) -> Dict[str, Any]
```

**Formula:**

```
AIPW = E[μ₁(X) - μ₀(X) + T(Y - μ₁(X))/e(X) - (1-T)(Y - μ₀(X))/(1-e(X))]
```

**Properties:**
- Consistent if EITHER propensity OR outcome model is correct
- More efficient than IPW when outcome model is correct
- Semiparametrically efficient

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Covariates for both models |
| `trim_threshold` | `float` | `0.01` | Trim propensity scores |

**Returns:**

```python
{
    'method': 'aipw',
    'ate': float,
    'se': float,            # Influence function SE
    'ci_lower': float,
    'ci_upper': float,
    'propensity_mean': float,
    'propensity_std': float,
    'n_used': int
}
```

**Example:**

```python
from causal_methods import estimate_aipw

result = estimate_aipw(data, 'iv', 'dv', covariates=['resp_age', 'resp_gender'])
print(f"AIPW ATE: {result['ate']:.2f} (SE: {result['se']:.2f})")
```

---

### estimate_psm()

Propensity Score Matching Estimator.

```python
def estimate_psm(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    n_neighbors: int = 1,
    caliper: float = 0.2,
    with_replacement: bool = False
) -> Dict[str, Any]
```

**Algorithm:**
1. Estimate propensity scores `e(X)`
2. Match each treated unit to nearest control(s) on `e(X)`
3. Estimate ATT from matched sample

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Covariates for propensity model |
| `n_neighbors` | `int` | `1` | Number of control matches per treated unit |
| `caliper` | `float` | `0.2` | Maximum propensity score distance (in SD units) |
| `with_replacement` | `bool` | `False` | Whether to match with replacement |

**Returns:**

```python
{
    'method': 'psm',
    'ate': float,
    'se': float,
    'ci_lower': float,
    'ci_upper': float,
    'n_matched': int,      # Number of matched pairs
    'match_rate': float,   # Proportion matched
    'n_treated': int,
    'n_control': int,
    'caliper_used': float
}
```

**Example:**

```python
from causal_methods import estimate_psm

result = estimate_psm(
    data, 'iv', 'dv',
    covariates=['resp_age', 'resp_gender'],
    n_neighbors=3,
    caliper=0.1
)
print(f"PSM ATE: {result['ate']:.2f}")
print(f"Match rate: {result['match_rate']:.1%}")
```

---

### estimate_causal_forest()

Causal Forest Estimator (Wager & Athey, 2018).

```python
def estimate_causal_forest(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    n_estimators: int = 500
) -> Dict[str, Any]
```

Uses random forest to estimate heterogeneous treatment effects `τ(X)`, then averages to get ATE.

**Requires:** `econml` package (`pip install econml`)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Covariates for effect heterogeneity |
| `n_estimators` | `int` | `500` | Number of trees |

**Returns:**

```python
{
    'method': 'causal_forest',
    'ate': float,
    'se': float,
    'ci_lower': float,
    'ci_upper': float,
    'te_std': float,   # Treatment effect heterogeneity
    'te_min': float,
    'te_max': float,
    'n_used': int
}
```

**Example:**

```python
from causal_methods import estimate_causal_forest

result = estimate_causal_forest(data, 'iv', 'dv', n_estimators=200)
print(f"Causal Forest ATE: {result['ate']:.2f}")
print(f"Effect heterogeneity (SD): {result['te_std']:.2f}")
```

---

## Classes

### CausalMethodEvaluator

Unified interface for evaluating causal inference methods.

```python
class CausalMethodEvaluator(
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `['resp_age', 'resp_gender', 'resp_polideo']` | Default covariates for adjustment methods |

**Class Attributes:**

```python
METHODS = {
    'naive': estimate_naive,
    'ipw': estimate_ipw,
    'outcome_regression': estimate_outcome_regression,
    'aipw': estimate_aipw,
    'psm': estimate_psm,
    'causal_forest': estimate_causal_forest
}
```

#### Methods

##### `evaluate_method()`

Evaluate a single method on data.

```python
def evaluate_method(
    self,
    data: pd.DataFrame,
    method: str,
    ground_truth_ate: float = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `method` | `str` | Required | Method name |
| `ground_truth_ate` | `float` | `None` | True ATE for computing bias |
| `**kwargs` | | | Additional arguments for the method |

**Returns:**

Method results plus (if `ground_truth_ate` provided):
- `ground_truth_ate`: The true ATE
- `bias`: Estimated - True
- `abs_bias`: Absolute bias
- `relative_bias`: Bias / True ATE
- `covers_truth`: Whether 95% CI covers true ATE

##### `evaluate_all()`

Evaluate all methods on a single dataset.

```python
def evaluate_all(
    self,
    data: pd.DataFrame,
    ground_truth_ate: float = None,
    methods: List[str] = None,
    skip_causal_forest: bool = False
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Observational data |
| `ground_truth_ate` | `float` | `None` | True ATE |
| `methods` | `List[str]` | `None` | Methods to evaluate (default: all) |
| `skip_causal_forest` | `bool` | `False` | Whether to skip causal forest (slow) |

**Returns:**

`DataFrame` with results for all methods

**Example:**

```python
from causal_methods import CausalMethodEvaluator

evaluator = CausalMethodEvaluator(
    treatment_col='iv',
    outcome_col='dv',
    covariates=['resp_age', 'resp_gender', 'resp_polideo']
)

results = evaluator.evaluate_all(
    data,
    ground_truth_ate=1555.67,
    skip_causal_forest=True
)

print(results[['method', 'ate', 'bias', 'covers_truth']])
```

---

## Utility Functions

### bootstrap_se()

Compute standard error via bootstrap.

```python
def bootstrap_se(
    data: pd.DataFrame,
    estimator_func: Callable,
    n_bootstrap: int = 200,
    random_state: int = 42
) -> float
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | Required | Input data |
| `estimator_func` | `Callable` | Required | Function that takes data and returns point estimate |
| `n_bootstrap` | `int` | `200` | Number of bootstrap samples |
| `random_state` | `int` | `42` | Random seed |

**Returns:** `float` - Bootstrap standard error

---

### compute_smd()

Compute Standardized Mean Difference for covariate balance.

```python
def compute_smd(
    data: pd.DataFrame,
    treatment_col: str,
    covariate: str
) -> float
```

**Formula:**

```
SMD = (mean_treated - mean_control) / pooled_std
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Data with treatment and covariate |
| `treatment_col` | `str` | Name of treatment column |
| `covariate` | `str` | Name of covariate column |

**Returns:** `float` - Standardized mean difference

**Interpretation:**
- `|SMD| < 0.1`: Good balance
- `|SMD| < 0.25`: Acceptable balance
- `|SMD| >= 0.25`: Imbalanced

---

### compute_heterogeneity_metrics()

Compute metrics characterizing treatment effect heterogeneity across sites.

```python
def compute_heterogeneity_metrics(
    site_ates: pd.DataFrame
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `site_ates` | `DataFrame` | Must have `'ate'` and `'ate_se'` columns |

**Returns:**

```python
{
    'n_sites': int,
    'pooled_ate_fe': float,        # Fixed-effect pooled estimate
    'pooled_se_fe': float,
    'pooled_ate_re': float,        # Random-effects pooled estimate
    'pooled_se_re': float,
    'Q_statistic': float,          # Cochran's Q
    'Q_df': int,
    'Q_pvalue': float,
    'I2': float,                   # I² statistic (%)
    'tau2': float,                 # Between-study variance
    'tau': float,
    'prediction_interval_lower': float,
    'prediction_interval_upper': float,
    'ate_range': tuple,
    'ate_mean': float,
    'ate_std': float
}
```

**Interpretation of I²:**
- `I² < 25%`: Low heterogeneity
- `25% <= I² < 50%`: Moderate heterogeneity
- `50% <= I² < 75%`: Substantial heterogeneity
- `I² >= 75%`: Considerable heterogeneity

**Example:**

```python
from causal_methods import compute_heterogeneity_metrics

site_results = pd.DataFrame({
    'site': ['A', 'B', 'C', 'D'],
    'ate': [1.5, 1.8, 1.2, 1.6],
    'ate_se': [0.2, 0.3, 0.25, 0.22]
})

metrics = compute_heterogeneity_metrics(site_results)
print(f"Pooled ATE (RE): {metrics['pooled_ate_re']:.2f}")
print(f"I² statistic: {metrics['I2']:.1f}%")
print(f"Prediction interval: [{metrics['prediction_interval_lower']:.2f}, {metrics['prediction_interval_upper']:.2f}]")
```
