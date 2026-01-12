# OSRCT Module API Reference

`osrct.py` - Observational Sampling from Randomized Controlled Trials

---

## Classes

### OSRCTSampler

Main class for generating confounded observational datasets from RCT data.

```python
class OSRCTSampler(
    biasing_covariates: List[str],
    biasing_coefficients: Optional[Dict[str, float]] = None,
    intercept: float = 0.0,
    standardize: bool = True,
    random_seed: Optional[int] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `biasing_covariates` | `List[str]` | Required | Names of pre-treatment covariates to use for biasing |
| `biasing_coefficients` | `Dict[str, float]` | `None` | Mapping of covariate names to coefficients. If None, random coefficients are generated |
| `intercept` | `float` | `0.0` | Intercept term for biasing function |
| `standardize` | `bool` | `True` | Whether to standardize covariates before applying biasing function |
| `random_seed` | `int` | `None` | Random seed for reproducibility |

#### Methods

##### `sample()`

Generate confounded observational sample from RCT data.

```python
def sample(
    self,
    rct_data: pd.DataFrame,
    treatment_col: str = 'iv',
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rct_data` | `DataFrame` | Required | Original RCT dataset with treatment, outcome, and covariates |
| `treatment_col` | `str` | `'iv'` | Name of treatment column (must be binary 0/1) |
| `verbose` | `bool` | `True` | Whether to print sampling statistics |

**Returns:**

| Return | Type | Description |
|--------|------|-------------|
| `observational_sample` | `DataFrame` | Biased observational dataset with added columns `_selection_prob` and `_preferred_treatment` |
| `selection_probs` | `ndarray` | Selection probability for each unit in original RCT |

**Example:**

```python
from osrct import OSRCTSampler

sampler = OSRCTSampler(
    biasing_covariates=['resp_age', 'resp_gender'],
    biasing_coefficients={'resp_age': 0.5, 'resp_gender': 0.8},
    random_seed=42
)

obs_data, probs = sampler.sample(rct_data, treatment_col='iv')
```

**Output:**
```
OSRCT Sampling Results:
  Original RCT size: 5,362
  Sampled size: 2,681 (50.0%)
  Selection prob range: [0.182, 0.818]
  Selection prob mean: 0.500
  Treatment group: 1,423 (53.1%)
  Control group: 1,258 (46.9%)
```

---

##### `get_confounding_strength()`

Measure confounding strength by computing correlations between biasing covariates and outcome.

```python
def get_confounding_strength(
    self,
    rct_data: pd.DataFrame,
    outcome_col: str = 'dv'
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rct_data` | `DataFrame` | Required | RCT dataset |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |

**Returns:**

`Dict[str, float]` - Correlation between each biasing covariate and outcome

**Example:**

```python
correlations = sampler.get_confounding_strength(rct_data, outcome_col='dv')
# {'resp_age': 0.15, 'resp_gender': -0.08}
```

---

## Functions

### select_biasing_covariates()

Select biasing covariates based on correlation with outcome.

```python
def select_biasing_covariates(
    rct_data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    candidate_covariates: Optional[List[str]] = None,
    min_correlation: float = 0.1,
    max_covariates: Optional[int] = None
) -> List[str]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rct_data` | `DataFrame` | Required | RCT dataset |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `candidate_covariates` | `List[str]` | `None` | List of candidate covariates. If None, uses all numeric columns except treatment/outcome |
| `min_correlation` | `float` | `0.1` | Minimum absolute correlation with outcome to be selected |
| `max_covariates` | `int` | `None` | Maximum number of covariates to select |

**Returns:**

`List[str]` - Selected biasing covariates, ordered by correlation strength

**Example:**

```python
from osrct import select_biasing_covariates

selected = select_biasing_covariates(
    rct_data,
    candidate_covariates=['resp_age', 'resp_gender', 'resp_polideo'],
    min_correlation=0.05,
    max_covariates=3
)
# ['resp_age', 'resp_polideo', 'resp_gender']
```

---

### evaluate_osrct_sample()

Evaluate OSRCT sample by comparing to original RCT.

```python
def evaluate_osrct_sample(
    rct_data: pd.DataFrame,
    obs_data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: Optional[List[str]] = None
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rct_data` | `DataFrame` | Required | Original RCT dataset |
| `obs_data` | `DataFrame` | Required | OSRCT observational sample |
| `treatment_col` | `str` | `'iv'` | Name of treatment column |
| `outcome_col` | `str` | `'dv'` | Name of outcome column |
| `covariates` | `List[str]` | `None` | Covariates to check balance for |

**Returns:**

`Dict[str, float]` with keys:

| Key | Description |
|-----|-------------|
| `rct_ate` | True ATE from RCT |
| `rct_ate_se` | Standard error of RCT ATE |
| `obs_ate_naive` | Naive observational ATE (biased) |
| `obs_ate_se` | Standard error of observational ATE |
| `confounding_bias` | Difference between naive and true |
| `sample_size_rct` | Original RCT size |
| `sample_size_obs` | Observational sample size |
| `sample_retention_rate` | Proportion retained |
| `rct_treatment_rate` | Treatment rate in RCT |
| `obs_treatment_rate` | Treatment rate in observational data |
| `covariate_balance` | SMD values per covariate (if provided) |

**Example:**

```python
from osrct import evaluate_osrct_sample

metrics = evaluate_osrct_sample(
    rct_data, obs_data,
    covariates=['resp_age', 'resp_gender']
)
print(f"True ATE: {metrics['rct_ate']:.2f}")
print(f"Naive ATE: {metrics['obs_ate_naive']:.2f}")
print(f"Confounding Bias: {metrics['confounding_bias']:.2f}")
```

---

### load_manylabs1_data()

Load ManyLabs1 data from pickle, CSV, or RData file.

```python
def load_manylabs1_data(
    data_path: str,
    study_filter: Optional[Union[str, List[str]]] = None,
    site_filter: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | Required | Path to data file (.pkl, .csv, or .RData) |
| `study_filter` | `str` or `List[str]` | `None` | Filter to specific studies |
| `site_filter` | `str` or `List[str]` | `None` | Filter to specific sites |

**Returns:**

`DataFrame` - Loaded and filtered ManyLabs1 data

**Supported formats:**
- `.pkl` - Python pickle (fastest)
- `.csv` - CSV format
- `.RData` - R data format (requires pyreadr)

**Example:**

```python
from osrct import load_manylabs1_data

# Load all data
data = load_manylabs1_data('Manylabs1_data.pkl')

# Load specific study
anchoring = load_manylabs1_data(
    'Manylabs1_data.pkl',
    study_filter='anchoring1'
)

# Load multiple studies from specific sites
subset = load_manylabs1_data(
    'Manylabs1_data.pkl',
    study_filter=['anchoring1', 'anchoring2'],
    site_filter=['mturk', 'abington']
)
```

---

### load_pipeline_data()

Load Pipeline data from processed CSV files.

```python
def load_pipeline_data(
    data_dir: str,
    study_id: Optional[Union[int, List[int]]] = None
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str` | Required | Directory containing processed Pipeline CSV files |
| `study_id` | `int` or `List[int]` | `None` | Filter to specific study IDs (5, 7, 8) |

**Returns:**

`DataFrame` - Loaded and filtered Pipeline data

**Available studies:**
- `5`: Moral Inversion
- `7`: Intuitive Economics
- `8`: Burn in Hell (within-subjects, not suitable for OSRCT)

**Example:**

```python
from osrct import load_pipeline_data

# Load study 7
data = load_pipeline_data('Pipeline/pre-process/', study_id=7)
```

---

## Algorithm Details

### OSRCT Sampling Algorithm

The OSRCT procedure implements Algorithm 2 from Gentzel et al. (2021):

```
Input: D_RCT (RCT dataset), C^b (biasing covariates)
Output: D_OSRCT (confounded observational dataset)

For each unit i in D_RCT:
    1. Compute selection probability: p_i = sigmoid(β₀ + Σ(βⱼ * C^b_ij))
    2. Sample preferred treatment: t_s ~ Bernoulli(p_i)
    3. If actual treatment T_i == t_s:
         Include unit i in D_OSRCT
       Else:
         Discard unit i
```

**Key Properties:**
- Expected sample size: ~50% of original RCT (for balanced treatment)
- Ground-truth ATE is preserved from original RCT
- Naive estimates from D_OSRCT will be biased
- Confounding strength controlled by β coefficients

### Biasing Function

The selection probability is computed using a logistic function:

```
P(T=1|C) = 1 / (1 + exp(-β₀ - Σ(βⱼ * Cⱼ)))
```

Where:
- `β₀`: Intercept (controls baseline selection rate)
- `βⱼ`: Coefficient for covariate j (controls confounding strength)
- `Cⱼ`: Standardized covariate values

Higher absolute values of βⱼ produce stronger confounding.
