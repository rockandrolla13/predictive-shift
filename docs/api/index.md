# API Reference

Complete documentation for OSRCT Benchmark modules.

---

## Modules

| Module | Description |
|--------|-------------|
| [osrct](osrct.md) | OSRCT sampling algorithm and data loaders |
| [causal_methods](causal_methods.md) | Causal inference method implementations |
| [evaluation](evaluation.md) | Evaluation utilities and metrics |

---

## Quick Reference

### OSRCT Module (`osrct.py`)

**Classes:**
- `OSRCTSampler` - Main sampling class for generating confounded data

**Functions:**
- `select_biasing_covariates()` - Automated covariate selection
- `evaluate_osrct_sample()` - Compare RCT vs observational estimates
- `load_manylabs1_data()` - Load ManyLabs1 dataset
- `load_pipeline_data()` - Load Pipeline dataset

### Causal Methods Module (`causal_methods.py`)

**Estimators:**
- `estimate_naive()` - Difference-in-means
- `estimate_ipw()` - Inverse probability weighting
- `estimate_outcome_regression()` - G-computation
- `estimate_aipw()` - Augmented IPW (doubly robust)
- `estimate_psm()` - Propensity score matching
- `estimate_causal_forest()` - Causal forest (requires econml)

**Classes:**
- `CausalMethodEvaluator` - Unified evaluation interface

**Utilities:**
- `bootstrap_se()` - Bootstrap standard errors
- `compute_smd()` - Standardized mean difference
- `compute_heterogeneity_metrics()` - Meta-analysis metrics

---

## Common Patterns

### Standard Method Interface

All estimation methods follow this interface:

```python
def estimate_method(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    **method_specific_args
) -> Dict[str, Any]:
    """
    Returns
    -------
    dict with keys:
        'method': str - Method name
        'ate': float - Point estimate
        'se': float - Standard error
        'ci_lower': float - 95% CI lower bound
        'ci_upper': float - 95% CI upper bound
        ... method-specific outputs
    """
```

### Data Requirements

**Required columns:**
- `iv`: Treatment indicator (binary 0/1)
- `dv`: Outcome variable (continuous)

**Standard covariates:**
- `resp_age`: Respondent age
- `resp_gender`: Respondent gender (1=female, 0=male)
- `resp_polideo`: Political ideology (0=conservative to 6=liberal)

### Error Handling

Methods return `np.nan` values when estimation fails:

```python
result = estimate_ipw(data, 'iv', 'dv', covariates)
if np.isnan(result['ate']):
    print(f"Estimation failed: {result.get('error', 'Unknown error')}")
```
