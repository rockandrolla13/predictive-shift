# OSRCT Benchmark Leaderboard

**Last Updated:** 2025-12-08

This leaderboard tracks the performance of causal inference methods on the OSRCT Benchmark.

---

## Overall Rankings (by RMSE)

| Rank | Method | Type | RMSE | Bias | Abs Bias | Coverage | Paper |
|:----:|--------|------|-----:|-----:|---------:|---------:|:-----:|
| 1 | IPW | Propensity | 23.24 | -4.03 | 9.45 | 0.2% | - |
| 2 | Outcome Regression | Outcome | 24.20 | +5.67 | 9.99 | 0.2% | - |
| 3 | AIPW | Doubly Robust | 24.33 | -5.15 | 9.86 | 0.2% | - |
| 4 | Naive | Unadjusted | 28.72 | +0.90 | 9.16 | 0.2% | - |
| 5 | PSM | Matching | 79.49 | +9.75 | 33.41 | 0.2% | - |

---

## Performance by Confounding Strength

### RMSE by Beta

| Method | β=0.1 | β=0.25 | β=0.5 | β=0.75 | β=1.0 | β=1.5 | β=2.0 |
|--------|------:|-------:|------:|-------:|------:|------:|------:|
| IPW | 13.52 | 15.28 | 17.58 | 17.32 | 27.28 | 25.02 | 37.23 |
| Outcome Reg. | 13.48 | 15.23 | 15.42 | 17.19 | 21.03 | 31.33 | 41.56 |
| AIPW | 13.49 | 15.24 | 17.63 | 17.30 | 25.69 | 27.05 | 41.56 |
| Naive | 12.17 | 12.76 | 13.91 | 20.60 | 26.44 | 39.57 | 50.80 |
| PSM | 68.80 | 55.12 | 66.97 | 66.78 | 79.48 | 93.27 | 111.78 |

---

## Performance by Covariate Pattern

| Method | age | gender | polideo | demo_basic | demo_full |
|--------|----:|-------:|--------:|-----------:|----------:|
| IPW | 19.68 | 21.48 | 20.97 | 25.31 | 27.79 |
| Outcome Reg. | 19.67 | 22.68 | 19.90 | 31.97 | 24.65 |
| AIPW | 23.96 | 21.30 | 20.83 | 26.62 | 28.09 |
| Naive | 27.49 | 46.04 | 13.17 | 26.70 | 19.04 |
| PSM | 70.63 | 95.21 | 62.14 | 72.53 | 91.75 |

---

## How to Submit Your Method

1. **Implement your method** following the standard interface:

```python
def your_method(data, treatment_col, outcome_col, covariates):
    """
    Parameters
    ----------
    data : pd.DataFrame
    treatment_col : str (default 'iv')
    outcome_col : str (default 'dv')
    covariates : list of str

    Returns
    -------
    dict with keys: 'ate', 'se', 'ci_lower', 'ci_upper'
    """
    # Your implementation
    return {'ate': ..., 'se': ..., 'ci_lower': ..., 'ci_upper': ...}
```

2. **Test on sample datasets**:

```python
from osrct_benchmark import load_dataset, evaluate_method

data, meta = load_dataset('anchoring1', beta=0.5, pattern='age')
result = your_method(data, 'iv', 'dv', ['resp_age'])
print(f"Estimated ATE: {result['ate']:.2f}, True ATE: {meta['ground_truth_ate']:.2f}")
```

3. **Open a submission issue** using the Method Submission template

4. **Include benchmark results** from running on the full dataset suite

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| RMSE | √(Σ(τ̂-τ)²/n) | Root mean squared error |
| Bias | E[τ̂] - τ | Average deviation from truth |
| Abs Bias | E[|τ̂-τ|] | Average absolute error |
| Coverage | P(τ ∈ CI) | 95% CI contains true ATE |
| CI Width | E[CI_upper - CI_lower] | Average confidence interval width |

---

## Method Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Propensity-based** | Model P(T\|X) | IPW, PSM |
| **Outcome-based** | Model E[Y\|T,X] | Outcome Regression |
| **Doubly Robust** | Model both | AIPW, TMLE |
| **ML-based** | Machine learning | Causal Forest, BART |
| **Hybrid** | Custom combinations | Meta-learners |

---

## Benchmark Statistics

- **Total Datasets:** 525
- **Studies:** 15
- **Confounding Patterns:** 5
- **Beta Values:** 7 (0.1 to 2.0)
- **Observations:** ~1.3 million total

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{osrct_benchmark_2025,
  title={OSRCT Benchmark: Semi-Synthetic Datasets for Causal Inference Evaluation},
  author={[Authors]},
  year={2025}
}
```
