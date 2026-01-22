# Experiment 4: Confounding Sweep Summary

**Author:** Andreas Koukorinis

## Key Finding

**Width is CONSTANT across all (αu, βu) values.** Confounding has no effect on bound width.

## Width by Configuration

| (αu, βu) | Width (mean) | Coverage | Naive Bias |
|----------|--------------|----------|------------|
| (0.25, 0.25) | 0.302 | 100% | 0.001 |
| (0.25, 2.0) | 0.311 | 100% | 0.016 |
| (0.5, 0.5) | 0.301 | 100% | 0.008 |
| (1.0, 1.0) | 0.306 | 100% | 0.042 |
| (1.5, 1.5) | 0.308 | 100% | 0.088 |
| (2.0, 2.0) | 0.311 | 100% | 0.136 |

## ANOVA Results

| Coefficient | Estimate | Std Error | p-value |
|-------------|----------|-----------|---------|
| Intercept | 0.300 | 0.002 | < 0.001 |
| αu | ~0 | 0.001 | 1.000 |
| βu | 0.006 | 0.001 | < 0.001 |
| αu × βu | ~0 | 0.001 | 1.000 |

**R² = 0.043** (confounding explains only 4% of width variance)

## Dominance Test

| Test | Result |
|------|--------|
| βu coefficient | 0.006 |
| αu coefficient | ~0 |
| Difference | 0.006 |
| t-statistic | 5.29 |
| p-value (one-sided) | < 0.001 |
| **βu dominates** | Yes |

βu has a statistically significant effect, but the effect size is negligible (0.006 per unit βu).

## Naive Bias Increases with Confounding

| (αu, βu) | Naive Bias |
|----------|------------|
| (0.25, 0.25) | 0.001 |
| (1.0, 1.0) | 0.042 |
| (2.0, 2.0) | 0.136 |

This confirms confounding IS present in the data, but the LP bounds are protected because they use only experimental data.

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total runs | 1,250 |
| Overall feasibility | 100% |
| Overall coverage | 100% |
| Width range | [0.262, 0.346] |

## Interpretation

The LP uses only F=on (experimental) data where treatment is randomized. Confounding affects observational data but NOT the experimental data used for bounds.

**Width ≈ 2ε + noise** regardless of confounding strength.
