# Experiment 3: Instrument Strength Sweep Summary

**Author:** Andreas Koukorinis

## Key Finding

**Width is CONSTANT across all αz values.** Instrument strength has no effect on bound width.

## Width by Epsilon (across all αz)

| ε | Width (mean) | Width (median) | Std | Coverage | Feasibility |
|---|--------------|----------------|-----|----------|-------------|
| 0.05 | 0.103 | 0.100 | 0.017 | 96% | 96% |
| 0.10 | 0.301 | 0.298 | 0.018 | 100% | 100% |
| 0.15 | 0.501 | 0.498 | 0.018 | 100% | 100% |
| 0.20 | 0.701 | 0.698 | 0.018 | 100% | 100% |

## Instrument Strength by αz

| αz | P(X=1|Z=1) - P(X=1|Z=0) |
|----|-------------------------|
| 0.25 | 0.058 (weak) |
| 0.50 | 0.115 |
| 0.75 | 0.167 |
| 1.00 | 0.213 |
| 1.50 | 0.288 |
| 2.00 | 0.341 |
| 2.50 | 0.376 |
| 3.00 | 0.400 |
| 4.00 | 0.424 (strong) |

## Width Threshold Analysis

No αz value causes width to exceed 0.8 threshold for any ε tested.

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total runs | 1,800 |
| Overall feasibility | 99% |
| Overall coverage | 100% |
| Width range | [0.073, 0.740] |

## Interpretation

The LP only uses F=on (experimental) data where treatment is randomized. Instrument strength affects observational data selection but NOT the experimental data used for bounds.

**Width ≈ 2ε + noise** is the dominant relationship.
