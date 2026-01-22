# Experiment 7: Real Data Summary (Raw Outcomes)

**Author:** Andreas Koukorinis

## Configuration

| Parameter | Value |
|-----------|-------|
| Studies | anchoring1, gainloss, flag, reciprocity |
| Patterns | age, gender, demo_basic |
| Betas | 0.25, 0.5, 1.0 |
| Epsilons | 0.05, 0.10, 0.15, 0.20 |
| Total configs | 144 |
| Successful | 144 |

## Key Finding

**100% of bounds are inverted.** The naturalness assumption is violated.

## Coverage by Epsilon

| ε | Coverage | Width/SD (mean) | Expected |
|---|----------|-----------------|----------|
| 0.05 | 0% | -2.21 | 0.10 |
| 0.10 | 0% | -2.01 | 0.20 |
| 0.15 | 0% | -1.81 | 0.30 |
| 0.20 | 0% | -1.61 | 0.40 |

Note: Negative width ratios indicate inverted bounds (lower > upper).

## Coverage by Study

| Study | Coverage |
|-------|----------|
| anchoring1 | 0% |
| flag | 0% |
| gainloss | 0% |
| reciprocity | 0% |

## Coverage by Confounding (β)

| β | Coverage |
|---|----------|
| 0.25 | 0% |
| 0.5 | 0% |
| 1.0 | 0% |

## Hypothesis Tests

| Hypothesis | Result | Details |
|------------|--------|---------|
| H1: Coverage ≥ 95% (ε ≥ 0.10) | **FAIL** | 0% coverage |
| H2: Width ≈ 2ε | **FAIL** | r = 0.27 |
| H3: Coverage independent of β | N/A | All zero |

## Cross-Site Heterogeneity

| Metric | Value |
|--------|-------|
| Mean θ₁ range / SD(Y) | 0.90 |
| Mean θ₀ range / SD(Y) | 0.92 |
| % Inverted bounds | 100% |
| **Min ε required** | **0.96** |

## Naive Bias by β

| β | Naive Bias |
|---|------------|
| 0.25 | -2.03 |
| 0.5 | 0.46 |
| 1.0 | 10.60 |

## Conclusion

Raw outcomes have too much cross-site heterogeneity. Sites have different baseline means (1000+ units for anchoring studies), violating the naturalness assumption.

**Recommendation:** Use site-standardized outcomes.
