# Experiment 2: Coverage Calibration Summary

**Author:** Andreas Koukorinis

## Results by DGP Difficulty

| DGP | N | Coverage | CI Lower | CI Upper | Width (median) | Width (mean) |
|-----|---|----------|----------|----------|----------------|--------------|
| Easy | 100 | **99%** | 94.6% | 99.8% | 0.100 | 0.100 |
| Medium | 100 | **100%** | 96.3% | 100% | 0.268 | 0.267 |
| Hard | 100 | **100%** | 96.3% | 100% | 0.590 | 0.589 |

## CATE Coverage

| DGP | CATE Z=0 | CATE Z=1 | Average |
|-----|----------|----------|---------|
| Easy | 98% | 97% | 97.5% |
| Medium | 100% | 100% | 100% |
| Hard | 100% | 100% | 100% |

## Width Statistics

| DGP | Min | Max | Std |
|-----|-----|-----|-----|
| Easy | 0.047 | 0.146 | 0.027 |
| Medium | 0.183 | 0.332 | 0.038 |
| Hard | 0.483 | 0.686 | 0.051 |

## LP Convergence

| DGP | Convergence Rate |
|-----|------------------|
| Easy | 99% |
| Medium | 100% |
| Hard | 100% |

## Success Criteria

| Criterion | Passed |
|-----------|--------|
| Easy coverage ≥ 95% | Yes |
| Medium coverage ≥ 95% | Yes |
| Hard coverage ≥ 95% | Yes |
| Easy width reasonable | Yes |
| Medium width reasonable | Yes |
| Hard width reasonable | Yes |

**All criteria passed: YES**
