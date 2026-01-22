# Experiment 1: Proof of Concept Results

**Author:** Andreas Koukorinis

## CATE Bounds

| Covariate | Lower | Upper |
|-----------|-------|-------|
| Z=0 | 0.194 | 0.309 |
| Z=1 | 0.195 | 0.272 |

## ATE Bounds

| Metric | Value |
|--------|-------|
| Lower | 0.195 |
| Upper | 0.290 |
| Width | 0.096 |
| True ATE | 0.228 |

## Coverage

| Target | Covered |
|--------|---------|
| ATE | Yes |
| CATE (Z=0) | Yes |
| CATE (Z=1) | Yes |

## Diagnostics

| Metric | Value |
|--------|-------|
| Instrument strength | 0.461 |
| Naive ATE | 0.224 |
| Naive bias | -0.004 |
| LP status | optimal |
| Runtime | 0.04s |

## Success Criteria

**All criteria passed: YES**
