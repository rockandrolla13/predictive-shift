# Experiment 2: Coverage Calibration

**Author:** Andreas Koukorinis

## Overview

This experiment verifies that the LP bounds method achieves nominal coverage rate (~95%) across multiple replications and difficulty levels. It establishes the statistical reliability of the bounds method by running 100 Monte Carlo replications for each of three DGP configurations.

## Research Question

Does the LP bounds method maintain valid coverage (>=95%) across:
- Different DGP difficulty levels (Easy, Medium, Hard)
- Multiple random seeds
- Varying instrument strengths and confounding levels

## Theoretical Background

For partial identification bounds to be valid, they must satisfy:

```
P(true_parameter in [lower, upper]) >= 1 - alpha
```

The bounds are derived from the naturalness constraint which allows epsilon tolerance in cross-site variation. Under correct specification, coverage should be at least 95% (for appropriately calibrated epsilon).

### Coverage-Width Tradeoff

There is a fundamental tradeoff:
- **Smaller epsilon:** Tighter bounds, but potentially undercoverage if sites have genuine heterogeneity
- **Larger epsilon:** Wider bounds (less informative), but guaranteed coverage

This experiment calibrates epsilon per DGP to achieve 95% coverage.

## Data Generating Process

Three difficulty levels with increasing challenge:

### Easy DGP
```python
DGPConfig(
    name='Easy',
    n_per_env=2000,
    alpha_z=4.0,    # Strong instrument
    alpha_u=0.1,    # Minimal confounding
    beta_u=0.1,
    epsilon=0.05    # Tight tolerance
)
```

### Medium DGP
```python
DGPConfig(
    name='Medium',
    n_per_env=1000,
    alpha_z=2.0,    # Moderate instrument
    alpha_u=0.5,    # Moderate confounding
    beta_u=0.5,
    epsilon=0.10    # Moderate tolerance
)
```

### Hard DGP
```python
DGPConfig(
    name='Hard',
    n_per_env=400,
    alpha_z=1.0,    # Weak instrument
    alpha_u=1.0,    # Strong confounding
    beta_u=1.0,
    epsilon=0.20    # Relaxed tolerance
)
```

## Experimental Design

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_replications` | 100 | Seeds per DGP configuration |
| `K` | 5 | Training environments |
| `beta_x` | 1.0 | True causal effect |
| **Total runs** | **300** | 100 seeds x 3 DGPs |

### Epsilon Calibration

Epsilon values are calibrated per DGP to achieve ~95% coverage:

| DGP | epsilon | Rationale |
|-----|---------|-----------|
| Easy | 0.05 | Low cross-site variance allows tight bounds |
| Medium | 0.10 | Moderate variance requires relaxation |
| Hard | 0.20 | High variance requires significant relaxation |

## Method

### Step 1: Configure DGPs

```python
DGP_CONFIGS = {
    'Easy': DGPConfig(
        name='Easy',
        n_per_env=2000,
        alpha_z=4.0,
        alpha_u=0.1,
        beta_u=0.1,
        description='Strong instrument, minimal confounding, large sample'
    ),
    'Medium': DGPConfig(
        name='Medium',
        n_per_env=1000,
        alpha_z=2.0,
        alpha_u=0.5,
        beta_u=0.5,
        description='Moderate instrument, moderate confounding'
    ),
    'Hard': DGPConfig(
        name='Hard',
        n_per_env=400,
        alpha_z=1.0,
        alpha_u=1.0,
        beta_u=1.0,
        description='Weak instrument, strong confounding, small sample'
    )
}

EPSILON_BY_DGP = {
    'Easy': 0.05,
    'Medium': 0.10,
    'Hard': 0.20
}
```

### Step 2: Run Replications

```python
for dgp_name, dgp_config in DGP_CONFIGS.items():
    for seed in range(1, 101):  # 100 replications
        result = run_single_replication(
            dgp_name=dgp_name,
            dgp_config=dgp_config,
            seed=seed,
            epsilon=EPSILON_BY_DGP[dgp_name],
            beta_x=1.0,
            K=5
        )
        results.append(result)
```

### Step 3: Compute Coverage

```python
# Coverage rate per DGP
for dgp_name in DGP_CONFIGS.keys():
    dgp_results = results_df[results_df['dgp'] == dgp_name]
    coverage_rate = dgp_results['coverage_ate'].mean()

    # Wilson score confidence interval
    n = len(dgp_results)
    z = 1.96
    p = coverage_rate
    ci_lower = (p + z**2/(2*n) - z*sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    ci_upper = (p + z**2/(2*n) + z*sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
```

## Success Criteria

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Easy coverage | >= 95% | Strong signal should yield valid bounds |
| Medium coverage | >= 95% | Moderate conditions should still work |
| Hard coverage | >= 90% | Challenging conditions may underperform |
| Easy width | <= 0.15 | Tight bounds expected |
| Medium width | <= 0.35 | Moderate width expected |
| Hard width | <= 0.70 | Wide bounds acceptable |

## Reproducibility

### Running the Experiment

```bash
cd /path/to/predictive-shift
python experiments/exp2_coverage_calibration.py
```

### Expected Output

```
============================================================
EXPERIMENT 2: Coverage Calibration
============================================================

Configuration:
  n_replications: 100
  K: 5
  epsilon: 0.05
  beta_x: 1.0

Running 300 replications...
  100 seeds x 3 DGPs
  epsilon = 0.05

DGP: Easy (Strong instrument, minimal confounding, large sample)
  Completed 50/300 replications...
  Completed 100/300 replications...
DGP: Medium (Moderate instrument, moderate confounding)
  Completed 150/300 replications...
  Completed 200/300 replications...
DGP: Hard (Weak instrument, strong confounding, small sample)
  Completed 250/300 replications...
  Completed 300/300 replications...

Completed in 45.3 seconds

============================================================
EXPERIMENT 2: Coverage Calibration Summary
============================================================

Easy DGP:
  Replications: 100
  ATE Coverage: 99.0% [94.6%, 99.8%]
  CATE Coverage (Z=0): 98.0%
  CATE Coverage (Z=1): 97.0%
  Median Width: 0.1000
  LP Convergence: 99.0%

Medium DGP:
  Replications: 100
  ATE Coverage: 100.0% [96.3%, 100.0%]
  CATE Coverage (Z=0): 100.0%
  CATE Coverage (Z=1): 100.0%
  Median Width: 0.2680
  LP Convergence: 100.0%

Hard DGP:
  Replications: 100
  ATE Coverage: 100.0% [96.3%, 100.0%]
  CATE Coverage (Z=0): 100.0%
  CATE Coverage (Z=1): 100.0%
  Median Width: 0.5900
  LP Convergence: 100.0%

------------------------------------------------------------
Success Criteria:
  Easy_coverage: PASS
  Medium_coverage: PASS
  Hard_coverage: PASS
  Easy_width: PASS
  Medium_width: PASS
  Hard_width: PASS
------------------------------------------------------------
ALL PASSED: YES
============================================================
```

### Minimal Reproducible Example

```python
"""Minimal example for Experiment 2 - single DGP."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/path/to/predictive-shift')

from causal_grounding.confounded_instrument_dgp import (
    ConfoundedInstrumentDGP,
    generate_confounded_instrument_data,
    logistic
)
from causal_grounding.lp_solver import solve_all_bounds_binary_lp

def run_replication(alpha_z, alpha_u, beta_u, epsilon, n_per_env, K, seed):
    """Run a single replication and return coverage."""
    dgp = ConfoundedInstrumentDGP(
        alpha_z=alpha_z, alpha_u=alpha_u, beta_u=beta_u,
        beta_x=1.0, prob_u=0.5, prob_z=0.5
    )

    # Generate data
    np.random.seed(seed)
    training_data = {}
    for k in range(K):
        # Observational
        idle = generate_confounded_instrument_data(dgp, n_per_env, seed=seed+k*1000)
        idle['F'] = 'idle'
        # Experimental (randomized X)
        np.random.seed(seed + k*1000 + 500)
        Z = np.random.binomial(1, 0.5, n_per_env)
        X = np.random.binomial(1, 0.5, n_per_env)
        U = np.random.binomial(1, 0.5, n_per_env)
        Y = np.random.binomial(1, logistic(dgp.beta_x * X + dgp.beta_u * U), n_per_env)
        on = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'F': 'on'})
        training_data[f'env_{k}'] = pd.concat([idle, on], ignore_index=True)

    # Compute bounds
    bounds = solve_all_bounds_binary_lp(
        training_data, ['Z'], 'X', 'Y', epsilon, 'F'
    )

    if not bounds:
        return None

    # Check coverage
    true_ate = 0.22  # Approximate for these parameters
    ate_lower = 0.5 * bounds[(0,)][0] + 0.5 * bounds[(1,)][0]
    ate_upper = 0.5 * bounds[(0,)][1] + 0.5 * bounds[(1,)][1]

    return ate_lower <= true_ate <= ate_upper

# Run 20 replications for Easy DGP
coverages = []
for seed in range(1, 21):
    cov = run_replication(
        alpha_z=4.0, alpha_u=0.1, beta_u=0.1,
        epsilon=0.05, n_per_env=2000, K=5, seed=seed
    )
    if cov is not None:
        coverages.append(cov)

print(f"Coverage rate: {np.mean(coverages):.1%}")
```

## Results

See `results/exp2_summary.md` for detailed output.

### Key Findings

| DGP | Coverage | 95% CI | Width (median) |
|-----|----------|--------|----------------|
| Easy | 99% | [94.6%, 99.8%] | 0.100 |
| Medium | 100% | [96.3%, 100%] | 0.268 |
| Hard | 100% | [96.3%, 100%] | 0.590 |

**All criteria passed.** The bounds method achieves valid coverage across all difficulty levels when epsilon is appropriately calibrated.

## Visualizations

### Coverage Distribution

```
Coverage by DGP:
  Easy   |#################### 99%
  Medium |#################### 100%
  Hard   |#################### 100%
         0%              50%             100%
```

### Width vs. Coverage Tradeoff

```
Width (median) by DGP:
  Easy   |##     0.10
  Medium |#####  0.27
  Hard   |########## 0.59
         0.0    0.2    0.4    0.6    0.8
```

## Output Files

| File | Description |
|------|-------------|
| `results/exp2_results.csv` | Raw results for all 300 runs |
| `results/exp2_summary.md` | Summary statistics in markdown |

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
```

## Interpretation

1. **Coverage meets target:** All DGPs achieve >= 95% coverage with calibrated epsilon
2. **Width scales with difficulty:** Harder problems require wider bounds (more uncertainty)
3. **LP convergence is robust:** 99-100% convergence across all conditions
4. **Epsilon calibration is critical:** Must adjust epsilon based on DGP characteristics

## References

- Imbens, G. W., & Manski, C. F. (2004). Confidence intervals for partially identified parameters.
- Stoye, J. (2009). More on confidence intervals for partially identified parameters.
