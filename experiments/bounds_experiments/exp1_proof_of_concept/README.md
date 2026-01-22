# Experiment 1: Proof of Concept

**Author:** Andreas Koukorinis

## Overview

This experiment validates that the LP bounds method for partial identification is fundamentally correct under ideal conditions. It serves as a sanity check before running more complex experiments.

## Research Question

Can the LP bounds method produce tight bounds with valid coverage in a controlled setting with:
- Strong instrument
- Minimal confounding
- Large sample size
- Binary outcomes

## Theoretical Background

The bounds method computes partial identification bounds on the Conditional Average Treatment Effect (CATE) using cross-site naturalness constraints. For each covariate stratum z, the bounds satisfy:

```
CATE(z) = E[Y(1) - Y(0) | Z=z] in [lower(z), upper(z)]
```

The LP formulation enforces that the target conditional means satisfy:
```
|theta_target - theta_k| <= epsilon  for all sites k
```

where `theta_k` is the observed conditional mean at site k and epsilon is the naturalness tolerance.

## Data Generating Process (DGP)

The experiment uses `ConfoundedInstrumentDGP` with the following structural equations:

```
Z ~ Bernoulli(0.5)                    # Instrument
U ~ Bernoulli(0.5)                    # Unobserved confounder
X = sigma(alpha_z * Z + alpha_u * U)  # Treatment (confounded)
Y = sigma(beta_x * X + beta_u * U)    # Outcome (binary)
```

Where `sigma` is the logistic function.

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_per_env` | 2,000 | Samples per environment |
| `K` | 5 | Number of training environments |
| `alpha_z` | 4.0 | Instrument strength (strong) |
| `alpha_u` | 0.1 | Treatment confounding (minimal) |
| `beta_x` | 1.0 | True causal effect |
| `beta_u` | 0.1 | Outcome confounding (minimal) |
| `epsilon` | 0.05 | Naturalness tolerance |
| `seed` | 42 | Random seed |

### Multi-Environment Data Structure

Each environment k contains:
- **F=idle (observational):** Treatment X depends on instrument Z and confounder U
- **F=on (experimental):** Treatment X is randomized (50/50), independent of Z and U

The LP bounds use only F=on data where treatment is randomized, ensuring the bounds are protected from confounding.

## Method

### Step 1: Data Generation

```python
from causal_grounding.confounded_instrument_dgp import ConfoundedInstrumentDGP

dgp = ConfoundedInstrumentDGP(
    alpha_z=4.0,   # Strong instrument
    alpha_u=0.1,   # Minimal confounding
    beta_x=1.0,    # True effect
    beta_u=0.1,    # Minimal outcome confounding
)

# Generate K=5 environments with n=2000 per environment
training_data = generate_multi_environment_data(dgp, n_per_env=2000, K=5, seed=42)
```

### Step 2: Compute Ground Truth

```python
from causal_grounding.confounded_instrument_dgp import compute_ground_truth_effects

effects = compute_ground_truth_effects(dgp)
true_ate = effects['ate']  # Monte Carlo estimate
```

### Step 3: Solve LP Bounds

```python
from causal_grounding.lp_solver import solve_all_bounds_binary_lp

bounds = solve_all_bounds_binary_lp(
    training_data=training_data,
    covariates=['Z'],
    treatment='X',
    outcome='Y',
    epsilon=0.05,
    regime_col='F',
    use_cvxpy=False  # Use closed-form for speed
)
```

### Step 4: Evaluate Coverage

```python
# ATE bounds (weighted average over Z)
ate_lower = 0.5 * bounds[(0,)][0] + 0.5 * bounds[(1,)][0]
ate_upper = 0.5 * bounds[(0,)][1] + 0.5 * bounds[(1,)][1]

coverage = ate_lower <= true_ate <= ate_upper
width = ate_upper - ate_lower
```

## Success Criteria

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Coverage | True (bounds contain true ATE) | Primary validity check |
| Width | < 0.30 | Tight bounds indicate informative identification |
| LP Status | Optimal | No numerical issues |

## Reproducibility

### Running the Experiment

```bash
cd /path/to/predictive-shift
python experiments/exp1_proof_of_concept.py
```

### Expected Output

```
============================================================
EXPERIMENT 1: Proof of Concept
Bounds Validation Under Ideal Conditions
============================================================

Configuration:
  n_per_env: 2000
  alpha_z: 4.0
  alpha_u: 0.1
  beta_x: 1.0
  beta_u: 0.1
  K: 5
  seed: 42
  epsilon: 0.05

DGP: Scenario [4.0, 0.1, 0.0] - IV moderately strong, minimal confounding
  Instrument strength (alpha_z): 4.0
  Confounding (alpha_u, beta_u): (0.1, 0.1)

Generating data for 5 environments...
  Total samples: 20000

Computing ground truth...
  True ATE: 0.2196
  True CATE(z=0): 0.2196
  True CATE(z=1): 0.2196

Solving LP bounds (epsilon=0.05)...

Bounds computed:
  CATE(z=0): [0.1712, 0.2648] (width=0.0936)
  CATE(z=1): [0.1688, 0.2688] (width=0.1000)
  ATE: [0.1700, 0.2668] (width=0.0968)

Coverage:
  ATE covered: True
  CATE(z=0) covered: True
  CATE(z=1) covered: True

Success Criteria:
  Width < 0.30: True (actual: 0.0968)
  Coverage: True
  LP Status: optimal
  ALL PASSED: True
```

### Minimal Reproducible Example

```python
"""Minimal example for Experiment 1."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/path/to/predictive-shift')

from causal_grounding.confounded_instrument_dgp import (
    ConfoundedInstrumentDGP,
    generate_confounded_instrument_data,
    compute_ground_truth_effects,
    logistic
)
from causal_grounding.lp_solver import solve_all_bounds_binary_lp

# 1. Create DGP
dgp = ConfoundedInstrumentDGP(
    alpha_0=0.0, alpha_z=4.0, alpha_u=0.1, alpha_w=0.0,
    beta_0=0.0, beta_x=1.0, beta_u=0.1, beta_w=0.0, beta_z=0.0,
    prob_u=0.5, prob_z=0.5
)

# 2. Generate multi-environment data
np.random.seed(42)
training_data = {}
for k in range(5):
    # Observational data (F=idle)
    idle = generate_confounded_instrument_data(dgp, 2000, seed=42+k*1000)
    idle['F'] = 'idle'

    # Experimental data (F=on) - X randomized
    np.random.seed(42 + k*1000 + 500)
    Z = np.random.binomial(1, 0.5, 2000)
    X = np.random.binomial(1, 0.5, 2000)  # Randomized!
    U = np.random.binomial(1, 0.5, 2000)
    Y = np.random.binomial(1, logistic(dgp.beta_x * X + dgp.beta_u * U), 2000)
    on = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'F': 'on'})

    training_data[f'env_{k}'] = pd.concat([idle, on], ignore_index=True)

# 3. Compute bounds
bounds = solve_all_bounds_binary_lp(
    training_data=training_data,
    covariates=['Z'],
    treatment='X',
    outcome='Y',
    epsilon=0.05,
    regime_col='F'
)

# 4. Check coverage
true_ate = compute_ground_truth_effects(dgp)['ate']
ate_lower = 0.5 * bounds[(0,)][0] + 0.5 * bounds[(1,)][0]
ate_upper = 0.5 * bounds[(0,)][1] + 0.5 * bounds[(1,)][1]

print(f"True ATE: {true_ate:.4f}")
print(f"Bounds: [{ate_lower:.4f}, {ate_upper:.4f}]")
print(f"Coverage: {ate_lower <= true_ate <= ate_upper}")
```

## Results

See `results/exp1_results.md` for detailed output.

### Key Findings

1. **Coverage achieved:** The bounds contain the true ATE
2. **Tight bounds:** Width < 0.10 (informative identification)
3. **LP converged:** No numerical issues

## Output Files

| File | Description |
|------|-------------|
| `results/exp1_results.md` | Detailed results in markdown format |

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
```

## References

- Manski, C. F. (1990). Nonparametric bounds on treatment effects. American Economic Review.
- Balke, A., & Pearl, J. (1997). Bounds on treatment effects from studies with imperfect compliance.
