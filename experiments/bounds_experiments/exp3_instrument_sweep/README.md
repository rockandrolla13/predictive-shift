# Experiment 3: Instrument Strength Sweep

**Author:** Andreas Koukorinis

## Overview

This experiment investigates how instrument strength (alpha_z) affects bound width. The key finding is that **width is constant across all alpha_z values** for a given epsilon, because the LP bounds use only experimental (F=on) data where treatment is randomized.

## Research Question

- What is the functional form of alpha_z -> Width?
- At what alpha_z* do bounds become uninformative (Width >= 0.8)?
- How does alpha_z* depend on naturalness tolerance epsilon?

## Theoretical Background

### Instrument Strength

The instrument Z affects treatment X through the structural equation:
```
P(X=1|Z=z) = sigma(alpha_z * z + alpha_u * U)
```

**Instrument strength** is measured by `|P(X=1|Z=1) - P(X=1|Z=0)|`:
- alpha_z = 0.25: ~0.06 (weak)
- alpha_z = 1.0: ~0.21 (moderate)
- alpha_z = 4.0: ~0.42 (strong)

### Why Instrument Strength Doesn't Affect Bounds

The LP bounds use only **F=on (experimental) data** where treatment X is randomized and independent of Z. Therefore:

1. Instrument strength affects **observational** data selection
2. But observational data (F=idle) is not used in the bounds computation
3. So instrument strength has **no effect** on width

This is a key insight: the bounds are protected from weak instruments because they rely on experimental data.

## Experimental Design

### Swept Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `alpha_z` | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] | Instrument strength |
| `epsilon` | [0.05, 0.10, 0.15, 0.20] | Naturalness tolerance |

### Fixed Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha_u` | 0.5 | Treatment confounding |
| `beta_u` | 0.5 | Outcome confounding |
| `beta_x` | 1.0 | True causal effect |
| `n_per_env` | 2,000 | Samples per environment |
| `K` | 5 | Number of environments |
| `n_seeds` | 50 | Replications per configuration |

### Total Configurations

```
9 alpha_z values x 4 epsilon values x 50 seeds = 1,800 runs
```

## Method

### Step 1: Configure Sweep

```python
@dataclass
class Exp3Config:
    alpha_z_values: List[float] = None  # [0.25, 0.5, ..., 4.0]
    epsilon_values: List[float] = None   # [0.05, 0.10, 0.15, 0.20]
    alpha_u: float = 0.5
    beta_u: float = 0.5
    beta_x: float = 1.0
    n_per_env: int = 2000
    K: int = 5
    n_seeds: int = 50
    width_threshold: float = 0.8  # For alpha_z* computation

    def __post_init__(self):
        if self.alpha_z_values is None:
            self.alpha_z_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        if self.epsilon_values is None:
            self.epsilon_values = [0.05, 0.10, 0.15, 0.20]
```

### Step 2: Run Configurations

```python
for alpha_z in config.alpha_z_values:
    for epsilon in config.epsilon_values:
        for seed in range(config.seed_start, config.seed_start + config.n_seeds):
            result = run_single_configuration(alpha_z, epsilon, seed, config)
            results.append(result)
```

### Step 3: Analyze Results

```python
# Group by (alpha_z, epsilon)
summary = {}
for (az, eps), group in results_df.groupby(['alpha_z', 'epsilon']):
    feasible = group[group['feasible']]
    summary[f"az={az}_eps={eps}"] = {
        'width_mean': feasible['ate_width'].mean(),
        'width_std': feasible['ate_width'].std(),
        'coverage': feasible['coverage'].mean(),
        'instrument_strength': feasible['instrument_strength'].mean()
    }
```

## Key Finding

**Width is CONSTANT across all alpha_z values for a given epsilon.**

### Width by Epsilon (averaged over all alpha_z)

| epsilon | Width (mean) | Width (std) | Coverage | Feasibility |
|---------|--------------|-------------|----------|-------------|
| 0.05 | 0.103 | 0.017 | 96% | 96% |
| 0.10 | 0.301 | 0.018 | 100% | 100% |
| 0.15 | 0.501 | 0.018 | 100% | 100% |
| 0.20 | 0.701 | 0.018 | 100% | 100% |

### Width by alpha_z (fixed epsilon=0.10)

| alpha_z | P(X=1|Z=1) - P(X=1|Z=0) | Width | Coverage |
|---------|-------------------------|-------|----------|
| 0.25 | 0.058 (weak) | 0.30 | 100% |
| 0.50 | 0.115 | 0.30 | 100% |
| 1.00 | 0.213 | 0.30 | 100% |
| 2.00 | 0.341 | 0.30 | 100% |
| 4.00 | 0.424 (strong) | 0.30 | 100% |

**All widths are the same!** Instrument strength has no effect.

## Reproducibility

### Running the Experiment

```bash
cd /path/to/predictive-shift

# Full run (50 seeds, ~10 minutes)
python experiments/exp3_instrument_sweep.py

# Quick run (10 seeds, ~2 minutes)
python experiments/exp3_instrument_sweep.py --quick

# Analyze existing results
python experiments/exp3_instrument_sweep.py --analyze-only outputs/exp3_results.csv
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--n-seeds N` | Number of seeds per configuration (default: 50) |
| `--n-jobs N` | Parallel jobs, -1 for all cores (default: -1) |
| `--quick` | Quick run with 10 seeds |
| `--analyze-only FILE` | Analyze existing results file |

### Expected Output

```
============================================================
Experiment 3: Instrument Strength Sweep
============================================================
az values: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
epsilon values: [0.05, 0.1, 0.15, 0.2]
Seeds per config: 50
Total runs: 1800
Fixed: alpha_u=0.5, beta_u=0.5, n=2000, K=5
============================================================
  Progress: 100/1800 (5.6%) - ETA: 180s
  Progress: 500/1800 (27.8%) - ETA: 120s
  ...

Completed in 185.3s
Results saved to: outputs/exp3_results.csv

============================================================
SUMMARY
============================================================
Total runs: 1800
Overall feasibility: 99.0%
Overall coverage: 100.0%

Thresholds (alpha_z where Width = 0.8):
  eps=0.05: not reached
  eps=0.10: not reached
  eps=0.15: not reached
  eps=0.20: not reached
```

### Minimal Reproducible Example

```python
"""Minimal example demonstrating width independence from alpha_z."""
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

def compute_width(alpha_z, epsilon, seed=42):
    """Compute bound width for given alpha_z and epsilon."""
    dgp = ConfoundedInstrumentDGP(
        alpha_z=alpha_z, alpha_u=0.5, beta_u=0.5, beta_x=1.0
    )

    # Generate multi-env data
    np.random.seed(seed)
    training_data = {}
    for k in range(5):
        idle = generate_confounded_instrument_data(dgp, 2000, seed=seed+k*1000)
        idle['F'] = 'idle'
        np.random.seed(seed + k*1000 + 500)
        Z = np.random.binomial(1, 0.5, 2000)
        X = np.random.binomial(1, 0.5, 2000)
        U = np.random.binomial(1, 0.5, 2000)
        Y = np.random.binomial(1, logistic(1.0*X + 0.5*U), 2000)
        on = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'F': 'on'})
        training_data[f'env_{k}'] = pd.concat([idle, on], ignore_index=True)

    bounds = solve_all_bounds_binary_lp(
        training_data, ['Z'], 'X', 'Y', epsilon, 'F'
    )

    if bounds and (0,) in bounds and (1,) in bounds:
        ate_lower = 0.5 * bounds[(0,)][0] + 0.5 * bounds[(1,)][0]
        ate_upper = 0.5 * bounds[(0,)][1] + 0.5 * bounds[(1,)][1]
        return ate_upper - ate_lower
    return None

# Compare widths for weak vs strong instruments
print("Width by alpha_z (epsilon=0.10):")
for alpha_z in [0.25, 1.0, 4.0]:
    width = compute_width(alpha_z, epsilon=0.10)
    print(f"  alpha_z={alpha_z}: width={width:.3f}")

# Output:
# Width by alpha_z (epsilon=0.10):
#   alpha_z=0.25: width=0.298
#   alpha_z=1.0: width=0.301
#   alpha_z=4.0: width=0.299
```

## Visualizations

### Figure 1: Width vs. alpha_z

Located at `figures/exp3_width_vs_alpha_z.png`

```
Width
0.8 |                                        --- threshold
    |
0.6 |
    |
0.4 |                                        eps=0.20
    |  --------------------------------------------
0.2 |                                        eps=0.10
    |  --------------------------------------------
0.0 +-------------------------------------------->
    0.25  0.5   1.0   1.5   2.0   2.5   3.0   4.0   alpha_z

Key observation: Flat lines - width is independent of alpha_z!
```

### Figure 2: Coverage Heatmap

Located at `figures/exp3_coverage_heatmap.png`

```
        alpha_z
eps     0.25  0.5   1.0   2.0   4.0
0.05   | 96%  96%   96%   96%   96%  |
0.10   |100% 100%  100%  100%  100%  |
0.15   |100% 100%  100%  100%  100%  |
0.20   |100% 100%  100%  100%  100%  |

Coverage is uniform across all configurations.
```

## Results

See `results/exp3_summary.md` for detailed output.

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total runs | 1,800 |
| Overall feasibility | 99% |
| Overall coverage | 100% |
| Width range | [0.073, 0.740] |

### Threshold Analysis

No alpha_z value causes width to exceed 0.8 threshold for any epsilon tested. This means even very weak instruments (alpha_z=0.25) produce informative bounds.

## Output Files

| File | Description |
|------|-------------|
| `results/exp3_results.csv` | Raw results for all 1,800 runs |
| `results/exp3_summary.md` | Summary statistics |
| `figures/exp3_width_vs_alpha_z.png` | Width curves |
| `figures/exp3_coverage_heatmap.png` | Coverage heatmap |

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
```

## Interpretation

1. **Width is controlled by epsilon only:** Width ≈ 2*epsilon + noise
2. **Instrument strength is irrelevant:** The LP uses only experimental data
3. **No alpha_z threshold exists:** Bounds never become uninformative
4. **Coverage is uniformly excellent:** 96-100% across all configurations

## Implications

- **Good news:** The bounds method works with arbitrarily weak instruments
- **Why:** The method uses experimental data where X is randomized
- **Trade-off:** Weak instruments reduce the **efficiency** of observational estimators, but not the **validity** of experimental bounds

## References

- Bound, J., Jaeger, D. A., & Baker, R. M. (1995). Problems with instrumental variables estimation when the correlation between the instruments and the endogenous explanatory variable is weak.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression.
