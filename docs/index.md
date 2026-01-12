# OSRCT Benchmark Documentation

**Observational Sampling from Randomized Controlled Trials**

---

## Overview

The OSRCT Benchmark provides tools for:

1. **Generating confounded observational datasets** from RCT data while preserving known ground-truth treatment effects
2. **Evaluating causal inference methods** against these known ground truths
3. **Benchmarking** new causal inference approaches

## Quick Links

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Setup and installation guide |
| [Quick Start](quickstart.md) | Get started in 5 minutes |
| [API Reference](api/index.md) | Complete API documentation |
| [Tutorials](tutorials.md) | Step-by-step guides |

---

## Package Structure

```
osrct_benchmark_v1.0/
├── confounded_datasets/     # 525 generated observational datasets
│   └── by_study/            # Organized by study name
├── ground_truth/            # True ATEs from original RCTs
│   └── rct_ates.csv
├── code/                    # Core Python modules
│   ├── osrct.py             # OSRCT sampling algorithm
│   ├── causal_methods.py    # Causal inference implementations
│   └── ...
├── metadata/                # Dataset documentation
│   ├── dataset_catalog.csv
│   └── data_dictionary.json
└── analysis_results/        # Pre-computed method evaluations
```

---

## Core Concepts

### OSRCT Algorithm

The OSRCT procedure (Gentzel et al., 2021) generates realistic confounded observational data:

1. **Input**: RCT data with treatment `T`, outcome `Y`, and covariates `X`
2. **Compute selection probability**: `p_i = f(X_i)` using a biasing function
3. **Sample preferred treatment**: `t_s ~ Bernoulli(p_i)`
4. **Selection**: Keep unit if `T_i == t_s`, discard otherwise
5. **Output**: Confounded observational dataset with ~50% retention

**Key property**: The ground-truth ATE from the RCT remains valid, but naive estimation from the observational data will be biased.

### Confounding Patterns

The benchmark includes 5 confounding patterns:

| Pattern | Biasing Covariates |
|---------|-------------------|
| `age` | `resp_age` only |
| `gender` | `resp_gender` only |
| `polideo` | `resp_polideo` (political ideology) only |
| `demo_basic` | `resp_age`, `resp_gender` |
| `demo_full` | `resp_age`, `resp_gender`, `resp_polideo` |

### Confounding Strengths (Beta)

Seven beta values control confounding intensity:

| Beta | Confounding | Description |
|------|-------------|-------------|
| 0.1 | Very weak | Nearly random selection |
| 0.25 | Weak | Slight bias |
| 0.5 | Moderate | Noticeable confounding |
| 0.75 | Moderate-strong | Substantial bias |
| 1.0 | Strong | Heavy confounding |
| 1.5 | Very strong | Extreme bias |
| 2.0 | Severe | Maximum confounding |

---

## Causal Inference Methods

Six methods are implemented for evaluation:

| Method | Description | Key Assumption |
|--------|-------------|----------------|
| **Naive** | Difference-in-means | No confounding |
| **IPW** | Inverse probability weighting | Correct propensity model |
| **OR** | Outcome regression | Correct outcome model |
| **AIPW** | Augmented IPW (doubly robust) | Either model correct |
| **PSM** | Propensity score matching | Overlap, correct PS |
| **Causal Forest** | ML-based heterogeneous effects | Regularity conditions |

---

## References

- Gentzel, M., Garant, D., & Steeg, D. (2021). *The Case for Evaluating Causal Models Using Controlled Experiments*. NeurIPS 2021.
- Klein et al. (2014). *Investigating Variation in Replicability: A "Many Labs" Replication Project*. Social Psychology, 45:142-152.
- Hernan & Robins (2020). *Causal Inference: What If*.
- Wager & Athey (2018). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*. JASA.

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Citation

```bibtex
@misc{osrct_benchmark,
  title={OSRCT Benchmark: Evaluating Causal Inference Methods with Known Ground Truth},
  author={OSRCT Benchmark Contributors},
  year={2025},
  url={https://github.com/[username]/osrct-benchmark}
}
```
