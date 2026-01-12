# OSRCT Benchmark v1.0

**Semi-Synthetic Confounded Observational Datasets for Causal Inference Evaluation**

Generated: 2025-12-08

---

## Overview

This benchmark provides **525 confounded observational datasets** derived from real
randomized controlled trials (ManyLabs1) using the OSRCT algorithm (Gentzel et al., 2021).

### Key Features

- **Ground-truth ATEs**: True treatment effects known from original RCTs
- **Controlled confounding**: Systematic confounding via OSRCT sampling
- **Multiple studies**: 15 psychological experiments
- **Multiple confounding patterns**: 5 covariate configurations
- **7 confounding strengths**: beta from 0.1 (weak) to 2.0 (extreme)

---

## Quick Start

### Load a Dataset (Python)

```python
import pandas as pd

# Load a confounded dataset
data = pd.read_csv('confounded_datasets/by_study/anchoring1/age_beta0.5_seed42.csv')

# Load ground truth
ground_truth = pd.read_csv('ground_truth/rct_ates.csv')
true_ate = ground_truth[ground_truth['study'] == 'anchoring1']['ate'].values[0]

# Your causal inference method
estimated_ate = your_method(data, treatment='iv', outcome='dv', covariates=['resp_age'])

# Evaluate
bias = estimated_ate - true_ate
print(f"True ATE: {true_ate:.3f}, Estimated: {estimated_ate:.3f}, Bias: {bias:.3f}")
```

### Load a Dataset (R)

```r
library(readr)

# Load confounded dataset
data <- read_csv('confounded_datasets/by_study/anchoring1/age_beta0.5_seed42.csv')

# Load ground truth
ground_truth <- read_csv('ground_truth/rct_ates.csv')
true_ate <- ground_truth$ate[ground_truth$study == 'anchoring1']
```

---

## Directory Structure

```
osrct_benchmark_v1.0/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
│
├── raw_rct_data/                      # Original preprocessed RCT data
│   └── data_dictionary.json           # Variable descriptions
│
├── confounded_datasets/               # Main benchmark datasets
│   ├── by_study/                      # Organized by psychological study
│   │   ├── anchoring1/
│   │   ├── anchoring2/
│   │   └── ...
│   └── by_confounding_strength/       # Alternative organization
│
├── ground_truth/                      # True treatment effects
│   ├── rct_ates.csv                   # Study-level ATEs
│   └── site_stratified_summary.csv    # Site-level ATEs
│
├── analysis_results/                  # Pre-computed method evaluations
│   ├── figures/
│   ├── method_evaluation/
│   └── findings_summary.md
│
├── metadata/                          # Dataset metadata
│   ├── dataset_catalog.csv            # Full dataset catalog
│   ├── generation_summary.csv         # Generation details
│   └── checksums.txt                  # MD5 checksums
│
└── code/                              # Source code
    ├── osrct.py                       # OSRCT algorithm
    └── examples/                      # Usage examples
```

---

## Dataset Naming Convention

```
{pattern}_beta{strength}_seed{seed}.csv

Examples:
  age_beta0.5_seed42.csv        - Age confounding, moderate strength
  gender_beta1.0_seed42.csv     - Gender confounding, strong
  demo_full_beta0.1_seed42.csv  - Full demographics, weak confounding
```

---

## Studies Included

| Study | Outcome Type | True ATE | Description |
|-------|--------------|----------|-------------|
| anchoring1 | Continuous | ~1556 | NYC population estimation |
| anchoring2 | Continuous | ~2029 | Chicago population estimation |
| anchoring3 | Continuous | ~2418 | Mt. Everest height estimation |
| anchoring4 | Continuous | ~2495 | Daily baby births estimation |
| gainloss | Binary | ~0.29 | Gain vs Loss framing |
| sunk | Ordinal | ~0.61 | Sunk cost effect |
| flag | Continuous | ~0.03 | Flag priming on conservatism |
| quote | Continuous | ~0.70 | Quote attribution |
| reciprocity | Binary | ~0.13 | Norm of reciprocity |
| gambfal | Continuous | ~1.70 | Gambler's fallacy |
| scales | Continuous | ~0.17 | Scale anchoring |
| contact | Continuous | ~0.25 | Imagined contact |
| money | Continuous | ~-0.02 | Currency priming |
| iat | Continuous | ~0.26 | Implicit Association Test |
| allowedforbidden | Binary | ~-0.17 | Allowed/Forbidden framing |

---

## Confounding Patterns

| Pattern | Covariates | Type |
|---------|------------|------|
| age | resp_age | Single continuous |
| gender | resp_gender | Single binary |
| polideo | resp_polideo | Single ordinal |
| demo_basic | resp_age, resp_gender | Multi-covariate |
| demo_full | resp_age, resp_gender, resp_polideo | Multi-covariate |

---

## Confounding Strength (Beta)

| Beta | Interpretation | Expected Naive Bias |
|------|----------------|---------------------|
| 0.1 | Very weak | Minimal |
| 0.25 | Weak | Small |
| 0.5 | Moderate | Moderate |
| 0.75 | Moderate-strong | Substantial |
| 1.0 | Strong | Large |
| 1.5 | Very strong | Very large |
| 2.0 | Extreme | Extreme |

---

## Method Evaluation Results

From our Phase 5 analysis (5 methods evaluated on 525 datasets):

| Rank | Method | RMSE | Mean Bias |
|------|--------|------|-----------|
| 1 | IPW | 23.24 | -4.03 |
| 2 | Outcome Regression | 24.20 | +5.67 |
| 3 | AIPW (Doubly Robust) | 24.33 | -5.15 |
| 4 | Naive | 28.72 | +0.90 |
| 5 | PSM | 79.49 | +9.75 |

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{osrct_benchmark_2025,
  title={OSRCT Benchmark: Semi-Synthetic Datasets for Causal Inference Evaluation},
  author={[Authors]},
  year={2025},
  note={Based on ManyLabs1 (Klein et al., 2014) and OSRCT (Gentzel et al., 2021)}
}
```

### Original Data Sources

- **ManyLabs1**: Klein, R. A., et al. (2014). Investigating variation in replicability. *Social Psychology*, 45(3), 142-152.
- **OSRCT Algorithm**: Gentzel, M., Garant, D., & Jensen, D. (2021). The case for evaluating causal models using interventional measures and empirical data. *NeurIPS 2021*.

---

## License

This benchmark is released under the MIT License.
The original ManyLabs1 data is released under CC0 license.

---

## Contact

For questions or issues, please open an issue on the GitHub repository.
