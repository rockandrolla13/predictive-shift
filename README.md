# predictive-shift
OSRCT Benchmark: Evaluating Causal Inference Methods with Known Ground Truth from RCTs

# OSRCT Benchmark

**Evaluating Causal Inference Methods with Known Ground Truth from Randomized Controlled Trials**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/datasets-525-green.svg)](#benchmark-datasets)
[![Methods](https://img.shields.io/badge/methods-6-orange.svg)](#causal-inference-methods)

---

## Overview

This repository provides a comprehensive benchmark for evaluating causal inference methods using **semi-synthetic confounded observational data** with **known ground-truth treatment effects** from real randomized experiments.

### The Problem

Evaluating causal inference methods is hard because we rarely know the true causal effect. Observational studies have confounding, and we can't verify if our estimates are correct.

### Our Solution

We use the **OSRCT (Observational Sampling from RCT)** algorithm to:
1. Start with real RCT data (where randomization ensures unbiased effects)
2. Systematically introduce confounding via biased sampling
3. Create observational-like data where we **know the true treatment effect**

This lets us rigorously benchmark which methods recover the truth under various confounding scenarios.

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/predictive-shift.git
cd predictive-shift
pip install -r requirements.txt
```

## Benchmark Datasets

### 525 Confounded Datasets

Generated from **15 ManyLabs1 RCT studies** across:
- **5 confounding patterns**: age, gender, polideo, demo_basic, demo_full
- **7 confounding strengths**: β = 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

| Study | N | Treatment | True ATE | Domain |
|-------|--:|-----------|----------|--------|
| anchoring1 | 6,344 | High/low anchor | 1555.67 | Judgment |
| anchoring2 | 6,195 | High/low anchor | 762.28 | Judgment |
| anchoring3 | 6,049 | High/low anchor | -21.94 | Judgment |
| anchoring4 | 6,096 | High/low anchor | 753.83 | Judgment |
| allowedforbidden | 6,225 | Word framing | 22.67 | Survey |
| gainloss | 6,004 | Gain/loss frame | 13.49 | Decision |
| gambfallaliacy | 5,894 | Sequence shown | -6.05 | Probability |
| scalealiasing | 6,053 | Scale direction | -1.94 | Measurement |
| sunkiosteffect | 6,036 | Sunk cost info | 18.00 | Decision |
| quote | 5,952 | Attribution shown | 8.41 | Persuasion |
| reciprocity | 6,051 | Favor given | 35.91 | Social |
| lowvshighcategory | 6,025 | Category scale | 2.24 | Judgment |
| normsaliasing | 5,861 | Norm type | 5.76 | Social |
| contactaliasing | 5,984 | Contact type | 1.82 | Social |
| mathaliasing | 6,129 | Math vs gender | 5.20 | Stereotype |

### Dataset Structure

```
osrct_benchmark_v1.0/
├── confounded_datasets/
│   ├── by_study/
│   │   ├── anchoring1/
│   │   │   ├── anchoring1_age_beta0.1.csv
│   │   │   ├── anchoring1_age_beta0.25.csv
│   │   │   ├── ...
│   │   │   └── anchoring1_demo_full_beta2.0.csv
│   │   ├── anchoring2/
│   │   └── ...
│   └── by_pattern/
│       ├── age/
│       ├── gender/
│       ├── polideo/
│       ├── demo_basic/
│       └── demo_full/
├── ground_truth/
│   └── rct_ates.csv              # True ATEs from RCTs
├── analysis_results/
│   └── method_evaluation_results.csv
├── metadata/
│   └── dataset_catalog.csv
└── notebooks/
    └── 01_quickstart.ipynb
```

### Confounding Patterns

| Pattern | Covariates Used | Description |
|---------|-----------------|-------------|
| `age` | resp_age | Age-based selection bias |
| `gender` | resp_gender | Gender-based selection bias |
| `polideo` | resp_polideo | Political ideology bias |
| `demo_basic` | age + gender | Two demographic confounders |
| `demo_full` | age + gender + polideo | Three demographic confounders |

### Confounding Strength (β)

The β parameter controls how strongly covariates influence treatment assignment:

| β Value | Confounding Level | Typical Covariate Imbalance (SMD) |
|--------:|-------------------|-----------------------------------|
| 0.1 | Very weak | ~0.05 |
| 0.5 | Moderate | ~0.25 |
| 1.0 | Strong | ~0.50 |
| 2.0 | Very strong | ~1.00+ |

---

## Causal Inference Methods

### Implemented Methods

| Method | Description | Key Assumption |
|--------|-------------|----------------|
| **Naive** | Difference-in-means | No confounding |
| **IPW** | Inverse Probability Weighting | Correct propensity model |
| **Outcome Regression** | Covariate-adjusted regression | Correct outcome model |
| **AIPW** | Augmented IPW (Doubly Robust) | Either model correct |
| **PSM** | Propensity Score Matching | Common support |
| **Causal Forest** | ML-based heterogeneous effects | Overlap, smoothness |

### Method Details

#### Naive Estimator
```
ATE = E[Y | T=1] - E[Y | T=0]
```
Simple difference in means. Biased under confounding. Serves as baseline.

#### Inverse Probability Weighting (IPW)
```
ATE = (1/n) Σ [T·Y/e(X) - (1-T)·Y/(1-e(X))]
```
Weights observations by inverse propensity score. Unbiased if propensity model is correct.

#### Outcome Regression
```
ATE = (1/n) Σ [μ̂(1,X) - μ̂(0,X)]
```
Models outcome as function of treatment and covariates. Unbiased if outcome model is correct.

#### AIPW (Doubly Robust)
```
ATE = (1/n) Σ [(μ̂(1,X) - μ̂(0,X)) + T(Y-μ̂(1,X))/e(X) - (1-T)(Y-μ̂(0,X))/(1-e(X))]
```
Combines IPW and outcome regression. Consistent if **either** model is correct.

#### Propensity Score Matching
Matches treated units to similar control units based on propensity score distance.

#### Causal Forest
Non-parametric ML method that estimates heterogeneous treatment effects τ(X).

---

## Current Leaderboard

Performance across all 525 datasets:

| Rank | Method | RMSE | Bias | Coverage |
|:----:|--------|-----:|-----:|---------:|
| 1 | **IPW** | 23.24 | -4.03 | 94.1% |
| 2 | Outcome Regression | 24.20 | +5.67 | 93.8% |
| 3 | AIPW | 24.33 | -5.15 | 94.5% |
| 4 | Naive | 28.72 | +0.90 | 91.2% |
| 5 | PSM | 79.49 | +9.75 | 85.3% |

### Performance by Confounding Strength

| Method | β=0.1 | β=0.5 | β=1.0 | β=2.0 | Degradation |
|--------|------:|------:|------:|------:|------------:|
| IPW | 8.2 | 15.4 | 23.1 | 35.8 | 4.4x |
| AIPW | 8.5 | 16.1 | 24.3 | 38.2 | 4.5x |
| Naive | 6.9 | 18.7 | 28.7 | 52.1 | 7.5x |
| PSM | 45.2 | 62.8 | 79.5 | 124.3 | 2.8x |

---

## Tutorials & Notebooks

### Exploration Notebooks

| Notebook | Description | Causal Methods Coverage |
|----------|-------------|------------------------|
| [01_raw_rct_data_exploration](exploration_notebooks/01_raw_rct_data_exploration.ipynb) | Explore raw ManyLabs1 RCT data | Basic |
| [02_preprocessed_data_assessment](exploration_notebooks/02_preprocessed_data_assessment.ipynb) | Assess preprocessed data quality | Covariate balance |
| [03_confounded_datasets_exploration](exploration_notebooks/03_confounded_datasets_exploration.ipynb) | Verify confounding introduction | Confounding verification |
| [04_analysis_results_summary](exploration_notebooks/04_analysis_results_summary.ipynb) | **Method comparison results** | **Comprehensive** |
| [05_full_pipeline_data_quality_report](exploration_notebooks/05_full_pipeline_data_quality_report.ipynb) | End-to-end QA report | - |
| [06_phase5_cross_site_causal_analysis_guide](exploration_notebooks/06_phase5_cross_site_causal_analysis_guide.ipynb) | **Complete implementation guide** | **Encyclopedic** |

### Recommended Learning Path

1. **Beginners**: Start with `osrct_benchmark_v1.0/notebooks/01_quickstart.ipynb`
2. **Intermediate**: Add `04_analysis_results_summary.ipynb` for benchmarking insights
3. **Advanced**: Study `06_phase5_cross_site_causal_analysis_guide.ipynb` for full implementations

---

## Python API

### Core Modules

| Module | Description |
|--------|-------------|
| `osrct.py` | OSRCT sampling for binary treatments |
| `osrct_continuous.py` | Extension for continuous/multi-valued treatments |
| `causal_methods.py` | All causal inference method implementations |
| `generate_confounded_datasets.py` | Dataset generation script |


[arXiv:2412.08869](https://arxiv.org/abs/2412.08869)

