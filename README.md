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

## Causal Grounding Module

In addition to the OSRCT benchmark, this repository includes a **causal grounding module** for computing partial identification bounds on CATE under unmeasured confounding.

### Overview

The `causal_grounding` module implements bounds estimation following Silva's "Causal Discovery Grounding and the Naturalness Assumption":

- **Leverages multi-environment data** (RCT + observational) to learn bounds on treatment effects
- **Uses conditional independence tests** (CMI) to score covariates as potential instruments
- **Solves linear programs** to compute partial identification bounds under naturalness assumptions
- **Transfers bounds** from training environments to a target population

### Module Structure

```
causal_grounding/
├── __init__.py            # Public API exports
├── estimator.py           # CausalGroundingEstimator class
├── ci_tests.py            # Conditional independence testing (CMI)
├── lp_solver.py           # Linear programming for bounds
├── covariate_scoring.py   # EHS criteria for instrument selection
├── discretize.py          # Covariate discretization
├── train_target_split.py  # Environment splitting
└── transfer.py            # Bound transfer methods
```

### Quick Example

```python
from causal_grounding import (
    CausalGroundingEstimator,
    create_train_target_split,
    load_rct_data
)
import pandas as pd

# Load data
osrct_data = pd.read_csv('confounded_datasets/anchoring1/age_beta0.1_seed42.csv')
rct_data = load_rct_data('anchoring1', 'ManyLabs1/pre-process/Manylabs1_data.pkl')

# Create train/target split (hold out mturk as target)
training_data, target_data = create_train_target_split(
    osrct_data, rct_data, target_site='mturk'
)

# Initialize and fit estimator
estimator = CausalGroundingEstimator(
    epsilon=0.1,           # Naturalness tolerance
    n_permutations=500,    # For CI tests
    random_seed=42
)
estimator.fit(training_data, treatment='iv', outcome='dv')

# Get CATE bounds
bounds_df = estimator.predict_bounds()
print(bounds_df[['z', 'cate_lower', 'cate_upper', 'width']])
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epsilon` | Naturalness tolerance (0 = strict, higher = relaxed) | 0.1 |
| `transfer_method` | How to combine bounds: `'conservative'`, `'average'`, `'weighted'` | `'conservative'` |
| `n_permutations` | Permutations for CI testing | 500 |

### Experiment Scripts

| Script | Description |
|--------|-------------|
| `experiments/run_grounding.py` | Full grid experiments with plotting and LaTeX output |
| `experiments/compare_ci_engines.py` | Compare CMI vs L1-Regression CI testing |
| `experiments/multi_instrument.py` | Multi-instrument aggregation analysis |
| `experiments/loco_vs_cmi.py` | LOCO vs CMI statistical comparison |
| `experiments/ehs_instrument_validity_experiment.py` | EHS instrument validity under realistic confounding |
| `experiments/compare_to_baselines.py` | Compare bounds to baseline causal methods |
| `experiments/validate_cate_coverage.py` | Validate CATE bounds against RCT ground truth |
| `experiments/run_sensitivity_sweep.py` | Parameter sensitivity analysis (epsilon sweep) |
| `experiments/integration_test.py` | End-to-end pipeline validation (7 tests) |

### Running Experiments

```bash
# Single experiment
python experiments/run_grounding.py --study anchoring1 --beta 0.3

# Full grid experiment
python experiments/run_grounding.py --grid --output results/

# Integration test
python experiments/integration_test.py --study anchoring1 --beta 0.1

# CATE coverage validation
python experiments/validate_cate_coverage.py --study anchoring1 --beta 0.1

# Epsilon sensitivity sweep
python experiments/run_sensitivity_sweep.py --study anchoring1 --beta 0.25 --output results/sensitivity/
```

### Sensitivity Analysis

The module includes tools for analyzing parameter sensitivity to characterize the precision-coverage tradeoff:

```python
from causal_grounding import SweepConfig, SensitivityAnalyzer

# Configure epsilon sweep
config = SweepConfig(
    parameter_name='epsilon',
    values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
)

# Run sweep
analyzer = SensitivityAnalyzer(
    sweep_config=config,
    base_estimator_params={'transfer_method': 'conservative'}
)

results = analyzer.run_sweep(
    training_data=training_data,
    ground_truth=true_cates  # For coverage computation
)

# Get recommended epsilon for 50% coverage target
print(results.get_recommended_epsilon(target_coverage=0.5))

# Generate Pareto plot
analyzer.plot_results(output_dir='results/sensitivity/')
```

Or via YAML configuration:

```bash
python experiments/run_sensitivity_sweep.py --config experiments/configs/sensitivity_sweep.yaml
```

### Test Suite

```bash
# Run all tests (144 unit tests + 7 integration tests)
pytest tests/test_causal_grounding/ -v
```

**Test Coverage:**
- `test_ci_tests.py` - 21 tests for CMI and CI engine
- `test_discretize.py` - 18 tests for covariate discretization
- `test_estimator.py` - 32 tests for the main estimator
- `test_lp_solver.py` - 21 tests for LP bounds
- `test_train_target_split.py` - 19 tests for environment splitting
- `test_transfer.py` - 33 tests for bound transfer
- `test_sensitivity.py` - Tests for parameter sensitivity analysis

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/predictive-shift.git
cd predictive-shift
pip install -r requirements.txt
```

### Run Your First Evaluation

```python
import pandas as pd
from causal_methods import CausalMethodEvaluator

# Load a confounded dataset
data = pd.read_csv('osrct_benchmark_v1.0/confounded_datasets/by_study/anchoring1/anchoring1_demo_full_beta1.0.csv')

# Load ground truth ATE from the original RCT
gt = pd.read_csv('osrct_benchmark_v1.0/ground_truth/rct_ates.csv')
true_ate = gt[gt['study'] == 'anchoring1']['ate'].values[0]

# Evaluate all methods
evaluator = CausalMethodEvaluator()
results = evaluator.evaluate_all(data, ground_truth_ate=true_ate, skip_causal_forest=True)

# See results
print(results[['method', 'ate', 'bias', 'covers_truth']])
```

**Output:**
```
              method          ate       bias  covers_truth
0              Naive  1528.279000  -27.38700         False
1                IPW  1553.421000   -2.24500          True
2  Outcome Regression  1560.112000    4.44600          True
3               AIPW  1555.890000    0.22400          True
4                PSM  1489.234000  -66.43200         False
```

---

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

### Generate Your Own Confounded Data

```python
from osrct import OSRCTSampler, load_manylabs1_data

# Load RCT data
rct_data = load_manylabs1_data(
    'ManyLabs1/pre-process/Manylabs1_data.pkl',
    study_filter='anchoring1'
)

# Create sampler with custom confounding
sampler = OSRCTSampler(
    biasing_covariates=['resp_age', 'resp_gender'],
    biasing_coefficients={'resp_age': 0.5, 'resp_gender': 0.8},
    random_seed=42
)

# Generate confounded sample
obs_data, selection_probs = sampler.sample(rct_data, treatment_col='iv')

# True ATE is still the RCT difference-in-means
# But naive estimate on obs_data will be biased
```

### Evaluate Your Own Method

```python
from causal_methods import CausalMethodEvaluator
import pandas as pd

# Your custom estimator
def my_estimator(data, treatment_col='iv', outcome_col='dv', covariates=None):
    # Your implementation here
    return {
        'method': 'my_method',
        'ate': estimated_ate,
        'se': standard_error,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se
    }

# Evaluate on benchmark
evaluator = CausalMethodEvaluator()
evaluator.add_method('my_method', my_estimator)

results = evaluator.evaluate_all(data, ground_truth_ate=true_ate)
```

---

## Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [Tutorials](docs/tutorials.md)
- [API Reference](docs/api/index.md)
  - [OSRCT Sampler](docs/api/osrct.md)
  - [Causal Methods](docs/api/causal_methods.md)
  - [Evaluation Utilities](docs/api/evaluation.md)
  - [Continuous Treatments](docs/api/continuous_treatments.md)

---

## Original Paper Code

This repository also contains reproduction code for:

**"Beyond reweighting: On the predictive role of covariate shift in effect generalization"**
Ying Jin, Naoki Egami, Dominik Rothenhäusler
[arXiv:2412.08869](https://arxiv.org/abs/2412.08869)

### Paper Code Structure

```
├── master.R                 # Main workflow script
├── ManyLabs1/
│   ├── pre-process/         # Data preprocessing
│   ├── explanatory/         # Distribution shift analysis
│   ├── predictive/          # Shift measures
│   └── generalization/      # KL-based prediction intervals
├── Pipeline/                # Pipeline dataset analysis
├── summary/                 # Cross-dataset analyses
└── plots_main.R             # Figure generation
```

### Run Paper Analysis (R)

```r
ROOT_DIR <- "/path/to/predictive-shift"
source("master.R")
```

---

## Citation

### OSRCT Benchmark

```bibtex
@misc{osrct_benchmark_2025,
  title={OSRCT Benchmark: Evaluating Causal Inference Methods with Known Ground Truth},
  author={OSRCT Benchmark Contributors},
  year={2025},
  url={https://github.com/YOUR_USERNAME/predictive-shift}
}
```

### Original Paper

```bibtex
@article{jin2024beyond,
  title={Beyond reweighting: On the predictive role of covariate shift in effect generalization},
  author={Jin, Ying and Egami, Naoki and Rothenh{\"a}usler, Dominik},
  journal={arXiv preprint arXiv:2412.08869},
  year={2024}
}
```

### OSRCT Algorithm

```bibtex
@inproceedings{gentzel2021osrct,
  title={The Case for Evaluating Causal Models Using Controlled Experiments},
  author={Gentzel, Michael and Garant, Dan and Jensen, David},
  booktitle={NeurIPS},
  year={2021}
}
```

### ManyLabs1 Data

```bibtex
@article{klein2014manylabs,
  title={Investigating Variation in Replicability: A "Many Labs" Replication Project},
  author={Klein, Richard A and others},
  journal={Social Psychology},
  volume={45},
  pages={142--152},
  year={2014}
}
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Reporting bugs
- Submitting new causal methods
- Adding datasets
- Improving documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Diana Da In Le for ManyLabs1 data pre-processing code
- ManyLabs1 Project for original RCT data
- Gentzel et al. for the OSRCT algorithm
- All contributors to the benchmark

---

## Links

- [Paper: Beyond Reweighting](https://arxiv.org/abs/2412.08869)
- [Awesome Replicability Data](https://github.com/ying531/awesome-replicability-data)
- [ManyLabs1 Project](https://osf.io/wx7ck/)
