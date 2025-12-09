# OSRCT Implementation for ManyLabs1 and Pipeline Datasets

This directory contains a Python implementation of the OSRCT (Observational Sampling from Randomized Controlled Trials) procedure from Gentzel et al. (2021).

## Overview

OSRCT generates semi-synthetic confounded observational datasets from RCT data while preserving known ground-truth treatment effects. This enables rigorous evaluation of causal inference methods.

## Files Created (November 30, 2025)

### Core Implementation
- **`osrct.py`** (18 KB): Main OSRCT module
  - `OSRCTSampler`: Main sampling class with logistic biasing function
  - `select_biasing_covariates()`: Automatic covariate selection
  - `evaluate_osrct_sample()`: Evaluation metrics
  - `load_manylabs1_data()`: ManyLabs1 data loader
  - `load_pipeline_data()`: Pipeline data loader

### Demonstration
- **`osrct_demo.py`** (17 KB): Demonstration script with CLI
  - `demo_synthetic()`: Test with synthetic data (no data required)
  - `demo_manylabs1()`: Demo with ManyLabs1 dataset
  - `demo_pipeline()`: Demo with Pipeline dataset

### Documentation
- **`Nov30MemoryRicardoWork.yaml`** (20 KB): Complete session memory
  - Detailed data structure documentation
  - Algorithm implementation details
  - Usage examples and next steps
  - **Use this file to resume the session**

- **`README_OSRCT.md`** (this file): Quick reference

## Quick Start

### 1. Test with Synthetic Data (No Data Required)

```bash
# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Run synthetic data demo
python osrct_demo.py --dataset synthetic
```

This will:
- Generate 2,000 synthetic RCT observations
- Create confounding via age, gender, and income
- Show that naive observational estimate is biased
- Print evaluation metrics

### 2. Use with ManyLabs1 Data

**Prerequisites:**
1. Download ManyLabs1 data from [OSF](https://osf.io/wx7ck/)
2. Process data using R scripts:
   ```R
   source("ManyLabs1/pre-process/ML1_data_process.R")
   source("ManyLabs1/pre-process/ML1_data_process_2.R")
   ```
3. Install pyreadr: `pip install pyreadr`

**Run demo:**
```bash
python osrct_demo.py \
  --dataset manylabs1 \
  --data-path ManyLabs1/pre-process/Manylabs1_data.RData \
  --study anchoring1
```

### 3. Use with Pipeline Data

**Prerequisites:**
1. Download Pipeline data from [OSF](https://osf.io/wx7ck/)
2. Process data: Run `Pipeline/pre-process/process.Rmd` in R

**Run demo:**
```bash
python osrct_demo.py \
  --dataset pipeline \
  --data-path Pipeline/pre-process/ \
  --study-id 7
```

## Programmatic Usage

```python
from osrct import OSRCTSampler, select_biasing_covariates
import pandas as pd

# Load your RCT data
rct_data = pd.read_csv('your_rct_data.csv')

# Select biasing covariates (automatic)
biasing_covs = select_biasing_covariates(
    rct_data,
    treatment_col='treatment',
    outcome_col='outcome',
    candidate_covariates=['age', 'gender', 'income'],
    min_correlation=0.1,
    max_covariates=5
)

# Create sampler with moderate confounding
sampler = OSRCTSampler(
    biasing_covariates=biasing_covs,
    biasing_coefficients={'age': 0.8, 'gender': 0.6, 'income': 0.5},
    intercept=0.0,
    standardize=True,
    random_seed=42
)

# Generate observational sample
obs_data, selection_probs = sampler.sample(
    rct_data,
    treatment_col='treatment',
    verbose=True
)

# Save results
obs_data.to_csv('observational_sample.csv', index=False)
```

## Data Structures

### ManyLabs1
- **Participants:** ~6,000+
- **Studies:** 13 psychology experiments
- **Labs:** 36
- **Treatment:** `iv` (binary 0/1)
- **Outcome:** `dv` (continuous)
- **Covariates:** demographics, political views, study characteristics

Example studies: `anchoring1`, `flag`, `gainloss`, `iat`, `sunk`

### Pipeline
- **Participants:** ~3,000+
- **Studies:** 10 moral judgment studies
- **Labs:** 25
- **Covariates:** demographics, socioeconomic, study experience

Example studies: Study 7 (Intuitive Economics), Study 8 (Burn in Hell)

## How OSRCT Works

1. **For each unit in the RCT:**
   - Compute selection probability: `p_i = 1 / (1 + exp(-β₀ - Σ(βⱼ*Cⱼ)))`
   - Sample preferred treatment: `t_s ~ Bernoulli(p_i)`
   - Keep unit if actual treatment matches preferred treatment

2. **Result:**
   - ~50% sample retention (on average)
   - Realistic confounding introduced
   - Ground-truth treatment effect preserved

## Next Steps

1. **Test with real data:** Download and process ManyLabs1/Pipeline data
2. **Create Jupyter tutorial:** Interactive walkthrough
3. **Benchmark causal methods:** IPW, regression adjustment, doubly robust
4. **Generate datasets:** Apply to all 13 ManyLabs1 studies

See `Nov30MemoryRicardoWork.yaml` for detailed next steps.

## Session Resumption

To continue this work in a future session:

1. **Read the memory file:**
   ```bash
   cat Nov30MemoryRicardoWork.yaml
   ```

2. **Review implementation:**
   - `osrct.py`: Core algorithm
   - `osrct_demo.py`: Examples

3. **Run quick test:**
   ```bash
   python osrct_demo.py --dataset synthetic
   ```

4. **Follow prioritized tasks** in `Nov30MemoryRicardoWork.yaml > next_steps`

## References

- **OSRCT Paper:** Gentzel et al. (2021) "The Case for Evaluating Causal Models Using Controlled Experiments" (NeurIPS 2021)
- **ManyLabs1:** [OSF Repository](https://osf.io/wx7ck/)
- **Pipeline:** [OSF Repository](https://osf.io/wx7ck/)
- **Predictive Shift Paper:** Jin, Egami, & Rothenhäusler (2024) [arXiv:2412.08869](https://arxiv.org/abs/2412.08869)

## Implementation Status

✅ **Completed:**
- Core OSRCT algorithm
- Logistic biasing function
- Covariate selection
- Evaluation metrics
- Demonstration scripts
- Session documentation

⏳ **Remaining:**
- Test with real ManyLabs1 data
- Test with real Pipeline data
- Create Jupyter notebook tutorial
- Add unit tests
- Package for distribution

## Contact

Session Date: November 30, 2025
Implementation: Python 3.x
Status: Development (v1.0.0)

For questions or to resume work, refer to `Nov30MemoryRicardoWork.yaml`.
