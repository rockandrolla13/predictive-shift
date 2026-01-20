# How to Run Experiments

A complete guide for running all experiments in the predictive-shift repository.

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt              # Core dependencies
pip install -r requirements-grounding.txt    # Causal grounding specific
```

### 2. Verify Data Files Exist

The experiments require two data sources:

| Data | Location | Description |
|------|----------|-------------|
| OSRCT Confounded Datasets | `confounded_datasets/` | Semi-synthetic datasets with controlled confounding |
| ManyLabs1 RCT Data | `ManyLabs1/pre-process/Manylabs1_data.pkl` | Ground-truth RCT data |

### 3. Quick Verification

```bash
python experiments/integration_test.py
```

If all 7 tests pass, you're ready to run experiments.

---

## Experiment Scripts Overview

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_grounding.py` | Main CATE bounds experiment | 5-30 min |
| `validate_cate_coverage.py` | Validate bounds contain true effects | 10-20 min |
| `compare_to_baselines.py` | Compare to IPW, AIPW, PSM, etc. | 10-30 min |
| `compare_lp_methods.py` | Compare heuristic vs LP bounds | 5-10 min |
| `run_sensitivity_sweep.py` | Epsilon parameter sweep | 30-60 min |
| `compare_ci_engines.py` | CMI vs L1-Regression comparison | 20-40 min |
| `multi_instrument.py` | Multi-instrument aggregation | 20-40 min |
| `loco_vs_cmi.py` | LOCO vs CMI statistical comparison | 30-60 min |

---

## Quick Start (5 minutes)

### Run a Single Experiment

```bash
python experiments/run_grounding.py \
    --study anchoring1 \
    --beta 0.25 \
    --epsilon 0.1
```

This computes CATE bounds for the "anchoring1" study with moderate confounding (beta=0.25).

### Check Output

Results are saved to `results/` with:
- JSON files containing numerical results
- PNG figures (forest plots, coverage, etc.)
- LaTeX tables for papers

---

## Core Experiments

### 1. Main Causal Grounding Experiment

**Purpose:** Compute CATE bounds on confounded data using naturalness constraints.

```bash
# Single study
python experiments/run_grounding.py --study anchoring1 --beta 0.3

# With baseline comparisons (IPW, AIPW, PSM)
python experiments/run_grounding.py --study anchoring1 --beta 0.3 --baselines

# Full grid (all studies, all betas) - takes hours
python experiments/run_grounding.py --grid --output results/full_grid/
```

**Key Parameters:**
- `--study`: Study name (anchoring1, gamblerfallacy, sunkfallacy, etc.)
- `--beta`: Confounding strength (0.1=weak, 0.5=strong, 1.0+=very strong)
- `--epsilon`: Naturalness tolerance (0.1 typical, higher=wider bounds)
- `--pattern`: Confounding pattern (age, gender, polideo, demo_basic, demo_full)

---

### 2. CATE Coverage Validation

**Purpose:** Verify that bounds contain ground-truth treatment effects.

```bash
# Single validation
python experiments/validate_cate_coverage.py \
    --study anchoring1 \
    --beta 0.25 \
    --epsilon 0.1

# Grid validation (multiple studies/betas)
python experiments/validate_cate_coverage.py --mode grid
```

**Output:** Coverage rate (% of strata where bounds contain true CATE).

---

### 3. Baseline Methods Comparison

**Purpose:** Compare causal grounding to standard causal inference methods.

```bash
python experiments/compare_to_baselines.py \
    --study anchoring1 \
    --beta 0.25

# Grid mode
python experiments/compare_to_baselines.py --mode grid
```

**Methods compared:** Naive, IPW, Outcome Regression, AIPW, PSM

---

### 4. LP Methods Comparison

**Purpose:** Compare heuristic bounds vs true LP bounds.

```bash
python experiments/compare_lp_methods.py \
    --study anchoring1 \
    --beta 0.25 \
    --binarize  # Required for LP bounds
```

**Output:** Comparison showing LP bounds are typically tighter than heuristic.

---

### 5. Sensitivity Analysis (Epsilon Sweep)

**Purpose:** Characterize precision-coverage tradeoff across epsilon values.

```bash
# Using command line
python experiments/run_sensitivity_sweep.py \
    --study anchoring1 \
    --beta 0.25 \
    --epsilon-values 0.01 0.05 0.1 0.2 0.3 0.5 \
    --output results/sensitivity/

# Using config file
python experiments/run_sensitivity_sweep.py \
    --config experiments/configs/sensitivity_sweep.yaml
```

**Output:** Pareto curve showing tradeoff between bound width and coverage.

---

## Advanced Experiments

### 6. CI Engine Comparison (CMI vs L1-Regression)

```bash
python experiments/compare_ci_engines.py
```

Compares two conditional independence testing approaches for instrument selection.

### 7. Multi-Instrument Aggregation

```bash
python experiments/multi_instrument.py
```

Evaluates whether using multiple instruments (k>1) produces tighter bounds.

### 8. LOCO vs CMI Comparison

```bash
python experiments/loco_vs_cmi.py
```

Compares statistical properties (Type I error, power) of LOCO vs CMI tests.

---

## Parameter Reference

### Studies (15 available)
```
anchoring1, anchoring2, anchoring3, anchoring4,
gamblerfallacy, sunkfallacy, gainloss, quote,
allowforbid, reciprocity, scaleframe, contact,
imaginedcontact, flagprime, moneypriming
```

### Confounding Patterns
| Pattern | Covariates |
|---------|------------|
| `age` | Age only |
| `gender` | Gender only |
| `polideo` | Political ideology only |
| `demo_basic` | Age + Gender |
| `demo_full` | Age + Gender + Political ideology |

### Beta Values (Confounding Strength)
| Beta | Interpretation |
|------|----------------|
| 0.1 | Weak confounding |
| 0.25 | Moderate confounding |
| 0.5 | Strong confounding |
| 1.0+ | Very strong confounding |

### Epsilon (Naturalness Tolerance)
| Epsilon | Effect |
|---------|--------|
| 0.05 | Tight bounds, may undercover |
| 0.1 | Typical choice, good tradeoff |
| 0.2-0.3 | Wider bounds, better coverage |
| 0.5+ | Very wide, conservative |

---

## Example Workflows

### Workflow 1: Quick Exploration (15 min)

```bash
# Test setup
python experiments/integration_test.py

# Run one experiment
python experiments/run_grounding.py --study anchoring1 --beta 0.25

# Validate coverage
python experiments/validate_cate_coverage.py --study anchoring1 --beta 0.25
```

### Workflow 2: Method Comparison (1 hour)

```bash
# Compare to baselines
python experiments/compare_to_baselines.py --study anchoring1 --beta 0.25

# Compare LP methods
python experiments/compare_lp_methods.py --study anchoring1 --beta 0.25 --binarize

# Sensitivity analysis
python experiments/run_sensitivity_sweep.py \
    --study anchoring1 --beta 0.25 \
    --epsilon-range 0.01 0.5 --n-points 10
```

### Workflow 3: Full Evaluation (4+ hours)

```bash
# Full grid experiment
python experiments/run_grounding.py --grid --output results/full_grid/

# CI engine comparison
python experiments/compare_ci_engines.py

# Multi-instrument analysis
python experiments/multi_instrument.py
```

---

## Output Structure

```
results/
├── grounding/
│   ├── anchoring1/
│   └── anchoring2/
├── cate_validation/
├── lp_comparison/
├── multi_instrument/
├── loco_vs_cmi/
├── ci_engine_comparison/
├── method_evaluation/
└── reports/
    └── figures/
```

---

## Troubleshooting

### "FileNotFoundError: confounded_datasets/..."
- Check that data files exist in `confounded_datasets/{study}/`
- File naming must be: `{pattern}_beta{beta}_seed42.csv`

### Slow Runtime
- Reduce `--n-permutations` (e.g., 100 instead of 500)
- Start with smaller studies (anchoring1)
- Use `--quiet` to suppress progress output

### Memory Issues
- Run single experiments instead of grid
- Reduce permutations
- Close other applications

### Import Errors
```bash
pip install --upgrade -r requirements-grounding.txt
```

---

## Getting Help

```bash
# Any script with --help
python experiments/run_grounding.py --help
python experiments/validate_cate_coverage.py --help
```
