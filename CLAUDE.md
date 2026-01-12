# CLAUDE.md - Project Intelligence for predictive-shift

## Project Overview

**predictive-shift** is an OSRCT (Observational Sampling from Randomized Controlled Trials) benchmark for evaluating causal inference methods. The project generates semi-synthetic confounded observational datasets from ManyLabs1 RCT data while preserving ground-truth treatment effects.

**Core Purpose:** Provide a rigorous benchmark to evaluate how well causal inference methods recover true treatment effects from confounded observational data.

**Reference:** Gentzel, M., Garant, D., & Jensen, D. (2021). *The Case for Evaluating Causal Models Using Controlled Experiments.* NeurIPS 2021.

---

## Architecture Diagram

```
predictive-shift/
│
├── CORE PIPELINE (Existing) ────────────────────────────────────────────────
│
│   osrct.py                    # OSRCTSampler class - generates confounded data
│         │
│         ▼
│   generate_confounded_datasets.py   # Full grid: 15 studies x 5 patterns x 7 betas
│         │
│         ▼
│   confounded_datasets/        # 525 generated datasets
│         │
│         ▼
│   causal_methods.py           # Baseline estimators (IPW, AIPW, OR, PSM, CF)
│         │
│         ▼
│   run_phase5_analysis.py      # Evaluation pipeline with CausalMethodEvaluator
│         │
│         ▼
│   analysis_results/           # Performance metrics, figures, findings
│
├── CONFIGURATION ───────────────────────────────────────────────────────────
│
│   experimental_grid.py        # Study/beta/pattern definitions
│
├── DATA ────────────────────────────────────────────────────────────────────
│
│   ManyLabs1/                  # Raw and preprocessed RCT data
│   └── pre-process/
│       └── Manylabs1_data.pkl
│
│   ground_truth/               # True ATEs from RCTs
│   └── rct_ates.csv
│
├── CAUSAL GROUNDING MODULE (New) ───────────────────────────────────────────
│
│   causal_grounding/           # Silva's causal discovery grounding algorithm
│   ├── __init__.py
│   ├── discretize.py           # Covariate discretization for LP
│   ├── train_target_split.py   # Train/target environment splitting
│   ├── ci_tests.py             # Conditional independence tests (EHS criteria)
│   ├── covariate_scoring.py    # h_X, h_Y scoring on modified EHS
│   ├── lp_solver.py            # Naturalness-constrained LP bounds
│   ├── estimator.py            # CausalGroundingEstimator main class
│   └── transfer.py             # Bound transfer across environments
│
├── EXPERIMENTS ─────────────────────────────────────────────────────────────
│
│   experiments/                # Grounding experiments and baselines
│
└── TESTS ───────────────────────────────────────────────────────────────────

    tests/
    └── test_causal_grounding/
```

---

## Key Abstractions

### 1. OSRCTSampler (osrct.py)

The core class for generating confounded observational data from RCT data.

```python
class OSRCTSampler:
    def __init__(
        self,
        biasing_covariates: List[str],      # Covariates that induce confounding
        biasing_coefficients: Dict[str, float],  # Coefficients for selection
        intercept: float = 0.0,
        standardize: bool = True,
        random_seed: int = None
    )

    def sample(
        self,
        rct_data: pd.DataFrame,
        treatment_col: str = 'iv',
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate confounded sample from RCT.

        Algorithm:
        1. Compute p_i = sigmoid(beta @ covariates) for each unit
        2. Sample preferred treatment t_s ~ Bernoulli(p_i)
        3. Keep only units where actual treatment == preferred treatment

        Returns:
            observational_sample: Biased dataset
            selection_probs: Selection probability per unit
        """
```

### 2. Causal Estimator Function Signature (causal_methods.py)

All estimators follow this interface:

```python
def estimate_<method>(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    **method_specific_args
) -> Dict[str, Any]:
    """
    Returns:
        {
            'method': str,
            'ate': float,
            'se': float,
            'ci_lower': float,
            'ci_upper': float,
            'n_used': int,
            ...method-specific fields...
        }
    """
```

**Available methods:** `naive`, `ipw`, `outcome_regression`, `aipw`, `psm`, `causal_forest`

### 3. CausalMethodEvaluator (causal_methods.py)

```python
class CausalMethodEvaluator:
    def __init__(
        self,
        treatment_col: str = 'iv',
        outcome_col: str = 'dv',
        covariates: List[str] = None
    )

    def evaluate_method(
        self,
        data: pd.DataFrame,
        method: str,
        ground_truth_ate: float = None,
        **kwargs
    ) -> Dict[str, Any]

    def evaluate_all(
        self,
        data: pd.DataFrame,
        ground_truth_ate: float = None,
        methods: List[str] = None
    ) -> pd.DataFrame
```

### 4. Data Schema

Standard column names across all datasets:

| Column | Type | Description |
|--------|------|-------------|
| `iv` | int (0/1) | Treatment indicator (independent variable) |
| `dv` | float | Outcome (dependent variable) |
| `resp_age` | float | Respondent age (continuous) |
| `resp_gender` | int (0/1) | Respondent gender |
| `resp_polideo` | int (0-6) | Political ideology (ordinal) |
| `site` | str | Data collection site (36 total in ManyLabs1) |
| `study` | str | Study name (e.g., 'anchoring1', 'gainloss') |

---

## Current Task: causal_grounding Module

### Theoretical Foundation

The module implements the algorithm from Silva's "Causal Discovery Grounding and the Naturalness Assumption":

**Goal:** Estimate CATE bounds in a target environment using:
- Modified EHS criteria for covariate selection
- LP-based partial identification with naturalness constraints
- Transfer from training to target environments

### Key Concepts

#### Regime Variable F

```
F in {idle, on}
├── F = idle: Observational regime (confounded)
│             X generated by p(X | Z, U) where U is unobserved confounder
│
└── F = on:   Interventional regime (unconfounded)
              X generated by p(X | Z) - designed policy, U -> X edge severed
```

The RCT data represents F=on (randomized, unconfounded).
The OSRCT-sampled data represents F=idle (confounded).

#### Modified EHS Criteria for Covariate Selection

Uses Conditional Mutual Information (CMI) tests to score covariates.
For candidate covariate Z_a (with remaining covariates Z_b):

1. **(i.a)** X ⊥̸ Z_a | {Z_b, F=idle} — want HIGH CMI (h_X score)
2. **(ii.a)** Y ⊥ Z_a | {X, Z_b, F=idle} — want LOW CMI (h_Y score)

**Good instrument:** High h_X, low h_Y

#### LP Solver for Partial Identification

Solve for bounds on:
```
θ_{yz_az_b}^{k,a,*} = P(X=1 | Z_a=z_a, Z_b=z_b, Y_*=y, F=idle)
```

This confounding function is unidentified but bounded by:
- Box constraints: 0 ≤ θ ≤ 1
- Identification constraint: weighted sum equals P(X=1|z,idle)
- Naturalness constraint: bounded variation across Z_a values (ε tolerance)

#### Transfer Rule

Bounds learned from training environments transfer to target:
- **Conservative:** Take widest bounds (max upper, min lower)
- Can be tightened with distributional assumptions

### Module Files

| File | Purpose |
|------|---------|
| `discretize.py` | Convert continuous covariates to discrete for LP |
| `train_target_split.py` | Split data into training/target environments |
| `ci_tests.py` | Conditional independence test engine |
| `covariate_scoring.py` | Score covariates on modified EHS (h_X, h_Y) |
| `lp_solver.py` | Naturalness-constrained LP for θ bounds |
| `estimator.py` | `CausalGroundingEstimator` main algorithm |
| `transfer.py` | Bound transfer across environments |

---

## Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Install dependencies
.venv/bin/pip install -r requirements-grounding.txt
```

### Running Existing Pipeline
```bash
# Generate confounded datasets
python generate_confounded_datasets.py

# Run Phase 5 evaluation
python run_phase5_analysis.py

# Skip causal forest (faster)
python run_phase5_analysis.py --skip-causal-forest

# Specific studies only
python run_phase5_analysis.py --studies anchoring1 gainloss
```

### Testing
```bash
# Run tests for causal_grounding module
.venv/bin/pytest tests/test_causal_grounding/ -v
```

---

## Style Guidelines

### Code Style

1. **Type hints** on all function signatures:
   ```python
   def estimate_ate(
       data: pd.DataFrame,
       treatment_col: str = 'iv',
       outcome_col: str = 'dv'
   ) -> Dict[str, float]:
   ```

2. **Docstrings** with Parameters/Returns sections:
   ```python
   """
   Brief description.

   Parameters
   ----------
   data : DataFrame
       Input data with treatment and outcome

   Returns
   -------
   result : dict
       Dictionary with 'ate', 'se', 'ci_lower', 'ci_upper'
   """
   ```

3. **Return dicts** from estimators with standard keys:
   - `method`, `ate`, `se`, `ci_lower`, `ci_upper`

4. **Constants** in UPPER_SNAKE_CASE at module level

5. **Class naming**: PascalCase (e.g., `OSRCTSampler`, `CausalMethodEvaluator`)

### Existing Patterns to Follow

- Use `pandas` for data manipulation
- Use `numpy` for numerical operations
- Use `scipy.optimize.linprog` or `cvxpy` for LP
- Bootstrap for standard errors when analytical formula unavailable
- Clip propensity scores to [0.01, 0.99] to avoid instability

---

## Related Documentation

- `revised_implementation_plan.md` - Detailed implementation roadmap
- `role_of_F_vs_X.tex` - Clarifies regime F vs treatment X semantics
- `intervention_semantics.tex` - Soft interventions with known targets
- `draft_main.tex` - Paper draft with theoretical framework
