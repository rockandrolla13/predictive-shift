# Synthetic Data Experiments for Causal Grounding

This module provides experiments on synthetic data with binary treatment, binary outcome, and discrete covariates. The synthetic data has known ground truth CATEs, allowing for precise validation of the method.

## Module Structure

```
experiments_synthetic/
├── __init__.py           # Package exports
├── synthetic_data.py     # Data generation with known ground truth
├── run_experiment.py     # Main experiment runner
├── integration_test.py   # Comprehensive tests
├── validate_cate.py      # CATE coverage validation
└── README.md             # This file
```

## Quick Start

### Generate Synthetic Data

```python
from experiments_synthetic import SyntheticDataGenerator, SyntheticDataConfig

# Create configuration
config = SyntheticDataConfig(
    n_z_values=3,          # Number of discrete covariate values
    beta=0.3,              # Confounding strength (U -> X)
    gamma=0.2,             # Confounding strength (U -> Y)
    treatment_effect=0.2,  # Main treatment effect (ATE component)
    treatment_z_interaction=0.1,  # CATE heterogeneity
    n_per_site=500,        # Samples per training site
    seed=42
)

# Create generator
generator = SyntheticDataGenerator(config, n_sites=10)

# Generate training data
training_data = generator.generate_training_data()

# Get ground truth
true_cates = generator.get_true_cates()  # {(z,): CATE(z)}
true_ate = generator.get_true_ate()      # E[CATE(Z)]
```

### Run Single Experiment

```python
from experiments_synthetic import run_synthetic_experiment

result = run_synthetic_experiment(
    n_sites=10,
    n_per_site=500,
    n_z_values=3,
    beta=0.3,
    treatment_effect=0.2,
    treatment_z_interaction=0.1,
    epsilon=0.1,
    n_permutations=100,
    random_seed=42
)

# Access results
print(f"ATE covered: {result['metrics']['ate_covered']}")
print(f"CATE coverage: {result['metrics']['cate_coverage_rate']:.1%}")
print(f"Mean width: {result['metrics']['mean_width']:.3f}")
```

### Run Grid of Experiments

```python
from experiments_synthetic import run_synthetic_grid

results_df = run_synthetic_grid(
    betas=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    epsilons=[0.1],
    n_sites_list=[10],
    n_z_values=3,
    n_per_site=500,
    output_dir='results_synthetic'
)
```

## Command Line Usage

### Single Experiment
```bash
python experiments_synthetic/run_experiment.py --beta 0.3 --epsilon 0.1 --n-sites 10
```

### Grid Experiment with Report
```bash
python experiments_synthetic/run_experiment.py --grid --report --output results_synthetic
```

### Comprehensive Sweep
```bash
python experiments_synthetic/run_experiment.py --all --report
```

### Run Integration Tests
```bash
python experiments_synthetic/integration_test.py
python experiments_synthetic/integration_test.py --quick  # Faster subset
```

### Validate CATE Coverage
```bash
# Single configuration
python experiments_synthetic/validate_cate.py --beta 0.3 --epsilon 0.1

# Sweep over beta values
python experiments_synthetic/validate_cate.py --sweep-beta

# Sweep over epsilon values
python experiments_synthetic/validate_cate.py --sweep-epsilon
```

## Data Generating Process

The synthetic data follows this structural causal model:

```
Z ~ Categorical(p_z)           # Discrete covariate
U ~ Bernoulli(p_u)             # Unmeasured confounder

# Treatment assignment
Under F=on (RCT):
    X ~ Bernoulli(0.5)         # Randomized
Under F=idle (Obs):
    X|Z,U ~ Bernoulli(π(Z) + β·U)  # Confounded

# Outcome
Y|X,Z,U ~ Bernoulli(μ(X,Z,U))
where μ(X,Z,U) = base + τ·X + δ·Z + γ_xz·X·Z + γ·U
```

The **true CATE** for each covariate stratum is:
```
CATE(z) = τ + γ_xz · z
```

Key parameters:
- `beta` (β): Controls confounding strength in treatment assignment
- `gamma` (γ): Controls confounding strength in outcome
- `treatment_effect` (τ): Main treatment effect
- `treatment_z_interaction` (γ_xz): Treatment effect heterogeneity

## Output Files

Grid experiments produce:
- `synthetic_grid_TIMESTAMP.csv` - Main results
- `synthetic_grid_TIMESTAMP.json` - Detailed results
- `figures/` - Coverage and width plots
- `tables/` - LaTeX tables
- `EXPERIMENT_REPORT.md` - Summary report

## Integration with CausalGroundingEstimator

The module adapts synthetic data for use with `CausalGroundingEstimator`:

```python
from experiments_synthetic import adapt_training_data_columns

# Adapt column names (X->iv, Y->dv, Z->Z_cat)
adapted_data = adapt_training_data_columns(training_data)

# Now can use with estimator
from causal_grounding import CausalGroundingEstimator
estimator = CausalGroundingEstimator(epsilon=0.1, discretize=False)
estimator.fit(adapted_data, treatment='iv', outcome='dv', covariates=['Z_cat'])
bounds = estimator.predict_bounds()
```

## Key Differences from OSRCT Experiments

| Aspect | OSRCT Experiments | Synthetic Experiments |
|--------|------------------|----------------------|
| Data source | ManyLabs1 studies | Generated |
| Ground truth | Estimated from RCT | Analytically known |
| Covariates | Age, gender, polideo | Discrete Z |
| Sites | Lab locations | Simulated environments |
| Confounding | Pattern-based | Parameter-controlled |

## Example Results

Typical output from a single experiment:

```
Configuration:
  Sites: 10
  Samples per site: 500
  Z values: 3
  Beta (confounding): 0.3
  Treatment effect: 0.2
  Interaction: 0.1
  Epsilon: 0.1

Ground Truth & Coverage:
  True ATE: 0.300
  ATE covered: True
  CATE coverage rate: 100.0%

Per-stratum Results:
  Z=(0,): True=0.200, Bounds=[-0.138, 0.499] ✓
  Z=(1,): True=0.300, Bounds=[0.119, 0.650] ✓
  Z=(2,): True=0.400, Bounds=[0.122, 0.603] ✓
```
