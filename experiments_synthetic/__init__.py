"""
Synthetic Data Experiments for Causal Grounding
================================================

This module provides experiments on synthetic data with binary treatment,
binary outcome, and discrete covariates. The synthetic data has known
ground truth CATEs, allowing for precise validation of the method.

Modules:
    synthetic_data - Generate synthetic data with controlled confounding
    run_experiment - Main experiment runner
    integration_test - Validation tests
    validate_cate - CATE coverage validation

Usage Examples:
    # Generate synthetic data
    from experiments_synthetic import SyntheticDataGenerator, SyntheticDataConfig
    config = SyntheticDataConfig(n_z_values=3, beta=0.3)
    generator = SyntheticDataGenerator(config, n_sites=10)
    training_data = generator.generate_training_data()
    
    # Run experiment
    from experiments_synthetic import run_synthetic_experiment
    result = run_synthetic_experiment(beta=0.3, epsilon=0.1)
    
    # Validate coverage
    from experiments_synthetic import validate_single_experiment
    validation = validate_single_experiment(beta=0.3, epsilon=0.1)

Command Line:
    # Run single experiment
    python -m experiments_synthetic.run_experiment --beta 0.3
    
    # Run grid of experiments
    python -m experiments_synthetic.run_experiment --grid --report
    
    # Run integration tests
    python -m experiments_synthetic.integration_test
    
    # Validate coverage
    python -m experiments_synthetic.validate_cate --sweep-beta
"""

from .synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_multi_site_data,
    compute_true_cate,
    create_confounding_sweep_configs,
    create_heterogeneity_sweep_configs,
)

from .run_experiment import (
    run_synthetic_experiment,
    run_synthetic_grid,
    run_confounding_sweep,
    run_epsilon_sweep,
    adapt_training_data_columns,
    generate_report_synthetic,
    BETAS,
    EPSILONS,
)

from .validate_cate import (
    validate_single_experiment,
    validate_coverage_sweep,
    compute_coverage_statistics,
    generate_validation_report,
)

__all__ = [
    # Synthetic data generation
    'SyntheticDataGenerator',
    'SyntheticDataConfig',
    'generate_multi_site_data',
    'compute_true_cate',
    'create_confounding_sweep_configs',
    'create_heterogeneity_sweep_configs',
    
    # Experiment runners
    'run_synthetic_experiment',
    'run_synthetic_grid',
    'run_confounding_sweep',
    'run_epsilon_sweep',
    'adapt_training_data_columns',
    'generate_report_synthetic',
    
    # Validation
    'validate_single_experiment',
    'validate_coverage_sweep',
    'compute_coverage_statistics',
    'generate_validation_report',
    
    # Constants
    'BETAS',
    'EPSILONS',
]
