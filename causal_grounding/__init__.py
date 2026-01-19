"""
Causal Grounding Module
=======================

Partial identification bounds for CATE estimation under unmeasured confounding,
following Silva's "Causal Discovery Grounding and the Naturalness Assumption".

Main Class:
    CausalGroundingEstimator - Fit on training environments, predict bounds

Supporting Classes:
    CITestEngine - Conditional independence testing via CMI

Key Functions:
    discretize_covariates - Discretize continuous covariates for LP
    create_train_target_split - Split data into train/target environments
    rank_covariates - Rank covariates by EHS criteria
    solve_cate_bounds - LP solver for bounds

Example:
    from causal_grounding import CausalGroundingEstimator

    estimator = CausalGroundingEstimator(epsilon=0.1)
    estimator.fit(training_data, treatment='iv', outcome='dv')
    bounds = estimator.predict_bounds()
"""

__version__ = "0.1.0"

from .estimator import CausalGroundingEstimator

from .ci_tests import (
    CITestEngine,
    compute_cmi,
    permutation_test_cmi,
    combine_conditioning_vars,
)

from .discretize import (
    discretize_covariates,
    discretize_age,
    discretize_polideo,
    get_discretized_covariate_names,
)

from .train_target_split import (
    create_train_target_split,
    add_regime_indicator,
    get_available_sites,
    load_rct_data,
)

from .covariate_scoring import (
    rank_covariates,
    rank_covariates_across_sites,
    select_best_instrument,
)

from .lp_solver import (
    solve_cate_bounds_single_z,
    solve_all_bounds,
    estimate_identified_probs,
    estimate_observed_probs,
    solve_cate_bounds_lp_binary,
    solve_all_bounds_binary_lp,
)

from .transfer import (
    transfer_bounds_conservative,
    transfer_bounds_average,
    transfer_bounds_weighted,
    compute_bound_metrics,
    bounds_to_dataframe,
)

from .sensitivity import (
    SweepConfig,
    SweepResults,
    SensitivityAnalyzer,
    run_epsilon_sweep,
)

__all__ = [
    # Main estimator
    'CausalGroundingEstimator',

    # CI testing
    'CITestEngine',
    'compute_cmi',
    'permutation_test_cmi',
    'combine_conditioning_vars',

    # Preprocessing
    'discretize_covariates',
    'discretize_age',
    'discretize_polideo',
    'get_discretized_covariate_names',

    # Train/target split
    'create_train_target_split',
    'add_regime_indicator',
    'get_available_sites',
    'load_rct_data',

    # Covariate scoring
    'rank_covariates',
    'rank_covariates_across_sites',
    'select_best_instrument',

    # LP solver
    'solve_cate_bounds_single_z',
    'solve_all_bounds',
    'estimate_identified_probs',
    'estimate_observed_probs',
    'solve_cate_bounds_lp_binary',
    'solve_all_bounds_binary_lp',

    # Transfer
    'transfer_bounds_conservative',
    'transfer_bounds_average',
    'transfer_bounds_weighted',
    'compute_bound_metrics',
    'bounds_to_dataframe',

    # Sensitivity analysis
    'SweepConfig',
    'SweepResults',
    'SensitivityAnalyzer',
    'run_epsilon_sweep',
]
