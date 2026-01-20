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

from .ci_tests_l1 import (
    L1RegressionCIEngine,
    create_ci_engine,
)

# LOCO CI testing (optional - requires group-lasso for lasso function class)
try:
    from .ci_tests_loco import LOCOCIEngine
    _LOCO_AVAILABLE = True
except ImportError:
    _LOCO_AVAILABLE = False

from .predictors import (
    BasePredictorModel,
    EmpiricalPredictor,
    create_predictor,
)

# XGBoost predictor is optional
try:
    from .predictors import XGBoostPredictor, XGBoostBackdoorModel
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from .marginal_effects import (
    MCEstimationResult,
    EmpiricalCovariateDistribution,
    MarginalCATEEstimator,
    estimate_marginal_cate_simple,
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
    select_top_k_instruments,
    get_instrument_bounds_for_aggregation,
)

from .lp_solver import (
    solve_cate_bounds_single_z,
    solve_all_bounds,
    estimate_identified_probs,
    estimate_observed_probs,
    solve_cate_bounds_lp_binary,
    solve_all_bounds_binary_lp,
)

from .lp_solver_extended import (
    ExtendedLPResult,
    ExtendedLPSolver,
    solve_extended_bounds_all_strata,
    compare_simple_vs_extended,
    create_lp_solver,
)

from .transfer import (
    transfer_bounds_conservative,
    transfer_bounds_average,
    transfer_bounds_weighted,
    compute_bound_metrics,
    bounds_to_dataframe,
    aggregate_across_instruments,
    aggregate_with_weights,
    compute_instrument_agreement,
)

from .sensitivity import (
    SweepConfig,
    SweepResults,
    SensitivityAnalyzer,
    run_epsilon_sweep,
)

from .evaluation import (
    compute_coverage_rate,
    compute_informativeness,
    compute_interval_score,
    compute_sharpness,
    summarize_bound_quality,
    compare_method_quality,
    per_stratum_coverage,
)

from .comparison import (
    MethodComparator,
    ComparisonResults,
    MethodConfig,
    compare_ci_engines,
    compare_predictors,
    compare_lp_solvers,
    quick_compare,
)

from .visualization import (
    plot_cate_bounds,
    plot_bounds_forest,
    plot_bounds_comparison,
    plot_coverage_by_stratum,
    plot_coverage_heatmap,
    plot_width_distribution,
    plot_width_vs_sample_size,
    plot_ehs_scores,
    plot_cmi_distribution,
    plot_method_comparison_summary,
    plot_runtime_comparison,
    plot_agreement_matrix,
    save_figure,
    create_multi_panel_figure,
)

from .simulator import (
    BinarySyntheticDGP,
    generate_random_dgp,
    simulate_observational,
    simulate_rct,
    compute_true_cate,
    compute_true_ate,
    generate_multi_environment_data,
    add_regime_indicator,
    get_covariate_stratum,
)

# Ricardo adapter is optional - may not be available
try:
    from .ricardo_adapter import (
        RicardoMethodAdapter,
        compare_with_ricardo,
        create_adapter,
    )
    _RICARDO_AVAILABLE = True
except ImportError:
    _RICARDO_AVAILABLE = False

__all__ = [
    # Main estimator
    'CausalGroundingEstimator',

    # CI testing
    'CITestEngine',
    'compute_cmi',
    'permutation_test_cmi',
    'combine_conditioning_vars',

    # L1-Regression CI testing
    'L1RegressionCIEngine',
    'create_ci_engine',

    # LOCO CI testing (optional)
    'LOCOCIEngine',

    # Prediction models
    'BasePredictorModel',
    'EmpiricalPredictor',
    'XGBoostPredictor',
    'XGBoostBackdoorModel',
    'create_predictor',

    # Marginal effects
    'MCEstimationResult',
    'EmpiricalCovariateDistribution',
    'MarginalCATEEstimator',
    'estimate_marginal_cate_simple',

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
    'select_top_k_instruments',
    'get_instrument_bounds_for_aggregation',

    # LP solver
    'solve_cate_bounds_single_z',
    'solve_all_bounds',
    'estimate_identified_probs',
    'estimate_observed_probs',
    'solve_cate_bounds_lp_binary',
    'solve_all_bounds_binary_lp',

    # Extended LP solver
    'ExtendedLPResult',
    'ExtendedLPSolver',
    'solve_extended_bounds_all_strata',
    'compare_simple_vs_extended',
    'create_lp_solver',

    # Transfer
    'transfer_bounds_conservative',
    'transfer_bounds_average',
    'transfer_bounds_weighted',
    'compute_bound_metrics',
    'bounds_to_dataframe',
    'aggregate_across_instruments',
    'aggregate_with_weights',
    'compute_instrument_agreement',

    # Sensitivity analysis
    'SweepConfig',
    'SweepResults',
    'SensitivityAnalyzer',
    'run_epsilon_sweep',

    # Evaluation metrics
    'compute_coverage_rate',
    'compute_informativeness',
    'compute_interval_score',
    'compute_sharpness',
    'summarize_bound_quality',
    'compare_method_quality',
    'per_stratum_coverage',

    # Comparison utilities
    'MethodComparator',
    'ComparisonResults',
    'MethodConfig',
    'compare_ci_engines',
    'compare_predictors',
    'compare_lp_solvers',
    'quick_compare',

    # Visualization
    'plot_cate_bounds',
    'plot_bounds_forest',
    'plot_bounds_comparison',
    'plot_coverage_by_stratum',
    'plot_coverage_heatmap',
    'plot_width_distribution',
    'plot_width_vs_sample_size',
    'plot_ehs_scores',
    'plot_cmi_distribution',
    'plot_method_comparison_summary',
    'plot_runtime_comparison',
    'plot_agreement_matrix',
    'save_figure',
    'create_multi_panel_figure',

    # Simulator
    'BinarySyntheticDGP',
    'generate_random_dgp',
    'simulate_observational',
    'simulate_rct',
    'compute_true_cate',
    'compute_true_ate',
    'generate_multi_environment_data',
    'add_regime_indicator',
    'get_covariate_stratum',

    # Ricardo adapter (optional)
    'RicardoMethodAdapter',
    'compare_with_ricardo',
    'create_adapter',
]
