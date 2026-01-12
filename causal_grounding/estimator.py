"""
CausalGroundingEstimator - Main Estimator Class

This module assembles all components for CATE bound estimation using
causal grounding with naturalness constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from .discretize import discretize_covariates, get_discretized_covariate_names
from .ci_tests import CITestEngine
from .covariate_scoring import rank_covariates_across_sites, select_best_instrument
from .lp_solver import solve_all_bounds
from .transfer import (
    transfer_bounds_conservative,
    transfer_bounds_average,
    compute_bound_metrics,
    bounds_to_dataframe
)


class CausalGroundingEstimator:
    """
    Estimate CATE bounds using causal grounding with naturalness constraints.

    Algorithm:
    1. Discretize continuous covariates
    2. Score covariates on modified EHS criteria (using CMI)
    3. Select best instrumental covariate Z_a
    4. Learn P(Y_x|z) from F=on (RCT) data
    5. Solve LP bounds for each training site
    6. Transfer bounds to target environment

    Example:
        estimator = CausalGroundingEstimator(epsilon=0.1)
        estimator.fit(training_data, treatment='iv', outcome='dv')
        bounds_df = estimator.predict_bounds(target_data)

    References:
        Silva, "Causal Discovery Grounding and the Naturalness Assumption"
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        transfer_method: str = 'conservative',
        ci_alpha: float = 0.05,
        n_permutations: int = 500,
        discretize: bool = True,
        age_bins: int = 5,
        use_full_lp: bool = False,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            epsilon: Naturalness tolerance for LP constraints
            transfer_method: 'conservative' or 'average'
            ci_alpha: Significance level for CI tests
            n_permutations: Permutations for CMI p-values
            discretize: Whether to discretize continuous covariates
            age_bins: Number of bins for age discretization
            use_full_lp: Use full LP solver (slower but more accurate)
            random_seed: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.epsilon = epsilon
        self.transfer_method = transfer_method
        self.ci_alpha = ci_alpha
        self.n_permutations = n_permutations
        self.discretize = discretize
        self.age_bins = age_bins
        self.use_full_lp = use_full_lp
        self.random_seed = random_seed
        self.verbose = verbose

        # Fitted attributes (set by fit())
        self.is_fitted_ = False
        self.covariates_ = None
        self.best_instrument_ = None
        self.covariate_scores_ = None
        self.training_bounds_ = None
        self.transferred_bounds_ = None
        self.ci_engine_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(
        self,
        training_data: Dict[str, pd.DataFrame],
        treatment: str = 'iv',
        outcome: str = 'dv',
        covariates: Optional[List[str]] = None,
        regime_col: str = 'F'
    ) -> 'CausalGroundingEstimator':
        """
        Fit on training environments.

        Args:
            training_data: Dict[site_id -> DataFrame with regime_col]
                Each DataFrame must have 'F' column with 'on'/'idle' values
            treatment: Treatment column name
            outcome: Outcome column name
            covariates: Covariate columns (default: discretized standard covariates)
            regime_col: Regime indicator column name

        Returns:
            self
        """
        self._log("Fitting CausalGroundingEstimator...")

        # Step 1: Discretize if needed
        if self.discretize:
            self._log("  Step 1: Discretizing covariates...")
            training_data = {
                site: discretize_covariates(df, age_bins=self.age_bins)
                for site, df in training_data.items()
            }

        # Set covariates
        if covariates is None:
            self.covariates_ = get_discretized_covariate_names()
        else:
            self.covariates_ = covariates

        self._log(f"  Using covariates: {self.covariates_}")

        # Step 2: Initialize CI engine
        self.ci_engine_ = CITestEngine(
            test_alpha=self.ci_alpha,
            n_permutations=self.n_permutations,
            random_seed=self.random_seed
        )

        # Step 3: Score covariates on EHS criteria
        self._log("  Step 2: Scoring covariates on EHS criteria...")
        self.covariate_scores_ = rank_covariates_across_sites(
            training_data,
            self.covariates_,
            treatment,
            outcome,
            ci_engine=self.ci_engine_,
            use_permutation_test=True,
            regime_col=regime_col
        )

        self.best_instrument_ = select_best_instrument(
            self.covariate_scores_,
            require_passes_ehs=False
        )
        self._log(f"  Best instrument: {self.best_instrument_}")

        # Step 4: Solve LP bounds for each training site
        self._log("  Step 3: Solving LP bounds...")
        self.training_bounds_ = solve_all_bounds(
            training_data,
            self.covariates_,
            treatment,
            outcome,
            epsilon=self.epsilon,
            regime_col=regime_col,
            use_full_lp=self.use_full_lp
        )
        self._log(f"  Computed bounds for {len(self.training_bounds_)} sites")

        # Step 5: Transfer bounds
        self._log("  Step 4: Transferring bounds...")
        if self.transfer_method == 'conservative':
            self.transferred_bounds_ = transfer_bounds_conservative(self.training_bounds_)
        elif self.transfer_method == 'average':
            self.transferred_bounds_ = transfer_bounds_average(self.training_bounds_)
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")

        self.is_fitted_ = True
        self._log("  Done!")

        return self

    def predict_bounds(
        self,
        target_data: Optional[pd.DataFrame] = None,
        z_values: Optional[List[Tuple]] = None,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, Dict[Tuple, Tuple]]:
        """
        Predict CATE bounds for target environment.

        Args:
            target_data: Optional target DataFrame (for z support)
            z_values: Optional specific z values to predict
            return_dataframe: Return DataFrame (True) or dict (False)

        Returns:
            DataFrame with columns: z components, cate_lower, cate_upper, width
            Or dict: {z_value: (lower, upper)}
        """
        self._check_is_fitted()

        if z_values is None:
            z_values = list(self.transferred_bounds_.keys())

        bounds = {z: self.transferred_bounds_.get(z, (-1, 1)) for z in z_values}

        if return_dataframe:
            return bounds_to_dataframe(bounds, self.covariates_)
        return bounds

    def get_diagnostics(self) -> Dict:
        """Return fitting diagnostics."""
        self._check_is_fitted()

        metrics = compute_bound_metrics(self.transferred_bounds_)

        return {
            'n_training_sites': len(self.training_bounds_),
            'n_z_values': len(self.transferred_bounds_),
            'best_instrument': self.best_instrument_,
            'epsilon': self.epsilon,
            'transfer_method': self.transfer_method,
            **metrics
        }

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "Estimator not fitted. Call fit() before predict_bounds()."
            )


# Test
if __name__ == "__main__":
    print("CausalGroundingEstimator test")
    print("=" * 50)

    # This test requires actual data - will skip if not available
    from pathlib import Path
    from .train_target_split import create_train_target_split, load_rct_data

    osrct_path = Path('confounded_datasets/anchoring1/anchoring1_age_beta_0.3.pkl')
    rct_path = Path('ManyLabs1/pre-process/Manylabs1_data.pkl')

    if osrct_path.exists() and rct_path.exists():
        osrct_data = pd.read_pickle(osrct_path)
        rct_data = load_rct_data('anchoring1', str(rct_path))

        training, target = create_train_target_split(
            osrct_data, rct_data, target_site='mturk'
        )

        print(f"Training sites: {list(training.keys())[:5]}...")
        print(f"Target size: {len(target)}")

        # Fit estimator (use fewer permutations for speed)
        estimator = CausalGroundingEstimator(
            epsilon=0.1,
            n_permutations=100,
            random_seed=42
        )

        estimator.fit(training, treatment='iv', outcome='dv')

        print("\nDiagnostics:")
        for k, v in estimator.get_diagnostics().items():
            print(f"  {k}: {v}")

        print("\nBounds:")
        bounds_df = estimator.predict_bounds()
        print(bounds_df.head(10))
    else:
        print("Test data not available - skipping integration test")
