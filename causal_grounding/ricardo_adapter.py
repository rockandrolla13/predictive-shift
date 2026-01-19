"""
Ricardo Method Adapter

This module provides an adapter to run Ricardo's original implementation
for comparison with our production causal_grounding code.

The adapter wraps Ricardo's code from the 'Ricardo Code/' directory
and exposes it through a consistent interface.

Note: Ricardo's code requires xgboost, multipledispatch, and sklearn.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


# Path to Ricardo's code directory (relative to this file)
RICARDO_CODE_PATH = Path(__file__).parent.parent / 'Ricardo Code'


@dataclass
class RicardoMethodConfig:
    """Configuration for Ricardo's methods."""
    use_xgboost: bool = True
    num_mc_samples: int = 1000
    sparse_pattern: Optional[np.ndarray] = None


class RicardoMethodAdapter:
    """
    Adapter to run Ricardo's implementation for comparison.

    This class dynamically imports Ricardo's code and provides
    a consistent interface for running his methods.

    Example:
        adapter = RicardoMethodAdapter()
        if adapter.is_available():
            # Generate synthetic data
            model = adapter.create_binary_synthetic_model(n_covariates=5)
            data = adapter.simulate_observational(model, n_samples=1000)

            # Compute CATE
            cate = adapter.compute_cate(data, model)
    """

    def __init__(self, ricardo_path: Optional[Union[str, Path]] = None):
        """
        Initialize adapter.

        Args:
            ricardo_path: Path to Ricardo's code directory.
                         Uses default RICARDO_CODE_PATH if None.
        """
        self.ricardo_path = Path(ricardo_path) if ricardo_path else RICARDO_CODE_PATH
        self._module = None
        self._available = None
        self._load_error = None

    def is_available(self) -> bool:
        """Check if Ricardo's code is available and can be imported."""
        if self._available is not None:
            return self._available

        try:
            self._load_module()
            self._available = True
        except Exception as e:
            self._available = False
            self._load_error = str(e)

        return self._available

    def get_load_error(self) -> Optional[str]:
        """Get error message if loading failed."""
        return self._load_error

    def _load_module(self):
        """Dynamically load Ricardo's simulator module."""
        if self._module is not None:
            return

        simulator_path = self.ricardo_path / 'simulator.py'

        if not simulator_path.exists():
            raise FileNotFoundError(
                f"Ricardo's simulator.py not found at {simulator_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "ricardo_simulator",
            simulator_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["ricardo_simulator"] = module
        spec.loader.exec_module(module)

        self._module = module

    def _ensure_loaded(self):
        """Ensure module is loaded, raise if not available."""
        if not self.is_available():
            raise RuntimeError(
                f"Ricardo's code not available: {self._load_error}"
            )

    # =========================================================================
    # Binary Synthetic Model Methods
    # =========================================================================

    def create_binary_synthetic_model(
        self,
        n_covariates: int,
        sparse_pattern: Optional[np.ndarray] = None
    ) -> Any:
        """
        Create a BinarySyntheticBackdoorModel with random parameters.

        Args:
            n_covariates: Number of binary covariates
            sparse_pattern: Optional sparsity pattern for coefficients

        Returns:
            BinarySyntheticBackdoorModel instance
        """
        self._ensure_loaded()
        return self._module.simulate_binary_synthetic_backdoor_parameters(
            n_covariates, sparse_pattern
        )

    def simulate_observational(
        self,
        model: Any,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Simulate observational (confounded) data from model.

        Args:
            model: BinarySyntheticBackdoorModel or XGBoostBackdoorModel
            n_samples: Number of samples to generate

        Returns:
            DataFrame with covariates, treatment, and outcome
        """
        self._ensure_loaded()
        return self._module.simulate_from_backdoor_model(n_samples, model)

    def simulate_rct(
        self,
        model: Any,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Simulate RCT (randomized) data from model.

        Args:
            model: BinarySyntheticBackdoorModel or XGBoostBackdoorModel
            n_samples: Number of samples to generate

        Returns:
            DataFrame with covariates, treatment (randomized), and outcome
        """
        self._ensure_loaded()
        return self._module.simulate_from_controlled_backdoor_model(n_samples, model)

    def compute_cate(
        self,
        data: pd.DataFrame,
        model: Any
    ) -> np.ndarray:
        """
        Compute true CATE from model for given covariate values.

        Args:
            data: DataFrame with covariate values
            model: BinarySyntheticBackdoorModel or XGBoostBackdoorModel

        Returns:
            Array of CATE values
        """
        self._ensure_loaded()
        return self._module.get_cate(data, model)

    def compute_montecarlo_cate(
        self,
        data: pd.DataFrame,
        observed_covariates: List[str],
        n_simulations: int,
        model: Any,
        use_logistic: bool = False
    ) -> np.ndarray:
        """
        Compute Monte Carlo CATE estimate for subset of covariates.

        Args:
            data: DataFrame with observed covariate values
            observed_covariates: List of observed covariate names
            n_simulations: Number of MC simulations
            model: XGBoostBackdoorModel
            use_logistic: Use logistic regression instead of XGBoost

        Returns:
            Array of marginal CATE estimates
        """
        self._ensure_loaded()
        return self._module.get_montecarlo_cate(
            data, observed_covariates, n_simulations, model, use_logistic
        )

    # =========================================================================
    # XGBoost Model Methods
    # =========================================================================

    def learn_backdoor_model(
        self,
        data: pd.DataFrame,
        covariate_cols: List[str],
        treatment_col: str,
        outcome_col: str
    ) -> Any:
        """
        Learn XGBoostBackdoorModel from data.

        Args:
            data: Training DataFrame
            covariate_cols: List of covariate column names
            treatment_col: Treatment column name
            outcome_col: Outcome column name

        Returns:
            XGBoostBackdoorModel instance
        """
        self._ensure_loaded()
        return self._module.learn_backdoor_model(
            data, covariate_cols, treatment_col, outcome_col
        )

    def select_confounders(
        self,
        model: Any,
        keep_k: int
    ) -> List[str]:
        """
        Select top-k confounders based on feature importance.

        Args:
            model: XGBoostBackdoorModel
            keep_k: Number of covariates to keep

        Returns:
            List of selected covariate names
        """
        self._ensure_loaded()
        return self._module.confounder_column_keeper(keep_k, model)

    def estimate_cate_tlearner(
        self,
        data: pd.DataFrame,
        covariate_cols: List[str],
        model: Any
    ) -> tuple:
        """
        Estimate CATE using T-learner.

        Args:
            data: DataFrame with data
            covariate_cols: Covariate columns to use
            model: Model specifying column names

        Returns:
            Tuple of (cate_estimates, outcome_models)
        """
        self._ensure_loaded()
        return self._module.estimate_cate_tlearner(data, covariate_cols, model)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def categorize_dataframe(
        self,
        df: pd.DataFrame,
        cat_threshold: int = 10,
        full_cat: bool = False
    ) -> None:
        """
        Convert columns to categorical where unique values <= threshold.

        Args:
            df: DataFrame to modify (in-place)
            cat_threshold: Threshold for categorization
            full_cat: If True, use KMeans clustering for continuous vars
        """
        self._ensure_loaded()
        self._module.dataframe_categorize(df, cat_threshold, full_cat)

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Any Ricardo model type

        Returns:
            Dict with model information
        """
        info = {
            'type': type(model).__name__,
            'x_idx': getattr(model, 'x_idx', None),
            'a_idx': getattr(model, 'a_idx', None),
            'y_idx': getattr(model, 'y_idx', None),
        }

        if hasattr(model, 'prob_x'):
            info['n_covariates'] = len(model.prob_x)
            info['coeff_a'] = model.coeff_a.tolist()
            info['coeff_y0'] = model.coeff_y0.tolist()
            info['coeff_y1'] = model.coeff_y1.tolist()

        return info


def compare_with_ricardo(
    our_results: Dict[str, Any],
    ricardo_results: Dict[str, Any],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Compare our implementation results with Ricardo's.

    Args:
        our_results: Dict with our method's outputs
        ricardo_results: Dict with Ricardo's method's outputs
        tolerance: Tolerance for numerical comparisons

    Returns:
        Dict with comparison metrics
    """
    comparison = {
        'matches': {},
        'differences': {},
        'metrics': {}
    }

    # Compare bounds if present
    if 'bounds' in our_results and 'bounds' in ricardo_results:
        our_bounds = our_results['bounds']
        ric_bounds = ricardo_results['bounds']

        if isinstance(our_bounds, pd.DataFrame) and isinstance(ric_bounds, pd.DataFrame):
            # Compare lower bounds
            if 'lower' in our_bounds.columns and 'lower' in ric_bounds.columns:
                lower_diff = np.abs(our_bounds['lower'].values - ric_bounds['lower'].values)
                comparison['metrics']['lower_bound_mae'] = float(np.mean(lower_diff))
                comparison['metrics']['lower_bound_max_diff'] = float(np.max(lower_diff))
                comparison['matches']['lower_bounds'] = np.all(lower_diff < tolerance)

            # Compare upper bounds
            if 'upper' in our_bounds.columns and 'upper' in ric_bounds.columns:
                upper_diff = np.abs(our_bounds['upper'].values - ric_bounds['upper'].values)
                comparison['metrics']['upper_bound_mae'] = float(np.mean(upper_diff))
                comparison['metrics']['upper_bound_max_diff'] = float(np.max(upper_diff))
                comparison['matches']['upper_bounds'] = np.all(upper_diff < tolerance)

    # Compare CATE if present
    if 'cate' in our_results and 'cate' in ricardo_results:
        our_cate = np.asarray(our_results['cate'])
        ric_cate = np.asarray(ricardo_results['cate'])

        if len(our_cate) == len(ric_cate):
            cate_diff = np.abs(our_cate - ric_cate)
            comparison['metrics']['cate_mae'] = float(np.mean(cate_diff))
            comparison['metrics']['cate_correlation'] = float(np.corrcoef(our_cate, ric_cate)[0, 1])
            comparison['matches']['cate'] = np.all(cate_diff < tolerance)

    # Compare coverage if present
    if 'coverage' in our_results and 'coverage' in ricardo_results:
        comparison['metrics']['our_coverage'] = our_results['coverage']
        comparison['metrics']['ricardo_coverage'] = ricardo_results['coverage']
        comparison['metrics']['coverage_diff'] = abs(
            our_results['coverage'] - ricardo_results['coverage']
        )

    # Overall match
    comparison['overall_match'] = all(comparison['matches'].values()) if comparison['matches'] else None

    return comparison


def create_adapter() -> RicardoMethodAdapter:
    """
    Factory function to create a RicardoMethodAdapter.

    Returns:
        RicardoMethodAdapter instance

    Raises:
        RuntimeError if Ricardo's code is not available
    """
    adapter = RicardoMethodAdapter()
    if not adapter.is_available():
        raise RuntimeError(
            f"Ricardo's code is not available: {adapter.get_load_error()}"
        )
    return adapter


# Module test
if __name__ == "__main__":
    print("Ricardo Method Adapter")
    print("=" * 50)

    adapter = RicardoMethodAdapter()

    if adapter.is_available():
        print("Status: Ricardo's code is available")

        # Test creating a model
        print("\nTesting binary synthetic model creation...")
        model = adapter.create_binary_synthetic_model(n_covariates=5)
        print(f"  Created model with {len(model.x_idx)} covariates")

        # Test simulation
        print("\nTesting simulation...")
        obs_data = adapter.simulate_observational(model, n_samples=100)
        rct_data = adapter.simulate_rct(model, n_samples=100)
        print(f"  Observational data shape: {obs_data.shape}")
        print(f"  RCT data shape: {rct_data.shape}")

        # Test CATE computation
        print("\nTesting CATE computation...")
        cate = adapter.compute_cate(obs_data, model)
        print(f"  CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
        print(f"  CATE mean: {cate.mean():.4f}")

        print("\nAll tests passed!")
    else:
        print(f"Status: Ricardo's code is NOT available")
        print(f"Error: {adapter.get_load_error()}")
        print("\nTo use Ricardo's methods, ensure:")
        print(f"  1. Directory exists: {RICARDO_CODE_PATH}")
        print("  2. Required packages installed: xgboost, multipledispatch, sklearn")
