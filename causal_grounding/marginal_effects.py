"""
Marginal Effects via Monte Carlo Integration

This module provides Monte Carlo methods for estimating marginal CATE
when only a subset of covariates is observed.

Based on Ricardo's get_montecarlo_cate implementation from method.ipynb.

Key Classes:
    MarginalCATEEstimator - Estimates E[CATE(X_obs, X_unobs)] by integrating
                           over the distribution of unobserved covariates

    EmpiricalCovariateDistribution - Samples unobserved covariate values
                                    from the empirical distribution
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

from .predictors import BasePredictorModel, EmpiricalPredictor, create_predictor


@dataclass
class MCEstimationResult:
    """
    Result of Monte Carlo CATE estimation.

    Attributes:
        mean_cate: Mean CATE estimate across MC samples
        std_cate: Standard deviation of CATE estimates
        cate_samples: Array of individual MC estimates (if requested)
        n_samples: Number of MC samples used
        observed_indices: Indices of observed covariates
        unobserved_indices: Indices of unobserved covariates
    """
    mean_cate: float
    std_cate: float
    cate_samples: Optional[np.ndarray] = None
    n_samples: int = 0
    observed_indices: Optional[List[int]] = None
    unobserved_indices: Optional[List[int]] = None


class EmpiricalCovariateDistribution:
    """
    Empirical distribution for sampling unobserved covariate values.

    Samples from the empirical marginal distribution of covariates
    observed in the training data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        covariate_names: List[str],
        joint_sampling: bool = False
    ):
        """
        Initialize empirical covariate distribution.

        Args:
            data: Training DataFrame
            covariate_names: List of covariate column names
            joint_sampling: If True, sample covariate vectors jointly;
                           if False, sample each covariate independently
        """
        self.covariate_names = covariate_names
        self.joint_sampling = joint_sampling
        self.n_covariates = len(covariate_names)

        if joint_sampling:
            # Store full covariate vectors for joint sampling
            self._covariate_data = data[covariate_names].values
        else:
            # Store each covariate's empirical distribution separately
            self._marginal_values = {}
            for name in covariate_names:
                self._marginal_values[name] = data[name].values

    def sample(
        self,
        n_samples: int,
        observed_values: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample covariate vectors from empirical distribution.

        Args:
            n_samples: Number of samples to draw
            observed_values: Dict of covariate_name -> value for observed covariates.
                           These will be fixed across all samples.
            seed: Random seed

        Returns:
            DataFrame with sampled covariate values
        """
        if seed is not None:
            np.random.seed(seed)

        observed_values = observed_values or {}

        if self.joint_sampling:
            # Sample full vectors from training data
            idx = np.random.choice(len(self._covariate_data), n_samples, replace=True)
            samples = pd.DataFrame(
                self._covariate_data[idx],
                columns=self.covariate_names
            )
        else:
            # Sample each covariate independently
            samples = {}
            for name in self.covariate_names:
                if name in observed_values:
                    # Use fixed observed value
                    samples[name] = np.full(n_samples, observed_values[name])
                else:
                    # Sample from empirical marginal
                    vals = self._marginal_values[name]
                    idx = np.random.choice(len(vals), n_samples, replace=True)
                    samples[name] = vals[idx]
            samples = pd.DataFrame(samples)

        # Override with any fixed observed values
        for name, value in observed_values.items():
            if name in samples.columns:
                samples[name] = value

        return samples

    def get_covariate_probabilities(
        self,
        covariate_name: str
    ) -> Dict[Any, float]:
        """
        Get empirical probability distribution for a single covariate.

        Args:
            covariate_name: Name of the covariate

        Returns:
            Dict mapping values to their probabilities
        """
        if self.joint_sampling:
            idx = self.covariate_names.index(covariate_name)
            values = self._covariate_data[:, idx]
        else:
            values = self._marginal_values[covariate_name]

        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)

        return dict(zip(unique, probs))


class MarginalCATEEstimator:
    """
    Monte Carlo estimator for marginal CATE.

    When only a subset of covariates is observed, estimates the marginal
    CATE by integrating over the distribution of unobserved covariates:

        E[CATE(X_obs)] = E_{X_unobs}[CATE(X_obs, X_unobs)]

    This is approximated via Monte Carlo:

        CATE_marginal(x_obs) ≈ (1/M) Σ_m CATE(x_obs, x_unobs^m)

    where x_unobs^m ~ P(X_unobs) from the empirical distribution.

    Based on Ricardo's get_montecarlo_cate implementation.
    """

    def __init__(
        self,
        predictor: BasePredictorModel,
        covariate_distribution: EmpiricalCovariateDistribution,
        n_mc_samples: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize marginal CATE estimator.

        Args:
            predictor: Fitted predictor model (must be able to predict CATE)
            covariate_distribution: Distribution for sampling unobserved covariates
            n_mc_samples: Default number of MC samples
            random_state: Random seed for reproducibility
        """
        self.predictor = predictor
        self.covariate_distribution = covariate_distribution
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self._all_covariate_names = covariate_distribution.covariate_names

    def estimate_marginal_cate(
        self,
        observed_values: Dict[str, Any],
        n_samples: Optional[int] = None,
        return_samples: bool = False
    ) -> MCEstimationResult:
        """
        Estimate marginal CATE for given observed covariate values.

        Args:
            observed_values: Dict of covariate_name -> value for observed covariates
            n_samples: Number of MC samples (uses default if not specified)
            return_samples: Whether to return individual CATE samples

        Returns:
            MCEstimationResult with mean, std, and optionally samples

        Example:
            >>> estimator = MarginalCATEEstimator(predictor, covar_dist)
            >>> result = estimator.estimate_marginal_cate({'age': 1, 'gender': 0})
            >>> print(f"Marginal CATE: {result.mean_cate:.3f} ± {result.std_cate:.3f}")
        """
        n_samples = n_samples or self.n_mc_samples

        # Identify observed vs unobserved covariates
        observed_names = list(observed_values.keys())
        unobserved_names = [n for n in self._all_covariate_names if n not in observed_names]

        observed_idx = [self._all_covariate_names.index(n) for n in observed_names]
        unobserved_idx = [self._all_covariate_names.index(n) for n in unobserved_names]

        # Sample unobserved covariates from empirical distribution
        covariate_samples = self.covariate_distribution.sample(
            n_samples,
            observed_values=observed_values,
            seed=self.random_state
        )

        # Compute CATE for each sampled covariate vector
        cate_samples = self.predictor.predict_cate(covariate_samples)

        # Aggregate
        mean_cate = float(np.mean(cate_samples))
        std_cate = float(np.std(cate_samples))

        result = MCEstimationResult(
            mean_cate=mean_cate,
            std_cate=std_cate,
            n_samples=n_samples,
            observed_indices=observed_idx,
            unobserved_indices=unobserved_idx
        )

        if return_samples:
            result.cate_samples = cate_samples

        return result

    def estimate_marginal_cate_batch(
        self,
        observed_data: pd.DataFrame,
        observed_columns: List[str],
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate marginal CATE for multiple observation points.

        Args:
            observed_data: DataFrame with observed covariate values
            observed_columns: Column names that are observed
            n_samples: Number of MC samples per point

        Returns:
            Array of marginal CATE estimates
        """
        n_samples = n_samples or self.n_mc_samples
        n_points = len(observed_data)

        cate_estimates = np.zeros(n_points)

        for i in range(n_points):
            observed_values = {col: observed_data[col].iloc[i] for col in observed_columns}
            result = self.estimate_marginal_cate(observed_values, n_samples)
            cate_estimates[i] = result.mean_cate

        return cate_estimates

    def estimate_marginal_bounds(
        self,
        observed_values: Dict[str, Any],
        bounds_df: pd.DataFrame,
        stratum_column: str = 'stratum',
        lower_column: str = 'lower',
        upper_column: str = 'upper',
        n_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate marginal bounds by averaging over stratum-specific bounds.

        For partial identification, we need to aggregate bounds across
        the strata defined by unobserved covariates.

        Conservative approach: max of lowers, min of uppers weighted by P(stratum)

        Args:
            observed_values: Dict of covariate_name -> value for observed covariates
            bounds_df: DataFrame with stratum-specific bounds
            stratum_column: Column name for stratum identifier
            lower_column: Column name for lower bound
            upper_column: Column name for upper bound
            n_samples: Number of MC samples for weighting

        Returns:
            Dict with 'lower', 'upper', and 'width' for marginal bounds
        """
        n_samples = n_samples or self.n_mc_samples

        # Sample unobserved covariates
        covariate_samples = self.covariate_distribution.sample(
            n_samples,
            observed_values=observed_values,
            seed=self.random_state
        )

        # Create stratum keys for each sample
        stratum_keys = covariate_samples.astype(str).agg('-'.join, axis=1).values

        # Count frequency of each stratum in samples (weighting)
        unique_strata, counts = np.unique(stratum_keys, return_counts=True)
        stratum_weights = dict(zip(unique_strata, counts / n_samples))

        # Match strata in bounds_df
        bounds_dict = bounds_df.set_index(stratum_column).to_dict('index')

        # Compute weighted bounds
        weighted_lowers = []
        weighted_uppers = []
        total_weight = 0

        for stratum, weight in stratum_weights.items():
            if stratum in bounds_dict:
                lower = bounds_dict[stratum][lower_column]
                upper = bounds_dict[stratum][upper_column]
                weighted_lowers.append(lower * weight)
                weighted_uppers.append(upper * weight)
                total_weight += weight

        if total_weight > 0:
            marginal_lower = sum(weighted_lowers) / total_weight
            marginal_upper = sum(weighted_uppers) / total_weight
        else:
            # Fallback: use overall bounds
            marginal_lower = bounds_df[lower_column].min()
            marginal_upper = bounds_df[upper_column].max()

        return {
            'lower': marginal_lower,
            'upper': marginal_upper,
            'width': marginal_upper - marginal_lower
        }


def estimate_marginal_cate_simple(
    predictor: BasePredictorModel,
    training_data: pd.DataFrame,
    covariate_names: List[str],
    observed_values: Dict[str, Any],
    n_mc_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Simple interface for marginal CATE estimation.

    Convenience function that creates necessary objects and computes marginal CATE.

    Args:
        predictor: Fitted predictor model
        training_data: Training data for empirical distribution
        covariate_names: Names of all covariates
        observed_values: Dict of observed covariate values
        n_mc_samples: Number of MC samples
        seed: Random seed

    Returns:
        Dict with 'mean', 'std', 'ci_lower', 'ci_upper'

    Example:
        >>> predictor = EmpiricalPredictor()
        >>> predictor.fit(train_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])
        >>> result = estimate_marginal_cate_simple(
        ...     predictor, train_data, ['Z1', 'Z2', 'Z3'],
        ...     {'Z1': 1},  # Only Z1 is observed
        ...     n_mc_samples=1000
        ... )
    """
    # Create distribution
    covar_dist = EmpiricalCovariateDistribution(
        training_data, covariate_names, joint_sampling=False
    )

    # Create estimator
    estimator = MarginalCATEEstimator(
        predictor, covar_dist, n_mc_samples, random_state=seed
    )

    # Estimate
    result = estimator.estimate_marginal_cate(observed_values, return_samples=True)

    # Compute confidence interval
    samples = result.cate_samples
    ci_lower = float(np.percentile(samples, 2.5))
    ci_upper = float(np.percentile(samples, 97.5))

    return {
        'mean': result.mean_cate,
        'std': result.std_cate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': result.n_samples
    }


def analyze_covariate_importance(
    predictor: BasePredictorModel,
    training_data: pd.DataFrame,
    covariate_names: List[str],
    n_mc_samples: int = 500,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Analyze which covariates contribute most to CATE variation.

    For each covariate, estimates the variance in CATE explained by
    knowing vs not knowing that covariate.

    Args:
        predictor: Fitted predictor model
        training_data: Training data
        covariate_names: Names of all covariates
        n_mc_samples: Number of MC samples
        seed: Random seed

    Returns:
        DataFrame with covariate importance scores
    """
    if seed is not None:
        np.random.seed(seed)

    # Get baseline CATE predictions (full information)
    full_cate = predictor.predict_cate(training_data[covariate_names])
    full_variance = np.var(full_cate)

    results = []

    for name in covariate_names:
        # Compute marginal CATE without this covariate
        other_covariates = [n for n in covariate_names if n != name]

        # For each unique value of this covariate, compute average CATE
        marginal_cates = []
        for idx in range(len(training_data)):
            observed_vals = {n: training_data[n].iloc[idx] for n in other_covariates}

            covar_dist = EmpiricalCovariateDistribution(
                training_data, covariate_names, joint_sampling=False
            )
            estimator = MarginalCATEEstimator(
                predictor, covar_dist, n_mc_samples=100
            )
            result = estimator.estimate_marginal_cate(observed_vals)
            marginal_cates.append(result.mean_cate)

        marginal_variance = np.var(marginal_cates)

        # Importance = variance reduction from knowing this covariate
        # Higher = more important
        importance = full_variance - marginal_variance if full_variance > 0 else 0

        results.append({
            'covariate': name,
            'full_variance': full_variance,
            'marginal_variance': marginal_variance,
            'importance': importance,
            'importance_ratio': importance / full_variance if full_variance > 0 else 0
        })

    return pd.DataFrame(results).sort_values('importance', ascending=False)


# Module test
if __name__ == "__main__":
    print("Marginal Effects Module Test")
    print("=" * 50)

    # Create test data
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    X = np.random.binomial(1, 0.3 + 0.3 * Z1 + 0.2 * Z2, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z1, n)

    df = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'X': X, 'Y': Y})

    print("\n1. Fitting predictor...")
    predictor = EmpiricalPredictor()
    predictor.fit(df, 'X', 'Y', ['Z1', 'Z2'])

    print("\n2. Testing EmpiricalCovariateDistribution...")
    covar_dist = EmpiricalCovariateDistribution(df, ['Z1', 'Z2'])
    samples = covar_dist.sample(10, observed_values={'Z1': 1})
    print(f"   Sampled {len(samples)} covariate vectors with Z1=1 fixed")
    print(f"   Z1 values: {samples['Z1'].unique()}")

    print("\n3. Testing MarginalCATEEstimator...")
    estimator = MarginalCATEEstimator(predictor, covar_dist, n_mc_samples=500)

    # Estimate marginal CATE when only Z1 is observed
    result = estimator.estimate_marginal_cate({'Z1': 1}, return_samples=True)
    print(f"   Marginal CATE (Z1=1): {result.mean_cate:.4f} ± {result.std_cate:.4f}")

    result = estimator.estimate_marginal_cate({'Z1': 0}, return_samples=True)
    print(f"   Marginal CATE (Z1=0): {result.mean_cate:.4f} ± {result.std_cate:.4f}")

    print("\n4. Testing simple interface...")
    simple_result = estimate_marginal_cate_simple(
        predictor, df, ['Z1', 'Z2'],
        {'Z1': 1}, n_mc_samples=500, seed=42
    )
    print(f"   Result: mean={simple_result['mean']:.4f}, "
          f"95% CI=[{simple_result['ci_lower']:.4f}, {simple_result['ci_upper']:.4f}]")

    print("\n5. Comparing full vs marginal CATE...")
    full_cate = predictor.predict_cate(df[['Z1', 'Z2']])
    print(f"   Full CATE range: [{full_cate.min():.4f}, {full_cate.max():.4f}]")
    print(f"   Full CATE mean: {full_cate.mean():.4f}")

    print("\nAll tests completed!")
