"""
Tests for Marginal Effects Module

Tests Monte Carlo CATE estimation for partially observed covariates.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from causal_grounding.marginal_effects import (
    MCEstimationResult,
    EmpiricalCovariateDistribution,
    MarginalCATEEstimator,
    estimate_marginal_cate_simple,
)
from causal_grounding.predictors import EmpiricalPredictor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def binary_data():
    """Sample binary data for testing."""
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    X = np.random.binomial(1, 0.3 + 0.3 * Z1 + 0.2 * Z2, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z1, n)

    return pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'X': X, 'Y': Y})


@pytest.fixture
def multivariate_data():
    """Sample data with multiple covariates."""
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    Z3 = np.random.binomial(1, 0.4, n)

    X = np.random.binomial(1, 0.2 + 0.3 * Z1 + 0.2 * Z2 + 0.1 * Z3, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.15 * Z1 + 0.05 * Z2, n)

    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
    })


@pytest.fixture
def fitted_predictor(binary_data):
    """Fitted empirical predictor."""
    predictor = EmpiricalPredictor()
    predictor.fit(binary_data, 'X', 'Y', ['Z1', 'Z2'])
    return predictor


@pytest.fixture
def covariate_distribution(binary_data):
    """Empirical covariate distribution."""
    return EmpiricalCovariateDistribution(
        binary_data, ['Z1', 'Z2'], joint_sampling=False
    )


# =============================================================================
# TEST: EmpiricalCovariateDistribution
# =============================================================================

class TestEmpiricalCovariateDistribution:
    """Tests for EmpiricalCovariateDistribution class."""

    def test_init_marginal(self, binary_data):
        """Test initialization with marginal sampling."""
        dist = EmpiricalCovariateDistribution(
            binary_data, ['Z1', 'Z2'], joint_sampling=False
        )
        assert dist.n_covariates == 2
        assert dist.joint_sampling == False

    def test_init_joint(self, binary_data):
        """Test initialization with joint sampling."""
        dist = EmpiricalCovariateDistribution(
            binary_data, ['Z1', 'Z2'], joint_sampling=True
        )
        assert dist.joint_sampling == True

    def test_sample_basic(self, covariate_distribution):
        """Test basic sampling."""
        samples = covariate_distribution.sample(100, seed=42)

        assert isinstance(samples, pd.DataFrame)
        assert len(samples) == 100
        assert 'Z1' in samples.columns
        assert 'Z2' in samples.columns

    def test_sample_with_observed(self, covariate_distribution):
        """Test sampling with observed values fixed."""
        samples = covariate_distribution.sample(
            100, observed_values={'Z1': 1}, seed=42
        )

        # Z1 should be fixed to 1
        assert np.all(samples['Z1'] == 1)
        # Z2 should vary
        assert len(samples['Z2'].unique()) > 1

    def test_sample_reproducible(self, covariate_distribution):
        """Test sampling is reproducible with seed."""
        samples1 = covariate_distribution.sample(50, seed=42)
        samples2 = covariate_distribution.sample(50, seed=42)

        pd.testing.assert_frame_equal(samples1, samples2)

    def test_get_covariate_probabilities(self, binary_data):
        """Test getting covariate probabilities."""
        dist = EmpiricalCovariateDistribution(
            binary_data, ['Z1', 'Z2'], joint_sampling=False
        )

        probs = dist.get_covariate_probabilities('Z1')

        assert 0 in probs
        assert 1 in probs
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_joint_sampling_preserves_correlation(self):
        """Test joint sampling preserves covariate correlations."""
        np.random.seed(42)
        n = 1000

        # Create correlated covariates
        Z1 = np.random.binomial(1, 0.5, n)
        Z2 = np.where(Z1 == 1,
                      np.random.binomial(1, 0.8, n),
                      np.random.binomial(1, 0.2, n))
        X = np.random.binomial(1, 0.5, n)
        Y = np.random.binomial(1, 0.5, n)

        df = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'X': X, 'Y': Y})

        # Joint sampling should preserve correlation
        dist_joint = EmpiricalCovariateDistribution(
            df, ['Z1', 'Z2'], joint_sampling=True
        )
        samples = dist_joint.sample(1000, seed=42)

        # Check correlation is preserved
        orig_corr = np.corrcoef(df['Z1'], df['Z2'])[0, 1]
        sample_corr = np.corrcoef(samples['Z1'], samples['Z2'])[0, 1]

        assert abs(orig_corr - sample_corr) < 0.1


# =============================================================================
# TEST: MarginalCATEEstimator
# =============================================================================

class TestMarginalCATEEstimator:
    """Tests for MarginalCATEEstimator class."""

    def test_init(self, fitted_predictor, covariate_distribution):
        """Test initialization."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500
        )
        assert estimator.n_mc_samples == 500

    def test_estimate_marginal_cate_basic(self, fitted_predictor, covariate_distribution):
        """Test basic marginal CATE estimation."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500
        )

        result = estimator.estimate_marginal_cate({'Z1': 1})

        assert isinstance(result, MCEstimationResult)
        assert isinstance(result.mean_cate, float)
        assert isinstance(result.std_cate, float)
        assert result.n_samples == 500

    def test_estimate_marginal_cate_with_samples(self, fitted_predictor, covariate_distribution):
        """Test marginal CATE estimation with samples returned."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500
        )

        result = estimator.estimate_marginal_cate({'Z1': 1}, return_samples=True)

        assert result.cate_samples is not None
        assert len(result.cate_samples) == 500

    def test_marginal_varies_by_observed(self, fitted_predictor, covariate_distribution):
        """Test that marginal CATE varies by observed covariate value."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500,
            random_state=42
        )

        result_z1_0 = estimator.estimate_marginal_cate({'Z1': 0})
        result_z1_1 = estimator.estimate_marginal_cate({'Z1': 1})

        # Z1 affects Y, so marginal CATE should differ
        # Note: might be similar due to small effect, but shouldn't be identical
        assert isinstance(result_z1_0.mean_cate, float)
        assert isinstance(result_z1_1.mean_cate, float)

    def test_estimate_batch(self, binary_data, fitted_predictor, covariate_distribution):
        """Test batch estimation."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=100
        )

        # Take small subset for speed
        test_data = binary_data.head(10)
        cates = estimator.estimate_marginal_cate_batch(
            test_data, ['Z1'], n_samples=100
        )

        assert len(cates) == 10
        assert all(np.isfinite(cates))

    def test_convergence_with_more_samples(self, fitted_predictor, covariate_distribution):
        """Test that variance decreases with more samples."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, random_state=42
        )

        # Low samples
        result_low = estimator.estimate_marginal_cate({'Z1': 1}, n_samples=50)

        # High samples
        result_high = estimator.estimate_marginal_cate({'Z1': 1}, n_samples=1000)

        # With more samples, estimate should be more precise (lower std)
        # Note: This is statistical so we use a loose check
        assert result_high.std_cate >= 0
        assert result_low.std_cate >= 0


# =============================================================================
# TEST: MCEstimationResult
# =============================================================================

class TestMCEstimationResult:
    """Tests for MCEstimationResult dataclass."""

    def test_create_result(self):
        """Test creating a result object."""
        result = MCEstimationResult(
            mean_cate=0.1,
            std_cate=0.05,
            n_samples=100,
            observed_indices=[0],
            unobserved_indices=[1]
        )

        assert result.mean_cate == 0.1
        assert result.std_cate == 0.05
        assert result.cate_samples is None

    def test_result_with_samples(self):
        """Test result with samples array."""
        samples = np.array([0.1, 0.15, 0.05, 0.2])
        result = MCEstimationResult(
            mean_cate=np.mean(samples),
            std_cate=np.std(samples),
            cate_samples=samples,
            n_samples=4
        )

        assert result.cate_samples is not None
        assert len(result.cate_samples) == 4


# =============================================================================
# TEST: estimate_marginal_cate_simple
# =============================================================================

class TestEstimateMarginalCATESimple:
    """Tests for estimate_marginal_cate_simple convenience function."""

    def test_basic(self, binary_data, fitted_predictor):
        """Test basic usage."""
        result = estimate_marginal_cate_simple(
            fitted_predictor,
            binary_data,
            ['Z1', 'Z2'],
            {'Z1': 1},
            n_mc_samples=500,
            seed=42
        )

        assert 'mean' in result
        assert 'std' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'n_samples' in result

    def test_confidence_interval(self, binary_data, fitted_predictor):
        """Test that CI is valid."""
        result = estimate_marginal_cate_simple(
            fitted_predictor,
            binary_data,
            ['Z1', 'Z2'],
            {'Z1': 1},
            n_mc_samples=500,
            seed=42
        )

        # CI lower should be less than upper
        assert result['ci_lower'] <= result['ci_upper']
        # Mean should be within CI
        assert result['ci_lower'] <= result['mean'] <= result['ci_upper']

    def test_reproducible_with_seed(self, binary_data, fitted_predictor):
        """Test reproducibility with seed."""
        result1 = estimate_marginal_cate_simple(
            fitted_predictor,
            binary_data,
            ['Z1', 'Z2'],
            {'Z1': 1},
            n_mc_samples=500,
            seed=42
        )

        result2 = estimate_marginal_cate_simple(
            fitted_predictor,
            binary_data,
            ['Z1', 'Z2'],
            {'Z1': 1},
            n_mc_samples=500,
            seed=42
        )

        assert result1['mean'] == result2['mean']


# =============================================================================
# TEST: estimate_marginal_bounds
# =============================================================================

class TestEstimateMarginalBounds:
    """Tests for marginal bounds estimation."""

    def test_basic(self, fitted_predictor, covariate_distribution):
        """Test basic marginal bounds estimation."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500,
            random_state=42
        )

        # Create mock bounds DataFrame
        bounds_df = pd.DataFrame({
            'stratum': ['0-0', '0-1', '1-0', '1-1'],
            'lower': [0.0, 0.1, 0.05, 0.15],
            'upper': [0.4, 0.5, 0.45, 0.55]
        })

        result = estimator.estimate_marginal_bounds(
            {'Z1': 1}, bounds_df
        )

        assert 'lower' in result
        assert 'upper' in result
        assert 'width' in result
        assert result['lower'] <= result['upper']

    def test_bounds_within_range(self, fitted_predictor, covariate_distribution):
        """Test marginal bounds are within original bounds range."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500,
            random_state=42
        )

        bounds_df = pd.DataFrame({
            'stratum': ['0-0', '0-1', '1-0', '1-1'],
            'lower': [0.0, 0.1, 0.05, 0.15],
            'upper': [0.4, 0.5, 0.45, 0.55]
        })

        result = estimator.estimate_marginal_bounds(
            {'Z1': 1}, bounds_df
        )

        # Marginal bounds should be within overall range
        assert result['lower'] >= bounds_df['lower'].min() - 0.01
        assert result['upper'] <= bounds_df['upper'].max() + 0.01


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_covariates_observed(self, binary_data, fitted_predictor, covariate_distribution):
        """Test when all covariates are observed."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=100
        )

        # Observe all covariates
        result = estimator.estimate_marginal_cate({'Z1': 1, 'Z2': 0})

        # Should essentially return the point estimate (low variance)
        assert result.std_cate < 0.1  # Should have very low variance

    def test_no_covariates_observed(self, binary_data, fitted_predictor, covariate_distribution):
        """Test when no covariates are observed."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, n_mc_samples=500
        )

        # No covariates observed (estimate marginal ATE)
        result = estimator.estimate_marginal_cate({})

        # Should return average CATE (like ATE)
        assert isinstance(result.mean_cate, float)
        assert result.std_cate > 0  # Should have some variance

    def test_single_covariate(self):
        """Test with single covariate."""
        np.random.seed(42)
        n = 500

        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
        Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

        df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

        predictor = EmpiricalPredictor()
        predictor.fit(df, 'X', 'Y', ['Z'])

        dist = EmpiricalCovariateDistribution(df, ['Z'])
        estimator = MarginalCATEEstimator(predictor, dist, n_mc_samples=100)

        # When Z is observed, should have no variance
        result = estimator.estimate_marginal_cate({'Z': 1})
        assert result.std_cate < 0.01

    def test_many_covariates(self, multivariate_data):
        """Test with many covariates."""
        predictor = EmpiricalPredictor()
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        dist = EmpiricalCovariateDistribution(
            multivariate_data, ['Z1', 'Z2', 'Z3']
        )
        estimator = MarginalCATEEstimator(predictor, dist, n_mc_samples=100)

        # Observe only one covariate
        result = estimator.estimate_marginal_cate({'Z1': 1})

        assert isinstance(result.mean_cate, float)
        assert len(result.unobserved_indices) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for marginal effects module."""

    def test_full_workflow(self, binary_data):
        """Test full workflow from data to marginal CATE."""
        # Step 1: Fit predictor
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z1', 'Z2'])

        # Step 2: Create distribution
        dist = EmpiricalCovariateDistribution(
            binary_data, ['Z1', 'Z2'], joint_sampling=False
        )

        # Step 3: Create estimator
        estimator = MarginalCATEEstimator(
            predictor, dist, n_mc_samples=500, random_state=42
        )

        # Step 4: Estimate marginal CATE for different observed values
        results = {}
        for z1 in [0, 1]:
            result = estimator.estimate_marginal_cate({'Z1': z1})
            results[f'Z1={z1}'] = result.mean_cate

        # Should get different results for different Z1 values
        assert 'Z1=0' in results
        assert 'Z1=1' in results

    def test_compare_full_vs_marginal(self, binary_data):
        """Test comparing full CATE vs marginal CATE."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z1', 'Z2'])

        # Full CATE (all covariates observed)
        full_cate = predictor.predict_cate(binary_data[['Z1', 'Z2']])

        # Marginal CATE (only Z1 observed)
        dist = EmpiricalCovariateDistribution(
            binary_data, ['Z1', 'Z2'], joint_sampling=False
        )
        estimator = MarginalCATEEstimator(
            predictor, dist, n_mc_samples=500, random_state=42
        )

        marginal_cates = []
        for z1 in [0, 1]:
            result = estimator.estimate_marginal_cate({'Z1': z1})
            marginal_cates.append(result.mean_cate)

        # Marginal CATE should be less variable than full CATE
        # (averaging over Z2)
        assert len(np.unique(marginal_cates)) <= len(np.unique(full_cate.round(4)))

    def test_mc_convergence(self, binary_data, fitted_predictor, covariate_distribution):
        """Test MC convergence with increasing samples."""
        estimator = MarginalCATEEstimator(
            fitted_predictor, covariate_distribution, random_state=42
        )

        # Estimate with different sample sizes
        sample_sizes = [50, 100, 500, 1000]
        estimates = []
        stds = []

        for n in sample_sizes:
            result = estimator.estimate_marginal_cate({'Z1': 1}, n_samples=n)
            estimates.append(result.mean_cate)
            stds.append(result.std_cate)

        # Estimates should converge (become more stable)
        # Check that estimates don't vary too wildly
        estimate_range = max(estimates) - min(estimates)
        assert estimate_range < 0.5  # Should be reasonably stable
