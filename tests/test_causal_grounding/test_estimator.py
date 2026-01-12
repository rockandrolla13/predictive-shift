"""
Test CausalGroundingEstimator main class.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding.estimator import CausalGroundingEstimator


@pytest.fixture
def synthetic_training_data():
    """Create synthetic training data for multiple sites."""
    np.random.seed(42)
    sites = {}

    for site_id in ['site_A', 'site_B', 'site_C']:
        n = 400  # Larger dataset

        # Covariates
        resp_age = np.random.uniform(18, 65, size=n)
        resp_gender = np.random.choice([0, 1], size=n)
        resp_polideo = np.random.choice(range(7), size=n)

        # Treatment and outcome
        X = np.random.choice([0, 1], size=n)
        Y = 100 + 30 * X + 5 * resp_gender + np.random.normal(0, 20, size=n)

        # Regime indicator - ensure balanced split
        F = np.array(['on'] * (n // 2) + ['idle'] * (n // 2))
        np.random.shuffle(F)

        sites[site_id] = pd.DataFrame({
            'resp_age': resp_age,
            'resp_gender': resp_gender,
            'resp_polideo': resp_polideo,
            'iv': X,
            'dv': Y,
            'F': F
        })

    return sites


@pytest.fixture
def minimal_training_data():
    """Create minimal training data for fast tests."""
    np.random.seed(42)
    sites = {}

    for site_id in ['site_A', 'site_B']:
        n = 200  # Need enough for F=on/idle split

        # Ensure balanced F split
        F = np.array(['on'] * (n // 2) + ['idle'] * (n // 2))
        np.random.shuffle(F)

        sites[site_id] = pd.DataFrame({
            'resp_age': np.random.uniform(18, 65, size=n),
            'resp_gender': np.random.choice([0, 1], size=n),
            'resp_polideo': np.random.choice(range(7), size=n),
            'iv': np.random.choice([0, 1], size=n),
            'dv': np.random.normal(100, 20, size=n),
            'F': F
        })

    return sites


class TestEstimatorInitialization:
    """Test CausalGroundingEstimator initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        estimator = CausalGroundingEstimator()

        assert estimator.epsilon == 0.1
        assert estimator.transfer_method == 'conservative'
        assert estimator.ci_alpha == 0.05
        assert estimator.n_permutations == 500
        assert estimator.discretize is True
        assert estimator.age_bins == 5
        assert estimator.use_full_lp is False
        assert estimator.random_seed is None
        assert estimator.verbose is True

    def test_custom_initialization(self):
        """Test custom parameter values."""
        estimator = CausalGroundingEstimator(
            epsilon=0.2,
            transfer_method='average',
            ci_alpha=0.10,
            n_permutations=100,
            discretize=False,
            age_bins=3,
            use_full_lp=True,
            random_seed=123,
            verbose=False
        )

        assert estimator.epsilon == 0.2
        assert estimator.transfer_method == 'average'
        assert estimator.ci_alpha == 0.10
        assert estimator.n_permutations == 100
        assert estimator.discretize is False
        assert estimator.age_bins == 3
        assert estimator.use_full_lp is True
        assert estimator.random_seed == 123
        assert estimator.verbose is False

    def test_initial_fitted_state(self):
        """Test that estimator starts unfitted."""
        estimator = CausalGroundingEstimator()

        assert estimator.is_fitted_ is False
        assert estimator.covariates_ is None
        assert estimator.best_instrument_ is None
        assert estimator.covariate_scores_ is None
        assert estimator.training_bounds_ is None
        assert estimator.transferred_bounds_ is None
        assert estimator.ci_engine_ is None


class TestEstimatorFit:
    """Test CausalGroundingEstimator.fit() method."""

    def test_fit_returns_self(self, minimal_training_data):
        """Test that fit returns self for method chaining."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        result = estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert result is estimator

    def test_fit_sets_is_fitted(self, minimal_training_data):
        """Test that fit sets is_fitted_ to True."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.is_fitted_ is True

    def test_fit_sets_covariates(self, minimal_training_data):
        """Test that fit sets covariates_."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.covariates_ is not None
        assert len(estimator.covariates_) > 0

    def test_fit_with_custom_covariates(self, minimal_training_data):
        """Test fit with custom covariate list."""
        # Add pre-discretized covariates
        for site_df in minimal_training_data.values():
            site_df['Z1'] = np.random.choice([0, 1, 2], size=len(site_df))
            site_df['Z2'] = np.random.choice([0, 1], size=len(site_df))

        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            discretize=False,
            random_seed=42
        )

        estimator.fit(
            minimal_training_data,
            treatment='iv',
            outcome='dv',
            covariates=['Z1', 'Z2']
        )

        assert estimator.covariates_ == ['Z1', 'Z2']

    def test_fit_sets_training_bounds(self, minimal_training_data):
        """Test that fit computes training bounds."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.training_bounds_ is not None
        assert isinstance(estimator.training_bounds_, dict)

    def test_fit_sets_transferred_bounds(self, minimal_training_data):
        """Test that fit computes transferred bounds."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.transferred_bounds_ is not None
        assert isinstance(estimator.transferred_bounds_, dict)

    def test_fit_conservative_transfer(self, minimal_training_data):
        """Test fit with conservative transfer method."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            transfer_method='conservative',
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.transfer_method == 'conservative'
        assert estimator.transferred_bounds_ is not None

    def test_fit_average_transfer(self, minimal_training_data):
        """Test fit with average transfer method."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            transfer_method='average',
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.transfer_method == 'average'
        assert estimator.transferred_bounds_ is not None

    def test_fit_invalid_transfer_method(self, minimal_training_data):
        """Test that invalid transfer method raises error."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            transfer_method='invalid_method',
            verbose=False,
            random_seed=42
        )

        with pytest.raises(ValueError, match="Unknown transfer method"):
            estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

    def test_fit_sets_covariate_scores(self, minimal_training_data):
        """Test that fit computes covariate scores."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.covariate_scores_ is not None

    def test_fit_sets_best_instrument(self, minimal_training_data):
        """Test that fit selects best instrument."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        assert estimator.best_instrument_ is not None


class TestEstimatorPredictBounds:
    """Test CausalGroundingEstimator.predict_bounds() method."""

    @pytest.fixture
    def fitted_estimator(self, minimal_training_data):
        """Return a fitted estimator."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')
        return estimator

    def test_predict_bounds_returns_dataframe(self, fitted_estimator):
        """Test that predict_bounds returns DataFrame by default."""
        result = fitted_estimator.predict_bounds()

        assert isinstance(result, pd.DataFrame)

    def test_predict_bounds_returns_dict(self, fitted_estimator):
        """Test that predict_bounds can return dict."""
        result = fitted_estimator.predict_bounds(return_dataframe=False)

        assert isinstance(result, dict)

    def test_predict_bounds_dataframe_columns(self, fitted_estimator):
        """Test that DataFrame has expected columns."""
        result = fitted_estimator.predict_bounds()

        assert 'cate_lower' in result.columns
        assert 'cate_upper' in result.columns
        assert 'width' in result.columns

    def test_predict_bounds_with_z_values(self, fitted_estimator):
        """Test predict_bounds with specific z_values."""
        # Get available z values
        available_z = list(fitted_estimator.transferred_bounds_.keys())

        if len(available_z) >= 2:
            subset = available_z[:2]
            result = fitted_estimator.predict_bounds(z_values=subset)

            assert len(result) == 2

    def test_predict_bounds_unfitted_raises(self):
        """Test that predict_bounds on unfitted estimator raises error."""
        estimator = CausalGroundingEstimator()

        with pytest.raises(RuntimeError, match="Estimator not fitted"):
            estimator.predict_bounds()

    def test_predict_bounds_width_positive(self, fitted_estimator):
        """Test that all bound widths are positive."""
        result = fitted_estimator.predict_bounds()

        assert (result['width'] >= 0).all()

    def test_predict_bounds_lower_less_than_upper(self, fitted_estimator):
        """Test that lower bounds are less than upper bounds."""
        result = fitted_estimator.predict_bounds()

        assert (result['cate_lower'] <= result['cate_upper']).all()


class TestEstimatorGetDiagnostics:
    """Test CausalGroundingEstimator.get_diagnostics() method."""

    @pytest.fixture
    def fitted_estimator(self, minimal_training_data):
        """Return a fitted estimator."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')
        return estimator

    def test_get_diagnostics_returns_dict(self, fitted_estimator):
        """Test that get_diagnostics returns dict."""
        result = fitted_estimator.get_diagnostics()

        assert isinstance(result, dict)

    def test_get_diagnostics_contains_expected_keys(self, fitted_estimator):
        """Test that diagnostics contains expected keys."""
        result = fitted_estimator.get_diagnostics()

        assert 'n_training_sites' in result
        assert 'n_z_values' in result
        assert 'best_instrument' in result
        assert 'epsilon' in result
        assert 'transfer_method' in result
        assert 'mean_width' in result

    def test_get_diagnostics_unfitted_raises(self):
        """Test that get_diagnostics on unfitted estimator raises error."""
        estimator = CausalGroundingEstimator()

        with pytest.raises(RuntimeError, match="Estimator not fitted"):
            estimator.get_diagnostics()

    def test_get_diagnostics_epsilon_matches(self, minimal_training_data):
        """Test that diagnostics epsilon matches initialization."""
        estimator = CausalGroundingEstimator(
            epsilon=0.15,
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        diagnostics = estimator.get_diagnostics()

        assert diagnostics['epsilon'] == 0.15

    def test_get_diagnostics_transfer_method_matches(self, minimal_training_data):
        """Test that diagnostics transfer_method matches initialization."""
        estimator = CausalGroundingEstimator(
            transfer_method='average',
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        diagnostics = estimator.get_diagnostics()

        assert diagnostics['transfer_method'] == 'average'


class TestEstimatorCheckIsFitted:
    """Test CausalGroundingEstimator._check_is_fitted() method."""

    def test_check_is_fitted_raises_when_not_fitted(self):
        """Test that _check_is_fitted raises when not fitted."""
        estimator = CausalGroundingEstimator()

        with pytest.raises(RuntimeError, match="Estimator not fitted"):
            estimator._check_is_fitted()

    def test_check_is_fitted_passes_when_fitted(self, minimal_training_data):
        """Test that _check_is_fitted passes when fitted."""
        estimator = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator.fit(minimal_training_data, treatment='iv', outcome='dv')

        # Should not raise
        estimator._check_is_fitted()


class TestEstimatorIntegration:
    """Integration tests for full estimator pipeline."""

    def test_full_pipeline(self, synthetic_training_data):
        """Test complete fit -> predict pipeline."""
        estimator = CausalGroundingEstimator(
            epsilon=0.1,
            n_permutations=50,
            verbose=False,
            random_seed=42
        )

        # Fit
        estimator.fit(synthetic_training_data, treatment='iv', outcome='dv')

        # Predict
        bounds_df = estimator.predict_bounds()

        # Diagnostics
        diagnostics = estimator.get_diagnostics()

        # Assertions
        assert estimator.is_fitted_
        assert len(bounds_df) > 0
        assert diagnostics['n_training_sites'] > 0

    def test_reproducibility_with_seed(self, minimal_training_data):
        """Test that results are reproducible with same seed."""
        estimator1 = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator1.fit(minimal_training_data, treatment='iv', outcome='dv')

        estimator2 = CausalGroundingEstimator(
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator2.fit(minimal_training_data, treatment='iv', outcome='dv')

        bounds1 = estimator1.predict_bounds(return_dataframe=False)
        bounds2 = estimator2.predict_bounds(return_dataframe=False)

        # Same z values should have same bounds
        common_z = set(bounds1.keys()) & set(bounds2.keys())
        for z in common_z:
            assert bounds1[z] == bounds2[z]

    def test_different_epsilon_different_widths(self, minimal_training_data):
        """Test that different epsilon gives different bound widths."""
        estimator_small = CausalGroundingEstimator(
            epsilon=0.05,
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator_small.fit(minimal_training_data, treatment='iv', outcome='dv')

        estimator_large = CausalGroundingEstimator(
            epsilon=0.2,
            n_permutations=50,
            verbose=False,
            random_seed=42
        )
        estimator_large.fit(minimal_training_data, treatment='iv', outcome='dv')

        diag_small = estimator_small.get_diagnostics()
        diag_large = estimator_large.get_diagnostics()

        # Larger epsilon should give wider bounds
        assert diag_large['mean_width'] > diag_small['mean_width']

    def test_method_chaining(self, minimal_training_data):
        """Test that method chaining works."""
        bounds = (
            CausalGroundingEstimator(n_permutations=50, verbose=False, random_seed=42)
            .fit(minimal_training_data, treatment='iv', outcome='dv')
            .predict_bounds()
        )

        assert isinstance(bounds, pd.DataFrame)
        assert len(bounds) > 0
