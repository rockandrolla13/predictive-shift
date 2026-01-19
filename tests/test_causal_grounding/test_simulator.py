"""
Tests for Synthetic Data Simulator

Tests the simulator.py module for generating synthetic data
with known ground truth CATE.
"""

import pytest
import numpy as np
import pandas as pd

from causal_grounding.simulator import (
    BinarySyntheticDGP,
    generate_random_dgp,
    simulate_observational,
    simulate_rct,
    compute_true_cate,
    compute_true_ate,
    generate_multi_environment_data,
    add_regime_indicator,
    get_covariate_stratum,
    logistic,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_dgp():
    """Simple DGP with 3 covariates."""
    return generate_random_dgp(n_covariates=3, seed=42)


@pytest.fixture
def sparse_dgp():
    """DGP with sparsity (some covariates have zero effect)."""
    return generate_random_dgp(n_covariates=5, sparsity=0.4, seed=42)


@pytest.fixture
def observational_data(simple_dgp):
    """Sample observational data."""
    return simulate_observational(simple_dgp, n_samples=1000, seed=42)


@pytest.fixture
def rct_data(simple_dgp):
    """Sample RCT data."""
    return simulate_rct(simple_dgp, n_samples=1000, seed=42)


# =============================================================================
# TEST: BinarySyntheticDGP
# =============================================================================

class TestBinarySyntheticDGP:
    """Tests for BinarySyntheticDGP dataclass."""

    def test_init(self):
        """Test DGP initialization."""
        dgp = BinarySyntheticDGP(
            n_covariates=3,
            prob_x=np.array([0.3, 0.5, 0.7]),
            coeff_a=np.array([0.1, 0.2, -0.1, 0.3]),
            coeff_y0=np.array([0.0, 0.1, 0.2, 0.3]),
            coeff_y1=np.array([0.5, 0.1, 0.2, 0.3])
        )
        assert dgp.n_covariates == 3
        assert len(dgp.prob_x) == 3
        assert len(dgp.coeff_a) == 4  # n_covariates + 1 (intercept)

    def test_default_names(self):
        """Test default covariate names."""
        dgp = BinarySyntheticDGP(
            n_covariates=3,
            prob_x=np.zeros(3),
            coeff_a=np.zeros(4),
            coeff_y0=np.zeros(4),
            coeff_y1=np.zeros(4)
        )
        assert dgp.x_names == ['X0', 'X1', 'X2']
        assert dgp.a_name == 'A'
        assert dgp.y_name == 'Y'

    def test_get_propensity_params(self, simple_dgp):
        """Test propensity parameter extraction."""
        params = simple_dgp.get_propensity_params()
        assert 'intercept' in params
        assert 'coefficients' in params
        assert len(params['coefficients']) == simple_dgp.n_covariates

    def test_get_outcome_params(self, simple_dgp):
        """Test outcome parameter extraction."""
        params0 = simple_dgp.get_outcome_params(a=0)
        params1 = simple_dgp.get_outcome_params(a=1)

        assert params0['treatment'] == 0
        assert params1['treatment'] == 1
        assert 'intercept' in params0
        assert 'coefficients' in params0


# =============================================================================
# TEST: generate_random_dgp
# =============================================================================

class TestGenerateRandomDGP:
    """Tests for generate_random_dgp function."""

    def test_basic_generation(self):
        """Test basic DGP generation."""
        dgp = generate_random_dgp(n_covariates=5)
        assert dgp.n_covariates == 5
        assert len(dgp.prob_x) == 5
        assert len(dgp.coeff_a) == 6
        assert len(dgp.coeff_y0) == 6
        assert len(dgp.coeff_y1) == 6

    def test_reproducibility(self):
        """Test that same seed produces same DGP."""
        dgp1 = generate_random_dgp(n_covariates=5, seed=42)
        dgp2 = generate_random_dgp(n_covariates=5, seed=42)

        np.testing.assert_array_equal(dgp1.prob_x, dgp2.prob_x)
        np.testing.assert_array_equal(dgp1.coeff_a, dgp2.coeff_a)

    def test_different_seeds(self):
        """Test that different seeds produce different DGPs."""
        dgp1 = generate_random_dgp(n_covariates=5, seed=42)
        dgp2 = generate_random_dgp(n_covariates=5, seed=43)

        assert not np.allclose(dgp1.prob_x, dgp2.prob_x)

    def test_sparsity_pattern(self):
        """Test that sparsity sets some coefficients to zero."""
        dgp = generate_random_dgp(n_covariates=10, sparsity=0.5, seed=42)

        # At least some coefficients should be zero (excluding intercept)
        n_zero_y0 = np.sum(dgp.coeff_y0[1:] == 0)
        n_zero_y1 = np.sum(dgp.coeff_y1[1:] == 0)

        assert n_zero_y0 >= 4  # ~50% sparsity
        assert n_zero_y1 >= 4

    def test_prob_x_bounds(self):
        """Test that marginal probabilities are in valid range."""
        dgp = generate_random_dgp(n_covariates=10, seed=42)
        assert np.all(dgp.prob_x >= 0.2)
        assert np.all(dgp.prob_x <= 0.8)


# =============================================================================
# TEST: simulate_observational
# =============================================================================

class TestSimulateObservational:
    """Tests for simulate_observational function."""

    def test_output_shape(self, simple_dgp):
        """Test output DataFrame shape."""
        data = simulate_observational(simple_dgp, n_samples=100)
        assert len(data) == 100
        assert len(data.columns) == 5  # 3 X + A + Y

    def test_column_names(self, simple_dgp):
        """Test output column names."""
        data = simulate_observational(simple_dgp, n_samples=100)
        assert all(x in data.columns for x in simple_dgp.x_names)
        assert simple_dgp.a_name in data.columns
        assert simple_dgp.y_name in data.columns

    def test_binary_values(self, simple_dgp):
        """Test that X, A, Y are binary."""
        data = simulate_observational(simple_dgp, n_samples=1000, seed=42)

        for col in simple_dgp.x_names + [simple_dgp.a_name, simple_dgp.y_name]:
            assert set(data[col].unique()).issubset({0, 1})

    def test_reproducibility(self, simple_dgp):
        """Test reproducibility with same seed."""
        data1 = simulate_observational(simple_dgp, 100, seed=42)
        data2 = simulate_observational(simple_dgp, 100, seed=42)

        pd.testing.assert_frame_equal(data1, data2)

    def test_confounding_exists(self, simple_dgp):
        """Test that treatment is confounded by covariates."""
        data = simulate_observational(simple_dgp, 10000, seed=42)

        # Check correlation between X and A
        X = data[simple_dgp.x_names].values
        A = data[simple_dgp.a_name].values

        # At least some covariate should be correlated with treatment
        correlations = [np.corrcoef(X[:, i], A)[0, 1] for i in range(X.shape[1])]
        assert np.max(np.abs(correlations)) > 0.01  # Some non-zero correlation


# =============================================================================
# TEST: simulate_rct
# =============================================================================

class TestSimulateRCT:
    """Tests for simulate_rct function."""

    def test_output_shape(self, simple_dgp):
        """Test output DataFrame shape."""
        data = simulate_rct(simple_dgp, n_samples=100)
        assert len(data) == 100

    def test_treatment_rate(self, simple_dgp):
        """Test that treatment rate matches specified probability."""
        data = simulate_rct(simple_dgp, 10000, treatment_prob=0.3, seed=42)
        actual_rate = data[simple_dgp.a_name].mean()
        assert abs(actual_rate - 0.3) < 0.02  # Within 2%

    def test_no_confounding(self, simple_dgp):
        """Test that treatment is independent of covariates in RCT."""
        data = simulate_rct(simple_dgp, 10000, seed=42)

        X = data[simple_dgp.x_names].values
        A = data[simple_dgp.a_name].values

        # Correlations should be near zero
        correlations = [np.corrcoef(X[:, i], A)[0, 1] for i in range(X.shape[1])]
        assert np.max(np.abs(correlations)) < 0.05  # Near-zero correlation

    def test_custom_treatment_prob(self, simple_dgp):
        """Test with non-standard treatment probability."""
        data = simulate_rct(simple_dgp, 10000, treatment_prob=0.7, seed=42)
        actual_rate = data[simple_dgp.a_name].mean()
        assert abs(actual_rate - 0.7) < 0.02


# =============================================================================
# TEST: compute_true_cate
# =============================================================================

class TestComputeTrueCate:
    """Tests for compute_true_cate function."""

    def test_output_shape(self, simple_dgp, observational_data):
        """Test output shape matches input."""
        cate = compute_true_cate(simple_dgp, observational_data)
        assert len(cate) == len(observational_data)

    def test_cate_range(self, simple_dgp, observational_data):
        """Test CATE is in valid probability range."""
        cate = compute_true_cate(simple_dgp, observational_data)
        # CATE = P(Y=1|A=1,X) - P(Y=1|A=0,X), should be in [-1, 1]
        assert np.all(cate >= -1)
        assert np.all(cate <= 1)

    def test_accepts_array_input(self, simple_dgp):
        """Test that function accepts numpy array."""
        X = np.random.binomial(1, 0.5, (100, simple_dgp.n_covariates))
        cate = compute_true_cate(simple_dgp, X)
        assert len(cate) == 100

    def test_accepts_dataframe_input(self, simple_dgp, observational_data):
        """Test that function accepts DataFrame."""
        cate = compute_true_cate(simple_dgp, observational_data)
        assert isinstance(cate, np.ndarray)

    def test_deterministic(self, simple_dgp, observational_data):
        """Test CATE computation is deterministic."""
        cate1 = compute_true_cate(simple_dgp, observational_data)
        cate2 = compute_true_cate(simple_dgp, observational_data)
        np.testing.assert_array_equal(cate1, cate2)


# =============================================================================
# TEST: compute_true_ate
# =============================================================================

class TestComputeTrueATE:
    """Tests for compute_true_ate function."""

    def test_output_type(self, simple_dgp):
        """Test output is float."""
        ate = compute_true_ate(simple_dgp, n_samples=1000)
        assert isinstance(ate, float)

    def test_ate_range(self, simple_dgp):
        """Test ATE is in valid range."""
        ate = compute_true_ate(simple_dgp, n_samples=10000)
        assert -1 <= ate <= 1

    def test_convergence(self, simple_dgp):
        """Test ATE converges with more samples."""
        ate_small = compute_true_ate(simple_dgp, n_samples=100)
        ate_large = compute_true_ate(simple_dgp, n_samples=100000)

        # Both should be finite
        assert np.isfinite(ate_small)
        assert np.isfinite(ate_large)


# =============================================================================
# TEST: generate_multi_environment_data
# =============================================================================

class TestGenerateMultiEnvironmentData:
    """Tests for generate_multi_environment_data function."""

    def test_number_of_environments(self, simple_dgp):
        """Test correct number of environments created."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=3, n_per_env=100, include_rct=True
        )
        assert len(envs) == 4  # 3 obs + 1 rct

    def test_no_rct_option(self, simple_dgp):
        """Test excluding RCT environment."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=3, n_per_env=100, include_rct=False
        )
        assert len(envs) == 3
        assert 'rct' not in envs

    def test_samples_per_env(self, simple_dgp):
        """Test correct samples per environment."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=2, n_per_env=500
        )
        for name, df in envs.items():
            assert len(df) == 500

    def test_environment_column(self, simple_dgp):
        """Test environment identifier column added."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=2, n_per_env=100
        )
        for name, df in envs.items():
            assert 'environment' in df.columns
            assert df['environment'].iloc[0] == name

    def test_heterogeneity(self, simple_dgp):
        """Test that heterogeneity creates variation in treatment rates."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=5, n_per_env=1000,
            heterogeneity=0.5, include_rct=False, seed=42
        )

        treatment_rates = [df['A'].mean() for df in envs.values()]

        # With heterogeneity, treatment rates should vary
        assert np.std(treatment_rates) > 0.01

    def test_reproducibility(self, simple_dgp):
        """Test reproducibility with seed."""
        envs1 = generate_multi_environment_data(
            simple_dgp, n_environments=2, n_per_env=100, seed=42
        )
        envs2 = generate_multi_environment_data(
            simple_dgp, n_environments=2, n_per_env=100, seed=42
        )

        for name in envs1:
            pd.testing.assert_frame_equal(envs1[name], envs2[name])


# =============================================================================
# TEST: add_regime_indicator
# =============================================================================

class TestAddRegimeIndicator:
    """Tests for add_regime_indicator function."""

    def test_adds_column(self, simple_dgp):
        """Test regime column is added."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=1, n_per_env=100
        )
        envs = add_regime_indicator(envs)

        for df in envs.values():
            assert 'F' in df.columns

    def test_rct_value(self, simple_dgp):
        """Test RCT environments get correct value."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=1, n_per_env=100, include_rct=True
        )
        envs = add_regime_indicator(envs, rct_value='idle', obs_value='on')

        assert all(envs['rct']['F'] == 'idle')
        assert all(envs['obs_0']['F'] == 'on')

    def test_custom_column_name(self, simple_dgp):
        """Test custom regime column name."""
        envs = generate_multi_environment_data(
            simple_dgp, n_environments=1, n_per_env=100
        )
        envs = add_regime_indicator(envs, regime_col='regime')

        for df in envs.values():
            assert 'regime' in df.columns


# =============================================================================
# TEST: get_covariate_stratum
# =============================================================================

class TestGetCovariateStratum:
    """Tests for get_covariate_stratum function."""

    def test_output_type(self, observational_data, simple_dgp):
        """Test output is integer array."""
        strata = get_covariate_stratum(observational_data, simple_dgp.x_names)
        assert strata.dtype in [np.int32, np.int64]

    def test_same_values_same_stratum(self):
        """Test same covariate values get same stratum."""
        df = pd.DataFrame({
            'X0': [0, 0, 1, 1],
            'X1': [0, 0, 0, 1]
        })
        strata = get_covariate_stratum(df, ['X0', 'X1'])

        # First two rows should have same stratum
        assert strata[0] == strata[1]
        # Different values should have different strata
        assert len(np.unique(strata)) == 3


# =============================================================================
# TEST: logistic function
# =============================================================================

class TestLogistic:
    """Tests for logistic function."""

    def test_at_zero(self):
        """Test logistic(0) = 0.5."""
        assert logistic(np.array([0.0]))[0] == 0.5

    def test_bounds(self):
        """Test output is in (0, 1)."""
        x = np.array([-10, -1, 0, 1, 10])
        y = logistic(x)
        assert np.all(y > 0)
        assert np.all(y < 1)

    def test_monotonic(self):
        """Test function is monotonically increasing."""
        x = np.linspace(-5, 5, 100)
        y = logistic(x)
        assert np.all(np.diff(y) > 0)

    def test_numerical_stability(self):
        """Test with extreme values."""
        x = np.array([-1000, 1000])
        y = logistic(x)
        assert np.isfinite(y).all()
        assert y[0] < 0.01
        assert y[1] > 0.99


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for simulator module."""

    def test_full_pipeline(self):
        """Test complete simulation and evaluation pipeline."""
        # Create DGP
        dgp = generate_random_dgp(n_covariates=5, sparsity=0.2, seed=42)

        # Generate data
        obs_data = simulate_observational(dgp, 1000, seed=42)
        rct_data = simulate_rct(dgp, 1000, seed=42)

        # Compute ground truth
        cate_obs = compute_true_cate(dgp, obs_data)
        cate_rct = compute_true_cate(dgp, rct_data)
        ate = compute_true_ate(dgp)

        # Validate
        assert len(cate_obs) == len(obs_data)
        assert len(cate_rct) == len(rct_data)
        assert -1 <= ate <= 1

        # Check ATE is close to mean CATE
        assert abs(np.mean(cate_obs) - ate) < 0.1
        assert abs(np.mean(cate_rct) - ate) < 0.1

    def test_multi_environment_pipeline(self):
        """Test multi-environment data generation pipeline."""
        dgp = generate_random_dgp(n_covariates=4, seed=42)

        envs = generate_multi_environment_data(
            dgp, n_environments=3, n_per_env=500,
            heterogeneity=0.2, include_rct=True, seed=42
        )
        envs = add_regime_indicator(envs)

        # Validate structure
        assert len(envs) == 4
        assert 'rct' in envs

        # Validate regime indicator
        assert envs['rct']['F'].iloc[0] == 'idle'
        assert envs['obs_0']['F'].iloc[0] == 'on'

        # Validate ground truth can be computed
        for name, df in envs.items():
            cate = compute_true_cate(dgp, df)
            assert len(cate) == len(df)
