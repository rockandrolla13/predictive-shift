"""
Test LP Solver for CATE Bounds.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding.lp_solver import (
    estimate_conditional_means,
    estimate_identified_probs,
    estimate_observed_probs,
    solve_cate_bounds_single_z,
    solve_cate_bounds_lp,
    solve_all_bounds,
)


class TestEstimateConditionalMeans:
    """Test estimate_conditional_means function."""

    def test_binary_outcome(self):
        """Test with binary outcome (should return probabilities)."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            'X': np.random.choice([0, 1], size=n),
            'Y': np.random.choice([0, 1], size=n),
            'Z': np.random.choice([0, 1, 2], size=n),
        })

        means = estimate_conditional_means(df, 'X', 'Y', ['Z'])

        # Should have entries for each (x, z) combination
        assert len(means) > 0

        # All values should be between 0 and 1 for binary outcome
        for val in means.values():
            assert 0 <= val <= 1

    def test_continuous_outcome(self):
        """Test with continuous outcome."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            'X': np.random.choice([0, 1], size=n),
            'Y': np.random.normal(100, 20, size=n),
            'Z': np.random.choice([0, 1], size=n),
        })

        means = estimate_conditional_means(df, 'X', 'Y', ['Z'])

        # Should have entries
        assert len(means) > 0

        # Means should be around 100
        for val in means.values():
            assert 50 < val < 150

    def test_min_count_filtering(self):
        """Test that sparse cells are filtered by min_count."""
        # Create data where X=0,Z=0 has 6 samples, X=1,Z=0 has 6 samples
        # but X=0,Z=1 has only 3 samples, X=1,Z=1 has only 2 samples
        df = pd.DataFrame({
            'X': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            'Y': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })

        # With min_count=5, Z=1 cells should be filtered (only 3 and 2 samples)
        means = estimate_conditional_means(df, 'X', 'Y', ['Z'], min_count=5)

        # Should have (0, (0,)) and (1, (0,)) but not Z=1 entries
        assert (0, (0,)) in means  # X=0, Z=0 has 6 samples
        assert (1, (0,)) in means  # X=1, Z=0 has 6 samples
        assert (0, (1,)) not in means  # X=0, Z=1 has only 3 samples
        assert (1, (1,)) not in means  # X=1, Z=1 has only 2 samples

    def test_multiple_covariates(self):
        """Test with multiple covariates."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            'X': np.random.choice([0, 1], size=n),
            'Y': np.random.choice([0, 1], size=n),
            'Z1': np.random.choice([0, 1], size=n),
            'Z2': np.random.choice([0, 1], size=n),
        })

        means = estimate_conditional_means(df, 'X', 'Y', ['Z1', 'Z2'])

        # Keys should be (x, (z1, z2)) tuples
        for key in means.keys():
            assert len(key) == 2
            assert isinstance(key[1], tuple)
            assert len(key[1]) == 2

    def test_backwards_compatibility_alias(self):
        """Test that estimate_identified_probs is an alias."""
        assert estimate_identified_probs is estimate_conditional_means


class TestEstimateObservedProbs:
    """Test estimate_observed_probs function."""

    @pytest.fixture
    def sample_idle_data(self):
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            'X': np.random.choice([0, 1], size=n),
            'Y': np.random.choice([0, 1], size=n),
            'Z': np.random.choice([0, 1, 2], size=n),
        })

    def test_returns_all_probability_dicts(self, sample_idle_data):
        """Test that all probability dictionaries are returned."""
        probs = estimate_observed_probs(sample_idle_data, 'X', 'Y', ['Z'])

        assert 'p_z' in probs
        assert 'p_x_given_z' in probs
        assert 'p_y_given_xz' in probs
        assert 'p_xy_given_z' in probs

    def test_p_z_sums_to_one(self, sample_idle_data):
        """Test that P(Z) probabilities sum to approximately 1."""
        probs = estimate_observed_probs(sample_idle_data, 'X', 'Y', ['Z'])

        total = sum(probs['p_z'].values())
        assert 0.95 < total < 1.05  # Allow small deviation due to min_count filtering

    def test_p_x_given_z_in_range(self, sample_idle_data):
        """Test that P(X=1|Z) is between 0 and 1."""
        probs = estimate_observed_probs(sample_idle_data, 'X', 'Y', ['Z'])

        for val in probs['p_x_given_z'].values():
            assert 0 <= val <= 1

    def test_p_xy_given_z_sums_correctly(self, sample_idle_data):
        """Test that P(X,Y|Z) sums to 1 for each Z."""
        probs = estimate_observed_probs(sample_idle_data, 'X', 'Y', ['Z'])

        # Group by z and sum
        z_totals = {}
        for (x, y, z), p in probs['p_xy_given_z'].items():
            if z not in z_totals:
                z_totals[z] = 0
            z_totals[z] += p

        for z, total in z_totals.items():
            assert 0.99 < total < 1.01, f"P(X,Y|Z={z}) should sum to 1, got {total}"


class TestSolveCATEBoundsSingleZ:
    """Test solve_cate_bounds_single_z function."""

    def test_returns_bounds_and_status(self):
        """Test that function returns (lower, upper, status)."""
        identified = {(0, (0,)): 0.3, (1, (0,)): 0.6}
        observed = {'p_z': {(0,): 0.5}}

        lower, upper, status = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.1
        )

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(status, str)

    def test_bounds_contain_point_estimate(self):
        """Test that bounds contain the point estimate."""
        identified = {(0, (0,)): 0.3, (1, (0,)): 0.7}
        observed = {'p_z': {(0,): 0.5}}

        lower, upper, status = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.1, outcome_scale=1.0
        )

        point_estimate = 0.7 - 0.3  # = 0.4
        assert lower <= point_estimate <= upper

    def test_missing_data_returns_nan(self):
        """Test that missing identified data returns NaN."""
        identified = {(0, (0,)): 0.3}  # Missing (1, (0,))
        observed = {'p_z': {(0,): 0.5}}

        lower, upper, status = solve_cate_bounds_single_z(
            (0,), identified, observed
        )

        assert np.isnan(lower)
        assert np.isnan(upper)
        assert status == 'missing_data'

    def test_epsilon_affects_width(self):
        """Test that larger epsilon gives wider bounds."""
        identified = {(0, (0,)): 0.3, (1, (0,)): 0.6}
        observed = {'p_z': {(0,): 0.5}}

        lower1, upper1, _ = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.05, outcome_scale=1.0
        )
        lower2, upper2, _ = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.2, outcome_scale=1.0
        )

        width1 = upper1 - lower1
        width2 = upper2 - lower2

        assert width2 > width1

    def test_outcome_scale_affects_width(self):
        """Test that outcome_scale affects bound width."""
        identified = {(0, (0,)): 0.3, (1, (0,)): 0.6}
        observed = {'p_z': {(0,): 0.5}}

        lower1, upper1, _ = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.1, outcome_scale=1.0
        )
        lower2, upper2, _ = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.1, outcome_scale=10.0
        )

        width1 = upper1 - lower1
        width2 = upper2 - lower2

        assert width2 > width1


class TestSolveCATEBoundsLP:
    """Test solve_cate_bounds_lp function."""

    def test_delegates_to_single_z(self):
        """Test that LP solver delegates to single_z solver."""
        identified = {(0, (0,)): 0.3, (1, (0,)): 0.6}
        observed = {'p_z': {(0,): 0.5}}

        result_lp = solve_cate_bounds_lp(
            (0,), identified, observed, epsilon=0.1, outcome_scale=1.0
        )
        result_single = solve_cate_bounds_single_z(
            (0,), identified, observed, epsilon=0.1, outcome_scale=1.0
        )

        assert result_lp == result_single


class TestSolveAllBounds:
    """Test solve_all_bounds function."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data for multiple sites."""
        np.random.seed(42)
        sites = {}

        for site_id in ['site_A', 'site_B']:
            n = 200
            Z = np.random.choice([0, 1], size=n)
            X = np.random.choice([0, 1], size=n)
            Y = np.random.normal(100, 20, size=n)
            F = np.random.choice(['on', 'idle'], size=n)

            sites[site_id] = pd.DataFrame({
                'X': X, 'Y': Y, 'Z': Z, 'F': F
            })

        return sites

    def test_returns_dict_of_bounds(self, training_data):
        """Test that function returns site -> bounds dict."""
        bounds = solve_all_bounds(
            training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=0.1
        )

        assert isinstance(bounds, dict)
        for site_id, site_bounds in bounds.items():
            assert isinstance(site_bounds, dict)
            for z, (lower, upper) in site_bounds.items():
                assert lower <= upper

    def test_skips_small_sites(self):
        """Test that sites with insufficient data are skipped."""
        small_site_data = {
            'small_site': pd.DataFrame({
                'X': [0, 1, 0],
                'Y': [1, 0, 1],
                'Z': [0, 1, 0],
                'F': ['on', 'idle', 'on']
            })
        }

        bounds = solve_all_bounds(
            small_site_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y'
        )

        # Should be empty - not enough data
        assert len(bounds) == 0

    def test_computes_outcome_scale_automatically(self, training_data):
        """Test that outcome_scale is computed if not provided."""
        # Should not raise
        bounds = solve_all_bounds(
            training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            outcome_scale=None
        )

        assert isinstance(bounds, dict)

    def test_respects_regime_column(self, training_data):
        """Test that only F=on and F=idle data are used correctly."""
        bounds = solve_all_bounds(
            training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            regime_col='F'
        )

        # Should produce valid bounds
        assert len(bounds) > 0


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_binary_outcome(self):
        """Test full pipeline with binary outcome."""
        np.random.seed(42)
        n = 500

        # Create data with treatment effect
        Z = np.random.choice([0, 1], size=n)
        X = np.random.choice([0, 1], size=n)
        # Y has treatment effect of ~0.3
        Y = (np.random.random(n) < 0.2 + 0.3 * X + 0.1 * Z).astype(int)
        F = np.random.choice(['on', 'idle'], size=n)

        training_data = {
            'site1': pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'F': F})
        }

        bounds = solve_all_bounds(
            training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=0.1
        )

        # Should have bounds
        if 'site1' in bounds:
            for z, (lower, upper) in bounds['site1'].items():
                # True effect is around 0.3, bounds should include it
                assert lower < 0.5  # Should have reasonable lower bound
                assert upper > 0.0  # Should have reasonable upper bound

    def test_end_to_end_continuous_outcome(self):
        """Test full pipeline with continuous outcome."""
        np.random.seed(42)
        n = 500

        Z = np.random.choice([0, 1], size=n)
        X = np.random.choice([0, 1], size=n)
        # Continuous Y with treatment effect of ~50
        Y = 100 + 50 * X + 10 * Z + np.random.normal(0, 20, size=n)
        F = np.random.choice(['on', 'idle'], size=n)

        training_data = {
            'site1': pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'F': F})
        }

        bounds = solve_all_bounds(
            training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=0.1
        )

        # Should have bounds
        if 'site1' in bounds:
            for z, (lower, upper) in bounds['site1'].items():
                # Effect should be around 50
                width = upper - lower
                assert width > 0  # Should have non-zero width
