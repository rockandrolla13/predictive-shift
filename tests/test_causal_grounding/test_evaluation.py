"""
Tests for Ground Truth Evaluation Metrics

Tests the evaluation.py module functions for computing coverage,
informativeness, interval score, and other quality metrics.
"""

import pytest
import numpy as np
import pandas as pd

from causal_grounding.evaluation import (
    compute_coverage_rate,
    compute_informativeness,
    compute_interval_score,
    compute_sharpness,
    summarize_bound_quality,
    compare_method_quality,
    per_stratum_coverage,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_bounds():
    """Simple bounds DataFrame for testing."""
    return pd.DataFrame({
        'stratum': ['A', 'B', 'C', 'D'],
        'lower': [0.1, 0.2, 0.3, 0.4],
        'upper': [0.5, 0.6, 0.7, 0.8]
    })


@pytest.fixture
def perfect_coverage_bounds():
    """Bounds that perfectly cover ground truth of 0.5."""
    return pd.DataFrame({
        'stratum': ['A', 'B', 'C'],
        'lower': [0.4, 0.3, 0.45],
        'upper': [0.6, 0.7, 0.55]
    })


@pytest.fixture
def partial_coverage_bounds():
    """Bounds with partial coverage for truth=0.5."""
    return pd.DataFrame({
        'stratum': ['covered1', 'covered2', 'miss_below', 'miss_above'],
        'lower': [0.4, 0.3, 0.6, 0.1],
        'upper': [0.6, 0.7, 0.8, 0.4]
    })


@pytest.fixture
def varying_width_bounds():
    """Bounds with different widths for sharpness testing."""
    return pd.DataFrame({
        'stratum': ['tight', 'medium', 'wide'],
        'lower': [0.45, 0.3, 0.1],
        'upper': [0.55, 0.7, 0.9]
    })


# =============================================================================
# TEST: compute_coverage_rate
# =============================================================================

class TestComputeCoverageRate:
    """Tests for compute_coverage_rate function."""

    def test_perfect_coverage_scalar(self, perfect_coverage_bounds):
        """Test 100% coverage with scalar ground truth."""
        rate = compute_coverage_rate(perfect_coverage_bounds, ground_truth=0.5)
        assert rate == 1.0

    def test_zero_coverage(self):
        """Test 0% coverage when truth is outside all bounds."""
        bounds = pd.DataFrame({
            'lower': [0.1, 0.2],
            'upper': [0.3, 0.4]
        })
        rate = compute_coverage_rate(bounds, ground_truth=0.8)
        assert rate == 0.0

    def test_partial_coverage(self, partial_coverage_bounds):
        """Test partial coverage rate."""
        rate = compute_coverage_rate(partial_coverage_bounds, ground_truth=0.5)
        # 2 out of 4 strata cover 0.5
        assert rate == 0.5

    def test_per_stratum_ground_truth(self):
        """Test with per-stratum ground truth values."""
        bounds = pd.DataFrame({
            'lower': [0.1, 0.3, 0.5],
            'upper': [0.2, 0.5, 0.7]
        })
        truth = [0.15, 0.4, 0.8]  # First two covered, third not
        rate = compute_coverage_rate(bounds, ground_truth=truth)
        assert abs(rate - 2/3) < 1e-10

    def test_boundary_cases(self):
        """Test coverage at exact boundaries."""
        bounds = pd.DataFrame({
            'lower': [0.5, 0.5],
            'upper': [0.5, 0.6]
        })
        # Truth at exact lower bound should be covered
        rate = compute_coverage_rate(bounds, ground_truth=0.5)
        assert rate == 1.0

    def test_custom_column_names(self):
        """Test with custom column names."""
        bounds = pd.DataFrame({
            'lb': [0.1, 0.2],
            'ub': [0.5, 0.6]
        })
        rate = compute_coverage_rate(bounds, ground_truth=0.3, lower_col='lb', upper_col='ub')
        assert rate == 1.0

    def test_length_mismatch_error(self):
        """Test error when ground truth length doesn't match bounds."""
        bounds = pd.DataFrame({'lower': [0.1, 0.2], 'upper': [0.5, 0.6]})
        with pytest.raises(ValueError, match="length"):
            compute_coverage_rate(bounds, ground_truth=[0.3, 0.4, 0.5])


# =============================================================================
# TEST: compute_informativeness
# =============================================================================

class TestComputeInformativeness:
    """Tests for compute_informativeness function."""

    def test_lower_improves_mode(self):
        """Test informativeness when lower bounds exceed naive."""
        bounds = pd.DataFrame({
            'lower': [0.3, 0.4, 0.1],  # 2 above naive=0.2
            'upper': [0.6, 0.7, 0.5]
        })
        info = compute_informativeness(bounds, naive_estimates=0.2, mode='lower_improves')
        assert abs(info - 2/3) < 1e-10

    def test_upper_improves_mode(self):
        """Test informativeness when upper bounds below naive."""
        bounds = pd.DataFrame({
            'lower': [0.1, 0.2, 0.3],
            'upper': [0.4, 0.5, 0.6]  # 2 below naive=0.55
        })
        info = compute_informativeness(bounds, naive_estimates=0.55, mode='upper_improves')
        assert abs(info - 2/3) < 1e-10

    def test_width_improves_mode(self):
        """Test informativeness based on width."""
        bounds = pd.DataFrame({
            'lower': [0.4, 0.1],  # widths: 0.1, 0.8
            'upper': [0.5, 0.9]
        })
        # Naive uncertainty = 2*|0.3| + 0.1 = 0.7
        info = compute_informativeness(bounds, naive_estimates=0.3, mode='width_improves')
        assert info == 0.5  # Only first bound (width=0.1) is narrower than 0.7

    def test_per_stratum_naive(self):
        """Test with per-stratum naive estimates."""
        bounds = pd.DataFrame({
            'lower': [0.3, 0.4],
            'upper': [0.5, 0.6]
        })
        naive = [0.2, 0.5]  # First lower > naive[0], second lower < naive[1]
        info = compute_informativeness(bounds, naive_estimates=naive, mode='lower_improves')
        assert info == 0.5

    def test_invalid_mode(self):
        """Test error with invalid mode."""
        bounds = pd.DataFrame({'lower': [0.1], 'upper': [0.5]})
        with pytest.raises(ValueError, match="Unknown mode"):
            compute_informativeness(bounds, naive_estimates=0.3, mode='invalid')


# =============================================================================
# TEST: compute_interval_score
# =============================================================================

class TestComputeIntervalScore:
    """Tests for compute_interval_score function."""

    def test_perfect_coverage_score(self, perfect_coverage_bounds):
        """Test score when all bounds cover truth."""
        score = compute_interval_score(perfect_coverage_bounds, ground_truth=0.5)
        # Score should equal mean width (no penalties)
        widths = perfect_coverage_bounds['upper'] - perfect_coverage_bounds['lower']
        expected = widths.mean()
        assert abs(score - expected) < 1e-10

    def test_penalty_for_miss_below(self):
        """Test penalty when truth is below lower bound."""
        bounds = pd.DataFrame({'lower': [0.5], 'upper': [0.6]})
        truth = 0.3  # Below lower bound by 0.2
        score = compute_interval_score(bounds, ground_truth=truth, alpha=0.05)
        # Score = width + (2/alpha) * (lower - truth)
        expected = 0.1 + (2/0.05) * 0.2
        assert abs(score - expected) < 1e-10

    def test_penalty_for_miss_above(self):
        """Test penalty when truth is above upper bound."""
        bounds = pd.DataFrame({'lower': [0.1], 'upper': [0.3]})
        truth = 0.5  # Above upper bound by 0.2
        score = compute_interval_score(bounds, ground_truth=truth, alpha=0.05)
        # Score = width + (2/alpha) * (truth - upper)
        expected = 0.2 + (2/0.05) * 0.2
        assert abs(score - expected) < 1e-10

    def test_lower_alpha_higher_penalty(self):
        """Test that lower alpha gives higher penalty."""
        bounds = pd.DataFrame({'lower': [0.5], 'upper': [0.6]})
        truth = 0.3

        score_low_alpha = compute_interval_score(bounds, truth, alpha=0.01)
        score_high_alpha = compute_interval_score(bounds, truth, alpha=0.10)

        assert score_low_alpha > score_high_alpha

    def test_tighter_bounds_better_score(self):
        """Test that tighter bounds give better (lower) score when covered."""
        bounds_tight = pd.DataFrame({'lower': [0.45], 'upper': [0.55]})
        bounds_wide = pd.DataFrame({'lower': [0.1], 'upper': [0.9]})

        score_tight = compute_interval_score(bounds_tight, ground_truth=0.5)
        score_wide = compute_interval_score(bounds_wide, ground_truth=0.5)

        assert score_tight < score_wide


# =============================================================================
# TEST: compute_sharpness
# =============================================================================

class TestComputeSharpness:
    """Tests for compute_sharpness function."""

    def test_sharpness_metrics(self, varying_width_bounds):
        """Test sharpness metrics computation."""
        sharpness = compute_sharpness(varying_width_bounds)

        # Widths: 0.1, 0.4, 0.8
        assert 'mean_width' in sharpness
        assert 'median_width' in sharpness
        assert 'std_width' in sharpness
        assert 'min_width' in sharpness
        assert 'max_width' in sharpness

        assert abs(sharpness['min_width'] - 0.1) < 1e-10
        assert abs(sharpness['max_width'] - 0.8) < 1e-10

    def test_uniform_widths(self):
        """Test with uniform widths."""
        bounds = pd.DataFrame({
            'lower': [0.1, 0.2, 0.3],
            'upper': [0.3, 0.4, 0.5]  # All width 0.2
        })
        sharpness = compute_sharpness(bounds)

        assert abs(sharpness['mean_width'] - 0.2) < 1e-10
        assert abs(sharpness['median_width'] - 0.2) < 1e-10
        assert abs(sharpness['std_width']) < 1e-10  # Zero variance


# =============================================================================
# TEST: summarize_bound_quality
# =============================================================================

class TestSummarizeBoundQuality:
    """Tests for summarize_bound_quality function."""

    def test_full_summary(self, simple_bounds):
        """Test full summary with all metrics."""
        summary = summarize_bound_quality(
            simple_bounds,
            ground_truth=0.5,
            naive_estimates=0.2
        )

        assert 'coverage_rate' in summary
        assert 'interval_score' in summary
        assert 'mean_width' in summary
        assert 'median_width' in summary
        assert 'informativeness_lower' in summary
        assert 'informativeness_upper' in summary
        assert 'n_strata' in summary

        assert summary['n_strata'] == 4

    def test_summary_without_naive(self, simple_bounds):
        """Test summary without naive estimates."""
        summary = summarize_bound_quality(simple_bounds, ground_truth=0.5)

        assert 'coverage_rate' in summary
        assert 'informativeness_lower' not in summary

    def test_summary_consistency(self, simple_bounds):
        """Test that summary values match individual function calls."""
        summary = summarize_bound_quality(simple_bounds, ground_truth=0.5)

        coverage = compute_coverage_rate(simple_bounds, ground_truth=0.5)
        interval = compute_interval_score(simple_bounds, ground_truth=0.5)
        sharpness = compute_sharpness(simple_bounds)

        assert abs(summary['coverage_rate'] - coverage) < 1e-10
        assert abs(summary['interval_score'] - interval) < 1e-10
        assert abs(summary['mean_width'] - sharpness['mean_width']) < 1e-10


# =============================================================================
# TEST: compare_method_quality
# =============================================================================

class TestCompareMethodQuality:
    """Tests for compare_method_quality function."""

    def test_compare_two_methods(self):
        """Test comparing two methods."""
        bounds_a = pd.DataFrame({'lower': [0.4, 0.3], 'upper': [0.6, 0.7]})
        bounds_b = pd.DataFrame({'lower': [0.45, 0.35], 'upper': [0.55, 0.65]})

        comparison = compare_method_quality(
            {'method_a': bounds_a, 'method_b': bounds_b},
            ground_truth=0.5
        )

        assert len(comparison) == 2
        assert 'method' in comparison.columns
        assert 'coverage_rate' in comparison.columns
        assert set(comparison['method']) == {'method_a', 'method_b'}

    def test_tighter_method_better_sharpness(self):
        """Test that tighter method has better sharpness."""
        bounds_wide = pd.DataFrame({'lower': [0.1], 'upper': [0.9]})
        bounds_tight = pd.DataFrame({'lower': [0.45], 'upper': [0.55]})

        comparison = compare_method_quality(
            {'wide': bounds_wide, 'tight': bounds_tight},
            ground_truth=0.5
        )

        wide_width = comparison.loc[comparison['method'] == 'wide', 'mean_width'].values[0]
        tight_width = comparison.loc[comparison['method'] == 'tight', 'mean_width'].values[0]

        assert tight_width < wide_width


# =============================================================================
# TEST: per_stratum_coverage
# =============================================================================

class TestPerStratumCoverage:
    """Tests for per_stratum_coverage function."""

    def test_per_stratum_output(self, partial_coverage_bounds):
        """Test per-stratum output columns."""
        result = per_stratum_coverage(partial_coverage_bounds, ground_truth=0.5)

        assert 'ground_truth' in result.columns
        assert 'covered' in result.columns
        assert 'miss_distance' in result.columns

    def test_covered_strata_zero_miss(self, partial_coverage_bounds):
        """Test that covered strata have zero miss distance."""
        result = per_stratum_coverage(partial_coverage_bounds, ground_truth=0.5)

        # Covered strata should have miss_distance = 0
        covered_miss = result.loc[result['covered'], 'miss_distance']
        assert all(covered_miss == 0)

    def test_miss_distance_calculation(self):
        """Test miss distance calculation for uncovered strata."""
        bounds = pd.DataFrame({
            'stratum': ['below', 'above'],
            'lower': [0.6, 0.1],
            'upper': [0.8, 0.3]
        })
        result = per_stratum_coverage(bounds, ground_truth=0.5)

        # 'below' misses by 0.1 (truth=0.5 < lower=0.6)
        below_miss = result.loc[result['stratum'] == 'below', 'miss_distance'].values[0]
        assert abs(below_miss - 0.1) < 1e-10

        # 'above' misses by 0.2 (truth=0.5 > upper=0.3)
        above_miss = result.loc[result['stratum'] == 'above', 'miss_distance'].values[0]
        assert abs(above_miss - 0.2) < 1e-10

    def test_per_stratum_ground_truth(self):
        """Test with per-stratum ground truth values."""
        bounds = pd.DataFrame({
            'stratum': ['A', 'B'],
            'lower': [0.1, 0.5],
            'upper': [0.4, 0.8]
        })
        truth = [0.3, 0.9]  # A covered, B not

        result = per_stratum_coverage(bounds, ground_truth=truth)

        assert result.loc[result['stratum'] == 'A', 'covered'].values[0] == True
        assert result.loc[result['stratum'] == 'B', 'covered'].values[0] == False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for evaluation module."""

    def test_realistic_bounds_evaluation(self):
        """Test with realistic-looking CATE bounds."""
        # Simulate bounds similar to experiment output
        np.random.seed(42)
        n_strata = 30

        bounds = pd.DataFrame({
            'stratum': [f'stratum_{i}' for i in range(n_strata)],
            'lower': 1000 + np.random.uniform(-200, 200, n_strata),
            'upper': 1700 + np.random.uniform(-200, 200, n_strata)
        })

        # True ATE around 1550
        true_ate = 1550

        # Naive observational estimate (biased)
        naive = 1200

        summary = summarize_bound_quality(bounds, true_ate, naive)

        # Sanity checks
        assert 0 <= summary['coverage_rate'] <= 1
        assert summary['mean_width'] > 0
        assert summary['n_strata'] == n_strata

    def test_evaluation_with_estimator_output_format(self):
        """Test that evaluation works with expected estimator output format."""
        # Format expected from CausalGroundingEstimator
        bounds = pd.DataFrame({
            'z_value': ['young_male', 'young_female', 'old_male', 'old_female'],
            'lower_bound': [1000, 1100, 1200, 1150],
            'upper_bound': [1800, 1700, 1900, 1850],
            'width': [800, 600, 700, 700],
            'n_samples': [100, 120, 90, 110]
        })

        true_ate = 1550

        # Should work with custom column names
        coverage = compute_coverage_rate(
            bounds, true_ate,
            lower_col='lower_bound', upper_col='upper_bound'
        )

        assert 0 <= coverage <= 1
