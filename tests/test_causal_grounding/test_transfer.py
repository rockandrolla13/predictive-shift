"""
Test Bound Transfer Across Environments.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding.transfer import (
    transfer_bounds_conservative,
    transfer_bounds_average,
    transfer_bounds_weighted,
    compute_bound_metrics,
    bounds_to_dataframe,
)


@pytest.fixture
def sample_training_bounds():
    """Sample training bounds from 3 sites."""
    return {
        'site_A': {
            (0,): (-0.1, 0.3),
            (1,): (0.1, 0.4),
            (2,): (0.2, 0.5),
        },
        'site_B': {
            (0,): (-0.2, 0.2),
            (1,): (0.0, 0.5),
            (2,): (0.1, 0.6),
        },
        'site_C': {
            (0,): (0.0, 0.25),
            (1,): (0.15, 0.35),
            (2,): (0.25, 0.45),
        },
    }


class TestTransferBoundsConservative:
    """Test transfer_bounds_conservative function."""

    def test_returns_dict(self, sample_training_bounds):
        """Test that function returns a dictionary."""
        result = transfer_bounds_conservative(sample_training_bounds)
        assert isinstance(result, dict)

    def test_takes_min_lower_max_upper(self, sample_training_bounds):
        """Test that conservative takes min(lower) and max(upper)."""
        result = transfer_bounds_conservative(sample_training_bounds)

        # For z=(0,): lowers are [-0.1, -0.2, 0.0], uppers are [0.3, 0.2, 0.25]
        # min(lowers) = -0.2, max(uppers) = 0.3
        assert result[(0,)] == (-0.2, 0.3)

        # For z=(1,): lowers are [0.1, 0.0, 0.15], uppers are [0.4, 0.5, 0.35]
        # min(lowers) = 0.0, max(uppers) = 0.5
        assert result[(1,)] == (0.0, 0.5)

    def test_wider_than_any_individual_site(self, sample_training_bounds):
        """Test that conservative bounds are at least as wide as any site."""
        result = transfer_bounds_conservative(sample_training_bounds)

        for z in result.keys():
            cons_lower, cons_upper = result[z]
            cons_width = cons_upper - cons_lower

            for site_bounds in sample_training_bounds.values():
                if z in site_bounds:
                    site_lower, site_upper = site_bounds[z]
                    site_width = site_upper - site_lower
                    assert cons_width >= site_width - 1e-10

    def test_respects_z_values_subset(self, sample_training_bounds):
        """Test filtering by z_values subset."""
        result = transfer_bounds_conservative(
            sample_training_bounds,
            z_values=[(0,), (1,)]
        )

        assert (0,) in result
        assert (1,) in result
        assert (2,) not in result

    def test_missing_z_returns_uninformative(self, sample_training_bounds):
        """Test that missing z values get uninformative bounds."""
        result = transfer_bounds_conservative(
            sample_training_bounds,
            z_values=[(0,), (99,)]  # (99,) doesn't exist
        )

        assert result[(99,)] == (-1.0, 1.0)

    def test_empty_training_bounds(self):
        """Test with empty training bounds."""
        result = transfer_bounds_conservative({})
        assert result == {}


class TestTransferBoundsAverage:
    """Test transfer_bounds_average function."""

    def test_returns_dict(self, sample_training_bounds):
        """Test that function returns a dictionary."""
        result = transfer_bounds_average(sample_training_bounds)
        assert isinstance(result, dict)

    def test_computes_mean_bounds(self, sample_training_bounds):
        """Test that average computes mean of bounds."""
        result = transfer_bounds_average(sample_training_bounds)

        # For z=(0,): lowers are [-0.1, -0.2, 0.0], uppers are [0.3, 0.2, 0.25]
        expected_lower = np.mean([-0.1, -0.2, 0.0])
        expected_upper = np.mean([0.3, 0.2, 0.25])

        assert abs(result[(0,)][0] - expected_lower) < 1e-10
        assert abs(result[(0,)][1] - expected_upper) < 1e-10

    def test_narrower_than_conservative(self, sample_training_bounds):
        """Test that average bounds are narrower than conservative."""
        conservative = transfer_bounds_conservative(sample_training_bounds)
        average = transfer_bounds_average(sample_training_bounds)

        for z in average.keys():
            cons_width = conservative[z][1] - conservative[z][0]
            avg_width = average[z][1] - average[z][0]
            assert avg_width <= cons_width + 1e-10

    def test_respects_z_values_subset(self, sample_training_bounds):
        """Test filtering by z_values subset."""
        result = transfer_bounds_average(
            sample_training_bounds,
            z_values=[(1,)]
        )

        assert len(result) == 1
        assert (1,) in result


class TestTransferBoundsWeighted:
    """Test transfer_bounds_weighted function."""

    def test_returns_dict(self, sample_training_bounds):
        """Test that function returns a dictionary."""
        weights = {'site_A': 1.0, 'site_B': 1.0, 'site_C': 1.0}
        result = transfer_bounds_weighted(sample_training_bounds, weights)
        assert isinstance(result, dict)

    def test_equal_weights_equals_average(self, sample_training_bounds):
        """Test that equal weights produce same result as average."""
        weights = {'site_A': 1.0, 'site_B': 1.0, 'site_C': 1.0}
        weighted = transfer_bounds_weighted(sample_training_bounds, weights)
        average = transfer_bounds_average(sample_training_bounds)

        for z in weighted.keys():
            assert abs(weighted[z][0] - average[z][0]) < 1e-10
            assert abs(weighted[z][1] - average[z][1]) < 1e-10

    def test_single_site_weight(self, sample_training_bounds):
        """Test that single non-zero weight returns that site's bounds."""
        weights = {'site_A': 1.0, 'site_B': 0.0, 'site_C': 0.0}
        result = transfer_bounds_weighted(sample_training_bounds, weights)

        # Should equal site_A bounds
        for z in sample_training_bounds['site_A'].keys():
            assert result[z] == sample_training_bounds['site_A'][z]

    def test_higher_weight_more_influence(self, sample_training_bounds):
        """Test that higher weights have more influence."""
        # site_A has lower=(−0.1) for z=(0,), site_B has lower=(-0.2)
        # With higher weight on site_A, result should be closer to -0.1

        weights_a_high = {'site_A': 10.0, 'site_B': 1.0, 'site_C': 1.0}
        weights_b_high = {'site_A': 1.0, 'site_B': 10.0, 'site_C': 1.0}

        result_a = transfer_bounds_weighted(sample_training_bounds, weights_a_high)
        result_b = transfer_bounds_weighted(sample_training_bounds, weights_b_high)

        # result_a lower should be closer to -0.1 (site_A)
        # result_b lower should be closer to -0.2 (site_B)
        assert result_a[(0,)][0] > result_b[(0,)][0]

    def test_missing_weight_uses_default(self, sample_training_bounds):
        """Test that missing weights default to 1.0."""
        weights = {'site_A': 1.0}  # Missing site_B and site_C
        result = transfer_bounds_weighted(sample_training_bounds, weights)

        # Should still produce valid bounds
        assert len(result) > 0
        for z, (lower, upper) in result.items():
            assert lower <= upper


class TestComputeBoundMetrics:
    """Test compute_bound_metrics function."""

    @pytest.fixture
    def sample_bounds(self):
        return {
            (0,): (0.0, 0.5),
            (1,): (0.1, 0.4),
            (2,): (0.2, 0.6),
        }

    def test_returns_required_keys(self, sample_bounds):
        """Test that all required metric keys are present."""
        metrics = compute_bound_metrics(sample_bounds)

        assert 'mean_width' in metrics
        assert 'median_width' in metrics
        assert 'min_width' in metrics
        assert 'max_width' in metrics
        assert 'n_z_values' in metrics

    def test_width_calculations(self, sample_bounds):
        """Test that width calculations are correct."""
        metrics = compute_bound_metrics(sample_bounds)

        # Widths: 0.5, 0.3, 0.4
        assert abs(metrics['min_width'] - 0.3) < 1e-10
        assert abs(metrics['max_width'] - 0.5) < 1e-10
        assert abs(metrics['mean_width'] - 0.4) < 1e-10
        assert abs(metrics['median_width'] - 0.4) < 1e-10

    def test_n_z_values(self, sample_bounds):
        """Test that n_z_values is correct."""
        metrics = compute_bound_metrics(sample_bounds)
        assert metrics['n_z_values'] == 3

    def test_coverage_with_true_cate(self, sample_bounds):
        """Test coverage calculation with ground truth."""
        true_cate = {
            (0,): 0.25,  # In [0.0, 0.5] -> covered
            (1,): 0.5,   # Not in [0.1, 0.4] -> not covered
            (2,): 0.4,   # In [0.2, 0.6] -> covered
        }

        metrics = compute_bound_metrics(sample_bounds, true_cate)

        assert 'coverage' in metrics
        assert metrics['coverage'] == 2/3  # 2 out of 3 covered
        assert metrics['n_evaluated'] == 3

    def test_coverage_perfect(self):
        """Test 100% coverage when all true values in bounds."""
        bounds = {(0,): (0.0, 1.0), (1,): (0.0, 1.0)}
        true_cate = {(0,): 0.5, (1,): 0.5}

        metrics = compute_bound_metrics(bounds, true_cate)
        assert metrics['coverage'] == 1.0

    def test_coverage_zero(self):
        """Test 0% coverage when no true values in bounds."""
        bounds = {(0,): (0.0, 0.1), (1,): (0.0, 0.1)}
        true_cate = {(0,): 0.5, (1,): 0.5}

        metrics = compute_bound_metrics(bounds, true_cate)
        assert metrics['coverage'] == 0.0

    def test_coverage_partial_overlap_true_cate(self):
        """Test coverage when true_cate has different z values."""
        bounds = {(0,): (0.0, 0.5), (1,): (0.0, 0.5), (2,): (0.0, 0.5)}
        true_cate = {(0,): 0.25, (1,): 0.25}  # Only 2 of 3 z values

        metrics = compute_bound_metrics(bounds, true_cate)
        assert metrics['n_evaluated'] == 2
        assert metrics['coverage'] == 1.0  # Both evaluated are covered


class TestBoundsToDataframe:
    """Test bounds_to_dataframe function."""

    @pytest.fixture
    def sample_bounds(self):
        return {
            (0,): (0.0, 0.5),
            (1,): (0.1, 0.4),
            (2,): (0.2, 0.6),
        }

    def test_returns_dataframe(self, sample_bounds):
        """Test that function returns a DataFrame."""
        df = bounds_to_dataframe(sample_bounds)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sample_bounds):
        """Test that DataFrame has required columns."""
        df = bounds_to_dataframe(sample_bounds)

        assert 'cate_lower' in df.columns
        assert 'cate_upper' in df.columns
        assert 'width' in df.columns
        assert 'z' in df.columns

    def test_correct_number_of_rows(self, sample_bounds):
        """Test that DataFrame has correct number of rows."""
        df = bounds_to_dataframe(sample_bounds)
        assert len(df) == 3

    def test_width_calculation(self, sample_bounds):
        """Test that width is correctly calculated."""
        df = bounds_to_dataframe(sample_bounds)

        for _, row in df.iterrows():
            expected_width = row['cate_upper'] - row['cate_lower']
            assert abs(row['width'] - expected_width) < 1e-10

    def test_covariate_names_expansion(self):
        """Test that covariate names expand z tuple."""
        bounds = {
            (0, 1): (0.0, 0.5),
            (1, 0): (0.1, 0.4),
        }
        df = bounds_to_dataframe(bounds, covariate_names=['Z1', 'Z2'])

        assert 'Z1' in df.columns
        assert 'Z2' in df.columns

    def test_covariate_values_correct(self):
        """Test that covariate values are extracted correctly."""
        bounds = {
            (0, 1): (0.0, 0.5),
            (1, 0): (0.1, 0.4),
        }
        df = bounds_to_dataframe(bounds, covariate_names=['Z1', 'Z2'])

        # Find row with z=(0, 1)
        row = df[df['z'] == (0, 1)].iloc[0]
        assert row['Z1'] == 0
        assert row['Z2'] == 1

    def test_column_order(self):
        """Test that columns are ordered correctly."""
        bounds = {(0, 1): (0.0, 0.5)}
        df = bounds_to_dataframe(bounds, covariate_names=['Z1', 'Z2'])

        # Covariate names should come first
        assert list(df.columns[:2]) == ['Z1', 'Z2']


class TestIntegration:
    """Integration tests for transfer functions."""

    def test_conservative_then_metrics(self, sample_training_bounds):
        """Test pipeline: conservative transfer -> metrics."""
        bounds = transfer_bounds_conservative(sample_training_bounds)
        metrics = compute_bound_metrics(bounds)

        assert metrics['n_z_values'] == 3
        assert metrics['mean_width'] > 0

    def test_average_then_dataframe(self, sample_training_bounds):
        """Test pipeline: average transfer -> dataframe."""
        bounds = transfer_bounds_average(sample_training_bounds)
        df = bounds_to_dataframe(bounds, covariate_names=['Z'])

        assert len(df) == 3
        assert 'Z' in df.columns

    def test_all_transfer_methods_same_keys(self, sample_training_bounds):
        """Test that all transfer methods produce same z keys."""
        weights = {'site_A': 1.0, 'site_B': 1.0, 'site_C': 1.0}

        conservative = transfer_bounds_conservative(sample_training_bounds)
        average = transfer_bounds_average(sample_training_bounds)
        weighted = transfer_bounds_weighted(sample_training_bounds, weights)

        assert set(conservative.keys()) == set(average.keys())
        assert set(average.keys()) == set(weighted.keys())

    def test_conservative_widest_average_middle_weighted_varies(self, sample_training_bounds):
        """Test relative widths of different transfer methods."""
        weights = {'site_A': 1.0, 'site_B': 1.0, 'site_C': 1.0}

        conservative = transfer_bounds_conservative(sample_training_bounds)
        average = transfer_bounds_average(sample_training_bounds)

        cons_metrics = compute_bound_metrics(conservative)
        avg_metrics = compute_bound_metrics(average)

        # Conservative should be wider on average
        assert cons_metrics['mean_width'] >= avg_metrics['mean_width']
