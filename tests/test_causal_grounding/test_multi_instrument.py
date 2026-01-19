"""
Tests for Multi-Instrument Aggregation

Tests the functions for aggregating bounds across multiple instruments.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from causal_grounding.transfer import (
    aggregate_across_instruments,
    aggregate_with_weights,
    compute_instrument_agreement,
)

from causal_grounding.covariate_scoring import (
    select_top_k_instruments,
    rank_covariates,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_instrument_bounds():
    """Simple bounds from two instruments."""
    return {
        'Z1': {
            (0,): (0.1, 0.5),
            (1,): (0.2, 0.6),
        },
        'Z2': {
            (0,): (0.15, 0.45),
            (1,): (0.18, 0.55),
        }
    }


@pytest.fixture
def multi_instrument_bounds():
    """Bounds from three instruments with varying coverage."""
    return {
        'Z1': {
            (0,): (0.1, 0.5),
            (1,): (0.2, 0.6),
            (2,): (0.3, 0.7),
        },
        'Z2': {
            (0,): (0.15, 0.45),
            (1,): (0.18, 0.58),
            # Missing (2,) - not covered by Z2
        },
        'Z3': {
            (0,): (0.12, 0.48),
            (1,): (0.22, 0.52),
            (2,): (0.28, 0.65),
        }
    }


@pytest.fixture
def conflicting_bounds():
    """Bounds that don't overlap (instruments disagree)."""
    return {
        'Z1': {
            (0,): (0.6, 0.8),  # High bounds
        },
        'Z2': {
            (0,): (0.1, 0.3),  # Low bounds - no overlap with Z1
        }
    }


@pytest.fixture
def sample_scores_df():
    """Sample covariate scores DataFrame."""
    return pd.DataFrame({
        'z_a': ['Z1', 'Z2', 'Z3', 'Z4', 'Z5'],
        'score': [0.8, 0.6, 0.5, 0.3, 0.1],
        'passes_ehs': [True, True, True, False, False],
        'passes_full_ehs': [True, True, False, False, False],
        'weak_instrument_warning': [False, False, False, True, False],
    })


# =============================================================================
# TEST: aggregate_across_instruments
# =============================================================================

class TestAggregateAcrossInstruments:
    """Tests for aggregate_across_instruments function."""

    def test_intersection_basic(self, simple_instrument_bounds):
        """Test intersection aggregation."""
        result = aggregate_across_instruments(
            simple_instrument_bounds, method='intersection'
        )

        # For (0,): max(0.1, 0.15) = 0.15, min(0.5, 0.45) = 0.45
        assert abs(result[(0,)][0] - 0.15) < 0.001
        assert abs(result[(0,)][1] - 0.45) < 0.001

        # For (1,): max(0.2, 0.18) = 0.2, min(0.6, 0.55) = 0.55
        assert abs(result[(1,)][0] - 0.2) < 0.001
        assert abs(result[(1,)][1] - 0.55) < 0.001

    def test_union_basic(self, simple_instrument_bounds):
        """Test union aggregation."""
        result = aggregate_across_instruments(
            simple_instrument_bounds, method='union'
        )

        # For (0,): min(0.1, 0.15) = 0.1, max(0.5, 0.45) = 0.5
        assert abs(result[(0,)][0] - 0.1) < 0.001
        assert abs(result[(0,)][1] - 0.5) < 0.001

    def test_intersection_tighter_than_union(self, simple_instrument_bounds):
        """Test that intersection gives tighter bounds than union."""
        intersection = aggregate_across_instruments(
            simple_instrument_bounds, method='intersection'
        )
        union = aggregate_across_instruments(
            simple_instrument_bounds, method='union'
        )

        for z in intersection.keys():
            int_width = intersection[z][1] - intersection[z][0]
            union_width = union[z][1] - union[z][0]
            assert int_width <= union_width + 0.001

    def test_partial_coverage(self, multi_instrument_bounds):
        """Test with instruments that have different coverage."""
        result = aggregate_across_instruments(
            multi_instrument_bounds, method='intersection'
        )

        # All strata should be covered
        assert (0,) in result
        assert (1,) in result
        assert (2,) in result  # Should work even though Z2 doesn't have it

    def test_conflicting_bounds(self, conflicting_bounds):
        """Test handling of non-overlapping bounds."""
        result = aggregate_across_instruments(
            conflicting_bounds, method='intersection'
        )

        # Non-overlapping bounds should collapse to midpoint
        lower, upper = result[(0,)]
        assert lower == upper  # Collapsed to single point

    def test_invalid_method(self, simple_instrument_bounds):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="Unknown method"):
            aggregate_across_instruments(
                simple_instrument_bounds, method='invalid'
            )

    def test_single_instrument(self):
        """Test with single instrument."""
        single = {'Z1': {(0,): (0.2, 0.6)}}

        result = aggregate_across_instruments(single, method='intersection')

        assert result[(0,)] == (0.2, 0.6)

    def test_three_instruments(self, multi_instrument_bounds):
        """Test with three instruments."""
        result = aggregate_across_instruments(
            multi_instrument_bounds, method='intersection'
        )

        # For stratum (0,): max(0.1, 0.15, 0.12) = 0.15
        #                   min(0.5, 0.45, 0.48) = 0.45
        lower, upper = result[(0,)]
        assert abs(lower - 0.15) < 0.001
        assert abs(upper - 0.45) < 0.001


# =============================================================================
# TEST: aggregate_with_weights
# =============================================================================

class TestAggregateWithWeights:
    """Tests for aggregate_with_weights function."""

    def test_equal_weights(self, simple_instrument_bounds):
        """Test with equal weights (should be like average)."""
        weights = {'Z1': 1.0, 'Z2': 1.0}

        result = aggregate_with_weights(simple_instrument_bounds, weights)

        # For (0,): (0.1 + 0.15) / 2 = 0.125, (0.5 + 0.45) / 2 = 0.475
        assert abs(result[(0,)][0] - 0.125) < 0.001
        assert abs(result[(0,)][1] - 0.475) < 0.001

    def test_different_weights(self, simple_instrument_bounds):
        """Test with different weights."""
        weights = {'Z1': 2.0, 'Z2': 1.0}

        result = aggregate_with_weights(simple_instrument_bounds, weights)

        # For (0,): (2*0.1 + 1*0.15) / 3 = 0.35/3 ≈ 0.117
        #           (2*0.5 + 1*0.45) / 3 = 1.45/3 ≈ 0.483
        expected_lower = (2 * 0.1 + 1 * 0.15) / 3
        expected_upper = (2 * 0.5 + 1 * 0.45) / 3
        assert abs(result[(0,)][0] - expected_lower) < 0.001
        assert abs(result[(0,)][1] - expected_upper) < 0.001

    def test_zero_weight_ignored(self, simple_instrument_bounds):
        """Test that zero-weight instruments are ignored."""
        weights = {'Z1': 1.0, 'Z2': 0.0}

        result = aggregate_with_weights(simple_instrument_bounds, weights)

        # Should only use Z1
        assert result[(0,)] == (0.1, 0.5)

    def test_missing_weight_default(self, simple_instrument_bounds):
        """Test that missing weights default to 1.0."""
        weights = {'Z1': 2.0}  # Z2 not specified

        result = aggregate_with_weights(simple_instrument_bounds, weights)

        # Z2 should have weight 1.0
        # For (0,): (2*0.1 + 1*0.15) / 3, (2*0.5 + 1*0.45) / 3
        expected_lower = (2 * 0.1 + 1 * 0.15) / 3
        assert abs(result[(0,)][0] - expected_lower) < 0.001


# =============================================================================
# TEST: compute_instrument_agreement
# =============================================================================

class TestComputeInstrumentAgreement:
    """Tests for compute_instrument_agreement function."""

    def test_basic_agreement(self, simple_instrument_bounds):
        """Test basic agreement computation."""
        result = compute_instrument_agreement(simple_instrument_bounds)

        assert isinstance(result, pd.DataFrame)
        assert 'stratum' in result.columns
        assert 'overlap' in result.columns
        assert 'agreement_score' in result.columns

    def test_overlapping_bounds(self, simple_instrument_bounds):
        """Test agreement for overlapping bounds."""
        result = compute_instrument_agreement(simple_instrument_bounds)

        # All bounds should overlap
        assert result['overlap'].all()

        # Agreement scores should be positive
        assert (result['agreement_score'] > 0).all()

    def test_non_overlapping_bounds(self, conflicting_bounds):
        """Test agreement for non-overlapping bounds."""
        result = compute_instrument_agreement(conflicting_bounds)

        # Should not overlap
        assert not result['overlap'].iloc[0]

        # Agreement score should be 0
        assert result['agreement_score'].iloc[0] == 0.0

    def test_agreement_score_range(self, multi_instrument_bounds):
        """Test that agreement scores are in [0, 1]."""
        result = compute_instrument_agreement(multi_instrument_bounds)

        assert (result['agreement_score'] >= 0).all()
        assert (result['agreement_score'] <= 1).all()

    def test_n_instruments_correct(self, multi_instrument_bounds):
        """Test that instrument count is correct."""
        result = compute_instrument_agreement(multi_instrument_bounds)

        # Stratum (0,) is covered by all 3 instruments
        stratum_0 = result[result['stratum'] == "(0,)"]
        if len(stratum_0) > 0:
            assert stratum_0['n_instruments'].iloc[0] == 3


# =============================================================================
# TEST: select_top_k_instruments
# =============================================================================

class TestSelectTopKInstruments:
    """Tests for select_top_k_instruments function."""

    def test_basic_top_k(self, sample_scores_df):
        """Test basic top-k selection."""
        result = select_top_k_instruments(sample_scores_df, k=3)

        assert len(result) == 3
        assert result == ['Z1', 'Z2', 'Z3']

    def test_k_larger_than_available(self, sample_scores_df):
        """Test when k is larger than available instruments."""
        result = select_top_k_instruments(sample_scores_df, k=10)

        # Should return all available
        assert len(result) == 5

    def test_require_passes_ehs(self, sample_scores_df):
        """Test filtering by EHS criterion."""
        result = select_top_k_instruments(
            sample_scores_df, k=5, require_passes_ehs=True
        )

        # Only Z1, Z2, Z3 pass EHS
        assert len(result) == 3
        assert all(z in result for z in ['Z1', 'Z2', 'Z3'])

    def test_require_full_ehs(self, sample_scores_df):
        """Test filtering by full EHS criterion."""
        result = select_top_k_instruments(
            sample_scores_df, k=5, require_full_ehs=True
        )

        # Only Z1, Z2 pass full EHS
        assert len(result) == 2
        assert all(z in result for z in ['Z1', 'Z2'])

    def test_exclude_weak_instruments(self, sample_scores_df):
        """Test excluding weak instruments."""
        result = select_top_k_instruments(
            sample_scores_df, k=5, exclude_weak_instruments=True
        )

        # Z4 should be excluded
        assert 'Z4' not in result

    def test_empty_result(self, sample_scores_df):
        """Test when no instruments qualify."""
        # Require full EHS and exclude weak, but also require high pass rate
        result = select_top_k_instruments(
            sample_scores_df, k=5,
            require_full_ehs=True,
            min_pass_rate=0.99  # Very high threshold
        )

        # May return empty or very few
        assert isinstance(result, list)

    def test_k_equals_one(self, sample_scores_df):
        """Test with k=1."""
        result = select_top_k_instruments(sample_scores_df, k=1)

        assert len(result) == 1
        assert result[0] == 'Z1'  # Highest score


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for multi-instrument aggregation."""

    def test_full_workflow(self):
        """Test full multi-instrument workflow."""
        # Step 1: Create sample data
        np.random.seed(42)
        n = 500

        Z1 = np.random.binomial(1, 0.5, n)
        Z2 = np.random.binomial(1, 0.6, n)
        Z3 = np.random.binomial(1, 0.4, n)
        X = np.random.binomial(1, 0.3 + 0.2 * Z1 + 0.15 * Z2, n)
        Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z1, n)

        data = pd.DataFrame({
            'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
        })

        # Step 2: Score instruments
        from causal_grounding.ci_tests import CITestEngine
        engine = CITestEngine(n_permutations=100, random_seed=42)

        scores = rank_covariates(
            data, ['Z1', 'Z2', 'Z3'],
            treatment='X', outcome='Y',
            ci_engine=engine, use_permutation_test=True
        )

        # Step 3: Select top instruments
        top_instruments = select_top_k_instruments(scores, k=2)
        assert len(top_instruments) <= 2

        # Step 4: Create mock bounds per instrument
        instrument_bounds = {
            inst: {(0,): (0.1, 0.5), (1,): (0.2, 0.6)}
            for inst in top_instruments
        }

        # Step 5: Aggregate
        if len(instrument_bounds) > 0:
            aggregated = aggregate_across_instruments(
                instrument_bounds, method='intersection'
            )
            assert len(aggregated) > 0

    def test_aggregation_tightens_bounds(self):
        """Test that multi-instrument aggregation tightens bounds."""
        # Create bounds where intersection is strictly tighter
        instrument_bounds = {
            'Z1': {(0,): (0.0, 0.6)},
            'Z2': {(0,): (0.2, 0.8)},
            'Z3': {(0,): (0.1, 0.7)},
        }

        intersection = aggregate_across_instruments(
            instrument_bounds, method='intersection'
        )
        union = aggregate_across_instruments(
            instrument_bounds, method='union'
        )

        int_width = intersection[(0,)][1] - intersection[(0,)][0]
        union_width = union[(0,)][1] - union[(0,)][0]

        # Intersection: max(0, 0.2, 0.1) = 0.2, min(0.6, 0.8, 0.7) = 0.6
        # Union: min(0, 0.2, 0.1) = 0, max(0.6, 0.8, 0.7) = 0.8
        assert int_width < union_width

        # Check specific values
        assert abs(intersection[(0,)][0] - 0.2) < 0.001
        assert abs(intersection[(0,)][1] - 0.6) < 0.001
        assert abs(union[(0,)][0] - 0.0) < 0.001
        assert abs(union[(0,)][1] - 0.8) < 0.001

    def test_agreement_helps_identify_issues(self):
        """Test that agreement analysis helps identify instrument conflicts."""
        # Create instruments that disagree for some strata
        instrument_bounds = {
            'Z1': {
                (0,): (0.1, 0.4),  # Reasonable
                (1,): (0.6, 0.9),  # High
            },
            'Z2': {
                (0,): (0.15, 0.45),  # Similar to Z1
                (1,): (0.1, 0.3),  # Low - conflicts with Z1
            },
        }

        agreement = compute_instrument_agreement(instrument_bounds)

        # Stratum (0,) should have good agreement
        stratum_0 = agreement[agreement['stratum'] == "(0,)"]
        if len(stratum_0) > 0:
            assert stratum_0['overlap'].iloc[0] == True

        # Stratum (1,) should have conflict
        stratum_1 = agreement[agreement['stratum'] == "(1,)"]
        if len(stratum_1) > 0:
            assert stratum_1['overlap'].iloc[0] == False
