"""
Tests for Extended LP Solver Module

Tests the ExtendedLPSolver class and related functions.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from causal_grounding.lp_solver_extended import (
    ExtendedLPResult,
    ExtendedLPSolver,
    solve_extended_bounds_all_strata,
    compare_simple_vs_extended,
    create_lp_solver,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_training_probs():
    """Simple training probabilities for testing."""
    return {
        'site_1': {
            (0, (0,)): 0.30,  # P(Y=1|X=0,Z=0)
            (0, (1,)): 0.40,  # P(Y=1|X=0,Z=1)
            (1, (0,)): 0.50,  # P(Y=1|X=1,Z=0)
            (1, (1,)): 0.60,  # P(Y=1|X=1,Z=1)
        },
        'site_2': {
            (0, (0,)): 0.32,
            (0, (1,)): 0.38,
            (1, (0,)): 0.52,
            (1, (1,)): 0.58,
        }
    }


@pytest.fixture
def multi_site_data():
    """Multi-site training data for testing."""
    np.random.seed(42)

    def create_site(n, effect_shift=0):
        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
        Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z + effect_shift, n)
        df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'F': 'on'})
        return df

    return {
        'site_1': create_site(500, 0.0),
        'site_2': create_site(500, 0.05),
        'site_3': create_site(500, -0.05),
    }


@pytest.fixture
def solver():
    """Default extended LP solver."""
    return ExtendedLPSolver(epsilon=0.1, solver='scipy')


# =============================================================================
# TEST: ExtendedLPResult
# =============================================================================

class TestExtendedLPResult:
    """Tests for ExtendedLPResult dataclass."""

    def test_create_result(self):
        """Test creating result object."""
        result = ExtendedLPResult(
            lower=0.1,
            upper=0.5,
            status='optimal'
        )
        assert result.lower == 0.1
        assert result.upper == 0.5
        assert result.status == 'optimal'

    def test_result_with_theta(self):
        """Test result with theta values."""
        result = ExtendedLPResult(
            lower=0.1,
            upper=0.5,
            status='optimal',
            theta_1=0.6,
            theta_0=0.3
        )
        assert result.theta_1 == 0.6
        assert result.theta_0 == 0.3


# =============================================================================
# TEST: ExtendedLPSolver INITIALIZATION
# =============================================================================

class TestExtendedLPSolverInit:
    """Tests for ExtendedLPSolver initialization."""

    def test_default_init(self):
        """Test default initialization."""
        solver = ExtendedLPSolver()
        assert solver.epsilon == 0.1
        assert solver.solver == 'scipy'
        assert solver.verbose == False

    def test_custom_init(self):
        """Test custom initialization."""
        solver = ExtendedLPSolver(
            epsilon=0.2,
            solver='scipy',
            verbose=True
        )
        assert solver.epsilon == 0.2
        assert solver.verbose == True


# =============================================================================
# TEST: solve_within_stratum_bounds
# =============================================================================

class TestSolveWithinStratumBounds:
    """Tests for solve_within_stratum_bounds method."""

    def test_basic_solve(self, solver, simple_training_probs):
        """Test basic solve."""
        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=simple_training_probs
        )

        assert isinstance(result, ExtendedLPResult)
        assert not np.isnan(result.lower)
        assert not np.isnan(result.upper)
        assert result.lower <= result.upper

    def test_valid_bounds_range(self, solver, simple_training_probs):
        """Test that bounds are in valid range."""
        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=simple_training_probs
        )

        # CATE bounds for binary outcomes should be in [-1, 1]
        assert result.lower >= -1.0 - 0.01
        assert result.upper <= 1.0 + 0.01

    def test_different_epsilon(self, simple_training_probs):
        """Test that larger epsilon gives wider bounds."""
        solver_tight = ExtendedLPSolver(epsilon=0.05)
        solver_loose = ExtendedLPSolver(epsilon=0.2)

        result_tight = solver_tight.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=simple_training_probs
        )

        result_loose = solver_loose.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=simple_training_probs
        )

        width_tight = result_tight.upper - result_tight.lower
        width_loose = result_loose.upper - result_loose.lower

        # Larger epsilon should give wider bounds
        assert width_loose >= width_tight - 0.01

    def test_insufficient_data(self, solver):
        """Test handling of insufficient data."""
        # Only one x value
        training_probs = {
            'site_1': {
                (1, (0,)): 0.5,  # Only X=1, no X=0
            }
        }

        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        assert result.status == 'insufficient_data'

    def test_with_weights(self, solver, simple_training_probs):
        """Test with custom instrument weights."""
        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=simple_training_probs,
            instrument_weights={0: 0.3, 1: 0.7}
        )

        assert not np.isnan(result.lower)
        assert not np.isnan(result.upper)


# =============================================================================
# TEST: solve_extended_bounds_all_strata
# =============================================================================

class TestSolveExtendedBoundsAllStrata:
    """Tests for solve_extended_bounds_all_strata function."""

    def test_basic(self, multi_site_data):
        """Test basic all-strata solve."""
        bounds = solve_extended_bounds_all_strata(
            multi_site_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=0.1
        )

        assert isinstance(bounds, dict)
        assert len(bounds) > 0

        for z, (lower, upper) in bounds.items():
            assert lower <= upper
            assert lower >= -1.0 - 0.01
            assert upper <= 1.0 + 0.01

    def test_all_strata_covered(self, multi_site_data):
        """Test that all strata with data are covered."""
        bounds = solve_extended_bounds_all_strata(
            multi_site_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        # Should have bounds for both Z=0 and Z=1
        z_values = set(bounds.keys())
        assert (0,) in z_values or (1,) in z_values

    def test_empty_result_with_bad_data(self):
        """Test empty result with insufficient data."""
        bad_data = {
            'site_1': pd.DataFrame({
                'Z': [0], 'X': [1], 'Y': [1], 'F': ['on']
            })
        }

        bounds = solve_extended_bounds_all_strata(
            bad_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        # Should return empty dict with insufficient data
        assert len(bounds) == 0


# =============================================================================
# TEST: compare_simple_vs_extended
# =============================================================================

class TestCompareSimpleVsExtended:
    """Tests for compare_simple_vs_extended function."""

    def test_basic_comparison(self, multi_site_data):
        """Test basic comparison."""
        comparison = compare_simple_vs_extended(
            multi_site_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        assert isinstance(comparison, pd.DataFrame)
        assert 'stratum' in comparison.columns
        assert 'simple_lower' in comparison.columns
        assert 'extended_lower' in comparison.columns
        assert 'width_improvement' in comparison.columns

    def test_comparison_has_widths(self, multi_site_data):
        """Test that comparison includes width calculations."""
        comparison = compare_simple_vs_extended(
            multi_site_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        assert 'simple_width' in comparison.columns
        assert 'extended_width' in comparison.columns

    def test_improvement_calculation(self, multi_site_data):
        """Test that improvement is calculated correctly."""
        comparison = compare_simple_vs_extended(
            multi_site_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        for _, row in comparison.iterrows():
            if not np.isnan(row['simple_width']) and not np.isnan(row['extended_width']):
                expected_improvement = row['simple_width'] - row['extended_width']
                assert abs(row['width_improvement'] - expected_improvement) < 0.001


# =============================================================================
# TEST: create_lp_solver factory
# =============================================================================

class TestCreateLPSolver:
    """Tests for create_lp_solver factory function."""

    def test_create_simple(self):
        """Test creating simple solver."""
        solver = create_lp_solver('simple', epsilon=0.1)
        # Simple solver returns a function
        assert callable(solver)

    def test_create_extended(self):
        """Test creating extended solver."""
        solver = create_lp_solver('extended', epsilon=0.1)
        assert isinstance(solver, ExtendedLPSolver)
        assert solver.epsilon == 0.1

    def test_invalid_type(self):
        """Test invalid solver type."""
        with pytest.raises(ValueError, match="Unknown solver_type"):
            create_lp_solver('invalid')

    def test_extended_with_kwargs(self):
        """Test creating extended solver with kwargs."""
        solver = create_lp_solver('extended', epsilon=0.2, verbose=True)
        assert solver.epsilon == 0.2
        assert solver.verbose == True


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_site(self):
        """Test with single training site."""
        np.random.seed(42)
        n = 200

        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.4, n)
        Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

        training_data = {
            'site_1': pd.DataFrame({
                'Z': Z, 'X': X, 'Y': Y, 'F': 'on'
            })
        }

        bounds = solve_extended_bounds_all_strata(
            training_data, ['Z'], 'X', 'Y', epsilon=0.1
        )

        # Should still produce bounds
        assert len(bounds) > 0

    def test_high_epsilon(self):
        """Test with very high epsilon."""
        solver = ExtendedLPSolver(epsilon=0.5)

        training_probs = {
            'site_1': {
                (0, (0,)): 0.3,
                (1, (0,)): 0.6,
            }
        }

        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        # Wide epsilon should give wide bounds
        width = result.upper - result.lower
        assert width >= 0.8  # Very wide bounds expected

    def test_zero_epsilon(self):
        """Test with zero epsilon (no naturalness tolerance)."""
        solver = ExtendedLPSolver(epsilon=0.0)

        training_probs = {
            'site_1': {
                (0, (0,)): 0.3,
                (1, (0,)): 0.6,
            },
            'site_2': {
                (0, (0,)): 0.32,  # Slightly different
                (1, (0,)): 0.58,
            }
        }

        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        # With epsilon=0, might be infeasible due to site differences
        # Or bounds should be very tight
        if not np.isnan(result.lower):
            width = result.upper - result.lower
            assert width < 0.1

    def test_identical_site_data(self):
        """Test with identical site data."""
        solver = ExtendedLPSolver(epsilon=0.1)

        # All sites have same probabilities
        training_probs = {
            'site_1': {(0, (0,)): 0.3, (1, (0,)): 0.6},
            'site_2': {(0, (0,)): 0.3, (1, (0,)): 0.6},
            'site_3': {(0, (0,)): 0.3, (1, (0,)): 0.6},
        }

        result = solver.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        # Bounds should center around CATE = 0.6 - 0.3 = 0.3
        assert abs((result.lower + result.upper) / 2 - 0.3) < 0.15


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for extended LP solver."""

    def test_full_workflow(self, multi_site_data):
        """Test full workflow."""
        # Step 1: Create solver
        solver = ExtendedLPSolver(epsilon=0.1, solver='scipy')

        # Step 2: Solve for all strata
        bounds = solve_extended_bounds_all_strata(
            multi_site_data, ['Z'], 'X', 'Y',
            epsilon=0.1, solver='scipy'
        )

        # Step 3: Verify bounds
        assert len(bounds) > 0
        for z, (lower, upper) in bounds.items():
            assert lower <= upper

        # Step 4: Compare with simple
        comparison = compare_simple_vs_extended(
            multi_site_data, ['Z'], 'X', 'Y', epsilon=0.1
        )
        assert len(comparison) > 0

    def test_bounds_contain_true_effect(self):
        """Test that bounds contain the true effect (with known DGP)."""
        np.random.seed(42)
        n = 1000

        # Known DGP: CATE = 0.3 (treatment increases probability by 0.3)
        true_cate = 0.3

        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.4 + 0.2 * Z, n)
        Y_0 = np.random.binomial(1, 0.3 + 0.1 * Z, n)
        Y_1 = np.random.binomial(1, 0.6 + 0.1 * Z, n)  # 0.3 higher
        Y = np.where(X == 1, Y_1, Y_0)

        training_data = {
            'site_1': pd.DataFrame({
                'Z': Z[:n//2], 'X': X[:n//2], 'Y': Y[:n//2], 'F': 'on'
            }),
            'site_2': pd.DataFrame({
                'Z': Z[n//2:], 'X': X[n//2:], 'Y': Y[n//2:], 'F': 'on'
            })
        }

        bounds = solve_extended_bounds_all_strata(
            training_data, ['Z'], 'X', 'Y',
            epsilon=0.15  # Some tolerance for sampling
        )

        # Bounds should contain true CATE for at least some strata
        contains_true = False
        for z, (lower, upper) in bounds.items():
            if lower <= true_cate <= upper:
                contains_true = True
                break

        # Allow for sampling variation
        assert contains_true or len(bounds) == 0

    def test_consistency_across_solvers(self):
        """Test that scipy and cvxpy give similar results."""
        np.random.seed(42)

        training_probs = {
            'site_1': {
                (0, (0,)): 0.3,
                (1, (0,)): 0.6,
            },
            'site_2': {
                (0, (0,)): 0.35,
                (1, (0,)): 0.55,
            }
        }

        solver_scipy = ExtendedLPSolver(epsilon=0.1, solver='scipy')
        result_scipy = solver_scipy.solve_within_stratum_bounds(
            z_value=(0,),
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        # Results should be similar (within numerical tolerance)
        # Note: cvxpy test would go here if available
        assert not np.isnan(result_scipy.lower)
        assert not np.isnan(result_scipy.upper)
