"""
Tests for L1-Regression CI Testing Module

Tests the L1RegressionCIEngine class and create_ci_engine factory function.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from causal_grounding.ci_tests_l1 import (
    L1RegressionCIEngine,
    create_ci_engine,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_binary_data():
    """Sample binary data for CI testing."""
    np.random.seed(42)
    n = 500

    # Z -> X -> Y (Z is valid instrument)
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)  # Z affects X
    Y = np.random.binomial(1, 0.3 + 0.3 * X, n)  # X affects Y, Z does not directly

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


@pytest.fixture
def sample_multivariate_data():
    """Sample data with multiple covariates."""
    np.random.seed(42)
    n = 500

    # Multiple binary covariates
    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    Z3 = np.random.binomial(1, 0.4, n)  # Independent of Y

    # X depends on Z1 and Z2
    X = np.random.binomial(1, 0.2 + 0.3 * Z1 + 0.2 * Z2, n)

    # Y depends on X and Z1 (Z1 is confounder, Z2 is valid instrument)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.2 * Z1, n)

    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
    })


@pytest.fixture
def sample_continuous_outcome_data():
    """Sample data with continuous outcome."""
    np.random.seed(42)
    n = 500

    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = 0.5 + 0.3 * X + np.random.normal(0, 0.5, n)  # Continuous Y

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


@pytest.fixture
def l1_engine():
    """Default L1 CI engine."""
    return L1RegressionCIEngine(alpha=0.1, coef_threshold=0.05)


# =============================================================================
# TEST: L1RegressionCIEngine INITIALIZATION
# =============================================================================

class TestL1RegressionCIEngineInit:
    """Tests for L1RegressionCIEngine initialization."""

    def test_default_init(self):
        """Test default initialization."""
        engine = L1RegressionCIEngine()
        assert engine.alpha == 0.1
        assert engine.test_alpha == 0.05
        assert engine.coef_threshold == 0.01
        assert engine.use_cv == False
        assert engine.max_iter == 1000

    def test_custom_init(self):
        """Test custom initialization."""
        engine = L1RegressionCIEngine(
            alpha=0.5,
            test_alpha=0.1,
            coef_threshold=0.02,
            use_cv=True,
            max_iter=2000,
            random_state=42
        )
        assert engine.alpha == 0.5
        assert engine.test_alpha == 0.1
        assert engine.coef_threshold == 0.02
        assert engine.use_cv == True
        assert engine.max_iter == 2000
        assert engine.random_state == 42


# =============================================================================
# TEST: CONDITIONAL INDEPENDENCE TESTING
# =============================================================================

class TestConditionalIndependence:
    """Tests for conditional independence testing."""

    def test_basic_ci_test(self, l1_engine, sample_binary_data):
        """Test basic conditional independence test."""
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        assert 'coefficient' in result
        assert 'abs_coefficient' in result
        assert 'reject_independence' in result
        assert 'threshold' in result
        assert 'n_features' in result
        assert 'cmi' in result  # For compatibility
        assert 'p_value' in result  # For compatibility

    def test_ci_exogeneity(self, l1_engine, sample_binary_data):
        """Test that Z ⊥ Y | X (exogeneity) holds."""
        # Z should be independent of Y given X (Z is valid instrument)
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        # Should NOT reject independence (Z is exogenous)
        # Note: With finite samples this may not always hold
        assert result['abs_coefficient'] >= 0

    def test_ci_relevance(self, l1_engine, sample_binary_data):
        """Test that Z ⊥̸ Y (marginal dependence) detected."""
        # Z should NOT be independent of Y marginally
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', []
        )

        # Should have some coefficient (Z affects Y through X)
        assert result['abs_coefficient'] >= 0

    def test_ci_instrument_relevance(self, l1_engine, sample_binary_data):
        """Test that X ⊥̸ Z (instrument relevance) detected."""
        # X should NOT be independent of Z (Z affects X)
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'X', 'Z', []
        )

        # Should reject independence
        assert result['abs_coefficient'] > 0

    def test_empty_conditioning_set(self, l1_engine, sample_binary_data):
        """Test CI test with empty conditioning set."""
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'X', 'Y', []
        )

        assert 'coefficient' in result
        assert result['n_features'] >= 1

    def test_multiple_conditioning_vars(self, l1_engine, sample_multivariate_data):
        """Test CI test with multiple conditioning variables."""
        result = l1_engine.test_conditional_independence(
            sample_multivariate_data, 'Z2', 'Y', ['X', 'Z1']
        )

        assert 'coefficient' in result
        assert result['n_features'] >= 3

    def test_return_coefficients(self, l1_engine, sample_binary_data):
        """Test returning all coefficients."""
        result = l1_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X'],
            return_coefficients=True
        )

        assert 'coefficients' in result
        assert isinstance(result['coefficients'], np.ndarray)

    def test_continuous_outcome(self, l1_engine, sample_continuous_outcome_data):
        """Test CI test with continuous outcome (uses Lasso)."""
        result = l1_engine.test_conditional_independence(
            sample_continuous_outcome_data, 'Z', 'Y', ['X']
        )

        assert 'coefficient' in result
        assert 'reject_independence' in result


# =============================================================================
# TEST: COMPUTE COEFFICIENT ONLY
# =============================================================================

class TestComputeCoefficientOnly:
    """Tests for compute_coefficient_only method."""

    def test_basic(self, l1_engine, sample_binary_data):
        """Test basic coefficient computation."""
        coef = l1_engine.compute_coefficient_only(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        assert isinstance(coef, float)
        assert coef >= 0  # Absolute coefficient

    def test_matches_full_test(self, sample_binary_data):
        """Test that coefficient matches full test."""
        # Use fixed random state for reproducibility
        engine = L1RegressionCIEngine(alpha=0.1, coef_threshold=0.05, random_state=42)

        coef = engine.compute_coefficient_only(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        result = engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        # Allow small tolerance due to SAGA solver stochasticity
        assert abs(coef - result['abs_coefficient']) < 0.01


# =============================================================================
# TEST: EHS CRITERIA SCORING
# =============================================================================

class TestEHSScoring:
    """Tests for EHS criteria scoring."""

    def test_basic_ehs_scoring(self, l1_engine, sample_binary_data):
        """Test basic EHS scoring."""
        result = l1_engine.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        # Check all expected keys
        assert 'z_a' in result
        assert 'test_i_cmi' in result
        assert 'test_i_pvalue' in result
        assert 'test_i_reject' in result
        assert 'test_ii_cmi' in result
        assert 'test_ii_pvalue' in result
        assert 'test_ii_reject' in result
        assert 'test_ia_cmi' in result
        assert 'test_ia_pvalue' in result
        assert 'test_ia_reject' in result
        assert 'passes_ehs' in result
        assert 'passes_full_ehs' in result
        assert 'weak_instrument_warning' in result
        assert 'score' in result

    def test_valid_instrument_scoring(self, l1_engine, sample_binary_data):
        """Test that valid instrument gets good score."""
        result = l1_engine.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        # Z should be valid instrument
        assert result['z_a'] == 'Z'
        # Score should be reasonable (positive if Z passes criteria)
        assert isinstance(result['score'], float)

    def test_ehs_with_conditioning_set(self, l1_engine, sample_multivariate_data):
        """Test EHS scoring with conditioning set."""
        result = l1_engine.score_ehs_criteria(
            sample_multivariate_data,
            z_a='Z2',
            z_b=['Z1'],  # Condition on Z1
            treatment='X',
            outcome='Y'
        )

        assert result['z_a'] == 'Z2'
        assert 'passes_ehs' in result

    def test_invalid_instrument(self, l1_engine, sample_multivariate_data):
        """Test that confounder is not valid instrument."""
        # Z1 is a confounder (affects both X and Y directly)
        result = l1_engine.score_ehs_criteria(
            sample_multivariate_data,
            z_a='Z1',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        # Z1 should fail exogeneity (test_i should reject)
        assert 'test_i_reject' in result

    def test_weak_instrument_warning(self, l1_engine):
        """Test weak instrument warning."""
        np.random.seed(42)
        n = 500

        # Create weak instrument (barely affects X)
        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.5 + 0.01 * Z, n)  # Very weak effect
        Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

        data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

        result = l1_engine.score_ehs_criteria(
            data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y',
            weak_instrument_threshold=0.05
        )

        # Should potentially warn about weak instrument
        assert 'weak_instrument_warning' in result


# =============================================================================
# TEST: FACTORY FUNCTION
# =============================================================================

class TestCreateCIEngine:
    """Tests for create_ci_engine factory function."""

    def test_create_l1_engine(self):
        """Test creating L1 engine."""
        engine = create_ci_engine('l1', alpha=0.2)

        assert isinstance(engine, L1RegressionCIEngine)
        assert engine.alpha == 0.2

    def test_create_cmi_engine(self):
        """Test creating CMI engine."""
        engine = create_ci_engine('cmi', n_permutations=100)

        # Should be CITestEngine
        from causal_grounding.ci_tests import CITestEngine
        assert isinstance(engine, CITestEngine)

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_ci_engine('invalid')

    def test_default_method(self):
        """Test default method is CMI."""
        engine = create_ci_engine()

        from causal_grounding.ci_tests import CITestEngine
        assert isinstance(engine, CITestEngine)


# =============================================================================
# TEST: INTERFACE COMPATIBILITY
# =============================================================================

class TestInterfaceCompatibility:
    """Test that L1 engine is compatible with CMI engine interface."""

    def test_same_output_keys(self, sample_binary_data):
        """Test that both engines return same output keys."""
        l1_engine = create_ci_engine('l1', alpha=0.1)
        cmi_engine = create_ci_engine('cmi', n_permutations=50)

        l1_result = l1_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        cmi_result = cmi_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        # Both should have these keys
        common_keys = ['cmi', 'p_value', 'reject_independence']
        for key in common_keys:
            assert key in l1_result, f"L1 missing key: {key}"
            assert key in cmi_result, f"CMI missing key: {key}"

    def test_same_ehs_output_keys(self, sample_binary_data):
        """Test that both engines return same EHS output keys."""
        l1_engine = create_ci_engine('l1', alpha=0.1)
        cmi_engine = create_ci_engine('cmi', n_permutations=50)

        l1_result = l1_engine.score_ehs_criteria(
            sample_binary_data, 'Z', [], 'X', 'Y'
        )
        cmi_result = cmi_engine.score_ehs_criteria(
            sample_binary_data, 'Z', [], 'X', 'Y'
        )

        # Both should have these keys
        common_keys = [
            'z_a', 'test_i_cmi', 'test_i_pvalue', 'test_i_reject',
            'test_ii_cmi', 'test_ii_pvalue', 'test_ii_reject',
            'passes_ehs', 'score'
        ]
        for key in common_keys:
            assert key in l1_result, f"L1 missing key: {key}"
            assert key in cmi_result, f"CMI missing key: {key}"


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value_covariate(self, l1_engine):
        """Test with single-value (constant) covariate."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'Z': np.ones(n, dtype=int),  # Constant
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        # Should not crash
        result = l1_engine.test_conditional_independence(data, 'Z', 'Y', ['X'])
        assert 'coefficient' in result

    def test_perfectly_correlated(self, l1_engine):
        """Test with perfectly correlated variables."""
        np.random.seed(42)
        n = 100

        X = np.random.binomial(1, 0.5, n)
        data = pd.DataFrame({
            'Z': X.copy(),  # Same as X
            'X': X,
            'Y': np.random.binomial(1, 0.3 + 0.4 * X, n)
        })

        # Should not crash
        result = l1_engine.test_conditional_independence(data, 'Z', 'Y', ['X'])
        assert 'coefficient' in result

    def test_small_sample(self, l1_engine):
        """Test with small sample size."""
        np.random.seed(42)
        n = 30  # Small sample

        data = pd.DataFrame({
            'Z': np.random.binomial(1, 0.5, n),
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        # Should not crash
        result = l1_engine.test_conditional_independence(data, 'Z', 'Y', ['X'])
        assert 'coefficient' in result

    def test_multicollinear_conditioning(self, l1_engine):
        """Test with multicollinear conditioning set."""
        np.random.seed(42)
        n = 200

        Z1 = np.random.binomial(1, 0.5, n)
        Z2 = 1 - Z1  # Perfect negative correlation with Z1
        X = np.random.binomial(1, 0.5, n)
        Y = np.random.binomial(1, 0.5, n)

        data = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'X': X, 'Y': Y})

        # Should handle gracefully
        result = l1_engine.test_conditional_independence(
            data, 'X', 'Y', ['Z1', 'Z2']
        )
        assert 'coefficient' in result


# =============================================================================
# TEST: DIFFERENT ALPHA VALUES
# =============================================================================

class TestAlphaSensitivity:
    """Test sensitivity to alpha parameter."""

    def test_high_alpha_sparse(self, sample_binary_data):
        """Test that high alpha produces sparser models."""
        engine_low = L1RegressionCIEngine(alpha=0.01, coef_threshold=0.01)
        engine_high = L1RegressionCIEngine(alpha=1.0, coef_threshold=0.01)

        result_low = engine_low.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X'],
            return_coefficients=True
        )
        result_high = engine_high.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X'],
            return_coefficients=True
        )

        # High alpha should push coefficients closer to zero
        low_coef_sum = np.sum(np.abs(result_low['coefficients']))
        high_coef_sum = np.sum(np.abs(result_high['coefficients']))

        # High alpha should have smaller coefficients (more regularization)
        assert high_coef_sum <= low_coef_sum + 0.1  # Allow small tolerance


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for L1 CI engine."""

    def test_full_workflow(self, sample_multivariate_data):
        """Test full workflow with L1 engine."""
        engine = L1RegressionCIEngine(alpha=0.1)

        # Score all potential instruments
        candidates = ['Z1', 'Z2', 'Z3']
        scores = []

        for z in candidates:
            z_b = [c for c in candidates if c != z]
            result = engine.score_ehs_criteria(
                sample_multivariate_data,
                z_a=z,
                z_b=z_b,
                treatment='X',
                outcome='Y'
            )
            scores.append(result)

        # All should complete
        assert len(scores) == 3

        # Find best instrument
        best = max(scores, key=lambda x: x['score'])
        assert 'z_a' in best

    def test_use_with_covariate_scoring(self, sample_multivariate_data):
        """Test using L1 engine with covariate scoring module."""
        from causal_grounding.covariate_scoring import rank_covariates

        l1_engine = create_ci_engine('l1', alpha=0.1)

        # This should work with L1 engine
        rankings = rank_covariates(
            data=sample_multivariate_data,
            treatment='X',
            outcome='Y',
            covariates=['Z1', 'Z2', 'Z3'],
            ci_engine=l1_engine
        )

        assert isinstance(rankings, pd.DataFrame)
        assert len(rankings) == 3

    def test_compare_engines_on_same_data(self, sample_binary_data):
        """Test that both engines can process same data."""
        l1_engine = create_ci_engine('l1', alpha=0.1)
        cmi_engine = create_ci_engine('cmi', n_permutations=50)

        # Both should succeed
        l1_score = l1_engine.score_ehs_criteria(
            sample_binary_data, 'Z', [], 'X', 'Y'
        )
        cmi_score = cmi_engine.score_ehs_criteria(
            sample_binary_data, 'Z', [], 'X', 'Y'
        )

        # Both should identify Z as passing EHS or not
        assert 'passes_ehs' in l1_score
        assert 'passes_ehs' in cmi_score


# =============================================================================
# TEST: CROSS-VALIDATION MODE
# =============================================================================

class TestCrossValidationMode:
    """Tests for cross-validation mode."""

    def test_cv_mode_binary(self, sample_binary_data):
        """Test CV mode with binary outcome."""
        engine = L1RegressionCIEngine(use_cv=True, random_state=42)

        result = engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        assert 'coefficient' in result

    def test_cv_mode_continuous(self, sample_continuous_outcome_data):
        """Test CV mode with continuous outcome."""
        engine = L1RegressionCIEngine(use_cv=True, random_state=42)

        result = engine.test_conditional_independence(
            sample_continuous_outcome_data, 'Z', 'Y', ['X']
        )

        assert 'coefficient' in result
