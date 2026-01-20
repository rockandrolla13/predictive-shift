"""
Tests for LOCO CI Testing Module

Tests the LOCOCIEngine class for conditional independence testing using
the Leave-One-Covariate-Out predictive comparison method.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from causal_grounding.ci_tests_loco import LOCOCIEngine
from causal_grounding.ci_tests_l1 import create_ci_engine


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
def sample_null_true_data():
    """Data where Y is independent of X given W."""
    np.random.seed(42)
    n = 500

    # W affects Y, but X does not affect Y given W
    W1 = np.random.choice(['A', 'B', 'C'], size=n)
    W2 = np.random.choice([0, 1], size=n)
    X = np.random.binomial(1, 0.5, n)

    # Y depends only on W, not X
    W1_effect = (W1 == 'A').astype(int) * 0.5
    W2_effect = W2 * 0.3
    logit_Y = -0.5 + W1_effect + W2_effect
    prob_Y = 1 / (1 + np.exp(-logit_Y))
    Y = np.random.binomial(1, prob_Y)

    return pd.DataFrame({'W1': W1, 'W2': W2, 'X': X, 'Y': Y})


@pytest.fixture
def sample_null_false_data():
    """Data where Y depends on X given W."""
    np.random.seed(42)
    n = 500

    # W and X both affect Y
    W1 = np.random.choice(['A', 'B', 'C'], size=n)
    W2 = np.random.choice([0, 1], size=n)
    X = np.random.binomial(1, 0.5, n)

    # Y depends on W and X
    W1_effect = (W1 == 'A').astype(int) * 0.5
    W2_effect = W2 * 0.3
    logit_Y = -0.5 + W1_effect + W2_effect + 1.5 * X  # Strong X effect
    prob_Y = 1 / (1 + np.exp(-logit_Y))
    Y = np.random.binomial(1, prob_Y)

    return pd.DataFrame({'W1': W1, 'W2': W2, 'X': X, 'Y': Y})


@pytest.fixture
def sample_multivariate_data():
    """Sample data with multiple covariates."""
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    Z3 = np.random.binomial(1, 0.4, n)  # Independent of Y

    X = np.random.binomial(1, 0.2 + 0.3 * Z1 + 0.2 * Z2, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.2 * Z1, n)

    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
    })


@pytest.fixture
def loco_engine_gbm():
    """Default LOCO CI engine with GBM."""
    return LOCOCIEngine(
        function_class='gbm',
        test_alpha=0.05,
        n_estimators=50,  # Reduced for faster tests
        max_depth=2,
        random_state=42
    )


# =============================================================================
# TEST: LOCOICIENGINE INITIALIZATION
# =============================================================================

class TestLOCOCIEngineInit:
    """Tests for LOCOCIEngine initialization."""

    def test_default_init(self):
        """Test default initialization."""
        engine = LOCOCIEngine()
        assert engine.function_class == 'gbm'
        assert engine.test_prop == 0.3
        assert engine.test_alpha == 0.05
        assert engine.cv_folds == 5
        assert engine.n_estimators == 100
        assert engine.learning_rate == 0.1
        assert engine.max_depth == 3

    def test_custom_init(self):
        """Test custom initialization."""
        engine = LOCOCIEngine(
            function_class='gbm',
            test_prop=0.4,
            test_alpha=0.1,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=123
        )
        assert engine.function_class == 'gbm'
        assert engine.test_prop == 0.4
        assert engine.test_alpha == 0.1
        assert engine.n_estimators == 200
        assert engine.learning_rate == 0.05
        assert engine.max_depth == 4
        assert engine.random_state == 123

    def test_invalid_function_class(self):
        """Test that invalid function_class raises error."""
        with pytest.raises(ValueError, match="function_class must be"):
            LOCOCIEngine(function_class='invalid')


# =============================================================================
# TEST: CONDITIONAL INDEPENDENCE TESTING
# =============================================================================

class TestConditionalIndependence:
    """Tests for conditional independence testing."""

    def test_basic_ci_test(self, loco_engine_gbm, sample_binary_data):
        """Test basic conditional independence test returns expected keys."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        # Check all expected keys present
        assert 'cmi' in result
        assert 'p_value' in result
        assert 'reject_independence' in result
        assert 'f0_loss' in result
        assert 'f1_loss' in result
        assert 'test_statistic' in result
        assert 'loss_reduction' in result
        assert 'n_test' in result

    def test_cmi_non_negative(self, loco_engine_gbm, sample_binary_data):
        """Test that CMI proxy is non-negative."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        assert result['cmi'] >= 0

    def test_p_value_range(self, loco_engine_gbm, sample_binary_data):
        """Test that p-value is in valid range."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        assert 0 <= result['p_value'] <= 1

    def test_null_true_high_pvalue(self, loco_engine_gbm, sample_null_true_data):
        """Test that when null is true, p-value tends to be high."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_null_true_data, 'X', 'Y', ['W1', 'W2']
        )

        # Should NOT reject (p-value > alpha)
        # This may occasionally fail due to randomness, so we use a relaxed threshold
        assert result['p_value'] > 0.01  # Very liberal threshold

    def test_null_false_low_pvalue(self, loco_engine_gbm, sample_null_false_data):
        """Test that when null is false (strong effect), p-value tends to be low."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_null_false_data, 'X', 'Y', ['W1', 'W2']
        )

        # Should reject (p-value < alpha)
        # With strong effect (coef=1.5), this should be detected
        assert result['p_value'] < 0.2  # Liberal threshold for finite samples

    def test_empty_conditioning_set(self, loco_engine_gbm, sample_binary_data):
        """Test with empty conditioning set."""
        result = loco_engine_gbm.test_conditional_independence(
            sample_binary_data, 'X', 'Y', []
        )

        assert 'cmi' in result
        assert 'p_value' in result
        assert result['n_test'] > 0


# =============================================================================
# TEST: EHS SCORING
# =============================================================================

class TestEHSScoring:
    """Tests for EHS criteria scoring."""

    def test_ehs_score_keys(self, loco_engine_gbm, sample_binary_data):
        """Test that EHS scoring returns all expected keys."""
        result = loco_engine_gbm.score_ehs_criteria(
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

    def test_ehs_z_a_returned(self, loco_engine_gbm, sample_binary_data):
        """Test that z_a is correctly returned."""
        result = loco_engine_gbm.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )
        assert result['z_a'] == 'Z'

    def test_ehs_cmi_non_negative(self, loco_engine_gbm, sample_binary_data):
        """Test that all CMI values are non-negative."""
        result = loco_engine_gbm.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        assert result['test_i_cmi'] >= 0
        assert result['test_ii_cmi'] >= 0
        assert result['test_ia_cmi'] >= 0

    def test_ehs_pvalues_in_range(self, loco_engine_gbm, sample_binary_data):
        """Test that all p-values are in valid range."""
        result = loco_engine_gbm.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        assert 0 <= result['test_i_pvalue'] <= 1
        assert 0 <= result['test_ii_pvalue'] <= 1
        assert 0 <= result['test_ia_pvalue'] <= 1

    def test_ehs_booleans(self, loco_engine_gbm, sample_binary_data):
        """Test that boolean fields are actual booleans."""
        result = loco_engine_gbm.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        assert isinstance(result['test_i_reject'], bool)
        assert isinstance(result['test_ii_reject'], bool)
        assert isinstance(result['test_ia_reject'], bool)
        assert isinstance(result['passes_ehs'], bool)
        assert isinstance(result['passes_full_ehs'], bool)
        assert isinstance(result['weak_instrument_warning'], bool)

    def test_ehs_with_conditioning_set(self, loco_engine_gbm, sample_multivariate_data):
        """Test EHS scoring with non-empty conditioning set."""
        result = loco_engine_gbm.score_ehs_criteria(
            sample_multivariate_data,
            z_a='Z2',
            z_b=['Z1', 'Z3'],
            treatment='X',
            outcome='Y'
        )

        # Should return valid results
        assert 'score' in result
        assert isinstance(result['score'], float)


# =============================================================================
# TEST: INTERFACE COMPATIBILITY
# =============================================================================

class TestInterfaceCompatibility:
    """Tests for interface compatibility with CITestEngine."""

    def test_factory_function_loco(self):
        """Test that factory function can create LOCO engine."""
        engine = create_ci_engine('loco', function_class='gbm', random_state=42)
        assert isinstance(engine, LOCOCIEngine)

    def test_factory_function_invalid(self):
        """Test that factory function raises error for invalid method."""
        with pytest.raises(ValueError):
            create_ci_engine('invalid_method')

    def test_compatible_with_cmi_interface(self, sample_binary_data):
        """Test that LOCO returns same keys as CMI engine."""
        loco_engine = create_ci_engine('loco', function_class='gbm', random_state=42)

        result = loco_engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        # Must have these keys for compatibility
        assert 'cmi' in result
        assert 'p_value' in result
        assert 'reject_independence' in result

    def test_ehs_compatible_with_cmi_interface(self, sample_binary_data):
        """Test that LOCO EHS scoring returns same keys as CMI engine."""
        loco_engine = create_ci_engine('loco', function_class='gbm', random_state=42)

        result = loco_engine.score_ehs_criteria(
            sample_binary_data,
            z_a='Z',
            z_b=[],
            treatment='X',
            outcome='Y'
        )

        # Must have these keys for compatibility with covariate_scoring
        required_keys = [
            'z_a', 'test_i_cmi', 'test_i_pvalue', 'test_i_reject',
            'test_ii_cmi', 'test_ii_pvalue', 'test_ii_reject',
            'test_ia_cmi', 'test_ia_pvalue', 'test_ia_reject',
            'passes_ehs', 'passes_full_ehs', 'weak_instrument_warning', 'score'
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"


# =============================================================================
# TEST: COMPUTE CMI ONLY
# =============================================================================

class TestComputeCMIOnly:
    """Tests for compute_cmi_only method."""

    def test_compute_cmi_only_returns_float(self, loco_engine_gbm, sample_binary_data):
        """Test that compute_cmi_only returns a float."""
        result = loco_engine_gbm.compute_cmi_only(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        assert isinstance(result, float)
        assert result >= 0


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_sample_size(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 50  # Small sample

        X = np.random.binomial(1, 0.5, n)
        Y = np.random.binomial(1, 0.5, n)
        W = np.random.binomial(1, 0.5, n)

        df = pd.DataFrame({'X': X, 'Y': Y, 'W': W})

        engine = LOCOCIEngine(
            function_class='gbm',
            test_prop=0.3,
            n_estimators=20,
            random_state=42
        )

        # Should not crash
        result = engine.test_conditional_independence(df, 'X', 'Y', ['W'])
        assert 'p_value' in result

    def test_imbalanced_outcome(self):
        """Test with imbalanced outcome."""
        np.random.seed(42)
        n = 200

        X = np.random.binomial(1, 0.5, n)
        W = np.random.binomial(1, 0.5, n)
        # Highly imbalanced Y
        Y = np.random.binomial(1, 0.1, n)

        df = pd.DataFrame({'X': X, 'Y': Y, 'W': W})

        engine = LOCOCIEngine(
            function_class='gbm',
            test_prop=0.3,
            n_estimators=20,
            random_state=42
        )

        # Should not crash
        result = engine.test_conditional_independence(df, 'X', 'Y', ['W'])
        assert 'p_value' in result

    def test_many_categories(self):
        """Test with many categorical levels."""
        np.random.seed(42)
        n = 300

        X = np.random.binomial(1, 0.5, n)
        Y = np.random.binomial(1, 0.5, n)
        W = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], n)

        df = pd.DataFrame({'X': X, 'Y': Y, 'W': W})

        engine = LOCOCIEngine(
            function_class='gbm',
            n_estimators=20,
            random_state=42
        )

        result = engine.test_conditional_independence(df, 'X', 'Y', ['W'])
        assert 'p_value' in result

    def test_reproducibility(self, sample_binary_data):
        """Test that results are reproducible with same random_state."""
        engine1 = LOCOCIEngine(function_class='gbm', random_state=42)
        engine2 = LOCOCIEngine(function_class='gbm', random_state=42)

        result1 = engine1.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        result2 = engine2.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )

        assert result1['p_value'] == result2['p_value']
        assert result1['cmi'] == result2['cmi']


# =============================================================================
# TEST: GBM VS LASSO (if lasso available)
# =============================================================================

class TestFunctionClasses:
    """Tests for different function classes."""

    def test_gbm_runs(self, sample_binary_data):
        """Test that GBM function class runs."""
        engine = LOCOCIEngine(
            function_class='gbm',
            n_estimators=20,
            random_state=42
        )
        result = engine.test_conditional_independence(
            sample_binary_data, 'Z', 'Y', ['X']
        )
        assert 'p_value' in result

    @pytest.mark.skipif(
        not LOCOCIEngine(function_class='gbm').function_class == 'gbm',
        reason="Skipping lasso test (group-lasso may not be installed)"
    )
    def test_lasso_if_available(self, sample_binary_data):
        """Test that lasso function class runs if available."""
        try:
            engine = LOCOCIEngine(
                function_class='lasso',
                cv_folds=3,
                random_state=42
            )
            result = engine.test_conditional_independence(
                sample_binary_data, 'Z', 'Y', ['X']
            )
            assert 'p_value' in result
        except ImportError:
            pytest.skip("group-lasso not installed")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
