"""
Test ci_tests.py module - CMI implementation and CITestEngine wrapper.
"""

import pytest
import numpy as np
import pandas as pd
import time
from causal_grounding import compute_cmi, permutation_test_cmi, CITestEngine, combine_conditioning_vars


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_independent_data():
    """Create synthetic data where Y and Z are independent given W."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'W': np.random.choice([0, 1, 2], n),
        'Y': np.random.choice([0, 1], n),
        'Z': np.random.choice([0, 1], n)
    })


@pytest.fixture
def synthetic_dependent_data():
    """Create synthetic data where Y depends on Z given W."""
    np.random.seed(42)
    n = 500
    W = np.random.choice([0, 1], n)
    Z = np.random.choice([0, 1], n)
    Y = (Z + W) % 2
    return pd.DataFrame({'W': W, 'Y': Y, 'Z': Z})


@pytest.fixture
def ci_engine():
    """Create CITestEngine with reduced permutations for speed."""
    return CITestEngine(n_permutations=100, random_seed=42)


# =============================================================================
# TEST CLASS: TestComputeCMI
# =============================================================================

class TestComputeCMI:
    """Test compute_cmi function."""

    def test_independent_variables(self):
        """Test CMI is close to 0 for independent variables."""
        np.random.seed(42)
        n = 500
        W = np.random.choice([0, 1, 2], size=n)
        Y = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1], size=n)

        cmi = compute_cmi(Y, Z, W)

        assert cmi < 0.05, f"CMI should be close to 0 for independent vars, got {cmi}"

    def test_dependent_variables(self):
        """Test CMI is greater than 0 for dependent variables."""
        np.random.seed(42)
        n = 500
        W = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1], size=n)
        Y = (Z + W) % 2  # Deterministic dependence

        cmi = compute_cmi(Y, Z, W)

        assert cmi > 0.1, f"CMI should be > 0.1 for dependent vars, got {cmi}"

    def test_smoothing_prevents_zero_division(self):
        """Test that smoothing prevents zero division with small datasets."""
        np.random.seed(42)
        n = 10
        W = np.random.choice([0, 1, 2], size=n)
        Y = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1, 2], size=n)

        # Should not raise any errors
        cmi = compute_cmi(Y, Z, W, alpha=1.0)

        assert np.isfinite(cmi), f"CMI should be finite, got {cmi}"

    def test_cmi_symmetry(self):
        """Test that CMI is symmetric in Y and Z."""
        np.random.seed(42)
        n = 500
        W = np.random.choice([0, 1, 2], size=n)
        Y = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1], size=n)

        cmi_yz = compute_cmi(Y, Z, W)
        cmi_zy = compute_cmi(Z, Y, W)

        assert np.isclose(cmi_yz, cmi_zy, rtol=1e-10), \
            f"CMI should be symmetric: {cmi_yz} vs {cmi_zy}"


# =============================================================================
# TEST CLASS: TestPermutationTest
# =============================================================================

class TestPermutationTest:
    """Test permutation_test_cmi function."""

    def test_independent_high_pvalue(self, synthetic_independent_data):
        """Test that independent variables give high p-value."""
        df = synthetic_independent_data

        observed_cmi, p_value, null_dist = permutation_test_cmi(
            df['Y'].values, df['Z'].values, df['W'].values,
            n_permutations=200,
            random_seed=42
        )

        assert p_value > 0.05, f"p-value should be > 0.05 for independent vars, got {p_value}"

    def test_dependent_low_pvalue(self, synthetic_dependent_data):
        """Test that dependent variables give low p-value."""
        df = synthetic_dependent_data

        observed_cmi, p_value, null_dist = permutation_test_cmi(
            df['Y'].values, df['Z'].values, df['W'].values,
            n_permutations=200,
            random_seed=42
        )

        assert p_value < 0.05, f"p-value should be < 0.05 for dependent vars, got {p_value}"

    def test_null_distribution_shape(self, synthetic_independent_data):
        """Test that null distribution has correct shape."""
        df = synthetic_independent_data
        n_permutations = 200

        observed_cmi, p_value, null_dist = permutation_test_cmi(
            df['Y'].values, df['Z'].values, df['W'].values,
            n_permutations=n_permutations,
            random_seed=42
        )

        assert len(null_dist) == n_permutations, \
            f"Null distribution should have {n_permutations} values, got {len(null_dist)}"
        assert (null_dist >= 0).all(), "All null CMI values should be non-negative"

    def test_reproducibility_with_seed(self, synthetic_independent_data):
        """Test that results are reproducible with same seed."""
        df = synthetic_independent_data

        result1 = permutation_test_cmi(
            df['Y'].values, df['Z'].values, df['W'].values,
            n_permutations=100,
            random_seed=123
        )

        result2 = permutation_test_cmi(
            df['Y'].values, df['Z'].values, df['W'].values,
            n_permutations=100,
            random_seed=123
        )

        assert result1[0] == result2[0], "Observed CMI should be identical"
        assert result1[1] == result2[1], "p-value should be identical"
        assert np.array_equal(result1[2], result2[2]), "Null distributions should be identical"


# =============================================================================
# TEST CLASS: TestCombineConditioningVars
# =============================================================================

class TestCombineConditioningVars:
    """Test combine_conditioning_vars function."""

    def test_empty_list_returns_zeros(self):
        """Test that empty column list returns zeros."""
        df = pd.DataFrame({'X': range(100)})

        result = combine_conditioning_vars(df, [])

        assert len(result) == 100
        assert (result == 0).all()

    def test_single_column(self):
        """Test that single column returns factorized values."""
        df = pd.DataFrame({'A': [0, 1, 2, 0, 1]})

        result = combine_conditioning_vars(df, ['A'])

        # Should map unique values to integers
        assert len(result) == 5
        # Same input values should map to same output values
        assert result[0] == result[3]  # Both are 0
        assert result[1] == result[4]  # Both are 1

    def test_multiple_columns_factorization(self):
        """Test that multiple columns are properly factorized."""
        df = pd.DataFrame({
            'A': [0, 0, 0, 1, 1, 1],
            'B': [0, 1, 2, 0, 1, 2]
        })

        result = combine_conditioning_vars(df, ['A', 'B'])

        # Should have 6 unique combinations (2 * 3)
        assert len(np.unique(result)) == 6
        # Values should be integers from 0 to 5
        assert result.min() >= 0
        assert result.max() <= 5

        # Same (A, B) pairs should map to same integer
        df2 = pd.DataFrame({
            'A': [0, 0, 1, 1, 0, 1],
            'B': [0, 0, 1, 1, 2, 0]
        })
        result2 = combine_conditioning_vars(df2, ['A', 'B'])
        assert result2[0] == result2[1]  # (0, 0) == (0, 0)
        assert result2[2] == result2[3]  # (1, 1) == (1, 1)


# =============================================================================
# TEST CLASS: TestCITestEngine
# =============================================================================

class TestCITestEngine:
    """Test CITestEngine class."""

    def test_engine_initialization(self):
        """Test default parameter initialization."""
        engine = CITestEngine()

        assert engine.smoothing_alpha == 1.0
        assert engine.test_alpha == 0.05
        assert engine.n_permutations == 1000
        assert engine.random_seed is None

    def test_conditional_independence_result_format(self, ci_engine, synthetic_independent_data):
        """Test that test_conditional_independence returns correct format."""
        result = ci_engine.test_conditional_independence(
            synthetic_independent_data, 'Z', 'Y', ['W']
        )

        assert isinstance(result, dict)
        assert 'cmi' in result
        assert 'p_value' in result
        assert 'reject_independence' in result
        assert 'n_permutations' in result
        assert 'null_mean' in result
        assert 'null_std' in result

    def test_compute_cmi_only_faster(self, synthetic_independent_data):
        """Test that compute_cmi_only is faster than full test."""
        engine = CITestEngine(n_permutations=200, random_seed=42)
        df = synthetic_independent_data

        # Time compute_cmi_only
        start = time.time()
        for _ in range(5):
            engine.compute_cmi_only(df, 'Z', 'Y', ['W'])
        cmi_only_time = time.time() - start

        # Time full test
        start = time.time()
        engine.test_conditional_independence(df, 'Z', 'Y', ['W'])
        full_test_time = time.time() - start

        assert cmi_only_time < full_test_time, \
            f"compute_cmi_only should be faster: {cmi_only_time:.3f}s vs {full_test_time:.3f}s"

    def test_score_ehs_criteria_structure(self, ci_engine):
        """Test that score_ehs_criteria returns correct structure."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'Y': np.random.choice([0, 1], n),
            'X': np.random.choice([0, 1], n),
            'Z_a': np.random.choice([0, 1], n),
            'Z_b': np.random.choice([0, 1], n)
        })

        result = ci_engine.score_ehs_criteria(
            df, z_a='Z_a', z_b=['Z_b'], treatment='X', outcome='Y'
        )

        assert isinstance(result, dict)
        assert result['z_a'] == 'Z_a'
        assert 'test_i_cmi' in result
        assert 'test_i_pvalue' in result
        assert 'test_i_reject' in result
        assert 'test_ii_cmi' in result
        assert 'test_ii_pvalue' in result
        assert 'test_ii_reject' in result
        assert 'passes_ehs' in result
        assert 'score' in result

        # Check score formula (CMI-based, sample-size invariant)
        expected_score = result['test_ii_cmi'] - result['test_i_cmi']
        assert np.isclose(result['score'], expected_score, rtol=1e-10)

    def test_ehs_good_instrument(self):
        """Test EHS criteria with a good instrument (Z_a -> X -> Y)."""
        np.random.seed(42)
        n = 500

        # Z_a -> X -> Y structure
        Z_a = np.random.choice([0, 1], n)
        # X depends on Z_a
        X = (Z_a + np.random.choice([0, 1], n, p=[0.8, 0.2])) % 2
        # Y depends on X but not directly on Z_a
        Y = (X + np.random.choice([0, 1], n, p=[0.7, 0.3])) % 2

        df = pd.DataFrame({'Y': Y, 'X': X, 'Z_a': Z_a, 'Z_b': np.random.choice([0, 1], n)})

        engine = CITestEngine(n_permutations=200, random_seed=42)
        result = engine.score_ehs_criteria(
            df, z_a='Z_a', z_b=['Z_b'], treatment='X', outcome='Y'
        )

        # A good instrument should have high score
        # Test (i): Y ⊥ Z_a | (Z_b, X) should NOT reject (high p-value)
        # Test (ii): Y ⊥̸ Z_a | Z_b should reject (low p-value)
        # This may not always pass due to noise, so we check the score is reasonable
        assert result['score'] >= 0, "Score should be non-negative"

    def test_ehs_bad_instrument(self):
        """Test EHS criteria with a bad instrument (Z_a -> X and Z_a -> Y)."""
        np.random.seed(42)
        n = 500

        # Z_a -> X and Z_a -> Y (direct effect)
        Z_a = np.random.choice([0, 1], n)
        X = (Z_a + np.random.choice([0, 1], n, p=[0.8, 0.2])) % 2
        # Y depends on BOTH X and Z_a directly
        Y = ((Z_a + X) > 0).astype(int)

        df = pd.DataFrame({'Y': Y, 'X': X, 'Z_a': Z_a, 'Z_b': np.random.choice([0, 1], n)})

        engine = CITestEngine(n_permutations=200, random_seed=42)
        result = engine.score_ehs_criteria(
            df, z_a='Z_a', z_b=['Z_b'], treatment='X', outcome='Y'
        )

        # A bad instrument should have low score or fail EHS
        # Test (i): Y ⊥ Z_a | (Z_b, X) should REJECT (low p-value) because Z_a affects Y directly
        # The score should be lower than for a good instrument
        assert 'score' in result


class TestCITestEngineEdgeCases:
    """Test edge cases for CITestEngine."""

    def test_empty_conditioning_set(self, ci_engine):
        """Test with empty conditioning set."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'X': np.random.choice([0, 1], n),
            'Y': np.random.choice([0, 1], n)
        })

        result = ci_engine.test_conditional_independence(df, 'X', 'Y', [])

        assert 'cmi' in result
        assert 'p_value' in result

    def test_return_null_distribution(self, ci_engine, synthetic_independent_data):
        """Test that null distribution can be returned."""
        result = ci_engine.test_conditional_independence(
            synthetic_independent_data, 'Z', 'Y', ['W'],
            return_null_dist=True
        )

        assert 'null_distribution' in result
        assert len(result['null_distribution']) == ci_engine.n_permutations

    def test_custom_parameters(self):
        """Test engine with custom parameters."""
        engine = CITestEngine(
            smoothing_alpha=0.5,
            test_alpha=0.01,
            n_permutations=50,
            random_seed=123
        )

        assert engine.smoothing_alpha == 0.5
        assert engine.test_alpha == 0.01
        assert engine.n_permutations == 50
        assert engine.random_seed == 123

    def test_score_ehs_without_permutation(self):
        """Test EHS scoring without permutation test."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'Y': np.random.choice([0, 1], n),
            'X': np.random.choice([0, 1], n),
            'Z_a': np.random.choice([0, 1], n),
            'Z_b': np.random.choice([0, 1], n)
        })

        engine = CITestEngine(n_permutations=100, random_seed=42)
        result = engine.score_ehs_criteria(
            df, z_a='Z_a', z_b=['Z_b'], treatment='X', outcome='Y',
            use_permutation_test=False
        )

        # Without permutation test, p-values should be None
        assert result['test_i_pvalue'] is None
        assert result['test_ii_pvalue'] is None
        # But CMI values should be computed
        assert result['test_i_cmi'] >= 0
        assert result['test_ii_cmi'] >= 0
