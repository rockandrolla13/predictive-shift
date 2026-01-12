"""
Test David's CMI implementation and the CITestEngine wrapper.
"""

import pytest
import numpy as np
import pandas as pd
from causal_grounding.ci_tests import (
    compute_cmi,
    permutation_test_cmi,
    CITestEngine,
    combine_conditioning_vars
)


class TestComputeCMI:
    """Test David's compute_cmi function."""

    def test_independent_variables(self):
        """CMI should be near zero for independent variables."""
        np.random.seed(42)
        n = 1000

        Y = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1, 2], size=n)
        W = np.random.choice([0, 1], size=n)

        cmi = compute_cmi(Y, Z, W)
        assert cmi < 0.05, f"CMI should be near 0 for independent vars, got {cmi}"

    def test_dependent_variables(self):
        """CMI should be positive for dependent variables."""
        np.random.seed(42)
        n = 1000

        W = np.random.choice([0, 1, 2], size=n)
        Z = np.random.choice([0, 1, 2, 3], size=n)

        # Y depends on Z
        logits = -1.0 + 0.5 * Z
        probs = 1 / (1 + np.exp(-logits))
        Y = (np.random.random(n) < probs).astype(int)

        cmi = compute_cmi(Y, Z, W)
        assert cmi > 0.01, f"CMI should be positive for dependent vars, got {cmi}"

    def test_smoothing_prevents_zero(self):
        """Laplacian smoothing should prevent division by zero."""
        Y = np.array([0, 0, 1, 1])
        Z = np.array([0, 1, 0, 1])
        W = np.array([0, 0, 1, 1])

        # Should not raise
        cmi = compute_cmi(Y, Z, W, alpha=1.0)
        assert np.isfinite(cmi)


class TestPermutationTest:
    """Test permutation_test_cmi function."""

    def test_independent_high_pvalue(self):
        """Independent variables should have high p-value."""
        np.random.seed(42)
        n = 500

        Y = np.random.choice([0, 1], size=n)
        Z = np.random.choice([0, 1, 2], size=n)
        W = np.random.choice([0, 1], size=n)

        cmi, p_value, null_dist = permutation_test_cmi(
            Y, Z, W, n_permutations=500, random_seed=42
        )

        assert p_value > 0.05, f"p-value should be high for independent vars, got {p_value}"

    def test_dependent_low_pvalue(self):
        """Dependent variables should have low p-value."""
        np.random.seed(42)
        n = 500

        W = np.random.choice([0, 1, 2], size=n)
        Z = np.random.choice([0, 1, 2, 3], size=n)

        logits = -1.0 + 0.3 * Z
        probs = 1 / (1 + np.exp(-logits))
        Y = (np.random.random(n) < probs).astype(int)

        cmi, p_value, null_dist = permutation_test_cmi(
            Y, Z, W, n_permutations=500, random_seed=42
        )

        assert p_value < 0.1, f"p-value should be low for dependent vars, got {p_value}"

    def test_null_distribution_shape(self):
        """Null distribution should have correct shape."""
        Y = np.random.choice([0, 1], size=100)
        Z = np.random.choice([0, 1], size=100)
        W = np.random.choice([0, 1], size=100)

        n_perms = 200
        cmi, p_value, null_dist = permutation_test_cmi(
            Y, Z, W, n_permutations=n_perms
        )

        assert len(null_dist) == n_perms


class TestCITestEngine:
    """Test CITestEngine wrapper class."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 300
        return pd.DataFrame({
            'X': np.random.choice([0, 1, 2], size=n),
            'Y': np.random.choice([0, 1], size=n),
            'Z': np.random.choice([0, 1], size=n),
            'W': np.random.choice([0, 1, 2], size=n),
        })

    def test_engine_initialization(self):
        engine = CITestEngine(n_permutations=100)
        assert engine.n_permutations == 100
        assert engine.test_alpha == 0.05

    def test_conditional_independence_result_format(self, sample_data):
        engine = CITestEngine(n_permutations=100)
        result = engine.test_conditional_independence(
            sample_data, 'X', 'Y', ['Z']
        )

        assert 'cmi' in result
        assert 'p_value' in result
        assert 'reject_independence' in result
        assert isinstance(result['reject_independence'], (bool, np.bool_))

    def test_ehs_scoring(self, sample_data):
        engine = CITestEngine(n_permutations=100)
        result = engine.score_ehs_criteria(
            sample_data,
            z_a='X',
            z_b=['Z'],
            treatment='W',
            outcome='Y'
        )

        assert 'passes_ehs' in result
        assert 'score' in result
        assert 'test_i_pvalue' in result
        assert 'test_ii_pvalue' in result


class TestCombineConditioningVars:
    """Test helper function for combining variables."""

    def test_empty_list(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = combine_conditioning_vars(df, [])
        assert np.all(result == 0)

    def test_single_column(self):
        df = pd.DataFrame({'A': [0, 1, 0, 1]})
        result = combine_conditioning_vars(df, ['A'])
        assert len(np.unique(result)) == 2

    def test_multiple_columns(self):
        df = pd.DataFrame({
            'A': [0, 0, 1, 1],
            'B': [0, 1, 0, 1]
        })
        result = combine_conditioning_vars(df, ['A', 'B'])
        assert len(np.unique(result)) == 4
