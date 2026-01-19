"""
Tests for Comparison Module

Tests the comparison.py utilities including MethodComparator,
ComparisonResults, and comparison functions.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from causal_grounding.comparison import (
    MethodComparator,
    ComparisonResults,
    MethodConfig,
)
from causal_grounding.ricardo_adapter import compare_with_ricardo


# =============================================================================
# MOCK ESTIMATOR FOR TESTING
# =============================================================================

class MockEstimator:
    """Mock estimator for testing MethodComparator."""

    def __init__(self, noise_level: float = 0.1, base_lower: float = 0.2):
        self.noise_level = noise_level
        self.base_lower = base_lower
        self.covariate_scores_ = None
        self._fitted = False

    def fit(self, data, treatment, outcome, covariates, **kwargs):
        """Mock fit method."""
        self._n_samples = len(data)
        self._covariates = covariates
        self._fitted = True

        # Create mock covariate scores
        self.covariate_scores_ = pd.DataFrame({
            'z_a': covariates,
            'score': np.random.uniform(0, 1, len(covariates))
        })

        return self

    def predict_bounds(self, **kwargs):
        """Mock predict_bounds method."""
        if not self._fitted:
            raise RuntimeError("Must fit before predict")

        n_strata = 5
        np.random.seed(42)

        return pd.DataFrame({
            'stratum': [f'S{i}' for i in range(n_strata)],
            'lower': self.base_lower + np.random.uniform(-0.1, 0.1, n_strata) * self.noise_level,
            'upper': 0.8 + np.random.uniform(-0.1, 0.1, n_strata) * self.noise_level
        })


class MockSlowEstimator(MockEstimator):
    """Mock estimator that takes longer."""

    def fit(self, data, treatment, outcome, covariates, **kwargs):
        import time
        time.sleep(0.1)  # Simulate slower fitting
        return super().fit(data, treatment, outcome, covariates, **kwargs)


class MockFailingEstimator:
    """Mock estimator that raises an error."""

    def fit(self, data, treatment, outcome, covariates, **kwargs):
        raise ValueError("Mock error during fit")

    def predict_bounds(self, **kwargs):
        return pd.DataFrame()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'treatment': np.random.binomial(1, 0.5, n),
        'outcome': np.random.normal(0, 1, n),
        'cov1': np.random.normal(0, 1, n),
        'cov2': np.random.choice(['A', 'B', 'C'], n),
        'cov3': np.random.uniform(0, 1, n)
    })


@pytest.fixture
def sample_bounds_a():
    """Sample bounds from method A."""
    return pd.DataFrame({
        'stratum': ['S0', 'S1', 'S2'],
        'lower': [0.2, 0.3, 0.25],
        'upper': [0.6, 0.7, 0.65]
    })


@pytest.fixture
def sample_bounds_b():
    """Sample bounds from method B."""
    return pd.DataFrame({
        'stratum': ['S0', 'S1', 'S2'],
        'lower': [0.22, 0.28, 0.27],
        'upper': [0.58, 0.72, 0.63]
    })


@pytest.fixture
def sample_comparison_results(sample_bounds_a, sample_bounds_b):
    """Sample ComparisonResults object."""
    return ComparisonResults(
        method_names=['method_a', 'method_b'],
        bounds={'method_a': sample_bounds_a, 'method_b': sample_bounds_b},
        metrics={
            'method_a': {'coverage_rate': 0.8, 'mean_width': 0.4},
            'method_b': {'coverage_rate': 0.75, 'mean_width': 0.35}
        },
        runtimes={'method_a': 1.5, 'method_b': 0.8}
    )


# =============================================================================
# TEST: MethodConfig
# =============================================================================

class TestMethodConfig:
    """Tests for MethodConfig dataclass."""

    def test_basic_config(self):
        """Test creating basic config."""
        config = MethodConfig(
            name='test_method',
            estimator_class=MockEstimator
        )
        assert config.name == 'test_method'
        assert config.estimator_class == MockEstimator
        assert config.estimator_kwargs == {}

    def test_config_with_kwargs(self):
        """Test config with estimator kwargs."""
        config = MethodConfig(
            name='test',
            estimator_class=MockEstimator,
            estimator_kwargs={'noise_level': 0.5},
            description='Test method'
        )
        assert config.estimator_kwargs == {'noise_level': 0.5}
        assert config.description == 'Test method'


# =============================================================================
# TEST: ComparisonResults
# =============================================================================

class TestComparisonResults:
    """Tests for ComparisonResults dataclass."""

    def test_get_bounds_df(self, sample_comparison_results):
        """Test getting bounds for specific method."""
        bounds = sample_comparison_results.get_bounds_df('method_a')
        assert isinstance(bounds, pd.DataFrame)
        assert 'lower' in bounds.columns

    def test_get_bounds_df_missing(self, sample_comparison_results):
        """Test getting bounds for missing method."""
        bounds = sample_comparison_results.get_bounds_df('nonexistent')
        assert bounds is None

    def test_get_metrics_df(self, sample_comparison_results):
        """Test getting metrics as DataFrame."""
        df = sample_comparison_results.get_metrics_df()
        assert isinstance(df, pd.DataFrame)
        assert 'method' in df.columns
        assert len(df) == 2

    def test_get_summary(self, sample_comparison_results):
        """Test getting text summary."""
        summary = sample_comparison_results.get_summary()
        assert isinstance(summary, str)
        assert 'method_a' in summary
        assert 'method_b' in summary


# =============================================================================
# TEST: MethodComparator
# =============================================================================

class TestMethodComparator:
    """Tests for MethodComparator class."""

    def test_init(self, sample_data):
        """Test comparator initialization."""
        comparator = MethodComparator(
            sample_data,
            treatment='treatment',
            outcome='outcome',
            covariates=['cov1', 'cov2', 'cov3']
        )
        assert comparator.data is sample_data
        assert comparator.treatment == 'treatment'

    def test_add_method(self, sample_data):
        """Test adding methods."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        result = comparator.add_method('test', MockEstimator)
        assert result is comparator  # Check method chaining
        assert 'test' in comparator.methods

    def test_add_method_with_kwargs(self, sample_data):
        """Test adding method with kwargs."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        comparator.add_method(
            'test',
            MockEstimator,
            estimator_kwargs={'noise_level': 0.5}
        )
        assert comparator.methods['test'].estimator_kwargs == {'noise_level': 0.5}

    def test_run_all_single_method(self, sample_data):
        """Test running single method."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1'],
            ground_truth=0.5
        )
        comparator.add_method('mock', MockEstimator)

        results = comparator.run_all(verbose=False)

        assert isinstance(results, ComparisonResults)
        assert 'mock' in results.bounds
        assert 'mock' in results.runtimes

    def test_run_all_multiple_methods(self, sample_data):
        """Test running multiple methods."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1'],
            ground_truth=0.5
        )
        comparator.add_method('method_a', MockEstimator, {'base_lower': 0.2})
        comparator.add_method('method_b', MockEstimator, {'base_lower': 0.3})

        results = comparator.run_all(verbose=False)

        assert len(results.method_names) == 2
        assert 'method_a' in results.bounds
        assert 'method_b' in results.bounds

    def test_run_all_with_failing_method(self, sample_data):
        """Test handling of failing method."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        comparator.add_method('good', MockEstimator)
        comparator.add_method('bad', MockFailingEstimator)

        # Should not raise, but bad method should have empty bounds
        results = comparator.run_all(verbose=False)
        assert len(results.bounds['bad']) == 0

    def test_compare_bounds(self, sample_data):
        """Test compare_bounds method."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        comparator.add_method('a', MockEstimator)
        comparator.add_method('b', MockEstimator)
        comparator.run_all(verbose=False)

        comparison = comparator.compare_bounds()
        assert isinstance(comparison, pd.DataFrame)

    def test_compare_bounds_before_run(self, sample_data):
        """Test compare_bounds raises before run_all."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        comparator.add_method('a', MockEstimator)

        with pytest.raises(RuntimeError, match="run_all"):
            comparator.compare_bounds()

    def test_compare_coverage(self, sample_data):
        """Test compare_coverage method."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1'],
            ground_truth=0.5
        )
        comparator.add_method('a', MockEstimator)
        comparator.run_all(verbose=False)

        coverage = comparator.compare_coverage()
        assert isinstance(coverage, pd.DataFrame)
        assert 'coverage_rate' in coverage.columns

    def test_compare_runtime(self, sample_data):
        """Test compare_runtime method."""
        comparator = MethodComparator(
            sample_data, 'treatment', 'outcome', ['cov1']
        )
        comparator.add_method('a', MockEstimator)
        comparator.add_method('b', MockSlowEstimator)
        comparator.run_all(verbose=False)

        runtime = comparator.compare_runtime()
        assert isinstance(runtime, pd.DataFrame)
        assert 'runtime_seconds' in runtime.columns
        # Slow method should have higher runtime
        slow_time = runtime.loc[runtime['method'] == 'b', 'runtime_seconds'].values[0]
        fast_time = runtime.loc[runtime['method'] == 'a', 'runtime_seconds'].values[0]
        assert slow_time > fast_time


# =============================================================================
# TEST: compare_with_ricardo
# =============================================================================

class TestCompareWithRicardo:
    """Tests for compare_with_ricardo function."""

    def test_compare_bounds(self, sample_bounds_a, sample_bounds_b):
        """Test comparing bounds between methods."""
        our_results = {'bounds': sample_bounds_a}
        ricardo_results = {'bounds': sample_bounds_b}

        comparison = compare_with_ricardo(our_results, ricardo_results)

        assert 'matches' in comparison
        assert 'differences' in comparison
        assert 'metrics' in comparison
        assert 'lower_bound_mae' in comparison['metrics']

    def test_compare_cate(self):
        """Test comparing CATE values."""
        our_results = {'cate': np.array([0.1, 0.2, 0.3, 0.4])}
        ricardo_results = {'cate': np.array([0.11, 0.19, 0.31, 0.39])}

        comparison = compare_with_ricardo(our_results, ricardo_results)

        assert 'cate_mae' in comparison['metrics']
        assert 'cate_correlation' in comparison['metrics']

    def test_compare_coverage(self):
        """Test comparing coverage values."""
        our_results = {'coverage': 0.8}
        ricardo_results = {'coverage': 0.75}

        comparison = compare_with_ricardo(our_results, ricardo_results)

        assert abs(comparison['metrics']['coverage_diff'] - 0.05) < 1e-10

    def test_overall_match_true(self, sample_bounds_a):
        """Test overall match when results are similar."""
        our_results = {'bounds': sample_bounds_a}
        ricardo_results = {'bounds': sample_bounds_a}  # Same bounds

        comparison = compare_with_ricardo(our_results, ricardo_results, tolerance=0.01)

        assert comparison['matches']['lower_bounds'] == True
        assert comparison['matches']['upper_bounds'] == True
        assert comparison['overall_match'] == True

    def test_overall_match_false(self, sample_bounds_a, sample_bounds_b):
        """Test overall match when results differ."""
        our_results = {'bounds': sample_bounds_a}
        ricardo_results = {'bounds': sample_bounds_b}

        comparison = compare_with_ricardo(our_results, ricardo_results, tolerance=0.01)

        # With tight tolerance, should not match
        assert comparison['overall_match'] == False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for comparison module."""

    def test_full_comparison_workflow(self, sample_data, tmp_path):
        """Test complete comparison workflow."""
        comparator = MethodComparator(
            sample_data,
            treatment='treatment',
            outcome='outcome',
            covariates=['cov1', 'cov2', 'cov3'],
            ground_truth=0.3
        )

        # Add multiple methods
        comparator.add_method('default', MockEstimator, {})
        comparator.add_method('high_noise', MockEstimator, {'noise_level': 0.5})

        # Run all methods
        results = comparator.run_all(verbose=False)

        # Check results
        assert len(results.method_names) == 2
        assert all(m in results.bounds for m in results.method_names)
        assert all(m in results.runtimes for m in results.method_names)

        # Get comparisons
        bounds_comp = comparator.compare_bounds()
        runtime_comp = comparator.compare_runtime()
        coverage_comp = comparator.compare_coverage()

        assert len(bounds_comp) > 0
        assert len(runtime_comp) == 2
        assert len(coverage_comp) == 2

        # Generate report
        report_path = comparator.generate_comparison_report(
            tmp_path, include_plots=False
        )
        assert report_path.exists()
