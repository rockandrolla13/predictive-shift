"""
Tests for Visualization Module

Tests the visualization.py plotting functions.
Uses matplotlib's non-interactive backend for testing.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from causal_grounding.visualization import (
    plot_cate_bounds,
    plot_bounds_forest,
    plot_bounds_comparison,
    plot_coverage_by_stratum,
    plot_coverage_heatmap,
    plot_width_distribution,
    plot_width_vs_sample_size,
    plot_ehs_scores,
    plot_cmi_distribution,
    plot_method_comparison_summary,
    plot_runtime_comparison,
    plot_agreement_matrix,
    save_figure,
    create_multi_panel_figure,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_bounds():
    """Sample bounds DataFrame for testing."""
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        'stratum': [f'S{i}' for i in range(n)],
        'lower': 1000 + np.random.uniform(-200, 100, n),
        'upper': 1600 + np.random.uniform(-100, 200, n)
    })


@pytest.fixture
def sample_ehs_scores():
    """Sample EHS scores DataFrame."""
    return pd.DataFrame({
        'z_a': ['age', 'gender', 'education', 'income', 'region'],
        'score': [0.15, 0.12, 0.08, 0.05, 0.02],
        'test_i_pvalue': [0.4, 0.3, 0.2, 0.1, 0.05],
        'test_ii_pvalue': [0.01, 0.02, 0.05, 0.1, 0.2]
    })


@pytest.fixture
def sample_comparison_results():
    """Sample results dict for comparison plots."""
    return {
        'method_a': {'coverage': 0.8, 'mean_width': 300, 'runtime': 10.5},
        'method_b': {'coverage': 0.75, 'mean_width': 250, 'runtime': 5.2},
        'method_c': {'coverage': 0.85, 'mean_width': 350, 'runtime': 15.0}
    }


# =============================================================================
# TEST: Bounds Visualization
# =============================================================================

class TestBoundsVisualization:
    """Tests for bounds plotting functions."""

    def test_plot_cate_bounds_basic(self, sample_bounds):
        """Test basic CATE bounds plot."""
        fig = plot_cate_bounds(sample_bounds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cate_bounds_with_true_cate(self, sample_bounds):
        """Test CATE bounds plot with reference line."""
        fig = plot_cate_bounds(sample_bounds, true_cate=1500)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cate_bounds_with_stratum_col(self, sample_bounds):
        """Test CATE bounds plot with stratum labels."""
        fig = plot_cate_bounds(sample_bounds, stratum_col='stratum')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cate_bounds_custom_columns(self):
        """Test with custom column names."""
        bounds = pd.DataFrame({
            'lb': [0.1, 0.2, 0.3],
            'ub': [0.5, 0.6, 0.7]
        })
        fig = plot_cate_bounds(bounds, lower_col='lb', upper_col='ub')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bounds_forest_basic(self, sample_bounds):
        """Test basic forest plot."""
        fig = plot_bounds_forest(sample_bounds, top_k=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bounds_forest_with_true_cate(self, sample_bounds):
        """Test forest plot with reference line."""
        fig = plot_bounds_forest(sample_bounds, top_k=5, true_cate=1500)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bounds_comparison(self, sample_bounds):
        """Test bounds comparison plot."""
        bounds_dict = {
            'method_a': sample_bounds,
            'method_b': sample_bounds.assign(
                lower=sample_bounds['lower'] + 50,
                upper=sample_bounds['upper'] - 50
            )
        }
        fig = plot_bounds_comparison(bounds_dict)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# TEST: Coverage Visualization
# =============================================================================

class TestCoverageVisualization:
    """Tests for coverage plotting functions."""

    def test_plot_coverage_by_stratum_scalar(self, sample_bounds):
        """Test coverage plot with scalar ground truth."""
        fig = plot_coverage_by_stratum(sample_bounds, ground_truth=1500)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_coverage_by_stratum_array(self, sample_bounds):
        """Test coverage plot with per-stratum ground truth."""
        n = len(sample_bounds)
        truth = 1400 + np.random.uniform(0, 200, n)
        fig = plot_coverage_by_stratum(sample_bounds, ground_truth=truth)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_coverage_heatmap(self):
        """Test coverage heatmap."""
        coverage_matrix = np.random.binomial(1, 0.7, size=(3, 5))
        fig = plot_coverage_heatmap(
            coverage_matrix,
            method_names=['A', 'B', 'C'],
            stratum_names=['S1', 'S2', 'S3', 'S4', 'S5']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# TEST: Width / Informativeness
# =============================================================================

class TestWidthVisualization:
    """Tests for width plotting functions."""

    def test_plot_width_distribution(self, sample_bounds):
        """Test width histogram."""
        fig = plot_width_distribution(sample_bounds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_width_vs_sample_size(self, sample_bounds):
        """Test width vs sample size scatter."""
        sample_sizes = np.random.randint(50, 200, len(sample_bounds))
        fig = plot_width_vs_sample_size(sample_bounds, sample_sizes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# TEST: CI Test Diagnostics
# =============================================================================

class TestCIDiagnostics:
    """Tests for CI test diagnostic plots."""

    def test_plot_ehs_scores(self, sample_ehs_scores):
        """Test EHS scores bar chart."""
        fig = plot_ehs_scores(sample_ehs_scores)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_ehs_scores_no_highlight(self, sample_ehs_scores):
        """Test EHS scores without best highlight."""
        fig = plot_ehs_scores(sample_ehs_scores, highlight_best=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cmi_distribution(self):
        """Test CMI permutation test plot."""
        np.random.seed(42)
        null_dist = np.random.exponential(0.01, 1000)
        observed = 0.05
        fig = plot_cmi_distribution(null_dist, observed)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# TEST: Comparison Plots
# =============================================================================

class TestComparisonPlots:
    """Tests for comparison plotting functions."""

    def test_plot_method_comparison_summary(self, sample_comparison_results):
        """Test method comparison summary."""
        fig = plot_method_comparison_summary(sample_comparison_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_runtime_comparison(self):
        """Test runtime comparison bar chart."""
        timing = {'method_a': 10.5, 'method_b': 5.2, 'method_c': 15.0}
        fig = plot_runtime_comparison(timing)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_agreement_matrix(self):
        """Test decision agreement matrix."""
        np.random.seed(42)
        decisions1 = np.random.binomial(1, 0.6, 100).astype(bool)
        decisions2 = np.random.binomial(1, 0.7, 100).astype(bool)
        fig = plot_agreement_matrix(decisions1, decisions2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# TEST: Utility Functions
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_save_figure(self, sample_bounds, tmp_path):
        """Test saving figure to file."""
        fig = plot_cate_bounds(sample_bounds)
        filepath = tmp_path / 'test_plot.png'
        save_figure(fig, str(filepath))
        assert filepath.exists()

    def test_create_multi_panel_figure(self):
        """Test creating multi-panel figure."""
        fig, axes = create_multi_panel_figure(4, n_cols=2)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 4
        plt.close(fig)

    def test_create_multi_panel_figure_odd_panels(self):
        """Test multi-panel with odd number of panels."""
        fig, axes = create_multi_panel_figure(3, n_cols=2)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_create_multi_panel_figure_single(self):
        """Test single panel figure."""
        fig, axes = create_multi_panel_figure(1)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 1
        plt.close(fig)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for visualization module."""

    def test_full_visualization_pipeline(self, sample_bounds, tmp_path):
        """Test creating multiple plots and saving."""
        true_cate = 1500

        # Create all major plot types
        plots = []

        plots.append(('bounds', plot_cate_bounds(
            sample_bounds, true_cate=true_cate, stratum_col='stratum'
        )))

        plots.append(('forest', plot_bounds_forest(
            sample_bounds, top_k=5, true_cate=true_cate
        )))

        plots.append(('coverage', plot_coverage_by_stratum(
            sample_bounds, true_cate, stratum_col='stratum'
        )))

        plots.append(('width', plot_width_distribution(sample_bounds)))

        # Save all plots
        for name, fig in plots:
            filepath = tmp_path / f'{name}.png'
            save_figure(fig, str(filepath))
            assert filepath.exists()
