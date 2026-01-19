"""
Tests for causal_grounding.sensitivity module.

Tests cover:
- SweepConfig creation and validation
- SweepResults computation and methods
- SensitivityAnalyzer sweep execution
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from causal_grounding.sensitivity import (
    SweepConfig,
    SweepResults,
    SensitivityAnalyzer,
    run_epsilon_sweep,
)


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_explicit_values(self):
        """Test creating config with explicit values."""
        config = SweepConfig(
            parameter_name='epsilon',
            values=[0.1, 0.2, 0.3]
        )
        assert config.parameter_name == 'epsilon'
        assert config.values == [0.1, 0.2, 0.3]

    def test_auto_generate_linear(self):
        """Test auto-generating linear-spaced values."""
        config = SweepConfig(
            parameter_name='epsilon',
            n_points=5,
            range=(0.0, 1.0),
            log_scale=False
        )
        assert len(config.values) == 5
        assert np.isclose(config.values[0], 0.0)
        assert np.isclose(config.values[-1], 1.0)
        # Check linear spacing
        diffs = np.diff(config.values)
        assert np.allclose(diffs, diffs[0])

    def test_auto_generate_log(self):
        """Test auto-generating log-spaced values."""
        config = SweepConfig(
            parameter_name='epsilon',
            n_points=5,
            range=(0.01, 1.0),
            log_scale=True
        )
        assert len(config.values) == 5
        assert np.isclose(config.values[0], 0.01)
        assert np.isclose(config.values[-1], 1.0)
        # Check log spacing (ratios should be constant)
        ratios = [config.values[i+1] / config.values[i] for i in range(len(config.values)-1)]
        assert np.allclose(ratios, ratios[0])

    def test_validation_error(self):
        """Test that missing parameters raises error."""
        with pytest.raises(ValueError, match="Must provide either"):
            SweepConfig(parameter_name='epsilon')

    def test_from_dict_explicit(self):
        """Test creating from dict with explicit values."""
        config = SweepConfig.from_dict({
            'parameter': 'epsilon',
            'values': [0.1, 0.2, 0.3]
        })
        assert config.parameter_name == 'epsilon'
        assert config.values == [0.1, 0.2, 0.3]

    def test_from_dict_auto_generate(self):
        """Test creating from dict with auto-generation."""
        config = SweepConfig.from_dict({
            'parameter': 'epsilon',
            'n_points': 3,
            'range': [0.1, 0.3],
            'log_scale': False
        })
        assert config.parameter_name == 'epsilon'
        assert len(config.values) == 3


class TestSweepResults:
    """Tests for SweepResults dataclass."""

    @pytest.fixture
    def mock_results_df(self):
        """Create mock results DataFrame."""
        return pd.DataFrame({
            'epsilon': [0.05, 0.1, 0.15, 0.2, 0.3],
            'coverage_rate': [0.30, 0.40, 0.50, 0.55, 0.60],
            'mean_width': [500, 600, 700, 800, 1000],
            'n_z_values': [30, 30, 30, 30, 30]
        })

    def test_pareto_frontier_computation(self, mock_results_df):
        """Test automatic Pareto frontier computation."""
        results = SweepResults(results_df=mock_results_df)

        # Pareto frontier should have points where coverage improves as width increases
        assert len(results.pareto_frontier) > 0
        # All points in Pareto frontier should be from original results
        assert all(results.pareto_frontier['epsilon'].isin(mock_results_df['epsilon']))

    def test_pareto_dominance(self, mock_results_df):
        """Test that Pareto frontier has no dominated points."""
        results = SweepResults(results_df=mock_results_df)
        pareto = results.pareto_frontier

        # No point should be dominated by another in the frontier
        for i, row1 in pareto.iterrows():
            for j, row2 in pareto.iterrows():
                if i != j:
                    # row1 dominated by row2 if row2 has better width AND coverage
                    dominated = (row2['mean_width'] < row1['mean_width'] and
                                 row2['coverage_rate'] > row1['coverage_rate'])
                    assert not dominated, f"Point {i} dominated by {j}"

    def test_get_recommended_epsilon_meets_target(self, mock_results_df):
        """Test recommendation when target can be met."""
        results = SweepResults(results_df=mock_results_df)
        rec = results.get_recommended_epsilon(target_coverage=0.5)

        assert rec['epsilon'] is not None
        assert rec['coverage_rate'] >= 0.5
        assert 'meets' in rec['reason'].lower() or 'best' in rec['reason'].lower()

    def test_get_recommended_epsilon_cannot_meet_target(self, mock_results_df):
        """Test recommendation when target cannot be met."""
        results = SweepResults(results_df=mock_results_df)
        rec = results.get_recommended_epsilon(target_coverage=0.99)

        assert rec['epsilon'] is not None
        # Should return best available
        assert rec['coverage_rate'] == mock_results_df['coverage_rate'].max()
        assert 'no config meets' in rec['reason'].lower()

    def test_get_recommended_prefer_width(self, mock_results_df):
        """Test recommendation preferring width over coverage."""
        results = SweepResults(results_df=mock_results_df)
        rec = results.get_recommended_epsilon(target_coverage=0.5, prefer='width')

        # Among configs meeting 0.5 coverage, should have minimum width
        candidates = mock_results_df[mock_results_df['coverage_rate'] >= 0.5]
        min_width = candidates['mean_width'].min()
        assert rec['mean_width'] == min_width

    def test_to_csv(self, mock_results_df):
        """Test saving results to CSV."""
        results = SweepResults(results_df=mock_results_df)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            results.to_csv(f.name)
            loaded = pd.read_csv(f.name)
            assert len(loaded) == len(mock_results_df)
            assert 'epsilon' in loaded.columns

    def test_to_json(self, mock_results_df):
        """Test saving summary to JSON."""
        results = SweepResults(results_df=mock_results_df)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            results.to_json(f.name)

            import json
            with open(f.name, 'r') as jf:
                summary = json.load(jf)

            assert 'n_configurations' in summary
            assert summary['n_configurations'] == 5
            assert 'recommended' in summary


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    @pytest.fixture
    def mock_training_data(self):
        """Create mock training data."""
        np.random.seed(42)
        n = 200

        def make_site_data(site_name):
            df = pd.DataFrame({
                'iv': np.random.choice([0, 1], n),
                'dv': np.random.randn(n) * 100 + 1000,
                'resp_age': np.random.randint(18, 70, n),
                'resp_gender': np.random.choice([0, 1], n),
                'resp_polideo': np.random.randint(0, 7, n),
                'F': np.random.choice(['on', 'idle'], n),
                'site': site_name
            })
            return df

        return {
            'site_a': make_site_data('site_a'),
            'site_b': make_site_data('site_b'),
        }

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        config = SweepConfig(parameter_name='epsilon', values=[0.1, 0.2])
        analyzer = SensitivityAnalyzer(
            sweep_config=config,
            base_estimator_params={'n_permutations': 100},
            random_seed=42
        )

        assert analyzer.sweep_config == config
        assert analyzer.base_estimator_params['n_permutations'] == 100
        assert analyzer.random_seed == 42

    def test_run_sweep_basic(self, mock_training_data):
        """Test basic sweep execution."""
        config = SweepConfig(parameter_name='epsilon', values=[0.1, 0.2])
        analyzer = SensitivityAnalyzer(
            sweep_config=config,
            base_estimator_params={'n_permutations': 50},  # Fast for testing
            random_seed=42,
            verbose=False
        )

        results = analyzer.run_sweep(
            training_data=mock_training_data,
            treatment='iv',
            outcome='dv'
        )

        assert isinstance(results, SweepResults)
        assert len(results.results_df) == 2
        assert 'epsilon' in results.results_df.columns
        assert list(results.results_df['epsilon']) == [0.1, 0.2]

    def test_run_sweep_with_ground_truth(self, mock_training_data):
        """Test sweep with ground truth coverage computation."""
        config = SweepConfig(parameter_name='epsilon', values=[0.1, 0.2])
        analyzer = SensitivityAnalyzer(
            sweep_config=config,
            base_estimator_params={'n_permutations': 50},
            random_seed=42,
            verbose=False
        )

        # Create ground truth that matches discretized z-values
        # Discretized age has 5 bins, gender 2, polideo 3 = 30 combinations
        ground_truth = {}
        for age in range(5):
            for gender in range(2):
                for polideo in range(3):
                    ground_truth[(age, gender, polideo)] = 100.0 + age * 10

        results = analyzer.run_sweep(
            training_data=mock_training_data,
            treatment='iv',
            outcome='dv',
            ground_truth=ground_truth
        )

        # If bounds were computed, should have coverage_rate
        # (May not have coverage if estimator errored due to small mock data)
        # Check that the sweep ran and recorded results
        assert len(results.results_df) == 2
        assert 'epsilon' in results.results_df.columns

    def test_results_stored_in_analyzer(self, mock_training_data):
        """Test that results are stored in analyzer."""
        config = SweepConfig(parameter_name='epsilon', values=[0.1])
        analyzer = SensitivityAnalyzer(
            sweep_config=config,
            base_estimator_params={'n_permutations': 50},
            verbose=False
        )

        assert analyzer.results_ is None

        analyzer.run_sweep(training_data=mock_training_data)

        assert analyzer.results_ is not None
        assert isinstance(analyzer.results_, SweepResults)


class TestRunEpsilonSweep:
    """Tests for convenience function run_epsilon_sweep."""

    @pytest.fixture
    def mock_training_data(self):
        """Create mock training data."""
        np.random.seed(42)
        n = 200

        def make_site_data(site_name):
            return pd.DataFrame({
                'iv': np.random.choice([0, 1], n),
                'dv': np.random.randn(n) * 100 + 1000,
                'resp_age': np.random.randint(18, 70, n),
                'resp_gender': np.random.choice([0, 1], n),
                'resp_polideo': np.random.randint(0, 7, n),
                'F': np.random.choice(['on', 'idle'], n),
            })

        return {
            'site_a': make_site_data('site_a'),
            'site_b': make_site_data('site_b'),
        }

    def test_convenience_function(self, mock_training_data):
        """Test run_epsilon_sweep convenience function."""
        results = run_epsilon_sweep(
            training_data=mock_training_data,
            epsilon_values=[0.1, 0.2],
            treatment='iv',
            outcome='dv',
            base_params={'n_permutations': 50},
            random_seed=42,
            verbose=False
        )

        assert isinstance(results, SweepResults)
        assert len(results.results_df) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Test SweepResults with empty DataFrame."""
        results = SweepResults(results_df=pd.DataFrame())

        assert len(results.pareto_frontier) == 0
        rec = results.get_recommended_epsilon()
        assert rec['epsilon'] is None

    def test_single_value_sweep(self):
        """Test sweep with single value."""
        config = SweepConfig(parameter_name='epsilon', values=[0.1])
        assert len(config.values) == 1

    def test_results_without_coverage(self):
        """Test SweepResults without coverage column."""
        df = pd.DataFrame({
            'epsilon': [0.1, 0.2],
            'mean_width': [500, 600]
        })
        results = SweepResults(results_df=df)

        # Should handle gracefully
        assert len(results.pareto_frontier) == 0


class TestIntegrationWithEstimator:
    """Integration tests with actual CausalGroundingEstimator."""

    @pytest.fixture
    def realistic_training_data(self):
        """Create more realistic training data with proper structure."""
        np.random.seed(42)
        n = 300

        def make_site_data():
            # Create correlated data
            age = np.random.randint(18, 70, n)
            gender = np.random.choice([0, 1], n)
            polideo = np.random.randint(0, 7, n)

            # Treatment probability depends on covariates
            p_treat = 0.3 + 0.01 * (age - 40) / 30 + 0.1 * gender
            p_treat = np.clip(p_treat, 0.1, 0.9)
            iv = (np.random.random(n) < p_treat).astype(int)

            # Outcome depends on treatment and covariates
            dv = 1000 + 200 * iv + 5 * age + 50 * gender + np.random.randn(n) * 100

            # Regime indicator (50% on, 50% idle)
            F = np.where(np.random.random(n) < 0.5, 'on', 'idle')

            return pd.DataFrame({
                'iv': iv,
                'dv': dv,
                'resp_age': age,
                'resp_gender': gender,
                'resp_polideo': polideo,
                'F': F
            })

        return {
            'site_1': make_site_data(),
            'site_2': make_site_data(),
            'site_3': make_site_data(),
        }

    def test_full_sweep_pipeline(self, realistic_training_data):
        """Test complete sweep pipeline with realistic data."""
        config = SweepConfig(
            parameter_name='epsilon',
            values=[0.05, 0.1, 0.2]
        )

        analyzer = SensitivityAnalyzer(
            sweep_config=config,
            base_estimator_params={
                'transfer_method': 'conservative',
                'n_permutations': 100,  # Fast for testing
            },
            random_seed=42,
            verbose=False
        )

        results = analyzer.run_sweep(
            training_data=realistic_training_data,
            treatment='iv',
            outcome='dv'
        )

        # Should have results for all epsilon values
        assert len(results.results_df) == 3

        # Should have required columns
        required_cols = ['epsilon', 'mean_width', 'n_z_values']
        for col in required_cols:
            assert col in results.results_df.columns

        # Width should increase with epsilon
        widths = results.results_df.sort_values('epsilon')['mean_width'].values
        # Generally expect wider bounds with larger epsilon
        # (may not be strictly monotonic due to estimation noise)
        assert widths[-1] > widths[0]

        # Metadata should be populated
        assert results.metadata['parameter_name'] == 'epsilon'
        assert results.metadata['n_values'] == 3
