"""
Sensitivity Analysis for Parameter Sweeps

This module provides tools for analyzing how parameter choices affect
CATE bound estimation, particularly the precision-coverage tradeoff.

Key classes:
- SweepConfig: Configuration for parameter sweeps
- SweepResults: Structured results from sensitivity analysis
- SensitivityAnalyzer: Main class for running parameter sweeps
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

from .estimator import CausalGroundingEstimator
from .transfer import compute_bound_metrics


@dataclass
class SweepConfig:
    """
    Configuration for parameter sweeps.

    Can specify exact values OR auto-generate a grid.

    Attributes:
        parameter_name: Name of parameter to sweep (e.g., 'epsilon')
        values: Explicit list of values to test
        n_points: Number of points to auto-generate (if values not provided)
        range: (min, max) tuple for auto-generation
        log_scale: Whether to use log-spaced values

    Example:
        # Explicit values
        config = SweepConfig(
            parameter_name='epsilon',
            values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        )

        # Auto-generated
        config = SweepConfig(
            parameter_name='epsilon',
            n_points=20,
            range=(0.01, 0.5),
            log_scale=False
        )
    """
    parameter_name: str
    values: Optional[List[float]] = None
    n_points: Optional[int] = None
    range: Optional[Tuple[float, float]] = None
    log_scale: bool = False

    def __post_init__(self):
        """Generate values if not provided."""
        if self.values is None:
            if self.n_points is None or self.range is None:
                raise ValueError(
                    "Must provide either 'values' or both 'n_points' and 'range'"
                )
            self.values = self._generate_values()

    def _generate_values(self) -> List[float]:
        """Generate sweep values based on configuration."""
        min_val, max_val = self.range

        if self.log_scale:
            values = np.logspace(
                np.log10(min_val),
                np.log10(max_val),
                self.n_points
            )
        else:
            values = np.linspace(min_val, max_val, self.n_points)

        return values.tolist()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SweepConfig':
        """Create SweepConfig from dictionary."""
        return cls(
            parameter_name=config_dict['parameter'],
            values=config_dict.get('values'),
            n_points=config_dict.get('n_points'),
            range=tuple(config_dict['range']) if 'range' in config_dict else None,
            log_scale=config_dict.get('log_scale', False)
        )


@dataclass
class SweepResults:
    """
    Structured results from sensitivity analysis.

    Attributes:
        results_df: DataFrame with one row per parameter configuration
        pareto_frontier: DataFrame with Pareto-optimal points
        best_params: Dictionary with recommended parameters
        metadata: Experiment configuration and metadata
    """
    results_df: pd.DataFrame
    pareto_frontier: pd.DataFrame = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute Pareto frontier if not provided."""
        if self.pareto_frontier is None:
            if len(self.results_df) > 0:
                self.pareto_frontier = self._compute_pareto_frontier()
            else:
                self.pareto_frontier = pd.DataFrame()

    def _compute_pareto_frontier(self) -> pd.DataFrame:
        """
        Extract Pareto-optimal points (maximize coverage, minimize width).
        """
        df = self.results_df.copy()

        if 'coverage_rate' not in df.columns or 'mean_width' not in df.columns:
            return pd.DataFrame()

        # Sort by width (ascending)
        df = df.sort_values('mean_width')

        pareto_points = []
        max_coverage_seen = -np.inf

        for _, row in df.iterrows():
            if row['coverage_rate'] > max_coverage_seen:
                pareto_points.append(row)
                max_coverage_seen = row['coverage_rate']

        return pd.DataFrame(pareto_points)

    def get_recommended_epsilon(
        self,
        target_coverage: float = 0.5,
        prefer: str = 'coverage'
    ) -> Dict[str, Any]:
        """
        Recommend epsilon value based on target criteria.

        Args:
            target_coverage: Minimum acceptable coverage rate
            prefer: 'coverage' (maximize coverage) or 'width' (minimize width)

        Returns:
            Dict with recommended epsilon and associated metrics
        """
        df = self.results_df

        if len(df) == 0:
            return {'epsilon': None, 'reason': 'No results'}

        # Filter to points meeting coverage target
        candidates = df[df['coverage_rate'] >= target_coverage]

        if len(candidates) == 0:
            # No points meet target - return highest coverage
            best_idx = df['coverage_rate'].idxmax()
            return {
                'epsilon': df.loc[best_idx, 'epsilon'],
                'coverage_rate': df.loc[best_idx, 'coverage_rate'],
                'mean_width': df.loc[best_idx, 'mean_width'],
                'reason': f'No config meets {target_coverage:.0%} coverage; using best available'
            }

        if prefer == 'width':
            # Among candidates, choose narrowest bounds
            best_idx = candidates['mean_width'].idxmin()
        else:
            # Among candidates, choose highest coverage
            best_idx = candidates['coverage_rate'].idxmax()

        return {
            'epsilon': candidates.loc[best_idx, 'epsilon'],
            'coverage_rate': candidates.loc[best_idx, 'coverage_rate'],
            'mean_width': candidates.loc[best_idx, 'mean_width'],
            'reason': f'Best {prefer} meeting {target_coverage:.0%} target'
        }

    def to_csv(self, path: Union[str, Path]) -> None:
        """Save results to CSV."""
        self.results_df.to_csv(path, index=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save summary to JSON."""
        summary = {
            'n_configurations': len(self.results_df),
            'parameter_range': [
                self.results_df['epsilon'].min(),
                self.results_df['epsilon'].max()
            ],
            'coverage_range': [
                self.results_df['coverage_rate'].min(),
                self.results_df['coverage_rate'].max()
            ],
            'width_range': [
                self.results_df['mean_width'].min(),
                self.results_df['mean_width'].max()
            ],
            'recommended': self.get_recommended_epsilon(),
            'metadata': self.metadata
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


class SensitivityAnalyzer:
    """
    Analyze sensitivity of CATE bounds to parameter choices.

    This class wraps CausalGroundingEstimator and runs it across
    a grid of parameter values to characterize the precision-coverage
    tradeoff.

    Example:
        analyzer = SensitivityAnalyzer(
            sweep_config=SweepConfig(
                parameter_name='epsilon',
                values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
            ),
            base_estimator_params={
                'transfer_method': 'conservative',
                'n_permutations': 500
            }
        )

        results = analyzer.run_sweep(
            training_data=training_data,
            treatment='iv',
            outcome='dv',
            ground_truth=true_cates
        )

        print(results.get_recommended_epsilon(target_coverage=0.5))
    """

    def __init__(
        self,
        sweep_config: SweepConfig,
        base_estimator_params: Optional[Dict[str, Any]] = None,
        metrics_to_track: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            sweep_config: Configuration for parameter sweep
            base_estimator_params: Fixed parameters for CausalGroundingEstimator
            metrics_to_track: List of metrics to compute (default: all)
            random_seed: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.sweep_config = sweep_config
        self.base_estimator_params = base_estimator_params or {}
        self.metrics_to_track = metrics_to_track
        self.random_seed = random_seed
        self.verbose = verbose

        # Results storage
        self.results_: Optional[SweepResults] = None

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def run_sweep(
        self,
        training_data: Dict[str, pd.DataFrame],
        treatment: str = 'iv',
        outcome: str = 'dv',
        covariates: Optional[List[str]] = None,
        regime_col: str = 'F',
        ground_truth: Optional[Dict[Tuple, float]] = None
    ) -> SweepResults:
        """
        Run parameter sweep and collect results.

        Args:
            training_data: Dict[site_id -> DataFrame with F column]
            treatment: Treatment column name
            outcome: Outcome column name
            covariates: Covariate columns
            regime_col: Regime indicator column
            ground_truth: Optional true CATE values for coverage computation

        Returns:
            SweepResults with analysis results
        """
        param_name = self.sweep_config.parameter_name
        param_values = self.sweep_config.values

        self._log(f"Running sensitivity sweep on '{param_name}'")
        self._log(f"  Values: {param_values}")
        self._log(f"  {len(param_values)} configurations to test")

        results = []

        for i, param_value in enumerate(param_values):
            self._log(f"  [{i+1}/{len(param_values)}] {param_name}={param_value}")

            # Build estimator params
            estimator_params = self.base_estimator_params.copy()
            estimator_params[param_name] = param_value

            if self.random_seed is not None:
                estimator_params['random_seed'] = self.random_seed

            # Suppress verbose output for individual runs
            estimator_params['verbose'] = False

            # Create and fit estimator
            try:
                estimator = CausalGroundingEstimator(**estimator_params)

                estimator.fit(
                    training_data,
                    treatment=treatment,
                    outcome=outcome,
                    covariates=covariates,
                    regime_col=regime_col
                )

                # Get diagnostics
                diagnostics = estimator.get_diagnostics()

                # Compute coverage if ground truth provided
                if ground_truth is not None:
                    coverage_metrics = compute_bound_metrics(
                        estimator.transferred_bounds_,
                        true_cate=ground_truth
                    )
                    diagnostics['coverage_rate'] = coverage_metrics.get('coverage', np.nan)
                    diagnostics['n_evaluated'] = coverage_metrics.get('n_evaluated', 0)

                # Record result
                result = {
                    param_name: param_value,
                    **diagnostics
                }
                results.append(result)

            except Exception as e:
                self._log(f"    Error: {e}")
                results.append({
                    param_name: param_value,
                    'error': str(e)
                })

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Build metadata
        metadata = {
            'parameter_name': param_name,
            'n_values': len(param_values),
            'base_params': self.base_estimator_params,
            'random_seed': self.random_seed,
            'has_ground_truth': ground_truth is not None
        }

        self.results_ = SweepResults(
            results_df=results_df,
            metadata=metadata
        )

        self._log("  Sweep complete!")
        self._log(f"  Coverage range: {results_df.get('coverage_rate', pd.Series()).min():.1%} - {results_df.get('coverage_rate', pd.Series()).max():.1%}")

        return self.results_

    def plot_results(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = 'sensitivity'
    ) -> Dict[str, Any]:
        """
        Generate visualization plots for sweep results.

        Args:
            output_dir: Directory to save plots (None = don't save)
            prefix: Filename prefix for saved plots

        Returns:
            Dict with plot figure objects
        """
        if self.results_ is None:
            raise RuntimeError("No results available. Run run_sweep() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("matplotlib not available, skipping plots")
            return {}

        df = self.results_.results_df
        param_name = self.sweep_config.parameter_name

        plots = {}

        # Plot 1: Coverage vs Parameter
        if 'coverage_rate' in df.columns:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(df[param_name], df['coverage_rate'] * 100, 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel(param_name.capitalize(), fontsize=12)
            ax1.set_ylabel('Coverage Rate (%)', fontsize=12)
            ax1.set_title(f'Coverage vs {param_name.capitalize()}', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)

            plots['coverage_curve'] = fig1

            if output_dir:
                fig1.savefig(Path(output_dir) / f'{prefix}_coverage_curve.png', dpi=150, bbox_inches='tight')

        # Plot 2: Width vs Parameter
        if 'mean_width' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(df[param_name], df['mean_width'], 'r-o', linewidth=2, markersize=8)
            ax2.set_xlabel(param_name.capitalize(), fontsize=12)
            ax2.set_ylabel('Mean Bound Width', fontsize=12)
            ax2.set_title(f'Bound Width vs {param_name.capitalize()}', fontsize=14)
            ax2.grid(True, alpha=0.3)

            plots['width_curve'] = fig2

            if output_dir:
                fig2.savefig(Path(output_dir) / f'{prefix}_width_curve.png', dpi=150, bbox_inches='tight')

        # Plot 3: Pareto frontier (Width vs Coverage)
        if 'coverage_rate' in df.columns and 'mean_width' in df.columns:
            fig3, ax3 = plt.subplots(figsize=(8, 6))

            # All points
            scatter = ax3.scatter(
                df['mean_width'],
                df['coverage_rate'] * 100,
                c=df[param_name],
                cmap='viridis',
                s=100,
                alpha=0.7
            )

            # Pareto frontier
            pareto = self.results_.pareto_frontier
            if len(pareto) > 0:
                ax3.plot(
                    pareto['mean_width'],
                    pareto['coverage_rate'] * 100,
                    'r--',
                    linewidth=2,
                    label='Pareto Frontier'
                )

            plt.colorbar(scatter, label=param_name.capitalize())
            ax3.set_xlabel('Mean Bound Width', fontsize=12)
            ax3.set_ylabel('Coverage Rate (%)', fontsize=12)
            ax3.set_title('Coverage-Width Tradeoff', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            plots['pareto'] = fig3

            if output_dir:
                fig3.savefig(Path(output_dir) / f'{prefix}_pareto.png', dpi=150, bbox_inches='tight')

        if output_dir:
            plt.close('all')

        return plots


def run_epsilon_sweep(
    training_data: Dict[str, pd.DataFrame],
    epsilon_values: List[float],
    treatment: str = 'iv',
    outcome: str = 'dv',
    ground_truth: Optional[Dict[Tuple, float]] = None,
    base_params: Optional[Dict] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> SweepResults:
    """
    Convenience function to run epsilon sweep.

    Args:
        training_data: Training data dict
        epsilon_values: List of epsilon values to test
        treatment: Treatment column
        outcome: Outcome column
        ground_truth: Optional true CATE values
        base_params: Base estimator parameters
        random_seed: Random seed
        verbose: Print progress

    Returns:
        SweepResults
    """
    config = SweepConfig(
        parameter_name='epsilon',
        values=epsilon_values
    )

    analyzer = SensitivityAnalyzer(
        sweep_config=config,
        base_estimator_params=base_params or {},
        random_seed=random_seed,
        verbose=verbose
    )

    return analyzer.run_sweep(
        training_data=training_data,
        treatment=treatment,
        outcome=outcome,
        ground_truth=ground_truth
    )


# Test
if __name__ == "__main__":
    print("SensitivityAnalyzer Test")
    print("=" * 50)

    # Test SweepConfig
    print("\n1. Testing SweepConfig:")

    # Explicit values
    config1 = SweepConfig(
        parameter_name='epsilon',
        values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    )
    print(f"   Explicit values: {config1.values}")

    # Auto-generated (linear)
    config2 = SweepConfig(
        parameter_name='epsilon',
        n_points=5,
        range=(0.01, 0.5),
        log_scale=False
    )
    print(f"   Auto-generated (linear): {[f'{v:.3f}' for v in config2.values]}")

    # Auto-generated (log)
    config3 = SweepConfig(
        parameter_name='epsilon',
        n_points=5,
        range=(0.01, 1.0),
        log_scale=True
    )
    print(f"   Auto-generated (log): {[f'{v:.3f}' for v in config3.values]}")

    # Test from_dict
    config4 = SweepConfig.from_dict({
        'parameter': 'epsilon',
        'values': [0.1, 0.2, 0.3]
    })
    print(f"   From dict: {config4.values}")

    print("\n2. Testing SweepResults:")

    # Create mock results
    results_df = pd.DataFrame({
        'epsilon': [0.05, 0.1, 0.15, 0.2, 0.3],
        'coverage_rate': [0.30, 0.40, 0.50, 0.55, 0.60],
        'mean_width': [500, 600, 700, 800, 1000],
        'n_z_values': [30, 30, 30, 30, 30]
    })

    results = SweepResults(results_df=results_df)

    print(f"   Pareto frontier points: {len(results.pareto_frontier)}")
    print(f"   Recommended (50% target): {results.get_recommended_epsilon(0.5)}")
    print(f"   Recommended (60% target): {results.get_recommended_epsilon(0.6)}")

    print("\nAll tests passed!")
