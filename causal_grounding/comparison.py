"""
Method Comparison Utilities

This module provides infrastructure for comparing different implementations
and configurations of the causal grounding methods.

Key classes:
    MethodComparator: Run multiple methods on same data and compare
    ComparisonResults: Store and analyze comparison results

Convenience functions:
    compare_ci_engines: Compare CI testing methods
    compare_predictors: Compare prediction models
    compare_lp_solvers: Compare LP solver implementations
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

from .evaluation import summarize_bound_quality, compare_method_quality


@dataclass
class MethodConfig:
    """Configuration for a single method."""
    name: str
    estimator_class: Any  # Class or factory function
    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ComparisonResults:
    """Container for method comparison results."""
    method_names: List[str]
    bounds: Dict[str, pd.DataFrame]
    metrics: Dict[str, Dict[str, float]]
    runtimes: Dict[str, float]
    ci_scores: Optional[Dict[str, pd.DataFrame]] = None
    predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    def get_bounds_df(self, method: str) -> pd.DataFrame:
        """Get bounds DataFrame for a specific method."""
        return self.bounds.get(method)

    def get_metrics_df(self) -> pd.DataFrame:
        """Get metrics as a DataFrame with one row per method."""
        rows = []
        for method in self.method_names:
            row = {'method': method, **self.metrics.get(method, {})}
            row['runtime'] = self.runtimes.get(method, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_summary(self) -> str:
        """Get human-readable summary of comparison."""
        lines = ["=" * 60, "Method Comparison Summary", "=" * 60, ""]

        df = self.get_metrics_df()
        for _, row in df.iterrows():
            lines.append(f"\n{row['method']}:")
            for col in df.columns:
                if col != 'method':
                    lines.append(f"  {col}: {row[col]:.4f}" if isinstance(row[col], float) else f"  {col}: {row[col]}")

        return "\n".join(lines)


class MethodComparator:
    """
    Run multiple methods on the same data and compare results.

    Example:
        comparator = MethodComparator(data, 'treatment', 'outcome', covariates)
        comparator.add_method('cmi', CausalGroundingEstimator, {'ci_method': 'cmi'})
        comparator.add_method('l1', CausalGroundingEstimator, {'ci_method': 'l1'})
        results = comparator.run_all()
        print(results.get_summary())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        ground_truth: Optional[Union[float, np.ndarray]] = None,
        naive_estimate: Optional[float] = None
    ):
        """
        Initialize comparator with shared data.

        Args:
            data: DataFrame with all variables
            treatment: Treatment column name
            outcome: Outcome column name
            covariates: List of covariate column names
            ground_truth: Optional true CATE for evaluation
            naive_estimate: Optional naive estimate for informativeness
        """
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.ground_truth = ground_truth
        self.naive_estimate = naive_estimate

        self.methods: Dict[str, MethodConfig] = {}
        self._results: Optional[ComparisonResults] = None

    def add_method(
        self,
        name: str,
        estimator_class: Any,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> 'MethodComparator':
        """
        Register a method configuration to compare.

        Args:
            name: Unique identifier for this method
            estimator_class: Estimator class or factory function
            estimator_kwargs: Keyword arguments for estimator
            description: Human-readable description

        Returns:
            Self for method chaining
        """
        self.methods[name] = MethodConfig(
            name=name,
            estimator_class=estimator_class,
            estimator_kwargs=estimator_kwargs or {},
            description=description
        )
        return self

    def run_all(
        self,
        verbose: bool = True,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None
    ) -> ComparisonResults:
        """
        Execute all registered methods and collect results.

        Args:
            verbose: Print progress messages
            fit_kwargs: Additional kwargs for fit method
            predict_kwargs: Additional kwargs for predict_bounds method

        Returns:
            ComparisonResults object
        """
        fit_kwargs = fit_kwargs or {}
        predict_kwargs = predict_kwargs or {}

        bounds = {}
        metrics = {}
        runtimes = {}
        ci_scores = {}

        for name, config in self.methods.items():
            if verbose:
                print(f"Running method: {name}...")

            start_time = time.time()

            try:
                # Create estimator
                estimator = config.estimator_class(**config.estimator_kwargs)

                # Fit
                estimator.fit(
                    self.data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    covariates=self.covariates,
                    **fit_kwargs
                )

                # Predict bounds
                method_bounds = estimator.predict_bounds(**predict_kwargs)
                bounds[name] = method_bounds

                # Get CI scores if available
                if hasattr(estimator, 'covariate_scores_') and estimator.covariate_scores_ is not None:
                    ci_scores[name] = estimator.covariate_scores_

            except Exception as e:
                if verbose:
                    print(f"  Error in {name}: {e}")
                bounds[name] = pd.DataFrame()
                continue

            elapsed = time.time() - start_time
            runtimes[name] = elapsed

            # Compute metrics if ground truth available
            if self.ground_truth is not None and len(bounds[name]) > 0:
                metrics[name] = summarize_bound_quality(
                    bounds[name],
                    self.ground_truth,
                    self.naive_estimate
                )
            else:
                metrics[name] = {}

            if verbose:
                print(f"  Completed in {elapsed:.2f}s")

        self._results = ComparisonResults(
            method_names=list(self.methods.keys()),
            bounds=bounds,
            metrics=metrics,
            runtimes=runtimes,
            ci_scores=ci_scores if ci_scores else None
        )

        return self._results

    def compare_bounds(self) -> pd.DataFrame:
        """
        Return DataFrame comparing bounds across methods.

        Returns wide-format DataFrame with columns for each method's bounds.
        """
        if self._results is None:
            raise RuntimeError("Must call run_all() before compare_bounds()")

        dfs = []
        for name, bounds_df in self._results.bounds.items():
            if len(bounds_df) == 0:
                continue
            df = bounds_df.copy()
            df = df.rename(columns={
                'lower': f'{name}_lower',
                'upper': f'{name}_upper'
            })
            # Keep only the bound columns we renamed
            keep_cols = [c for c in df.columns if c.startswith(name)]
            if 'stratum' in bounds_df.columns:
                df['stratum'] = bounds_df['stratum']
                keep_cols = ['stratum'] + keep_cols
            elif 'z_value' in bounds_df.columns:
                df['stratum'] = bounds_df['z_value']
                keep_cols = ['stratum'] + keep_cols
            dfs.append(df[keep_cols])

        if not dfs:
            return pd.DataFrame()

        # Merge on stratum
        result = dfs[0]
        for df in dfs[1:]:
            if 'stratum' in result.columns and 'stratum' in df.columns:
                result = result.merge(df, on='stratum', how='outer')
            else:
                result = pd.concat([result.reset_index(drop=True),
                                   df.reset_index(drop=True)], axis=1)

        return result

    def compare_coverage(
        self,
        ground_truth: Optional[Union[float, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Return coverage comparison across methods.

        Args:
            ground_truth: Override ground truth for this comparison

        Returns:
            DataFrame with coverage metrics per method
        """
        if self._results is None:
            raise RuntimeError("Must call run_all() before compare_coverage()")

        truth = ground_truth if ground_truth is not None else self.ground_truth
        if truth is None:
            raise ValueError("ground_truth must be provided")

        return compare_method_quality(
            self._results.bounds,
            truth,
            self.naive_estimate
        )

    def compare_runtime(self) -> pd.DataFrame:
        """Return runtime comparison as DataFrame."""
        if self._results is None:
            raise RuntimeError("Must call run_all() before compare_runtime()")

        return pd.DataFrame({
            'method': list(self._results.runtimes.keys()),
            'runtime_seconds': list(self._results.runtimes.values())
        }).sort_values('runtime_seconds')

    def generate_comparison_report(
        self,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Path:
        """
        Generate full comparison report with visualizations.

        Args:
            output_dir: Directory to save report files
            include_plots: Whether to generate plots

        Returns:
            Path to main report file
        """
        from .visualization import (
            plot_bounds_comparison,
            plot_method_comparison_summary,
            plot_runtime_comparison,
            save_figure
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._results is None:
            raise RuntimeError("Must call run_all() before generating report")

        # Save metrics CSV
        metrics_df = self._results.get_metrics_df()
        metrics_df.to_csv(output_dir / 'metrics_comparison.csv', index=False)

        # Save bounds comparison CSV
        bounds_df = self.compare_bounds()
        if len(bounds_df) > 0:
            bounds_df.to_csv(output_dir / 'bounds_comparison.csv', index=False)

        # Generate plots
        if include_plots:
            try:
                # Bounds comparison plot
                if len(self._results.bounds) > 0:
                    fig = plot_bounds_comparison(
                        self._results.bounds,
                        true_cate=self.ground_truth
                    )
                    save_figure(fig, output_dir / 'bounds_comparison.png')

                # Summary metrics plot
                if self._results.metrics:
                    fig = plot_method_comparison_summary(self._results.metrics)
                    save_figure(fig, output_dir / 'metrics_summary.png')

                # Runtime comparison
                if self._results.runtimes:
                    fig = plot_runtime_comparison(self._results.runtimes)
                    save_figure(fig, output_dir / 'runtime_comparison.png')

            except Exception as e:
                print(f"Warning: Could not generate some plots: {e}")

        # Generate markdown report
        report_path = output_dir / 'COMPARISON_REPORT.md'
        self._write_markdown_report(report_path, metrics_df)

        return report_path

    def _write_markdown_report(
        self,
        path: Path,
        metrics_df: pd.DataFrame
    ) -> None:
        """Write markdown report file."""
        lines = [
            "# Method Comparison Report",
            "",
            f"**Methods compared:** {', '.join(self.methods.keys())}",
            f"**Data size:** {len(self.data)} samples",
            f"**Covariates:** {', '.join(self.covariates)}",
            "",
            "## Metrics Summary",
            "",
            metrics_df.to_markdown(index=False),
            "",
            "## Runtime Comparison",
            "",
            self.compare_runtime().to_markdown(index=False),
            "",
        ]

        if self.ground_truth is not None:
            lines.extend([
                "## Coverage Analysis",
                "",
                self.compare_coverage().to_markdown(index=False),
                "",
            ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compare_ci_engines(
    data: pd.DataFrame,
    engines_dict: Dict[str, Any],
    treatment: str,
    outcome: str,
    covariates: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare different CI testing engines.

    Args:
        data: DataFrame with all variables
        engines_dict: Dict mapping name to CITestEngine instance
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of covariate column names
        verbose: Print progress

    Returns:
        Dict with comparison results
    """
    results = {
        'scores': {},
        'runtimes': {},
        'best_instruments': {}
    }

    for name, engine in engines_dict.items():
        if verbose:
            print(f"Running CI engine: {name}...")

        start = time.time()

        # Score all covariates
        all_scores = []
        for z_a in covariates:
            z_b = [c for c in covariates if c != z_a]
            score = engine.score_ehs_criteria(
                data, z_a, z_b, treatment, outcome
            )
            all_scores.append(score)

        scores_df = pd.DataFrame(all_scores).sort_values('score', ascending=False)
        results['scores'][name] = scores_df
        results['runtimes'][name] = time.time() - start
        results['best_instruments'][name] = scores_df.iloc[0]['z_a'] if len(scores_df) > 0 else None

        if verbose:
            print(f"  Best instrument: {results['best_instruments'][name]}")
            print(f"  Runtime: {results['runtimes'][name]:.2f}s")

    return results


def compare_predictors(
    data: pd.DataFrame,
    predictors_dict: Dict[str, Any],
    treatment: str,
    outcome: str,
    covariates: List[str],
    test_data: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare different prediction models.

    Args:
        data: Training DataFrame
        predictors_dict: Dict mapping name to predictor instance
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of covariate column names
        test_data: Optional test DataFrame
        verbose: Print progress

    Returns:
        Dict with comparison results
    """
    results = {
        'predictions': {},
        'runtimes': {},
        'train_metrics': {},
        'test_metrics': {}
    }

    for name, predictor in predictors_dict.items():
        if verbose:
            print(f"Fitting predictor: {name}...")

        start = time.time()

        # Fit predictor
        predictor.fit(data, treatment, outcome, covariates)

        # Get predictions
        X = data[covariates].values
        A = data[treatment].values

        results['predictions'][name] = {
            'propensity': predictor.predict_propensity(X) if hasattr(predictor, 'predict_propensity') else None,
            'outcome': predictor.predict_outcome(X, A) if hasattr(predictor, 'predict_outcome') else None
        }

        results['runtimes'][name] = time.time() - start

        if verbose:
            print(f"  Runtime: {results['runtimes'][name]:.2f}s")

    return results


def compare_lp_solvers(
    probabilities: Dict[str, Any],
    solvers_dict: Dict[str, Any],
    epsilon: float = 0.1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare different LP solver implementations.

    Args:
        probabilities: Dict with probability estimates needed for LP
        solvers_dict: Dict mapping name to LP solver instance
        epsilon: Naturalness tolerance
        verbose: Print progress

    Returns:
        Dict with comparison results
    """
    results = {
        'bounds': {},
        'runtimes': {},
        'widths': {}
    }

    for name, solver in solvers_dict.items():
        if verbose:
            print(f"Running LP solver: {name}...")

        start = time.time()

        # Solve bounds
        bounds = solver.solve(probabilities, epsilon=epsilon)
        results['bounds'][name] = bounds
        results['runtimes'][name] = time.time() - start

        if bounds is not None:
            if isinstance(bounds, pd.DataFrame):
                widths = bounds['upper'] - bounds['lower']
            else:
                widths = np.array([b[1] - b[0] for b in bounds.values()])
            results['widths'][name] = {
                'mean': float(np.mean(widths)),
                'median': float(np.median(widths)),
                'min': float(np.min(widths)),
                'max': float(np.max(widths))
            }

        if verbose:
            print(f"  Runtime: {results['runtimes'][name]:.2f}s")
            if name in results['widths']:
                print(f"  Mean width: {results['widths'][name]['mean']:.4f}")

    return results


def quick_compare(
    data: pd.DataFrame,
    configs: List[Dict[str, Any]],
    treatment: str,
    outcome: str,
    covariates: List[str],
    ground_truth: Optional[float] = None
) -> pd.DataFrame:
    """
    Quick comparison of multiple configurations.

    Args:
        data: DataFrame with all variables
        configs: List of config dicts with 'name', 'class', 'kwargs' keys
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: Covariate column names
        ground_truth: Optional true CATE

    Returns:
        DataFrame with comparison results
    """
    comparator = MethodComparator(
        data, treatment, outcome, covariates,
        ground_truth=ground_truth
    )

    for config in configs:
        comparator.add_method(
            config['name'],
            config['class'],
            config.get('kwargs', {})
        )

    results = comparator.run_all()
    return results.get_metrics_df()


# Module test
if __name__ == "__main__":
    print("Comparison module loaded successfully.")
    print("\nAvailable classes:")
    print("  - MethodComparator")
    print("  - ComparisonResults")
    print("  - MethodConfig")
    print("\nAvailable functions:")
    print("  - compare_ci_engines")
    print("  - compare_predictors")
    print("  - compare_lp_solvers")
    print("  - quick_compare")
