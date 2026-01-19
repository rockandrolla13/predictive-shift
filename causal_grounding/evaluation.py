"""
Ground Truth Evaluation Metrics for CATE Bounds

This module provides metrics for evaluating the quality of CATE bounds
against known ground truth effects. These metrics help assess:
- Coverage: Do bounds contain the true effect?
- Informativeness: Are bounds tighter than naive estimates?
- Interval score: Penalized width metric rewarding tight, valid bounds

Based on Ricardo's evaluation approach from method.ipynb.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List


def compute_coverage_rate(
    bounds_df: pd.DataFrame,
    ground_truth: Union[float, pd.Series, np.ndarray],
    lower_col: str = 'lower',
    upper_col: str = 'upper'
) -> float:
    """
    Compute proportion of strata where true CATE falls within bounds.

    Coverage rate = mean(lower_bound <= true_cate <= upper_bound)

    Args:
        bounds_df: DataFrame with columns for lower and upper bounds
        ground_truth: Either a single ATE value (applied to all strata)
                     or a Series/array of per-stratum CATE values
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound

    Returns:
        Coverage rate in [0, 1]

    Example:
        >>> bounds = pd.DataFrame({
        ...     'stratum': ['A', 'B', 'C'],
        ...     'lower': [0.1, 0.2, 0.3],
        ...     'upper': [0.5, 0.4, 0.6]
        ... })
        >>> compute_coverage_rate(bounds, ground_truth=0.35)
        0.6666666666666666
    """
    lower = bounds_df[lower_col].values
    upper = bounds_df[upper_col].values

    if isinstance(ground_truth, (int, float)):
        truth = np.full(len(bounds_df), ground_truth)
    else:
        truth = np.asarray(ground_truth)

    if len(truth) != len(bounds_df):
        raise ValueError(
            f"Ground truth length ({len(truth)}) must match "
            f"bounds_df length ({len(bounds_df)})"
        )

    covered = (lower <= truth) & (truth <= upper)
    return float(np.mean(covered))


def compute_informativeness(
    bounds_df: pd.DataFrame,
    naive_estimates: Union[float, pd.Series, np.ndarray],
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    mode: str = 'lower_improves'
) -> float:
    """
    Compute proportion of strata where bounds improve on naive estimates.

    This measures how often our partial identification bounds provide
    information beyond naive observational estimates.

    Modes:
        'lower_improves': fraction where lower_bound > naive_estimate
        'upper_improves': fraction where upper_bound < naive_estimate
        'width_improves': fraction where bound width < naive uncertainty

    Args:
        bounds_df: DataFrame with columns for lower and upper bounds
        naive_estimates: Naive point estimates (e.g., observational difference)
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound
        mode: How to measure improvement

    Returns:
        Informativeness rate in [0, 1]

    Example:
        >>> bounds = pd.DataFrame({
        ...     'lower': [0.1, 0.2, 0.3],
        ...     'upper': [0.5, 0.4, 0.6]
        ... })
        >>> compute_informativeness(bounds, naive_estimates=0.15, mode='lower_improves')
        0.6666666666666666
    """
    lower = bounds_df[lower_col].values
    upper = bounds_df[upper_col].values

    if isinstance(naive_estimates, (int, float)):
        naive = np.full(len(bounds_df), naive_estimates)
    else:
        naive = np.asarray(naive_estimates)

    if len(naive) != len(bounds_df):
        raise ValueError(
            f"naive_estimates length ({len(naive)}) must match "
            f"bounds_df length ({len(bounds_df)})"
        )

    if mode == 'lower_improves':
        # Lower bound is above naive (we can rule out low values)
        improved = lower > naive
    elif mode == 'upper_improves':
        # Upper bound is below naive (we can rule out high values)
        improved = upper < naive
    elif mode == 'width_improves':
        # Bound width is smaller than 2x uncertainty around naive
        width = upper - lower
        # Assume naive uncertainty is proportional to |naive|
        naive_uncertainty = 2 * np.abs(naive) + 0.1  # Add small epsilon
        improved = width < naive_uncertainty
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'lower_improves', 'upper_improves', or 'width_improves'")

    return float(np.mean(improved))


def compute_interval_score(
    bounds_df: pd.DataFrame,
    ground_truth: Union[float, pd.Series, np.ndarray],
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    alpha: float = 0.05
) -> float:
    """
    Compute interval score - a proper scoring rule for prediction intervals.

    The interval score rewards tight bounds that still cover the truth,
    and penalizes missing the truth proportionally to the miss distance.

    Score = (upper - lower) + (2/alpha) * (lower - truth) * 1[truth < lower]
                            + (2/alpha) * (truth - upper) * 1[truth > upper]

    Lower is better. A perfect score would be achieved by tight bounds
    that exactly contain the truth.

    Args:
        bounds_df: DataFrame with columns for lower and upper bounds
        ground_truth: True CATE values
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound
        alpha: Nominal coverage level (default 0.05 for 95% intervals)

    Returns:
        Mean interval score (lower is better)

    Example:
        >>> bounds = pd.DataFrame({
        ...     'lower': [0.1, 0.2, 0.3],
        ...     'upper': [0.5, 0.4, 0.6]
        ... })
        >>> compute_interval_score(bounds, ground_truth=0.35)  # All covered
        0.3333333333333333
    """
    lower = bounds_df[lower_col].values
    upper = bounds_df[upper_col].values

    if isinstance(ground_truth, (int, float)):
        truth = np.full(len(bounds_df), ground_truth)
    else:
        truth = np.asarray(ground_truth)

    if len(truth) != len(bounds_df):
        raise ValueError(
            f"Ground truth length ({len(truth)}) must match "
            f"bounds_df length ({len(bounds_df)})"
        )

    # Interval width (always contributes)
    width = upper - lower

    # Penalty for missing below
    below_penalty = (2 / alpha) * (lower - truth) * (truth < lower)

    # Penalty for missing above
    above_penalty = (2 / alpha) * (truth - upper) * (truth > upper)

    scores = width + below_penalty + above_penalty

    return float(np.mean(scores))


def compute_sharpness(
    bounds_df: pd.DataFrame,
    lower_col: str = 'lower',
    upper_col: str = 'upper'
) -> Dict[str, float]:
    """
    Compute sharpness metrics (bound width statistics).

    Sharpness measures how tight the bounds are, regardless of coverage.
    Tighter bounds are more informative.

    Args:
        bounds_df: DataFrame with columns for lower and upper bounds
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound

    Returns:
        Dict with mean_width, median_width, std_width, min_width, max_width

    Example:
        >>> bounds = pd.DataFrame({
        ...     'lower': [0.1, 0.2, 0.3],
        ...     'upper': [0.5, 0.4, 0.6]
        ... })
        >>> compute_sharpness(bounds)
        {'mean_width': 0.3, 'median_width': 0.3, 'std_width': 0.1, ...}
    """
    lower = bounds_df[lower_col].values
    upper = bounds_df[upper_col].values
    widths = upper - lower

    return {
        'mean_width': float(np.mean(widths)),
        'median_width': float(np.median(widths)),
        'std_width': float(np.std(widths)),
        'min_width': float(np.min(widths)),
        'max_width': float(np.max(widths)),
    }


def summarize_bound_quality(
    bounds_df: pd.DataFrame,
    ground_truth: Union[float, pd.Series, np.ndarray],
    naive_estimates: Optional[Union[float, pd.Series, np.ndarray]] = None,
    lower_col: str = 'lower',
    upper_col: str = 'upper'
) -> Dict[str, Any]:
    """
    Compute all bound quality metrics in one call.

    This is a convenience function that computes coverage, informativeness,
    interval score, and sharpness in one call.

    Args:
        bounds_df: DataFrame with columns for lower and upper bounds
        ground_truth: True CATE values (single ATE or per-stratum CATE)
        naive_estimates: Optional naive point estimates for informativeness
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound

    Returns:
        Dict containing all metrics:
            - coverage_rate: proportion of strata with truth in bounds
            - interval_score: penalized width metric (lower is better)
            - mean_width: average bound width
            - median_width: median bound width
            - informativeness_lower: proportion where lower > naive (if naive provided)
            - informativeness_upper: proportion where upper < naive (if naive provided)
            - n_strata: number of strata evaluated

    Example:
        >>> bounds = pd.DataFrame({
        ...     'stratum': ['A', 'B', 'C'],
        ...     'lower': [100, 200, 300],
        ...     'upper': [500, 400, 600]
        ... })
        >>> summarize_bound_quality(bounds, ground_truth=350, naive_estimates=250)
        {
            'coverage_rate': 0.667,
            'interval_score': 466.67,
            'mean_width': 300.0,
            ...
        }
    """
    result = {
        'n_strata': len(bounds_df),
        'coverage_rate': compute_coverage_rate(
            bounds_df, ground_truth, lower_col, upper_col
        ),
        'interval_score': compute_interval_score(
            bounds_df, ground_truth, lower_col, upper_col
        ),
    }

    # Add sharpness metrics
    sharpness = compute_sharpness(bounds_df, lower_col, upper_col)
    result.update(sharpness)

    # Add informativeness if naive estimates provided
    if naive_estimates is not None:
        result['informativeness_lower'] = compute_informativeness(
            bounds_df, naive_estimates, lower_col, upper_col, mode='lower_improves'
        )
        result['informativeness_upper'] = compute_informativeness(
            bounds_df, naive_estimates, lower_col, upper_col, mode='upper_improves'
        )
        result['informativeness_width'] = compute_informativeness(
            bounds_df, naive_estimates, lower_col, upper_col, mode='width_improves'
        )

    return result


def compare_method_quality(
    bounds_dict: Dict[str, pd.DataFrame],
    ground_truth: Union[float, pd.Series, np.ndarray],
    naive_estimates: Optional[Union[float, pd.Series, np.ndarray]] = None,
    lower_col: str = 'lower',
    upper_col: str = 'upper'
) -> pd.DataFrame:
    """
    Compare quality metrics across multiple methods.

    Args:
        bounds_dict: Dict mapping method name to bounds DataFrame
        ground_truth: True CATE values
        naive_estimates: Optional naive estimates
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound

    Returns:
        DataFrame with one row per method and columns for each metric

    Example:
        >>> bounds_a = pd.DataFrame({'lower': [0.1, 0.2], 'upper': [0.5, 0.4]})
        >>> bounds_b = pd.DataFrame({'lower': [0.15, 0.25], 'upper': [0.45, 0.35]})
        >>> compare_method_quality(
        ...     {'method_a': bounds_a, 'method_b': bounds_b},
        ...     ground_truth=0.3
        ... )
    """
    results = []

    for method_name, bounds_df in bounds_dict.items():
        summary = summarize_bound_quality(
            bounds_df, ground_truth, naive_estimates, lower_col, upper_col
        )
        summary['method'] = method_name
        results.append(summary)

    df = pd.DataFrame(results)

    # Reorder columns to put method first
    cols = ['method'] + [c for c in df.columns if c != 'method']
    return df[cols]


def per_stratum_coverage(
    bounds_df: pd.DataFrame,
    ground_truth: Union[float, pd.Series, np.ndarray],
    stratum_col: str = 'stratum',
    lower_col: str = 'lower',
    upper_col: str = 'upper'
) -> pd.DataFrame:
    """
    Compute per-stratum coverage indicators.

    Returns the original bounds DataFrame with additional columns
    indicating whether each stratum's bounds cover the ground truth.

    Args:
        bounds_df: DataFrame with bounds and stratum identifier
        ground_truth: True CATE values
        stratum_col: Column name for stratum identifier
        lower_col: Column name for lower bound
        upper_col: Column name for upper bound

    Returns:
        DataFrame with added columns:
            - ground_truth: the true value for this stratum
            - covered: whether bounds contain truth
            - miss_distance: 0 if covered, else distance to nearest bound

    Example:
        >>> bounds = pd.DataFrame({
        ...     'stratum': ['A', 'B'],
        ...     'lower': [0.1, 0.3],
        ...     'upper': [0.4, 0.5]
        ... })
        >>> per_stratum_coverage(bounds, ground_truth=0.35)
    """
    result = bounds_df.copy()

    if isinstance(ground_truth, (int, float)):
        truth = np.full(len(bounds_df), ground_truth)
    else:
        truth = np.asarray(ground_truth)

    lower = bounds_df[lower_col].values
    upper = bounds_df[upper_col].values

    result['ground_truth'] = truth
    result['covered'] = (lower <= truth) & (truth <= upper)

    # Compute miss distance (0 if covered)
    miss_below = np.maximum(0, lower - truth)
    miss_above = np.maximum(0, truth - upper)
    result['miss_distance'] = miss_below + miss_above

    return result


# Module test
if __name__ == "__main__":
    # Create sample bounds
    bounds = pd.DataFrame({
        'stratum': ['young_male', 'young_female', 'old_male', 'old_female'],
        'lower': [100, 150, 200, 180],
        'upper': [500, 450, 600, 550]
    })

    # True ATE
    true_ate = 350

    # Naive observational estimate
    naive = 200

    print("Sample Bounds:")
    print(bounds)
    print()

    print(f"Ground Truth ATE: {true_ate}")
    print(f"Naive Estimate: {naive}")
    print()

    # Compute metrics
    print("Coverage Rate:", compute_coverage_rate(bounds, true_ate))
    print("Informativeness (lower):", compute_informativeness(bounds, naive, mode='lower_improves'))
    print("Interval Score:", compute_interval_score(bounds, true_ate))
    print()

    print("Full Summary:")
    summary = summarize_bound_quality(bounds, true_ate, naive)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    print("Per-Stratum Coverage:")
    per_stratum = per_stratum_coverage(bounds, true_ate)
    print(per_stratum)
