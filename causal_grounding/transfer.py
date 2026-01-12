"""
Bound Transfer Across Environments

This module handles transferring bounds from training to target environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def transfer_bounds_conservative(
    training_bounds: Dict[str, Dict[Tuple, Tuple]],
    z_values: Optional[List[Tuple]] = None
) -> Dict[Tuple, Tuple]:
    """
    Conservative transfer: take widest bounds across training sites.

    For each z: lower = min(lowers), upper = max(uppers)

    Args:
        training_bounds: {site_id: {z_value: (lower, upper)}}
        z_values: Optional subset of z values

    Returns:
        {z_value: (lower, upper)}
    """
    # Collect all z values if not specified
    if z_values is None:
        z_values = set()
        for site_bounds in training_bounds.values():
            z_values.update(site_bounds.keys())
        z_values = list(z_values)

    result = {}

    for z in z_values:
        lowers = []
        uppers = []

        for site_id, site_bounds in training_bounds.items():
            if z in site_bounds:
                lower, upper = site_bounds[z]
                lowers.append(lower)
                uppers.append(upper)

        if lowers:
            result[z] = (min(lowers), max(uppers))
        else:
            result[z] = (-1.0, 1.0)  # Uninformative bounds

    return result


def transfer_bounds_average(
    training_bounds: Dict[str, Dict[Tuple, Tuple]],
    z_values: Optional[List[Tuple]] = None
) -> Dict[Tuple, Tuple]:
    """
    Average transfer: take mean bounds across training sites.

    Less conservative, assumes training sites are representative.
    """
    if z_values is None:
        z_values = set()
        for site_bounds in training_bounds.values():
            z_values.update(site_bounds.keys())
        z_values = list(z_values)

    result = {}

    for z in z_values:
        lowers = []
        uppers = []

        for site_id, site_bounds in training_bounds.items():
            if z in site_bounds:
                lower, upper = site_bounds[z]
                lowers.append(lower)
                uppers.append(upper)

        if lowers:
            result[z] = (np.mean(lowers), np.mean(uppers))
        else:
            result[z] = (-1.0, 1.0)

    return result


def transfer_bounds_weighted(
    training_bounds: Dict[str, Dict[Tuple, Tuple]],
    site_weights: Dict[str, float],
    z_values: Optional[List[Tuple]] = None
) -> Dict[Tuple, Tuple]:
    """
    Weighted transfer: weight bounds by site importance.

    Args:
        training_bounds: {site_id: {z_value: (lower, upper)}}
        site_weights: {site_id: weight} (higher = more important)
        z_values: Optional subset of z values
    """
    if z_values is None:
        z_values = set()
        for site_bounds in training_bounds.values():
            z_values.update(site_bounds.keys())
        z_values = list(z_values)

    result = {}

    for z in z_values:
        weighted_lowers = []
        weighted_uppers = []
        total_weight = 0.0

        for site_id, site_bounds in training_bounds.items():
            if z in site_bounds:
                weight = site_weights.get(site_id, 1.0)
                lower, upper = site_bounds[z]
                weighted_lowers.append(weight * lower)
                weighted_uppers.append(weight * upper)
                total_weight += weight

        if total_weight > 0:
            result[z] = (
                sum(weighted_lowers) / total_weight,
                sum(weighted_uppers) / total_weight
            )
        else:
            result[z] = (-1.0, 1.0)

    return result


def compute_bound_metrics(
    bounds: Dict[Tuple, Tuple],
    true_cate: Optional[Dict[Tuple, float]] = None
) -> Dict[str, float]:
    """
    Compute metrics for bound quality.

    Args:
        bounds: {z_value: (lower, upper)}
        true_cate: Optional ground truth {z_value: cate}

    Returns:
        {
            'mean_width': float,
            'median_width': float,
            'min_width': float,
            'max_width': float,
            'coverage': float (if true_cate provided),
            'n_z_values': int
        }
    """
    widths = [upper - lower for lower, upper in bounds.values()]

    metrics = {
        'mean_width': np.mean(widths),
        'median_width': np.median(widths),
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'n_z_values': len(bounds)
    }

    if true_cate is not None:
        covered = 0
        total = 0
        for z, (lower, upper) in bounds.items():
            if z in true_cate:
                total += 1
                if lower <= true_cate[z] <= upper:
                    covered += 1

        metrics['coverage'] = covered / total if total > 0 else np.nan
        metrics['n_evaluated'] = total

    return metrics


def bounds_to_dataframe(
    bounds: Dict[Tuple, Tuple],
    covariate_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert bounds dict to readable DataFrame.

    Args:
        bounds: {z_value: (lower, upper)}
        covariate_names: Names for z components

    Returns:
        DataFrame with z components, lower, upper, width
    """
    records = []

    for z, (lower, upper) in bounds.items():
        record = {
            'z': z,
            'cate_lower': lower,
            'cate_upper': upper,
            'width': upper - lower
        }

        # Add individual z components
        if covariate_names:
            for i, name in enumerate(covariate_names):
                record[name] = z[i] if i < len(z) else None

        records.append(record)

    df = pd.DataFrame(records)

    # Reorder columns
    z_cols = covariate_names if covariate_names else []
    other_cols = ['cate_lower', 'cate_upper', 'width']
    col_order = z_cols + other_cols + ['z']

    return df[[c for c in col_order if c in df.columns]]


# Test
if __name__ == "__main__":
    # Create mock bounds
    training_bounds = {
        'site_A': {
            (0,): (-0.1, 0.3),
            (1,): (0.1, 0.4),
            (2,): (0.2, 0.5),
        },
        'site_B': {
            (0,): (-0.2, 0.2),
            (1,): (0.0, 0.5),
            (2,): (0.1, 0.6),
        },
        'site_C': {
            (0,): (0.0, 0.25),
            (1,): (0.15, 0.35),
            (2,): (0.25, 0.45),
        },
    }

    print("Conservative transfer:")
    conservative = transfer_bounds_conservative(training_bounds)
    for z, (l, u) in sorted(conservative.items()):
        print(f"  Z={z}: [{l:.2f}, {u:.2f}]")

    print("\nAverage transfer:")
    average = transfer_bounds_average(training_bounds)
    for z, (l, u) in sorted(average.items()):
        print(f"  Z={z}: [{l:.2f}, {u:.2f}]")

    print("\nMetrics (conservative):")
    metrics = compute_bound_metrics(conservative)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    print("\nAs DataFrame:")
    df = bounds_to_dataframe(conservative, covariate_names=['Z'])
    print(df)
