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


def aggregate_across_instruments(
    instrument_bounds: Dict[str, Dict[Tuple, Tuple]],
    method: str = 'intersection'
) -> Dict[Tuple, Tuple]:
    """
    Aggregate bounds across multiple instruments.

    This is Ricardo's multi-instrument approach:
    - For each stratum, combine bounds from all valid instruments
    - Intersection: lower = max(lowers), upper = min(uppers)
    - Union: lower = min(lowers), upper = max(uppers) (conservative)

    The intersection approach gives tighter bounds by combining
    information from multiple instruments.

    Args:
        instrument_bounds: Dict[instrument_name -> {z_value: (lower, upper)}]
        method: 'intersection' (tighter) or 'union' (conservative)

    Returns:
        {z_value: (lower, upper)} - aggregated bounds

    Example:
        >>> bounds_z1 = {(0,): (0.1, 0.5), (1,): (0.2, 0.6)}
        >>> bounds_z2 = {(0,): (0.15, 0.45), (1,): (0.18, 0.55)}
        >>> agg = aggregate_across_instruments({'Z1': bounds_z1, 'Z2': bounds_z2})
        >>> # For stratum (0,): max(0.1, 0.15) = 0.15, min(0.5, 0.45) = 0.45
    """
    # Collect all z values across instruments
    all_z_values = set()
    for inst_bounds in instrument_bounds.values():
        all_z_values.update(inst_bounds.keys())

    result = {}

    for z in all_z_values:
        lowers = []
        uppers = []

        for inst_name, inst_bounds in instrument_bounds.items():
            if z in inst_bounds:
                lower, upper = inst_bounds[z]
                lowers.append(lower)
                uppers.append(upper)

        if not lowers:
            continue

        if method == 'intersection':
            # Tighter bounds: max of lowers, min of uppers
            agg_lower = max(lowers)
            agg_upper = min(uppers)

            # Check for empty interval
            if agg_lower > agg_upper:
                # Instruments disagree - take midpoint as both bounds
                midpoint = (agg_lower + agg_upper) / 2
                result[z] = (midpoint, midpoint)
            else:
                result[z] = (agg_lower, agg_upper)

        elif method == 'union':
            # Conservative: min of lowers, max of uppers
            result[z] = (min(lowers), max(uppers))

        else:
            raise ValueError(f"Unknown method: {method}. Use 'intersection' or 'union'.")

    return result


def aggregate_with_weights(
    instrument_bounds: Dict[str, Dict[Tuple, Tuple]],
    instrument_weights: Dict[str, float]
) -> Dict[Tuple, Tuple]:
    """
    Weighted aggregation across instruments.

    Args:
        instrument_bounds: Dict[instrument_name -> {z_value: (lower, upper)}]
        instrument_weights: Dict[instrument_name -> weight] (e.g., from EHS scores)

    Returns:
        {z_value: (lower, upper)} - weighted average bounds
    """
    all_z_values = set()
    for inst_bounds in instrument_bounds.values():
        all_z_values.update(inst_bounds.keys())

    result = {}

    for z in all_z_values:
        weighted_lowers = []
        weighted_uppers = []
        total_weight = 0.0

        for inst_name, inst_bounds in instrument_bounds.items():
            if z in inst_bounds:
                weight = instrument_weights.get(inst_name, 1.0)
                lower, upper = inst_bounds[z]
                weighted_lowers.append(weight * lower)
                weighted_uppers.append(weight * upper)
                total_weight += weight

        if total_weight > 0:
            result[z] = (
                sum(weighted_lowers) / total_weight,
                sum(weighted_uppers) / total_weight
            )

    return result


def compute_instrument_agreement(
    instrument_bounds: Dict[str, Dict[Tuple, Tuple]]
) -> pd.DataFrame:
    """
    Analyze agreement between instruments.

    For each stratum, computes:
    - Overlap: whether all instrument bounds overlap
    - Agreement score: proportion of interval overlap
    - Range of lowers/uppers across instruments

    Args:
        instrument_bounds: Dict[instrument_name -> {z_value: (lower, upper)}]

    Returns:
        DataFrame with agreement metrics per stratum
    """
    all_z_values = set()
    for inst_bounds in instrument_bounds.values():
        all_z_values.update(inst_bounds.keys())

    records = []

    for z in all_z_values:
        lowers = []
        uppers = []
        inst_names = []

        for inst_name, inst_bounds in instrument_bounds.items():
            if z in inst_bounds:
                lower, upper = inst_bounds[z]
                lowers.append(lower)
                uppers.append(upper)
                inst_names.append(inst_name)

        if len(lowers) < 2:
            continue

        # Check overlap
        max_lower = max(lowers)
        min_upper = min(uppers)
        overlap = max_lower <= min_upper

        # Agreement score: width of intersection / width of union
        if overlap:
            intersection_width = min_upper - max_lower
            union_width = max(uppers) - min(lowers)
            agreement_score = intersection_width / union_width if union_width > 0 else 1.0
        else:
            agreement_score = 0.0

        records.append({
            'stratum': str(z),
            'n_instruments': len(lowers),
            'instruments': ', '.join(inst_names),
            'overlap': overlap,
            'agreement_score': agreement_score,
            'lower_range': max(lowers) - min(lowers),
            'upper_range': max(uppers) - min(uppers),
            'intersection_lower': max_lower,
            'intersection_upper': min_upper if overlap else max_lower,
        })

    return pd.DataFrame(records)


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
