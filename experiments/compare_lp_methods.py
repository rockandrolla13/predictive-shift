#!/usr/bin/env python3
"""
Compare LP Bound Methods: Heuristic vs True LP

This script compares three approaches to computing CATE bounds:
1. Heuristic: Current sensitivity band approach (CATE ± ε*σ)
2. LP closed-form: True LP bounds using naturalness constraints
3. LP CVXPY: Same as closed-form, verified with CVXPY solver

Requires binary outcome Y for LP methods.

Usage:
    python experiments/compare_lp_methods.py --study anchoring1 --beta 0.25
    python experiments/compare_lp_methods.py --study anchoring1 --beta 0.25 --binarize
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causal_grounding import (
    create_train_target_split,
    load_rct_data,
    discretize_covariates,
    get_discretized_covariate_names,
)
from causal_grounding.lp_solver import (
    solve_all_bounds,
    solve_all_bounds_binary_lp,
    estimate_conditional_means,
)
from causal_grounding.transfer import (
    transfer_bounds_conservative,
    compute_bound_metrics,
)


def binarize_outcome(data: pd.DataFrame, outcome: str, method: str = 'median') -> pd.DataFrame:
    """
    Binarize continuous outcome for LP bounds.

    Args:
        data: DataFrame with outcome column
        outcome: Outcome column name
        method: 'median' (split at median) or 'positive' (Y > 0)

    Returns:
        DataFrame with binarized outcome
    """
    data = data.copy()

    if method == 'median':
        threshold = data[outcome].median()
        data[outcome] = (data[outcome] > threshold).astype(int)
    elif method == 'positive':
        data[outcome] = (data[outcome] > 0).astype(int)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return data


def compute_ground_truth_cate_binary(
    rct_data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str]
) -> Dict[Tuple, float]:
    """
    Compute ground truth CATE from RCT data for binary outcome.

    Returns:
        Dict[z_value -> CATE]
    """
    true_cates = {}

    for z_value, group in rct_data.groupby(covariates, observed=True):
        z_tuple = tuple(z_value) if hasattr(z_value, '__iter__') else (z_value,)

        treated = group[group[treatment] == 1][outcome]
        control = group[group[treatment] == 0][outcome]

        if len(treated) >= 5 and len(control) >= 5:
            # For binary Y, CATE = P(Y=1|X=1) - P(Y=1|X=0)
            true_cates[z_tuple] = treated.mean() - control.mean()

    return true_cates


def run_comparison(
    training_data: Dict[str, pd.DataFrame],
    rct_data: pd.DataFrame,
    epsilon: float = 0.1,
    treatment: str = 'iv',
    outcome: str = 'dv',
    covariates: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comparison of bound methods.

    Returns:
        DataFrame with comparison results
    """
    if covariates is None:
        covariates = get_discretized_covariate_names()

    # Compute ground truth
    true_cates = compute_ground_truth_cate_binary(rct_data, treatment, outcome, covariates)

    if verbose:
        print(f"Ground truth CATE computed for {len(true_cates)} strata")

    results = []

    # Method 1: Heuristic (current approach)
    if verbose:
        print("\n1. Running heuristic method...")
    t0 = time.time()

    heuristic_bounds_per_site = solve_all_bounds(
        training_data, covariates, treatment, outcome,
        epsilon=epsilon, use_full_lp=False
    )
    heuristic_bounds = transfer_bounds_conservative(heuristic_bounds_per_site)

    t_heuristic = time.time() - t0
    heuristic_metrics = compute_bound_metrics(heuristic_bounds, true_cates)

    results.append({
        'method': 'Heuristic (sensitivity)',
        'mean_width': heuristic_metrics['mean_width'],
        'coverage': heuristic_metrics.get('coverage', np.nan),
        'n_z_values': heuristic_metrics['n_z_values'],
        'time_seconds': t_heuristic
    })

    if verbose:
        print(f"   Width: {heuristic_metrics['mean_width']:.4f}")
        print(f"   Coverage: {heuristic_metrics.get('coverage', 'N/A')}")

    # Method 2: LP closed-form
    if verbose:
        print("\n2. Running LP closed-form method...")
    t0 = time.time()

    lp_closed_bounds = solve_all_bounds_binary_lp(
        training_data, covariates, treatment, outcome,
        epsilon=epsilon, use_cvxpy=False
    )

    t_lp_closed = time.time() - t0
    lp_closed_metrics = compute_bound_metrics(lp_closed_bounds, true_cates)

    results.append({
        'method': 'LP closed-form',
        'mean_width': lp_closed_metrics['mean_width'],
        'coverage': lp_closed_metrics.get('coverage', np.nan),
        'n_z_values': lp_closed_metrics['n_z_values'],
        'time_seconds': t_lp_closed
    })

    if verbose:
        print(f"   Width: {lp_closed_metrics['mean_width']:.4f}")
        print(f"   Coverage: {lp_closed_metrics.get('coverage', 'N/A')}")

    # Method 3: LP CVXPY
    if verbose:
        print("\n3. Running LP CVXPY method...")
    t0 = time.time()

    try:
        lp_cvxpy_bounds = solve_all_bounds_binary_lp(
            training_data, covariates, treatment, outcome,
            epsilon=epsilon, use_cvxpy=True
        )

        t_lp_cvxpy = time.time() - t0
        lp_cvxpy_metrics = compute_bound_metrics(lp_cvxpy_bounds, true_cates)

        results.append({
            'method': 'LP CVXPY',
            'mean_width': lp_cvxpy_metrics['mean_width'],
            'coverage': lp_cvxpy_metrics.get('coverage', np.nan),
            'n_z_values': lp_cvxpy_metrics['n_z_values'],
            'time_seconds': t_lp_cvxpy
        })

        if verbose:
            print(f"   Width: {lp_cvxpy_metrics['mean_width']:.4f}")
            print(f"   Coverage: {lp_cvxpy_metrics.get('coverage', 'N/A')}")

    except ImportError:
        if verbose:
            print("   CVXPY not available, skipping")
        results.append({
            'method': 'LP CVXPY',
            'mean_width': np.nan,
            'coverage': np.nan,
            'n_z_values': 0,
            'time_seconds': np.nan
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compare LP bound methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--study', '-s', type=str, default='anchoring1',
                        help='ManyLabs study name')
    parser.add_argument('--beta', '-b', type=float, default=0.25,
                        help='Confounding strength')
    parser.add_argument('--epsilon', '-e', type=float, default=0.1,
                        help='Naturalness tolerance')
    parser.add_argument('--target-site', type=str, default='mturk',
                        help='Target site to hold out')
    parser.add_argument('--binarize', action='store_true',
                        help='Binarize outcome (median split)')
    parser.add_argument('--output', '-o', type=str, default='results/lp_comparison/',
                        help='Output directory')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("LP Bounds Method Comparison")
        print("=" * 60)
        print(f"Study: {args.study}")
        print(f"Beta: {args.beta}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Binarize: {args.binarize}")
        print("=" * 60)

    # Load data
    osrct_path = project_root / 'confounded_datasets' / args.study / f'age_beta{args.beta}_seed42.csv'
    rct_path = project_root / 'ManyLabs1' / 'pre-process' / 'Manylabs1_data.pkl'

    if not osrct_path.exists():
        print(f"Error: OSRCT data not found at {osrct_path}")
        return 1

    if verbose:
        print("\nLoading data...")

    osrct_data = pd.read_csv(osrct_path)
    rct_data = load_rct_data(args.study, str(rct_path))

    # Binarize if requested
    if args.binarize:
        if verbose:
            print("Binarizing outcome (median split)...")
        osrct_data = binarize_outcome(osrct_data, 'dv', method='median')
        rct_data = binarize_outcome(rct_data, 'dv', method='median')

    # Discretize covariates
    osrct_data = discretize_covariates(osrct_data)
    rct_data = discretize_covariates(rct_data)

    # Create train/target split
    training_data, target_data = create_train_target_split(
        osrct_data, rct_data, target_site=args.target_site
    )

    if verbose:
        print(f"Training sites: {len(training_data)}")
        print(f"Target samples: {len(target_data)}")

    # Run comparison
    results = run_comparison(
        training_data, rct_data,
        epsilon=args.epsilon,
        verbose=verbose
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f'{args.study}_beta{args.beta}_comparison.csv'
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    heuristic = results[results['method'] == 'Heuristic (sensitivity)'].iloc[0]
    lp_closed = results[results['method'] == 'LP closed-form'].iloc[0]

    width_reduction = (heuristic['mean_width'] - lp_closed['mean_width']) / heuristic['mean_width'] * 100

    print(f"Heuristic width:   {heuristic['mean_width']:.4f}")
    print(f"LP width:          {lp_closed['mean_width']:.4f}")
    print(f"Width reduction:   {width_reduction:.1f}%")

    if not np.isnan(heuristic['coverage']) and not np.isnan(lp_closed['coverage']):
        print(f"\nHeuristic coverage: {heuristic['coverage']:.1%}")
        print(f"LP coverage:        {lp_closed['coverage']:.1%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
