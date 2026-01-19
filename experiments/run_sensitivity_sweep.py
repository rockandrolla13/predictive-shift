#!/usr/bin/env python3
"""
Run Sensitivity Sweep for Epsilon Parameter

This script runs a parameter sweep to characterize the precision-coverage
tradeoff for CATE bound estimation.

Usage:
    # Using YAML config
    python experiments/run_sensitivity_sweep.py --config experiments/configs/sensitivity_sweep.yaml

    # Using command-line arguments
    python experiments/run_sensitivity_sweep.py \
        --study anchoring1 \
        --beta 0.25 \
        --epsilon-values 0.01 0.05 0.1 0.15 0.2 0.3 0.5 \
        --output results/sensitivity/

    # Auto-generate epsilon values
    python experiments/run_sensitivity_sweep.py \
        --study anchoring1 \
        --beta 0.25 \
        --epsilon-range 0.01 0.5 \
        --n-points 20 \
        --output results/sensitivity/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List

import yaml
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causal_grounding import (
    SweepConfig,
    SensitivityAnalyzer,
    create_train_target_split,
    load_rct_data,
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_ground_truth_cate(
    rct_data: pd.DataFrame,
    treatment: str = 'iv',
    outcome: str = 'dv',
    covariates: List[str] = None
) -> Dict:
    """
    Compute ground truth CATE from RCT data.

    Args:
        rct_data: RCT DataFrame
        treatment: Treatment column
        outcome: Outcome column
        covariates: Covariate columns for stratification

    Returns:
        Dict mapping z_value tuples to true CATE
    """
    from causal_grounding import discretize_covariates, get_discretized_covariate_names

    # Discretize
    rct_disc = discretize_covariates(rct_data)

    if covariates is None:
        covariates = get_discretized_covariate_names()

    # Compute CATE for each stratum
    true_cates = {}

    for z_value, group in rct_disc.groupby(covariates, observed=True):
        z_tuple = tuple(z_value) if hasattr(z_value, '__iter__') else (z_value,)

        treated = group[group[treatment] == 1][outcome]
        control = group[group[treatment] == 0][outcome]

        if len(treated) >= 5 and len(control) >= 5:
            true_cates[z_tuple] = treated.mean() - control.mean()

    return true_cates


def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity sweep for epsilon parameter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file option
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )

    # Direct argument options
    parser.add_argument(
        '--study', '-s',
        type=str,
        default='anchoring1',
        help='ManyLabs study name (default: anchoring1)'
    )
    parser.add_argument(
        '--beta', '-b',
        type=float,
        default=0.25,
        help='Confounding strength beta (default: 0.25)'
    )
    parser.add_argument(
        '--target-site',
        type=str,
        default='mturk',
        help='Site to use as target (default: mturk)'
    )

    # Epsilon sweep options
    parser.add_argument(
        '--epsilon-values',
        type=float,
        nargs='+',
        help='Explicit epsilon values to test'
    )
    parser.add_argument(
        '--epsilon-range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='Range for auto-generating epsilon values'
    )
    parser.add_argument(
        '--n-points',
        type=int,
        default=10,
        help='Number of epsilon points to generate (default: 10)'
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use log-scale for epsilon values'
    )

    # Estimator options
    parser.add_argument(
        '--transfer-method',
        type=str,
        choices=['conservative', 'average'],
        default='conservative',
        help='Transfer method (default: conservative)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=500,
        help='Number of permutations for CI tests (default: 500)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/sensitivity/',
        help='Output directory (default: results/sensitivity/)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)

        # Extract values from config
        study = config.get('experiment', {}).get('study', args.study)
        beta = config.get('experiment', {}).get('beta', args.beta)
        target_site = config.get('experiment', {}).get('target_site', args.target_site)
        seed = config.get('experiment', {}).get('random_seed', args.seed)

        sweep_config = config.get('sweep', {})
        epsilon_values = sweep_config.get('values')
        epsilon_range = sweep_config.get('range')
        n_points = sweep_config.get('n_points', args.n_points)
        log_scale = sweep_config.get('log_scale', False)

        base_params = config.get('base_estimator', {})
        output_dir = config.get('output', {}).get('output_dir', args.output)
        generate_plots = config.get('output', {}).get('plot_pareto', not args.no_plots)
    else:
        study = args.study
        beta = args.beta
        target_site = args.target_site
        seed = args.seed
        epsilon_values = args.epsilon_values
        epsilon_range = args.epsilon_range
        n_points = args.n_points
        log_scale = args.log_scale
        base_params = {
            'transfer_method': args.transfer_method,
            'n_permutations': args.n_permutations,
        }
        output_dir = args.output
        generate_plots = not args.no_plots

    verbose = not args.quiet

    # Validate epsilon configuration
    if epsilon_values is None and epsilon_range is None:
        # Default epsilon values
        epsilon_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    # Create sweep config
    if epsilon_values:
        sweep_cfg = SweepConfig(
            parameter_name='epsilon',
            values=epsilon_values
        )
    else:
        sweep_cfg = SweepConfig(
            parameter_name='epsilon',
            n_points=n_points,
            range=tuple(epsilon_range),
            log_scale=log_scale
        )

    if verbose:
        print("=" * 60)
        print("Sensitivity Sweep for Epsilon Parameter")
        print("=" * 60)
        print(f"Study: {study}")
        print(f"Beta: {beta}")
        print(f"Target site: {target_site}")
        print(f"Epsilon values: {sweep_cfg.values}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if verbose:
        print("\nLoading data...")

    osrct_path = project_root / 'confounded_datasets' / study / f'{study}_age_beta_{beta}.csv'
    if not osrct_path.exists():
        # Try alternate naming
        osrct_path = project_root / 'confounded_datasets' / study / f'age_beta{beta}_seed42.csv'

    rct_path = project_root / 'ManyLabs1' / 'pre-process' / 'Manylabs1_data.pkl'

    if not osrct_path.exists():
        print(f"Error: OSRCT data not found at {osrct_path}")
        print("Available files:")
        study_dir = project_root / 'confounded_datasets' / study
        if study_dir.exists():
            for f in list(study_dir.iterdir())[:5]:
                print(f"  {f.name}")
        sys.exit(1)

    if not rct_path.exists():
        print(f"Error: RCT data not found at {rct_path}")
        sys.exit(1)

    osrct_data = pd.read_csv(osrct_path)
    rct_data = load_rct_data(study, str(rct_path))

    # Create train/target split
    if verbose:
        print("Creating train/target split...")

    training_data, target_data = create_train_target_split(
        osrct_data, rct_data, target_site=target_site
    )

    if verbose:
        print(f"  Training sites: {len(training_data)}")
        print(f"  Target size: {len(target_data)}")

    # Compute ground truth CATE from RCT
    if verbose:
        print("Computing ground truth CATE from RCT data...")

    ground_truth = compute_ground_truth_cate(rct_data)
    if verbose:
        print(f"  Ground truth strata: {len(ground_truth)}")

    # Create and run analyzer
    if verbose:
        print("\nRunning sensitivity sweep...")

    analyzer = SensitivityAnalyzer(
        sweep_config=sweep_cfg,
        base_estimator_params=base_params,
        random_seed=seed,
        verbose=verbose
    )

    results = analyzer.run_sweep(
        training_data=training_data,
        treatment='iv',
        outcome='dv',
        ground_truth=ground_truth
    )

    # Save results
    prefix = f'{study}_beta{beta}'

    # CSV
    csv_path = output_path / f'{prefix}_sweep.csv'
    results.to_csv(csv_path)
    if verbose:
        print(f"\nResults saved to: {csv_path}")

    # JSON summary
    json_path = output_path / f'{prefix}_summary.json'
    results.to_json(json_path)
    if verbose:
        print(f"Summary saved to: {json_path}")

    # Plots
    if generate_plots:
        if verbose:
            print("Generating plots...")
        analyzer.plot_results(output_dir=output_path, prefix=prefix)
        if verbose:
            print(f"Plots saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nResults DataFrame:")
    print(results.results_df[['epsilon', 'coverage_rate', 'mean_width']].to_string(index=False))

    recommended = results.get_recommended_epsilon(target_coverage=0.5)
    print(f"\nRecommended epsilon (for 50% coverage target):")
    print(f"  epsilon = {recommended['epsilon']}")
    print(f"  coverage = {recommended['coverage_rate']:.1%}")
    print(f"  mean_width = {recommended['mean_width']:.1f}")
    print(f"  reason: {recommended['reason']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
