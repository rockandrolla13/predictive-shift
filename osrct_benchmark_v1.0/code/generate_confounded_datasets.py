#!/usr/bin/env python3
"""
Generate Confounded Datasets from ManyLabs1 RCT Data

This script systematically applies the OSRCT algorithm to generate a comprehensive
benchmark dataset of confounded observational samples from ManyLabs1 RCT data.

Features:
- Generates datasets for all 15 ManyLabs1 studies
- Multiple confounding strengths (beta = 0.1 to 2.0)
- Single and multi-covariate confounding patterns
- Site-stratified datasets for heterogeneity analysis
- Comprehensive metadata and validation

Usage:
    # Generate all configurations
    python generate_confounded_datasets.py

    # Generate specific studies
    python generate_confounded_datasets.py --studies anchoring1 gainloss

    # Generate with specific beta values
    python generate_confounded_datasets.py --beta-values 0.5 1.0

    # Generate site-stratified datasets only
    python generate_confounded_datasets.py --site-stratified-only

Reference:
Gentzel, M., Garant, D., & Jensen, D. (2021). The Case for Evaluating Causal
Models Using Controlled Experiments. NeurIPS 2021.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Import OSRCT components
from osrct import OSRCTSampler, evaluate_osrct_sample, load_manylabs1_data
from experimental_grid import (
    EXPERIMENTAL_GRID,
    STUDIES,
    BETA_VALUES,
    COVARIATE_PATTERNS,
    get_total_configurations,
    get_grid_summary
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_PATH = 'ManyLabs1/pre-process/Manylabs1_data.pkl'
DEFAULT_OUTPUT_DIR = 'confounded_datasets'
VERSION = '1.0.0'


# =============================================================================
# CORE GENERATION FUNCTIONS
# =============================================================================

def generate_single_configuration(
    data: pd.DataFrame,
    study: str,
    beta: float,
    pattern_name: str,
    pattern_config: dict,
    seed: int,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate one confounded dataset for a specific configuration.

    Parameters
    ----------
    data : DataFrame
        Preprocessed ManyLabs1 data (filtered to study)
    study : str
        Study identifier
    beta : float
        Confounding strength
    pattern_name : str
        Covariate pattern identifier
    pattern_config : dict
        Covariate names and coefficient function
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print detailed output

    Returns
    -------
    obs_data : DataFrame
        Confounded observational sample
    metrics : dict
        Evaluation metrics including ground-truth ATE
    """
    # Extract configuration
    covariates = pattern_config['covariates']
    coefficients = pattern_config['coefficients'](beta)

    # Validate covariates exist
    missing = set(covariates) - set(data.columns)
    if missing:
        raise ValueError(f"Missing covariates: {missing}")

    # Create sampler
    sampler = OSRCTSampler(
        biasing_covariates=covariates,
        biasing_coefficients=coefficients,
        intercept=0.0,
        standardize=True,
        random_seed=seed
    )

    # Generate sample
    obs_data, selection_probs = sampler.sample(
        data,
        treatment_col='iv',
        verbose=verbose
    )

    # Compute metrics
    metrics = evaluate_osrct_sample(
        rct_data=data,
        obs_data=obs_data,
        treatment_col='iv',
        outcome_col='dv',
        covariates=covariates
    )

    # Add configuration metadata
    metrics['study'] = study
    metrics['beta'] = beta
    metrics['pattern'] = pattern_name
    metrics['seed'] = seed
    metrics['covariates'] = covariates
    metrics['coefficients'] = coefficients
    metrics['selection_prob_min'] = selection_probs.min()
    metrics['selection_prob_max'] = selection_probs.max()
    metrics['selection_prob_mean'] = selection_probs.mean()
    metrics['selection_prob_std'] = selection_probs.std()

    return obs_data, metrics


def generate_study_batch(
    data: pd.DataFrame,
    study: str,
    patterns: Dict[str, dict],
    beta_values: List[float],
    output_dir: Path,
    seeds: List[int] = [42],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate all configurations for a single study.

    Parameters
    ----------
    data : DataFrame
        Full preprocessed data
    study : str
        Study to process
    patterns : dict
        Covariate pattern specifications
    beta_values : list
        Beta values to use
    output_dir : Path
        Output directory
    seeds : List[int]
        Random seeds for reproducibility
    verbose : bool
        Print detailed output

    Returns
    -------
    all_metrics : List[dict]
        Metrics for all configurations
    """
    # Filter to study
    study_data = data[data['original_study'] == study].copy()

    if len(study_data) == 0:
        raise ValueError(f"No data for study: {study}")

    all_metrics = []
    study_dir = output_dir / study
    study_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all configurations
    for pattern_name, pattern_config in patterns.items():
        for beta in beta_values:
            for seed in seeds:
                # Generate configuration ID
                config_id = f"{pattern_name}_beta{beta}_seed{seed}"

                try:
                    obs_data, metrics = generate_single_configuration(
                        data=study_data,
                        study=study,
                        beta=beta,
                        pattern_name=pattern_name,
                        pattern_config=pattern_config,
                        seed=seed,
                        verbose=verbose
                    )

                    # Save dataset
                    output_path = study_dir / f"{config_id}.csv"
                    obs_data.to_csv(output_path, index=False)

                    metrics['output_file'] = str(output_path)
                    metrics['success'] = True
                    metrics['error'] = None

                except Exception as e:
                    metrics = {
                        'study': study,
                        'beta': beta,
                        'pattern': pattern_name,
                        'seed': seed,
                        'success': False,
                        'error': str(e),
                        'output_file': None
                    }

                all_metrics.append(metrics)

    return all_metrics


def generate_site_stratified(
    data: pd.DataFrame,
    study: str,
    beta: float,
    pattern_config: dict,
    output_dir: Path,
    min_site_n: int = 50,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate confounded datasets for each site separately.

    Parameters
    ----------
    data : DataFrame
        Full preprocessed data
    study : str
        Study to process
    beta : float
        Confounding strength
    pattern_config : dict
        Covariate pattern specification
    output_dir : Path
        Output directory
    min_site_n : int
        Minimum site sample size to include
    seed : int
        Random seed

    Returns
    -------
    site_metrics : List[dict]
        Metrics for each site
    """
    study_data = data[data['original_study'] == study]
    sites = study_data['site'].unique()

    site_metrics = []

    for site in sites:
        site_data = study_data[study_data['site'] == site].copy()

        if len(site_data) < min_site_n:
            site_metrics.append({
                'study': study,
                'site': site,
                'beta': beta,
                'success': False,
                'error': f'Insufficient sample size: {len(site_data)} < {min_site_n}',
                'site_n_rct': len(site_data)
            })
            continue

        try:
            obs_data, metrics = generate_single_configuration(
                data=site_data,
                study=study,
                beta=beta,
                pattern_name='demo_basic',
                pattern_config=pattern_config,
                seed=seed,
                verbose=False
            )

            metrics['site'] = site
            metrics['site_n_rct'] = len(site_data)

            # Save
            output_path = output_dir / f"{study}_{site}_beta{beta}.csv"
            obs_data.to_csv(output_path, index=False)

            metrics['output_file'] = str(output_path)
            metrics['success'] = True
            metrics['error'] = None

        except Exception as e:
            metrics = {
                'study': study,
                'site': site,
                'beta': beta,
                'success': False,
                'error': str(e),
                'site_n_rct': len(site_data)
            }

        site_metrics.append(metrics)

    return site_metrics


def compute_ground_truth_ates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute true ATEs from RCT data for all studies.

    Parameters
    ----------
    data : DataFrame
        Full preprocessed ManyLabs1 data

    Returns
    -------
    ates_df : DataFrame
        Ground-truth ATEs for each study
    """
    results = []

    for study in data['original_study'].unique():
        study_data = data[data['original_study'] == study]

        treated = study_data[study_data['iv'] == 1]['dv']
        control = study_data[study_data['iv'] == 0]['dv']

        ate = treated.mean() - control.mean()
        se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))

        results.append({
            'study': study,
            'n_total': len(study_data),
            'n_treated': len(treated),
            'n_control': len(control),
            'mean_y1': treated.mean(),
            'mean_y0': control.mean(),
            'std_y1': treated.std(),
            'std_y0': control.std(),
            'ate': ate,
            'ate_se': se,
            'ate_ci_lower': ate - 1.96 * se,
            'ate_ci_upper': ate + 1.96 * se
        })

    return pd.DataFrame(results)


def generate_full_grid(
    data_path: str,
    output_dir: str,
    studies: List[str] = None,
    beta_values: List[float] = None,
    patterns: List[str] = None,
    seeds: List[int] = [42],
    include_site_stratified: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate complete experimental grid.

    Parameters
    ----------
    data_path : str
        Path to Manylabs1_data.pkl
    output_dir : str
        Root output directory
    studies : List[str], optional
        Subset of studies (None = all)
    beta_values : List[float], optional
        Beta values to use (None = all)
    patterns : List[str], optional
        Patterns to use (None = all)
    seeds : List[int]
        Random seeds
    include_site_stratified : bool
        Whether to generate site-stratified datasets
    verbose : bool
        Print progress

    Returns
    -------
    results_df : DataFrame
        Summary of all configurations with metrics
    """
    start_time = time.time()

    # Load data
    if verbose:
        print(f"Loading data from: {data_path}")
    data = pd.read_pickle(data_path)
    if verbose:
        print(f"  Loaded {len(data):,} observations")

    # Set defaults
    if studies is None:
        studies = STUDIES
    if beta_values is None:
        beta_values = BETA_VALUES
    if patterns is None:
        pattern_configs = COVARIATE_PATTERNS
    else:
        pattern_configs = {k: v for k, v in COVARIATE_PATTERNS.items() if k in patterns}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total configurations
    n_configs = len(studies) * len(beta_values) * len(pattern_configs) * len(seeds)
    if verbose:
        print(f"\nGenerating {n_configs} pooled configurations...")
        print(f"  Studies: {len(studies)}")
        print(f"  Beta values: {beta_values}")
        print(f"  Patterns: {list(pattern_configs.keys())}")
        print(f"  Seeds: {seeds}")

    # Compute and save ground-truth ATEs
    if verbose:
        print("\nComputing ground-truth ATEs...")
    ates_df = compute_ground_truth_ates(data)
    ates_df.to_csv(output_dir / 'ground_truth_ates.csv', index=False)
    if verbose:
        print(f"  Saved to: {output_dir / 'ground_truth_ates.csv'}")

    # Generate pooled datasets
    all_results = []
    completed = 0

    for i, study in enumerate(studies):
        if verbose:
            print(f"\n[{i+1}/{len(studies)}] Processing {study}...")

        study_metrics = generate_study_batch(
            data=data,
            study=study,
            patterns=pattern_configs,
            beta_values=beta_values,
            output_dir=output_dir,
            seeds=seeds,
            verbose=False
        )

        all_results.extend(study_metrics)
        completed += len(study_metrics)

        if verbose:
            success = sum(1 for m in study_metrics if m.get('success', False))
            print(f"    Generated {success}/{len(study_metrics)} datasets")

    # Generate site-stratified datasets
    site_stratified_results = []

    if include_site_stratified:
        if verbose:
            print("\n\nGenerating site-stratified datasets...")

        site_dir = output_dir / 'site_stratified'
        site_dir.mkdir(parents=True, exist_ok=True)

        grid_config = EXPERIMENTAL_GRID['site_stratified']
        site_studies = [s for s in grid_config['studies'] if s in studies]
        site_betas = grid_config['beta_values']
        site_pattern = COVARIATE_PATTERNS[grid_config['pattern']]
        min_site_n = grid_config['min_site_n']

        for study in site_studies:
            study_site_dir = site_dir / study
            study_site_dir.mkdir(parents=True, exist_ok=True)

            for beta in site_betas:
                if verbose:
                    print(f"  {study} (beta={beta})...")

                site_metrics = generate_site_stratified(
                    data=data,
                    study=study,
                    beta=beta,
                    pattern_config=site_pattern,
                    output_dir=study_site_dir,
                    min_site_n=min_site_n,
                    seed=42
                )

                site_stratified_results.extend(site_metrics)

                success = sum(1 for m in site_metrics if m.get('success', False))
                if verbose:
                    print(f"    {success}/{len(site_metrics)} sites")

    # Save site-stratified summary
    if site_stratified_results:
        site_df = pd.DataFrame(site_stratified_results)
        site_df.to_csv(output_dir / 'site_stratified_summary.csv', index=False)

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Save summary
    results_df.to_csv(output_dir / 'generation_summary.csv', index=False)

    # Generate metadata
    elapsed = time.time() - start_time
    metadata = {
        'version': VERSION,
        'timestamp': datetime.now().isoformat(),
        'source_data': data_path,
        'total_configurations': len(all_results),
        'successful_generations': results_df['success'].sum() if 'success' in results_df else 0,
        'failed_generations': (~results_df['success']).sum() if 'success' in results_df else 0,
        'studies': studies,
        'beta_values': beta_values,
        'patterns': list(pattern_configs.keys()),
        'seeds': seeds,
        'site_stratified_configurations': len(site_stratified_results),
        'elapsed_seconds': elapsed
    }

    # Save metadata
    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"  Pooled datasets: {metadata['successful_generations']}/{metadata['total_configurations']} successful")
        print(f"  Site-stratified: {len(site_stratified_results)} configurations")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Output directory: {output_dir}")

    return results_df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_generated_datasets(
    output_dir: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate all generated datasets meet quality criteria.

    Parameters
    ----------
    output_dir : str
        Directory containing generated datasets
    verbose : bool
        Print validation results

    Returns
    -------
    report : dict
        Validation report
    """
    output_dir = Path(output_dir)

    # Load generation summary
    summary_path = output_dir / 'generation_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f"Generation summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    report = {
        'total_datasets': len(summary_df),
        'successful': summary_df['success'].sum(),
        'failed': (~summary_df['success']).sum(),
        'checks': {}
    }

    # Check 1: All successful datasets have output files
    if 'output_file' in summary_df.columns:
        successful = summary_df[summary_df['success']]
        files_exist = successful['output_file'].apply(
            lambda x: Path(x).exists() if pd.notna(x) else False
        )
        report['checks']['files_exist'] = {
            'passed': files_exist.sum(),
            'failed': len(files_exist) - files_exist.sum(),
            'pass_rate': files_exist.mean()
        }

    # Check 2: Sample retention rates
    if 'sample_retention_rate' in summary_df.columns:
        valid_retention = (
            (summary_df['sample_retention_rate'] > 0.3) &
            (summary_df['sample_retention_rate'] < 0.7)
        )
        report['checks']['retention_rate'] = {
            'passed': valid_retention.sum(),
            'failed': (~valid_retention).sum(),
            'pass_rate': valid_retention.mean(),
            'mean': summary_df['sample_retention_rate'].mean(),
            'std': summary_df['sample_retention_rate'].std()
        }

    # Check 3: Confounding bias introduced
    if 'confounding_bias' in summary_df.columns:
        bias_stats = {
            'mean': summary_df['confounding_bias'].mean(),
            'std': summary_df['confounding_bias'].std(),
            'min': summary_df['confounding_bias'].min(),
            'max': summary_df['confounding_bias'].max()
        }
        report['checks']['confounding_bias'] = bias_stats

    # Save report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    if verbose:
        print("\nValidation Report")
        print("=" * 40)
        print(f"Total datasets: {report['total_datasets']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")

        for check_name, check_result in report['checks'].items():
            print(f"\n{check_name}:")
            for k, v in check_result.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate confounded datasets from ManyLabs1 RCT data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all configurations
  python generate_confounded_datasets.py

  # Generate specific studies
  python generate_confounded_datasets.py --studies anchoring1 gainloss sunk

  # Generate with specific beta values
  python generate_confounded_datasets.py --beta-values 0.5 1.0

  # Generate specific patterns
  python generate_confounded_datasets.py --patterns age gender demo_basic

  # Skip site-stratified generation
  python generate_confounded_datasets.py --no-site-stratified

  # Validate generated datasets
  python generate_confounded_datasets.py --validate-only
        """
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f'Path to preprocessed data (default: {DEFAULT_DATA_PATH})'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--studies', '-s',
        type=str,
        nargs='+',
        default=None,
        help='Specific studies to process (default: all 15)'
    )

    parser.add_argument(
        '--beta-values', '-b',
        type=float,
        nargs='+',
        default=None,
        help='Confounding strength values (default: 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)'
    )

    parser.add_argument(
        '--patterns', '-p',
        type=str,
        nargs='+',
        default=None,
        help='Covariate patterns to use (default: all 6)'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Random seeds for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--no-site-stratified',
        action='store_true',
        help='Skip site-stratified dataset generation'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing datasets (skip generation)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'%(prog)s {VERSION}'
    )

    args = parser.parse_args()

    # Validate only mode
    if args.validate_only:
        validate_generated_datasets(args.output_dir, verbose=not args.quiet)
        return

    # Check data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("Please run preprocessing first to generate Manylabs1_data.pkl")
        sys.exit(1)

    # Run generation
    results = generate_full_grid(
        data_path=args.data_path,
        output_dir=args.output_dir,
        studies=args.studies,
        beta_values=args.beta_values,
        patterns=args.patterns,
        seeds=args.seeds,
        include_site_stratified=not args.no_site_stratified,
        verbose=not args.quiet
    )

    # Run validation
    if not args.quiet:
        print("\n")
    validate_generated_datasets(args.output_dir, verbose=not args.quiet)


if __name__ == '__main__':
    main()
