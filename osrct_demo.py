"""
OSRCT Demonstration Script

This script demonstrates how to use the OSRCT module to generate confounded
observational datasets from ManyLabs1 and Pipeline RCT data.

Usage:
    python osrct_demo.py --dataset manylabs1 --data-path path/to/data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings

from osrct import (
    OSRCTSampler,
    select_biasing_covariates,
    evaluate_osrct_sample,
    load_manylabs1_data,
    load_pipeline_data
)


def demo_manylabs1(data_path: str, study: str = 'anchoring1'):
    """
    Demonstrate OSRCT on ManyLabs1 data.

    Parameters
    ----------
    data_path : str
        Path to Manylabs1_data.RData file
    study : str
        Which study to use (e.g., 'anchoring1', 'flag', 'gainloss')
    """
    print("\n" + "=" * 80)
    print("OSRCT Demo: ManyLabs1 Dataset")
    print("=" * 80)

    # Load data
    print(f"\n1. Loading ManyLabs1 data (study: {study})...")
    try:
        data = load_manylabs1_data(data_path, study_filter=study)
        print(f"   Loaded {len(data):,} observations")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Please ensure you have:")
        print("   - Processed the data using ML1_data_process.R and ML1_data_process_2.R")
        print("   - Installed pyreadr: pip install pyreadr")
        return

    # Display data structure
    print("\n2. Data structure:")
    print(f"   Columns: {list(data.columns[:10])}... ({len(data.columns)} total)")
    print(f"   Treatment (iv) distribution: {data['iv'].value_counts().to_dict()}")
    print(f"   Outcome (dv) summary: mean={data['dv'].mean():.3f}, std={data['dv'].std():.3f}")

    # Define candidate covariates (ManyLabs1 specific)
    candidate_covariates = [
        'resp_gender', 'resp_age', 'resp_ethnicity_hisp',
        'resp_polideo', 'resp_american', 'resp_american_pid', 'resp_american_ideo',
        'RACE_white', 'RACE_black_american', 'RACE_east_asian'
    ]

    # Filter to existing columns
    candidate_covariates = [
        c for c in candidate_covariates
        if c in data.columns
    ]

    print(f"\n3. Selecting biasing covariates from {len(candidate_covariates)} candidates...")
    biasing_covariates = select_biasing_covariates(
        data,
        treatment_col='iv',
        outcome_col='dv',
        candidate_covariates=candidate_covariates,
        min_correlation=0.05,
        max_covariates=5
    )

    print(f"   Selected {len(biasing_covariates)} covariates:")
    for cov in biasing_covariates:
        corr = np.corrcoef(
            data['dv'].values,
            data[cov].fillna(data[cov].mean()).values
        )[0, 1]
        print(f"     - {cov}: correlation with outcome = {corr:.3f}")

    if len(biasing_covariates) == 0:
        print("   No covariates met selection criteria. Using default set.")
        biasing_covariates = ['resp_age', 'resp_gender']

    # Create OSRCT sampler with moderate confounding
    print("\n4. Creating OSRCT sampler...")
    print("   Confounding strength: MODERATE")
    print("   Strategy: Age and gender have stronger effect on treatment selection")

    # Define biasing coefficients for moderate confounding
    biasing_coefficients = {cov: 0.5 for cov in biasing_covariates}
    if 'resp_age' in biasing_coefficients:
        biasing_coefficients['resp_age'] = 0.8  # Stronger effect
    if 'resp_gender' in biasing_coefficients:
        biasing_coefficients['resp_gender'] = 0.6

    sampler = OSRCTSampler(
        biasing_covariates=biasing_covariates,
        biasing_coefficients=biasing_coefficients,
        intercept=0.0,
        standardize=True,
        random_seed=42
    )

    # Generate observational sample
    print("\n5. Generating confounded observational sample...")
    obs_data, selection_probs = sampler.sample(
        data,
        treatment_col='iv',
        verbose=True
    )

    # Evaluate sample
    print("\n6. Evaluating OSRCT sample...")
    metrics = evaluate_osrct_sample(
        rct_data=data,
        obs_data=obs_data,
        treatment_col='iv',
        outcome_col='dv',
        covariates=biasing_covariates
    )

    print(f"\n   Treatment Effect Estimates:")
    print(f"     RCT (true) ATE: {metrics['rct_ate']:.4f} (SE: {metrics['rct_ate_se']:.4f})")
    print(f"     Observational (naive) ATE: {metrics['obs_ate_naive']:.4f} (SE: {metrics['obs_ate_se']:.4f})")
    print(f"     Confounding Bias: {metrics['confounding_bias']:.4f}")

    print(f"\n   Sample Characteristics:")
    print(f"     Original RCT size: {metrics['sample_size_rct']:,}")
    print(f"     Observational size: {metrics['sample_size_obs']:,}")
    print(f"     Retention rate: {metrics['sample_retention_rate']:.1%}")
    print(f"     RCT treatment rate: {metrics['rct_treatment_rate']:.1%}")
    print(f"     Obs treatment rate: {metrics['obs_treatment_rate']:.1%}")

    print(f"\n   Covariate Balance (Standardized Mean Differences):")
    if 'covariate_balance' in metrics:
        for cov, balance in metrics['covariate_balance'].items():
            print(f"     {cov}:")
            print(f"       RCT SMD: {balance['rct_smd']:>7.3f}")
            print(f"       Obs SMD: {balance['obs_smd']:>7.3f}")
            print(f"       Change:  {balance['smd_change']:>7.3f}")

    # Create visualization
    print("\n7. Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Selection probability distribution
    axes[0, 0].hist(selection_probs, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(selection_probs.mean(), color='red', linestyle='--',
                      label=f'Mean: {selection_probs.mean():.3f}')
    axes[0, 0].set_xlabel('Selection Probability P(T=1|C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Selection Probabilities')
    axes[0, 0].legend()

    # Plot 2: Treatment distribution comparison
    treatment_comparison = pd.DataFrame({
        'RCT': data['iv'].value_counts(normalize=True),
        'Observational': obs_data['iv'].value_counts(normalize=True)
    })
    treatment_comparison.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Treatment')
    axes[0, 1].set_ylabel('Proportion')
    axes[0, 1].set_title('Treatment Distribution: RCT vs Observational')
    axes[0, 1].set_xticklabels(['Control', 'Treatment'], rotation=0)
    axes[0, 1].legend()

    # Plot 3: Outcome distribution by treatment
    rct_plot_data = data[['iv', 'dv']].copy()
    rct_plot_data['source'] = 'RCT'
    obs_plot_data = obs_data[['iv', 'dv']].copy()
    obs_plot_data['source'] = 'Observational'
    combined = pd.concat([rct_plot_data, obs_plot_data])

    for source, color in [('RCT', 'blue'), ('Observational', 'red')]:
        for treatment in [0, 1]:
            subset = combined[(combined['source'] == source) & (combined['iv'] == treatment)]
            axes[1, 0].hist(subset['dv'], bins=20, alpha=0.3, color=color,
                          label=f'{source}, T={treatment}')
    axes[1, 0].set_xlabel('Outcome (dv)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Outcome Distribution by Treatment and Source')
    axes[1, 0].legend()

    # Plot 4: Covariate balance (SMD)
    if 'covariate_balance' in metrics:
        covs = list(metrics['covariate_balance'].keys())[:5]  # Top 5
        rct_smds = [metrics['covariate_balance'][c]['rct_smd'] for c in covs]
        obs_smds = [metrics['covariate_balance'][c]['obs_smd'] for c in covs]

        x = np.arange(len(covs))
        width = 0.35

        axes[1, 1].bar(x - width/2, rct_smds, width, label='RCT', alpha=0.7)
        axes[1, 1].bar(x + width/2, obs_smds, width, label='Observational', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].axhline(y=0.1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1, 1].axhline(y=-0.1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1, 1].set_xlabel('Covariate')
        axes[1, 1].set_ylabel('Standardized Mean Difference')
        axes[1, 1].set_title('Covariate Balance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([c[:15] for c in covs], rotation=45, ha='right')
        axes[1, 1].legend()

    plt.tight_layout()
    output_path = Path('osrct_manylabs1_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")

    # Save observational sample
    output_csv = Path(f'osrct_sample_{study}.csv')
    obs_data.to_csv(output_csv, index=False)
    print(f"   Saved observational sample to: {output_csv}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_pipeline(data_dir: str, study_id: int = 7):
    """
    Demonstrate OSRCT on Pipeline data.

    Parameters
    ----------
    data_dir : str
        Directory containing processed Pipeline CSV files
    study_id : int
        Which study to use (5, 7, or 8)
    """
    print("\n" + "=" * 80)
    print("OSRCT Demo: Pipeline Dataset")
    print("=" * 80)

    # Load data
    print(f"\n1. Loading Pipeline data (study ID: {study_id})...")
    try:
        data = load_pipeline_data(data_dir, study_id=study_id)
        print(f"   Loaded {len(data):,} observations")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Please ensure you have processed the data using process.Rmd")
        return

    # Study-specific configuration
    if study_id == 7:  # Intuitive Economics
        treatment_col = 'condition'
        outcome_col = 'htxfair'  # or 'htxgood'
        candidate_covariates = [
            'poltclid', 'gender', 'yearbirth', 'familyinc'
        ]
    else:
        print(f"   Study ID {study_id} not fully configured. Using defaults.")
        treatment_col = 'condition' if 'condition' in data.columns else data.columns[1]
        outcome_col = data.select_dtypes(include=[np.number]).columns[2]
        candidate_covariates = list(data.select_dtypes(include=[np.number]).columns[3:10])

    # Ensure treatment is binary
    if treatment_col in data.columns:
        data[treatment_col] = data[treatment_col].fillna(0).astype(int)
        # Binarize if needed
        unique_vals = data[treatment_col].unique()
        if len(unique_vals) > 2:
            print(f"   Warning: Treatment has {len(unique_vals)} values. Binarizing...")
            data[treatment_col] = (data[treatment_col] > data[treatment_col].median()).astype(int)

    print(f"\n2. Data structure:")
    print(f"   Treatment column: {treatment_col}")
    print(f"   Outcome column: {outcome_col}")
    print(f"   Treatment distribution: {data[treatment_col].value_counts().to_dict()}")

    # Filter covariates
    candidate_covariates = [c for c in candidate_covariates if c in data.columns]

    print(f"\n3. Selecting biasing covariates from {len(candidate_covariates)} candidates...")
    biasing_covariates = select_biasing_covariates(
        data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        candidate_covariates=candidate_covariates,
        min_correlation=0.05,
        max_covariates=5
    )

    if len(biasing_covariates) == 0:
        print("   No covariates met criteria. Using first few candidates.")
        biasing_covariates = candidate_covariates[:min(3, len(candidate_covariates))]

    print(f"   Selected covariates: {biasing_covariates}")

    # Create OSRCT sampler
    sampler = OSRCTSampler(
        biasing_covariates=biasing_covariates,
        biasing_coefficients={cov: 0.6 for cov in biasing_covariates},
        intercept=0.0,
        standardize=True,
        random_seed=42
    )

    # Generate sample
    print("\n4. Generating confounded observational sample...")
    obs_data, selection_probs = sampler.sample(
        data,
        treatment_col=treatment_col,
        verbose=True
    )

    # Evaluate
    print("\n5. Evaluating OSRCT sample...")
    metrics = evaluate_osrct_sample(
        rct_data=data,
        obs_data=obs_data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariates=biasing_covariates
    )

    print(f"\n   Treatment Effect Estimates:")
    print(f"     RCT (true) ATE: {metrics['rct_ate']:.4f}")
    print(f"     Observational (naive) ATE: {metrics['obs_ate_naive']:.4f}")
    print(f"     Confounding Bias: {metrics['confounding_bias']:.4f}")

    # Save results
    output_csv = Path(f'osrct_sample_pipeline_{study_id}.csv')
    obs_data.to_csv(output_csv, index=False)
    print(f"\n   Saved observational sample to: {output_csv}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_synthetic():
    """
    Demonstrate OSRCT on synthetic data for testing.
    """
    print("\n" + "=" * 80)
    print("OSRCT Demo: Synthetic Dataset")
    print("=" * 80)

    # Generate synthetic RCT data
    print("\n1. Generating synthetic RCT data...")
    np.random.seed(42)
    n = 2000

    # Covariates
    age = np.random.normal(35, 10, n)
    gender = np.random.binomial(1, 0.5, n)
    income = np.random.gamma(2, 20, n)

    # Random treatment assignment (RCT)
    treatment = np.random.binomial(1, 0.5, n)

    # Outcome with true treatment effect = 2.0
    # Outcome also depends on covariates
    true_ate = 2.0
    outcome = (
        10 +                      # baseline
        true_ate * treatment +    # treatment effect
        0.1 * age +              # age effect
        1.5 * gender +           # gender effect
        0.05 * income +          # income effect
        np.random.normal(0, 2, n)  # noise
    )

    # Create dataframe
    rct_data = pd.DataFrame({
        'iv': treatment,
        'dv': outcome,
        'age': age,
        'gender': gender,
        'income': income
    })

    print(f"   Generated {len(rct_data):,} observations")
    print(f"   True ATE: {true_ate}")

    # Select biasing covariates
    print("\n2. Selecting biasing covariates...")
    biasing_covariates = select_biasing_covariates(
        rct_data,
        treatment_col='iv',
        outcome_col='dv',
        candidate_covariates=['age', 'gender', 'income'],
        min_correlation=0.0,
        max_covariates=3
    )
    print(f"   Selected: {biasing_covariates}")

    # Create OSRCT sampler with strong confounding
    print("\n3. Creating OSRCT sampler (STRONG confounding)...")
    sampler = OSRCTSampler(
        biasing_covariates=biasing_covariates,
        biasing_coefficients={'age': 1.2, 'gender': 1.5, 'income': 0.8},
        intercept=-0.5,
        standardize=True,
        random_seed=42
    )

    # Generate observational sample
    print("\n4. Generating confounded observational sample...")
    obs_data, selection_probs = sampler.sample(rct_data, treatment_col='iv', verbose=True)

    # Evaluate
    print("\n5. Evaluating OSRCT sample...")
    metrics = evaluate_osrct_sample(
        rct_data=rct_data,
        obs_data=obs_data,
        treatment_col='iv',
        outcome_col='dv',
        covariates=biasing_covariates
    )

    print(f"\n   Treatment Effect Estimates:")
    print(f"     TRUE ATE: {true_ate:.4f}")
    print(f"     RCT ATE: {metrics['rct_ate']:.4f} (SE: {metrics['rct_ate_se']:.4f})")
    print(f"     Observational (naive) ATE: {metrics['obs_ate_naive']:.4f} (SE: {metrics['obs_ate_se']:.4f})")
    print(f"     Confounding Bias: {metrics['confounding_bias']:.4f}")

    print("\n   Note: The observational estimate is biased due to confounding!")
    print("   Methods like IPW or regression adjustment would be needed to recover true ATE.")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='OSRCT Demonstration Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['manylabs1', 'pipeline', 'synthetic'],
        default='synthetic',
        help='Which dataset to use for demonstration'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data file or directory'
    )

    parser.add_argument(
        '--study',
        type=str,
        default='anchoring1',
        help='Study name for ManyLabs1 (e.g., anchoring1, flag, gainloss)'
    )

    parser.add_argument(
        '--study-id',
        type=int,
        default=7,
        help='Study ID for Pipeline (5, 7, or 8)'
    )

    args = parser.parse_args()

    # Run appropriate demo
    if args.dataset == 'manylabs1':
        if args.data_path is None:
            print("Error: --data-path required for ManyLabs1 dataset")
            print("Please provide path to Manylabs1_data.RData")
            return
        demo_manylabs1(args.data_path, args.study)

    elif args.dataset == 'pipeline':
        if args.data_path is None:
            print("Error: --data-path required for Pipeline dataset")
            print("Please provide path to Pipeline data directory")
            return
        demo_pipeline(args.data_path, args.study_id)

    else:  # synthetic
        demo_synthetic()


if __name__ == "__main__":
    main()
