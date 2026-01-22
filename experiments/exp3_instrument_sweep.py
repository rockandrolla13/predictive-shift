"""
Experiment 3: Instrument Strength Sweep

This experiment investigates how instrument strength (αz) affects bound width,
holding confounding and sample size constant.

Research Questions:
- What is the functional form of αz → Width?
- At what αz* do bounds become uninformative (Width ≥ 0.8)?
- How does αz* depend on naturalness tolerance ε?

Reference: BoundsExperiments/prd_exp3_exp4_v2.md
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding.confounded_instrument_dgp import (
    ConfoundedInstrumentDGP,
    generate_confounded_instrument_data,
    logistic
)
from causal_grounding.lp_solver import (
    solve_all_bounds_binary_lp,
    estimate_conditional_means
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Exp3Config:
    """Configuration for Experiment 3."""
    # Swept parameters
    alpha_z_values: List[float] = None
    epsilon_values: List[float] = None

    # Fixed parameters
    alpha_u: float = 0.5       # Moderate treatment confounding
    beta_u: float = 0.5        # Moderate outcome confounding
    beta_x: float = 1.0        # True causal effect
    n_per_env: int = 2000      # Sample size per environment
    K: int = 5                 # Number of training environments

    # Replication
    n_seeds: int = 50          # Seeds per configuration
    seed_start: int = 42

    # Analysis
    width_threshold: float = 0.8  # For αz* computation

    # Parallelization
    n_jobs: int = -1

    def __post_init__(self):
        if self.alpha_z_values is None:
            self.alpha_z_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        if self.epsilon_values is None:
            self.epsilon_values = [0.05, 0.10, 0.15, 0.20]


# =============================================================================
# Data Generation
# =============================================================================

def generate_multi_environment_data(
    dgp: ConfoundedInstrumentDGP,
    n_per_env: int,
    K: int,
    seed: int
) -> Dict[str, pd.DataFrame]:
    """Generate data for K training environments."""
    np.random.seed(seed)
    training_data = {}

    for k in range(K):
        env_seed = seed + k * 1000

        # Generate observational data (F=idle)
        idle_data = generate_confounded_instrument_data(
            dgp, n_per_env, seed=env_seed, include_unobserved=False
        )
        idle_data['F'] = 'idle'

        # Generate experimental data (F=on) - treatment randomized
        np.random.seed(env_seed + 500)
        n = n_per_env

        Z = np.random.binomial(1, dgp.prob_z, n)
        X = np.random.binomial(1, 0.5, n)  # Randomized
        U = np.random.binomial(1, dgp.prob_u, n)

        logit_y = dgp.beta_0 + dgp.beta_x * X + dgp.beta_u * U + dgp.beta_z * Z
        prob_y = logistic(logit_y)
        Y = np.random.binomial(1, prob_y, n)

        on_data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'F': 'on'})
        env_data = pd.concat([idle_data, on_data], ignore_index=True)
        training_data[f'env_{k}'] = env_data

    return training_data


def compute_true_ate(dgp: ConfoundedInstrumentDGP, n_mc: int = 50000) -> float:
    """Compute true ATE via Monte Carlo."""
    np.random.seed(42)
    U = np.random.binomial(1, dgp.prob_u, n_mc)
    Z = np.random.binomial(1, dgp.prob_z, n_mc)

    logit_y1 = dgp.beta_0 + dgp.beta_x * 1 + dgp.beta_u * U + dgp.beta_z * Z
    logit_y0 = dgp.beta_0 + dgp.beta_x * 0 + dgp.beta_u * U + dgp.beta_z * Z

    return float(np.mean(logistic(logit_y1) - logistic(logit_y0)))


# =============================================================================
# Single Run
# =============================================================================

def run_single_configuration(
    alpha_z: float,
    epsilon: float,
    seed: int,
    config: Exp3Config
) -> Dict:
    """Run a single configuration and return results."""

    # Create DGP
    dgp = ConfoundedInstrumentDGP(
        alpha_0=0.0,
        alpha_z=alpha_z,
        alpha_u=config.alpha_u,
        alpha_w=0.0,
        beta_0=0.0,
        beta_x=config.beta_x,
        beta_u=config.beta_u,
        beta_w=0.0,
        beta_z=0.0,
        prob_u=0.5,
        prob_z=0.5,
        prob_w=0.5
    )

    # Generate data
    training_data = generate_multi_environment_data(
        dgp, config.n_per_env, config.K, seed
    )

    # Compute true ATE
    true_ate = compute_true_ate(dgp)

    # Solve bounds using LP
    try:
        bounds = solve_all_bounds_binary_lp(
            training_data=training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=epsilon,
            regime_col='F',
            use_cvxpy=False
        )

        if bounds and len(bounds) >= 2:
            # Compute ATE bounds (weighted average over Z)
            z0_bounds = bounds.get((0,), (float('nan'), float('nan')))
            z1_bounds = bounds.get((1,), (float('nan'), float('nan')))

            ate_lower = 0.5 * z0_bounds[0] + 0.5 * z1_bounds[0]
            ate_upper = 0.5 * z0_bounds[1] + 0.5 * z1_bounds[1]
            ate_width = ate_upper - ate_lower

            coverage = ate_lower <= true_ate <= ate_upper
            feasible = True
            lp_status = 'optimal'

            cate_width_z0 = z0_bounds[1] - z0_bounds[0]
            cate_width_z1 = z1_bounds[1] - z1_bounds[0]
        else:
            ate_lower = ate_upper = ate_width = float('nan')
            cate_width_z0 = cate_width_z1 = float('nan')
            coverage = False
            feasible = False
            lp_status = 'infeasible'

    except Exception as e:
        ate_lower = ate_upper = ate_width = float('nan')
        cate_width_z0 = cate_width_z1 = float('nan')
        coverage = False
        feasible = False
        lp_status = f'error: {str(e)}'

    # Compute empirical instrument strength
    all_idle = pd.concat([
        d[d['F'] == 'idle'] for d in training_data.values()
    ])
    p_x_z1 = all_idle[all_idle['Z'] == 1]['X'].mean()
    p_x_z0 = all_idle[all_idle['Z'] == 0]['X'].mean()
    instrument_strength = abs(p_x_z1 - p_x_z0)

    return {
        'alpha_z': alpha_z,
        'epsilon': epsilon,
        'seed': seed,
        'ate_lower': ate_lower,
        'ate_upper': ate_upper,
        'ate_width': ate_width,
        'true_ate': true_ate,
        'coverage': coverage,
        'cate_width_z0': cate_width_z0,
        'cate_width_z1': cate_width_z1,
        'feasible': feasible,
        'lp_status': lp_status,
        'instrument_strength': instrument_strength
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(config: Exp3Config = None, verbose: bool = True) -> pd.DataFrame:
    """Run Experiment 3: Instrument Strength Sweep."""

    if config is None:
        config = Exp3Config()

    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Calculate total runs
    total_runs = len(config.alpha_z_values) * len(config.epsilon_values) * config.n_seeds

    if verbose:
        print("=" * 60)
        print("Experiment 3: Instrument Strength Sweep")
        print("=" * 60)
        print(f"αz values: {config.alpha_z_values}")
        print(f"ε values: {config.epsilon_values}")
        print(f"Seeds per config: {config.n_seeds}")
        print(f"Total runs: {total_runs}")
        print(f"Fixed: αu={config.alpha_u}, βu={config.beta_u}, n={config.n_per_env}, K={config.K}")
        print("=" * 60)

    # Generate all configurations
    configs_to_run = []
    for alpha_z in config.alpha_z_values:
        for epsilon in config.epsilon_values:
            for i in range(config.n_seeds):
                seed = config.seed_start + i
                configs_to_run.append((alpha_z, epsilon, seed))

    # Run configurations
    results = []
    completed = 0

    # Determine number of workers
    n_jobs = config.n_jobs
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 4

    if n_jobs > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(run_single_configuration, az, eps, s, config): (az, eps, s)
                for az, eps, s in configs_to_run
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                if verbose and completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (total_runs - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%) "
                          f"- ETA: {remaining:.0f}s")
    else:
        # Sequential execution
        for az, eps, seed in configs_to_run:
            result = run_single_configuration(az, eps, seed, config)
            results.append(result)
            completed += 1

            if verbose and completed % 50 == 0:
                print(f"  Progress: {completed}/{total_runs}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save raw results
    results_df.to_csv(output_dir / 'exp3_results.csv', index=False)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Results saved to: {output_dir / 'exp3_results.csv'}")

    return results_df


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results(results_df: pd.DataFrame, config: Exp3Config = None) -> Dict:
    """Analyze Experiment 3 results."""

    if config is None:
        config = Exp3Config()

    summary = {
        'by_alpha_z_epsilon': {},
        'width_curves': {},
        'thresholds': {},
        'overall': {}
    }

    # Group by (alpha_z, epsilon)
    for (az, eps), group in results_df.groupby(['alpha_z', 'epsilon']):
        key = f"az={az}_eps={eps}"

        feasible = group[group['feasible']]

        summary['by_alpha_z_epsilon'][key] = {
            'alpha_z': az,
            'epsilon': eps,
            'n_runs': len(group),
            'n_feasible': len(feasible),
            'feasibility_rate': len(feasible) / len(group) if len(group) > 0 else 0,
            'width': {
                'mean': feasible['ate_width'].mean() if len(feasible) > 0 else float('nan'),
                'median': feasible['ate_width'].median() if len(feasible) > 0 else float('nan'),
                'std': feasible['ate_width'].std() if len(feasible) > 0 else float('nan'),
                'min': feasible['ate_width'].min() if len(feasible) > 0 else float('nan'),
                'max': feasible['ate_width'].max() if len(feasible) > 0 else float('nan')
            },
            'coverage': feasible['coverage'].mean() if len(feasible) > 0 else float('nan'),
            'instrument_strength': {
                'mean': feasible['instrument_strength'].mean() if len(feasible) > 0 else float('nan'),
                'std': feasible['instrument_strength'].std() if len(feasible) > 0 else float('nan')
            }
        }

    # Width curves by epsilon
    for eps in config.epsilon_values:
        eps_data = results_df[results_df['epsilon'] == eps]
        curve_data = eps_data.groupby('alpha_z').agg({
            'ate_width': ['mean', 'std', 'median'],
            'coverage': 'mean',
            'feasible': 'mean'
        }).reset_index()

        summary['width_curves'][f'eps={eps}'] = {
            'alpha_z': curve_data['alpha_z'].tolist(),
            'width_mean': curve_data[('ate_width', 'mean')].tolist(),
            'width_std': curve_data[('ate_width', 'std')].tolist(),
            'width_median': curve_data[('ate_width', 'median')].tolist(),
            'coverage': curve_data[('coverage', 'mean')].tolist(),
            'feasibility': curve_data[('feasible', 'mean')].tolist()
        }

    # Estimate thresholds (αz where width = 0.8)
    for eps in config.epsilon_values:
        eps_data = results_df[results_df['epsilon'] == eps]
        grouped = eps_data.groupby('alpha_z')['ate_width'].median().reset_index()

        # Find threshold via interpolation
        threshold = None
        for i in range(len(grouped) - 1):
            w1, w2 = grouped.iloc[i]['ate_width'], grouped.iloc[i+1]['ate_width']
            az1, az2 = grouped.iloc[i]['alpha_z'], grouped.iloc[i+1]['alpha_z']

            if w1 >= config.width_threshold >= w2:
                # Linear interpolation
                t = (w1 - config.width_threshold) / (w1 - w2) if w1 != w2 else 0
                threshold = az1 + t * (az2 - az1)
                break

        summary['thresholds'][f'eps={eps}'] = {
            'alpha_z_threshold': threshold,
            'width_threshold': config.width_threshold
        }

    # Overall statistics
    summary['overall'] = {
        'total_runs': len(results_df),
        'overall_feasibility': results_df['feasible'].mean(),
        'overall_coverage': results_df[results_df['feasible']]['coverage'].mean(),
        'width_range': {
            'min': results_df[results_df['feasible']]['ate_width'].min(),
            'max': results_df[results_df['feasible']]['ate_width'].max()
        }
    }

    return summary


def generate_figures(results_df: pd.DataFrame, config: Exp3Config = None):
    """Generate figures for Experiment 3."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for figures")
        return

    if config is None:
        config = Exp3Config()

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Width vs αz for each ε
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(config.epsilon_values)))

    for eps, color in zip(config.epsilon_values, colors):
        eps_data = results_df[results_df['epsilon'] == eps]
        grouped = eps_data.groupby('alpha_z').agg({
            'ate_width': ['mean', 'std']
        }).reset_index()

        ax.plot(grouped['alpha_z'], grouped[('ate_width', 'mean')],
                'o-', color=color, label=f'ε = {eps}', markersize=6)
        ax.fill_between(
            grouped['alpha_z'],
            grouped[('ate_width', 'mean')] - grouped[('ate_width', 'std')],
            grouped[('ate_width', 'mean')] + grouped[('ate_width', 'std')],
            alpha=0.2, color=color
        )

    ax.axhline(y=config.width_threshold, color='red', linestyle='--',
               label=f'Uninformative threshold ({config.width_threshold})')
    ax.set_xlabel('Instrument Strength (αz)', fontsize=12)
    ax.set_ylabel('ATE Bound Width', fontsize=12)
    ax.set_title('Experiment 3: Width vs Instrument Strength', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_width_vs_alpha_z.png', dpi=150)
    plt.close()

    # Figure 2: Coverage heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = results_df.pivot_table(
        values='coverage',
        index='epsilon',
        columns='alpha_z',
        aggfunc='mean'
    )

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.8, vmax=1.0, ax=ax)
    ax.set_title('Coverage Rate by (αz, ε)', fontsize=14)
    ax.set_xlabel('Instrument Strength (αz)', fontsize=12)
    ax.set_ylabel('Naturalness Tolerance (ε)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_coverage_heatmap.png', dpi=150)
    plt.close()

    print(f"Figures saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 3: Instrument Strength Sweep')
    parser.add_argument('--n-seeds', type=int, default=50, help='Seeds per configuration')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (-1 = all)')
    parser.add_argument('--quick', action='store_true', help='Quick run (10 seeds)')
    parser.add_argument('--analyze-only', type=str, help='Analyze existing results file')
    args = parser.parse_args()

    if args.analyze_only:
        results_df = pd.read_csv(args.analyze_only)
        summary = analyze_results(results_df)

        output_dir = Path(__file__).parent.parent / 'outputs'
        with open(output_dir / 'exp3_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        generate_figures(results_df)
        print(f"Analysis saved to: {output_dir / 'exp3_summary.json'}")
        return

    config = Exp3Config(
        n_seeds=10 if args.quick else args.n_seeds,
        n_jobs=args.n_jobs
    )

    # Run experiment
    results_df = run_experiment(config, verbose=True)

    # Analyze results
    summary = analyze_results(results_df, config)

    output_dir = Path(__file__).parent.parent / 'outputs'
    with open(output_dir / 'exp3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Generate figures
    generate_figures(results_df, config)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs: {summary['overall']['total_runs']}")
    print(f"Overall feasibility: {summary['overall']['overall_feasibility']:.1%}")
    print(f"Overall coverage: {summary['overall']['overall_coverage']:.1%}")
    print(f"\nThresholds (αz where Width = {config.width_threshold}):")
    for eps_key, thresh in summary['thresholds'].items():
        az_star = thresh['alpha_z_threshold']
        print(f"  {eps_key}: αz* = {az_star:.2f}" if az_star else f"  {eps_key}: not reached")


if __name__ == '__main__':
    main()
