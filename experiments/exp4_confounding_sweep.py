"""
Experiment 4: Confounding Strength Sweep (αu vs βu)

This experiment investigates how treatment confounding (αu) and outcome
confounding (βu) affect bound width, using a factorial design.

Research Questions:
- Does βu have a larger effect on Width than αu?
- Is the interaction multiplicative (Width ≈ f(αu) × g(βu))?
- What is the "exchange rate" ρ = (∂W/∂βu) / (∂W/∂αu)?

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
class Exp4Config:
    """Configuration for Experiment 4."""
    # Factorial grid
    alpha_u_values: List[float] = None  # Treatment confounding
    beta_u_values: List[float] = None   # Outcome confounding

    # Fixed parameters
    alpha_z: float = 2.0       # Medium instrument strength
    epsilon: float = 0.10      # Calibrated naturalness tolerance
    beta_x: float = 1.0        # True causal effect
    n_per_env: int = 2000      # Sample size per environment
    K: int = 5                 # Number of training environments

    # Replication
    n_seeds: int = 50          # Seeds per configuration
    seed_start: int = 42

    # Analysis
    delta_for_derivatives: float = 0.1  # For exchange rate computation

    # Parallelization
    n_jobs: int = -1

    def __post_init__(self):
        if self.alpha_u_values is None:
            self.alpha_u_values = [0.25, 0.5, 1.0, 1.5, 2.0]
        if self.beta_u_values is None:
            self.beta_u_values = [0.25, 0.5, 1.0, 1.5, 2.0]


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


def compute_naive_ate(training_data: Dict[str, pd.DataFrame]) -> float:
    """Compute naive (confounded) ATE estimate."""
    all_idle = pd.concat([d[d['F'] == 'idle'] for d in training_data.values()])
    treated = all_idle[all_idle['X'] == 1]['Y'].mean()
    control = all_idle[all_idle['X'] == 0]['Y'].mean()
    return treated - control


# =============================================================================
# Single Run
# =============================================================================

def run_single_configuration(
    alpha_u: float,
    beta_u: float,
    seed: int,
    config: Exp4Config
) -> Dict:
    """Run a single configuration and return results."""

    # Create DGP
    dgp = ConfoundedInstrumentDGP(
        alpha_0=0.0,
        alpha_z=config.alpha_z,
        alpha_u=alpha_u,
        alpha_w=0.0,
        beta_0=0.0,
        beta_x=config.beta_x,
        beta_u=beta_u,
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

    # Compute naive ATE (confounded)
    naive_ate = compute_naive_ate(training_data)
    naive_bias = naive_ate - true_ate

    # Solve bounds using LP
    try:
        bounds = solve_all_bounds_binary_lp(
            training_data=training_data,
            covariates=['Z'],
            treatment='X',
            outcome='Y',
            epsilon=config.epsilon,
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

    return {
        'alpha_u': alpha_u,
        'beta_u': beta_u,
        'seed': seed,
        'ate_lower': ate_lower,
        'ate_upper': ate_upper,
        'ate_width': ate_width,
        'true_ate': true_ate,
        'naive_ate': naive_ate,
        'naive_bias': naive_bias,
        'coverage': coverage,
        'cate_width_z0': cate_width_z0,
        'cate_width_z1': cate_width_z1,
        'feasible': feasible,
        'lp_status': lp_status
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(config: Exp4Config = None, verbose: bool = True) -> pd.DataFrame:
    """Run Experiment 4: Confounding Sweep."""

    if config is None:
        config = Exp4Config()

    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Calculate total runs
    total_runs = len(config.alpha_u_values) * len(config.beta_u_values) * config.n_seeds

    if verbose:
        print("=" * 60)
        print("Experiment 4: Confounding Sweep (αu vs βu)")
        print("=" * 60)
        print(f"αu values: {config.alpha_u_values}")
        print(f"βu values: {config.beta_u_values}")
        print(f"Seeds per config: {config.n_seeds}")
        print(f"Total runs: {total_runs}")
        print(f"Fixed: αz={config.alpha_z}, ε={config.epsilon}, n={config.n_per_env}, K={config.K}")
        print("=" * 60)

    # Generate all configurations
    configs_to_run = []
    for alpha_u in config.alpha_u_values:
        for beta_u in config.beta_u_values:
            for i in range(config.n_seeds):
                seed = config.seed_start + i
                configs_to_run.append((alpha_u, beta_u, seed))

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
                executor.submit(run_single_configuration, au, bu, s, config): (au, bu, s)
                for au, bu, s in configs_to_run
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
        for au, bu, seed in configs_to_run:
            result = run_single_configuration(au, bu, seed, config)
            results.append(result)
            completed += 1

            if verbose and completed % 50 == 0:
                print(f"  Progress: {completed}/{total_runs}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save raw results
    results_df.to_csv(output_dir / 'exp4_results.csv', index=False)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Results saved to: {output_dir / 'exp4_results.csv'}")

    return results_df


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results(results_df: pd.DataFrame, config: Exp4Config = None) -> Dict:
    """Analyze Experiment 4 results with ANOVA and exchange rate."""

    if config is None:
        config = Exp4Config()

    summary = {
        'by_configuration': {},
        'anova': {},
        'exchange_rate': {},
        'dominance_test': {},
        'overall': {}
    }

    # Group by (alpha_u, beta_u)
    for (au, bu), group in results_df.groupby(['alpha_u', 'beta_u']):
        key = f"au={au}_bu={bu}"

        feasible = group[group['feasible']]

        summary['by_configuration'][key] = {
            'alpha_u': au,
            'beta_u': bu,
            'n_runs': len(group),
            'n_feasible': len(feasible),
            'feasibility_rate': len(feasible) / len(group) if len(group) > 0 else 0,
            'width': {
                'mean': feasible['ate_width'].mean() if len(feasible) > 0 else float('nan'),
                'median': feasible['ate_width'].median() if len(feasible) > 0 else float('nan'),
                'std': feasible['ate_width'].std() if len(feasible) > 0 else float('nan')
            },
            'coverage': feasible['coverage'].mean() if len(feasible) > 0 else float('nan'),
            'naive_bias': {
                'mean': feasible['naive_bias'].mean() if len(feasible) > 0 else float('nan'),
                'std': feasible['naive_bias'].std() if len(feasible) > 0 else float('nan')
            }
        }

    # ANOVA analysis
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        # Filter to feasible runs
        feasible_df = results_df[results_df['feasible']].copy()

        if len(feasible_df) > 10:
            # Fit interaction model: width ~ alpha_u * beta_u
            model = ols('ate_width ~ alpha_u * beta_u', data=feasible_df).fit()

            anova_table = anova_lm(model, typ=2)

            summary['anova'] = {
                'coefficients': {
                    'intercept': float(model.params['Intercept']),
                    'alpha_u': float(model.params['alpha_u']),
                    'beta_u': float(model.params['beta_u']),
                    'interaction': float(model.params['alpha_u:beta_u'])
                },
                'std_errors': {
                    'intercept': float(model.bse['Intercept']),
                    'alpha_u': float(model.bse['alpha_u']),
                    'beta_u': float(model.bse['beta_u']),
                    'interaction': float(model.bse['alpha_u:beta_u'])
                },
                'p_values': {
                    'intercept': float(model.pvalues['Intercept']),
                    'alpha_u': float(model.pvalues['alpha_u']),
                    'beta_u': float(model.pvalues['beta_u']),
                    'interaction': float(model.pvalues['alpha_u:beta_u'])
                },
                'r_squared': float(model.rsquared),
                'r_squared_adj': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue)
            }

            # Dominance test: β_βu > β_αu?
            beta_au = model.params['alpha_u']
            beta_bu = model.params['beta_u']
            diff = beta_bu - beta_au

            # Get covariance for standard error of difference
            cov_matrix = model.cov_params()
            var_diff = (cov_matrix.loc['alpha_u', 'alpha_u'] +
                       cov_matrix.loc['beta_u', 'beta_u'] -
                       2 * cov_matrix.loc['alpha_u', 'beta_u'])
            se_diff = np.sqrt(var_diff) if var_diff > 0 else 0.001

            from scipy import stats
            t_stat = diff / se_diff
            p_value = 1 - stats.t.cdf(t_stat, df=model.df_resid)  # One-sided

            summary['dominance_test'] = {
                'beta_u_coefficient': float(beta_bu),
                'alpha_u_coefficient': float(beta_au),
                'difference': float(diff),
                'se_difference': float(se_diff),
                't_statistic': float(t_stat),
                'p_value_one_sided': float(p_value),
                'beta_u_dominates': bool(p_value < 0.05 and diff > 0)
            }

            # Exchange rate at center (αu=1.0, βu=1.0)
            beta_interaction = model.params['alpha_u:beta_u']
            exchange_rate = (beta_bu + beta_interaction * 1.0) / (beta_au + beta_interaction * 1.0)

            summary['exchange_rate'] = {
                'at_center': float(exchange_rate),
                'interpretation': f"1 unit increase in βu has {exchange_rate:.2f}x the effect of 1 unit increase in αu"
            }

    except ImportError:
        summary['anova'] = {'error': 'statsmodels not available'}
        summary['dominance_test'] = {'error': 'statsmodels not available'}
        summary['exchange_rate'] = {'error': 'statsmodels not available'}
    except Exception as e:
        summary['anova'] = {'error': str(e)}
        summary['dominance_test'] = {'error': str(e)}
        summary['exchange_rate'] = {'error': str(e)}

    # Overall statistics
    feasible = results_df[results_df['feasible']]
    summary['overall'] = {
        'total_runs': len(results_df),
        'total_feasible': len(feasible),
        'overall_feasibility': results_df['feasible'].mean(),
        'overall_coverage': feasible['coverage'].mean() if len(feasible) > 0 else float('nan'),
        'width_range': {
            'min': feasible['ate_width'].min() if len(feasible) > 0 else float('nan'),
            'max': feasible['ate_width'].max() if len(feasible) > 0 else float('nan')
        }
    }

    return summary


def generate_figures(results_df: pd.DataFrame, config: Exp4Config = None):
    """Generate figures for Experiment 4."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for figures")
        return

    if config is None:
        config = Exp4Config()

    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Width heatmap (αu × βu)
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = results_df[results_df['feasible']].pivot_table(
        values='ate_width',
        index='beta_u',
        columns='alpha_u',
        aggfunc='median'
    )

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Median Width'})
    ax.set_title('Experiment 4: ATE Bound Width by Confounding Strength', fontsize=14)
    ax.set_xlabel('Treatment Confounding (αu)', fontsize=12)
    ax.set_ylabel('Outcome Confounding (βu)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_width_heatmap.png', dpi=150)
    plt.close()

    # Figure 2: Coverage heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_cov = results_df[results_df['feasible']].pivot_table(
        values='coverage',
        index='beta_u',
        columns='alpha_u',
        aggfunc='mean'
    )

    sns.heatmap(pivot_cov, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.8, vmax=1.0, ax=ax, cbar_kws={'label': 'Coverage Rate'})
    ax.set_title('Coverage Rate by Confounding Strength', fontsize=14)
    ax.set_xlabel('Treatment Confounding (αu)', fontsize=12)
    ax.set_ylabel('Outcome Confounding (βu)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_coverage_heatmap.png', dpi=150)
    plt.close()

    # Figure 3: Marginal effects comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Width vs αu for different βu
    feasible = results_df[results_df['feasible']]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(config.beta_u_values)))

    for bu, color in zip(config.beta_u_values, colors):
        bu_data = feasible[feasible['beta_u'] == bu]
        grouped = bu_data.groupby('alpha_u')['ate_width'].agg(['mean', 'std']).reset_index()
        axes[0].plot(grouped['alpha_u'], grouped['mean'], 'o-', color=color,
                     label=f'βu = {bu}', markersize=6)

    axes[0].set_xlabel('Treatment Confounding (αu)', fontsize=12)
    axes[0].set_ylabel('Mean ATE Width', fontsize=12)
    axes[0].set_title('Width vs αu (by βu level)', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Right: Width vs βu for different αu
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(config.alpha_u_values)))

    for au, color in zip(config.alpha_u_values, colors):
        au_data = feasible[feasible['alpha_u'] == au]
        grouped = au_data.groupby('beta_u')['ate_width'].agg(['mean', 'std']).reset_index()
        axes[1].plot(grouped['beta_u'], grouped['mean'], 'o-', color=color,
                     label=f'αu = {au}', markersize=6)

    axes[1].set_xlabel('Outcome Confounding (βu)', fontsize=12)
    axes[1].set_ylabel('Mean ATE Width', fontsize=12)
    axes[1].set_title('Width vs βu (by αu level)', fontsize=12)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_marginal_effects.png', dpi=150)
    plt.close()

    # Figure 4: Contour plot
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = feasible.pivot_table(
        values='ate_width',
        index='beta_u',
        columns='alpha_u',
        aggfunc='median'
    )

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    contour = ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd')
    ax.contour(X, Y, Z, levels=[0.3, 0.5, 0.7], colors='black', linewidths=1)
    plt.colorbar(contour, ax=ax, label='ATE Width')

    ax.set_xlabel('Treatment Confounding (αu)', fontsize=12)
    ax.set_ylabel('Outcome Confounding (βu)', fontsize=12)
    ax.set_title('Width Contours (αu, βu)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_contour.png', dpi=150)
    plt.close()

    print(f"Figures saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment 4: Confounding Sweep')
    parser.add_argument('--n-seeds', type=int, default=50, help='Seeds per configuration')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (-1 = all)')
    parser.add_argument('--quick', action='store_true', help='Quick run (10 seeds)')
    parser.add_argument('--analyze-only', type=str, help='Analyze existing results file')
    args = parser.parse_args()

    if args.analyze_only:
        results_df = pd.read_csv(args.analyze_only)
        summary = analyze_results(results_df)

        output_dir = Path(__file__).parent.parent / 'outputs'
        with open(output_dir / 'exp4_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        generate_figures(results_df)
        print(f"Analysis saved to: {output_dir / 'exp4_summary.json'}")
        return

    config = Exp4Config(
        n_seeds=10 if args.quick else args.n_seeds,
        n_jobs=args.n_jobs
    )

    # Run experiment
    results_df = run_experiment(config, verbose=True)

    # Analyze results
    summary = analyze_results(results_df, config)

    output_dir = Path(__file__).parent.parent / 'outputs'
    with open(output_dir / 'exp4_summary.json', 'w') as f:
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

    if 'coefficients' in summary.get('anova', {}):
        print(f"\nANOVA Results:")
        print(f"  R² = {summary['anova']['r_squared']:.3f}")
        print(f"  αu coefficient: {summary['anova']['coefficients']['alpha_u']:.4f}")
        print(f"  βu coefficient: {summary['anova']['coefficients']['beta_u']:.4f}")
        print(f"  Interaction: {summary['anova']['coefficients']['interaction']:.4f}")

    if 'beta_u_dominates' in summary.get('dominance_test', {}):
        dom = summary['dominance_test']
        print(f"\nDominance Test:")
        print(f"  βu dominates αu: {dom['beta_u_dominates']}")
        print(f"  p-value (one-sided): {dom['p_value_one_sided']:.4f}")

    if 'at_center' in summary.get('exchange_rate', {}):
        print(f"\nExchange Rate: {summary['exchange_rate']['at_center']:.2f}")
        print(f"  {summary['exchange_rate']['interpretation']}")


if __name__ == '__main__':
    main()
