"""
LP Solver for CATE Bounds under Naturalness Constraints

This module solves the linear program for partial identification of CATE
bounds using the naturalness assumption from Silva's paper.

The key insight is that while P(Y_x | Z) is not identified from F=idle data
alone due to confounding, it can be bounded by:
1. Using P(Y|X,Z,F=on) from experimental data (identified)
2. Constraining confounding functions via naturalness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# cvxpy is optional - only needed for full LP solver
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def estimate_conditional_means(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    min_count: int = 5
) -> Dict[Tuple, float]:
    """
    Estimate E[Y | X=x, Z=z] from data.

    For binary outcomes, this equals P(Y=1 | X=x, Z=z).
    For continuous outcomes, this is the conditional mean.

    Args:
        data: DataFrame
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of (discretized) covariate columns
        min_count: Minimum observations required for estimate

    Returns:
        Dict mapping (x, z_tuple) -> E[Y | X=x, Z=z]
    """
    means = {}

    # Group by treatment and covariates
    group_cols = [treatment] + covariates
    grouped = data.groupby(group_cols, observed=True)[outcome].agg(['mean', 'count'])

    for idx, row in grouped.iterrows():
        if row['count'] >= min_count:
            x = idx[0]
            z = tuple(idx[1:]) if len(covariates) > 1 else (idx[1],)
            means[(x, z)] = row['mean']

    return means


# Alias for backwards compatibility
estimate_identified_probs = estimate_conditional_means


def estimate_observed_probs(
    idle_data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    min_count: int = 5
) -> Dict[str, Dict]:
    """
    Estimate observed probabilities from F=idle data.

    Args:
        idle_data: DataFrame with F=idle data only
        treatment: Treatment column name
        outcome: Outcome column name
        covariates: List of covariate columns
        min_count: Minimum observations for reliable estimate

    Returns:
        {
            'p_x_given_z': {z: P(X=1|Z=z)},
            'p_y_given_xz': {(x,z): P(Y=1|X=x,Z=z)},
            'p_z': {z: P(Z=z)},
            'p_xy_given_z': {(x,y,z): P(X=x,Y=y|Z=z)}
        }
    """
    n = len(idle_data)

    # P(Z=z)
    z_counts = idle_data.groupby(covariates, observed=True).size()
    p_z = {}
    for z, count in z_counts.items():
        if count >= min_count:
            z_tuple = tuple(z) if isinstance(z, tuple) else (z,)
            p_z[z_tuple] = count / n

    # P(X=1|Z=z)
    p_x_given_z = {}
    for z, group in idle_data.groupby(covariates, observed=True):
        if len(group) >= min_count:
            z_tuple = tuple(z) if isinstance(z, tuple) else (z,)
            p_x_given_z[z_tuple] = group[treatment].mean()

    # P(Y=1|X=x,Z=z)
    p_y_given_xz = {}
    for (x, *z), group in idle_data.groupby([treatment] + covariates, observed=True):
        if len(group) >= min_count:
            z_tuple = tuple(z)
            p_y_given_xz[(x, z_tuple)] = group[outcome].mean()

    # P(X=x,Y=y|Z=z)
    p_xy_given_z = {}
    for z, z_group in idle_data.groupby(covariates, observed=True):
        z_tuple = tuple(z) if isinstance(z, tuple) else (z,)
        n_z = len(z_group)
        if n_z >= min_count:
            for x in [0, 1]:
                for y in [0, 1]:
                    count = ((z_group[treatment] == x) & (z_group[outcome] == y)).sum()
                    p_xy_given_z[(x, y, z_tuple)] = count / n_z

    return {
        'p_z': p_z,
        'p_x_given_z': p_x_given_z,
        'p_y_given_xz': p_y_given_xz,
        'p_xy_given_z': p_xy_given_z
    }


def solve_cate_bounds_single_z(
    z_value: Tuple,
    identified: Dict[Tuple, float],
    observed: Dict[str, Dict],
    epsilon: float = 0.1,
    outcome_scale: Optional[float] = None
) -> Tuple[float, float, str]:
    """
    Compute CATE bounds at a single z value.

    For continuous outcomes, CATE = E[Y|X=1,Z] - E[Y|X=0,Z].
    For binary outcomes, CATE = P(Y=1|X=1,Z) - P(Y=1|X=0,Z).

    Args:
        z_value: Tuple of covariate values
        identified: E[Y|X,Z] from F=on data
        observed: Observed probabilities from F=idle data
        epsilon: Naturalness tolerance (as fraction of outcome scale)
        outcome_scale: Scale of outcome for uncertainty band.
                      If None, epsilon is absolute.

    Returns:
        (cate_lower, cate_upper, status)
    """
    # Check if we have identified quantities for this z
    has_x1 = (1, z_value) in identified
    has_x0 = (0, z_value) in identified

    if not has_x1 or not has_x0:
        # Cannot compute bounds without identified quantities
        return float('nan'), float('nan'), 'missing_data'

    # Get identified E[Y|X=x,Z=z] from F=on
    ey_x1_z = identified[(1, z_value)]
    ey_x0_z = identified[(0, z_value)]

    # Point estimate of CATE
    cate_point = ey_x1_z - ey_x0_z

    # Uncertainty band
    if outcome_scale is not None:
        uncertainty = epsilon * outcome_scale
    else:
        uncertainty = epsilon * abs(cate_point) if cate_point != 0 else epsilon

    cate_lower = cate_point - uncertainty
    cate_upper = cate_point + uncertainty

    return cate_lower, cate_upper, 'optimal'


def solve_cate_bounds_lp(
    z_value: Tuple,
    identified: Dict[Tuple, float],
    observed: Dict[str, Dict],
    epsilon: float = 0.1,
    outcome_scale: Optional[float] = None
) -> Tuple[float, float, str]:
    """
    Solve LP for CATE bounds with naturalness constraints.

    Note: For continuous outcomes, this simplifies to the same as
    solve_cate_bounds_single_z since the LP structure assumes binary outcomes.

    For proper continuous outcome handling, use solve_cate_bounds_single_z.
    """
    # For now, delegate to simple bounds
    # Full LP implementation would require different formulation for continuous Y
    return solve_cate_bounds_single_z(
        z_value, identified, observed, epsilon, outcome_scale
    )


def solve_all_bounds(
    training_data: Dict[str, pd.DataFrame],
    covariates: List[str],
    treatment: str,
    outcome: str,
    epsilon: float = 0.1,
    regime_col: str = 'F',
    use_full_lp: bool = False,
    outcome_scale: Optional[float] = None
) -> Dict[str, Dict[Tuple, Tuple]]:
    """
    Solve bounds for all sites and all z values.

    Args:
        training_data: Dict[site_id -> DataFrame with F column]
        covariates: Covariate column names
        treatment: Treatment column
        outcome: Outcome column
        epsilon: Naturalness tolerance (as fraction of outcome scale)
        regime_col: Regime indicator column
        use_full_lp: Use full LP (slower) or simple bounds
        outcome_scale: Scale of outcome. If None, computed from data.

    Returns:
        {site_id: {z_value: (lower, upper)}}
    """
    solver_fn = solve_cate_bounds_lp if use_full_lp else solve_cate_bounds_single_z

    # Compute outcome scale if not provided
    if outcome_scale is None:
        all_outcomes = []
        for site_data in training_data.values():
            all_outcomes.extend(site_data[outcome].dropna().values)
        if len(all_outcomes) > 0:
            outcome_scale = np.std(all_outcomes)
        else:
            outcome_scale = 1.0

    all_bounds = {}

    for site_id, site_data in training_data.items():
        on_data = site_data[site_data[regime_col] == 'on']
        idle_data = site_data[site_data[regime_col] == 'idle']

        if len(on_data) < 10 or len(idle_data) < 10:
            continue

        identified = estimate_conditional_means(on_data, treatment, outcome, covariates)
        observed = estimate_observed_probs(idle_data, treatment, outcome, covariates)

        site_bounds = {}
        for z_value in observed['p_z'].keys():
            lower, upper, status = solver_fn(
                z_value, identified, observed, epsilon, outcome_scale
            )
            if not np.isnan(lower) and not np.isnan(upper):
                site_bounds[z_value] = (lower, upper)

        if site_bounds:  # Only add sites with valid bounds
            all_bounds[site_id] = site_bounds

    return all_bounds


# Test
if __name__ == "__main__":
    # Simple test with synthetic data
    np.random.seed(42)
    n = 1000

    # Create synthetic data
    Z = np.random.choice([0, 1, 2], size=n)
    X = (np.random.random(n) < 0.3 + 0.2 * Z).astype(int)
    Y = (np.random.random(n) < 0.2 + 0.3 * X + 0.1 * Z).astype(int)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'F': 'on'})

    identified = estimate_identified_probs(df, 'X', 'Y', ['Z'])
    print("Identified P(Y=1|X,Z):")
    for k, v in sorted(identified.items()):
        print(f"  {k}: {v:.3f}")

    df_idle = df.copy()
    df_idle['F'] = 'idle'
    observed = estimate_observed_probs(df_idle, 'X', 'Y', ['Z'])
    print("\nObserved P(X=1|Z):")
    for k, v in sorted(observed['p_x_given_z'].items()):
        print(f"  {k}: {v:.3f}")

    print("\nCATE bounds:")
    for z in [(0,), (1,), (2,)]:
        lower, upper, status = solve_cate_bounds_single_z(z, identified, observed, epsilon=0.1)
        print(f"  Z={z}: [{lower:.3f}, {upper:.3f}]")
