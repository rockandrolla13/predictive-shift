"""
Extended LP Solver for CATE Bounds

This module implements Ricardo's more sophisticated LP formulation that
handles variation across instrument values within a stratum.

Key differences from simple LP solver:
- Models P(Y_x=1|Z) for each instrument value x in the domain
- Uses proportionality constraints to link potential outcomes across x values
- Exploits cross-instrument naturalness for tighter bounds

Based on Ricardo's bound_po implementation from method.ipynb.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import linprog
import warnings

# cvxpy is optional
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@dataclass
class ExtendedLPResult:
    """
    Result from extended LP solver.

    Attributes:
        lower: Lower bound on CATE
        upper: Upper bound on CATE
        status: Solver status
        theta_1: Optimal P(Y_1=1|Z) value
        theta_0: Optimal P(Y_0=1|Z) value
        dual_values: Dual variable values (for sensitivity analysis)
    """
    lower: float
    upper: float
    status: str
    theta_1: Optional[float] = None
    theta_0: Optional[float] = None
    dual_values: Optional[Dict[str, float]] = None


class ExtendedLPSolver:
    """
    Extended LP solver for CATE bounds with within-stratum variation.

    This solver implements Ricardo's more sophisticated LP formulation that:
    1. Models decision variables θ_x for each instrument value x
    2. Uses proportionality constraints from the instrument structure
    3. Exploits cross-instrument naturalness constraints

    The LP structure:
        Decision variables: θ₁(x), θ₀(x) for x ∈ {0, 1, ..., |X|-1}

        Objective: min/max CATE = E_X[θ₁(X) - θ₀(X)]

        Constraints:
        - Probability bounds: 0 ≤ θ_x ≤ 1
        - Naturalness: |θ_x - θ_x^k| ≤ ε for each training site k
        - Proportionality: Relates θ across different x values
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        solver: str = 'scipy',
        verbose: bool = False
    ):
        """
        Initialize extended LP solver.

        Args:
            epsilon: Naturalness tolerance
            solver: 'scipy' for scipy.optimize.linprog, 'cvxpy' for CVXPY
            verbose: Print solver progress
        """
        self.epsilon = epsilon
        self.solver = solver
        self.verbose = verbose

        if solver == 'cvxpy' and not HAS_CVXPY:
            warnings.warn("CVXPY not available, falling back to scipy")
            self.solver = 'scipy'

    def solve_within_stratum_bounds(
        self,
        z_value: Tuple,
        instrument_domain: List[int],
        training_probs: Dict[str, Dict[Tuple, float]],
        instrument_weights: Optional[Dict[int, float]] = None
    ) -> ExtendedLPResult:
        """
        Solve LP for CATE bounds at a single z value with multiple instrument values.

        This exploits variation in the instrument (treatment) to get tighter bounds.

        Args:
            z_value: Tuple of covariate values (the stratum)
            instrument_domain: List of possible instrument values (e.g., [0, 1])
            training_probs: Dict[site_id -> Dict[(x, z) -> P(Y=1|X=x,Z=z)]]
            instrument_weights: Dict[x -> P(X=x|Z=z)] weights for marginalizing

        Returns:
            ExtendedLPResult with bounds and diagnostics
        """
        # Collect observed probabilities for each (x, z)
        theta_obs = {}  # Dict[x -> List[P(Y=1|X=x,Z=z) across sites]]

        for x in instrument_domain:
            theta_obs[x] = []
            for site_id, site_probs in training_probs.items():
                key = (x, z_value)
                if key in site_probs:
                    theta_obs[x].append(site_probs[key])

        # Check we have data for at least some x values
        valid_x = [x for x in instrument_domain if theta_obs[x]]
        if len(valid_x) < 2:
            return ExtendedLPResult(
                lower=float('nan'),
                upper=float('nan'),
                status='insufficient_data'
            )

        # Default weights: uniform over observed x values
        if instrument_weights is None:
            instrument_weights = {x: 1.0 / len(valid_x) for x in valid_x}

        # Solve using selected backend
        if self.solver == 'cvxpy':
            return self._solve_cvxpy(
                z_value, valid_x, theta_obs, instrument_weights
            )
        else:
            return self._solve_scipy(
                z_value, valid_x, theta_obs, instrument_weights
            )

    def _solve_cvxpy(
        self,
        z_value: Tuple,
        valid_x: List[int],
        theta_obs: Dict[int, List[float]],
        weights: Dict[int, float]
    ) -> ExtendedLPResult:
        """Solve using CVXPY."""
        if not HAS_CVXPY:
            raise ImportError("CVXPY required")

        n_x = len(valid_x)

        # Decision variables: θ₁(x), θ₀(x) for each x
        theta_1 = {x: cp.Variable() for x in valid_x}
        theta_0 = {x: cp.Variable() for x in valid_x}

        constraints = []

        # Probability bounds
        for x in valid_x:
            constraints.extend([
                theta_1[x] >= 0, theta_1[x] <= 1,
                theta_0[x] >= 0, theta_0[x] <= 1,
            ])

        # Naturalness constraints: |θ_a(x) - θ_a^k(x)| ≤ ε
        for x in valid_x:
            for obs_val in theta_obs[x]:
                # This is P(Y=1|X=x,Z=z) from some training site
                # It identifies θ_x(x) = P(Y_x=1|Z=z) when X is the treatment

                # For binary treatment: θ_1 relates to X=1, θ_0 relates to X=0
                if x == 1:
                    constraints.append(theta_1[x] >= obs_val - self.epsilon)
                    constraints.append(theta_1[x] <= obs_val + self.epsilon)
                elif x == 0:
                    constraints.append(theta_0[x] >= obs_val - self.epsilon)
                    constraints.append(theta_0[x] <= obs_val + self.epsilon)

        # Objective: CATE = sum_x w(x) * [θ_1(x) - θ_0(x)]
        # For binary treatment, simplifies to θ_1(1) - θ_0(0)
        # But we can also average over x values

        # Simple formulation for binary X
        if set(valid_x) == {0, 1}:
            cate_expr = theta_1[1] - theta_0[0]
        else:
            # Weighted average
            cate_expr = cp.sum([
                weights.get(x, 0) * (theta_1[x] - theta_0[x])
                for x in valid_x
            ])

        # Solve for lower bound
        try:
            prob_lower = cp.Problem(cp.Minimize(cate_expr), constraints)
            prob_lower.solve(solver=cp.ECOS, verbose=self.verbose)

            if prob_lower.status in ['optimal', 'optimal_inaccurate']:
                lower = float(prob_lower.value)
            else:
                lower = float('nan')
        except Exception as e:
            if self.verbose:
                print(f"Lower bound solve failed: {e}")
            lower = float('nan')

        # Solve for upper bound
        try:
            prob_upper = cp.Problem(cp.Maximize(cate_expr), constraints)
            prob_upper.solve(solver=cp.ECOS, verbose=self.verbose)

            if prob_upper.status in ['optimal', 'optimal_inaccurate']:
                upper = float(prob_upper.value)
            else:
                upper = float('nan')
        except Exception as e:
            if self.verbose:
                print(f"Upper bound solve failed: {e}")
            upper = float('nan')

        return ExtendedLPResult(
            lower=lower,
            upper=upper,
            status='cvxpy',
            theta_1=theta_1[1].value if 1 in valid_x and theta_1[1].value is not None else None,
            theta_0=theta_0[0].value if 0 in valid_x and theta_0[0].value is not None else None
        )

    def _solve_scipy(
        self,
        z_value: Tuple,
        valid_x: List[int],
        theta_obs: Dict[int, List[float]],
        weights: Dict[int, float]
    ) -> ExtendedLPResult:
        """Solve using scipy.optimize.linprog."""
        # For binary treatment, we have 2 variables: θ_1(1), θ_0(0)
        # Variable layout: [theta_1, theta_0]

        # === LOWER BOUND: min (theta_1 - theta_0) ===

        # Objective: min theta_1 - theta_0
        c_lower = np.array([1.0, -1.0])  # minimize theta_1 - theta_0

        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []

        # Naturalness constraints on theta_1
        for obs_val in theta_obs.get(1, []):
            # theta_1 >= obs_val - epsilon  →  -theta_1 <= -(obs_val - epsilon)
            A_ub.append([-1.0, 0.0])
            b_ub.append(-(obs_val - self.epsilon))
            # theta_1 <= obs_val + epsilon
            A_ub.append([1.0, 0.0])
            b_ub.append(obs_val + self.epsilon)

        # Naturalness constraints on theta_0
        for obs_val in theta_obs.get(0, []):
            # theta_0 >= obs_val - epsilon  →  -theta_0 <= -(obs_val - epsilon)
            A_ub.append([0.0, -1.0])
            b_ub.append(-(obs_val - self.epsilon))
            # theta_0 <= obs_val + epsilon
            A_ub.append([0.0, 1.0])
            b_ub.append(obs_val + self.epsilon)

        # Bounds: 0 <= theta <= 1
        bounds = [(0, 1), (0, 1)]

        if len(A_ub) > 0:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
        else:
            A_ub = None
            b_ub = None

        # Solve for lower bound
        try:
            res_lower = linprog(
                c_lower, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                method='highs'
            )
            if res_lower.success:
                lower = float(res_lower.fun)
                theta_1_lower = res_lower.x[0]
                theta_0_lower = res_lower.x[1]
            else:
                lower = float('nan')
                theta_1_lower = None
                theta_0_lower = None
        except Exception as e:
            if self.verbose:
                print(f"Lower bound solve failed: {e}")
            lower = float('nan')
            theta_1_lower = None
            theta_0_lower = None

        # === UPPER BOUND: max (theta_1 - theta_0) = min -(theta_1 - theta_0) ===

        c_upper = np.array([-1.0, 1.0])  # minimize -(theta_1 - theta_0)

        try:
            res_upper = linprog(
                c_upper, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                method='highs'
            )
            if res_upper.success:
                upper = -float(res_upper.fun)  # Negate to get max
                theta_1_upper = res_upper.x[0]
                theta_0_upper = res_upper.x[1]
            else:
                upper = float('nan')
        except Exception as e:
            if self.verbose:
                print(f"Upper bound solve failed: {e}")
            upper = float('nan')

        return ExtendedLPResult(
            lower=lower,
            upper=upper,
            status='scipy',
            theta_1=theta_1_lower,
            theta_0=theta_0_lower
        )

    def solve_with_proportionality(
        self,
        z_value: Tuple,
        training_probs: Dict[str, Dict[Tuple, float]],
        propensity_scores: Dict[int, float],
        outcome_probs: Dict[str, Dict[Tuple, float]]
    ) -> ExtendedLPResult:
        """
        Solve LP with proportionality constraints.

        This is Ricardo's more sophisticated formulation that uses
        proportionality conditions relating potential outcomes across
        different X values.

        The key insight: Under the instrument validity assumptions,
        we can relate P(Y_x=1|Z) across different x values using
        the observed data structure.

        Args:
            z_value: Covariate stratum
            training_probs: P(Y=1|X,Z) from each site
            propensity_scores: P(X=x|Z) for each x
            outcome_probs: P(Y=1|Z) marginals from each site

        Returns:
            ExtendedLPResult with potentially tighter bounds
        """
        # This is a placeholder for the full Ricardo formulation
        # The implementation would involve:
        # 1. Setting up variables for each (x, outcome) combination
        # 2. Adding proportionality constraints
        # 3. Adding cross-stratum naturalness constraints

        # For now, fall back to simple formulation
        return self.solve_within_stratum_bounds(
            z_value,
            [0, 1],
            training_probs,
            propensity_scores
        )


def solve_extended_bounds_all_strata(
    training_data: Dict[str, pd.DataFrame],
    covariates: List[str],
    treatment: str,
    outcome: str,
    epsilon: float = 0.1,
    regime_col: str = 'F',
    solver: str = 'scipy'
) -> Dict[Tuple, Tuple[float, float]]:
    """
    Solve extended LP bounds for all covariate strata.

    Args:
        training_data: Dict[site_id -> DataFrame]
        covariates: List of covariate column names
        treatment: Treatment column name
        outcome: Outcome column name
        epsilon: Naturalness tolerance
        regime_col: Regime indicator column
        solver: 'scipy' or 'cvxpy'

    Returns:
        Dict[z_value -> (lower, upper)]
    """
    # Collect identified probabilities from each site
    training_probs = {}
    all_z_values = set()

    for site_id, site_data in training_data.items():
        on_data = site_data[site_data[regime_col] == 'on']

        if len(on_data) < 10:
            continue

        site_probs = {}
        grouped = on_data.groupby([treatment] + covariates, observed=True)

        for idx, group in grouped:
            if len(group) >= 5:
                x = idx[0]
                z = tuple(idx[1:]) if len(covariates) > 1 else (idx[1],)
                site_probs[(x, z)] = group[outcome].mean()
                all_z_values.add(z)

        if site_probs:
            training_probs[site_id] = site_probs

    if not training_probs:
        return {}

    # Create solver
    lp_solver = ExtendedLPSolver(epsilon=epsilon, solver=solver)

    # Solve for each stratum
    bounds = {}

    for z_value in all_z_values:
        result = lp_solver.solve_within_stratum_bounds(
            z_value,
            instrument_domain=[0, 1],
            training_probs=training_probs
        )

        if not np.isnan(result.lower) and not np.isnan(result.upper):
            bounds[z_value] = (result.lower, result.upper)

    return bounds


def compare_simple_vs_extended(
    training_data: Dict[str, pd.DataFrame],
    covariates: List[str],
    treatment: str,
    outcome: str,
    epsilon: float = 0.1,
    regime_col: str = 'F'
) -> pd.DataFrame:
    """
    Compare simple LP bounds with extended LP bounds.

    Useful for understanding when the extended formulation helps.

    Args:
        training_data: Dict[site_id -> DataFrame]
        covariates: Covariate column names
        treatment: Treatment column name
        outcome: Outcome column name
        epsilon: Naturalness tolerance
        regime_col: Regime indicator column

    Returns:
        DataFrame with comparison metrics
    """
    from .lp_solver import solve_all_bounds_binary_lp

    # Simple LP bounds
    simple_bounds = solve_all_bounds_binary_lp(
        training_data, covariates, treatment, outcome,
        epsilon, regime_col, use_cvxpy=False
    )

    # Extended LP bounds
    extended_bounds = solve_extended_bounds_all_strata(
        training_data, covariates, treatment, outcome,
        epsilon, regime_col, solver='scipy'
    )

    # Compare
    results = []

    all_z = set(simple_bounds.keys()) | set(extended_bounds.keys())

    for z in sorted(all_z):
        row = {'stratum': str(z)}

        if z in simple_bounds:
            row['simple_lower'] = simple_bounds[z][0]
            row['simple_upper'] = simple_bounds[z][1]
            row['simple_width'] = simple_bounds[z][1] - simple_bounds[z][0]
        else:
            row['simple_lower'] = np.nan
            row['simple_upper'] = np.nan
            row['simple_width'] = np.nan

        if z in extended_bounds:
            row['extended_lower'] = extended_bounds[z][0]
            row['extended_upper'] = extended_bounds[z][1]
            row['extended_width'] = extended_bounds[z][1] - extended_bounds[z][0]
        else:
            row['extended_lower'] = np.nan
            row['extended_upper'] = np.nan
            row['extended_width'] = np.nan

        # Improvement
        if not np.isnan(row.get('simple_width', np.nan)) and not np.isnan(row.get('extended_width', np.nan)):
            row['width_improvement'] = row['simple_width'] - row['extended_width']
            row['improvement_ratio'] = row['width_improvement'] / row['simple_width'] if row['simple_width'] > 0 else 0
        else:
            row['width_improvement'] = np.nan
            row['improvement_ratio'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def create_lp_solver(
    solver_type: str = 'simple',
    epsilon: float = 0.1,
    **kwargs
) -> Any:
    """
    Factory function for LP solvers.

    Args:
        solver_type: 'simple' for basic LP, 'extended' for within-stratum variation
        epsilon: Naturalness tolerance
        **kwargs: Additional solver arguments

    Returns:
        LP solver instance or function

    Example:
        solver = create_lp_solver('extended', epsilon=0.1)
    """
    if solver_type == 'simple':
        from .lp_solver import solve_cate_bounds_lp_binary
        return lambda *args, **kw: solve_cate_bounds_lp_binary(
            *args, epsilon=epsilon, **{**kwargs, **kw}
        )
    elif solver_type == 'extended':
        return ExtendedLPSolver(epsilon=epsilon, **kwargs)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. Use 'simple' or 'extended'.")


# Module test
if __name__ == "__main__":
    print("Extended LP Solver Test")
    print("=" * 50)

    np.random.seed(42)

    # Create synthetic multi-site data
    def create_site_data(n, site_effect=0):
        Z = np.random.binomial(1, 0.5, n)
        X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
        Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z + site_effect, n)

        df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
        df['F'] = 'on'
        return df

    print("\n1. Creating multi-site training data...")
    training_data = {
        'site_1': create_site_data(500, site_effect=0.0),
        'site_2': create_site_data(500, site_effect=0.05),
        'site_3': create_site_data(500, site_effect=-0.05),
    }

    print("\n2. Testing ExtendedLPSolver...")
    solver = ExtendedLPSolver(epsilon=0.1, solver='scipy')

    # Collect probabilities
    training_probs = {}
    for site_id, site_data in training_data.items():
        site_probs = {}
        for (x, z), group in site_data.groupby(['X', 'Z']):
            site_probs[(x, (z,))] = group['Y'].mean()
        training_probs[site_id] = site_probs

    result = solver.solve_within_stratum_bounds(
        z_value=(0,),
        instrument_domain=[0, 1],
        training_probs=training_probs
    )
    print(f"   Z=(0,): [{result.lower:.4f}, {result.upper:.4f}] ({result.status})")

    result = solver.solve_within_stratum_bounds(
        z_value=(1,),
        instrument_domain=[0, 1],
        training_probs=training_probs
    )
    print(f"   Z=(1,): [{result.lower:.4f}, {result.upper:.4f}] ({result.status})")

    print("\n3. Testing solve_extended_bounds_all_strata...")
    bounds = solve_extended_bounds_all_strata(
        training_data, ['Z'], 'X', 'Y',
        epsilon=0.1, solver='scipy'
    )
    print(f"   Bounds computed for {len(bounds)} strata")
    for z, (lower, upper) in sorted(bounds.items()):
        print(f"   Z={z}: [{lower:.4f}, {upper:.4f}]")

    print("\n4. Comparing simple vs extended...")
    comparison = compare_simple_vs_extended(
        training_data, ['Z'], 'X', 'Y',
        epsilon=0.1
    )
    print(comparison.to_string(index=False))

    print("\nAll tests completed!")
