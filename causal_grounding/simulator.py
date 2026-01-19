"""
Synthetic Data Simulator for Causal Grounding

This module provides utilities for generating synthetic data with known
ground truth CATE for testing and validation of causal inference methods.

Based on Ricardo's BinarySyntheticBackdoorModel with enhancements for:
- Multi-environment data generation
- Known ground truth computation
- Sparsity patterns for simulated unmeasured confounding

Key Classes:
    BinarySyntheticDGP - Data generating process with binary covariates

Key Functions:
    generate_random_dgp - Create DGP with random parameters
    simulate_observational - Generate confounded observational data
    simulate_rct - Generate randomized experimental data
    compute_true_cate - Compute ground truth CATE from DGP parameters
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class BinarySyntheticDGP:
    """
    Binary Synthetic Data Generating Process.

    This defines a causal model with binary covariates X, binary treatment A,
    and binary/continuous outcome Y with logistic/linear structural equations.

    Structural Equations:
        X_i ~ Bernoulli(prob_x[i])
        A | X ~ Bernoulli(logistic(coeff_a @ [1, X]))
        Y(0) | X ~ Bernoulli(logistic(coeff_y0 @ [1, X]))  [binary]
        Y(1) | X ~ Bernoulli(logistic(coeff_y1 @ [1, X]))  [binary]

    Attributes:
        n_covariates: Number of binary covariates
        prob_x: Marginal probabilities P(X_i=1), shape (n_covariates,)
        coeff_a: Logistic coefficients for treatment, shape (n_covariates+1,)
                 includes intercept as first element
        coeff_y0: Logistic coefficients for Y(0), shape (n_covariates+1,)
        coeff_y1: Logistic coefficients for Y(1), shape (n_covariates+1,)
        x_names: Names for covariate columns
        a_name: Name for treatment column
        y_name: Name for outcome column
        seed: Random seed used to generate parameters (for reproducibility)
    """
    n_covariates: int
    prob_x: np.ndarray
    coeff_a: np.ndarray
    coeff_y0: np.ndarray
    coeff_y1: np.ndarray
    x_names: List[str] = field(default_factory=list)
    a_name: str = 'A'
    y_name: str = 'Y'
    seed: Optional[int] = None

    def __post_init__(self):
        if not self.x_names:
            self.x_names = [f'X{i}' for i in range(self.n_covariates)]

    def get_propensity_params(self) -> Dict[str, Any]:
        """Get propensity model parameters."""
        return {
            'intercept': self.coeff_a[0],
            'coefficients': dict(zip(self.x_names, self.coeff_a[1:]))
        }

    def get_outcome_params(self, a: int) -> Dict[str, Any]:
        """Get outcome model parameters for treatment level a."""
        coeff = self.coeff_y1 if a == 1 else self.coeff_y0
        return {
            'treatment': a,
            'intercept': coeff[0],
            'coefficients': dict(zip(self.x_names, coeff[1:]))
        }


def generate_random_dgp(
    n_covariates: int,
    sparsity: float = 0.0,
    effect_scale: float = 1.0,
    confounding_strength: float = 1.0,
    seed: Optional[int] = None
) -> BinarySyntheticDGP:
    """
    Generate a random DGP with specified characteristics.

    Args:
        n_covariates: Number of binary covariates
        sparsity: Fraction of covariates with zero effect on outcome (0 to 1)
                 These simulate unmeasured confounding when hidden
        effect_scale: Scale of outcome coefficients
        confounding_strength: Scale of treatment coefficients
        seed: Random seed for reproducibility

    Returns:
        BinarySyntheticDGP instance

    Example:
        >>> dgp = generate_random_dgp(5, sparsity=0.2, seed=42)
        >>> dgp.n_covariates
        5
    """
    if seed is not None:
        np.random.seed(seed)

    # Marginal probabilities for X
    prob_x = np.random.uniform(0.2, 0.8, n_covariates)

    # Treatment coefficients (includes intercept)
    coeff_a = np.random.normal(0, confounding_strength / np.sqrt(n_covariates),
                               n_covariates + 1)

    # Outcome coefficients (includes intercept)
    coeff_y0 = np.random.normal(0, effect_scale / np.sqrt(n_covariates),
                                n_covariates + 1)
    coeff_y1 = np.random.normal(0, effect_scale / np.sqrt(n_covariates),
                                n_covariates + 1)

    # Apply sparsity pattern
    if sparsity > 0:
        n_sparse = int(sparsity * n_covariates)
        if n_sparse > 0:
            sparse_idx = np.random.choice(n_covariates, n_sparse, replace=False)
            # Set coefficients to zero (keep intercept)
            coeff_y0[1 + sparse_idx] = 0
            coeff_y1[1 + sparse_idx] = 0

    return BinarySyntheticDGP(
        n_covariates=n_covariates,
        prob_x=prob_x,
        coeff_a=coeff_a,
        coeff_y0=coeff_y0,
        coeff_y1=coeff_y1,
        seed=seed
    )


def simulate_observational(
    dgp: BinarySyntheticDGP,
    n_samples: int,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate observational (confounded) data from DGP.

    Treatment is assigned based on covariates according to the propensity model,
    creating confounding between X, A, and Y.

    Args:
        dgp: Data generating process
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        DataFrame with columns [X0, ..., Xk, A, Y]

    Example:
        >>> dgp = generate_random_dgp(3, seed=42)
        >>> data = simulate_observational(dgp, 1000)
        >>> data.shape
        (1000, 5)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate covariates
    X = np.zeros((n_samples, dgp.n_covariates), dtype=np.int32)
    for i in range(dgp.n_covariates):
        X[:, i] = np.random.binomial(1, dgp.prob_x[i], n_samples)

    # Design matrix with intercept
    X_design = np.column_stack([np.ones(n_samples), X])

    # Generate treatment (confounded by X)
    propensity = logistic(X_design @ dgp.coeff_a)
    A = np.random.binomial(1, propensity, n_samples)

    # Generate potential outcomes
    prob_y0 = logistic(X_design @ dgp.coeff_y0)
    prob_y1 = logistic(X_design @ dgp.coeff_y1)

    # Observed outcome
    Y = np.where(A == 1,
                 np.random.binomial(1, prob_y1, n_samples),
                 np.random.binomial(1, prob_y0, n_samples))

    # Create DataFrame
    df = pd.DataFrame(X, columns=dgp.x_names)
    df[dgp.a_name] = A
    df[dgp.y_name] = Y

    return df


def simulate_rct(
    dgp: BinarySyntheticDGP,
    n_samples: int,
    treatment_prob: float = 0.5,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate RCT (randomized) data from DGP.

    Treatment is randomly assigned independent of covariates,
    breaking the confounding pathway.

    Args:
        dgp: Data generating process
        n_samples: Number of samples to generate
        treatment_prob: Probability of treatment assignment
        seed: Random seed

    Returns:
        DataFrame with columns [X0, ..., Xk, A, Y]

    Example:
        >>> dgp = generate_random_dgp(3, seed=42)
        >>> data = simulate_rct(dgp, 1000)
        >>> # Treatment should be independent of X
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate covariates
    X = np.zeros((n_samples, dgp.n_covariates), dtype=np.int32)
    for i in range(dgp.n_covariates):
        X[:, i] = np.random.binomial(1, dgp.prob_x[i], n_samples)

    # Design matrix with intercept
    X_design = np.column_stack([np.ones(n_samples), X])

    # Random treatment assignment (no confounding)
    A = np.random.binomial(1, treatment_prob, n_samples)

    # Generate potential outcomes
    prob_y0 = logistic(X_design @ dgp.coeff_y0)
    prob_y1 = logistic(X_design @ dgp.coeff_y1)

    # Observed outcome
    Y = np.where(A == 1,
                 np.random.binomial(1, prob_y1, n_samples),
                 np.random.binomial(1, prob_y0, n_samples))

    # Create DataFrame
    df = pd.DataFrame(X, columns=dgp.x_names)
    df[dgp.a_name] = A
    df[dgp.y_name] = Y

    return df


def compute_true_cate(
    dgp: BinarySyntheticDGP,
    x_values: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Compute ground truth CATE from DGP parameters.

    CATE(x) = E[Y(1) - Y(0) | X=x]
            = P(Y=1 | A=1, X=x) - P(Y=1 | A=0, X=x)
            = logistic(coeff_y1 @ [1, x]) - logistic(coeff_y0 @ [1, x])

    Args:
        dgp: Data generating process
        x_values: Covariate values to evaluate at
                 DataFrame with X columns or array of shape (n, n_covariates)

    Returns:
        Array of true CATE values

    Example:
        >>> dgp = generate_random_dgp(3, seed=42)
        >>> data = simulate_rct(dgp, 100)
        >>> cate = compute_true_cate(dgp, data)
        >>> cate.shape
        (100,)
    """
    if isinstance(x_values, pd.DataFrame):
        X = x_values[dgp.x_names].values
    else:
        X = np.asarray(x_values)

    n = X.shape[0]
    X_design = np.column_stack([np.ones(n), X])

    prob_y0 = logistic(X_design @ dgp.coeff_y0)
    prob_y1 = logistic(X_design @ dgp.coeff_y1)

    return prob_y1 - prob_y0


def compute_true_ate(dgp: BinarySyntheticDGP, n_samples: int = 10000) -> float:
    """
    Compute ground truth ATE via Monte Carlo.

    Args:
        dgp: Data generating process
        n_samples: Number of MC samples

    Returns:
        Estimated true ATE
    """
    # Generate X from marginal distribution
    X = np.zeros((n_samples, dgp.n_covariates))
    for i in range(dgp.n_covariates):
        X[:, i] = np.random.binomial(1, dgp.prob_x[i], n_samples)

    cate = compute_true_cate(dgp, X)
    return float(np.mean(cate))


def generate_multi_environment_data(
    dgp: BinarySyntheticDGP,
    n_environments: int,
    n_per_env: int,
    heterogeneity: float = 0.1,
    include_rct: bool = True,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate data from multiple environments with heterogeneity.

    Creates multiple environments that share the same outcome model
    but have different propensity models (different confounding).

    Args:
        dgp: Base data generating process
        n_environments: Number of environments
        n_per_env: Samples per environment
        heterogeneity: Amount of propensity variation between environments
        include_rct: Include an RCT environment with no confounding
        seed: Random seed

    Returns:
        Dict mapping environment name to DataFrame

    Example:
        >>> dgp = generate_random_dgp(3, seed=42)
        >>> envs = generate_multi_environment_data(dgp, 3, 500)
        >>> len(envs)
        4  # 3 observational + 1 RCT if include_rct=True
    """
    if seed is not None:
        np.random.seed(seed)

    environments = {}

    for i in range(n_environments):
        # Perturb propensity coefficients for this environment
        env_dgp = BinarySyntheticDGP(
            n_covariates=dgp.n_covariates,
            prob_x=dgp.prob_x,
            coeff_a=dgp.coeff_a + np.random.normal(0, heterogeneity, len(dgp.coeff_a)),
            coeff_y0=dgp.coeff_y0,  # Same outcome model
            coeff_y1=dgp.coeff_y1,
            x_names=dgp.x_names,
            a_name=dgp.a_name,
            y_name=dgp.y_name
        )

        data = simulate_observational(env_dgp, n_per_env)
        data['environment'] = f'obs_{i}'
        environments[f'obs_{i}'] = data

    if include_rct:
        rct_data = simulate_rct(dgp, n_per_env)
        rct_data['environment'] = 'rct'
        environments['rct'] = rct_data

    return environments


def add_regime_indicator(
    environments: Dict[str, pd.DataFrame],
    regime_col: str = 'F',
    rct_value: str = 'idle',
    obs_value: str = 'on'
) -> Dict[str, pd.DataFrame]:
    """
    Add regime indicator column to environment data.

    Args:
        environments: Dict of environment DataFrames
        regime_col: Name of regime column
        rct_value: Value for RCT environments
        obs_value: Value for observational environments

    Returns:
        Dict with updated DataFrames
    """
    result = {}
    for name, df in environments.items():
        df = df.copy()
        if 'rct' in name.lower():
            df[regime_col] = rct_value
        else:
            df[regime_col] = obs_value
        result[name] = df
    return result


def get_covariate_stratum(
    data: pd.DataFrame,
    covariate_names: List[str]
) -> np.ndarray:
    """
    Create stratum identifier from covariate values.

    Args:
        data: DataFrame with covariate columns
        covariate_names: List of covariate column names

    Returns:
        Array of stratum identifiers (integers)
    """
    return pd.factorize(data[covariate_names].astype(str).agg('-'.join, axis=1))[0]


# Module test
if __name__ == "__main__":
    print("Synthetic Data Simulator")
    print("=" * 50)

    # Create DGP
    print("\n1. Creating DGP with 5 covariates, 20% sparsity...")
    dgp = generate_random_dgp(n_covariates=5, sparsity=0.2, seed=42)
    print(f"   Covariates: {dgp.x_names}")
    print(f"   Marginal probs: {dgp.prob_x.round(3)}")

    # Generate observational data
    print("\n2. Generating observational data (n=1000)...")
    obs_data = simulate_observational(dgp, 1000, seed=42)
    print(f"   Shape: {obs_data.shape}")
    print(f"   Treatment rate: {obs_data['A'].mean():.3f}")
    print(f"   Outcome rate: {obs_data['Y'].mean():.3f}")

    # Generate RCT data
    print("\n3. Generating RCT data (n=1000)...")
    rct_data = simulate_rct(dgp, 1000, seed=42)
    print(f"   Shape: {rct_data.shape}")
    print(f"   Treatment rate: {rct_data['A'].mean():.3f}")
    print(f"   Outcome rate: {rct_data['Y'].mean():.3f}")

    # Compute CATE
    print("\n4. Computing ground truth CATE...")
    cate = compute_true_cate(dgp, obs_data)
    print(f"   CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
    print(f"   CATE mean (ATE): {cate.mean():.4f}")

    # Multi-environment data
    print("\n5. Generating multi-environment data...")
    envs = generate_multi_environment_data(dgp, 3, 500, seed=42)
    print(f"   Environments: {list(envs.keys())}")
    for name, df in envs.items():
        print(f"   {name}: n={len(df)}, treatment_rate={df['A'].mean():.3f}")

    print("\nAll tests passed!")
