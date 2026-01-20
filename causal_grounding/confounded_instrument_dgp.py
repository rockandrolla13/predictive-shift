"""
Confounded Instrument Data Generating Process

This module provides a DGP that correctly implements the causal structure
assumed in the EHS (Entner-Hoyer-Spirtes) framework:

    U (unobserved)
   / \\
  X → Y
  ↑
  Z (instrument)

Where:
- Z → X: Instrument affects treatment (relevance)
- U → X, U → Y: Unobserved confounding
- Z ⊥ U: Instrument is exogenous
- Z ⊥ Y | (X, U): Exclusion restriction (no direct effect on Y)

This structure allows testing:
- EHS test (i): Y ⊥ Z | X — should pass for valid instrument
- EHS test (ii): Y ⊥̸ Z — should reject (Z affects Y through X)
- EHS test (i.a): X ⊥̸ Z — should reject (Z affects X directly)

Key Classes:
    ConfoundedInstrumentDGP - DGP with explicit instrument and confounder

Key Functions:
    generate_confounded_instrument_data - Generate data from the DGP
    compute_ground_truth_effects - Compute true causal effects
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class ConfoundedInstrumentDGP:
    """
    Data Generating Process with explicit instrument and unobserved confounder.

    Causal Structure:
        U ~ Bernoulli(prob_u)              # Unobserved confounder
        Z ~ Bernoulli(prob_z)              # Instrument (independent of U)
        W ~ Bernoulli(prob_w)              # Observed confounder (optional)

        # Treatment model: X depends on Z, U, and optionally W
        P(X=1 | Z, U, W) = logistic(alpha_0 + alpha_z*Z + alpha_u*U + alpha_w*W)

        # Outcome model: Y depends on X, U, and optionally W, but NOT Z (unless exclusion violated)
        P(Y=1 | X, U, W, Z) = logistic(beta_0 + beta_x*X + beta_u*U + beta_w*W + beta_z*Z)

        When beta_z = 0, Z is a valid instrument (exclusion restriction holds).
        When beta_z ≠ 0, Z has a direct effect on Y (invalid instrument).

    Attributes:
        alpha_0: Intercept for treatment model
        alpha_z: Effect of instrument Z on treatment X (instrument strength)
        alpha_u: Effect of unobserved confounder U on treatment X
        alpha_w: Effect of observed confounder W on treatment X (if W included)
        beta_0: Intercept for outcome model
        beta_x: Effect of treatment X on outcome Y (causal effect)
        beta_u: Effect of unobserved confounder U on outcome Y
        beta_w: Effect of observed confounder W on outcome Y (if W included)
        beta_z: Direct effect of Z on Y (0 for valid instrument, >0 for invalid)
        prob_u: P(U=1) for unobserved confounder
        prob_z: P(Z=1) for instrument
        prob_w: P(W=1) for observed confounder (if included)
        include_observed_confounder: Whether to include W in the model
    """
    # Treatment model parameters
    alpha_0: float = 0.0  # Intercept
    alpha_z: float = 1.0  # Instrument effect on treatment
    alpha_u: float = 1.0  # Unobserved confounder effect on treatment
    alpha_w: float = 0.5  # Observed confounder effect on treatment

    # Outcome model parameters
    beta_0: float = 0.0   # Intercept
    beta_x: float = 1.0   # Treatment effect on outcome (causal effect)
    beta_u: float = 1.0   # Unobserved confounder effect on outcome
    beta_w: float = 0.5   # Observed confounder effect on outcome
    beta_z: float = 0.0   # Direct effect of Z on Y (0 = valid instrument)

    # Marginal probabilities
    prob_u: float = 0.5   # P(U=1)
    prob_z: float = 0.5   # P(Z=1)
    prob_w: float = 0.5   # P(W=1)

    # Structure options
    include_observed_confounder: bool = False

    # Column names
    z_name: str = 'Z'     # Instrument
    w_name: str = 'W'     # Observed confounder
    x_name: str = 'X'     # Treatment
    y_name: str = 'Y'     # Outcome
    u_name: str = 'U'     # Unobserved confounder (for ground truth only)

    def is_valid_instrument(self) -> bool:
        """Check if Z is a valid instrument (no direct effect on Y)."""
        return abs(self.beta_z) < 1e-10

    def is_weak_instrument(self, threshold: float = 0.3) -> bool:
        """Check if Z is a weak instrument."""
        return abs(self.alpha_z) < threshold

    def get_scenario_description(self) -> str:
        """Return a description of the causal scenario."""
        if self.is_valid_instrument():
            if self.is_weak_instrument():
                return "Scenario 3: Weak Instrument (Z → X weak, Z ⊥ Y|X)"
            elif self.include_observed_confounder:
                return "Scenario 4: Valid Instrument + Observed Confounder"
            else:
                return "Scenario 1: Valid Instrument (Z → X, Z ⊥ Y|X)"
        else:
            return f"Scenario 2: Invalid Instrument (Z → Y direct, beta_z={self.beta_z})"


def generate_confounded_instrument_data(
    dgp: ConfoundedInstrumentDGP,
    n_samples: int,
    seed: Optional[int] = None,
    include_unobserved: bool = False
) -> pd.DataFrame:
    """
    Generate data from the confounded instrument DGP.

    Args:
        dgp: Data generating process specification
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        include_unobserved: If True, include U in output (for ground truth analysis)

    Returns:
        DataFrame with columns [Z, W (if included), X, Y, U (if include_unobserved)]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate unobserved confounder U (independent of everything else)
    U = np.random.binomial(1, dgp.prob_u, n_samples)

    # Generate instrument Z (independent of U - this is the exogeneity assumption)
    Z = np.random.binomial(1, dgp.prob_z, n_samples)

    # Generate observed confounder W if included (independent of Z and U)
    if dgp.include_observed_confounder:
        W = np.random.binomial(1, dgp.prob_w, n_samples)
    else:
        W = np.zeros(n_samples)

    # Generate treatment X (depends on Z, U, and W)
    # P(X=1 | Z, U, W) = logistic(alpha_0 + alpha_z*Z + alpha_u*U + alpha_w*W)
    logit_x = (dgp.alpha_0 +
               dgp.alpha_z * Z +
               dgp.alpha_u * U +
               (dgp.alpha_w * W if dgp.include_observed_confounder else 0))
    prob_x = logistic(logit_x)
    X = np.random.binomial(1, prob_x, n_samples)

    # Generate outcome Y (depends on X, U, W, and possibly Z if exclusion violated)
    # P(Y=1 | X, U, W, Z) = logistic(beta_0 + beta_x*X + beta_u*U + beta_w*W + beta_z*Z)
    logit_y = (dgp.beta_0 +
               dgp.beta_x * X +
               dgp.beta_u * U +
               (dgp.beta_w * W if dgp.include_observed_confounder else 0) +
               dgp.beta_z * Z)  # This is 0 for valid instruments
    prob_y = logistic(logit_y)
    Y = np.random.binomial(1, prob_y, n_samples)

    # Create DataFrame
    df = pd.DataFrame({dgp.z_name: Z})

    if dgp.include_observed_confounder:
        df[dgp.w_name] = W

    df[dgp.x_name] = X
    df[dgp.y_name] = Y

    if include_unobserved:
        df[dgp.u_name] = U

    return df


def compute_ground_truth_effects(dgp: ConfoundedInstrumentDGP) -> Dict[str, float]:
    """
    Compute ground truth causal effects from DGP parameters.

    Returns:
        Dict with:
        - 'ate': Average treatment effect E[Y(1) - Y(0)]
        - 'causal_effect': The beta_x parameter (direct treatment effect)
        - 'confounding_bias': Bias from naive regression due to U
        - 'instrument_strength': The alpha_z parameter (Z → X effect)
        - 'exclusion_violation': The beta_z parameter (Z → Y direct effect)
    """
    # True causal effect is beta_x (coefficient of X in outcome model)
    causal_effect = dgp.beta_x

    # Monte Carlo estimation of ATE accounting for structure
    n_mc = 100000
    np.random.seed(42)  # Fixed seed for reproducibility

    U = np.random.binomial(1, dgp.prob_u, n_mc)
    Z = np.random.binomial(1, dgp.prob_z, n_mc)

    if dgp.include_observed_confounder:
        W = np.random.binomial(1, dgp.prob_w, n_mc)
    else:
        W = np.zeros(n_mc)

    # Potential outcome Y(1) - setting X=1 for everyone
    logit_y1 = (dgp.beta_0 +
                dgp.beta_x * 1 +
                dgp.beta_u * U +
                (dgp.beta_w * W if dgp.include_observed_confounder else 0) +
                dgp.beta_z * Z)
    prob_y1 = logistic(logit_y1)

    # Potential outcome Y(0) - setting X=0 for everyone
    logit_y0 = (dgp.beta_0 +
                dgp.beta_x * 0 +
                dgp.beta_u * U +
                (dgp.beta_w * W if dgp.include_observed_confounder else 0) +
                dgp.beta_z * Z)
    prob_y0 = logistic(logit_y0)

    ate = np.mean(prob_y1 - prob_y0)

    # Naive estimator bias (comparing treated vs untreated groups)
    # This is confounded because U affects both X and Y
    logit_x = (dgp.alpha_0 +
               dgp.alpha_z * Z +
               dgp.alpha_u * U +
               (dgp.alpha_w * W if dgp.include_observed_confounder else 0))
    prob_x = logistic(logit_x)
    X_obs = np.random.binomial(1, prob_x, n_mc)

    # Observed outcomes
    logit_y_obs = (dgp.beta_0 +
                   dgp.beta_x * X_obs +
                   dgp.beta_u * U +
                   (dgp.beta_w * W if dgp.include_observed_confounder else 0) +
                   dgp.beta_z * Z)
    prob_y_obs = logistic(logit_y_obs)
    Y_obs = np.random.binomial(1, prob_y_obs, n_mc)

    # Naive comparison
    naive_effect = Y_obs[X_obs == 1].mean() - Y_obs[X_obs == 0].mean()
    confounding_bias = naive_effect - ate

    return {
        'ate': ate,
        'causal_effect': causal_effect,
        'confounding_bias': confounding_bias,
        'instrument_strength': dgp.alpha_z,
        'exclusion_violation': dgp.beta_z
    }


# Pre-configured scenarios for experiments
def create_scenario_1_valid_instrument(
    alpha_z: float = 1.5,
    alpha_u: float = 1.0,
    beta_x: float = 1.5,
    beta_u: float = 1.0
) -> ConfoundedInstrumentDGP:
    """
    Scenario 1: Valid Instrument (Baseline)

    Z → X (strong), U → X, U → Y, X → Y
    Z ⊥ Y | X (exclusion holds)

    EHS Expected: Z passes all three tests → valid instrument
    """
    return ConfoundedInstrumentDGP(
        alpha_z=alpha_z,
        alpha_u=alpha_u,
        beta_x=beta_x,
        beta_u=beta_u,
        beta_z=0.0,  # No direct effect - valid instrument
        include_observed_confounder=False
    )


def create_scenario_2_exclusion_violated(
    alpha_z: float = 1.5,
    beta_z: float = 0.8,
    alpha_u: float = 1.0,
    beta_x: float = 1.5,
    beta_u: float = 1.0
) -> ConfoundedInstrumentDGP:
    """
    Scenario 2: Invalid Instrument (Direct Effect on Y)

    Z → X, Z → Y (direct!), U → X, U → Y, X → Y
    Z ⊥̸ Y | X (exclusion violated)

    EHS Expected: Z fails test (i) → invalid instrument
    """
    return ConfoundedInstrumentDGP(
        alpha_z=alpha_z,
        alpha_u=alpha_u,
        beta_x=beta_x,
        beta_u=beta_u,
        beta_z=beta_z,  # Direct effect - invalid instrument
        include_observed_confounder=False
    )


def create_scenario_3_weak_instrument(
    alpha_z: float = 0.2,
    alpha_u: float = 1.0,
    beta_x: float = 1.5,
    beta_u: float = 1.0
) -> ConfoundedInstrumentDGP:
    """
    Scenario 3: Weak Instrument

    Z → X (weak), U → X, U → Y, X → Y
    Z ⊥ Y | X (exclusion holds, but Z weakly affects X)

    EHS Expected: Z may fail test (i.a) → weak instrument warning
    """
    return ConfoundedInstrumentDGP(
        alpha_z=alpha_z,  # Weak instrument effect
        alpha_u=alpha_u,
        beta_x=beta_x,
        beta_u=beta_u,
        beta_z=0.0,  # No direct effect - valid but weak instrument
        include_observed_confounder=False
    )


def create_scenario_4_confounder_vs_instrument(
    alpha_z: float = 1.5,
    alpha_w: float = 1.2,
    alpha_u: float = 1.0,
    beta_x: float = 1.5,
    beta_u: float = 1.0,
    beta_w: float = 1.2
) -> ConfoundedInstrumentDGP:
    """
    Scenario 4: Confounder vs Instrument Discrimination

    Z → X (instrument, no direct Y effect)
    W → X, W → Y (observed confounder)
    U → X, U → Y (unobserved confounder)

    EHS Expected:
    - Z passes all tests → valid instrument
    - W fails test (i) because W → Y directly
    """
    return ConfoundedInstrumentDGP(
        alpha_z=alpha_z,
        alpha_w=alpha_w,
        alpha_u=alpha_u,
        beta_x=beta_x,
        beta_u=beta_u,
        beta_w=beta_w,  # W affects Y directly (confounder)
        beta_z=0.0,     # Z does not affect Y directly (valid instrument)
        include_observed_confounder=True
    )


# Module test
if __name__ == "__main__":
    print("Confounded Instrument DGP")
    print("=" * 60)

    print("\n1. Scenario 1: Valid Instrument")
    print("-" * 40)
    dgp1 = create_scenario_1_valid_instrument()
    data1 = generate_confounded_instrument_data(dgp1, 1000, seed=42)
    effects1 = compute_ground_truth_effects(dgp1)
    print(f"   Description: {dgp1.get_scenario_description()}")
    print(f"   Data shape: {data1.shape}")
    print(f"   Columns: {list(data1.columns)}")
    print(f"   True ATE: {effects1['ate']:.4f}")
    print(f"   Confounding bias: {effects1['confounding_bias']:.4f}")
    print(f"   Valid instrument: {dgp1.is_valid_instrument()}")

    print("\n2. Scenario 2: Invalid Instrument (Exclusion Violated)")
    print("-" * 40)
    dgp2 = create_scenario_2_exclusion_violated(beta_z=0.5)
    data2 = generate_confounded_instrument_data(dgp2, 1000, seed=42)
    effects2 = compute_ground_truth_effects(dgp2)
    print(f"   Description: {dgp2.get_scenario_description()}")
    print(f"   Direct effect of Z on Y: {dgp2.beta_z}")
    print(f"   Valid instrument: {dgp2.is_valid_instrument()}")

    print("\n3. Scenario 3: Weak Instrument")
    print("-" * 40)
    dgp3 = create_scenario_3_weak_instrument(alpha_z=0.1)
    data3 = generate_confounded_instrument_data(dgp3, 1000, seed=42)
    effects3 = compute_ground_truth_effects(dgp3)
    print(f"   Description: {dgp3.get_scenario_description()}")
    print(f"   Instrument strength (alpha_z): {dgp3.alpha_z}")
    print(f"   Weak instrument: {dgp3.is_weak_instrument()}")

    print("\n4. Scenario 4: Confounder vs Instrument")
    print("-" * 40)
    dgp4 = create_scenario_4_confounder_vs_instrument()
    data4 = generate_confounded_instrument_data(dgp4, 1000, seed=42)
    effects4 = compute_ground_truth_effects(dgp4)
    print(f"   Description: {dgp4.get_scenario_description()}")
    print(f"   Data shape: {data4.shape}")
    print(f"   Columns: {list(data4.columns)}")
    print(f"   Z is valid instrument: {dgp4.beta_z == 0}")
    print(f"   W is confounder (affects Y): {dgp4.beta_w != 0}")

    print("\n5. Verifying Causal Structure")
    print("-" * 40)
    # Generate data with U visible to verify structure
    dgp_test = create_scenario_1_valid_instrument()
    data_test = generate_confounded_instrument_data(dgp_test, 10000, seed=42, include_unobserved=True)

    # Check Z ⊥ U (instrument exogeneity)
    zu_corr = np.corrcoef(data_test['Z'], data_test['U'])[0, 1]
    print(f"   Z-U correlation (should be ~0): {zu_corr:.4f}")

    # Check Z → X (instrument relevance)
    zx_corr = np.corrcoef(data_test['Z'], data_test['X'])[0, 1]
    print(f"   Z-X correlation (should be positive): {zx_corr:.4f}")

    # Check U → X (confounding path)
    ux_corr = np.corrcoef(data_test['U'], data_test['X'])[0, 1]
    print(f"   U-X correlation (should be positive): {ux_corr:.4f}")

    # Check U → Y (confounding path)
    uy_corr = np.corrcoef(data_test['U'], data_test['Y'])[0, 1]
    print(f"   U-Y correlation (should be positive): {uy_corr:.4f}")

    print("\nAll scenarios created successfully!")
