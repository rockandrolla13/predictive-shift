"""
Conditional Independence Testing using CMI

This module implements conditional independence testing using Conditional
Mutual Information (CMI) with permutation testing for p-values.

Based on David's CMI implementation with Laplacian smoothing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


# =============================================================================
# DAVID'S CMI FUNCTIONS (COPY EXACTLY)
# =============================================================================

def compute_cmi(
    Y: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    alpha: float = 1.0
) -> float:
    """
    Compute conditional mutual information I(Y; Z | W) with Laplacian smoothing.

    Parameters:
    -----------
    Y : np.ndarray
        Binary variable (0 or 1)
    Z : np.ndarray
        Categorical variable
    W : np.ndarray
        Categorical variable (conditioning variable)
    alpha : float
        Laplacian pseudocount parameter (default: 1.0)

    Returns:
    --------
    float
        Conditional mutual information I(Y; Z | W)
    """
    n = len(Y)
    y_vals = np.unique(Y)
    z_vals = np.unique(Z)
    w_vals = np.unique(W)
    n_y = len(y_vals)
    n_z = len(z_vals)
    n_w = len(w_vals)

    # Compute P(W=w) with smoothing
    counts_w = np.zeros(n_w)
    for k, w in enumerate(w_vals):
        counts_w[k] = np.sum(W == w) + alpha
    p_w_vals = counts_w / (n + alpha * n_w)

    cmi = 0.0

    for k, w in enumerate(w_vals):
        mask_w = (W == w)
        n_w_count = np.sum(mask_w)
        p_w = p_w_vals[k]

        # P(Y, Z | W=w) with smoothing
        counts_yz_given_w = np.zeros((n_y, n_z))
        for i, y in enumerate(y_vals):
            for j, z in enumerate(z_vals):
                mask = mask_w & (Y == y) & (Z == z)
                counts_yz_given_w[i, j] = np.sum(mask) + alpha

        total_count_w = n_w_count + alpha * n_y * n_z
        p_yz_given_w = counts_yz_given_w / total_count_w

        p_y_given_w = np.sum(p_yz_given_w, axis=1)
        p_z_given_w = np.sum(p_yz_given_w, axis=0)

        mi_given_w = 0.0
        for i in range(n_y):
            for j in range(n_z):
                mi_given_w += p_yz_given_w[i, j] * np.log2(
                    p_yz_given_w[i, j] / (p_y_given_w[i] * p_z_given_w[j])
                )

        cmi += p_w * mi_given_w

    return cmi


def permutation_test_cmi(
    Y: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    alpha: float = 1.0,
    n_permutations: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float, np.ndarray]:
    """
    Perform permutation test for conditional mutual information.

    Tests null hypothesis: I(Y; Z | W) = 0

    Parameters:
    -----------
    Y : np.ndarray
        Binary variable (0 or 1)
    Z : np.ndarray
        Categorical variable
    W : np.ndarray
        Categorical variable (conditioning variable)
    alpha : float
        Laplacian pseudocount parameter (default: 1.0)
    n_permutations : int
        Number of permutations (default: 1000)
    random_seed : int
        Random seed for reproducibility (default: None)

    Returns:
    --------
    observed_cmi : float
        Observed conditional mutual information
    p_value : float
        Permutation p-value
    null_distribution : np.ndarray
        CMI values under null hypothesis
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    observed_cmi = compute_cmi(Y, Z, W, alpha)
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        Z_permuted = Z.copy()
        for w in np.unique(W):
            mask = W == w
            indices = np.where(mask)[0]
            Z_permuted[mask] = Z[indices[np.random.permutation(len(indices))]]

        null_distribution[i] = compute_cmi(Y, Z_permuted, W, alpha)

    p_value = (np.sum(null_distribution >= observed_cmi) + 1) / (n_permutations + 1)

    return observed_cmi, p_value, null_distribution


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def combine_conditioning_vars(
    data: pd.DataFrame,
    cols: List[str]
) -> np.ndarray:
    """
    Combine multiple categorical columns into single conditioning variable.

    Args:
        data: DataFrame with categorical columns
        cols: Column names to combine

    Returns:
        Integer array mapping each unique combination to 0, 1, 2, ...
    """
    if len(cols) == 0:
        return np.zeros(len(data), dtype=int)
    if len(cols) == 1:
        return pd.factorize(data[cols[0]])[0]

    combined = data[cols].astype(str).agg('-'.join, axis=1)
    return pd.factorize(combined)[0]


# =============================================================================
# CI TEST ENGINE CLASS
# =============================================================================

class CITestEngine:
    """
    Conditional independence testing using Conditional Mutual Information.

    Uses permutation testing to obtain p-values for H0: X ⊥ Y | Z.

    Example:
        engine = CITestEngine(n_permutations=1000)
        result = engine.test_conditional_independence(df, 'X', 'Y', ['Z1', 'Z2'])
        print(f"p-value: {result['p_value']:.4f}")
    """

    def __init__(
        self,
        smoothing_alpha: float = 1.0,
        test_alpha: float = 0.05,
        n_permutations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            smoothing_alpha: Laplacian pseudocount for CMI computation
            test_alpha: Significance level for independence tests
            n_permutations: Number of permutations for p-value
            random_seed: Random seed for reproducibility
        """
        self.smoothing_alpha = smoothing_alpha
        self.test_alpha = test_alpha
        self.n_permutations = n_permutations
        self.random_seed = random_seed

    def test_conditional_independence(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str],
        return_null_dist: bool = False
    ) -> Dict[str, Any]:
        """
        Test X ⊥ Y | conditioning_set using CMI permutation test.

        Args:
            data: DataFrame with all variables (must be categorical/discrete)
            x: First variable name
            y: Second variable name
            conditioning_set: List of conditioning variable names
            return_null_dist: Whether to include null distribution in result

        Returns:
            {
                'cmi': float,                 # I(X; Y | Z)
                'p_value': float,             # Permutation p-value
                'reject_independence': bool,  # p_value < test_alpha
                'n_permutations': int,
                'null_mean': float,
                'null_std': float,
                'null_distribution': array    # Only if return_null_dist=True
            }
        """
        Y_arr = data[y].values
        Z_arr = data[x].values
        W_arr = combine_conditioning_vars(data, conditioning_set)

        observed_cmi, p_value, null_dist = permutation_test_cmi(
            Y_arr, Z_arr, W_arr,
            alpha=self.smoothing_alpha,
            n_permutations=self.n_permutations,
            random_seed=self.random_seed
        )

        result = {
            'cmi': observed_cmi,
            'p_value': p_value,
            'reject_independence': p_value < self.test_alpha,
            'n_permutations': self.n_permutations,
            'null_mean': float(np.mean(null_dist)),
            'null_std': float(np.std(null_dist))
        }

        if return_null_dist:
            result['null_distribution'] = null_dist

        return result

    def compute_cmi_only(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str]
    ) -> float:
        """Compute CMI without permutation test (faster, no p-value)."""
        Y_arr = data[y].values
        Z_arr = data[x].values
        W_arr = combine_conditioning_vars(data, conditioning_set)
        return compute_cmi(Y_arr, Z_arr, W_arr, alpha=self.smoothing_alpha)

    def score_ehs_criteria(
        self,
        data: pd.DataFrame,
        z_a: str,
        z_b: List[str],
        treatment: str,
        outcome: str,
        use_permutation_test: bool = True
    ) -> Dict[str, Any]:
        """
        Score covariate z_a on modified EHS criteria.

        EHS (Entner-Hoyer-Spirtes) criteria for valid instrument:
        - Test (i):  Y ⊥ Z_a | (Z_b, X)  → should NOT reject
        - Test (ii): Y ⊥̸ Z_a | Z_b       → should reject

        Args:
            data: DataFrame with discretized variables
            z_a: Candidate instrumental covariate
            z_b: Other covariates
            treatment: Treatment column (X)
            outcome: Outcome column (Y)
            use_permutation_test: Use full test (True) or CMI only (False)

        Returns:
            {
                'z_a': str,
                'test_i_cmi': float,
                'test_i_pvalue': float or None,
                'test_i_reject': bool,
                'test_ii_cmi': float,
                'test_ii_pvalue': float or None,
                'test_ii_reject': bool,
                'passes_ehs': bool,
                'score': float
            }
        """
        if use_permutation_test:
            test_i = self.test_conditional_independence(
                data, z_a, outcome, z_b + [treatment]
            )
            test_ii = self.test_conditional_independence(
                data, z_a, outcome, z_b
            )

            passes_ehs = (not test_i['reject_independence']) and test_ii['reject_independence']
            # Use CMI-based scoring (sample-size invariant) instead of p-values
            score = test_ii['cmi'] - test_i['cmi']

            return {
                'z_a': z_a,
                'test_i_cmi': test_i['cmi'],
                'test_i_pvalue': test_i['p_value'],
                'test_i_reject': test_i['reject_independence'],
                'test_ii_cmi': test_ii['cmi'],
                'test_ii_pvalue': test_ii['p_value'],
                'test_ii_reject': test_ii['reject_independence'],
                'passes_ehs': passes_ehs,
                'score': score
            }
        else:
            test_i_cmi = self.compute_cmi_only(data, z_a, outcome, z_b + [treatment])
            test_ii_cmi = self.compute_cmi_only(data, z_a, outcome, z_b)

            cmi_threshold = 0.01
            test_i_reject = test_i_cmi > cmi_threshold
            test_ii_reject = test_ii_cmi > cmi_threshold

            passes_ehs = (not test_i_reject) and test_ii_reject
            score = test_ii_cmi - test_i_cmi

            return {
                'z_a': z_a,
                'test_i_cmi': test_i_cmi,
                'test_i_pvalue': None,
                'test_i_reject': test_i_reject,
                'test_ii_cmi': test_ii_cmi,
                'test_ii_pvalue': None,
                'test_ii_reject': test_ii_reject,
                'passes_ehs': passes_ehs,
                'score': score
            }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Reproduce David's example
    np.random.seed(42)
    n = 500

    W = np.random.choice([0, 1, 2, 3], size=n)
    Z = np.random.choice([0, 1, 2, 3, 4], size=n)

    logits = -1.0 + 0.1 * W + 0.1 * Z
    probs = 1 / (1 + np.exp(-logits))
    Y = (np.random.random(n) < probs).astype(int)

    df = pd.DataFrame({'Y': Y, 'Z': Z, 'W': W})

    engine = CITestEngine(n_permutations=1000, random_seed=42)

    print("Test Y ⊥ Z | W (expect rejection - Z affects Y):")
    result = engine.test_conditional_independence(df, 'Z', 'Y', ['W'])
    print(f"  CMI: {result['cmi']:.6f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Reject: {result['reject_independence']}")

    print("\nTest Y ⊥ W | Z (expect rejection - W affects Y):")
    result2 = engine.test_conditional_independence(df, 'W', 'Y', ['Z'])
    print(f"  CMI: {result2['cmi']:.6f}")
    print(f"  p-value: {result2['p_value']:.4f}")
    print(f"  Reject: {result2['reject_independence']}")
