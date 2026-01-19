"""
L1-Regularized Regression CI Testing

This module implements conditional independence testing using L1-regularized
(Lasso) regression as an alternative to CMI-based testing.

The key idea: If X ⊥ Y | Z, then X should have zero coefficient when
predicting Y from (X, Z) with L1 regularization.

Advantages over CMI:
- Faster for large datasets (no permutation testing needed)
- Can handle high-dimensional conditioning sets
- Provides coefficient estimates for interpretation

Based on Ricardo's regression_indep implementation from method.ipynb.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings


class L1RegressionCIEngine:
    """
    Conditional independence testing using L1-regularized regression.

    Tests X ⊥ Y | Z by fitting Y ~ X + Z with L1 penalty and checking
    if coefficient for X is zero (or near-zero).

    This provides the same interface as CITestEngine for compatibility.

    Example:
        engine = L1RegressionCIEngine(alpha=0.1)
        result = engine.test_conditional_independence(df, 'X', 'Y', ['Z1', 'Z2'])
        print(f"Reject independence: {result['reject_independence']}")
    """

    def __init__(
        self,
        alpha: float = 0.1,
        test_alpha: float = 0.05,
        coef_threshold: float = 0.01,
        use_cv: bool = False,
        max_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize L1 regression CI engine.

        Args:
            alpha: L1 regularization strength (higher = more sparse)
            test_alpha: Significance level for independence tests
            coef_threshold: Threshold below which coefficient is considered zero
            use_cv: Use cross-validation to select alpha
            max_iter: Maximum iterations for solver
            random_state: Random state for reproducibility
        """
        self.alpha = alpha
        self.test_alpha = test_alpha
        self.coef_threshold = coef_threshold
        self.use_cv = use_cv
        self.max_iter = max_iter
        self.random_state = random_state

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Prepare data for regression, handling categorical variables.

        Returns:
            X: Feature matrix
            y: Target vector
            is_binary: Whether target is binary
        """
        y = data[target].values

        # Check if target is binary
        unique_y = np.unique(y)
        is_binary = len(unique_y) == 2

        # Prepare features
        X_parts = []
        for col in features:
            vals = data[col].values
            unique_vals = np.unique(vals)

            if len(unique_vals) <= 10:  # Treat as categorical
                # One-hot encode (drop first to avoid multicollinearity)
                for i, val in enumerate(unique_vals[1:], 1):
                    X_parts.append((vals == val).astype(float).reshape(-1, 1))
            else:  # Treat as continuous
                X_parts.append(StandardScaler().fit_transform(vals.reshape(-1, 1)))

        if X_parts:
            X = np.hstack(X_parts)
        else:
            X = np.zeros((len(data), 1))

        return X, y, is_binary

    def _fit_l1_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_binary: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Fit L1-regularized model and return coefficients.

        Returns:
            coefficients: Array of feature coefficients
            intercept: Model intercept
        """
        if is_binary:
            if self.use_cv:
                # Use cross-validation to select C (inverse of alpha)
                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    C=1.0 / self.alpha  # C is inverse regularization
                )
            else:
                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=1.0 / self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
        else:
            if self.use_cv:
                model = LassoCV(
                    cv=5,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
            else:
                model = Lasso(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        return model.coef_.flatten(), model.intercept_

    def test_conditional_independence(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str],
        return_coefficients: bool = False
    ) -> Dict[str, Any]:
        """
        Test X ⊥ Y | conditioning_set using L1 regression.

        Fits Y ~ X + Z with L1 penalty and checks if X coefficient is zero.

        Args:
            data: DataFrame with all variables
            x: First variable name (test if this is independent)
            y: Second variable name (outcome)
            conditioning_set: List of conditioning variable names
            return_coefficients: Whether to return all coefficients

        Returns:
            {
                'coefficient': float,         # Coefficient of x
                'abs_coefficient': float,     # |coefficient|
                'reject_independence': bool,  # |coef| > threshold
                'threshold': float,           # Threshold used
                'n_features': int,            # Number of features in model
                'coefficients': array         # All coefficients (if requested)
            }
        """
        # Features: x first, then conditioning set
        features = [x] + conditioning_set

        X, y_arr, is_binary = self._prepare_data(data, y, features)

        # Count features from x (first variable)
        x_unique = len(np.unique(data[x].values))
        n_x_features = max(1, x_unique - 1) if x_unique <= 10 else 1

        # Fit model
        coefficients, intercept = self._fit_l1_model(X, y_arr, is_binary)

        # Extract coefficient for x (may be multiple if categorical)
        x_coefs = coefficients[:n_x_features]
        max_abs_coef = float(np.max(np.abs(x_coefs)))

        # Test: reject independence if coefficient is non-zero
        reject = max_abs_coef > self.coef_threshold

        result = {
            'coefficient': float(x_coefs[0]) if len(x_coefs) == 1 else float(np.mean(x_coefs)),
            'abs_coefficient': max_abs_coef,
            'reject_independence': reject,
            'threshold': self.coef_threshold,
            'n_features': X.shape[1],
            # For compatibility with CMI interface
            'cmi': max_abs_coef,  # Use coefficient as proxy for CMI
            'p_value': 1.0 - max_abs_coef if max_abs_coef < 1 else 0.0,  # Pseudo p-value
        }

        if return_coefficients:
            result['coefficients'] = coefficients

        return result

    def compute_coefficient_only(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str]
    ) -> float:
        """
        Compute coefficient without full test (faster).

        Returns absolute coefficient value.
        """
        result = self.test_conditional_independence(data, x, y, conditioning_set)
        return result['abs_coefficient']

    def score_ehs_criteria(
        self,
        data: pd.DataFrame,
        z_a: str,
        z_b: List[str],
        treatment: str,
        outcome: str,
        weak_instrument_threshold: float = 0.01,
        use_permutation_test: bool = True  # Ignored, for API compatibility
    ) -> Dict[str, Any]:
        """
        Score covariate z_a on EHS criteria using L1 regression.

        EHS (Entner-Hoyer-Spirtes) criteria for valid instrument:
        - Test (i):   Y ⊥ Z_a | (Z_b, X)  → should NOT reject (exogeneity)
        - Test (ii):  Y ⊥̸ Z_a | Z_b       → should reject (outcome relevance)
        - Test (i.a): X ⊥̸ Z_a | Z_b       → should reject (instrument relevance)

        Args:
            data: DataFrame with discretized variables
            z_a: Candidate instrumental covariate
            z_b: Other covariates
            treatment: Treatment column (X)
            outcome: Outcome column (Y)
            weak_instrument_threshold: Threshold for weak instrument warning

        Returns:
            Same format as CITestEngine.score_ehs_criteria for compatibility
        """
        # Test (i): Y ⊥ Z_a | (Z_b, X) — exogeneity (should NOT reject)
        test_i = self.test_conditional_independence(
            data, z_a, outcome, z_b + [treatment]
        )

        # Test (ii): Y ⊥̸ Z_a | Z_b — outcome relevance (should reject)
        test_ii = self.test_conditional_independence(
            data, z_a, outcome, z_b
        )

        # Test (i.a): X ⊥̸ Z_a | Z_b — instrument relevance (should reject)
        test_ia = self.test_conditional_independence(
            data, treatment, z_a, z_b
        )

        passes_ehs = (not test_i['reject_independence']) and test_ii['reject_independence']
        passes_full_ehs = passes_ehs and test_ia['reject_independence']

        # Score: high test_ii coefficient, low test_i coefficient
        score = test_ii['abs_coefficient'] - test_i['abs_coefficient']

        weak_instrument_warning = (test_ia['abs_coefficient'] < weak_instrument_threshold) and (score > 0)

        return {
            'z_a': z_a,
            'test_i_cmi': test_i['abs_coefficient'],
            'test_i_pvalue': test_i['p_value'],
            'test_i_reject': test_i['reject_independence'],
            'test_ii_cmi': test_ii['abs_coefficient'],
            'test_ii_pvalue': test_ii['p_value'],
            'test_ii_reject': test_ii['reject_independence'],
            'test_ia_cmi': test_ia['abs_coefficient'],
            'test_ia_pvalue': test_ia['p_value'],
            'test_ia_reject': test_ia['reject_independence'],
            'passes_ehs': passes_ehs,
            'passes_full_ehs': passes_full_ehs,
            'weak_instrument_warning': weak_instrument_warning,
            'score': score
        }


def create_ci_engine(
    method: str = 'cmi',
    **kwargs
) -> Any:
    """
    Factory function to create CI testing engine.

    Args:
        method: 'cmi' for CMI-based, 'l1' for L1-regression based
        **kwargs: Arguments passed to engine constructor

    Returns:
        CITestEngine or L1RegressionCIEngine instance

    Example:
        engine = create_ci_engine('l1', alpha=0.1)
        engine = create_ci_engine('cmi', n_permutations=1000)
    """
    if method == 'cmi':
        from .ci_tests import CITestEngine
        return CITestEngine(**kwargs)
    elif method == 'l1':
        return L1RegressionCIEngine(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cmi' or 'l1'.")


# Module test
if __name__ == "__main__":
    print("L1 Regression CI Testing")
    print("=" * 50)

    # Create test data with known structure
    np.random.seed(42)
    n = 500

    # Z -> X -> Y (Z is valid instrument for X->Y)
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)  # Z affects X
    Y = np.random.binomial(1, 0.3 + 0.3 * X, n)  # X affects Y, Z does not directly

    df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

    engine = L1RegressionCIEngine(alpha=0.1, coef_threshold=0.05)

    print("\n1. Testing Z ⊥ Y | X (should NOT reject - Z is valid IV):")
    result = engine.test_conditional_independence(df, 'Z', 'Y', ['X'])
    print(f"   Coefficient: {result['abs_coefficient']:.4f}")
    print(f"   Reject: {result['reject_independence']}")

    print("\n2. Testing Z ⊥ Y (should reject - Z affects Y through X):")
    result = engine.test_conditional_independence(df, 'Z', 'Y', [])
    print(f"   Coefficient: {result['abs_coefficient']:.4f}")
    print(f"   Reject: {result['reject_independence']}")

    print("\n3. Testing X ⊥ Z (should reject - Z affects X):")
    result = engine.test_conditional_independence(df, 'X', 'Z', [])
    print(f"   Coefficient: {result['abs_coefficient']:.4f}")
    print(f"   Reject: {result['reject_independence']}")

    print("\n4. EHS Scoring for Z as instrument:")
    score = engine.score_ehs_criteria(df, 'Z', [], 'X', 'Y')
    print(f"   Test (i) - exogeneity: reject={score['test_i_reject']}")
    print(f"   Test (ii) - outcome relevance: reject={score['test_ii_reject']}")
    print(f"   Test (i.a) - instrument relevance: reject={score['test_ia_reject']}")
    print(f"   Passes EHS: {score['passes_ehs']}")
    print(f"   Passes Full EHS: {score['passes_full_ehs']}")
    print(f"   Score: {score['score']:.4f}")

    print("\nAll tests completed!")
