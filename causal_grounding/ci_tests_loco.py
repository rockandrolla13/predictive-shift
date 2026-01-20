"""
LOCO (Leave-One-Covariate-Out) CI Testing

This module implements conditional independence testing using the LOCO method,
which compares predictive models with and without the variable of interest.

The key idea: If X ⊥ Y | W, then a model P(Y | X, W) should not significantly
outperform P(Y | W) on held-out data.

Method:
1. Split data into train/test
2. Fit f0: P(Y | W) - null model
3. Fit f1: P(Y | X, W) - full model
4. Compare per-sample log-losses via Wilcoxon rank-sum test
5. If f1 significantly better, reject H0 (X provides info beyond W)

Advantages over CMI:
- Handles high-dimensional conditioning sets via regularization
- Works with continuous covariates (no discretization)
- Model-agnostic (can use lasso, GBM, etc.)

References:
- Original implementation from LOCO.ipynb (Colab notebook)
"""

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Check for optional group-lasso dependency
try:
    from group_lasso import LogisticGroupLasso
    LogisticGroupLasso.LOG_LOSSES = True
    LOCO_LASSO_AVAILABLE = True
except ImportError:
    LOCO_LASSO_AVAILABLE = False


class LOCOCIEngine:
    """
    Conditional independence testing using Leave-One-Covariate-Out (LOCO).

    Tests X ⊥ Y | W by comparing predictive performance of:
    - f0: P(Y | W)     - null model (without X)
    - f1: P(Y | X, W)  - full model (with X)

    Uses Wilcoxon rank-sum test on per-sample log-losses to determine
    if including X significantly improves prediction.

    This provides the same interface as CITestEngine for compatibility.

    Example:
        engine = LOCOCIEngine(function_class='gbm')
        result = engine.test_conditional_independence(df, 'X', 'Y', ['Z1', 'Z2'])
        print(f"Reject independence: {result['reject_independence']}")
    """

    def __init__(
        self,
        function_class: str = 'gbm',
        test_prop: float = 0.3,
        test_alpha: float = 0.05,
        cv_folds: int = 5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None
    ):
        """
        Initialize LOCO CI engine.

        Args:
            function_class: Model class to use ('lasso' or 'gbm')
            test_prop: Proportion of data for testing (default 0.3)
            test_alpha: Significance level for Wilcoxon test
            cv_folds: Number of CV folds for lasso regularization selection
            n_estimators: Number of boosting stages for GBM
            learning_rate: Learning rate for GBM
            max_depth: Maximum tree depth for GBM
            random_state: Random seed for reproducibility
        """
        if function_class == 'lasso' and not LOCO_LASSO_AVAILABLE:
            raise ImportError(
                "group-lasso package required for function_class='lasso'. "
                "Install with: pip install group-lasso"
            )

        if function_class not in ('lasso', 'gbm'):
            raise ValueError(f"function_class must be 'lasso' or 'gbm', got {function_class}")

        self.function_class = function_class
        self.test_prop = test_prop
        self.test_alpha = test_alpha
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _prepare_features(
        self,
        data: pd.DataFrame,
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder]:
        """
        Prepare feature matrix with one-hot encoding.

        Returns:
            X_encoded: One-hot encoded feature matrix
            groups: Group indices for grouped lasso
            encoder: Fitted OneHotEncoder
        """
        if not features:
            # Empty feature set - return zero matrix
            return np.zeros((len(data), 1)), np.array([0]), None

        # Create DataFrame subset
        W_df = data[features].copy()

        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        W_encoded = encoder.fit_transform(W_df)

        # Create group indices for grouped lasso
        groups = []
        feature_names = encoder.get_feature_names_out()
        for col_idx, col_name in enumerate(features):
            col_features = [i for i, name in enumerate(feature_names)
                          if name.startswith(f'{col_name}_')]
            groups.extend([col_idx] * len(col_features))
        groups = np.array(groups)

        return W_encoded, groups, encoder

    def _fit_lasso_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        W_train: np.ndarray,
        W_test: np.ndarray,
        groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit grouped lasso models with CV for regularization selection."""

        reg_candidates = np.logspace(-3, 1, 20)

        # f0: P(Y | W) - null model (X omitted)
        if W_train.shape[1] > 0:
            best_reg_f0 = self._cv_select_reg(
                W_train, Y_train, groups, reg_candidates
            )
            model_f0 = LogisticGroupLasso(
                groups=groups.tolist(),
                n_iter=100,
                tol=1e-4,
                l1_reg=best_reg_f0
            )
            model_f0.fit(W_train, Y_train)
            f0_probs = model_f0.predict_proba(W_test)[:, 1]
        else:
            # No conditioning set - predict base rate
            f0_probs = np.full(len(Y_test), Y_train.mean())

        # f1: P(Y | X, W) - full model
        X_W_train = np.hstack([X_train, W_train])
        X_W_test = np.hstack([X_test, W_test])
        full_groups = np.concatenate([[0], groups + 1]) if len(groups) > 0 else np.array([0])

        best_reg_f1 = self._cv_select_reg(
            X_W_train, Y_train, full_groups, reg_candidates
        )
        model_f1 = LogisticGroupLasso(
            groups=full_groups.tolist(),
            n_iter=100,
            tol=1e-4,
            l1_reg=best_reg_f1
        )
        model_f1.fit(X_W_train, Y_train)
        f1_probs = model_f1.predict_proba(X_W_test)[:, 1]

        return f0_probs, f1_probs

    def _cv_select_reg(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        reg_candidates: np.ndarray
    ) -> float:
        """Select best regularization parameter via cross-validation."""

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for reg in reg_candidates:
            fold_losses = []

            for train_idx, val_idx in kf.split(X):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                Y_fold_train, Y_fold_val = Y[train_idx], Y[val_idx]

                model = LogisticGroupLasso(
                    groups=groups.tolist(),
                    n_iter=100,
                    tol=1e-4,
                    l1_reg=reg
                )

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_fold_train, Y_fold_train)
                    probs = model.predict_proba(X_fold_val)[:, 1]
                    probs = np.clip(probs, 1e-15, 1 - 1e-15)
                    loss = log_loss(Y_fold_val, probs)
                    fold_losses.append(loss)
                except Exception:
                    fold_losses.append(np.inf)

            cv_scores.append(np.mean(fold_losses))

        best_idx = np.argmin(cv_scores)
        return reg_candidates[best_idx]

    def _fit_gbm_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        W_train: np.ndarray,
        W_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit gradient boosting models."""

        # f0: P(Y | W) - null model
        if W_train.shape[1] > 0:
            model_f0 = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            model_f0.fit(W_train, Y_train)
            f0_probs = model_f0.predict_proba(W_test)[:, 1]
        else:
            # No conditioning set - predict base rate
            f0_probs = np.full(len(Y_test), Y_train.mean())

        # f1: P(Y | X, W) - full model
        X_W_train = np.hstack([X_train, W_train])
        X_W_test = np.hstack([X_test, W_test])

        model_f1 = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        model_f1.fit(X_W_train, Y_train)
        f1_probs = model_f1.predict_proba(X_W_test)[:, 1]

        return f0_probs, f1_probs

    def _run_loco_test(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str]
    ) -> Dict[str, Any]:
        """
        Run the core LOCO test.

        Returns raw LOCO results with losses and p-value.
        """
        # Get data as arrays
        X = data[x].values.reshape(-1, 1)
        Y = data[y].values.ravel()

        # Prepare conditioning set
        W_encoded, groups, encoder = self._prepare_features(data, conditioning_set)

        # Split data
        indices = np.arange(len(Y))

        # Check for sufficient samples per class
        unique_y, counts_y = np.unique(Y, return_counts=True)
        min_class_count = min(counts_y)

        # Need at least 2 samples per class in test set for stratified split
        if min_class_count < 3:
            # Fall back to non-stratified split
            train_idx, test_idx = train_test_split(
                indices, test_size=self.test_prop, random_state=self.random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, test_size=self.test_prop,
                random_state=self.random_state, stratify=Y
            )

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        W_train, W_test = W_encoded[train_idx], W_encoded[test_idx]

        # One-hot encode X for model fitting
        X_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_encoded = X_encoder.fit_transform(X_train)
        X_test_encoded = X_encoder.transform(X_test)

        # Fit models based on function class
        if self.function_class == 'lasso':
            f0_probs, f1_probs = self._fit_lasso_models(
                X_train_encoded, X_test_encoded, Y_train, Y_test,
                W_train, W_test, groups
            )
        else:  # gbm
            f0_probs, f1_probs = self._fit_gbm_models(
                X_train_encoded, X_test_encoded, Y_train, Y_test,
                W_train, W_test
            )

        # Compute log losses
        epsilon = 1e-15
        f0_probs = np.clip(f0_probs, epsilon, 1 - epsilon)
        f1_probs = np.clip(f1_probs, epsilon, 1 - epsilon)

        f0_losses = -Y_test * np.log(f0_probs) - (1 - Y_test) * np.log(1 - f0_probs)
        f1_losses = -Y_test * np.log(f1_probs) - (1 - Y_test) * np.log(1 - f1_probs)

        # Wilcoxon rank-sum test (one-sided: f1 should have lower loss)
        statistic, p_value = ranksums(f1_losses, f0_losses, alternative='less')

        return {
            'p_value': float(p_value),
            'test_statistic': float(statistic),
            'f0_loss': float(np.mean(f0_losses)),
            'f1_loss': float(np.mean(f1_losses)),
            'loss_reduction': float(np.mean(f0_losses) - np.mean(f1_losses)),
            'n_test': len(Y_test)
        }

    def test_conditional_independence(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Test conditional independence: X ⊥ Y | conditioning_set.

        Args:
            data: DataFrame with variables
            x: Variable to test
            y: Outcome variable
            conditioning_set: Variables to condition on

        Returns:
            {
                'cmi': float,                # Loss reduction as CMI proxy
                'p_value': float,            # Wilcoxon test p-value
                'reject_independence': bool, # p_value < test_alpha
                'f0_loss': float,           # Null model loss
                'f1_loss': float,           # Full model loss
                'test_statistic': float,    # Wilcoxon statistic
                'loss_reduction': float,    # f0_loss - f1_loss
                'n_test': int               # Test set size
            }
        """
        result = self._run_loco_test(data, x, y, conditioning_set)

        # Compute CMI proxy (loss reduction, clamped to non-negative)
        cmi_proxy = max(0.0, result['loss_reduction'])

        return {
            'cmi': cmi_proxy,
            'p_value': result['p_value'],
            'reject_independence': result['p_value'] < self.test_alpha,
            'f0_loss': result['f0_loss'],
            'f1_loss': result['f1_loss'],
            'test_statistic': result['test_statistic'],
            'loss_reduction': result['loss_reduction'],
            'n_test': result['n_test']
        }

    def compute_cmi_only(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set: List[str]
    ) -> float:
        """
        Compute only the CMI proxy (loss reduction) without full test.

        For LOCO, this still requires model fitting, so it's not faster.
        Provided for API compatibility.
        """
        result = self._run_loco_test(data, x, y, conditioning_set)
        return max(0.0, result['loss_reduction'])

    def score_ehs_criteria(
        self,
        data: pd.DataFrame,
        z_a: str,
        z_b: List[str],
        treatment: str,
        outcome: str,
        use_permutation_test: bool = True,  # Ignored (always uses Wilcoxon)
        weak_instrument_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Score covariate z_a on EHS criteria using LOCO tests.

        EHS (Entner-Hoyer-Spirtes) criteria for valid instrument:
        - Test (i):   Y ⊥ Z_a | (Z_b, X)  → should NOT reject (exogeneity)
        - Test (ii):  Y ⊥̸ Z_a | Z_b       → should reject (outcome relevance)
        - Test (i.a): X ⊥̸ Z_a | Z_b       → should reject (instrument relevance)

        Args:
            data: DataFrame with variables
            z_a: Candidate instrumental covariate
            z_b: Other covariates
            treatment: Treatment column (X)
            outcome: Outcome column (Y)
            use_permutation_test: Ignored (LOCO always uses Wilcoxon)
            weak_instrument_threshold: Loss reduction threshold for weak instrument

        Returns:
            Same format as CITestEngine.score_ehs_criteria()
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

        # EHS criteria
        passes_ehs = (not test_i['reject_independence']) and test_ii['reject_independence']
        passes_full_ehs = passes_ehs and test_ia['reject_independence']

        # Score: loss reduction from outcome relevance minus exogeneity violation
        score = test_ii['cmi'] - test_i['cmi']

        # Weak instrument warning
        weak_instrument_warning = (test_ia['cmi'] < weak_instrument_threshold) and (score > 0)

        return {
            'z_a': z_a,
            'test_i_cmi': test_i['cmi'],
            'test_i_pvalue': test_i['p_value'],
            'test_i_reject': test_i['reject_independence'],
            'test_ii_cmi': test_ii['cmi'],
            'test_ii_pvalue': test_ii['p_value'],
            'test_ii_reject': test_ii['reject_independence'],
            'test_ia_cmi': test_ia['cmi'],
            'test_ia_pvalue': test_ia['p_value'],
            'test_ia_reject': test_ia['reject_independence'],
            'passes_ehs': passes_ehs,
            'passes_full_ehs': passes_full_ehs,
            'weak_instrument_warning': weak_instrument_warning,
            'score': score
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("LOCO CI Testing")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Generate synthetic data
    # W: two categorical variables
    W1 = np.random.choice(['A', 'B', 'C'], size=n)
    W2 = np.random.choice(['X', 'Y'], size=n)

    # X: binary treatment
    X = np.random.binomial(1, 0.5, size=n)

    # Scenario 1: Y depends on W but NOT X given W (null true)
    W1_effect = (W1 == 'A').astype(int) * 0.5
    W2_effect = (W2 == 'X').astype(int) * 0.3
    logit_Y = -0.5 + W1_effect + W2_effect
    prob_Y = 1 / (1 + np.exp(-logit_Y))
    Y_null = np.random.binomial(1, prob_Y)

    df_null = pd.DataFrame({
        'W1': W1, 'W2': W2, 'X': X, 'Y': Y_null
    })

    print("\n1. Testing H_0: Y ⊥ X | W (null TRUE - should NOT reject)")

    engine = LOCOCIEngine(function_class='gbm', random_state=42)
    result = engine.test_conditional_independence(df_null, 'X', 'Y', ['W1', 'W2'])
    print(f"   Reject: {result['reject_independence']}")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Loss reduction (CMI proxy): {result['cmi']:.4f}")

    # Scenario 2: Y depends on X given W (null false)
    logit_Y2 = -0.5 + W1_effect + W2_effect + 1.5 * X
    prob_Y2 = 1 / (1 + np.exp(-logit_Y2))
    Y_alt = np.random.binomial(1, prob_Y2)

    df_alt = pd.DataFrame({
        'W1': W1, 'W2': W2, 'X': X, 'Y': Y_alt
    })

    print("\n2. Testing H_0: Y ⊥ X | W (null FALSE - SHOULD reject)")
    result = engine.test_conditional_independence(df_alt, 'X', 'Y', ['W1', 'W2'])
    print(f"   Reject: {result['reject_independence']}")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Loss reduction (CMI proxy): {result['cmi']:.4f}")

    # EHS scoring example
    print("\n3. EHS Scoring Example")
    # Create data where W1 is a valid instrument
    Z = np.random.choice([0, 1, 2], size=n)  # Instrument
    X_inst = np.random.binomial(1, 0.3 + 0.2 * Z, n)  # Z affects X
    Y_inst = np.random.binomial(1, 0.3 + 0.3 * X_inst, n)  # X affects Y

    df_inst = pd.DataFrame({
        'Z': Z, 'X': X_inst, 'Y': Y_inst
    })

    ehs_result = engine.score_ehs_criteria(
        df_inst, z_a='Z', z_b=[], treatment='X', outcome='Y'
    )
    print(f"   passes_ehs: {ehs_result['passes_ehs']}")
    print(f"   passes_full_ehs: {ehs_result['passes_full_ehs']}")
    print(f"   score: {ehs_result['score']:.4f}")
    print(f"   test_i (exogeneity) p-value: {ehs_result['test_i_pvalue']:.4f}")
    print(f"   test_ii (outcome) p-value: {ehs_result['test_ii_pvalue']:.4f}")
    print(f"   test_ia (instrument) p-value: {ehs_result['test_ia_pvalue']:.4f}")
