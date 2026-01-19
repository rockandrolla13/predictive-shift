"""
Prediction Models for CATE Estimation

This module provides pluggable prediction models for estimating
P(Y|X,A) and P(A|X) probabilities used in LP bounds computation.

Two approaches:
- EmpiricalPredictor: Empirical counts (current default approach)
- XGBoostPredictor: XGBoost-based models (Ricardo's approach from method.ipynb)

Based on Ricardo's learn_xgb and XGBoostBackdoorModel implementations.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import warnings

# XGBoost is optional
_XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    _XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore
    GridSearchCV = None  # type: ignore


class BasePredictorModel(ABC):
    """
    Abstract base class for prediction models.

    Defines interface for propensity and outcome prediction.
    """

    @abstractmethod
    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit propensity model P(A|X).

        Args:
            data: DataFrame with all variables
            treatment: Treatment column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit outcome model P(Y|X,A).

        Args:
            data: DataFrame with all variables
            outcome: Outcome column name
            treatment: Treatment column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict P(A=1|X).

        Args:
            X: Covariate values

        Returns:
            Array of propensity scores
        """
        pass

    @abstractmethod
    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """
        Predict P(Y=1|X,A) or E[Y|X,A].

        Args:
            X: Covariate values
            A: Treatment values (can be scalar for potential outcome prediction)

        Returns:
            Array of outcome predictions
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> 'BasePredictorModel':
        """
        Fit both propensity and outcome models.

        Args:
            data: DataFrame with all variables
            treatment: Treatment column name
            outcome: Outcome column name
            covariates: List of covariate column names

        Returns:
            self for chaining
        """
        self.fit_propensity(data, treatment, covariates)
        self.fit_outcome(data, outcome, treatment, covariates)
        return self

    def predict_potential_outcomes(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict potential outcomes Y(0) and Y(1).

        Args:
            X: Covariate values

        Returns:
            (Y0_pred, Y1_pred) arrays
        """
        Y0 = self.predict_outcome(X, 0)
        Y1 = self.predict_outcome(X, 1)
        return Y0, Y1

    def predict_cate(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict CATE = E[Y(1) - Y(0) | X].

        Args:
            X: Covariate values

        Returns:
            Array of CATE predictions
        """
        Y0, Y1 = self.predict_potential_outcomes(X)
        return Y1 - Y0


class EmpiricalPredictor(BasePredictorModel):
    """
    Empirical counts predictor.

    Uses empirical frequencies P(Y|X,A) and P(A|X) from the data.
    This is the default approach for the LP bounds computation.

    For discrete covariates, computes exact conditional probabilities.
    For continuous covariates, uses binning.
    """

    def __init__(
        self,
        smoothing: float = 0.01,
        n_bins: int = 5
    ):
        """
        Initialize empirical predictor.

        Args:
            smoothing: Laplace smoothing parameter for counts
            n_bins: Number of bins for continuous covariates
        """
        self.smoothing = smoothing
        self.n_bins = n_bins

        self._propensity_table = None
        self._outcome_tables = {}  # {A: table}
        self._covariates = None
        self._is_fitted_propensity = False
        self._is_fitted_outcome = False

    def _discretize_continuous(
        self,
        values: np.ndarray
    ) -> np.ndarray:
        """Discretize continuous values into bins."""
        if len(np.unique(values)) > self.n_bins:
            return pd.qcut(values, self.n_bins, labels=False, duplicates='drop')
        return values

    def _create_key(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Create string key from covariate values."""
        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        keys = []
        for row in X:
            key = '-'.join(str(int(v)) for v in row)
            keys.append(key)
        return np.array(keys)

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'EmpiricalPredictor':
        """Fit propensity model using empirical counts."""
        self._covariates = covariates

        # Create key for each row
        keys = self._create_key(data)

        # Count A=1 and total for each key
        self._propensity_table = {}

        for key in np.unique(keys):
            mask = keys == key
            n_treated = data.loc[mask, treatment].sum()
            n_total = mask.sum()

            # Laplace smoothing
            prob = (n_treated + self.smoothing) / (n_total + 2 * self.smoothing)
            self._propensity_table[key] = prob

        self._is_fitted_propensity = True
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'EmpiricalPredictor':
        """Fit outcome model using empirical counts."""
        self._covariates = covariates

        # Create key for each row
        keys = self._create_key(data)

        # Count Y=1 for each (key, A) combination
        for a in [0, 1]:
            self._outcome_tables[a] = {}
            a_mask = data[treatment] == a

            for key in np.unique(keys):
                key_mask = keys == key
                mask = key_mask & a_mask

                if mask.sum() > 0:
                    n_positive = data.loc[mask, outcome].sum()
                    n_total = mask.sum()
                    prob = (n_positive + self.smoothing) / (n_total + 2 * self.smoothing)
                else:
                    # No data for this combination, use marginal
                    prob = 0.5

                self._outcome_tables[a][key] = prob

        self._is_fitted_outcome = True
        return self

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict P(A=1|X) using empirical frequencies."""
        if not self._is_fitted_propensity:
            raise RuntimeError("Must fit propensity model first")

        keys = self._create_key(X)
        probs = np.array([
            self._propensity_table.get(key, 0.5)
            for key in keys
        ])
        return probs

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """Predict P(Y=1|X,A) using empirical frequencies."""
        if not self._is_fitted_outcome:
            raise RuntimeError("Must fit outcome model first")

        keys = self._create_key(X)

        if isinstance(A, (int, np.integer)):
            A = np.full(len(keys), A)

        probs = np.array([
            self._outcome_tables[int(a)].get(key, 0.5)
            for key, a in zip(keys, A)
        ])
        return probs

    def get_probability_table(self) -> pd.DataFrame:
        """
        Get full probability table for debugging.

        Returns:
            DataFrame with columns: key, P(A=1|X), P(Y=1|X,A=0), P(Y=1|X,A=1)
        """
        if not self._is_fitted_outcome:
            raise RuntimeError("Must fit models first")

        all_keys = set(self._propensity_table.keys())
        for table in self._outcome_tables.values():
            all_keys.update(table.keys())

        rows = []
        for key in sorted(all_keys):
            rows.append({
                'key': key,
                'P(A=1|X)': self._propensity_table.get(key, 0.5),
                'P(Y=1|X,A=0)': self._outcome_tables[0].get(key, 0.5),
                'P(Y=1|X,A=1)': self._outcome_tables[1].get(key, 0.5)
            })

        return pd.DataFrame(rows)


class XGBoostPredictor(BasePredictorModel):
    """
    XGBoost-based predictor.

    Uses XGBoost gradient boosting for propensity and outcome models.
    Based on Ricardo's learn_xgb implementation from method.ipynb.

    Advantages:
    - Handles high-dimensional covariates
    - Captures non-linear relationships
    - Provides feature importance
    - Better generalization to new covariate values
    """

    def __init__(
        self,
        use_grid_search: bool = True,
        n_estimators_range: List[int] = None,
        max_depth_range: List[int] = None,
        learning_rate_range: List[float] = None,
        cv_folds: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize XGBoost predictor.

        Args:
            use_grid_search: Use GridSearchCV for hyperparameter tuning
            n_estimators_range: Range of n_estimators to try
            max_depth_range: Range of max_depth to try
            learning_rate_range: Range of learning_rate to try
            cv_folds: Number of CV folds for grid search
            random_state: Random seed for reproducibility
        """
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not available. Install with: pip install xgboost"
            )

        self.use_grid_search = use_grid_search
        self.n_estimators_range = n_estimators_range or [100, 200, 400]
        self.max_depth_range = max_depth_range or [3, 5, 7]
        self.learning_rate_range = learning_rate_range or [0.05, 0.1, 0.2]
        self.cv_folds = cv_folds
        self.random_state = random_state

        self._propensity_model = None
        self._outcome_model = None  # Single model that takes A as feature
        self._covariates = None
        self._treatment = None
        self._is_binary_outcome = True

    def _learn_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        objective: str = 'binary:logistic'
    ) -> Any:
        """
        Learn XGBoost model with optional grid search.

        Based on Ricardo's learn_xgb implementation.

        Args:
            X: Feature matrix
            y: Target vector
            objective: XGBoost objective function

        Returns:
            Fitted XGBoost model (XGBClassifier)
        """
        base_params = {
            'objective': objective,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': self.random_state
        }

        if self.use_grid_search:
            param_grid = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'learning_rate': self.learning_rate_range
            }

            base_model = xgb.XGBClassifier(**base_params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring='neg_log_loss',
                    n_jobs=-1
                )
                grid_search.fit(X, y)

            return grid_search.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                **base_params
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)

            return model

    def fit_propensity(
        self,
        data: pd.DataFrame,
        treatment: str,
        covariates: List[str]
    ) -> 'XGBoostPredictor':
        """Fit propensity model P(A|X) using XGBoost."""
        self._covariates = covariates
        self._treatment = treatment

        X = data[covariates].values
        y = data[treatment].values

        self._propensity_model = self._learn_xgb(X, y)
        return self

    def fit_outcome(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: List[str]
    ) -> 'XGBoostPredictor':
        """Fit outcome model P(Y|X,A) using XGBoost."""
        self._covariates = covariates
        self._treatment = treatment

        # Check if outcome is binary
        unique_y = np.unique(data[outcome])
        self._is_binary_outcome = len(unique_y) == 2

        # Add treatment as feature
        X = np.column_stack([data[covariates].values, data[treatment].values])
        y = data[outcome].values

        if self._is_binary_outcome:
            self._outcome_model = self._learn_xgb(X, y)
        else:
            # For continuous outcome, use regression
            self._outcome_model = self._learn_xgb_regressor(X, y)

        return self

    def _learn_xgb_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Learn XGBoost regressor for continuous outcome."""
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state
        }

        if self.use_grid_search:
            param_grid = {
                'n_estimators': self.n_estimators_range,
                'max_depth': self.max_depth_range,
                'learning_rate': self.learning_rate_range
            }

            base_model = xgb.XGBRegressor(**base_params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X, y)

            return grid_search.best_estimator_
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                **base_params
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)

            return model

    def predict_propensity(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict P(A=1|X) using XGBoost."""
        if self._propensity_model is None:
            raise RuntimeError("Must fit propensity model first")

        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        return self._propensity_model.predict_proba(X)[:, 1]

    def predict_outcome(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        A: Union[np.ndarray, int]
    ) -> np.ndarray:
        """Predict P(Y=1|X,A) or E[Y|X,A] using XGBoost."""
        if self._outcome_model is None:
            raise RuntimeError("Must fit outcome model first")

        if isinstance(X, pd.DataFrame):
            X = X[self._covariates].values

        n = X.shape[0]

        if isinstance(A, (int, np.integer)):
            A = np.full(n, A)

        X_with_A = np.column_stack([X, A])

        if self._is_binary_outcome:
            return self._outcome_model.predict_proba(X_with_A)[:, 1]
        else:
            return self._outcome_model.predict(X_with_A)

    def get_feature_importance(
        self,
        model_type: str = 'outcome'
    ) -> pd.DataFrame:
        """
        Get feature importance from XGBoost model.

        Args:
            model_type: 'propensity' or 'outcome'

        Returns:
            DataFrame with feature importances
        """
        if model_type == 'propensity':
            if self._propensity_model is None:
                raise RuntimeError("Propensity model not fitted")
            model = self._propensity_model
            features = self._covariates
        else:
            if self._outcome_model is None:
                raise RuntimeError("Outcome model not fitted")
            model = self._outcome_model
            features = list(self._covariates) + [self._treatment]

        importance = model.feature_importances_

        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)


@dataclass
class XGBoostBackdoorModel:
    """
    Container for fitted XGBoost propensity and outcome models.

    Based on Ricardo's XGBoostBackdoorModel dataclass.

    Attributes:
        bst_propensity: Fitted propensity model
        bst_outcome: Fitted outcome model
        covariates: List of covariate names
        treatment: Treatment column name
        outcome: Outcome column name
    """
    bst_propensity: Any  # XGBClassifier
    bst_outcome: Any  # XGBClassifier or XGBRegressor
    covariates: List[str]
    treatment: str
    outcome: str
    is_binary_outcome: bool = True


def create_predictor(
    method: str = 'empirical',
    **kwargs
) -> BasePredictorModel:
    """
    Factory function to create predictor.

    Args:
        method: 'empirical' or 'xgboost'
        **kwargs: Arguments passed to predictor constructor

    Returns:
        BasePredictorModel instance

    Example:
        predictor = create_predictor('xgboost', use_grid_search=True)
        predictor = create_predictor('empirical', smoothing=0.1)
    """
    if method == 'empirical':
        return EmpiricalPredictor(**kwargs)
    elif method == 'xgboost':
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not available. Install with: pip install xgboost"
            )
        return XGBoostPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'empirical' or 'xgboost'.")


# Module test
if __name__ == "__main__":
    print("Prediction Models Test")
    print("=" * 50)

    # Create test data
    np.random.seed(42)
    n = 500

    # Simple DGP: Z -> X -> Y
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

    df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

    print("\n1. Testing EmpiricalPredictor...")
    emp = EmpiricalPredictor()
    emp.fit(df, treatment='X', outcome='Y', covariates=['Z'])

    prop = emp.predict_propensity(df[['Z']])
    print(f"   Propensity range: [{prop.min():.3f}, {prop.max():.3f}]")

    y0 = emp.predict_outcome(df[['Z']], 0)
    y1 = emp.predict_outcome(df[['Z']], 1)
    print(f"   Y(0) range: [{y0.min():.3f}, {y0.max():.3f}]")
    print(f"   Y(1) range: [{y1.min():.3f}, {y1.max():.3f}]")

    cate = emp.predict_cate(df[['Z']])
    print(f"   CATE range: [{cate.min():.3f}, {cate.max():.3f}]")

    if _XGBOOST_AVAILABLE:
        print("\n2. Testing XGBoostPredictor...")
        xgb_pred = XGBoostPredictor(use_grid_search=False, random_state=42)
        xgb_pred.fit(df, treatment='X', outcome='Y', covariates=['Z'])

        prop = xgb_pred.predict_propensity(df[['Z']])
        print(f"   Propensity range: [{prop.min():.3f}, {prop.max():.3f}]")

        y0 = xgb_pred.predict_outcome(df[['Z']], 0)
        y1 = xgb_pred.predict_outcome(df[['Z']], 1)
        print(f"   Y(0) range: [{y0.min():.3f}, {y0.max():.3f}]")
        print(f"   Y(1) range: [{y1.min():.3f}, {y1.max():.3f}]")

        cate = xgb_pred.predict_cate(df[['Z']])
        print(f"   CATE range: [{cate.min():.3f}, {cate.max():.3f}]")

        print("\n   Feature importance (outcome):")
        importance = xgb_pred.get_feature_importance('outcome')
        print(importance)
    else:
        print("\n2. XGBoost not available, skipping...")

    print("\n3. Testing factory function...")
    emp = create_predictor('empirical')
    print(f"   Empirical: {type(emp).__name__}")

    if _XGBOOST_AVAILABLE:
        xgb = create_predictor('xgboost', use_grid_search=False)
        print(f"   XGBoost: {type(xgb).__name__}")

    print("\nAll tests completed!")
