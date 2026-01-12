"""
Causal Inference Methods for OSRCT Benchmark Evaluation

This module implements standard causal inference methods for estimating
Average Treatment Effects (ATEs) from observational data.

Methods Implemented:
1. Naive Difference-in-Means
2. Inverse Probability Weighting (IPW)
3. Outcome Regression (OR)
4. Augmented IPW / Doubly Robust (AIPW/DR)
5. Propensity Score Matching (PSM)
6. Causal Forest (requires econml)

Reference:
- Hernan & Robins (2020). Causal Inference: What If.
- Wager & Athey (2018). Estimation and Inference of Heterogeneous
  Treatment Effects using Random Forests. JASA.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bootstrap_se(
    data: pd.DataFrame,
    estimator_func: Callable,
    n_bootstrap: int = 200,
    random_state: int = 42
) -> float:
    """
    Compute standard error via bootstrap.

    Parameters
    ----------
    data : DataFrame
        Input data
    estimator_func : Callable
        Function that takes data and returns point estimate
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed

    Returns
    -------
    se : float
        Bootstrap standard error
    """
    np.random.seed(random_state)
    n = len(data)
    estimates = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data.iloc[idx]

        try:
            est = estimator_func(boot_data)
            if not np.isnan(est):
                estimates.append(est)
        except:
            continue

    if len(estimates) < 10:
        return np.nan

    return np.std(estimates)


def compute_smd(
    data: pd.DataFrame,
    treatment_col: str,
    covariate: str
) -> float:
    """
    Compute Standardized Mean Difference for covariate balance.

    SMD = (mean_treated - mean_control) / pooled_std
    """
    treated = data[data[treatment_col] == 1][covariate].dropna()
    control = data[data[treatment_col] == 0][covariate].dropna()

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    pooled_std = np.sqrt((treated.var() + control.var()) / 2)

    if pooled_std == 0:
        return 0.0

    return (treated.mean() - control.mean()) / pooled_std


# =============================================================================
# CAUSAL INFERENCE METHODS
# =============================================================================

def estimate_naive(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None  # unused, for interface consistency
) -> Dict[str, Any]:
    """
    Naive Difference-in-Means Estimator.

    ATE = E[Y|T=1] - E[Y|T=0]

    Assumption: No confounding (violated in OSRCT data)

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column (binary 0/1)
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Unused, for interface consistency

    Returns
    -------
    result : dict
        Dictionary with ate, se, ci_lower, ci_upper, method
    """
    treated = data[data[treatment_col] == 1][outcome_col].dropna()
    control = data[data[treatment_col] == 0][outcome_col].dropna()

    if len(treated) == 0 or len(control) == 0:
        return {
            'method': 'naive',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_treated': len(treated),
            'n_control': len(control)
        }

    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    return {
        'method': 'naive',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se,
        'n_treated': len(treated),
        'n_control': len(control)
    }


def estimate_ipw(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    trim_threshold: float = 0.01,
    normalize_weights: bool = True
) -> Dict[str, Any]:
    """
    Inverse Probability Weighting (Horvitz-Thompson) Estimator.

    ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]

    where e(X) = P(T=1|X) is the propensity score.

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Covariates for propensity score model
    trim_threshold : float
        Trim propensity scores outside [threshold, 1-threshold]
    normalize_weights : bool
        Whether to normalize weights (Hajek estimator)

    Returns
    -------
    result : dict
        Estimation results
    """
    if covariates is None:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']

    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 50:
        return {
            'method': 'ipw',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'Insufficient data after removing missing values'
        }

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values

    # Standardize covariates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit propensity score model
    try:
        ps_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        ps_model.fit(X_scaled, T)
        e = ps_model.predict_proba(X_scaled)[:, 1]
    except Exception as ex:
        return {
            'method': 'ipw',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': f'Propensity model failed: {str(ex)}'
        }

    # Trim extreme propensity scores
    e = np.clip(e, trim_threshold, 1 - trim_threshold)

    # IPW weights
    weights_treated = T / e
    weights_control = (1 - T) / (1 - e)

    if normalize_weights:
        # Hajek estimator (normalized weights)
        ate_treated = np.sum(Y * weights_treated) / np.sum(weights_treated)
        ate_control = np.sum(Y * weights_control) / np.sum(weights_control)
    else:
        # Horvitz-Thompson estimator
        n = len(Y)
        ate_treated = np.sum(Y * weights_treated) / n
        ate_control = np.sum(Y * weights_control) / n

    ate = ate_treated - ate_control

    # Bootstrap SE
    def ipw_point(d):
        d_clean = d[[treatment_col, outcome_col] + covariates].dropna()
        if len(d_clean) < 30:
            return np.nan
        X_b = scaler.transform(d_clean[covariates].values)
        T_b = d_clean[treatment_col].values
        Y_b = d_clean[outcome_col].values
        e_b = np.clip(ps_model.predict_proba(X_b)[:, 1], trim_threshold, 1 - trim_threshold)
        w_t = T_b / e_b
        w_c = (1 - T_b) / (1 - e_b)
        return np.sum(Y_b * w_t) / np.sum(w_t) - np.sum(Y_b * w_c) / np.sum(w_c)

    se = bootstrap_se(data_clean, ipw_point, n_bootstrap=100)

    return {
        'method': 'ipw',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se if not np.isnan(se) else np.nan,
        'ci_upper': ate + 1.96 * se if not np.isnan(se) else np.nan,
        'propensity_mean': e.mean(),
        'propensity_std': e.std(),
        'propensity_min': e.min(),
        'propensity_max': e.max(),
        'n_used': len(data_clean)
    }


def estimate_outcome_regression(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    include_interactions: bool = False
) -> Dict[str, Any]:
    """
    Outcome Regression (G-computation) Estimator.

    Fit: E[Y|T,X] = α + βT + γX (+ δ(T*X) if interactions)
    ATE = E[μ(1,X) - μ(0,X)]

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Covariates for outcome model
    include_interactions : bool
        Whether to include treatment-covariate interactions

    Returns
    -------
    result : dict
        Estimation results
    """
    if covariates is None:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']

    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 50:
        return {
            'method': 'outcome_regression',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'Insufficient data'
        }

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values.reshape(-1, 1)
    Y = data_clean[outcome_col].values

    # Standardize covariates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if include_interactions:
        # Include treatment-covariate interactions
        TX = T * X_scaled
        X_full = np.hstack([T, X_scaled, TX])
    else:
        X_full = np.hstack([T, X_scaled])

    # Fit outcome model
    try:
        model = Ridge(alpha=0.1, random_state=42)
        model.fit(X_full, Y)
    except Exception as ex:
        return {
            'method': 'outcome_regression',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': f'Outcome model failed: {str(ex)}'
        }

    # Predict potential outcomes
    if include_interactions:
        X_t1 = np.hstack([np.ones((len(X_scaled), 1)), X_scaled, X_scaled])
        X_t0 = np.hstack([np.zeros((len(X_scaled), 1)), X_scaled, np.zeros_like(X_scaled)])
    else:
        X_t1 = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
        X_t0 = np.hstack([np.zeros((len(X_scaled), 1)), X_scaled])

    y1_pred = model.predict(X_t1)
    y0_pred = model.predict(X_t0)

    # ATE is average of individual treatment effects
    ate = (y1_pred - y0_pred).mean()

    # Bootstrap SE
    def or_point(d):
        d_clean = d[[treatment_col, outcome_col] + covariates].dropna()
        if len(d_clean) < 30:
            return np.nan
        X_b = scaler.transform(d_clean[covariates].values)
        T_b = d_clean[treatment_col].values.reshape(-1, 1)
        Y_b = d_clean[outcome_col].values
        if include_interactions:
            X_full_b = np.hstack([T_b, X_b, T_b * X_b])
        else:
            X_full_b = np.hstack([T_b, X_b])
        model_b = Ridge(alpha=0.1)
        model_b.fit(X_full_b, Y_b)
        if include_interactions:
            X_t1_b = np.hstack([np.ones((len(X_b), 1)), X_b, X_b])
            X_t0_b = np.hstack([np.zeros((len(X_b), 1)), X_b, np.zeros_like(X_b)])
        else:
            X_t1_b = np.hstack([np.ones((len(X_b), 1)), X_b])
            X_t0_b = np.hstack([np.zeros((len(X_b), 1)), X_b])
        return (model_b.predict(X_t1_b) - model_b.predict(X_t0_b)).mean()

    se = bootstrap_se(data_clean, or_point, n_bootstrap=100)

    return {
        'method': 'outcome_regression',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se if not np.isnan(se) else np.nan,
        'ci_upper': ate + 1.96 * se if not np.isnan(se) else np.nan,
        'r_squared': model.score(X_full, Y),
        'n_used': len(data_clean)
    }


def estimate_aipw(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    trim_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Augmented Inverse Probability Weighting (Doubly Robust) Estimator.

    AIPW = E[μ₁(X) - μ₀(X) + T(Y - μ₁(X))/e(X) - (1-T)(Y - μ₀(X))/(1-e(X))]

    Properties:
    - Consistent if EITHER propensity OR outcome model is correct
    - More efficient than IPW when outcome model is correct
    - Semiparametric efficient

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Covariates for both models
    trim_threshold : float
        Trim propensity scores

    Returns
    -------
    result : dict
        Estimation results
    """
    if covariates is None:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']

    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 50:
        return {
            'method': 'aipw',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'Insufficient data'
        }

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values
    n = len(Y)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit propensity score model
    try:
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, T)
        e = np.clip(ps_model.predict_proba(X_scaled)[:, 1], trim_threshold, 1 - trim_threshold)
    except Exception as ex:
        return {
            'method': 'aipw',
            'ate': np.nan,
            'se': np.nan,
            'error': f'Propensity model failed: {str(ex)}'
        }

    # Fit outcome models for treated and control separately
    try:
        treated_mask = T == 1
        control_mask = T == 0

        mu1_model = Ridge(alpha=0.1)
        mu1_model.fit(X_scaled[treated_mask], Y[treated_mask])
        mu1 = mu1_model.predict(X_scaled)

        mu0_model = Ridge(alpha=0.1)
        mu0_model.fit(X_scaled[control_mask], Y[control_mask])
        mu0 = mu0_model.predict(X_scaled)
    except Exception as ex:
        return {
            'method': 'aipw',
            'ate': np.nan,
            'se': np.nan,
            'error': f'Outcome model failed: {str(ex)}'
        }

    # AIPW estimator (influence function approach)
    aipw_scores = (
        (mu1 - mu0) +
        T * (Y - mu1) / e -
        (1 - T) * (Y - mu0) / (1 - e)
    )

    ate = aipw_scores.mean()

    # Asymptotic SE from influence function
    se = aipw_scores.std() / np.sqrt(n)

    return {
        'method': 'aipw',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se,
        'propensity_mean': e.mean(),
        'propensity_std': e.std(),
        'n_used': len(data_clean)
    }


def estimate_psm(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    n_neighbors: int = 1,
    caliper: float = 0.2,
    with_replacement: bool = False
) -> Dict[str, Any]:
    """
    Propensity Score Matching Estimator.

    1. Estimate propensity scores e(X)
    2. Match each treated unit to nearest control(s) on e(X)
    3. Estimate ATT from matched sample

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Covariates for propensity model
    n_neighbors : int
        Number of control matches per treated unit
    caliper : float
        Maximum propensity score distance (in SD units)
    with_replacement : bool
        Whether to match with replacement

    Returns
    -------
    result : dict
        Estimation results
    """
    if covariates is None:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']

    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna().copy()
    data_clean = data_clean.reset_index(drop=True)

    if len(data_clean) < 50:
        return {
            'method': 'psm',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'Insufficient data'
        }

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit propensity scores
    try:
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, T)
        e = ps_model.predict_proba(X_scaled)[:, 1]
    except Exception as ex:
        return {
            'method': 'psm',
            'ate': np.nan,
            'se': np.nan,
            'error': f'Propensity model failed: {str(ex)}'
        }

    # Separate treated and control
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {
            'method': 'psm',
            'ate': np.nan,
            'se': np.nan,
            'error': 'No treated or control units'
        }

    e_treated = e[treated_idx]
    e_control = e[control_idx]

    # Caliper in propensity score units
    ps_std = e.std()
    caliper_dist = caliper * ps_std

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_idx)), metric='euclidean')
    nn.fit(e_control.reshape(-1, 1))
    distances, indices = nn.kneighbors(e_treated.reshape(-1, 1))

    # Apply caliper
    valid_matches = distances[:, 0] <= caliper_dist

    if valid_matches.sum() == 0:
        return {
            'method': 'psm',
            'ate': np.nan,
            'se': np.nan,
            'n_matched': 0,
            'error': 'No valid matches within caliper'
        }

    # Compute ATT from matched sample
    matched_treated_idx = treated_idx[valid_matches]
    matched_control_idx = control_idx[indices[valid_matches, 0]]

    Y_treated_matched = Y[matched_treated_idx]
    Y_control_matched = Y[matched_control_idx]

    # ATT (Average Treatment effect on Treated)
    ate = Y_treated_matched.mean() - Y_control_matched.mean()

    # SE (assuming independent matched pairs)
    se = np.sqrt(
        Y_treated_matched.var() / len(Y_treated_matched) +
        Y_control_matched.var() / len(Y_control_matched)
    )

    return {
        'method': 'psm',
        'ate': ate,
        'se': se,
        'ci_lower': ate - 1.96 * se,
        'ci_upper': ate + 1.96 * se,
        'n_matched': valid_matches.sum(),
        'match_rate': valid_matches.mean(),
        'n_treated': len(treated_idx),
        'n_control': len(control_idx),
        'caliper_used': caliper_dist
    }


def estimate_causal_forest(
    data: pd.DataFrame,
    treatment_col: str = 'iv',
    outcome_col: str = 'dv',
    covariates: List[str] = None,
    n_estimators: int = 500
) -> Dict[str, Any]:
    """
    Causal Forest Estimator (Wager & Athey, 2018).

    Uses random forest to estimate heterogeneous treatment effects τ(X),
    then averages to get ATE.

    Requires: econml package

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Name of treatment column
    outcome_col : str
        Name of outcome column
    covariates : List[str]
        Covariates for effect heterogeneity
    n_estimators : int
        Number of trees

    Returns
    -------
    result : dict
        Estimation results
    """
    if covariates is None:
        covariates = ['resp_age', 'resp_gender', 'resp_polideo']

    # Check if econml is available
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        return {
            'method': 'causal_forest',
            'ate': np.nan,
            'se': np.nan,
            'error': 'econml not installed. Install with: pip install econml'
        }

    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 100:
        return {
            'method': 'causal_forest',
            'ate': np.nan,
            'se': np.nan,
            'error': 'Insufficient data (need at least 100 observations)'
        }

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # Fit causal forest
        cf = CausalForestDML(
            n_estimators=n_estimators,
            random_state=42,
            inference=True,
            cv=3
        )
        cf.fit(Y, T, X=X_scaled, W=None)

        # Get treatment effects
        te = cf.effect(X_scaled)
        ate = te.mean()

        # Get confidence intervals
        te_interval = cf.effect_interval(X_scaled, alpha=0.05)

        # SE from confidence interval
        se = (te_interval[1] - te_interval[0]).mean() / (2 * 1.96)

        return {
            'method': 'causal_forest',
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'te_std': te.std(),  # Heterogeneity measure
            'te_min': te.min(),
            'te_max': te.max(),
            'n_used': len(data_clean)
        }

    except Exception as ex:
        return {
            'method': 'causal_forest',
            'ate': np.nan,
            'se': np.nan,
            'error': f'Causal forest failed: {str(ex)}'
        }


# =============================================================================
# UNIFIED EVALUATOR CLASS
# =============================================================================

class CausalMethodEvaluator:
    """
    Unified interface for evaluating causal inference methods.
    """

    METHODS = {
        'naive': estimate_naive,
        'ipw': estimate_ipw,
        'outcome_regression': estimate_outcome_regression,
        'aipw': estimate_aipw,
        'psm': estimate_psm,
        'causal_forest': estimate_causal_forest
    }

    def __init__(
        self,
        treatment_col: str = 'iv',
        outcome_col: str = 'dv',
        covariates: List[str] = None
    ):
        """
        Initialize evaluator.

        Parameters
        ----------
        treatment_col : str
            Name of treatment column
        outcome_col : str
            Name of outcome column
        covariates : List[str]
            Default covariates for adjustment methods
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariates = covariates or ['resp_age', 'resp_gender', 'resp_polideo']

    def evaluate_method(
        self,
        data: pd.DataFrame,
        method: str,
        ground_truth_ate: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single method on data.

        Parameters
        ----------
        data : DataFrame
            Observational data
        method : str
            Method name (naive, ipw, outcome_regression, aipw, psm, causal_forest)
        ground_truth_ate : float, optional
            True ATE for computing bias
        **kwargs
            Additional arguments for the method

        Returns
        -------
        result : dict
            Evaluation results
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")

        method_func = self.METHODS[method]

        # Call method
        if method == 'naive':
            result = method_func(data, self.treatment_col, self.outcome_col)
        else:
            result = method_func(
                data, self.treatment_col, self.outcome_col,
                self.covariates, **kwargs
            )

        # Add ground truth comparison if provided
        if ground_truth_ate is not None:
            result['ground_truth_ate'] = ground_truth_ate
            result['bias'] = result['ate'] - ground_truth_ate if not np.isnan(result['ate']) else np.nan
            result['abs_bias'] = abs(result['bias']) if not np.isnan(result['bias']) else np.nan
            result['relative_bias'] = result['bias'] / ground_truth_ate if ground_truth_ate != 0 else np.nan
            result['covers_truth'] = (
                result.get('ci_lower', -np.inf) <= ground_truth_ate <= result.get('ci_upper', np.inf)
            ) if not np.isnan(result.get('ci_lower', np.nan)) else np.nan

        return result

    def evaluate_all(
        self,
        data: pd.DataFrame,
        ground_truth_ate: float = None,
        methods: List[str] = None,
        skip_causal_forest: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate all methods on a single dataset.

        Parameters
        ----------
        data : DataFrame
            Observational data
        ground_truth_ate : float
            True ATE
        methods : List[str], optional
            Methods to evaluate (default: all)
        skip_causal_forest : bool
            Whether to skip causal forest (slow)

        Returns
        -------
        results : DataFrame
            Results for all methods
        """
        if methods is None:
            methods = list(self.METHODS.keys())
            if skip_causal_forest and 'causal_forest' in methods:
                methods.remove('causal_forest')

        results = []

        for method_name in methods:
            result = self.evaluate_method(data, method_name, ground_truth_ate)
            results.append(result)

        return pd.DataFrame(results)


# =============================================================================
# HETEROGENEITY ANALYSIS
# =============================================================================

def compute_heterogeneity_metrics(site_ates: pd.DataFrame) -> Dict[str, float]:
    """
    Compute metrics characterizing treatment effect heterogeneity across sites.

    Metrics:
    - I² statistic (percentage of variability due to heterogeneity)
    - Q statistic (Cochran's Q test for heterogeneity)
    - τ² (between-study variance)
    - Prediction interval (expected range of effects in new sites)

    Parameters
    ----------
    site_ates : DataFrame
        Must have 'ate' and 'ate_se' columns

    Returns
    -------
    metrics : dict
        Heterogeneity metrics
    """
    from scipy.stats import chi2

    ates = site_ates['ate'].dropna().values
    ses = site_ates['ate_se'].dropna().values

    if len(ates) < 2:
        return {
            'pooled_ate': ates[0] if len(ates) == 1 else np.nan,
            'n_sites': len(ates),
            'error': 'Need at least 2 sites for heterogeneity analysis'
        }

    # Filter out invalid SEs
    valid = ses > 0
    ates = ates[valid]
    ses = ses[valid]

    if len(ates) < 2:
        return {'error': 'Insufficient valid standard errors'}

    weights = 1 / ses**2

    # Fixed-effect pooled estimate
    pooled_ate = np.sum(weights * ates) / np.sum(weights)
    pooled_se = 1 / np.sqrt(np.sum(weights))

    # Cochran's Q
    Q = np.sum(weights * (ates - pooled_ate)**2)
    df = len(ates) - 1
    Q_pvalue = 1 - chi2.cdf(Q, df)

    # I² statistic
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    # τ² (DerSimonian-Laird estimator)
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0
    tau = np.sqrt(tau2)

    # Random-effects pooled estimate
    if tau2 > 0:
        re_weights = 1 / (ses**2 + tau2)
        re_pooled_ate = np.sum(re_weights * ates) / np.sum(re_weights)
        re_pooled_se = 1 / np.sqrt(np.sum(re_weights))
    else:
        re_pooled_ate = pooled_ate
        re_pooled_se = pooled_se

    # Prediction interval (expected range in new site)
    pred_se = np.sqrt(re_pooled_se**2 + tau2)
    pred_lower = re_pooled_ate - 1.96 * pred_se
    pred_upper = re_pooled_ate + 1.96 * pred_se

    return {
        'n_sites': len(ates),
        'pooled_ate_fe': pooled_ate,
        'pooled_se_fe': pooled_se,
        'pooled_ate_re': re_pooled_ate,
        'pooled_se_re': re_pooled_se,
        'Q_statistic': Q,
        'Q_df': df,
        'Q_pvalue': Q_pvalue,
        'I2': I2,
        'tau2': tau2,
        'tau': tau,
        'prediction_interval_lower': pred_lower,
        'prediction_interval_upper': pred_upper,
        'ate_range': (ates.min(), ates.max()),
        'ate_mean': ates.mean(),
        'ate_std': ates.std()
    }


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("Causal Methods Module")
    print("=" * 60)
    print("\nAvailable methods:")
    for method, func in CausalMethodEvaluator.METHODS.items():
        print(f"  - {method}: {func.__doc__.split(chr(10))[1].strip()}")

    print("\nUsage:")
    print("  from causal_methods import CausalMethodEvaluator")
    print("  evaluator = CausalMethodEvaluator()")
    print("  results = evaluator.evaluate_all(data, ground_truth_ate=1.5)")
