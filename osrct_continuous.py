"""
OSRCT Extension for Continuous Treatments

This module extends the OSRCT framework to handle continuous and multi-valued
treatments using Generalized Propensity Score (GPS) methods.

The standard OSRCT algorithm (Gentzel et al., 2021) handles binary treatments.
This extension enables:
1. Continuous treatments via GPS-based sampling
2. Multi-valued (categorical) treatments
3. Dose-response curve estimation

References:
- Hirano & Imbens (2004). The Propensity Score with Continuous Treatments.
- Gentzel et al. (2021). The Case for Evaluating Causal Models Using Controlled Experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Callable
from scipy.stats import norm, truncnorm
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KernelDensity
import warnings


# =============================================================================
# GENERALIZED PROPENSITY SCORE SAMPLER
# =============================================================================

class GPSSampler:
    """
    Generalized Propensity Score Sampler for Continuous Treatments.

    Creates confounded observational data from RCT data with continuous treatments
    by using GPS-based sampling. The GPS represents the conditional density of
    treatment given covariates: f(T|X).

    Algorithm:
    1. Estimate GPS: f(T|X) for each unit
    2. Compute weights based on how "unusual" each treatment level is
    3. Sample units with probability proportional to their weights
    4. Result: observational data with confounding between X and T
    """

    def __init__(
        self,
        biasing_covariates: List[str],
        biasing_coefficients: Optional[Dict[str, float]] = None,
        treatment_noise_scale: float = 1.0,
        sampling_method: str = 'importance',
        standardize: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize GPS Sampler.

        Parameters
        ----------
        biasing_covariates : list of str
            Pre-treatment covariates that influence treatment assignment
        biasing_coefficients : dict, optional
            Coefficients for each covariate in the GPS model.
            If None, random coefficients are generated.
        treatment_noise_scale : float, default=1.0
            Scale of noise in treatment model (controls confounding strength)
        sampling_method : str, default='importance'
            Sampling method: 'importance', 'rejection', or 'threshold'
        standardize : bool, default=True
            Whether to standardize covariates
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.biasing_covariates = biasing_covariates
        self.biasing_coefficients = biasing_coefficients
        self.treatment_noise_scale = treatment_noise_scale
        self.sampling_method = sampling_method
        self.standardize = standardize
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self.scaler = StandardScaler() if standardize else None
        self.is_fitted = False
        self._treatment_model = None

    def _fit_treatment_model(
        self,
        X: np.ndarray,
        T: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Fit a linear model for treatment given covariates.

        E[T|X] = X @ beta
        T|X ~ N(X @ beta, sigma^2)

        Returns predicted treatment and residual std.
        """
        model = Ridge(alpha=0.1)
        model.fit(X, T)

        T_pred = model.predict(X)
        residuals = T - T_pred
        sigma = np.std(residuals)

        self._treatment_model = model
        return T_pred, sigma

    def compute_gps(
        self,
        data: pd.DataFrame,
        treatment_col: str
    ) -> np.ndarray:
        """
        Compute Generalized Propensity Score f(T|X) for each unit.

        Parameters
        ----------
        data : DataFrame
            Data with treatment and covariates
        treatment_col : str
            Name of continuous treatment column

        Returns
        -------
        gps : ndarray
            GPS values (densities) for each unit
        """
        # Extract covariates
        X = data[self.biasing_covariates].values.copy()

        # Handle missing values
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Standardize
        if self.standardize:
            if not self.is_fitted:
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X = self.scaler.transform(X)

        T = data[treatment_col].values

        # Fit treatment model: T|X ~ N(X @ beta, sigma^2)
        T_pred, sigma = self._fit_treatment_model(X, T)

        # Compute GPS as normal density
        gps = norm.pdf(T, loc=T_pred, scale=sigma)

        return gps

    def _compute_confounding_weights(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        confounding_strength: float = 1.0
    ) -> np.ndarray:
        """
        Compute sampling weights that induce confounding.

        Higher weights for units where treatment is "expected" given covariates.
        This creates positive confounding (X -> T -> Y structure becomes biased).
        """
        # Extract covariates
        X = data[self.biasing_covariates].values.copy()

        # Handle missing
        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        if self.standardize:
            if not self.is_fitted:
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X = self.scaler.transform(X)

        T = data[treatment_col].values

        # Fit treatment model
        T_pred, sigma = self._fit_treatment_model(X, T)

        # Compute "conformity" score: how close is actual T to predicted T?
        # Higher score = treatment matches what covariates predict
        z_scores = (T - T_pred) / sigma
        conformity = norm.pdf(z_scores)  # Higher for z near 0

        # Convert to sampling weights
        # confounding_strength controls how much we bias toward "expected" treatments
        weights = np.exp(confounding_strength * conformity)
        weights = weights / weights.sum()

        return weights

    def sample(
        self,
        rct_data: pd.DataFrame,
        treatment_col: str,
        confounding_strength: float = 1.0,
        target_size: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate confounded observational sample from RCT data.

        Parameters
        ----------
        rct_data : DataFrame
            Original RCT data with continuous treatment
        treatment_col : str
            Name of continuous treatment column
        confounding_strength : float, default=1.0
            Controls how much confounding to introduce (higher = more)
        target_size : int, optional
            Target sample size. If None, uses 50% of original.
        verbose : bool, default=True
            Print sampling statistics

        Returns
        -------
        observational_sample : DataFrame
            Confounded observational dataset
        sampling_weights : ndarray
            Weights used for sampling
        """
        n = len(rct_data)
        if target_size is None:
            target_size = n // 2

        # Compute confounding weights
        weights = self._compute_confounding_weights(
            rct_data, treatment_col, confounding_strength
        )

        # Sample based on method
        if self.sampling_method == 'importance':
            # Importance sampling with replacement
            idx = np.random.choice(
                n, size=target_size, replace=True, p=weights
            )
        elif self.sampling_method == 'rejection':
            # Rejection sampling
            max_weight = weights.max()
            idx = []
            attempts = 0
            max_attempts = n * 10

            while len(idx) < target_size and attempts < max_attempts:
                i = np.random.randint(n)
                u = np.random.uniform()
                if u < weights[i] / max_weight:
                    idx.append(i)
                attempts += 1

            if len(idx) < target_size:
                warnings.warn(
                    f"Rejection sampling only got {len(idx)} samples after {max_attempts} attempts"
                )
            idx = np.array(idx)

        elif self.sampling_method == 'threshold':
            # Keep units with weight above threshold
            threshold = np.percentile(weights, 100 * (1 - target_size / n))
            idx = np.where(weights >= threshold)[0]
            if len(idx) > target_size:
                idx = np.random.choice(idx, size=target_size, replace=False)

        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Create observational sample
        observational_sample = rct_data.iloc[idx].copy()
        observational_sample['_sampling_weight'] = weights[idx]

        if verbose:
            print(f"GPS Sampling Results:")
            print(f"  Original RCT size: {n:,}")
            print(f"  Sampled size: {len(observational_sample):,}")
            print(f"  Confounding strength: {confounding_strength}")
            print(f"  Treatment range: [{rct_data[treatment_col].min():.2f}, {rct_data[treatment_col].max():.2f}]")

            # Check for confounding by correlating X with T
            if len(self.biasing_covariates) > 0:
                rct_corr = rct_data[self.biasing_covariates[0]].corr(rct_data[treatment_col])
                obs_corr = observational_sample[self.biasing_covariates[0]].corr(
                    observational_sample[treatment_col]
                )
                print(f"  Correlation({self.biasing_covariates[0]}, {treatment_col}):")
                print(f"    RCT: {rct_corr:.3f}")
                print(f"    Obs: {obs_corr:.3f}")

        return observational_sample, weights


# =============================================================================
# MULTI-VALUED TREATMENT SAMPLER
# =============================================================================

class MultiValuedTreatmentSampler:
    """
    OSRCT Sampler for Multi-Valued (Categorical) Treatments.

    Extends binary OSRCT to treatments with K > 2 categories.
    Uses multinomial logistic model for treatment assignment.
    """

    def __init__(
        self,
        biasing_covariates: List[str],
        biasing_coefficients: Optional[Dict[str, Dict[int, float]]] = None,
        standardize: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Multi-Valued Treatment Sampler.

        Parameters
        ----------
        biasing_covariates : list of str
            Covariates for biasing
        biasing_coefficients : dict, optional
            Nested dict: {covariate: {treatment_level: coefficient}}
            If None, random coefficients generated.
        standardize : bool, default=True
            Standardize covariates
        random_seed : int, optional
            Random seed
        """
        self.biasing_covariates = biasing_covariates
        self.biasing_coefficients = biasing_coefficients
        self.standardize = standardize
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self.scaler = StandardScaler() if standardize else None
        self.is_fitted = False

    def _compute_treatment_probabilities(
        self,
        X: np.ndarray,
        treatment_levels: List[int]
    ) -> np.ndarray:
        """
        Compute P(T=k|X) for each treatment level k.

        Uses softmax over linear predictors.
        """
        K = len(treatment_levels)
        n = X.shape[0]

        # Initialize coefficients if needed
        if self.biasing_coefficients is None:
            self.biasing_coefficients = {
                cov: {k: np.random.randn() for k in treatment_levels[1:]}  # K-1 coefficients
                for cov in self.biasing_covariates
            }

        # Compute linear predictors for each class (reference = first level)
        linear_preds = np.zeros((n, K))

        for j, cov in enumerate(self.biasing_covariates):
            for idx, k in enumerate(treatment_levels[1:], 1):
                coef = self.biasing_coefficients.get(cov, {}).get(k, 0)
                linear_preds[:, idx] += coef * X[:, j]

        # Softmax
        exp_preds = np.exp(linear_preds - linear_preds.max(axis=1, keepdims=True))
        probs = exp_preds / exp_preds.sum(axis=1, keepdims=True)

        return probs

    def sample(
        self,
        rct_data: pd.DataFrame,
        treatment_col: str,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate confounded sample for multi-valued treatment.

        Similar to binary OSRCT:
        1. Compute P(T=k|X) for each treatment level
        2. Sample preferred treatment from this distribution
        3. Keep units where actual treatment matches preferred

        Parameters
        ----------
        rct_data : DataFrame
            RCT data with multi-valued treatment
        treatment_col : str
            Treatment column name
        verbose : bool
            Print statistics

        Returns
        -------
        observational_sample : DataFrame
        selection_probs : ndarray
        """
        # Extract treatment levels
        treatment_levels = sorted(rct_data[treatment_col].dropna().unique())
        K = len(treatment_levels)

        if K < 2:
            raise ValueError("Treatment must have at least 2 levels")

        # Extract and prepare covariates
        X = rct_data[self.biasing_covariates].values.copy()

        if np.any(np.isnan(X)):
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        if self.standardize:
            if not self.is_fitted:
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X = self.scaler.transform(X)

        # Compute treatment probabilities
        probs = self._compute_treatment_probabilities(X, treatment_levels)

        # Map treatment values to indices
        T = rct_data[treatment_col].values
        T_idx = np.array([treatment_levels.index(t) for t in T])

        # Sample preferred treatment
        preferred_idx = np.array([
            np.random.choice(K, p=probs[i])
            for i in range(len(probs))
        ])

        # Keep where actual matches preferred
        keep_mask = (T_idx == preferred_idx)
        observational_sample = rct_data[keep_mask].copy()

        # Selection probability is prob of getting the actual treatment
        selection_probs = probs[np.arange(len(T)), T_idx]
        observational_sample['_selection_prob'] = selection_probs[keep_mask]

        if verbose:
            print(f"Multi-Valued Treatment Sampling Results:")
            print(f"  Treatment levels: {treatment_levels}")
            print(f"  Original RCT size: {len(rct_data):,}")
            print(f"  Sampled size: {len(observational_sample):,} ({100*len(observational_sample)/len(rct_data):.1f}%)")

            # Distribution comparison
            print(f"\n  Treatment distribution:")
            for k in treatment_levels:
                rct_pct = (rct_data[treatment_col] == k).mean() * 100
                obs_pct = (observational_sample[treatment_col] == k).mean() * 100
                print(f"    Level {k}: RCT={rct_pct:.1f}%, Obs={obs_pct:.1f}%")

        return observational_sample, selection_probs


# =============================================================================
# CAUSAL METHODS FOR CONTINUOUS TREATMENTS
# =============================================================================

def estimate_gps_weighting(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str],
    n_bins: int = 5,
    trim_threshold: float = 0.01
) -> Dict:
    """
    GPS-based weighting estimator for continuous treatment effects.

    Estimates dose-response curve E[Y|T=t] using GPS weighting.

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Continuous treatment column
    outcome_col : str
        Outcome column
    covariates : list
        Covariates for GPS model
    n_bins : int
        Number of treatment bins for dose-response
    trim_threshold : float
        Trim GPS values

    Returns
    -------
    result : dict
        Dose-response curve estimates and diagnostics
    """
    # Prepare data
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 50:
        return {'error': 'Insufficient data'}

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit treatment model: T|X ~ N(X @ beta, sigma^2)
    model = Ridge(alpha=0.1)
    model.fit(X_scaled, T)
    T_pred = model.predict(X_scaled)
    sigma = np.std(T - T_pred)

    # Compute GPS
    gps = norm.pdf(T, loc=T_pred, scale=sigma)
    gps = np.clip(gps, trim_threshold, None)

    # Compute marginal density of T
    kde = KernelDensity(bandwidth=0.5 * sigma)
    kde.fit(T.reshape(-1, 1))
    marginal_density = np.exp(kde.score_samples(T.reshape(-1, 1)))

    # Stabilized weights
    weights = marginal_density / gps
    weights = weights / weights.mean()

    # Estimate dose-response at different treatment levels
    T_min, T_max = T.min(), T.max()
    T_bins = np.linspace(T_min, T_max, n_bins + 1)
    T_centers = (T_bins[:-1] + T_bins[1:]) / 2

    dose_response = []
    for i in range(n_bins):
        mask = (T >= T_bins[i]) & (T < T_bins[i+1])
        if mask.sum() > 0:
            weighted_y = np.average(Y[mask], weights=weights[mask])
            dose_response.append({
                'treatment_level': T_centers[i],
                'expected_outcome': weighted_y,
                'n_units': mask.sum()
            })

    return {
        'method': 'gps_weighting',
        'dose_response': dose_response,
        'gps_mean': gps.mean(),
        'gps_std': gps.std(),
        'weight_mean': weights.mean(),
        'weight_std': weights.std(),
        'n_used': len(data_clean)
    }


def estimate_dose_response_kernel(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str],
    treatment_grid: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None
) -> Dict:
    """
    Kernel-based dose-response curve estimation.

    Uses local linear regression with GPS-based kernel weights.

    Parameters
    ----------
    data : DataFrame
        Observational data
    treatment_col : str
        Continuous treatment
    outcome_col : str
        Outcome
    covariates : list
        Covariates for GPS
    treatment_grid : ndarray, optional
        Treatment values at which to estimate E[Y|T]
    bandwidth : float, optional
        Kernel bandwidth

    Returns
    -------
    result : dict
        Dose-response curve
    """
    data_clean = data[[treatment_col, outcome_col] + covariates].dropna()

    if len(data_clean) < 50:
        return {'error': 'Insufficient data'}

    X = data_clean[covariates].values
    T = data_clean[treatment_col].values
    Y = data_clean[outcome_col].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GPS model
    model = Ridge(alpha=0.1)
    model.fit(X_scaled, T)
    T_pred = model.predict(X_scaled)
    sigma = np.std(T - T_pred)

    # GPS
    gps = norm.pdf(T, loc=T_pred, scale=sigma)

    # Treatment grid
    if treatment_grid is None:
        treatment_grid = np.linspace(T.min(), T.max(), 20)

    # Bandwidth
    if bandwidth is None:
        bandwidth = 1.06 * sigma * len(T) ** (-1/5)  # Silverman's rule

    # Estimate E[Y|T=t] for each t in grid
    dose_response = []

    for t in treatment_grid:
        # Kernel weights based on distance from t
        kernel_weights = norm.pdf((T - t) / bandwidth)

        # GPS weights (stabilized)
        gps_at_t = norm.pdf(t, loc=T_pred, scale=sigma)
        gps_weights = gps_at_t / gps

        # Combined weights
        combined_weights = kernel_weights * gps_weights
        combined_weights = combined_weights / combined_weights.sum()

        # Weighted outcome
        expected_y = np.sum(combined_weights * Y)

        dose_response.append({
            'treatment_level': t,
            'expected_outcome': expected_y,
            'effective_n': 1 / np.sum(combined_weights ** 2)  # Effective sample size
        })

    return {
        'method': 'kernel_dose_response',
        'dose_response': dose_response,
        'bandwidth': bandwidth,
        'n_used': len(data_clean)
    }


# =============================================================================
# UTILITIES
# =============================================================================

def evaluate_continuous_treatment_sample(
    rct_data: pd.DataFrame,
    obs_data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: List[str]
) -> Dict:
    """
    Evaluate GPS sampling by comparing RCT vs observational data.

    Parameters
    ----------
    rct_data : DataFrame
        Original RCT data
    obs_data : DataFrame
        GPS-sampled observational data
    treatment_col : str
        Treatment column
    outcome_col : str
        Outcome column
    covariates : list
        Covariates used for biasing

    Returns
    -------
    metrics : dict
        Comparison metrics
    """
    metrics = {}

    # Sample sizes
    metrics['n_rct'] = len(rct_data)
    metrics['n_obs'] = len(obs_data)
    metrics['retention_rate'] = len(obs_data) / len(rct_data)

    # Treatment distribution
    metrics['treatment_mean_rct'] = rct_data[treatment_col].mean()
    metrics['treatment_mean_obs'] = obs_data[treatment_col].mean()
    metrics['treatment_std_rct'] = rct_data[treatment_col].std()
    metrics['treatment_std_obs'] = obs_data[treatment_col].std()

    # Confounding: correlation between covariates and treatment
    confounding = {}
    for cov in covariates:
        rct_corr = rct_data[cov].corr(rct_data[treatment_col])
        obs_corr = obs_data[cov].corr(obs_data[treatment_col])
        confounding[cov] = {
            'rct_correlation': rct_corr,
            'obs_correlation': obs_corr,
            'change': obs_corr - rct_corr
        }
    metrics['confounding'] = confounding

    # Outcome distribution
    metrics['outcome_mean_rct'] = rct_data[outcome_col].mean()
    metrics['outcome_mean_obs'] = obs_data[outcome_col].mean()

    return metrics


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OSRCT Extension for Continuous Treatments")
    print("=" * 60)

    # Create synthetic continuous treatment data
    np.random.seed(42)
    n = 1000

    # Covariates
    age = np.random.normal(40, 10, n)
    income = np.random.exponential(50000, n)

    # Continuous treatment (e.g., dosage, hours of training)
    # In RCT, assigned randomly
    treatment = np.random.uniform(0, 100, n)

    # Outcome with treatment effect
    # True dose-response: Y = 10 + 0.5*T + 0.1*age + noise
    outcome = 10 + 0.5 * treatment + 0.1 * age + np.random.normal(0, 5, n)

    # Create dataframe
    rct_data = pd.DataFrame({
        'age': age,
        'income': income,
        'treatment': treatment,
        'outcome': outcome
    })

    print("\n1. Creating GPS Sampler...")
    sampler = GPSSampler(
        biasing_covariates=['age', 'income'],
        confounding_strength=2.0,
        random_seed=42
    )

    print("\n2. Generating Confounded Sample...")
    obs_data, weights = sampler.sample(
        rct_data,
        treatment_col='treatment',
        confounding_strength=2.0
    )

    print("\n3. Evaluating Sample...")
    metrics = evaluate_continuous_treatment_sample(
        rct_data, obs_data, 'treatment', 'outcome', ['age', 'income']
    )

    print(f"\n  Confounding induced:")
    for cov, stats in metrics['confounding'].items():
        print(f"    {cov}: RCT corr={stats['rct_correlation']:.3f}, Obs corr={stats['obs_correlation']:.3f}")

    print("\n4. Estimating Dose-Response...")
    dr_result = estimate_gps_weighting(
        obs_data, 'treatment', 'outcome', ['age', 'income']
    )

    print(f"\n  Dose-Response Curve:")
    for point in dr_result['dose_response']:
        print(f"    T={point['treatment_level']:.1f}: E[Y]={point['expected_outcome']:.2f}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
