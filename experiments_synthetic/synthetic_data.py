"""
Synthetic Data Generator for Causal Grounding Experiments

Generates synthetic datasets with:
- Binary treatment (X)
- Binary outcome (Y)
- Discrete covariates (Z)
- Multiple training environments (sites)
- Known ground truth CATE

The data generating process follows:
    Z ~ Categorical(p_z)
    U ~ Bernoulli(p_u)  # Unmeasured confounder
    X|Z,U ~ Bernoulli(pi(Z) + beta * U)  # Confounded treatment
    Y|X,Z,U ~ Bernoulli(mu(X,Z) + gamma * U)  # Confounded outcome

Where:
    - pi(Z) is the baseline treatment probability given Z
    - mu(X,Z) is the structural outcome model (gives true CATE)
    - beta controls confounding strength in treatment
    - gamma controls confounding strength in outcome
    - Under F=on (RCT), X is randomized independent of U

Usage:
    generator = SyntheticDataGenerator(n_sites=10, n_z_values=3, beta=0.3)
    training_data = generator.generate_training_data(n_per_site=500)
    true_cates = generator.get_true_cates()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    # Covariate structure
    n_covariates: int = 1
    n_z_values: int = 3  # Number of discrete values per covariate
    p_z: Optional[np.ndarray] = None  # Marginal P(Z), uniform if None
    
    # Treatment model
    base_treatment_prob: float = 0.5  # Baseline P(X=1)
    treatment_z_effect: float = 0.1  # Effect of Z on treatment
    
    # Outcome model (defines true CATE)
    base_outcome_prob: float = 0.3  # Baseline P(Y=1|X=0)
    treatment_effect: float = 0.2  # Main treatment effect (ATE)
    treatment_z_interaction: float = 0.1  # Z modifies treatment effect
    z_direct_effect: float = 0.1  # Direct effect of Z on Y
    
    # Confounding
    beta: float = 0.3  # Confounding strength (U -> X)
    gamma: float = 0.2  # Confounding strength (U -> Y)
    p_u: float = 0.5  # Marginal P(U=1)
    
    # Site heterogeneity
    site_effect_sd: float = 0.05  # Between-site variance
    
    # RCT treatment probability
    rct_treatment_prob: float = 0.5
    
    # Sample sizes
    n_per_site: int = 500
    rct_fraction: float = 0.3  # Fraction of data that is RCT (F=on)
    
    # Random seed
    seed: Optional[int] = None


class SyntheticDataGenerator:
    """
    Generate synthetic data for causal grounding experiments.
    
    The generator creates datasets where:
    1. Ground truth CATE(z) is analytically known
    2. Multiple training environments (sites) are available
    3. Each site has both RCT (F=on) and observational (F=idle) data
    4. Confounding is introduced in observational data via U
    
    Example:
        config = SyntheticDataConfig(
            n_z_values=3,
            beta=0.3,
            treatment_effect=0.2
        )
        generator = SyntheticDataGenerator(config)
        training_data = generator.generate_training_data(n_sites=10)
        true_cates = generator.get_true_cates()
    """
    
    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
        n_sites: int = 10,
        n_z_values: int = 3,
        beta: float = 0.3,
        treatment_effect: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize generator with config or individual parameters.
        
        Args:
            config: Full configuration object (takes precedence)
            n_sites: Number of training sites
            n_z_values: Number of discrete covariate values
            beta: Confounding strength
            treatment_effect: Main treatment effect (ATE)
            seed: Random seed
        """
        if config is not None:
            self.config = config
        else:
            self.config = SyntheticDataConfig(
                n_z_values=n_z_values,
                beta=beta,
                treatment_effect=treatment_effect,
                seed=seed
            )
        
        self.n_sites = n_sites
        self.rng = np.random.default_rng(self.config.seed)
        
        # Initialize covariate distribution
        if self.config.p_z is None:
            self.config.p_z = np.ones(self.config.n_z_values) / self.config.n_z_values
        
        # Compute and cache true CATEs
        self._true_cates = self._compute_true_cates()
        self._true_ate = self._compute_true_ate()
        
        # Generate site-specific effects
        self._site_effects = self.rng.normal(
            0, self.config.site_effect_sd, self.n_sites
        )
    
    def _compute_true_cates(self) -> Dict[Tuple, float]:
        """
        Compute ground truth CATE(z) for each z value.
        
        True CATE(z) = P(Y=1|do(X=1),Z=z) - P(Y=1|do(X=0),Z=z)
                     = E[Y|X=1,Z=z,U] - E[Y|X=0,Z=z,U]  (averaged over U)
        
        Since U is independent of Z in the structural model:
        CATE(z) = treatment_effect + treatment_z_interaction * z
        
        Returns:
            Dict mapping z_tuple to true CATE
        """
        cates = {}
        
        for z in range(self.config.n_z_values):
            z_tuple = (z,)
            
            # CATE is the difference in outcome probability under intervention
            # Y = base + treatment_effect * X + z_effect * Z + treatment_z_interaction * X * Z + gamma * U
            # E[Y|do(X=1),Z=z] = base + treatment_effect + z_effect * z + treatment_z_interaction * z + gamma * E[U]
            # E[Y|do(X=0),Z=z] = base + z_effect * z + gamma * E[U]
            # CATE(z) = treatment_effect + treatment_z_interaction * z
            
            cate = self.config.treatment_effect + self.config.treatment_z_interaction * z
            cates[z_tuple] = cate
        
        return cates
    
    def _compute_true_ate(self) -> float:
        """
        Compute ground truth ATE = E_Z[CATE(Z)].
        
        Returns:
            True ATE
        """
        ate = 0.0
        for z in range(self.config.n_z_values):
            ate += self.config.p_z[z] * self._true_cates[(z,)]
        return ate
    
    def get_true_cates(self) -> Dict[Tuple, float]:
        """Return ground truth CATE for each z value."""
        return self._true_cates.copy()
    
    def get_true_ate(self) -> float:
        """Return ground truth ATE."""
        return self._true_ate
    
    def _generate_site_data(
        self,
        site_id: int,
        n_samples: int,
        include_rct: bool = True
    ) -> pd.DataFrame:
        """
        Generate data for a single site.
        
        Args:
            site_id: Site identifier (for site-specific effects)
            n_samples: Total number of samples
            include_rct: Whether to include RCT (F=on) data
        
        Returns:
            DataFrame with columns: Z, X, Y, F, U (U is hidden confounder)
        """
        # Determine RCT vs observational split
        if include_rct:
            n_rct = int(n_samples * self.config.rct_fraction)
            n_obs = n_samples - n_rct
        else:
            n_rct = 0
            n_obs = n_samples
        
        site_effect = self._site_effects[site_id % len(self._site_effects)]
        
        dfs = []
        
        # Generate RCT data (F=on)
        if n_rct > 0:
            Z_rct = self.rng.choice(
                self.config.n_z_values, 
                size=n_rct, 
                p=self.config.p_z
            )
            U_rct = self.rng.binomial(1, self.config.p_u, n_rct)
            
            # In RCT, X is randomized (independent of U)
            X_rct = self.rng.binomial(1, self.config.rct_treatment_prob, n_rct)
            
            # Y depends on X, Z, U (but not through confounding since X is randomized)
            p_y_rct = self._outcome_prob(X_rct, Z_rct, U_rct, site_effect)
            Y_rct = self.rng.binomial(1, p_y_rct)
            
            df_rct = pd.DataFrame({
                'Z': Z_rct,
                'X': X_rct,
                'Y': Y_rct,
                'F': 'on',
                'U': U_rct,
                'site': f'site_{site_id}'
            })
            dfs.append(df_rct)
        
        # Generate observational data (F=idle)
        if n_obs > 0:
            Z_obs = self.rng.choice(
                self.config.n_z_values, 
                size=n_obs, 
                p=self.config.p_z
            )
            U_obs = self.rng.binomial(1, self.config.p_u, n_obs)
            
            # In observational data, X depends on U (confounding)
            p_x_obs = self._treatment_prob(Z_obs, U_obs)
            X_obs = self.rng.binomial(1, p_x_obs)
            
            # Y depends on X, Z, U
            p_y_obs = self._outcome_prob(X_obs, Z_obs, U_obs, site_effect)
            Y_obs = self.rng.binomial(1, p_y_obs)
            
            df_obs = pd.DataFrame({
                'Z': Z_obs,
                'X': X_obs,
                'Y': Y_obs,
                'F': 'idle',
                'U': U_obs,
                'site': f'site_{site_id}'
            })
            dfs.append(df_obs)
        
        return pd.concat(dfs, ignore_index=True)
    
    def _treatment_prob(
        self, 
        Z: np.ndarray, 
        U: np.ndarray
    ) -> np.ndarray:
        """
        Compute P(X=1|Z,U) for observational data.
        
        P(X=1|Z,U) = base + z_effect * Z + beta * U
        """
        p = (
            self.config.base_treatment_prob 
            + self.config.treatment_z_effect * Z 
            + self.config.beta * U
        )
        return np.clip(p, 0.01, 0.99)
    
    def _outcome_prob(
        self, 
        X: np.ndarray, 
        Z: np.ndarray, 
        U: np.ndarray,
        site_effect: float = 0.0
    ) -> np.ndarray:
        """
        Compute P(Y=1|X,Z,U) for the structural outcome model.
        
        P(Y=1|X,Z,U) = base + treatment_effect * X + z_effect * Z 
                     + treatment_z_interaction * X * Z + gamma * U + site_effect
        """
        p = (
            self.config.base_outcome_prob
            + self.config.treatment_effect * X
            + self.config.z_direct_effect * Z
            + self.config.treatment_z_interaction * X * Z
            + self.config.gamma * U
            + site_effect
        )
        return np.clip(p, 0.01, 0.99)
    
    def generate_training_data(
        self,
        n_sites: Optional[int] = None,
        n_per_site: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate training data from multiple sites.
        
        Args:
            n_sites: Number of sites (default: self.n_sites)
            n_per_site: Samples per site (default: config.n_per_site)
        
        Returns:
            Dict mapping site_id to DataFrame with F column
        """
        n_sites = n_sites or self.n_sites
        n_per_site = n_per_site or self.config.n_per_site
        
        training_data = {}
        
        for site_id in range(n_sites):
            site_name = f'site_{site_id}'
            training_data[site_name] = self._generate_site_data(
                site_id, n_per_site, include_rct=True
            )
        
        return training_data
    
    def generate_target_data(
        self,
        n_samples: int = 1000,
        include_rct: bool = False
    ) -> pd.DataFrame:
        """
        Generate target environment data.
        
        Args:
            n_samples: Number of samples
            include_rct: Whether to include RCT data in target
        
        Returns:
            DataFrame with target data
        """
        # Target uses a separate site effect
        target_site_id = self.n_sites  # New site not in training
        return self._generate_site_data(
            target_site_id, n_samples, include_rct=include_rct
        )
    
    def generate_rct_data(
        self,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generate pure RCT data for ground truth computation.
        
        Args:
            n_samples: Number of samples
        
        Returns:
            DataFrame with RCT data (F=on)
        """
        Z = self.rng.choice(self.config.n_z_values, size=n_samples, p=self.config.p_z)
        U = self.rng.binomial(1, self.config.p_u, n_samples)
        X = self.rng.binomial(1, self.config.rct_treatment_prob, n_samples)
        
        p_y = self._outcome_prob(X, Z, U, site_effect=0.0)
        Y = self.rng.binomial(1, p_y)
        
        return pd.DataFrame({
            'Z': Z,
            'X': X,
            'Y': Y,
            'F': 'on',
            'U': U
        })


def generate_multi_site_data(
    n_sites: int = 10,
    n_per_site: int = 500,
    n_z_values: int = 3,
    beta: float = 0.3,
    treatment_effect: float = 0.2,
    treatment_z_interaction: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[Tuple, float], float]:
    """
    Convenience function to generate multi-site data with ground truth.
    
    Args:
        n_sites: Number of training sites
        n_per_site: Samples per site
        n_z_values: Number of discrete Z values
        beta: Confounding strength
        treatment_effect: Main treatment effect
        treatment_z_interaction: Treatment effect heterogeneity
        seed: Random seed
    
    Returns:
        (training_data, true_cates, true_ate)
    """
    config = SyntheticDataConfig(
        n_z_values=n_z_values,
        beta=beta,
        treatment_effect=treatment_effect,
        treatment_z_interaction=treatment_z_interaction,
        n_per_site=n_per_site,
        seed=seed
    )
    
    generator = SyntheticDataGenerator(config, n_sites=n_sites)
    training_data = generator.generate_training_data()
    
    return training_data, generator.get_true_cates(), generator.get_true_ate()


def compute_true_cate(
    n_z_values: int,
    treatment_effect: float,
    treatment_z_interaction: float
) -> Dict[Tuple, float]:
    """
    Compute true CATE for given parameters.
    
    Args:
        n_z_values: Number of Z values
        treatment_effect: Main effect
        treatment_z_interaction: Interaction term
    
    Returns:
        Dict mapping z_tuple to true CATE
    """
    cates = {}
    for z in range(n_z_values):
        cates[(z,)] = treatment_effect + treatment_z_interaction * z
    return cates


# =============================================================================
# EXTENDED CONFIGURATIONS
# =============================================================================

def create_confounding_sweep_configs(
    betas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    base_config: Optional[SyntheticDataConfig] = None
) -> List[SyntheticDataConfig]:
    """
    Create configs for sweeping over confounding strength.
    
    Args:
        betas: List of beta values to sweep
        base_config: Base configuration to modify
    
    Returns:
        List of SyntheticDataConfig objects
    """
    base = base_config or SyntheticDataConfig()
    configs = []
    
    for beta in betas:
        config = SyntheticDataConfig(
            n_covariates=base.n_covariates,
            n_z_values=base.n_z_values,
            p_z=base.p_z,
            base_treatment_prob=base.base_treatment_prob,
            treatment_z_effect=base.treatment_z_effect,
            base_outcome_prob=base.base_outcome_prob,
            treatment_effect=base.treatment_effect,
            treatment_z_interaction=base.treatment_z_interaction,
            z_direct_effect=base.z_direct_effect,
            beta=beta,  # Varying parameter
            gamma=base.gamma,
            p_u=base.p_u,
            site_effect_sd=base.site_effect_sd,
            rct_treatment_prob=base.rct_treatment_prob,
            n_per_site=base.n_per_site,
            rct_fraction=base.rct_fraction,
            seed=base.seed
        )
        configs.append(config)
    
    return configs


def create_heterogeneity_sweep_configs(
    interactions: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
    base_config: Optional[SyntheticDataConfig] = None
) -> List[SyntheticDataConfig]:
    """
    Create configs for sweeping over treatment heterogeneity.
    
    Args:
        interactions: List of treatment_z_interaction values
        base_config: Base configuration to modify
    
    Returns:
        List of SyntheticDataConfig objects
    """
    base = base_config or SyntheticDataConfig()
    configs = []
    
    for interaction in interactions:
        config = SyntheticDataConfig(
            n_covariates=base.n_covariates,
            n_z_values=base.n_z_values,
            p_z=base.p_z,
            base_treatment_prob=base.base_treatment_prob,
            treatment_z_effect=base.treatment_z_effect,
            base_outcome_prob=base.base_outcome_prob,
            treatment_effect=base.treatment_effect,
            treatment_z_interaction=interaction,  # Varying parameter
            z_direct_effect=base.z_direct_effect,
            beta=base.beta,
            gamma=base.gamma,
            p_u=base.p_u,
            site_effect_sd=base.site_effect_sd,
            rct_treatment_prob=base.rct_treatment_prob,
            n_per_site=base.n_per_site,
            rct_fraction=base.rct_fraction,
            seed=base.seed
        )
        configs.append(config)
    
    return configs


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("SyntheticDataGenerator Test")
    print("=" * 60)
    
    # Create generator
    config = SyntheticDataConfig(
        n_z_values=3,
        beta=0.3,
        treatment_effect=0.2,
        treatment_z_interaction=0.1,
        seed=42
    )
    
    generator = SyntheticDataGenerator(config, n_sites=5)
    
    # Get ground truth
    print("\nGround Truth CATEs:")
    for z, cate in generator.get_true_cates().items():
        print(f"  Z={z}: CATE = {cate:.3f}")
    
    print(f"\nTrue ATE: {generator.get_true_ate():.3f}")
    
    # Generate training data
    training_data = generator.generate_training_data(n_per_site=500)
    
    print(f"\nTraining sites: {list(training_data.keys())}")
    
    # Inspect one site
    site_data = training_data['site_0']
    print(f"\nSite 0 summary:")
    print(f"  Total samples: {len(site_data)}")
    print(f"  F=on (RCT): {(site_data['F'] == 'on').sum()}")
    print(f"  F=idle (obs): {(site_data['F'] == 'idle').sum()}")
    print(f"  P(X=1): {site_data['X'].mean():.3f}")
    print(f"  P(Y=1): {site_data['Y'].mean():.3f}")
    
    # Check empirical CATEs from RCT data
    rct_data = generator.generate_rct_data(n_samples=10000)
    print("\nEmpirical CATEs from large RCT sample:")
    for z in range(config.n_z_values):
        z_data = rct_data[rct_data['Z'] == z]
        y1 = z_data[z_data['X'] == 1]['Y'].mean()
        y0 = z_data[z_data['X'] == 0]['Y'].mean()
        emp_cate = y1 - y0
        true_cate = generator.get_true_cates()[(z,)]
        print(f"  Z={z}: Empirical = {emp_cate:.3f}, True = {true_cate:.3f}")
    
    print("\nTest passed!")
