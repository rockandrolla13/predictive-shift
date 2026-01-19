"""
Covariate Scoring by EHS Criteria

This module ranks covariates by EHS (Entner-Hoyer-Spirtes) criteria
across training environments to identify valid instrumental covariates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .ci_tests import CITestEngine


def rank_covariates(
    data: pd.DataFrame,
    covariates: List[str],
    treatment: str,
    outcome: str,
    ci_engine: Optional[CITestEngine] = None,
    use_permutation_test: bool = True
) -> pd.DataFrame:
    """
    Rank covariates by EHS score.

    For each covariate z_a, uses remaining covariates as z_b.

    Args:
        data: DataFrame with discretized covariates
        covariates: List of covariate column names
        treatment: Treatment column name
        outcome: Outcome column name
        ci_engine: CITestEngine instance (created if None)
        use_permutation_test: Use full permutation test

    Returns:
        DataFrame sorted by score (descending) with columns:
            z_a, test_i_cmi, test_i_pvalue, test_i_reject,
            test_ii_cmi, test_ii_pvalue, test_ii_reject,
            passes_ehs, score
    """
    if ci_engine is None:
        ci_engine = CITestEngine()

    results = []

    for z_a in covariates:
        z_b = [c for c in covariates if c != z_a]

        score_result = ci_engine.score_ehs_criteria(
            data, z_a, z_b, treatment, outcome,
            use_permutation_test=use_permutation_test
        )
        results.append(score_result)

    df = pd.DataFrame(results)
    return df.sort_values('score', ascending=False).reset_index(drop=True)


def rank_covariates_across_sites(
    training_data: Dict[str, pd.DataFrame],
    covariates: List[str],
    treatment: str,
    outcome: str,
    ci_engine: Optional[CITestEngine] = None,
    use_permutation_test: bool = True,
    regime_col: str = 'F'
) -> pd.DataFrame:
    """
    Rank covariates by aggregated EHS score across training sites.

    For each site, scores are computed on F=idle data only.
    Final score is mean across sites.

    Args:
        training_data: Dict[site_id -> DataFrame with 'F' column]
        covariates: List of covariate column names
        treatment: Treatment column name
        outcome: Outcome column name
        ci_engine: CITestEngine instance
        use_permutation_test: Use full permutation test
        regime_col: Regime indicator column name

    Returns:
        DataFrame with aggregated scores and per-site breakdown
    """
    if ci_engine is None:
        ci_engine = CITestEngine()

    all_results = []

    for site_id, site_data in training_data.items():
        # Use only F=idle data for scoring
        idle_data = site_data[site_data[regime_col] == 'idle']

        if len(idle_data) < 50:  # Skip small sites
            continue

        for z_a in covariates:
            z_b = [c for c in covariates if c != z_a]

            score_result = ci_engine.score_ehs_criteria(
                idle_data, z_a, z_b, treatment, outcome,
                use_permutation_test=use_permutation_test
            )
            score_result['site'] = site_id
            all_results.append(score_result)

    df = pd.DataFrame(all_results)

    # Aggregate across sites
    agg_df = df.groupby('z_a').agg({
        'test_i_cmi': 'mean',
        'test_i_pvalue': 'mean',
        'test_i_reject': 'mean',  # Proportion of sites
        'test_ii_cmi': 'mean',
        'test_ii_pvalue': 'mean',
        'test_ii_reject': 'mean',
        'test_ia_cmi': 'mean',
        'test_ia_pvalue': 'mean',
        'test_ia_reject': 'mean',  # Proportion of sites with instrument relevance
        'passes_ehs': 'mean',
        'passes_full_ehs': 'mean',  # Proportion passing all 3 criteria
        'weak_instrument_warning': 'max',  # True if any site flagged
        'score': 'mean',
        'site': 'count'  # Number of sites
    }).rename(columns={'site': 'n_sites'})

    return agg_df.sort_values('score', ascending=False)


def select_best_instrument(
    scores_df: pd.DataFrame,
    require_passes_ehs: bool = False,
    require_full_ehs: bool = False,
    min_pass_rate: float = 0.5,
    exclude_weak_instruments: bool = False
) -> Optional[str]:
    """
    Select best instrumental covariate from ranked scores.

    Args:
        scores_df: DataFrame from rank_covariates
        require_passes_ehs: Only consider covariates passing original EHS (test_i, test_ii)
        require_full_ehs: Only consider covariates passing full EHS (includes test_ia)
        min_pass_rate: Minimum proportion of sites passing EHS criteria
        exclude_weak_instruments: Exclude covariates with weak_instrument_warning

    Returns:
        Best covariate name, or None if none qualify
    """
    df = scores_df.copy()

    # Filter by full EHS (includes instrument relevance test_ia)
    if require_full_ehs:
        if 'passes_full_ehs' in df.columns:
            if df['passes_full_ehs'].dtype == float:
                df = df[df['passes_full_ehs'] >= min_pass_rate]
            else:
                df = df[df['passes_full_ehs'] == True]
    # Filter by original EHS (test_i, test_ii only)
    elif require_passes_ehs:
        if 'passes_ehs' in df.columns:
            if df['passes_ehs'].dtype == float:
                df = df[df['passes_ehs'] >= min_pass_rate]
            else:
                df = df[df['passes_ehs'] == True]

    # Exclude weak instruments
    if exclude_weak_instruments and 'weak_instrument_warning' in df.columns:
        df = df[df['weak_instrument_warning'] == False]

    if len(df) == 0:
        return None

    # Return highest scoring
    if 'z_a' in df.columns:
        return df.iloc[0]['z_a']
    else:
        return df.index[0]


def select_top_k_instruments(
    scores_df: pd.DataFrame,
    k: int = 3,
    require_passes_ehs: bool = False,
    require_full_ehs: bool = False,
    min_pass_rate: float = 0.5,
    exclude_weak_instruments: bool = False
) -> List[str]:
    """
    Select top-k instrumental covariates from ranked scores.

    This extends select_best_instrument to support multi-instrument
    aggregation as in Ricardo's approach.

    Args:
        scores_df: DataFrame from rank_covariates
        k: Number of instruments to select
        require_passes_ehs: Only consider covariates passing EHS (test_i, test_ii)
        require_full_ehs: Only consider covariates passing full EHS (includes test_ia)
        min_pass_rate: Minimum proportion of sites passing EHS criteria
        exclude_weak_instruments: Exclude covariates with weak_instrument_warning

    Returns:
        List of top-k covariate names (may be fewer if not enough qualify)

    Example:
        >>> scores = rank_covariates(data, covariates, 'X', 'Y')
        >>> top_3 = select_top_k_instruments(scores, k=3, require_passes_ehs=True)
        >>> # Use top_3 for multi-instrument bounds aggregation
    """
    df = scores_df.copy()

    # Filter by full EHS (includes instrument relevance test_ia)
    if require_full_ehs:
        if 'passes_full_ehs' in df.columns:
            if df['passes_full_ehs'].dtype == float:
                df = df[df['passes_full_ehs'] >= min_pass_rate]
            else:
                df = df[df['passes_full_ehs'] == True]
    # Filter by original EHS (test_i, test_ii only)
    elif require_passes_ehs:
        if 'passes_ehs' in df.columns:
            if df['passes_ehs'].dtype == float:
                df = df[df['passes_ehs'] >= min_pass_rate]
            else:
                df = df[df['passes_ehs'] == True]

    # Exclude weak instruments
    if exclude_weak_instruments and 'weak_instrument_warning' in df.columns:
        df = df[df['weak_instrument_warning'] == False]

    if len(df) == 0:
        return []

    # Get top-k
    if 'z_a' in df.columns:
        return df.head(k)['z_a'].tolist()
    else:
        return df.head(k).index.tolist()


def get_instrument_bounds_for_aggregation(
    training_data: Dict[str, pd.DataFrame],
    instruments: List[str],
    treatment: str,
    outcome: str,
    other_covariates: List[str],
    epsilon: float = 0.1,
    regime_col: str = 'F'
) -> Dict[str, Dict[tuple, tuple]]:
    """
    Compute bounds for each instrument separately for aggregation.

    Args:
        training_data: Dict[site_id -> DataFrame]
        instruments: List of instrument covariate names
        treatment: Treatment column name
        outcome: Outcome column name
        other_covariates: Non-instrument covariates (define strata)
        epsilon: Naturalness tolerance
        regime_col: Regime indicator column

    Returns:
        Dict[instrument_name -> {z_value: (lower, upper)}]

    Example:
        >>> top_instruments = select_top_k_instruments(scores, k=3)
        >>> bounds_per_inst = get_instrument_bounds_for_aggregation(
        ...     training_data, top_instruments, 'X', 'Y', ['age', 'gender']
        ... )
        >>> aggregated = aggregate_across_instruments(bounds_per_inst)
    """
    from .lp_solver import solve_all_bounds_binary_lp

    instrument_bounds = {}

    for instrument in instruments:
        # Use this instrument as the main covariate, others as stratifying
        covariates = [instrument] + other_covariates

        bounds = solve_all_bounds_binary_lp(
            training_data,
            covariates=covariates,
            treatment=treatment,
            outcome=outcome,
            epsilon=epsilon,
            regime_col=regime_col
        )

        instrument_bounds[instrument] = bounds

    return instrument_bounds


# Test
if __name__ == "__main__":
    from .discretize import discretize_covariates, get_discretized_covariate_names
    from pathlib import Path

    data_path = Path('confounded_datasets/anchoring1/age_beta0.5_seed42.csv')

    if data_path.exists():
        data = pd.read_csv(data_path)
        data = discretize_covariates(data)

        covariates = get_discretized_covariate_names()

        print("Ranking covariates (this may take a minute)...")
        engine = CITestEngine(n_permutations=500, random_seed=42)

        scores = rank_covariates(
            data, covariates,
            treatment='iv', outcome='dv',
            ci_engine=engine
        )

        print("\nCovariate Rankings:")
        print(scores[['z_a', 'test_i_pvalue', 'test_ii_pvalue', 'passes_ehs', 'score']])

        best = select_best_instrument(scores)
        print(f"\nBest instrument: {best}")
    else:
        print(f"Data file not found: {data_path}")
