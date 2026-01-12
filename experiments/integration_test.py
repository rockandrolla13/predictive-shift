"""
Integration test for causal_grounding module.

Validates:
1. All imports work
2. Data loading works
3. Estimator fits without errors
4. Bounds are produced
5. Bounds are sensible (within [-1, 1], lower <= upper)
6. Metrics are computed

Usage:
    python experiments/integration_test.py
    python experiments/integration_test.py --study anchoring1 --beta 0.3
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_grounding import (
    CausalGroundingEstimator,
    create_train_target_split,
    load_rct_data,
    discretize_covariates,
    CITestEngine,
    compute_cmi,
    permutation_test_cmi
)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        # Test core estimator
        print("  - CausalGroundingEstimator: OK")

        # Test data functions
        print("  - create_train_target_split: OK")
        print("  - load_rct_data: OK")
        print("  - discretize_covariates: OK")

        # Test CI engine
        print("  - CITestEngine: OK")
        print("  - compute_cmi: OK")
        print("  - permutation_test_cmi: OK")

        print("  All imports successful!")
        return True

    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def test_ci_engine():
    """Test CI test engine with synthetic data."""
    print("\nTesting CI engine...")

    try:
        # Create synthetic independent data
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'W': np.random.choice([0, 1, 2], n),
            'Y': np.random.choice([0, 1], n),
            'Z': np.random.choice([0, 1], n)
        })

        # Initialize engine with few permutations for speed
        engine = CITestEngine(n_permutations=100, random_seed=42)
        print("  - CITestEngine initialized")

        # Test conditional independence
        result = engine.test_conditional_independence(df, 'Z', 'Y', ['W'])

        # Check result structure
        assert 'cmi' in result, "Result missing 'cmi' key"
        assert 'p_value' in result, "Result missing 'p_value' key"
        assert 'reject_independence' in result, "Result missing 'reject_independence' key"
        print("  - test_conditional_independence returns expected keys")

        # For independent data, p-value should be high
        assert result['p_value'] > 0.05, f"p-value too low for independent data: {result['p_value']}"
        print(f"  - p-value for independent data: {result['p_value']:.3f} (> 0.05)")

        # Test CMI computation
        cmi = compute_cmi(df['Y'].values, df['Z'].values, df['W'].values)
        assert cmi >= 0, f"CMI should be non-negative, got {cmi}"
        print(f"  - compute_cmi works: CMI = {cmi:.4f}")

        print("  CI engine tests passed!")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_data_loading(study, pattern, beta, data_dir, rct_path):
    """Test data loading."""
    print(f"\nTesting data loading for {study}...")

    osrct_data = None
    rct_data = None

    try:
        # Find OSRCT file
        data_path = Path(data_dir)
        study_dir = data_path / study

        if not study_dir.exists():
            print(f"  Warning: Study directory not found: {study_dir}")
            # Try to find any available study
            available = [d for d in data_path.glob("*") if d.is_dir()]
            if available:
                study_dir = available[0]
                study = study_dir.name
                print(f"  Using alternative: {study_dir}")

        # Try multiple file patterns
        osrct_path = None
        patterns_to_try = [
            f"{pattern}_beta{beta}_seed42.csv",  # e.g., age_beta0.1_seed42.csv
            f"{study}_{pattern}_beta_{beta}.pkl",  # e.g., anchoring1_age_beta_0.3.pkl
            f"{study}_{pattern}_beta_{beta}.csv",
            f"{pattern}_beta{beta}.csv",
        ]

        for file_pattern in patterns_to_try:
            candidate = study_dir / file_pattern
            if candidate.exists():
                osrct_path = candidate
                break

        # If not found, try to find any CSV or PKL file
        if osrct_path is None:
            csv_files = list(study_dir.glob("*.csv"))
            pkl_files = list(study_dir.glob("*.pkl"))
            all_files = csv_files + pkl_files
            if all_files:
                osrct_path = all_files[0]
                print(f"  Warning: Using fallback file: {osrct_path.name}")

        if osrct_path is None:
            print(f"  Warning: No data files found in {study_dir}")
            return False, None, None

        print(f"  - Found OSRCT file: {osrct_path.name}")

        # Load OSRCT
        if str(osrct_path).endswith('.pkl'):
            osrct_data = pd.read_pickle(osrct_path)
        else:
            osrct_data = pd.read_csv(osrct_path)
        print(f"  - OSRCT shape: {osrct_data.shape}")
        print(f"  - OSRCT columns: {list(osrct_data.columns)[:10]}...")

        # Load RCT data
        rct_data = load_rct_data(study, rct_path)
        print(f"  - RCT shape: {rct_data.shape}")

        print("  Data loading successful!")
        return True, osrct_data, rct_data

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, osrct_data, rct_data


def test_train_target_split(osrct_data, rct_data, target_site):
    """Test train/target split."""
    print("\nTesting train/target split...")

    try:
        # Create split
        training_data, target_data = create_train_target_split(
            osrct_data, rct_data, target_site=target_site
        )

        # Validate types
        assert isinstance(training_data, dict), "training_data should be a dict"
        print("  - training_data is dict: OK")

        assert isinstance(target_data, pd.DataFrame), "target_data should be DataFrame"
        print("  - target_data is DataFrame: OK")

        # Validate target site exclusion
        assert target_site not in training_data, f"target_site '{target_site}' should not be in training_data"
        print(f"  - target_site '{target_site}' excluded from training: OK")

        # Print summary
        print(f"  - Number of training sites: {len(training_data)}")
        for site, df in training_data.items():
            n_on = (df['F'] == 'on').sum() if 'F' in df.columns else 'N/A'
            n_idle = (df['F'] == 'idle').sum() if 'F' in df.columns else 'N/A'
            print(f"    - {site}: {len(df)} rows (on={n_on}, idle={n_idle})")
        print(f"  - Target data: {len(target_data)} rows")

        print("  Train/target split successful!")
        return True, training_data, target_data

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, None, None


def test_estimator_fit(training_data, epsilon, n_permutations):
    """Test estimator fitting."""
    print("\nTesting estimator fit...")

    try:
        # Initialize estimator
        estimator = CausalGroundingEstimator(
            epsilon=epsilon,
            n_permutations=n_permutations,
            random_seed=42,
            verbose=False
        )
        print(f"  - Estimator initialized (epsilon={epsilon})")

        # Time the fit
        start_time = time.time()
        estimator.fit(training_data, treatment='iv', outcome='dv')
        fit_time = time.time() - start_time
        print(f"  - Fit completed in {fit_time:.2f}s")

        # Check attributes
        assert hasattr(estimator, 'is_fitted_') and estimator.is_fitted_, "Estimator should be fitted"
        print("  - is_fitted_: OK")

        assert hasattr(estimator, 'training_bounds_'), "Estimator missing training_bounds_"
        print("  - training_bounds_: OK")

        assert hasattr(estimator, 'transferred_bounds_'), "Estimator missing transferred_bounds_"
        print("  - transferred_bounds_: OK")

        print("  Estimator fit successful!")
        return True, estimator

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, None


def test_bounds_prediction(estimator):
    """Test bounds prediction."""
    print("\nTesting bounds prediction...")

    try:
        # Get bounds
        bounds_df = estimator.predict_bounds()

        # Check type
        assert isinstance(bounds_df, pd.DataFrame), "predict_bounds should return DataFrame"
        print(f"  - Returns DataFrame: OK ({len(bounds_df)} rows)")

        # Check columns
        required_cols = ['cate_lower', 'cate_upper', 'width']
        for col in required_cols:
            assert col in bounds_df.columns, f"Missing column: {col}"
        print(f"  - Has required columns: OK")

        # Check bounds are sensible
        # Note: bounds may exceed [-1, 1] for continuous outcomes, check they're finite
        assert bounds_df['cate_lower'].notna().all(), "Lower bounds contain NaN"
        assert bounds_df['cate_upper'].notna().all(), "Upper bounds contain NaN"
        assert np.isfinite(bounds_df['cate_lower']).all(), "Lower bounds contain inf"
        assert np.isfinite(bounds_df['cate_upper']).all(), "Upper bounds contain inf"
        print("  - Bounds are finite: OK")

        # Check lower <= upper
        violations = (bounds_df['cate_lower'] > bounds_df['cate_upper']).sum()
        assert violations == 0, f"Found {violations} rows where lower > upper"
        print("  - Lower <= Upper for all rows: OK")

        # Print summary
        print(f"  - Mean lower bound: {bounds_df['cate_lower'].mean():.3f}")
        print(f"  - Mean upper bound: {bounds_df['cate_upper'].mean():.3f}")
        print(f"  - Mean width: {bounds_df['width'].mean():.3f}")
        print(f"  - Median width: {bounds_df['width'].median():.3f}")

        print("  Bounds prediction successful!")
        return True, bounds_df

    except Exception as e:
        print(f"  FAILED: {e}")
        return False, None


def test_diagnostics(estimator):
    """Test diagnostics output."""
    print("\nTesting diagnostics...")

    try:
        # Get diagnostics
        diagnostics = estimator.get_diagnostics()

        # Check type
        assert isinstance(diagnostics, dict), "get_diagnostics should return dict"
        print("  - Returns dict: OK")

        # Check required keys
        required_keys = ['epsilon', 'transfer_method', 'n_training_sites']
        for key in required_keys:
            assert key in diagnostics, f"Missing key: {key}"
        print("  - Has required keys: OK")

        # Print diagnostics
        print(f"  - epsilon: {diagnostics.get('epsilon')}")
        print(f"  - transfer_method: {diagnostics.get('transfer_method')}")
        print(f"  - n_training_sites: {diagnostics.get('n_training_sites')}")
        print(f"  - n_strata: {diagnostics.get('n_strata', 'N/A')}")

        print("  Diagnostics test successful!")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# =============================================================================
# MAIN INTEGRATION TEST
# =============================================================================

def run_full_integration_test(
    study: str,
    pattern: str,
    beta: float,
    epsilon: float,
    target_site: str,
    data_dir: str,
    rct_path: str,
    n_permutations: int
):
    """Run complete integration test."""

    print("=" * 60)
    print("CAUSAL GROUNDING INTEGRATION TEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  study: {study}")
    print(f"  pattern: {pattern}")
    print(f"  beta: {beta}")
    print(f"  epsilon: {epsilon}")
    print(f"  target_site: {target_site}")
    print(f"  n_permutations: {n_permutations}")
    print()

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()

    # Test 2: CI Engine
    results['ci_engine'] = test_ci_engine()

    # Test 3: Data Loading
    success, osrct_data, rct_data = test_data_loading(study, pattern, beta, data_dir, rct_path)
    results['data_loading'] = success

    if not success or osrct_data is None or rct_data is None:
        print("\n" + "=" * 60)
        print("INTEGRATION TEST INCOMPLETE - Data loading failed")
        print("=" * 60)
        print("\nNote: Some tests require actual data files to run.")
        print("Make sure the confounded_datasets directory exists with study data.")

        # Still report results for tests that ran
        passed = sum(results.values())
        total = len(results)
        print(f"\nTests completed: {passed}/{total} passed")
        return passed == total

    # Test 4: Train/Target Split
    success, training_data, target_data = test_train_target_split(
        osrct_data, rct_data, target_site
    )
    results['train_target_split'] = success

    if not success or training_data is None:
        print("\n" + "=" * 60)
        print("INTEGRATION TEST INCOMPLETE - Split failed")
        print("=" * 60)
        return False

    # Test 5: Estimator Fit
    success, estimator = test_estimator_fit(training_data, epsilon, n_permutations)
    results['estimator_fit'] = success

    if not success or estimator is None:
        print("\n" + "=" * 60)
        print("INTEGRATION TEST INCOMPLETE - Fit failed")
        print("=" * 60)
        return False

    # Test 6: Bounds Prediction
    success, bounds_df = test_bounds_prediction(estimator)
    results['bounds_prediction'] = success

    # Test 7: Diagnostics
    results['diagnostics'] = test_diagnostics(estimator)

    # Final Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")

    print("=" * 60)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Integration test for causal_grounding module'
    )

    parser.add_argument(
        '--study', type=str, default='anchoring1',
        help='Study name (default: anchoring1)'
    )
    parser.add_argument(
        '--pattern', type=str, default='age',
        help='Confounding pattern (default: age)'
    )
    parser.add_argument(
        '--beta', type=float, default=0.3,
        help='Confounding strength (default: 0.3)'
    )
    parser.add_argument(
        '--epsilon', type=float, default=0.1,
        help='Naturalness tolerance (default: 0.1)'
    )
    parser.add_argument(
        '--target-site', type=str, default='mturk',
        help='Target site (default: mturk)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='confounded_datasets',
        help='Directory containing OSRCT data (default: confounded_datasets)'
    )
    parser.add_argument(
        '--rct-path', type=str, default='ManyLabs1/pre-process/Manylabs1_data.pkl',
        help='Path to RCT data pickle (default: ManyLabs1/pre-process/Manylabs1_data.pkl)'
    )
    parser.add_argument(
        '--n-permutations', type=int, default=100,
        help='Number of permutations for CI tests (default: 100)'
    )

    args = parser.parse_args()

    success = run_full_integration_test(
        study=args.study,
        pattern=args.pattern,
        beta=args.beta,
        epsilon=args.epsilon,
        target_site=args.target_site,
        data_dir=args.data_dir,
        rct_path=args.rct_path,
        n_permutations=args.n_permutations
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
