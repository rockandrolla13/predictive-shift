"""
Integration Tests for Synthetic Data Experiments

Validates:
1. Synthetic data generation works correctly
2. Ground truth CATEs match empirical estimates
3. Estimator fits on synthetic data
4. Bounds are produced and sensible
5. Coverage is computed correctly
6. Full experiment pipeline runs end-to-end

Usage:
    python experiments_synthetic/integration_test.py
    python experiments_synthetic/integration_test.py --quick
    python experiments_synthetic/integration_test.py --verbose
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_synthetic.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_multi_site_data,
    compute_true_cate,
)

from experiments_synthetic.run_experiment import (
    run_synthetic_experiment,
    run_synthetic_grid,
    adapt_training_data_columns,
)

from causal_grounding import CausalGroundingEstimator


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_data_generation():
    """Test that synthetic data generation works correctly."""
    print("Testing data generation...")
    
    try:
        # Create generator
        config = SyntheticDataConfig(
            n_z_values=3,
            beta=0.3,
            treatment_effect=0.2,
            treatment_z_interaction=0.1,
            n_per_site=200,
            seed=42
        )
        generator = SyntheticDataGenerator(config, n_sites=5)
        print("  - Generator initialized: OK")
        
        # Generate training data
        training_data = generator.generate_training_data()
        assert isinstance(training_data, dict), "Training data should be dict"
        assert len(training_data) == 5, f"Expected 5 sites, got {len(training_data)}"
        print(f"  - Generated {len(training_data)} sites: OK")
        
        # Check site data structure
        site_df = training_data['site_0']
        required_cols = ['X', 'Y', 'Z', 'F']
        for col in required_cols:
            assert col in site_df.columns, f"Missing column: {col}"
        print(f"  - Site data has required columns: OK")
        
        # Check regime indicator
        assert set(site_df['F'].unique()) == {'on', 'idle'}, "F should have 'on' and 'idle'"
        print("  - Regime indicator correct: OK")
        
        # Check data types
        assert site_df['X'].isin([0, 1]).all(), "X should be binary"
        assert site_df['Y'].isin([0, 1]).all(), "Y should be binary"
        assert site_df['Z'].isin(range(config.n_z_values)).all(), "Z should be discrete"
        print("  - Data types correct: OK")
        
        print("  Data generation test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_ground_truth_cates():
    """Test that ground truth CATEs are correctly computed."""
    print("\nTesting ground truth CATE computation...")
    
    try:
        # Use moderate parameters to avoid probability clipping at boundaries
        config = SyntheticDataConfig(
            n_z_values=3,
            base_outcome_prob=0.3,
            treatment_effect=0.15,  # Moderate effect
            treatment_z_interaction=0.05,  # Small interaction to avoid boundary clipping
            z_direct_effect=0.05,
            gamma=0.1,  # Lower confounding on outcome
            n_per_site=1000,
            rct_fraction=0.5,
            seed=42
        )
        generator = SyntheticDataGenerator(config, n_sites=1)
        
        # Get analytic CATEs
        true_cates = generator.get_true_cates()
        print(f"  - True CATEs: {true_cates}")
        
        # Generate large RCT sample
        rct_data = generator.generate_rct_data(n_samples=100000)
        
        # Compute empirical CATEs
        empirical_cates = {}
        for z in range(config.n_z_values):
            z_data = rct_data[rct_data['Z'] == z]
            y1 = z_data[z_data['X'] == 1]['Y'].mean()
            y0 = z_data[z_data['X'] == 0]['Y'].mean()
            empirical_cates[(z,)] = y1 - y0
        print(f"  - Empirical CATEs: {empirical_cates}")
        
        # Compare with reasonable tolerance (accounting for sampling variance)
        # SE for difference of proportions ~ sqrt(2*0.5*0.5/n_per_arm) ~ 0.02 for n=2500
        tolerance = 0.03
        for z_tuple in true_cates:
            true_val = true_cates[z_tuple]
            emp_val = empirical_cates[z_tuple]
            diff = abs(true_val - emp_val)
            assert diff < tolerance, f"CATE mismatch for Z={z_tuple}: true={true_val:.3f}, emp={emp_val:.3f}, diff={diff:.3f}"
        print(f"  - CATEs match (tolerance < {tolerance}): OK")
        
        # Check ATE
        true_ate = generator.get_true_ate()
        emp_ate = rct_data[rct_data['X'] == 1]['Y'].mean() - rct_data[rct_data['X'] == 0]['Y'].mean()
        assert abs(true_ate - emp_ate) < tolerance, f"ATE mismatch: true={true_ate:.3f}, emp={emp_ate:.3f}"
        print(f"  - ATE matches: true={true_ate:.3f}, emp={emp_ate:.3f}: OK")
        
        print("  Ground truth test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_data_adaptation():
    """Test that data adaptation for estimator works."""
    print("\nTesting data adaptation...")
    
    try:
        # Generate data
        config = SyntheticDataConfig(n_z_values=3, seed=42)
        generator = SyntheticDataGenerator(config, n_sites=3)
        training_data = generator.generate_training_data()
        
        # Adapt data
        adapted = adapt_training_data_columns(training_data)
        
        # Check adapted structure
        for site_id, df in adapted.items():
            assert 'iv' in df.columns, "Missing 'iv' column"
            assert 'dv' in df.columns, "Missing 'dv' column"
            assert 'Z_cat' in df.columns, "Missing 'Z_cat' column"
            assert df['iv'].equals(df['X']), "'iv' should equal 'X'"
            assert df['dv'].equals(df['Y']), "'dv' should equal 'Y'"
        print("  - Data adapted correctly: OK")
        
        print("  Data adaptation test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_estimator_fit():
    """Test that estimator fits on synthetic data."""
    print("\nTesting estimator fit...")
    
    try:
        # Generate data
        config = SyntheticDataConfig(
            n_z_values=3,
            beta=0.3,
            n_per_site=300,
            seed=42
        )
        generator = SyntheticDataGenerator(config, n_sites=5)
        training_data = generator.generate_training_data()
        
        # Adapt for estimator
        adapted = adapt_training_data_columns(training_data)
        
        # Initialize estimator
        estimator = CausalGroundingEstimator(
            epsilon=0.1,
            n_permutations=50,  # Few permutations for speed
            discretize=False,
            random_seed=42,
            verbose=False
        )
        print("  - Estimator initialized: OK")
        
        # Fit
        start = time.time()
        estimator.fit(adapted, treatment='iv', outcome='dv', covariates=['Z_cat'])
        fit_time = time.time() - start
        print(f"  - Fit completed in {fit_time:.2f}s: OK")
        
        # Check fitted state
        assert estimator.is_fitted_, "Estimator should be fitted"
        assert estimator.training_bounds_ is not None, "Training bounds should exist"
        assert estimator.transferred_bounds_ is not None, "Transferred bounds should exist"
        print("  - Fitted attributes present: OK")
        
        print("  Estimator fit test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_bounds_prediction():
    """Test that bounds prediction works and is sensible."""
    print("\nTesting bounds prediction...")
    
    try:
        # Generate and fit
        config = SyntheticDataConfig(
            n_z_values=3,
            beta=0.3,
            treatment_effect=0.2,
            treatment_z_interaction=0.1,
            n_per_site=300,
            seed=42
        )
        generator = SyntheticDataGenerator(config, n_sites=5)
        training_data = generator.generate_training_data()
        adapted = adapt_training_data_columns(training_data)
        
        estimator = CausalGroundingEstimator(
            epsilon=0.1,
            n_permutations=50,
            discretize=False,
            random_seed=42,
            verbose=False
        )
        estimator.fit(adapted, treatment='iv', outcome='dv', covariates=['Z_cat'])
        
        # Predict bounds
        bounds_df = estimator.predict_bounds()
        
        # Check structure
        assert isinstance(bounds_df, pd.DataFrame), "Should return DataFrame"
        assert 'cate_lower' in bounds_df.columns, "Missing cate_lower"
        assert 'cate_upper' in bounds_df.columns, "Missing cate_upper"
        assert 'width' in bounds_df.columns, "Missing width"
        print(f"  - Bounds DataFrame with {len(bounds_df)} rows: OK")
        
        # Check bounds are finite
        assert bounds_df['cate_lower'].notna().all(), "Lower bounds contain NaN"
        assert bounds_df['cate_upper'].notna().all(), "Upper bounds contain NaN"
        assert np.isfinite(bounds_df['cate_lower']).all(), "Lower bounds contain inf"
        assert np.isfinite(bounds_df['cate_upper']).all(), "Upper bounds contain inf"
        print("  - Bounds are finite: OK")
        
        # Check lower <= upper
        assert (bounds_df['cate_lower'] <= bounds_df['cate_upper']).all(), "Lower > Upper found"
        print("  - Lower <= Upper: OK")
        
        # Check width is positive
        assert (bounds_df['width'] >= 0).all(), "Negative width found"
        print("  - Width >= 0: OK")
        
        # Bounds should be reasonable (not too extreme)
        assert bounds_df['cate_lower'].min() > -2, "Lower bound too extreme"
        assert bounds_df['cate_upper'].max() < 2, "Upper bound too extreme"
        print("  - Bounds within reasonable range: OK")
        
        print("  Bounds prediction test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_coverage_computation():
    """Test that coverage is computed correctly."""
    print("\nTesting coverage computation...")
    
    try:
        # Run experiment
        result = run_synthetic_experiment(
            n_sites=10,
            n_per_site=300,
            n_z_values=3,
            beta=0.0,  # No confounding - should have good coverage
            treatment_effect=0.2,
            treatment_z_interaction=0.1,
            epsilon=0.15,  # Wider bounds for coverage
            n_permutations=50,
            random_seed=42,
            verbose=False
        )
        
        assert 'error' not in result, f"Experiment failed: {result.get('error')}"
        
        metrics = result['metrics']
        
        # Check coverage rate is computed
        assert 'ate_covered' in metrics, "Missing ate_covered"
        assert 'cate_coverage_rate' in metrics, "Missing cate_coverage_rate"
        print(f"  - Coverage metrics present: OK")
        
        # With no confounding and reasonable epsilon, should have good coverage
        print(f"  - ATE covered: {metrics['ate_covered']}")
        print(f"  - CATE coverage rate: {metrics['cate_coverage_rate']:.2%}")
        
        # Check per-stratum results
        assert 'per_stratum_results' in result, "Missing per_stratum_results"
        assert len(result['per_stratum_results']) > 0, "No per-stratum results"
        print(f"  - Per-stratum results: {len(result['per_stratum_results'])} strata: OK")
        
        # Verify coverage computation
        per_stratum = result['per_stratum_results']
        n_covered = sum(1 for r in per_stratum if r['is_covered'])
        expected_rate = n_covered / len(per_stratum)
        assert abs(metrics['cate_coverage_rate'] - expected_rate) < 0.01, "Coverage rate mismatch"
        print("  - Coverage rate verified: OK")
        
        print("  Coverage computation test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_grid_experiment():
    """Test grid experiment runs end-to-end."""
    print("\nTesting grid experiment...")
    
    try:
        # Run small grid
        results_df = run_synthetic_grid(
            betas=[0.0, 0.3],
            epsilons=[0.1],
            n_sites_list=[5],
            n_z_values=3,
            n_per_site=200,
            n_permutations=50,
            random_seed=42,
            output_dir='/tmp/test_synthetic_grid',
            verbose=False
        )
        
        # Check results
        assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
        assert len(results_df) == 2, f"Expected 2 results, got {len(results_df)}"
        print(f"  - Grid returned {len(results_df)} results: OK")
        
        # Check required columns
        required_cols = ['beta', 'ate_covered', 'cate_coverage_rate', 'mean_width']
        for col in required_cols:
            assert col in results_df.columns, f"Missing column: {col}"
        print("  - Required columns present: OK")
        
        print("  Grid experiment test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_confounding_effect():
    """Test that confounding actually biases naive estimates."""
    print("\nTesting confounding effect...")
    
    try:
        # Generate data with strong confounding
        config = SyntheticDataConfig(
            n_z_values=3,
            beta=0.5,  # Strong confounding
            gamma=0.3,
            treatment_effect=0.2,
            n_per_site=2000,
            seed=42
        )
        generator = SyntheticDataGenerator(config, n_sites=1)
        training_data = generator.generate_training_data()
        site_data = training_data['site_0']
        
        # Compute naive estimate from observational data only
        obs_data = site_data[site_data['F'] == 'idle']
        naive_effect = obs_data[obs_data['X'] == 1]['Y'].mean() - obs_data[obs_data['X'] == 0]['Y'].mean()
        
        # Compute RCT estimate
        rct_data = site_data[site_data['F'] == 'on']
        rct_effect = rct_data[rct_data['X'] == 1]['Y'].mean() - rct_data[rct_data['X'] == 0]['Y'].mean()
        
        true_ate = generator.get_true_ate()
        
        print(f"  - True ATE: {true_ate:.3f}")
        print(f"  - RCT estimate: {rct_effect:.3f}")
        print(f"  - Naive (obs) estimate: {naive_effect:.3f}")
        
        # Naive should be biased upward (U -> X positively, U -> Y positively)
        naive_bias = abs(naive_effect - true_ate)
        rct_bias = abs(rct_effect - true_ate)
        
        # Naive bias should be larger than RCT bias
        assert naive_bias > rct_bias, f"Expected naive bias ({naive_bias:.3f}) > RCT bias ({rct_bias:.3f})"
        print(f"  - Naive bias ({naive_bias:.3f}) > RCT bias ({rct_bias:.3f}): OK")
        
        print("  Confounding effect test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_full_pipeline():
    """Test the complete experiment pipeline."""
    print("\nTesting full pipeline...")
    
    try:
        start = time.time()
        
        # Run single experiment
        result = run_synthetic_experiment(
            n_sites=10,
            n_per_site=300,
            n_z_values=3,
            beta=0.3,
            treatment_effect=0.2,
            treatment_z_interaction=0.1,
            epsilon=0.1,
            n_permutations=100,
            random_seed=42,
            verbose=False
        )
        
        elapsed = time.time() - start
        
        assert 'error' not in result, f"Pipeline failed: {result.get('error')}"
        
        # Check all expected keys
        expected_keys = ['config', 'bounds', 'metrics', 'true_cates', 'per_stratum_results']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        print(f"  - Pipeline completed in {elapsed:.2f}s: OK")
        
        # Check metrics
        metrics = result['metrics']
        assert metrics['n_training_sites'] == 10, "Wrong number of sites"
        assert metrics['n_strata'] > 0, "No strata computed"
        print(f"  - {metrics['n_training_sites']} sites, {metrics['n_strata']} strata: OK")
        
        # Check ground truth available
        assert len(result['true_cates']) == 3, "Wrong number of true CATEs"
        print("  - Ground truth CATEs available: OK")
        
        print("  Full pipeline test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests(quick: bool = False):
    """Run all integration tests."""
    print("=" * 60)
    print("SYNTHETIC DATA INTEGRATION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Core tests
    results['data_generation'] = test_data_generation()
    results['ground_truth_cates'] = test_ground_truth_cates()
    results['data_adaptation'] = test_data_adaptation()
    results['estimator_fit'] = test_estimator_fit()
    results['bounds_prediction'] = test_bounds_prediction()
    results['coverage_computation'] = test_coverage_computation()
    
    if not quick:
        results['confounding_effect'] = test_confounding_effect()
        results['grid_experiment'] = test_grid_experiment()
        results['full_pipeline'] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
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


def main():
    parser = argparse.ArgumentParser(
        description='Integration tests for synthetic data experiments'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset of tests')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    success = run_all_tests(quick=args.quick)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
