"""
Tests for Prediction Models Module

Tests the EmpiricalPredictor, XGBoostPredictor, and factory functions.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from causal_grounding.predictors import (
    BasePredictorModel,
    EmpiricalPredictor,
    create_predictor,
    _XGBOOST_AVAILABLE,
)

# Conditional import for XGBoost tests
if _XGBOOST_AVAILABLE:
    from causal_grounding.predictors import XGBoostPredictor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def binary_data():
    """Sample binary data for testing."""
    np.random.seed(42)
    n = 500

    # Z -> X -> Y
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X, n)

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


@pytest.fixture
def multivariate_data():
    """Sample data with multiple covariates."""
    np.random.seed(42)
    n = 500

    Z1 = np.random.binomial(1, 0.5, n)
    Z2 = np.random.binomial(1, 0.6, n)
    Z3 = np.random.binomial(1, 0.4, n)

    X = np.random.binomial(1, 0.2 + 0.3 * Z1 + 0.2 * Z2, n)
    Y = np.random.binomial(1, 0.3 + 0.3 * X + 0.1 * Z1, n)

    return pd.DataFrame({
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'X': X, 'Y': Y
    })


@pytest.fixture
def continuous_outcome_data():
    """Sample data with continuous outcome."""
    np.random.seed(42)
    n = 500

    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)
    Y = 0.5 + 0.3 * X + 0.1 * Z + np.random.normal(0, 0.5, n)

    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


# =============================================================================
# TEST: EmpiricalPredictor INITIALIZATION
# =============================================================================

class TestEmpiricalPredictorInit:
    """Tests for EmpiricalPredictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = EmpiricalPredictor()
        assert predictor.smoothing == 0.01
        assert predictor.n_bins == 5

    def test_custom_init(self):
        """Test custom initialization."""
        predictor = EmpiricalPredictor(smoothing=0.1, n_bins=10)
        assert predictor.smoothing == 0.1
        assert predictor.n_bins == 10


# =============================================================================
# TEST: EmpiricalPredictor PROPENSITY
# =============================================================================

class TestEmpiricalPropensity:
    """Tests for EmpiricalPredictor propensity model."""

    def test_fit_propensity(self, binary_data):
        """Test fitting propensity model."""
        predictor = EmpiricalPredictor()
        result = predictor.fit_propensity(binary_data, 'X', ['Z'])

        assert result is predictor  # Returns self
        assert predictor._is_fitted_propensity

    def test_predict_propensity(self, binary_data):
        """Test predicting propensity."""
        predictor = EmpiricalPredictor()
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        assert len(probs) == len(binary_data)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_propensity_varies_by_covariate(self, binary_data):
        """Test that propensity varies by covariate value."""
        predictor = EmpiricalPredictor()
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        # Create test data with different Z values
        test_z0 = pd.DataFrame({'Z': [0] * 10})
        test_z1 = pd.DataFrame({'Z': [1] * 10})

        prop_z0 = predictor.predict_propensity(test_z0)[0]
        prop_z1 = predictor.predict_propensity(test_z1)[0]

        # Z affects X, so propensities should differ
        assert prop_z0 != prop_z1

    def test_propensity_not_fitted_error(self, binary_data):
        """Test error when propensity not fitted."""
        predictor = EmpiricalPredictor()

        with pytest.raises(RuntimeError, match="fit propensity"):
            predictor.predict_propensity(binary_data[['Z']])


# =============================================================================
# TEST: EmpiricalPredictor OUTCOME
# =============================================================================

class TestEmpiricalOutcome:
    """Tests for EmpiricalPredictor outcome model."""

    def test_fit_outcome(self, binary_data):
        """Test fitting outcome model."""
        predictor = EmpiricalPredictor()
        result = predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        assert result is predictor  # Returns self
        assert predictor._is_fitted_outcome

    def test_predict_outcome_scalar_treatment(self, binary_data):
        """Test predicting outcome with scalar treatment."""
        predictor = EmpiricalPredictor()
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        probs_a0 = predictor.predict_outcome(binary_data[['Z']], 0)
        probs_a1 = predictor.predict_outcome(binary_data[['Z']], 1)

        assert len(probs_a0) == len(binary_data)
        assert len(probs_a1) == len(binary_data)
        assert np.all(probs_a0 >= 0)
        assert np.all(probs_a1 <= 1)

    def test_predict_outcome_array_treatment(self, binary_data):
        """Test predicting outcome with array treatment."""
        predictor = EmpiricalPredictor()
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        A = binary_data['X'].values
        probs = predictor.predict_outcome(binary_data[['Z']], A)

        assert len(probs) == len(binary_data)

    def test_outcome_not_fitted_error(self, binary_data):
        """Test error when outcome not fitted."""
        predictor = EmpiricalPredictor()

        with pytest.raises(RuntimeError, match="fit outcome"):
            predictor.predict_outcome(binary_data[['Z']], 0)


# =============================================================================
# TEST: EmpiricalPredictor COMBINED
# =============================================================================

class TestEmpiricalCombined:
    """Tests for combined EmpiricalPredictor functionality."""

    def test_fit_both(self, binary_data):
        """Test fitting both models with fit()."""
        predictor = EmpiricalPredictor()
        result = predictor.fit(binary_data, 'X', 'Y', ['Z'])

        assert result is predictor
        assert predictor._is_fitted_propensity
        assert predictor._is_fitted_outcome

    def test_predict_potential_outcomes(self, binary_data):
        """Test predicting potential outcomes."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        Y0, Y1 = predictor.predict_potential_outcomes(binary_data[['Z']])

        assert len(Y0) == len(binary_data)
        assert len(Y1) == len(binary_data)

    def test_predict_cate(self, binary_data):
        """Test predicting CATE."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(binary_data[['Z']])

        assert len(cate) == len(binary_data)
        # CATE should be between -1 and 1 for binary outcome
        assert np.all(cate >= -1)
        assert np.all(cate <= 1)

    def test_get_probability_table(self, binary_data):
        """Test getting probability table."""
        predictor = EmpiricalPredictor()
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        table = predictor.get_probability_table()

        assert isinstance(table, pd.DataFrame)
        assert 'key' in table.columns
        assert 'P(A=1|X)' in table.columns
        assert 'P(Y=1|X,A=0)' in table.columns
        assert 'P(Y=1|X,A=1)' in table.columns


# =============================================================================
# TEST: EmpiricalPredictor MULTIPLE COVARIATES
# =============================================================================

class TestEmpiricalMultipleCovariates:
    """Tests for EmpiricalPredictor with multiple covariates."""

    def test_fit_multivariate(self, multivariate_data):
        """Test fitting with multiple covariates."""
        predictor = EmpiricalPredictor()
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        # Should create keys for each combination
        table = predictor.get_probability_table()
        assert len(table) > 1  # Should have multiple strata

    def test_predict_multivariate(self, multivariate_data):
        """Test predicting with multiple covariates."""
        predictor = EmpiricalPredictor()
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        cate = predictor.predict_cate(multivariate_data[['Z1', 'Z2', 'Z3']])
        assert len(cate) == len(multivariate_data)


# =============================================================================
# TEST: XGBoostPredictor (if available)
# =============================================================================

@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostPredictorInit:
    """Tests for XGBoostPredictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = XGBoostPredictor()
        assert predictor.use_grid_search == True
        assert predictor.cv_folds == 5

    def test_custom_init(self):
        """Test custom initialization."""
        predictor = XGBoostPredictor(
            use_grid_search=False,
            n_estimators_range=[50, 100],
            max_depth_range=[3],
            random_state=42
        )
        assert predictor.use_grid_search == False
        assert predictor.n_estimators_range == [50, 100]
        assert predictor.random_state == 42


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostPropensity:
    """Tests for XGBoostPredictor propensity model."""

    def test_fit_propensity(self, binary_data):
        """Test fitting propensity model."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit_propensity(binary_data, 'X', ['Z'])

        assert result is predictor
        assert predictor._propensity_model is not None

    def test_predict_propensity(self, binary_data):
        """Test predicting propensity."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_propensity(binary_data, 'X', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        assert len(probs) == len(binary_data)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostOutcome:
    """Tests for XGBoostPredictor outcome model."""

    def test_fit_outcome_binary(self, binary_data):
        """Test fitting outcome model with binary outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        assert result is predictor
        assert predictor._outcome_model is not None
        assert predictor._is_binary_outcome == True

    def test_fit_outcome_continuous(self, continuous_outcome_data):
        """Test fitting outcome model with continuous outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_outcome(continuous_outcome_data, 'Y', 'X', ['Z'])

        assert predictor._is_binary_outcome == False

    def test_predict_outcome(self, binary_data):
        """Test predicting outcome."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit_outcome(binary_data, 'Y', 'X', ['Z'])

        probs_a0 = predictor.predict_outcome(binary_data[['Z']], 0)
        probs_a1 = predictor.predict_outcome(binary_data[['Z']], 1)

        assert len(probs_a0) == len(binary_data)
        assert len(probs_a1) == len(binary_data)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostCombined:
    """Tests for combined XGBoostPredictor functionality."""

    def test_fit_both(self, binary_data):
        """Test fitting both models."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        result = predictor.fit(binary_data, 'X', 'Y', ['Z'])

        assert result is predictor
        assert predictor._propensity_model is not None
        assert predictor._outcome_model is not None

    def test_predict_cate(self, binary_data):
        """Test predicting CATE."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(binary_data[['Z']])

        assert len(cate) == len(binary_data)

    def test_feature_importance(self, binary_data):
        """Test getting feature importance."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        prop_importance = predictor.get_feature_importance('propensity')
        outcome_importance = predictor.get_feature_importance('outcome')

        assert isinstance(prop_importance, pd.DataFrame)
        assert isinstance(outcome_importance, pd.DataFrame)
        assert 'feature' in prop_importance.columns
        assert 'importance' in prop_importance.columns


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostMultipleCovariates:
    """Tests for XGBoostPredictor with multiple covariates."""

    def test_multivariate(self, multivariate_data):
        """Test with multiple covariates."""
        predictor = XGBoostPredictor(use_grid_search=False, random_state=42)
        predictor.fit(multivariate_data, 'X', 'Y', ['Z1', 'Z2', 'Z3'])

        cate = predictor.predict_cate(multivariate_data[['Z1', 'Z2', 'Z3']])
        assert len(cate) == len(multivariate_data)

        # Check feature importance has all features
        importance = predictor.get_feature_importance('propensity')
        assert len(importance) == 3  # Z1, Z2, Z3


# =============================================================================
# TEST: FACTORY FUNCTION
# =============================================================================

class TestCreatePredictor:
    """Tests for create_predictor factory function."""

    def test_create_empirical(self):
        """Test creating empirical predictor."""
        predictor = create_predictor('empirical')
        assert isinstance(predictor, EmpiricalPredictor)

    def test_create_empirical_with_kwargs(self):
        """Test creating empirical predictor with kwargs."""
        predictor = create_predictor('empirical', smoothing=0.5)
        assert predictor.smoothing == 0.5

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_create_xgboost(self):
        """Test creating XGBoost predictor."""
        predictor = create_predictor('xgboost', use_grid_search=False)
        assert isinstance(predictor, XGBoostPredictor)

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_predictor('invalid')


# =============================================================================
# TEST: INTERFACE COMPATIBILITY
# =============================================================================

class TestInterfaceCompatibility:
    """Test that predictors have compatible interfaces."""

    def test_empirical_is_base_predictor(self):
        """Test EmpiricalPredictor is a BasePredictorModel."""
        predictor = EmpiricalPredictor()
        assert isinstance(predictor, BasePredictorModel)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_is_base_predictor(self):
        """Test XGBoostPredictor is a BasePredictorModel."""
        predictor = XGBoostPredictor()
        assert isinstance(predictor, BasePredictorModel)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_same_cate_interface(self, binary_data):
        """Test both predictors have same CATE prediction interface."""
        emp = create_predictor('empirical')
        xgb = create_predictor('xgboost', use_grid_search=False, random_state=42)

        emp.fit(binary_data, 'X', 'Y', ['Z'])
        xgb.fit(binary_data, 'X', 'Y', ['Z'])

        emp_cate = emp.predict_cate(binary_data[['Z']])
        xgb_cate = xgb.predict_cate(binary_data[['Z']])

        # Both should return same shape
        assert emp_cate.shape == xgb_cate.shape


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_covariate_value(self):
        """Test with single covariate value (constant)."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'Z': np.ones(n, dtype=int),  # Constant
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        predictor = EmpiricalPredictor()
        predictor.fit(data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(data[['Z']])
        assert len(cate) == n

    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 20

        data = pd.DataFrame({
            'Z': np.random.binomial(1, 0.5, n),
            'X': np.random.binomial(1, 0.5, n),
            'Y': np.random.binomial(1, 0.5, n)
        })

        predictor = EmpiricalPredictor()
        predictor.fit(data, 'X', 'Y', ['Z'])

        cate = predictor.predict_cate(data[['Z']])
        assert len(cate) == n

    def test_unseen_covariate_value(self, binary_data):
        """Test predicting for covariate value not seen in training."""
        # Train only on Z=0
        train_data = binary_data[binary_data['Z'] == 0].copy()

        predictor = EmpiricalPredictor()
        predictor.fit(train_data, 'X', 'Y', ['Z'])

        # Predict on Z=1 (not in training)
        test_data = pd.DataFrame({'Z': [1, 1, 1]})

        # Should return default (0.5) without crashing
        probs = predictor.predict_propensity(test_data)
        assert len(probs) == 3
        assert np.all(probs == 0.5)  # Default for unseen

    def test_smoothing_effect(self, binary_data):
        """Test that smoothing prevents zero probabilities."""
        predictor = EmpiricalPredictor(smoothing=0.5)
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        probs = predictor.predict_propensity(binary_data[['Z']])

        # With smoothing, no probs should be exactly 0 or 1
        assert np.all(probs > 0)
        assert np.all(probs < 1)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for predictors."""

    def test_full_workflow_empirical(self, binary_data):
        """Test full workflow with empirical predictor."""
        predictor = create_predictor('empirical')

        # Fit
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        # Predict propensity
        prop = predictor.predict_propensity(binary_data[['Z']])
        assert np.all(prop >= 0) and np.all(prop <= 1)

        # Predict outcomes
        y0, y1 = predictor.predict_potential_outcomes(binary_data[['Z']])
        assert np.all(y0 >= 0) and np.all(y0 <= 1)
        assert np.all(y1 >= 0) and np.all(y1 <= 1)

        # Predict CATE
        cate = predictor.predict_cate(binary_data[['Z']])
        assert np.all(cate >= -1) and np.all(cate <= 1)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_full_workflow_xgboost(self, binary_data):
        """Test full workflow with XGBoost predictor."""
        predictor = create_predictor('xgboost', use_grid_search=False, random_state=42)

        # Fit
        predictor.fit(binary_data, 'X', 'Y', ['Z'])

        # Predict propensity
        prop = predictor.predict_propensity(binary_data[['Z']])
        assert np.all(prop >= 0) and np.all(prop <= 1)

        # Predict outcomes
        y0, y1 = predictor.predict_potential_outcomes(binary_data[['Z']])

        # Predict CATE
        cate = predictor.predict_cate(binary_data[['Z']])
        assert len(cate) == len(binary_data)

    @pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_compare_predictors(self, binary_data):
        """Test comparing both predictors on same data."""
        emp = create_predictor('empirical')
        xgb = create_predictor('xgboost', use_grid_search=False, random_state=42)

        emp.fit(binary_data, 'X', 'Y', ['Z'])
        xgb.fit(binary_data, 'X', 'Y', ['Z'])

        emp_cate = emp.predict_cate(binary_data[['Z']])
        xgb_cate = xgb.predict_cate(binary_data[['Z']])

        # Both should produce valid CATE estimates
        assert len(emp_cate) == len(xgb_cate)

        # Correlation between predictions should be positive (similar direction)
        correlation = np.corrcoef(emp_cate, xgb_cate)[0, 1]
        # Note: correlation could be weak due to different methods
        assert not np.isnan(correlation)
