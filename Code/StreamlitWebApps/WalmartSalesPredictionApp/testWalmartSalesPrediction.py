import pytest
import os
import tempfile
import joblib
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from walmartSalesPredictionCore import (
    recreate_arima_model,
    load_default_model,
    load_uploaded_model,
    predict_next_4_weeks,
    CONFIG
)

class TestWalmartSalesPrediction:
    
    def test_recreate_arima_model_valid_params(self):
        """Test ARIMA model recreation with valid parameters"""
        params = {'order': (1, 1, 1)}
        model = recreate_arima_model(params)
        assert model is not None
    
    def test_recreate_arima_model_invalid_params(self):
        """Test ARIMA model recreation with invalid parameters"""
        with pytest.raises(ValueError):
            recreate_arima_model("invalid")
        
        # Test with invalid order
        params = {'order': (1, 2)}  # Wrong length
        model = recreate_arima_model(params)
        assert model is None
    
    def test_load_default_model_invalid_type(self):
        """Test loading default model with invalid type"""
        with pytest.raises(ValueError):
            load_default_model("")
        
        model, error = load_default_model("Invalid Model")
        assert model is None
        assert "Invalid model type" in error
    
    @patch('os.path.exists')
    def test_load_default_model_file_not_found(self, mock_exists):
        """Test loading default model when file doesn't exist"""
        mock_exists.return_value = False
        model, error = load_default_model("Auto ARIMA")
        assert model is None
        assert "Default model not found" in error
    
    @patch('os.path.exists')
    @patch('joblib.load')
    def test_load_default_model_success(self, mock_joblib_load, mock_exists):
        """Test successful default model loading"""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        model, error = load_default_model("Auto ARIMA")
        assert model == mock_model
        assert error is None
    
    def test_load_uploaded_model_invalid_input(self):
        """Test loading uploaded model with invalid inputs"""
        with pytest.raises(ValueError):
            load_uploaded_model(None, "Auto ARIMA")
        
        with pytest.raises(ValueError):
            load_uploaded_model(Mock(), "")
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('joblib.load')
    @patch('os.unlink')
    def test_load_uploaded_model_success(self, mock_unlink, mock_joblib_load, mock_temp):
        """Test successful uploaded model loading"""
        mock_file = Mock()
        mock_file.getvalue.return_value = b"test data"
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test"
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        model, error = load_uploaded_model(mock_file, "Auto ARIMA")
        assert model == mock_model
        assert error is None
    
    def test_predict_next_4_weeks_invalid_input(self):
        """Test prediction with invalid inputs"""
        with pytest.raises(ValueError):
            predict_next_4_weeks(None, "Auto ARIMA")
        
        with pytest.raises(ValueError):
            predict_next_4_weeks(Mock(), "")
    
    def test_predict_next_4_weeks_arima_success(self):
        """Test successful ARIMA prediction"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        
        predictions, dates, error = predict_next_4_weeks(mock_model, "Auto ARIMA")
        
        assert predictions is not None
        assert len(predictions) == CONFIG['PREDICTION_PERIODS']
        assert dates is not None
        assert len(dates) == CONFIG['PREDICTION_PERIODS']
        assert error is None
        mock_model.predict.assert_called_once_with(n_periods=CONFIG['PREDICTION_PERIODS'])
    
    def test_predict_next_4_weeks_exponential_smoothing_success(self):
        """Test successful Exponential Smoothing prediction"""
        mock_model = Mock()
        mock_model.forecast.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        
        predictions, dates, error = predict_next_4_weeks(mock_model, "Exponential Smoothing (Holt-Winters)")
        
        assert predictions is not None
        assert len(predictions) == CONFIG['PREDICTION_PERIODS']
        assert dates is not None
        assert len(dates) == CONFIG['PREDICTION_PERIODS']
        assert error is None
        mock_model.forecast.assert_called_once_with(CONFIG['PREDICTION_PERIODS'])
    
    def test_predict_next_4_weeks_unknown_model_type(self):
        """Test prediction with unknown model type"""
        mock_model = Mock()
        predictions, dates, error = predict_next_4_weeks(mock_model, "Unknown Model")
        
        assert predictions is None
        assert dates is None
        assert "Unknown model type" in error
    
    def test_error_handling_prediction_failure(self):
        """Test error handling when prediction fails"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        predictions, dates, error = predict_next_4_weeks(mock_model, "Auto ARIMA")
        
        assert predictions is None
        assert dates is None
        assert "Error generating predictions" in error

# Pytest automation setup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])