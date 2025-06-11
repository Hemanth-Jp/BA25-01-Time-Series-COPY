"""
@brief Core module for Walmart sales prediction functionality
@details This module provides the core functionality for loading and using time series models
         to predict sales data. It supports both ARIMA and Exponential Smoothing models
         with fallback mechanisms for cross-platform compatibility.
@author Sales Prediction Team
@date 2025
"""

import os
import tempfile
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_model_path_simple():
    """
    @brief Determine the appropriate model path based on deployment environment
    @details Simple check for Streamlit Cloud vs local deployment by examining file structure
    @return Path string to the models directory
    @note This function helps with cross-platform deployment compatibility
    """
    # Check if we're on Streamlit Cloud by looking for specific environment
    if os.path.exists("Code/WalmartSalesPredictionApp"):
        return "Code/WalmartSalesPredictionApp/models/default/"
    else:
        return "models/default/"

# Configuration dictionary with all hardcoded values
CONFIG = {
    'PREDICTION_PERIODS': 4,  # Number of weeks to predict
    'MODEL_FILE_MAP': {
        "Auto ARIMA": "AutoARIMA",
        "Exponential Smoothing (Holt-Winters)": "ExponentialSmoothingHoltWinters"
    },
    'MODEL_FUNC_MAP': {
        "Auto ARIMA": "Auto ARIMA",
        "Exponential Smoothing (Holt-Winters)": "Exponential Smoothing (Holt-Winters)"
    },
    'DEFAULT_MODEL_PATH': get_model_path_simple(),  # Dynamic path based on environment
    'SUPPORTED_EXTENSIONS': ["pkl"],  # Supported file extensions for model files
    'DEFAULT_ARIMA_ORDER': (1, 1, 1)  # Default ARIMA parameters (p, d, q)
}

def recreate_arima_model(params):
    """
    @brief Attempt to recreate an ARIMA model from parameters if pickle loading fails
    @details This function serves as a fallback mechanism when model deserialization fails
             due to version incompatibilities or missing dependencies
    @param params Dictionary containing model parameters, must include 'order' key
    @return ARIMA model object or None if recreation fails
    @raises ValueError If parameters are not a dictionary or order is invalid
    @warning This creates a dummy model and may not preserve original training data
    """
    # Validate input parameter type
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a dictionary")
    
    try:
        # Extract ARIMA order parameters, use default if not provided
        order = params.get('order', CONFIG['DEFAULT_ARIMA_ORDER'])
        
        # Validate order tuple structure (p, d, q)
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError("Order must be a tuple of length 3")
        
        # Create dummy ARIMA model with specified order
        # Note: This creates a model with dummy data, not the original training data
        model = ARIMA(np.array([0]), order=order)
        return model
    except Exception as e:
        # Log warning and return None on failure
        warnings.warn(f"Failed to recreate ARIMA model: {str(e)}")
        return None

def load_default_model(model_type):
    """
    @brief Load default model from models/default/ directory with improved error handling
    @details Attempts to load pre-trained models using both joblib and pickle for compatibility
    @param model_type Type of model to load (must be in MODEL_FILE_MAP)
    @return Tuple of (model_object, error_message) where error_message is None on success
    @raises ValueError If model_type is empty or invalid
    @note Uses fallback loading mechanisms for cross-platform compatibility
    """
    # Validate input parameter
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    # Check if model type is supported
    if model_type not in CONFIG['MODEL_FILE_MAP']:
        return None, f"Invalid model type: {model_type}"
    
    # Construct file path using mapping
    file_name = CONFIG['MODEL_FILE_MAP'][model_type]
    model_path = f"{CONFIG['DEFAULT_MODEL_PATH']}{file_name}.pkl"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        return None, f"Default model not found at {model_path}"
    
    try:
        # First try joblib loading (preferred method)
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle as fallback
            try:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                return model, None
            except Exception as pickle_error:
                # Handle specific statsmodels compatibility issues
                if model_type == "Auto ARIMA" and ("statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error)):
                    return None, "Error loading model. Please check the model file or try another model type."
                # Raise generic error for other issues
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_uploaded_model(uploaded_file, model_type):
    """
    @brief Load model from uploaded file with improved error handling for cross-platform compatibility
    @details Saves uploaded file to temporary location and attempts loading with multiple methods
    @param uploaded_file Streamlit uploaded file object containing model data
    @param model_type Type of model being uploaded (for error handling context)
    @return Tuple of (model_object, error_message) where error_message is None on success
    @raises ValueError If uploaded_file is None or model_type is empty
    @note Automatically cleans up temporary files after processing
    """
    # Validate input parameters
    if not uploaded_file:
        raise ValueError("Uploaded file cannot be None")
    
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    tmp_path = None
    try:
        # Save uploaded file to temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # First try joblib loading (preferred method)
        try:
            model = joblib.load(tmp_path)
            # Clean up temporary file on success
            os.unlink(tmp_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle as fallback
            try:
                with open(tmp_path, 'rb') as file:
                    model = pickle.load(file)
                # Clean up temporary file on success
                os.unlink(tmp_path)
                return model, None
            except Exception as pickle_error:
                # Handle specific statsmodels compatibility issues
                if model_type == "Auto ARIMA" and ("statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error)):
                    # Clean up temporary file before returning error
                    os.unlink(tmp_path)
                    return None, "Error loading model. Please check the model file or try another model type."
                # Raise generic error for other loading failures
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    
    except Exception as e:
        # Clean up temporary file if any error occurs
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors
        return None, f"Invalid model file: {str(e)}. Please check format or retrain."

def predict_next_4_weeks(model, model_type):
    """
    @brief Predict next 4 weeks of sales using the provided model
    @details Generates predictions based on model type and creates corresponding date ranges
    @param model Trained model object (ARIMA or Exponential Smoothing)
    @param model_type Type of model being used for prediction routing
    @return Tuple of (predictions_array, dates_list, error_message) where error_message is None on success
    @raises ValueError If model is None or model_type is empty
    @note Dates are generated starting from next week after current date
    """
    # Validate input parameters
    if not model:
        raise ValueError("Model cannot be None")
    
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    # Generate dates for next 4 weeks starting from next week
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, CONFIG['PREDICTION_PERIODS'] + 1)]
    
    try:
        # Map model type to functional type for prediction routing
        functional_model_type = CONFIG['MODEL_FUNC_MAP'].get(model_type)
        if not functional_model_type:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Route to appropriate prediction method based on model type
        if functional_model_type == "Auto ARIMA":
            # Use ARIMA-specific prediction method
            predictions = model.predict(n_periods=CONFIG['PREDICTION_PERIODS'])
        elif functional_model_type == "Exponential Smoothing (Holt-Winters)":
            # Use Exponential Smoothing-specific forecast method
            predictions = model.forecast(CONFIG['PREDICTION_PERIODS'])
        else:
            # Unknown model type error
            raise ValueError(f"Unknown model type: {functional_model_type}")
        
        return predictions, dates, None
    except Exception as e:
        # Return error information on prediction failure
        return None, None, f"Error generating predictions: {str(e)}"