import os
import tempfile
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_model_path():
    """
    Dynamically determine the correct model path based on the environment
    
    Returns:
        str: The correct path to the models directory
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible paths
    possible_paths = [
        # Local development path (relative to script location)
        os.path.join(current_dir, "models", "default"),
        # Streamlit Cloud path (absolute path style)
        "Code/WalmartSalesPredictionApp/models/default",
        # Alternative local path
        "models/default",
        # Another possible cloud path
        os.path.join("Code", "WalmartSalesPredictionApp", "models", "default")
    ]
    
    # Check which path exists and has model files
    for path in possible_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            # Check if this directory contains any .pkl files
            try:
                files = os.listdir(full_path)
                if any(f.endswith('.pkl') for f in files):
                    return path + "/"
            except (OSError, PermissionError):
                continue
    
    # If no valid path found, return the default and let the error handling deal with it
    return "models/default/"

# Configuration dictionary with all hardcoded values
CONFIG = {
    'PREDICTION_PERIODS': 4,
    'MODEL_FILE_MAP': {
        "Auto ARIMA": "AutoARIMA",
        "Exponential Smoothing (Holt-Winters)": "ExponentialSmoothingHoltWinters"
    },
    'MODEL_FUNC_MAP': {
        "Auto ARIMA": "Auto ARIMA",
        "Exponential Smoothing (Holt-Winters)": "Exponential Smoothing (Holt-Winters)"
    },
    'DEFAULT_MODEL_PATH': get_model_path(),  # Now dynamically determined
    'SUPPORTED_EXTENSIONS': ["pkl"],
    'DEFAULT_ARIMA_ORDER': (1, 1, 1)
}

def recreate_arima_model(params):
    """
    Attempt to recreate an ARIMA model from parameters if pickle loading fails
    
    Args:
        params (dict): Dictionary containing model parameters
        
    Returns:
        ARIMA model or None if recreation fails
    """
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a dictionary")
    
    try:
        order = params.get('order', CONFIG['DEFAULT_ARIMA_ORDER'])
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError("Order must be a tuple of length 3")
        
        model = ARIMA(np.array([0]), order=order)
        return model
    except Exception as e:
        warnings.warn(f"Failed to recreate ARIMA model: {str(e)}")
        return None

def load_default_model(model_type):
    """
    Load default model from models/default/ directory with improved error handling
    
    Args:
        model_type (str): Type of model to load
        
    Returns:
        tuple: (model, error_message) where error_message is None on success
    """
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    if model_type not in CONFIG['MODEL_FILE_MAP']:
        return None, f"Invalid model type: {model_type}"
    
    file_name = CONFIG['MODEL_FILE_MAP'][model_type]
    model_path = f"{CONFIG['DEFAULT_MODEL_PATH']}{file_name}.pkl"
    
    # Debug information - you can remove this later
    print(f"DEBUG: Looking for model at: {model_path}")
    print(f"DEBUG: Model path exists: {os.path.exists(model_path)}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    if not os.path.exists(model_path):
        # Try to find the model file in common locations
        alternative_paths = [
            f"models/default/{file_name}.pkl",
            f"Code/WalmartSalesPredictionApp/models/default/{file_name}.pkl",
            f"./{file_name}.pkl",
            f"./models/{file_name}.pkl"
        ]
        
        found_path = None
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                found_path = alt_path
                model_path = alt_path
                break
        
        if not found_path:
            # List available files for debugging
            try:
                current_files = os.listdir(os.getcwd())
                print(f"DEBUG: Files in current directory: {current_files}")
                if os.path.exists("models"):
                    model_files = os.listdir("models")
                    print(f"DEBUG: Files in models directory: {model_files}")
                    if os.path.exists("models/default"):
                        default_files = os.listdir("models/default")
                        print(f"DEBUG: Files in models/default directory: {default_files}")
            except:
                pass
            
            return None, f"Default model not found. Searched paths: {model_path} and alternatives: {alternative_paths}"
    
    try:
        # First try joblib
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle
            try:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                return model, None
            except Exception as pickle_error:
                # Generic error for model loading issue
                if model_type == "Auto ARIMA" and ("statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error)):
                    return None, "Error loading model. Please check the model file or try another model type."
                # Other errors
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_uploaded_model(uploaded_file, model_type):
    """
    Load model from uploaded file with improved error handling for cross-platform compatibility
    
    Args:
        uploaded_file: Streamlit uploaded file object
        model_type (str): Type of model being uploaded
        
    Returns:
        tuple: (model, error_message) where error_message is None on success
    """
    if not uploaded_file:
        raise ValueError("Uploaded file cannot be None")
    
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # First try joblib
        try:
            model = joblib.load(tmp_path)
            # Clean up temporary file
            os.unlink(tmp_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle
            try:
                with open(tmp_path, 'rb') as file:
                    model = pickle.load(file)
                # Clean up temporary file
                os.unlink(tmp_path)
                return model, None
            except Exception as pickle_error:
                # Generic error message
                if model_type == "Auto ARIMA" and ("statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error)):
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    return None, "Error loading model. Please check the model file or try another model type."
                # Other errors
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    
    except Exception as e:
        # Clean up if error occurs
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass
        return None, f"Invalid model file: {str(e)}. Please check format or retrain."

def predict_next_4_weeks(model, model_type):
    """
    Predict next 4 weeks of sales
    
    Args:
        model: Trained model object
        model_type (str): Type of model
        
    Returns:
        tuple: (predictions, dates, error_message) where error_message is None on success
    """
    if not model:
        raise ValueError("Model cannot be None")
    
    if not model_type:
        raise ValueError("Model type cannot be empty")
    
    # Generate dates for next 4 weeks
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, CONFIG['PREDICTION_PERIODS'] + 1)]
    
    try:
        functional_model_type = CONFIG['MODEL_FUNC_MAP'].get(model_type)
        if not functional_model_type:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if functional_model_type == "Auto ARIMA":
            predictions = model.predict(n_periods=CONFIG['PREDICTION_PERIODS'])
        elif functional_model_type == "Exponential Smoothing (Holt-Winters)":
            predictions = model.forecast(CONFIG['PREDICTION_PERIODS'])
        else:
            raise ValueError(f"Unknown model type: {functional_model_type}")
        
        return predictions, dates, None
    except Exception as e:
        return None, None, f"Error generating predictions: {str(e)}"