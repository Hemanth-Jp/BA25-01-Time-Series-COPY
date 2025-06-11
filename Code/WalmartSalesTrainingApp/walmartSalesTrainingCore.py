"""
@brief Core module for Walmart sales model training functionality
@details This module provides comprehensive functionality for training time series models
         on Walmart sales data, including data preprocessing, feature engineering,
         model training (ARIMA and Exponential Smoothing), and model evaluation
@author Sales Prediction Team
@date 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import os

def get_model_path_simple():
    """
    @brief Determine the appropriate model path for training environment
    @details Simple check for Streamlit Cloud vs local deployment specific to Training App
    @return Path string to the models directory for training output
    @note This function helps with cross-platform deployment compatibility for training
    """
    # Check if we're on Streamlit Cloud by looking for specific training environment
    if os.path.exists("Code/WalmartSalesTrainingApp"):
        return "Code/WalmartSalesTrainingApp/models/default/"
    else:
        return "models/default/"

def get_data_path_simple():
    """
    @brief Determine the appropriate data path for training datasets
    @details Simple check for data path in Training App deployment environment
    @return Path string to the dataset directory
    @note Handles different deployment scenarios for accessing training datasets
    """
    # Check if we're on Streamlit Cloud by looking for specific dataset environment
    if os.path.exists("Code/WalmartDataset"):
        return "Code/WalmartDataset/"
    else:
        return "../WalmartDataset/"

# Configuration dictionary with all hardcoded values for training
CONFIG = {
    'TRAIN_TEST_SPLIT': 0.7,  # 70% for training, 30% for testing
    'DEFAULT_SEASONAL_PERIODS': 20,  # Default seasonal periods for Holt-Winters
    'DEFAULT_MAX_P': 20,  # Maximum AR terms for ARIMA
    'DEFAULT_MAX_Q': 20,  # Maximum MA terms for ARIMA
    'DEFAULT_MAX_P_SEASONAL': 20,  # Maximum seasonal AR terms
    'DEFAULT_MAX_Q_SEASONAL': 20,  # Maximum seasonal MA terms
    'DEFAULT_MAX_ITER': 200,  # Maximum iterations for model fitting
    'DEFAULT_MAX_D': 10,  # Maximum differencing order
    'HOLIDAY_DATES': {  # Known holiday dates affecting sales
        'SUPER_BOWL': ['2010-02-12', '2011-02-11', '2012-02-10'],
        'LABOR_DAY': ['2010-09-10', '2011-09-09', '2012-09-07'],
        'THANKSGIVING': ['2010-11-26', '2011-11-25'],
        'CHRISTMAS': ['2010-12-31', '2011-12-30']
    },
    'MODEL_FILE_MAP': {  # Mapping of model names to file names
        "Auto ARIMA": "AutoARIMA",
        "Exponential Smoothing (Holt-Winters)": "ExponentialSmoothingHoltWinters"
    },
    'DEFAULT_MODEL_PATH': get_model_path_simple(),  # Dynamic model output path
    'DEFAULT_DATA_PATH': get_data_path_simple(),  # Dynamic data input path
    'SUPPORTED_EXTENSIONS': ["pkl"],  # Supported model file extensions
    'DEFAULT_ARIMA_ORDER': (1, 1, 1)  # Default ARIMA parameters (p, d, q)
}

def load_and_merge_data(train_file, features_file, stores_file):
    """
    @brief Load and merge the three required CSV files for Walmart sales analysis
    @details Performs inner joins between train, features, and stores datasets,
             handles duplicate holiday columns, and ensures data consistency
    @param train_file CSV file containing training sales data with Store, Date, Weekly_Sales
    @param features_file CSV file containing feature data with Store, Date, external factors
    @param stores_file CSV file containing store metadata with Store, Type, Size
    @return Merged pandas DataFrame with all data combined
    @raises ValueError If any file is missing or data merging fails
    @note Automatically resolves IsHoliday column conflicts from multiple sources
    """
    # Validate that all required files are provided
    if not train_file or not features_file or not stores_file:
        raise ValueError("All three files (train, features, stores) must be provided")
    
    try:
        # Load individual CSV files into DataFrames
        df_store = pd.read_csv(stores_file)
        df_train = pd.read_csv(train_file)
        df_features = pd.read_csv(features_file)
        
        # Merge datasets using inner joins to ensure data consistency
        # First merge train with features on Store and Date
        df = df_train.merge(df_features, on=['Store', 'Date'], how='inner')
        # Then merge with store information on Store
        df = df.merge(df_store, on=['Store'], how='inner')
        
        # Handle duplicate IsHoliday columns created during merge
        df.drop(['IsHoliday_y'], axis=1, inplace=True)  # Remove features IsHoliday
        df.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)  # Keep train IsHoliday
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading and merging data: {str(e)}")

def clean_data(df):
    """
    @brief Clean and preprocess the merged sales data
    @details Removes invalid sales records, fills missing values, and creates
             specific holiday indicator features for improved model performance
    @param df Merged DataFrame containing sales and feature data
    @return Cleaned DataFrame ready for time series analysis
    @raises ValueError If input DataFrame is None or empty
    @note Creates binary holiday indicators for major sales-affecting holidays
    """
    # Validate input DataFrame
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be None or empty")
    
    try:
        # Remove records with non-positive sales (data quality issue)
        df = df.loc[df['Weekly_Sales'] > 0]
        
        # Fill missing values in markdown columns with zeros (common practice)
        df = df.fillna(0)
        
        # Create specific holiday indicators using configured dates
        holiday_dates = CONFIG['HOLIDAY_DATES']
        
        # Super Bowl indicator (major shopping event)
        df.loc[df['Date'].isin(holiday_dates['SUPER_BOWL']), 'Super_Bowl'] = True
        df.loc[~df['Date'].isin(holiday_dates['SUPER_BOWL']), 'Super_Bowl'] = False
        
        # Labor Day indicator (end of summer shopping)
        df.loc[df['Date'].isin(holiday_dates['LABOR_DAY']), 'Labor_Day'] = True
        df.loc[~df['Date'].isin(holiday_dates['LABOR_DAY']), 'Labor_Day'] = False
        
        # Thanksgiving indicator (major holiday shopping)
        df.loc[df['Date'].isin(holiday_dates['THANKSGIVING']), 'Thanksgiving'] = True
        df.loc[~df['Date'].isin(holiday_dates['THANKSGIVING']), 'Thanksgiving'] = False
        
        # Christmas indicator (peak shopping season)
        df.loc[df['Date'].isin(holiday_dates['CHRISTMAS']), 'Christmas'] = True
        df.loc[~df['Date'].isin(holiday_dates['CHRISTMAS']), 'Christmas'] = False
        
        return df
    except Exception as e:
        raise ValueError(f"Error cleaning data: {str(e)}")

def prepare_time_series_data(df):
    """
    @brief Prepare data for time series modeling by aggregation and differencing
    @details Converts data to weekly frequency, aggregates numerical features,
             and creates differenced series for stationarity
    @param df Cleaned DataFrame with date and numerical columns
    @return Tuple of (weekly_aggregated_data, differenced_sales_series)
    @raises ValueError If input DataFrame is None or empty
    @note Differencing helps achieve stationarity required for ARIMA modeling
    """
    # Validate input DataFrame
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be None or empty")
    
    try:
        # Convert date column to datetime and set as index for time series operations
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index('Date', inplace=True)
        
        # Create weekly aggregated data using mean for numerical features
        # This handles irregular date intervals and multiple stores per week
        df_week = df.select_dtypes(include='number').resample('W').mean()
        
        # Difference the sales data for stationarity (removes trend/seasonality)
        # First difference removes trend, making data suitable for ARIMA
        df_week_diff = df_week['Weekly_Sales'].diff().dropna()
        
        return df_week, df_week_diff
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def train_auto_arima(train_data_diff, hyperparams=None):
    """
    @brief Train Auto ARIMA model with automatic parameter selection
    @details Uses pmdarima's auto_arima to automatically select optimal ARIMA parameters
             through grid search and information criteria optimization
    @param train_data_diff Differenced training data (stationary time series)
    @param hyperparams Optional dictionary of hyperparameters to override defaults
    @return Fitted Auto ARIMA model object
    @raises ValueError If training data is None, empty, or training fails
    @note Auto ARIMA automatically handles parameter selection and seasonal components
    """
    # Validate training data
    if train_data_diff is None or len(train_data_diff) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Set default parameters for Auto ARIMA with extensive search
        default_params = {
            'start_p': 0,  # Starting AR order
            'start_q': 0,  # Starting MA order
            'start_P': 0,  # Starting seasonal AR order
            'start_Q': 0,  # Starting seasonal MA order
            'max_p': CONFIG['DEFAULT_MAX_P'],  # Maximum AR order to test
            'max_q': CONFIG['DEFAULT_MAX_Q'],  # Maximum MA order to test
            'max_P': CONFIG['DEFAULT_MAX_P_SEASONAL'],  # Maximum seasonal AR order
            'max_Q': CONFIG['DEFAULT_MAX_Q_SEASONAL'],  # Maximum seasonal MA order
            'seasonal': True,  # Enable seasonal components
            'maxiter': CONFIG['DEFAULT_MAX_ITER'],  # Maximum fitting iterations
            'information_criterion': 'aic',  # Use AIC for model selection
            'stepwise': False,  # Disable stepwise search for thorough exploration
            'suppress_warnings': True,  # Suppress convergence warnings
            'D': 1,  # Seasonal differencing order
            'max_D': CONFIG['DEFAULT_MAX_D'],  # Maximum seasonal differencing
            'error_action': 'ignore',  # Ignore errors and continue search
            'approximation': False  # Use exact likelihood for better accuracy
        }
        
        # Update parameters with user-provided hyperparameters
        if hyperparams:
            default_params.update(hyperparams)
        
        # Perform auto ARIMA model selection and training
        model_auto_arima = auto_arima(train_data_diff, trace=True, **default_params)
        # Fit the selected model to training data
        model_auto_arima.fit(train_data_diff)
        
        return model_auto_arima
    except Exception as e:
        raise ValueError(f"Error training Auto ARIMA model: {str(e)}")

def train_exponential_smoothing(train_data_diff, hyperparams=None):
    """
    @brief Train Exponential Smoothing (Holt-Winters) model
    @details Implements triple exponential smoothing with trend and seasonal components
             for robust time series forecasting
    @param train_data_diff Differenced training data for model fitting
    @param hyperparams Optional dictionary of hyperparameters to override defaults
    @return Fitted Exponential Smoothing model object
    @raises ValueError If training data is None, empty, or training fails
    @note Holt-Winters method handles trend and seasonality simultaneously
    """
    # Validate training data
    if train_data_diff is None or len(train_data_diff) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Set default parameters for Exponential Smoothing
        default_params = {
            'seasonal_periods': CONFIG['DEFAULT_SEASONAL_PERIODS'],  # Seasonal cycle length
            'seasonal': 'additive',  # Additive seasonal component
            'trend': 'additive',  # Additive trend component
            'damped': True  # Damped trend to prevent over-extrapolation
        }
        
        # Update parameters with user-provided hyperparameters
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create and fit Exponential Smoothing model
        model_holt_winters = ExponentialSmoothing(
            train_data_diff,
            **default_params
        ).fit()
        
        return model_holt_winters
    except Exception as e:
        raise ValueError(f"Error training Exponential Smoothing model: {str(e)}")

def wmae_ts(y_true, y_pred):
    """
    @brief Calculate weighted mean absolute error for time series evaluation
    @details Computes WMAE metric which is commonly used for sales forecasting
             evaluation, providing equal weighting to all observations
    @param y_true True/actual values from test dataset
    @param y_pred Predicted values from trained model
    @return WMAE score (lower is better)
    @raises ValueError If inputs are None or calculation fails
    @note WMAE is preferred over RMSE for sales data due to robustness to outliers
    """
    # Validate input parameters
    if y_true is None or y_pred is None:
        raise ValueError("True and predicted values cannot be None")
    
    try:
        # Convert pandas objects to numpy arrays for calculation
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            y_pred = y_pred.values
        
        # Create uniform weights (can be modified for time-weighted evaluation)
        weights = np.ones_like(y_true)
        
        # Calculate weighted mean absolute error
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    except Exception as e:
        raise ValueError(f"Error calculating WMAE: {str(e)}")

def create_diagnostic_plots(train_data, test_data, predictions, model_type):
    """
    @brief Create diagnostic plots for model evaluation and visualization
    @details Generates comprehensive plots showing training data, test data,
             and model predictions for visual assessment of model performance
    @param train_data Training dataset time series
    @param test_data Test dataset time series
    @param predictions Model predictions for test period
    @param model_type String identifier for the model type (for plot title)
    @return Matplotlib figure object containing the diagnostic plot
    @raises ValueError If any input data is None or plotting fails
    @note Visual inspection helps identify model biases and seasonal patterns
    """
    # Validate input data
    if train_data is None or test_data is None or predictions is None:
        raise ValueError("Training data, test data, and predictions cannot be None")
    
    try:
        # Create figure with appropriate size for detailed visualization
        plt.figure(figsize=(15, 6))
        plt.title(f'Prediction using {model_type}', fontsize=15)
        
        # Plot training data (blue line)
        plt.plot(train_data.index, train_data.values, label='Train')
        # Plot actual test data (orange line)
        plt.plot(test_data.index, test_data.values, label='Test')
        # Plot model predictions (green line)
        plt.plot(test_data.index, predictions, label='Prediction')
        
        # Add legend and labels for clarity
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales (Differenced)')
        plt.grid(True)  # Add grid for better readability
        
        return plt.gcf()  # Return current figure
    except Exception as e:
        raise ValueError(f"Error creating diagnostic plots: {str(e)}")

def save_model(model, model_type):
    """
    @brief Save trained model to default location with error handling
    @details Saves model using joblib serialization to the configured default path
             with automatic directory creation if needed
    @param model Trained model object to save
    @param model_type String identifier for model type (used for filename)
    @return Tuple of (success_boolean, error_message)
    @note Uses joblib for cross-platform compatibility and efficient serialization
    """
    try:
        # Get filename from model type mapping
        file_name = CONFIG['MODEL_FILE_MAP'][model_type]
        model_path = f"{CONFIG['DEFAULT_MODEL_PATH']}{file_name}.pkl"
        
        # Create directory if it doesn't exist (handles deployment scenarios)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model using joblib for efficient serialization
        joblib.dump(model, model_path)
        
        return True, None  # Success case
        
    except Exception as e:
        return False, f"Error saving model: {str(e)}"  # Error case