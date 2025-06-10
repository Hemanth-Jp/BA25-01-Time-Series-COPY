import os
import tempfile
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_model_path_simple():
    """Simple check for Streamlit Cloud vs local for Training App"""
    # Check if we're on Streamlit Cloud by looking for specific environment
    if os.path.exists("Code/WalmartSalesTrainingApp"):
        return "Code/WalmartSalesTrainingApp/models/default/"
    else:
        return "models/default/"

def get_data_path_simple():
    """Simple check for data path in Training App"""
    # Check if we're on Streamlit Cloud by looking for specific environment
    if os.path.exists("Code/WalmartDataset"):
        return "Code/WalmartDataset/"
    else:
        return "../WalmartDataset/"

# Configuration dictionary
CONFIG = {
    'TRAIN_TEST_SPLIT': 0.7,
    'DEFAULT_SEASONAL_PERIODS': 20,
    'HOLIDAY_DATES': [
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',  # Super Bowl
        '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06',  # Labor Day
        '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',  # Thanksgiving
        '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'   # Christmas
    ],
    'MODEL_FILE_MAP': {
        "Auto ARIMA": "AutoARIMA",
        "Exponential Smoothing (Holt-Winters)": "ExponentialSmoothingHoltWinters"
    },
    'DEFAULT_MODEL_PATH': get_model_path_simple(),
    'DEFAULT_DATA_PATH': get_data_path_simple(),
    'SUPPORTED_EXTENSIONS': ["pkl"],
    'DEFAULT_ARIMA_ORDER': (1, 1, 1)
}

def load_and_merge_data(train_file, features_file, stores_file):
    """
    Load and merge the three CSV files
    
    Args:
        train_file: Training data CSV file
        features_file: Features data CSV file  
        stores_file: Stores data CSV file
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    if not all([train_file, features_file, stores_file]):
        raise ValueError("All three files must be provided")
    
    try:
        # Load the datasets
        train_df = pd.read_csv(train_file)
        features_df = pd.read_csv(features_file)
        stores_df = pd.read_csv(stores_file)
        
        # Merge datasets
        merged_df = train_df.merge(features_df, on=['Store', 'Date'], how='left')
        merged_df = merged_df.merge(stores_df, on='Store', how='left')
        
        return merged_df
        
    except Exception as e:
        raise ValueError(f"Error loading and merging data: {str(e)}")

def clean_data(df):
    """
    Clean the merged dataframe
    
    Args:
        df (pd.DataFrame): Raw merged dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None or df.empty:
        raise ValueError("Dataframe cannot be None or empty")
    
    try:
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert Date to datetime
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Remove negative weekly sales
        df_clean = df_clean[df_clean['Weekly_Sales'] > 0]
        
        # Fill missing values for markdown columns
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Handle IsHoliday columns from merge (IsHoliday_x and IsHoliday_y)
        if 'IsHoliday_x' in df_clean.columns and 'IsHoliday_y' in df_clean.columns:
            # Use IsHoliday_x (from train data) as primary, fill with IsHoliday_y where missing
            df_clean['IsHoliday'] = df_clean['IsHoliday_x'].fillna(df_clean['IsHoliday_y'])
            df_clean = df_clean.drop(['IsHoliday_x', 'IsHoliday_y'], axis=1)
        
        # Create holiday indicators based on dates
        df_clean['Date_str'] = df_clean['Date'].dt.strftime('%Y-%m-%d')
        
        # Create specific holiday flags
        df_clean['Super_Bowl'] = df_clean['Date_str'].isin([
            '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'
        ]).astype(int)
        
        df_clean['Labor_Day'] = df_clean['Date_str'].isin([
            '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'
        ]).astype(int)
        
        df_clean['Thanksgiving'] = df_clean['Date_str'].isin([
            '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'
        ]).astype(int)
        
        df_clean['Christmas'] = df_clean['Date_str'].isin([
            '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'
        ]).astype(int)
        
        return df_clean
        
    except Exception as e:
        raise ValueError(f"Error cleaning data: {str(e)}")

def prepare_time_series_data(df):
    """
    Prepare data for time series modeling
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        tuple: (df_week, df_week_diff)
    """
    try:
        # Aggregate by week
        df_week = df.groupby('Date').agg({
            'Weekly_Sales': 'sum',
            'Temperature': 'mean',
            'Fuel_Price': 'mean' if 'Fuel_Price' in df.columns else lambda x: 0,
            'CPI': 'mean' if 'CPI' in df.columns else lambda x: 0,
            'Unemployment': 'mean' if 'Unemployment' in df.columns else lambda x: 0
        }).reset_index()
        
        df_week = df_week.sort_values('Date')
        
        # Calculate week-over-week difference for sales
        df_week_diff = df_week['Weekly_Sales'].diff().dropna()
        
        return df_week, df_week_diff
        
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def wmae_ts(y_true, y_pred):
    """
    Calculate Weighted Mean Absolute Error for time series
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: WMAE score
    """
    if y_true is None or y_pred is None:
        raise ValueError("Inputs cannot be None")
    
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate weights (higher for holiday weeks)
        weights = np.ones(len(y_true))
        
        # Calculate WMAE
        wmae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
        
        return wmae
        
    except Exception as e:
        raise ValueError(f"Error calculating WMAE: {str(e)}")

def train_auto_arima(train_data):
    """
    Train Auto ARIMA model
    
    Args:
        train_data: Training time series data
        
    Returns:
        Fitted ARIMA model
    """
    if train_data is None or len(train_data) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Fit Auto ARIMA
        model = auto_arima(
            train_data,
            seasonal=True,
            m=CONFIG['DEFAULT_SEASONAL_PERIODS'],
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore'
        )
        
        model.fit(train_data)
        return model
        
    except Exception as e:
        raise ValueError(f"Error training Auto ARIMA: {str(e)}")

def train_exponential_smoothing(train_data):
    """
    Train Exponential Smoothing (Holt-Winters) model
    
    Args:
        train_data: Training time series data
        
    Returns:
        Fitted Exponential Smoothing model
    """
    if train_data is None or len(train_data) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Fit Exponential Smoothing
        model = ExponentialSmoothing(
            train_data,
            trend='add',
            seasonal='add',
            seasonal_periods=CONFIG['DEFAULT_SEASONAL_PERIODS']
        )
        
        fitted_model = model.fit()
        return fitted_model
        
    except Exception as e:
        raise ValueError(f"Error training Exponential Smoothing: {str(e)}")

def save_model(model, model_type):
    """
    Save trained model to default location
    
    Args:
        model: Trained model object
        model_type (str): Type of model
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        file_name = CONFIG['MODEL_FILE_MAP'][model_type]
        model_path = f"{CONFIG['DEFAULT_MODEL_PATH']}{file_name}.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        
        return True, None
        
    except Exception as e:
        return False, f"Error saving model: {str(e)}"