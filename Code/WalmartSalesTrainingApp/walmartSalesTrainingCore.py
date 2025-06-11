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
import matplotlib.pyplot as plt
import seaborn as sns

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
    'DEFAULT_MAX_P': 20,
    'DEFAULT_MAX_Q': 20,
    'DEFAULT_MAX_P_SEASONAL': 20,
    'DEFAULT_MAX_Q_SEASONAL': 20,
    'DEFAULT_MAX_ITER': 200,
    'DEFAULT_MAX_D': 10,
    'HOLIDAY_DATES': {
        'SUPER_BOWL': ['2010-02-12', '2011-02-11', '2012-02-10'],
        'LABOR_DAY': ['2010-09-10', '2011-09-09', '2012-09-07'],
        'THANKSGIVING': ['2010-11-26', '2011-11-25'],
        'CHRISTMAS': ['2010-12-31', '2011-12-30']
    },
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
        
        # Create specific holiday flags using CONFIG
        df_clean['Super_Bowl'] = df_clean['Date_str'].isin(
            CONFIG['HOLIDAY_DATES']['SUPER_BOWL']
        ).astype(int)
        
        df_clean['Labor_Day'] = df_clean['Date_str'].isin(
            CONFIG['HOLIDAY_DATES']['LABOR_DAY']
        ).astype(int)
        
        df_clean['Thanksgiving'] = df_clean['Date_str'].isin(
            CONFIG['HOLIDAY_DATES']['THANKSGIVING']
        ).astype(int)
        
        df_clean['Christmas'] = df_clean['Date_str'].isin(
            CONFIG['HOLIDAY_DATES']['CHRISTMAS']
        ).astype(int)
        
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

def train_auto_arima(train_data, hyperparams=None):
    """
    Train Auto ARIMA model
    
    Args:
        train_data: Training time series data
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Fitted ARIMA model
    """
    if train_data is None or len(train_data) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Set default hyperparameters if not provided
        if hyperparams is None:
            hyperparams = {}
        
        # Fit Auto ARIMA with hyperparameters
        model = auto_arima(
            train_data,
            start_p=hyperparams.get('start_p', 0),
            start_q=hyperparams.get('start_q', 0),
            max_p=hyperparams.get('max_p', CONFIG['DEFAULT_MAX_P']),
            max_q=hyperparams.get('max_q', CONFIG['DEFAULT_MAX_Q']),
            start_P=hyperparams.get('start_P', 0),
            start_Q=hyperparams.get('start_Q', 0),
            max_P=hyperparams.get('max_P', CONFIG['DEFAULT_MAX_P_SEASONAL']),
            max_Q=hyperparams.get('max_Q', CONFIG['DEFAULT_MAX_Q_SEASONAL']),
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

def train_exponential_smoothing(train_data, hyperparams=None):
    """
    Train Exponential Smoothing (Holt-Winters) model
    
    Args:
        train_data: Training time series data
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Fitted Exponential Smoothing model
    """
    if train_data is None or len(train_data) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        # Set default hyperparameters if not provided
        if hyperparams is None:
            hyperparams = {}
        
        # Fit Exponential Smoothing with hyperparameters
        model = ExponentialSmoothing(
            train_data,
            trend=hyperparams.get('trend', 'add'),
            seasonal=hyperparams.get('seasonal', 'add'),
            seasonal_periods=hyperparams.get('seasonal_periods', CONFIG['DEFAULT_SEASONAL_PERIODS']),
            damped_trend=hyperparams.get('damped', True)
        )
        
        fitted_model = model.fit()
        return fitted_model
        
    except Exception as e:
        raise ValueError(f"Error training Exponential Smoothing: {str(e)}")

def create_diagnostic_plots(train_data, test_data, predictions, model_type):
    """
    Create diagnostic plots for model evaluation
    
    Args:
        train_data: Training time series data
        test_data: Test time series data
        predictions: Model predictions
        model_type: Type of model used
        
    Returns:
        matplotlib.figure.Figure: Figure with diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type} Model Diagnostics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training vs Test Data
    axes[0, 0].plot(range(len(train_data)), train_data, label='Training Data', color='blue')
    test_range = range(len(train_data), len(train_data) + len(test_data))
    axes[0, 0].plot(test_range, test_data, label='Test Data', color='green')
    axes[0, 0].plot(test_range, predictions, label='Predictions', color='red', linestyle='--')
    axes[0, 0].set_title('Time Series Forecast')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Sales Difference')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = np.array(test_data) - np.array(predictions)
    axes[0, 1].plot(residuals, color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Residuals')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals Distribution
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Actual vs Predicted
    axes[1, 1].scatter(test_data, predictions, alpha=0.6, color='purple')
    min_val = min(min(test_data), min(predictions))
    max_val = max(max(test_data), max(predictions))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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