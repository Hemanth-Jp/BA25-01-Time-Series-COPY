import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# Configuration dictionary with all hardcoded values
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
    }
}

def load_and_merge_data(train_file, features_file, stores_file):
    """Load and merge the three CSV files"""
    if not train_file or not features_file or not stores_file:
        raise ValueError("All three files (train, features, stores) must be provided")
    
    try:
        df_store = pd.read_csv(stores_file)
        df_train = pd.read_csv(train_file)
        df_features = pd.read_csv(features_file)
        
        # Merge datasets
        df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
        df.drop(['IsHoliday_y'], axis=1, inplace=True)
        df.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading and merging data: {str(e)}")

def clean_data(df):
    """Clean the merged data"""
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be None or empty")
    
    try:
        # Remove non-positive sales
        df = df.loc[df['Weekly_Sales'] > 0]
        
        # Fill missing values in markdown columns with zeros
        df = df.fillna(0)
        
        # Create specific holiday indicators
        holiday_dates = CONFIG['HOLIDAY_DATES']
        
        df.loc[df['Date'].isin(holiday_dates['SUPER_BOWL']), 'Super_Bowl'] = True
        df.loc[~df['Date'].isin(holiday_dates['SUPER_BOWL']), 'Super_Bowl'] = False
        
        df.loc[df['Date'].isin(holiday_dates['LABOR_DAY']), 'Labor_Day'] = True
        df.loc[~df['Date'].isin(holiday_dates['LABOR_DAY']), 'Labor_Day'] = False
        
        df.loc[df['Date'].isin(holiday_dates['THANKSGIVING']), 'Thanksgiving'] = True
        df.loc[~df['Date'].isin(holiday_dates['THANKSGIVING']), 'Thanksgiving'] = False
        
        df.loc[df['Date'].isin(holiday_dates['CHRISTMAS']), 'Christmas'] = True
        df.loc[~df['Date'].isin(holiday_dates['CHRISTMAS']), 'Christmas'] = False
        
        return df
    except Exception as e:
        raise ValueError(f"Error cleaning data: {str(e)}")

def prepare_time_series_data(df):
    """Prepare data for time series modeling"""
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be None or empty")
    
    try:
        # Convert date and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index('Date', inplace=True)
        
        # Create weekly aggregated data
        df_week = df.select_dtypes(include='number').resample('W').mean()
        
        # Difference the data for stationarity
        df_week_diff = df_week['Weekly_Sales'].diff().dropna()
        
        return df_week, df_week_diff
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def train_auto_arima(train_data_diff, hyperparams=None):
    """Train Auto ARIMA model"""
    if train_data_diff is None or len(train_data_diff) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        default_params = {
            'start_p': 0,
            'start_q': 0,
            'start_P': 0,
            'start_Q': 0,
            'max_p': CONFIG['DEFAULT_MAX_P'],
            'max_q': CONFIG['DEFAULT_MAX_Q'],
            'max_P': CONFIG['DEFAULT_MAX_P_SEASONAL'],
            'max_Q': CONFIG['DEFAULT_MAX_Q_SEASONAL'],
            'seasonal': True,
            'maxiter': CONFIG['DEFAULT_MAX_ITER'],
            'information_criterion': 'aic',
            'stepwise': False,
            'suppress_warnings': True,
            'D': 1,
            'max_D': CONFIG['DEFAULT_MAX_D'],
            'error_action': 'ignore',
            'approximation': False
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        model_auto_arima = auto_arima(train_data_diff, trace=True, **default_params)
        model_auto_arima.fit(train_data_diff)
        
        return model_auto_arima
    except Exception as e:
        raise ValueError(f"Error training Auto ARIMA model: {str(e)}")

def train_exponential_smoothing(train_data_diff, hyperparams=None):
    """Train Exponential Smoothing model"""
    if train_data_diff is None or len(train_data_diff) == 0:
        raise ValueError("Training data cannot be None or empty")
    
    try:
        default_params = {
            'seasonal_periods': CONFIG['DEFAULT_SEASONAL_PERIODS'],
            'seasonal': 'additive',
            'trend': 'additive',
            'damped': True
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        model_holt_winters = ExponentialSmoothing(
            train_data_diff,
            **default_params
        ).fit()
        
        return model_holt_winters
    except Exception as e:
        raise ValueError(f"Error training Exponential Smoothing model: {str(e)}")

def wmae_ts(y_true, y_pred):
    """Calculate weighted mean absolute error"""
    if y_true is None or y_pred is None:
        raise ValueError("True and predicted values cannot be None")
    
    try:
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            y_pred = y_pred.values
        
        weights = np.ones_like(y_true)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    except Exception as e:
        raise ValueError(f"Error calculating WMAE: {str(e)}")

def create_diagnostic_plots(train_data, test_data, predictions, model_type):
    """Create diagnostic plots for model evaluation"""
    if train_data is None or test_data is None or predictions is None:
        raise ValueError("Training data, test data, and predictions cannot be None")
    
    try:
        plt.figure(figsize=(15, 6))
        plt.title(f'Prediction using {model_type}', fontsize=15)
        plt.plot(train_data.index, train_data.values, label='Train')
        plt.plot(test_data.index, test_data.values, label='Test')
        plt.plot(test_data.index, predictions, label='Prediction')
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales (Differenced)')
        plt.grid(True)
        
        return plt.gcf()
    except Exception as e:
        raise ValueError(f"Error creating diagnostic plots: {str(e)}")
