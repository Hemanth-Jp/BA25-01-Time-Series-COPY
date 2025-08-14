"""
Basic Time Series Forecasting with Pmdarima

This module demonstrates fundamental time series forecasting using pmdarima's
auto_arima functionality. It covers data loading, automatic model selection,
and basic forecasting with visualization.


@version: 1.0
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load sample time series data and prepare for modeling.
    
    Returns:
        pandas.Series: Time series data with datetime index
    """
    try:
        # Load built-in wine sales dataset
        y = pm.datasets.load_wineind()
        
        # Create datetime index (monthly data from 1980)
        dates = pd.date_range(start='1980-01', periods=len(y), freq='M')
        ts_data = pd.Series(y, index=dates, name='Wine_Sales')
        
        print(f"Dataset loaded successfully: {len(ts_data)} observations")
        print(f"Date range: {ts_data.index[0]} to {ts_data.index[-1]}")
        
        return ts_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_forecasting_workflow(data, forecast_periods=24):
    """
    Demonstrate basic forecasting workflow with auto_arima.
    
    Args:
        data (pandas.Series): Time series data
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        tuple: (model, forecasts, confidence_intervals, metrics)
    """
    if data is None:
        return None, None, None, None
    
    # Split data into train and test sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Training set: {len(train_data)} observations")
    print(f"Test set: {len(test_data)} observations")
    
    # Automatic ARIMA model selection
    print("\nFitting auto_arima model...")
    model = pm.auto_arima(
        train_data,
        seasonal=True,
        m=12,  # Monthly seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        random_state=42
    )
    
    print(f"Best model: {model.order} x {model.seasonal_order}")
    
    # Generate forecasts with confidence intervals
    forecasts, conf_int = model.predict(
        n_periods=len(test_data),
        return_conf_int=True,
        alpha=0.05  # 95% confidence interval
    )
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(test_data, forecasts)
    rmse = np.sqrt(mean_squared_error(test_data, forecasts))
    mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return model, forecasts, conf_int, metrics

def visualize_results(data, model, forecasts, conf_int):
    """
    Create visualization of historical data and forecasts.
    
    Args:
        data (pandas.Series): Original time series
        model: Fitted pmdarima model
        forecasts (array): Forecast values
        conf_int (array): Confidence intervals
    """
    # Calculate split point
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create forecast index
    forecast_index = test_data.index
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot historical data
    plt.plot(train_data.index, train_data.values, 
             label='Training Data', color='blue', linewidth=1.5)
    plt.plot(test_data.index, test_data.values, 
             label='Actual', color='red', linewidth=1.5)
    
    # Plot forecasts
    plt.plot(forecast_index, forecasts, 
             label='Forecasts', color='green', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(forecast_index, 
                     conf_int[:, 0], conf_int[:, 1],
                     color='green', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Time Series Forecasting with Auto-ARIMA', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Wine Sales', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def main():
    """
    Main function demonstrating basic pmdarima forecasting workflow.
    """
    print("=== Basic Time Series Forecasting with Pmdarima ===\n")
    
    # Load and prepare data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Perform forecasting
    model, forecasts, conf_int, metrics = basic_forecasting_workflow(data)
    if model is None:
        return
    
    # Visualize results
    plt_obj = visualize_results(data, model, forecasts, conf_int)
    plt_obj.show()
    
    print("\n=== Forecasting Complete ===")
    print("Basic auto_arima workflow successfully demonstrated!")

if __name__ == "__main__":
    main()
