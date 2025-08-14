"""
Time Series Analysis with Statsmodels (Fixed Version)

This module demonstrates comprehensive time series analysis including
trend detection, seasonality, and ARIMA modeling using statsmodels.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

def generate_timeseries_data(n_periods=800, random_state=42):
    """
    Generate sample time series data with trend and seasonality.
    
    Args:
        n_periods (int): Number of time periods (increased to 800 for 2+ cycles)
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.Series: Generated time series
    """
    np.random.seed(random_state)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    # Generate components
    trend = np.linspace(100, 200, n_periods)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25)
    noise = np.random.normal(0, 5, n_periods)
    
    # Combine components
    ts_data = trend + seasonal + noise
    
    return pd.Series(ts_data, index=dates, name='value')

def check_stationarity(ts_data):
    """
    Check stationarity using Augmented Dickey-Fuller test.
    
    Args:
        ts_data (pandas.Series): Time series data
        
    Returns:
        bool: True if stationary, False otherwise
    """
    print("=== Stationarity Test ===")
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(ts_data.dropna())
    
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    is_stationary = adf_result[1] <= 0.05
    print(f"Result: {'Stationary' if is_stationary else 'Non-stationary'}")
    
    return is_stationary

def decompose_timeseries(ts_data, period=None):
    """
    Perform seasonal decomposition of time series.
    
    Args:
        ts_data (pandas.Series): Time series data
        period (int): Seasonal period. If None, attempts to detect automatically
        
    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: Decomposition results
    """
    print("\n=== Seasonal Decomposition ===")
    
    # Determine period if not specified
    if period is None:
        # For daily data, try common periods
        if len(ts_data) >= 730:  # At least 2 years of daily data
            period = 365
        elif len(ts_data) >= 104:  # At least 2 years of weekly data
            period = 52
        elif len(ts_data) >= 24:  # At least 2 years of monthly data
            period = 12
        else:
            print(f"Warning: Not enough data for seasonal decomposition. Need at least 2 complete cycles.")
            print(f"Data length: {len(ts_data)}, trying with period=7 (weekly)")
            period = 7
    
    print(f"Using period: {period}")
    
    # Check if we have enough data
    if len(ts_data) < 2 * period:
        print(f"Warning: Insufficient data for period {period}. Need {2 * period} observations, have {len(ts_data)}")
        # Try with a smaller period
        period = max(4, len(ts_data) // 3)
        print(f"Adjusting period to: {period}")
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
        
    except ValueError as e:
        print(f"Error in seasonal decomposition: {e}")
        print("Skipping seasonal decomposition due to insufficient data.")
        return None

def analyze_autocorrelation(ts_data, lags=40):
    """
    Analyze autocorrelation and partial autocorrelation.
    
    Args:
        ts_data (pandas.Series): Time series data
        lags (int): Number of lags to analyze
    """
    print("\n=== Autocorrelation Analysis ===")
    
    # Adjust lags if data is too short
    max_lags = min(lags, len(ts_data) // 4)
    print(f"Using {max_lags} lags for analysis")
    
    # Plot ACF and PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(ts_data.dropna(), lags=max_lags, ax=axes[0])
    plot_pacf(ts_data.dropna(), lags=max_lags, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def fit_arima_model(ts_data, order=(1, 1, 1)):
    """
    Fit ARIMA model to time series data.
    
    Args:
        ts_data (pandas.Series): Time series data
        order (tuple): ARIMA order (p, d, q)
        
    Returns:
        statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model
    """
    print(f"\n=== ARIMA Model Fitting ===")
    print(f"Fitting ARIMA{order} model...")
    
    try:
        # Fit ARIMA model
        model = ARIMA(ts_data, order=order)
        fitted_model = model.fit()
        
        # Display model summary
        print(fitted_model.summary())
        
        return fitted_model
        
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        print("Trying with simpler model ARIMA(1,0,1)...")
        try:
            model = ARIMA(ts_data, order=(1, 0, 1))
            fitted_model = model.fit()
            print(fitted_model.summary())
            return fitted_model
        except Exception as e2:
            print(f"Error with simpler model: {e2}")
            return None

def model_diagnostics(fitted_model):
    """
    Perform model diagnostic tests.
    
    Args:
        fitted_model: Fitted ARIMA model results
    """
    if fitted_model is None:
        print("No model to diagnose")
        return
        
    print("\n=== Model Diagnostics ===")
    
    # Residual analysis
    residuals = fitted_model.resid
    
    try:
        # Ljung-Box test for residual autocorrelation
        lb_result = sm.stats.diagnostic.acorr_ljungbox(
            residuals, lags=min(10, len(residuals)//4), return_df=True)
        
        print(f"Ljung-Box Test:")
        print(f"Test statistic: {lb_result['lb_stat'].iloc[-1]:.4f}")
        print(f"p-value: {lb_result['lb_pvalue'].iloc[-1]:.4f}")
        print(f"Result: {'No autocorrelation' if lb_result['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation present'}")
        
    except Exception as e:
        print(f"Error in Ljung-Box test: {e}")
    
    # Plot residual diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residuals over time
    residuals.plot(ax=axes[0, 0])
    axes[0, 0].set_title('Residuals')
    
    # Q-Q plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # ACF of residuals
    max_lags = min(20, len(residuals) // 4)
    plot_acf(residuals, ax=axes[1, 0], lags=max_lags)
    axes[1, 0].set_title('ACF of Residuals')
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=20, density=True, alpha=0.7)
    axes[1, 1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()

def generate_forecasts(fitted_model, steps=30):
    """
    Generate forecasts using fitted model.
    
    Args:
        fitted_model: Fitted ARIMA model
        steps (int): Number of forecast steps
        
    Returns:
        pandas.DataFrame: Forecast values and confidence intervals
    """
    if fitted_model is None:
        print("No fitted model available for forecasting")
        return None
        
    print(f"\n=== Forecasting ({steps} steps ahead) ===")
    
    try:
        # Generate forecasts
        forecast = fitted_model.forecast(steps=steps)
        forecast_ci = fitted_model.get_forecast(steps=steps).conf_int()
        
        # Create forecast dates
        if hasattr(fitted_model.data, 'dates') and fitted_model.data.dates is not None:
            # If dates are available
            last_date = fitted_model.data.dates[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=steps, freq='D')
        else:
            # If no dates, use simple range
            forecast_dates = range(len(fitted_model.data.endog), len(fitted_model.data.endog) + steps)
        
        # Display forecast summary
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Lower CI': forecast_ci.iloc[:, 0],
            'Upper CI': forecast_ci.iloc[:, 1]
        }, index=forecast_dates)
        
        print("Forecast Summary (first 10 periods):")
        print(forecast_df.head(10))
        
        return forecast_df
        
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return None

def main():
    """Main function to demonstrate time series analysis."""
    # Generate sample time series with more data points
    print("Generating sample time series data...")
    ts_data = generate_timeseries_data(n_periods=800)  # Increased from 300 to 800
    
    print(f"\nTime Series Overview:")
    print(f"Start Date: {ts_data.index[0]}")
    print(f"End Date: {ts_data.index[-1]}")
    print(f"Number of observations: {len(ts_data)}")
    print(f"Mean: {ts_data.mean():.2f}")
    print(f"Standard deviation: {ts_data.std():.2f}")
    
    # Plot original time series
    plt.figure(figsize=(12, 6))
    ts_data.plot()
    plt.title('Original Time Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    
    # Check stationarity
    is_stationary = check_stationarity(ts_data)
    
    # Seasonal decomposition
    decomposition = decompose_timeseries(ts_data)
    
    # Make series stationary if needed
    if not is_stationary:
        print("\nApplying first differencing...")
        ts_diff = ts_data.diff().dropna()
        is_stationary_diff = check_stationarity(ts_diff)
        ts_analysis = ts_diff if is_stationary_diff else ts_data
    else:
        ts_analysis = ts_data
    
    # Autocorrelation analysis
    analyze_autocorrelation(ts_analysis)
    
    # Fit ARIMA model
    fitted_model = fit_arima_model(ts_data)
    
    # Model diagnostics
    model_diagnostics(fitted_model)
    
    # Generate forecasts
    forecast_df = generate_forecasts(fitted_model)
    
    if forecast_df is not None:
        # Plot forecasts
        plt.figure(figsize=(14, 8))
        
        # Plot last 100 observations and forecasts
        ts_data.tail(100).plot(label='Observed', color='blue')
        forecast_df['Forecast'].plot(label='Forecast', color='red')
        plt.fill_between(forecast_df.index, 
                         forecast_df['Lower CI'], 
                         forecast_df['Upper CI'], 
                         color='red', alpha=0.3, label='Confidence Interval')
        
        plt.title('Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()