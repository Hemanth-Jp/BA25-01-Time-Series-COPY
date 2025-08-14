"""
Seasonal ARIMA with Exogenous Variables using Pmdarima 

This module demonstrates advanced seasonal ARIMA modeling with exogenous variables,
comprehensive model diagnostics, and production-ready forecasting workflows.


@version: 1.1 
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.model_selection import train_test_split
from pmdarima.arima import ndiffs, nsdiffs
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf  # Fixed import
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_synthetic_data_with_exog():
    """
    Create synthetic seasonal time series with exogenous variables.
    
    Returns:
        tuple: (endogenous_series, exogenous_dataframe)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range (5 years of monthly data)
    dates = pd.date_range(start='2019-01', end='2023-12', freq='M')
    n_periods = len(dates)
    
    # Create trend component
    trend = np.linspace(100, 200, n_periods)
    
    # Create seasonal component (annual seasonality)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    
    # Create exogenous variables
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12) + np.random.normal(0, 2, n_periods)
    marketing_spend = np.random.uniform(50, 150, n_periods)
    economic_index = 100 + np.cumsum(np.random.normal(0, 1, n_periods))
    
    # Combine components with exogenous effects
    exog_effect = 0.5 * temperature + 0.3 * marketing_spend + 0.1 * economic_index
    noise = np.random.normal(0, 5, n_periods)
    
    # Final endogenous series
    y = trend + seasonal + exog_effect + noise
    
    # Create pandas objects
    endogenous = pd.Series(y, index=dates, name='Sales')
    exogenous = pd.DataFrame({
        'temperature': temperature,
        'marketing_spend': marketing_spend,
        'economic_index': economic_index
    }, index=dates)
    
    print(f"Created synthetic dataset: {len(endogenous)} observations")
    print(f"Exogenous variables: {list(exogenous.columns)}")
    
    return endogenous, exogenous

def perform_stationarity_analysis(data):
    """
    Perform comprehensive stationarity analysis.
    
    Args:
        data (pandas.Series): Time series data
        
    Returns:
        dict: Analysis results
    """
    print("\n=== Stationarity Analysis ===")
    
    # Test for non-seasonal differencing
    n_diffs = ndiffs(data, test='adf', max_d=3)
    print(f"Recommended non-seasonal differencing (d): {n_diffs}")
    
    # Test for seasonal differencing  
    n_seasonal_diffs = nsdiffs(data, m=12, max_D=2)
    print(f"Recommended seasonal differencing (D): {n_seasonal_diffs}")
    
    # Perform ADF test manually for detailed results
    from pmdarima.arima.stationarity import ADFTest
    adf_test = ADFTest(alpha=0.05)
    adf_result = adf_test.should_diff(data)
    print(f"ADF Test - Should difference: {adf_result[0]}")
    print(f"ADF Test - P-value: {adf_result[1]:.4f}")
    
    return {
        'n_diffs': n_diffs,
        'n_seasonal_diffs': n_seasonal_diffs,
        'adf_should_diff': adf_result[0],
        'adf_pvalue': adf_result[1]
    }

def fit_sarimax_model(y_train, X_train, y_test, X_test):
    """
    Fit SARIMAX model with comprehensive parameter search.
    
    Args:
        y_train, X_train: Training data
        y_test, X_test: Test data
        
    Returns:
        tuple: (best_model, forecasts, performance_metrics)
    """
    print("\n=== SARIMAX Model Fitting ===")
    
    # Perform stationarity analysis
    stationarity_results = perform_stationarity_analysis(y_train)
    
    # Configure auto_arima with exogenous variables
    model = pm.auto_arima(
        y_train, 
        X=X_train,
        seasonal=True,
        m=12,
        d=stationarity_results['n_diffs'],
        D=stationarity_results['n_seasonal_diffs'],
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        information_criterion='aic',
        out_of_sample_size=int(len(y_train) * 0.1),
        random_state=42
    )
    
    print(f"Best SARIMAX model: {model.order} x {model.seasonal_order}")
    print(f"AIC: {model.aic():.2f}")
    print(f"BIC: {model.bic():.2f}")
    
    # Generate forecasts with exogenous variables
    forecasts, conf_int = model.predict(
        n_periods=len(y_test),
        X=X_test,
        return_conf_int=True,
        alpha=0.05
    )
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))
    mape = np.mean(np.abs((y_test - forecasts) / y_test)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'AIC': model.aic(),
        'BIC': model.bic()
    }
    
    print(f"\nModel Performance on Test Set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    return model, forecasts, conf_int, metrics

def analyze_model_diagnostics(model, y_train):
    """
    Perform comprehensive model diagnostics.
    
    Args:
        model: Fitted pmdarima model
        y_train: Training data
    """
    print("\n=== Model Diagnostics ===")
    
    # Get residuals
    residuals = model.resid()
    
    # Basic residual statistics
    print(f"Residual mean: {np.mean(residuals):.4f}")
    print(f"Residual std: {np.std(residuals):.4f}")
    print(f"Residual skewness: {pd.Series(residuals).skew():.4f}")
    print(f"Residual kurtosis: {pd.Series(residuals).kurtosis():.4f}")
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals plot
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=20, density=True, alpha=0.7)
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ACF of residuals (Fixed import)
    try:
        lags = np.arange(1, min(20, len(residuals)//4))
        acf_vals = acf(residuals, nlags=len(lags), fft=False)
        axes[1, 1].stem(lags, acf_vals[1:])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Residuals ACF')
        axes[1, 1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: Could not plot ACF due to: {e}")
        axes[1, 1].text(0.5, 0.5, 'ACF plot unavailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Residuals ACF (Error)')
    
    plt.tight_layout()
    return fig

def create_forecast_visualization(y_train, y_test, forecasts, conf_int, exog_vars):
    """
    Create comprehensive forecast visualization.
    
    Args:
        y_train, y_test: Training and test data
        forecasts: Model forecasts
        conf_int: Confidence intervals
        exog_vars: Exogenous variables for additional plots
        
    Returns:
        matplotlib.figure.Figure: Complete visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main forecast plot
    ax1 = axes[0, 0]
    ax1.plot(y_train.index, y_train.values, label='Training', color='blue', alpha=0.7)
    ax1.plot(y_test.index, y_test.values, label='Actual', color='red', linewidth=2)
    ax1.plot(y_test.index, forecasts, label='Forecast', color='green', linewidth=2)
    ax1.fill_between(y_test.index, conf_int[:, 0], conf_int[:, 1], 
                     color='green', alpha=0.2, label='95% CI')
    ax1.set_title('SARIMAX Forecast Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Forecast vs Actual scatter
    ax2 = axes[0, 1]
    ax2.scatter(y_test.values, forecasts, alpha=0.6)
    min_val, max_val = min(y_test.min(), forecasts.min()), max(y_test.max(), forecasts.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Forecast vs Actual')
    ax2.grid(True, alpha=0.3)
    
    # Exogenous variable effects
    ax3 = axes[1, 0]
    ax3.plot(exog_vars.index, exog_vars['temperature'], label='Temperature', alpha=0.7)
    ax3.set_ylabel('Temperature', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(exog_vars.index, exog_vars['marketing_spend'], 
                  label='Marketing', color='orange', alpha=0.7)
    ax3_twin.set_ylabel('Marketing Spend', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.set_title('Exogenous Variables')
    ax3.grid(True, alpha=0.3)
    
    # Forecast errors
    ax4 = axes[1, 1]
    errors = y_test.values - forecasts
    ax4.plot(y_test.index, errors, marker='o', linestyle='-', alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_title('Forecast Errors')
    ax4.set_ylabel('Error')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def perform_ljung_box_test(residuals):
    """
    Perform Ljung-Box test for residual autocorrelation.
    
    Args:
        residuals: Model residuals
        
    Returns:
        dict: Test results
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        print(f"\nLjung-Box Test Results:")
        print(f"P-values for lags 1-10: {lb_result['lb_pvalue'].iloc[:3].values}")
        
        # Check if any p-values are significant (< 0.05)
        significant_lags = lb_result[lb_result['lb_pvalue'] < 0.05].index.tolist()
        if significant_lags:
            print(f"Significant autocorrelation at lags: {significant_lags}")
        else:
            print("No significant autocorrelation detected")
            
        return {
            'test_results': lb_result,
            'significant_lags': significant_lags
        }
    except ImportError:
        print("Warning: statsmodels not available for Ljung-Box test")
        return None

def main():
    """
    Main function demonstrating advanced SARIMAX modeling workflow.
    """
    print("=== Advanced Seasonal ARIMA with Exogenous Variables ===\n")
    
    # Create synthetic data with exogenous variables
    y, X = create_synthetic_data_with_exog()
    
    # Split data for training and testing
    split_idx = int(len(y) * 0.8)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    print(f"Training period: {y_train.index[0]} to {y_train.index[-1]}")
    print(f"Test period: {y_test.index[0]} to {y_test.index[-1]}")
    
    # Fit SARIMAX model
    model, forecasts, conf_int, metrics = fit_sarimax_model(y_train, X_train, y_test, X_test)
    
    # Perform additional residual tests
    residuals = model.resid()
    ljung_box_results = perform_ljung_box_test(residuals)
    
    # Perform model diagnostics
    diag_fig = analyze_model_diagnostics(model, y_train)
    diag_fig.suptitle('SARIMAX Model Diagnostics', fontsize=14, fontweight='bold')
    
    # Create comprehensive visualization
    viz_fig = create_forecast_visualization(y_train, y_test, forecasts, conf_int, X)
    viz_fig.suptitle('SARIMAX Forecasting Results', fontsize=14, fontweight='bold')
    
    # Display model summary
    print(f"\n=== Final Model Summary ===")
    print(f"Model: SARIMAX{model.order}x{model.seasonal_order}")
    print(f"Exogenous variables: {list(X.columns)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    
    plt.show()
    
    print("\n=== Advanced SARIMAX Modeling Complete ===")

if __name__ == "__main__":
    main()