"""
@brief Exponential Smoothing Implementation with statsmodels

This example demonstrates the setup and usage of Exponential Smoothing (Holt-Winters)
algorithm using the statsmodels library. The code showcases proper parameter configuration,
model fitting, and forecasting with comprehensive error handling and diagnostics.

@author Hemanth Jadiswami Prabhakaran 7026000
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

def generateSampleData():
    """
    @brief Generate sample time series data with trend and seasonal components
    
    Creates synthetic time series data that exhibits both trending behavior
    and seasonal patterns suitable for Exponential Smoothing demonstration.
    
    @return pd.Series Time series with trend, seasonality, and noise
    @note Uses fixed random seed for reproducible results
    """
    np.random.seed(42)
    # Create 48 months of data (4 years)
    dates = pd.date_range('2020-01-01', periods=48, freq='M')
    
    # Generate base trend component (gradual increase)
    trend = np.linspace(100, 150, 48)
    
    # Add seasonal component (12-month cycle)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(48) / 12)
    
    # Add random noise component
    noise = np.random.normal(0, 3, 48)
    
    # Combine all components into final time series
    ts = pd.Series(trend + seasonal + noise, index=dates, name='sales')
    return ts

def fitExponentialSmoothing(data, **kwargs):
    """
    @brief Fit Exponential Smoothing model with comprehensive error handling
    
    Configures and fits an Exponential Smoothing model using statsmodels
    with automatic parameter optimization and proper error handling.
    
    @param data pd.Series Input time series data for model fitting
    @param kwargs dict Additional parameters for ExponentialSmoothing
    @return statsmodels.tsa.holtwinters.HoltWintersResults Fitted model object
    @raises Exception Model fitting failures with descriptive error messages
    @note Defaults to additive trend and seasonal components
    """
    try:
        # Configure Exponential Smoothing model parameters
        model = ExponentialSmoothing(
            data,
            trend='add',          # Additive trend component
            seasonal='add',       # Additive seasonal component
            seasonal_periods=12,  # Monthly seasonal cycle
            **kwargs
        )
        
        # Fit model with automatic parameter optimization
        fitted_model = model.fit(optimized=True, use_boxcox=False)
        return fitted_model
        
    except Exception as e:
        print(f"Error fitting Exponential Smoothing model: {e}")
        return None

def generateForecasts(model, steps=12):
    """
    @brief Generate forecasts with prediction intervals
    
    Creates future predictions using the fitted Exponential Smoothing model
    with confidence intervals for uncertainty quantification.
    
    @param model statsmodels.tsa.holtwinters.HoltWintersResults Fitted ES model
    @param steps int Number of future periods to forecast
    @return tuple (forecasts, prediction_intervals) Forecast results
    @raises Exception Forecasting failures with error handling
    @warning Prediction intervals may be unreliable for long horizons
    """
    try:
        # Generate point forecasts for specified horizon
        forecasts = model.forecast(steps=steps)
        
        # Calculate prediction intervals (approximate method)
        residuals = model.resid
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        # Create confidence intervals (95% level)
        confidence_level = 1.96  # 95% confidence
        lower_bound = forecasts - confidence_level * std_error
        upper_bound = forecasts + confidence_level * std_error
        
        prediction_intervals = pd.DataFrame({
            'lower': lower_bound,
            'upper': upper_bound
        }, index=forecasts.index)
        
        return forecasts, prediction_intervals
        
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return None, None

def evaluateModel(model, data):
    """
    @brief Evaluate model performance using multiple metrics
    
    Computes comprehensive evaluation metrics for the fitted Exponential
    Smoothing model including error measures and information criteria.
    
    @param model statsmodels.tsa.holtwinters.HoltWintersResults Fitted model
    @param data pd.Series Original time series data
    @return dict Dictionary containing evaluation metrics
    @note Includes MAE, RMSE, MAPE, and information criteria
    """
    try:
        # Extract fitted values and residuals
        fitted_values = model.fittedvalues
        residuals = model.resid
        
        # Calculate error metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / data.iloc[len(residuals):]) * 100)
        
        # Compile evaluation metrics dictionary
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'aic': model.aic,
            'bic': model.bic,
            'alpha': model.params['smoothing_level'],
            'beta': model.params['smoothing_trend'],
            'gamma': model.params['smoothing_seasonal']
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}

def main():
    """
    @brief Main function demonstrating Exponential Smoothing with statsmodels
    
    Orchestrates the complete workflow from data generation through model
    fitting, forecasting, and evaluation using statsmodels implementation.
    
    @note Provides comprehensive example of ES algorithm usage
    """
    print("Exponential Smoothing with statsmodels Example")
    print("=" * 50)
    
    # Generate synthetic time series data
    ts_data = generateSampleData()
    print(f"Generated time series with {len(ts_data)} observations")
    print(f"Data range: {ts_data.min():.2f} to {ts_data.max():.2f}")
    
    # Display basic data statistics
    print(f"Mean: {ts_data.mean():.2f}, Std: {ts_data.std():.2f}")
    
    # Fit Exponential Smoothing model
    print("\nFitting Exponential Smoothing model...")
    model = fitExponentialSmoothing(ts_data)
    
    if model is not None:
        # Display model parameters
        print(f"Model fitted successfully")
        print(f"Alpha (level): {model.params['smoothing_level']:.4f}")
        print(f"Beta (trend): {model.params['smoothing_trend']:.4f}")
        print(f"Gamma (seasonal): {model.params['smoothing_seasonal']:.4f}")
        
        # Evaluate model performance
        metrics = evaluateModel(model, ts_data)
        if metrics:
            print(f"\nModel Performance Metrics:")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  AIC: {metrics['aic']:.2f}")
            print(f"  BIC: {metrics['bic']:.2f}")
        
        # Generate forecasts
        print("\nGenerating forecasts...")
        forecasts, pred_intervals = generateForecasts(model, steps=6)
        
        if forecasts is not None:
            print("Forecast Results:")
            for i, (pred, interval) in enumerate(zip(forecasts, pred_intervals.iterrows()), 1):
                lower, upper = interval[1]['lower'], interval[1]['upper']
                print(f"  Month {i}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
        
        # Display model summary information
        print(f"\nModel Summary:")
        print(f"  Trend: {model.model.trend}")
        print(f"  Seasonal: {model.model.seasonal}")
        print(f"  Seasonal periods: {model.model.seasonal_periods}")
        
    else:
        print("Failed to fit Exponential Smoothing model")

if __name__ == "__main__":
    main()