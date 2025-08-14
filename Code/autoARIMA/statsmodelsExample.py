"""
@brief Auto ARIMA Implementation with statsmodels

This example demonstrates manual parameter selection approach using statsmodels
to implement Auto ARIMA functionality. Shows grid search optimization and
model comparison using information criteria.

@note Requires statsmodels library installation: pip install statsmodels
@warning Grid search can be computationally intensive for large parameter spaces

"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings during grid search

def generate_sample_data():
    """
    @brief Generate sample time series data for demonstration.
    
    Creates a synthetic time series with trend, seasonal, and noise components
    identical to pmdarima example for consistent comparison.
    
    @return Time series with trend and seasonal components
    @rtype pd.Series
    
    @note Uses fixed random seed (42) for reproducible results
    @note Monthly frequency with 100 observations spanning ~8 years
    """
    np.random.seed(42)  # Set seed for reproducible results
    dates = pd.date_range('2020-01-01', periods=100, freq='M')  # Monthly frequency
    
    # Create time series components
    trend = np.linspace(100, 200, 100)  # Linear upward trend from 100 to 200
    seasonal = 10 * np.sin(2 * np.pi * np.arange(100) / 12)  # Annual seasonality (12 months)
    noise = np.random.normal(0, 5, 100)  # Gaussian white noise with std=5
    
    # Combine all components into final time series
    ts = pd.Series(trend + seasonal + noise, index=dates, name='value')
    return ts

def check_stationarity(data):
    """
    @brief Check stationarity using Augmented Dickey-Fuller test.
    
    Performs statistical test to determine if the time series is stationary,
    which is a key assumption for ARIMA models.
    
    @param data Input time series data
    @type data pd.Series
        
    @return True if series is stationary (p-value <= 0.05)
    @rtype bool
    
    @raises Exception Catches any errors during statistical testing
    
    @note Uses 5% significance level for stationarity decision
    @note Stationary series have constant mean and variance over time
    """
    try:
        # Perform Augmented Dickey-Fuller test
        result = adfuller(data.dropna())  # Remove NaN values before testing
        return result[1] <= 0.05  # p-value <= 0.05 suggests stationarity
    except:
        return False  # Return False if test fails

def auto_arima_grid_search(data, max_p=3, max_d=2, max_q=3):
    """
    @brief Implement Auto ARIMA using grid search with statsmodels.
    
    Performs exhaustive search over ARIMA parameter space to find the model
    with lowest AIC (Akaike Information Criterion) value.
    
    @param data Input time series data
    @type data pd.Series
    @param max_p Maximum autoregressive order (default: 3)
    @type max_p int
    @param max_d Maximum differencing order (default: 2)
    @type max_d int
    @param max_q Maximum moving average order (default: 3)
    @type max_q int
        
    @return Tuple containing best fitted model and optimal parameters
    @rtype tuple(statsmodels.tsa.arima.model.ARIMAResults, tuple)
    
    @note AIC balances model fit and complexity (lower is better)
    @warning Computational complexity grows as O(p*d*q)
    """
    best_aic = np.inf  # Initialize with infinite AIC
    best_model = None
    best_params = None
    
    # Generate all possible parameter combinations
    p_values = range(0, max_p + 1)  # Autoregressive orders: 0 to max_p
    d_values = range(0, max_d + 1)  # Differencing orders: 0 to max_d
    q_values = range(0, max_q + 1)  # Moving average orders: 0 to max_q
    
    print("Searching optimal parameters...")
    
    # Iterate through all parameter combinations
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            # Fit ARIMA model with current parameter combination
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Compare AIC values to find best model
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic      # Update best AIC score
                best_model = fitted_model        # Store best model
                best_params = (p, d, q)         # Store optimal parameters
                
        except Exception as e:
            # Skip problematic parameter combinations (e.g., non-convergent)
            continue
    
    return best_model, best_params

def generate_forecasts(model, steps=12):
    """
    @brief Generate forecasts with confidence intervals using statsmodels.
    
    Produces point forecasts and prediction intervals using the fitted ARIMA model
    for the specified number of future time periods.
    
    @param model Fitted ARIMA model from statsmodels
    @type model statsmodels.tsa.arima.model.ARIMAResults
    @param steps Number of forecast steps (default: 12 for one year ahead)
    @type steps int
        
    @return Tuple containing forecasts and confidence intervals, or (None, None) if error
    @rtype tuple(pd.Series, pd.DataFrame) or tuple(None, None)
    
    @raises Exception Catches and logs forecasting errors
    
    @note Confidence intervals provide uncertainty quantification
    @note Uses get_forecast() method for comprehensive forecast results
    """
    try:
        # Generate forecast results object with confidence intervals
        forecast_result = model.get_forecast(steps=steps)
        forecasts = forecast_result.predicted_mean  # Point forecasts
        conf_int = forecast_result.conf_int()       # Confidence intervals DataFrame
        return forecasts, conf_int
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        return None, None

def evaluate_model(model, data):
    """
    @brief Evaluate model performance and diagnostics.
    
    Computes comprehensive model evaluation metrics including information criteria
    and residual statistics for model assessment.
    
    @param model Fitted ARIMA model from statsmodels
    @type model statsmodels.tsa.arima.model.ARIMAResults
    @param data Original time series data (for context)
    @type data pd.Series
        
    @return Dictionary containing model evaluation metrics
    @rtype dict
    
    @raises Exception Catches and logs evaluation errors
    
    @note AIC, BIC, HQIC are information criteria (lower is better)
    @note LLF is log-likelihood function (higher is better)
    """
    try:
        # Compile comprehensive model evaluation metrics
        metrics = {
            'aic': model.aic,           # Akaike Information Criterion
            'bic': model.bic,           # Bayesian Information Criterion
            'hqic': model.hqic,         # Hannan-Quinn Information Criterion
            'llf': model.llf,           # Log-likelihood function value
            'residual_std': model.resid.std()  # Standard deviation of residuals
        }
        return metrics
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}

def main():
    """
    @brief Main function demonstrating Auto ARIMA usage with statsmodels.
    
    Orchestrates the complete manual Auto ARIMA workflow including data generation,
    stationarity testing, grid search optimization, forecasting, and results display.
    
    @note Demonstrates manual implementation of Auto ARIMA functionality
    @note Shows more detailed control over model selection process
    """
    print("Auto ARIMA with statsmodels Example")
    print("=" * 40)
    
    # Generate synthetic time series data for demonstration
    ts_data = generate_sample_data()
    print(f"Generated time series with {len(ts_data)} observations")
    print(f"Data range: {ts_data.min():.2f} to {ts_data.max():.2f}")
    
    # Test for stationarity (important ARIMA assumption)
    is_stationary = check_stationarity(ts_data)
    print(f"Series is stationary: {is_stationary}")
    
    # Perform comprehensive parameter optimization via grid search
    print("\nPerforming parameter optimization...")
    best_model, best_params = auto_arima_grid_search(ts_data)
    
    # Proceed only if optimal model was found
    if best_model is not None:
        # Display optimal model configuration
        print(f"Optimal ARIMA order: {best_params}")
        
        # Evaluate model performance using multiple criteria
        metrics = evaluate_model(best_model, ts_data)
        print(f"Model Performance:")
        for metric, value in metrics.items():  # Display all evaluation metrics
            print(f"  {metric.upper()}: {value:.3f}")
        
        # Generate future predictions with uncertainty quantification
        print("\nGenerating forecasts...")
        forecasts, conf_intervals = generate_forecasts(best_model, steps=6)
        
        # Display forecast results if generation was successful
        if forecasts is not None:
            print("Forecast Results:")
            # Iterate through forecasts and confidence intervals
            for i, (pred, ci_row) in enumerate(zip(forecasts, conf_intervals.values), 1):
                print(f"  Step {i}: {pred:.2f} [{ci_row[0]:.2f}, {ci_row[1]:.2f}]")
        
        # Display comprehensive model summary information
        print(f"\nModel Summary:")
        print(f"  Parameters: {len(best_model.params)}")  # Number of estimated parameters
        print(f"  Observations: {best_model.nobs}")        # Number of observations used
        print(f"  Log-likelihood: {best_model.llf:.2f}")   # Final log-likelihood value
        
    else:
        print("Failed to find optimal ARIMA model")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()