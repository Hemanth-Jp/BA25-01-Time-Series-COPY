"""
@brief Custom Exponential Smoothing Implementation

This example demonstrates a manual implementation of the Holt-Winters Exponential
Smoothing algorithm with parameter optimization using scipy. The code provides
educational insight into the algorithm mechanics and component decomposition.

@author Hemanth Jadiswami Prabhakaran 7026000
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CustomExponentialSmoothing:
    """
    @brief Custom implementation of Holt-Winters Exponential Smoothing
    
    Implements the complete Holt-Winters algorithm with additive trend and
    seasonal components, including parameter optimization and forecasting.
    """
    
    def __init__(self, seasonal_periods=12):
        """
        @brief Initialize Exponential Smoothing model
        
        @param seasonal_periods int Number of periods in seasonal cycle
        @note Defaults to 12 for monthly data
        """
        self.seasonal_periods = seasonal_periods
        self.alpha = None  # Level smoothing parameter
        self.beta = None   # Trend smoothing parameter  
        self.gamma = None  # Seasonal smoothing parameter
        self.fitted_values = None
        self.residuals = None
        self.level = None
        self.trend = None
        self.seasonal = None
        
    def _initializeComponents(self, data):
        """
        @brief Initialize level, trend, and seasonal components
        
        Uses simple methods to estimate initial values for the three
        components before beginning the exponential smoothing process.
        
        @param data pd.Series Input time series data
        @return tuple (initial_level, initial_trend, initial_seasonal)
        @note Uses first year of data for seasonal initialization
        """
        n = len(data)
        m = self.seasonal_periods
        
        # Initialize level as mean of first season
        initial_level = np.mean(data[:m])
        
        # Initialize trend using linear regression on first two seasons
        if n >= 2 * m:
            first_season = np.mean(data[:m])
            second_season = np.mean(data[m:2*m])
            initial_trend = (second_season - first_season) / m
        else:
            initial_trend = 0
            
        # Initialize seasonal indices
        initial_seasonal = np.zeros(m)
        for i in range(m):
            # Average seasonal effect across available years
            seasonal_data = data[i::m]
            if len(seasonal_data) > 0:
                initial_seasonal[i] = np.mean(seasonal_data) - initial_level
                
        return initial_level, initial_trend, initial_seasonal
    
    def _exponentialSmoothing(self, data, alpha, beta, gamma):
        """
        @brief Apply Holt-Winters exponential smoothing equations
        
        Implements the core recursive equations for level, trend, and
        seasonal component updating throughout the time series.
        
        @param data pd.Series Input time series data
        @param alpha float Level smoothing parameter (0 < alpha <= 1)
        @param beta float Trend smoothing parameter (0 <= beta <= 1)
        @param gamma float Seasonal smoothing parameter (0 <= gamma <= 1)
        @return np.array Fitted values from the smoothing process
        @note Uses additive formulation for both trend and seasonality
        """
        n = len(data)
        m = self.seasonal_periods
        
        # Initialize components
        level, trend, seasonal = self._initializeComponents(data)
        
        # Storage for component evolution
        levels = np.zeros(n)
        trends = np.zeros(n)
        seasonals = np.zeros(n)
        fitted = np.zeros(n)
        
        # Apply exponential smoothing equations
        for t in range(n):
            # Get seasonal index (cycle through seasonal periods)
            s_index = t % m
            
            if t == 0:
                # First observation uses initial values
                levels[t] = level
                trends[t] = trend
                seasonals[t] = seasonal[s_index]
                fitted[t] = level + trend + seasonal[s_index]
            else:
                # Update level component (equation 1)
                prev_level = levels[t-1]
                prev_trend = trends[t-1]
                prev_seasonal = seasonals[t-m] if t >= m else seasonal[s_index]
                
                levels[t] = alpha * (data.iloc[t] - prev_seasonal) + (1 - alpha) * (prev_level + prev_trend)
                
                # Update trend component (equation 2)  
                trends[t] = beta * (levels[t] - prev_level) + (1 - beta) * prev_trend
                
                # Update seasonal component (equation 3)
                if t >= m:
                    seasonals[t] = gamma * (data.iloc[t] - levels[t]) + (1 - gamma) * seasonals[t-m]
                else:
                    seasonals[t] = seasonal[s_index]
                
                # Calculate fitted value
                fitted[t] = levels[t] + trends[t] + seasonals[t]
        
        # Store final component states
        self.level = levels[-1]
        self.trend = trends[-1]
        self.seasonal = seasonals[-m:] if n >= m else seasonal
        
        return fitted
    
    def _objectiveFunction(self, params, data):
        """
        @brief Objective function for parameter optimization
        
        Calculates sum of squared errors for given parameter values,
        used by optimization algorithm to find optimal smoothing parameters.
        
        @param params np.array Parameter values [alpha, beta, gamma]
        @param data pd.Series Input time series data
        @return float Sum of squared errors
        @note Includes parameter bounds checking
        """
        alpha, beta, gamma = params
        
        # Check parameter bounds
        if not (0 < alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
            return np.inf
            
        # Calculate fitted values
        fitted = self._exponentialSmoothing(data, alpha, beta, gamma)
        
        # Compute sum of squared errors
        errors = data.values - fitted
        sse = np.sum(errors**2)
        
        return sse
    
    def fit(self, data):
        """
        @brief Fit Exponential Smoothing model with parameter optimization
        
        Optimizes smoothing parameters using scipy minimize and fits
        the complete Holt-Winters model to the input time series.
        
        @param data pd.Series Input time series data for fitting
        @return self Fitted model instance for method chaining
        @raises Exception Optimization failures or invalid data
        """
        try:
            # Initial parameter guesses
            initial_params = [0.3, 0.1, 0.1]  # [alpha, beta, gamma]
            
            # Parameter bounds for optimization
            bounds = [(0.01, 1.0), (0.0, 1.0), (0.0, 1.0)]
            
            # Optimize parameters using L-BFGS-B method
            result = minimize(
                self._objectiveFunction,
                initial_params,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                # Extract optimized parameters
                self.alpha, self.beta, self.gamma = result.x
                
                # Calculate final fitted values and residuals
                self.fitted_values = self._exponentialSmoothing(data, self.alpha, self.beta, self.gamma)
                self.residuals = data.values - self.fitted_values
                
                return self
            else:
                raise Exception(f"Optimization failed: {result.message}")
                
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None
    
    def forecast(self, steps=12):
        """
        @brief Generate forecasts using fitted model
        
        Produces future predictions by extending the level, trend, and
        seasonal components according to Holt-Winters equations.
        
        @param steps int Number of future periods to forecast
        @return np.array Forecast values for specified horizon
        @raises Exception Missing fitted model or invalid parameters
        @warning Forecasts assume continuation of current patterns
        """
        if self.level is None or self.trend is None or self.seasonal is None:
            raise Exception("Model must be fitted before forecasting")
            
        forecasts = np.zeros(steps)
        
        # Generate forecasts using final component states
        for h in range(steps):
            # Seasonal index cycles through seasonal periods
            seasonal_index = h % self.seasonal_periods
            seasonal_component = self.seasonal[seasonal_index]
            
            # Forecast equation: level + h*trend + seasonal
            forecasts[h] = self.level + (h + 1) * self.trend + seasonal_component
            
        return forecasts

def generateSampleData():
    """
    @brief Generate sample time series with trend and seasonality
    
    @return pd.Series Synthetic time series for algorithm demonstration
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=48, freq='M')
    
    # Create components: trend + seasonal + noise
    trend = np.linspace(100, 150, 48)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.normal(0, 3, 48)
    
    ts = pd.Series(trend + seasonal + noise, index=dates, name='sales')
    return ts

def main():
    """
    @brief Main function demonstrating custom Exponential Smoothing
    
    Orchestrates complete workflow using custom implementation including
    parameter optimization, model fitting, and forecast generation.
    """
    print("Custom Exponential Smoothing Implementation")
    print("=" * 50)
    
    # Generate sample data
    ts_data = generateSampleData()
    print(f"Generated time series with {len(ts_data)} observations")
    print(f"Data range: {ts_data.min():.2f} to {ts_data.max():.2f}")
    
    # Initialize and fit custom model
    print("\nFitting custom Exponential Smoothing model...")
    model = CustomExponentialSmoothing(seasonal_periods=12)
    fitted_model = model.fit(ts_data)
    
    if fitted_model is not None:
        # Display optimized parameters
        print(f"Optimized Parameters:")
        print(f"  Alpha (level): {model.alpha:.4f}")
        print(f"  Beta (trend): {model.beta:.4f}")
        print(f"  Gamma (seasonal): {model.gamma:.4f}")
        
        # Calculate and display error metrics
        rmse = np.sqrt(np.mean(model.residuals**2))
        mae = np.mean(np.abs(model.residuals))
        print(f"\nModel Performance:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        
        # Generate and display forecasts
        print("\nGenerating forecasts...")
        forecasts = model.forecast(steps=6)
        
        print("Forecast Results:")
        for i, pred in enumerate(forecasts, 1):
            print(f"  Period {i}: {pred:.2f}")
            
        # Display final component states
        print(f"\nFinal Component States:")
        print(f"  Level: {model.level:.2f}")
        print(f"  Trend: {model.trend:.4f}")
        print(f"  Seasonal (last 3): {model.seasonal[-3:]}")
        
    else:
        print("Failed to fit custom Exponential Smoothing model")

if __name__ == "__main__":
    main()