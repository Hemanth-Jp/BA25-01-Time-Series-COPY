"""
Pipeline Integration and Model Persistence with Pmdarima 

This module demonstrates production-ready forecasting workflows using pmdarima
pipelines, advanced preprocessing, model serialization, and deployment patterns.


@version: 1.1 
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer, FourierFeaturizer
from pmdarima.model_selection import train_test_split
import pickle
import joblib
import warnings
from datetime import datetime, timedelta
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ProductionForecastingPipeline:
    """
    Production-ready forecasting pipeline with pmdarima.
    
    This class encapsulates the complete forecasting workflow including
    data preprocessing, model training, validation, and deployment.
    """
    
    def __init__(self, seasonal_period=12, forecast_horizon=12):
        """
        Initialize the forecasting pipeline.
        
        Args:
            seasonal_period (int): Seasonal period for the data
            forecast_horizon (int): Default forecast horizon
        """
        self.seasonal_period = seasonal_period
        self.forecast_horizon = forecast_horizon
        self.pipeline = None
        self.is_fitted = False
        self.model_metadata = {}
        
    def create_pipeline(self, use_boxcox=True, use_fourier=True):
        """
        Create preprocessing and modeling pipeline.
        
        Args:
            use_boxcox (bool): Whether to apply Box-Cox transformation
            use_fourier (bool): Whether to add Fourier features
            
        Returns:
            pmdarima.pipeline.Pipeline: Configured pipeline
        """
        steps = []
        
        # Add Box-Cox transformation if requested
        if use_boxcox:
            steps.append(('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)))
        
        # Add Fourier features if requested and seasonal period > 2
        # Fourier features require k <= m//2, so we need m >= 4 for k=2
        if use_fourier and self.seasonal_period > 2:
            k = min(2, self.seasonal_period // 2)
            steps.append(('fourier', FourierFeaturizer(m=self.seasonal_period, k=k)))
        
        # Determine if model should be seasonal
        is_seasonal = self.seasonal_period > 1
        
        # Configure ARIMA parameters based on seasonality
        if is_seasonal:
            # Seasonal ARIMA configuration
            arima_params = {
                'seasonal': True,
                'm': self.seasonal_period,
                'max_p': 3, 'max_q': 3,
                'max_P': 2, 'max_Q': 2,
                'start_P': 0, 'start_Q': 0,
                'max_D': 1
            }
        else:
            # Non-seasonal ARIMA configuration
            arima_params = {
                'seasonal': False,
                'm': 1,
                'max_p': 5, 'max_q': 5,  # Allow higher orders for non-seasonal
                'max_P': 0, 'max_Q': 0,  # No seasonal components
                'start_P': 0, 'start_Q': 0,
                'max_D': 0  # No seasonal differencing
            }
        
        # Add ARIMA model with proper configuration
        steps.append(('arima', pm.AutoARIMA(
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            random_state=42,
            n_jobs=1,  # Use single job for stability
            **arima_params
        )))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit(self, y, X=None):
        """
        Fit the forecasting pipeline.
        
        Args:
            y (pandas.Series): Target time series
            X (pandas.DataFrame, optional): Exogenous variables
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        print("Fitting forecasting pipeline...")
        start_time = datetime.now()
        
        # Fit the pipeline
        self.pipeline.fit(y, X=X)
        
        fit_time = datetime.now() - start_time
        self.is_fitted = True
        
        # Store metadata - access the fitted model attributes correctly
        fitted_model = self.pipeline.named_steps['arima'].model_
        self.model_metadata = {
            'fit_time': fit_time,
            'data_shape': len(y),
            'model_order': fitted_model.order,
            'seasonal_order': fitted_model.seasonal_order,
            'aic': fitted_model.aic(),
            'bic': fitted_model.bic(),
            'fit_date': datetime.now()
        }
        
        print(f"Pipeline fitted in {fit_time.total_seconds():.2f} seconds")
        print(f"Final model: ARIMA{self.model_metadata['model_order']} x {self.model_metadata['seasonal_order']}")
        
    def predict(self, n_periods=None, X=None, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts using the fitted pipeline.
        
        Args:
            n_periods (int): Number of periods to forecast
            X (pandas.DataFrame): Future exogenous variables
            return_conf_int (bool): Whether to return confidence intervals
            alpha (float): Significance level for confidence intervals
            
        Returns:
            tuple or array: Forecasts and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        if n_periods is None:
            n_periods = self.forecast_horizon
        
        return self.pipeline.predict(
            n_periods=n_periods,
            X=X,
            return_conf_int=return_conf_int,
            alpha=alpha
        )
    
    def save_model(self, filepath, method='pickle'):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath (str): Path to save the model
            method (str): Serialization method ('pickle' or 'joblib')
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        # Create directory if it doesn't exist and filesystem is writable
        try:
            if os.path.dirname(filepath):  # Only if there's a directory component
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot create directory. Using current directory. Error: {e}")
            # Use just the filename in current directory
            filepath = os.path.basename(filepath)
        
        model_data = {
            'pipeline': self.pipeline,
            'metadata': self.model_metadata,
            'seasonal_period': self.seasonal_period,
            'forecast_horizon': self.forecast_horizon
        }
        
        try:
            if method == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
            elif method == 'joblib':
                joblib.dump(model_data, filepath)
            else:
                raise ValueError("Method must be 'pickle' or 'joblib'")
            
            print(f"Model saved to {filepath} using {method}")
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot save model to disk. Error: {e}")
            print("Model remains in memory for this session.")
    
    @classmethod
    def load_model(cls, filepath, method='pickle'):
        """
        Load a saved pipeline from disk.
        
        Args:
            filepath (str): Path to the saved model
            method (str): Serialization method ('pickle' or 'joblib')
            
        Returns:
            ProductionForecastingPipeline: Loaded pipeline instance
        """
        try:
            if method == 'pickle':
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            elif method == 'joblib':
                model_data = joblib.load(filepath)
            else:
                raise ValueError("Method must be 'pickle' or 'joblib'")
            
            # Reconstruct the pipeline instance
            instance = cls(
                seasonal_period=model_data['seasonal_period'],
                forecast_horizon=model_data['forecast_horizon']
            )
            instance.pipeline = model_data['pipeline']
            instance.model_metadata = model_data['metadata']
            instance.is_fitted = True
            
            print(f"Model loaded from {filepath}")
            return instance
            
        except (OSError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Cannot load model from {filepath}. Error: {e}")
            print("Returning None - model persistence demonstration skipped.")
            return None
    
    def get_model_info(self):
        """
        Get comprehensive model information.
        
        Returns:
            dict: Model information and metadata
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "model_order": self.model_metadata['model_order'],
            "seasonal_order": self.model_metadata['seasonal_order'],
            "aic": self.model_metadata['aic'],
            "bic": self.model_metadata['bic'],
            "fit_time": self.model_metadata['fit_time'].total_seconds(),
            "data_points": self.model_metadata['data_shape'],
            "fit_date": self.model_metadata['fit_date'].strftime('%Y-%m-%d %H:%M:%S'),
            "pipeline_steps": list(self.pipeline.named_steps.keys())
        }

def demonstrate_pipeline_workflow():
    """
    Demonstrate complete pipeline workflow with real data.
    
    Returns:
        ProductionForecastingPipeline: Trained pipeline instance
    """
    print("=== Pipeline Workflow Demonstration ===\n")
    
    # Load sample data
    y = pm.datasets.load_lynx()  # Annual lynx trappings data
    
    # Create date index
    dates = pd.date_range(start='1821', periods=len(y), freq='A')
    ts_data = pd.Series(y, index=dates, name='Lynx_Trappings')
    
    print(f"Dataset: {len(ts_data)} annual observations")
    print(f"Period: {ts_data.index[0].year} - {ts_data.index[-1].year}")
    
    # Split data
    train_size = int(len(ts_data) * 0.85)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    print(f"Training: {len(train_data)} observations")
    print(f"Testing: {len(test_data)} observations")
    
    # Create and fit pipeline - NO Fourier features for annual data
    pipeline = ProductionForecastingPipeline(seasonal_period=1, forecast_horizon=10)
    pipeline.create_pipeline(use_boxcox=True, use_fourier=False)  # Disable Fourier for annual data
    pipeline.fit(train_data)
    
    # Generate forecasts
    forecasts, conf_int = pipeline.predict(n_periods=len(test_data), return_conf_int=True)
    
    # Calculate performance
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_data, forecasts)
        rmse = np.sqrt(mean_squared_error(test_data, forecasts))
        
        print(f"\nPipeline Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    except ImportError:
        print("\nSkipping performance metrics (sklearn not available)")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label='Training', alpha=0.7)
    plt.plot(test_data.index, test_data.values, label='Actual', linewidth=2)
    plt.plot(test_data.index, forecasts, label='Pipeline Forecast', linewidth=2)
    plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], 
                     alpha=0.2, label='95% CI')
    plt.title('Production Pipeline Forecasting Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return pipeline

def demonstrate_seasonal_pipeline():
    """
    Demonstrate pipeline with seasonal data (includes Fourier features).
    
    Returns:
        ProductionForecastingPipeline: Trained seasonal pipeline
    """
    print("\n=== Seasonal Pipeline Demonstration ===\n")
    
    # Load seasonal data
    y = pm.datasets.load_wineind()  # Monthly wine sales
    dates = pd.date_range(start='1980-01', periods=len(y), freq='M')
    ts_data = pd.Series(y, index=dates, name='Wine_Sales')
    
    print(f"Dataset: {len(ts_data)} monthly observations")
    print(f"Period: {ts_data.index[0].strftime('%Y-%m')} to {ts_data.index[-1].strftime('%Y-%m')}")
    
    # Split data
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    print(f"Training: {len(train_data)} observations")
    print(f"Testing: {len(test_data)} observations")
    
    # Create and fit seasonal pipeline WITH Fourier features
    pipeline = ProductionForecastingPipeline(seasonal_period=12, forecast_horizon=12)
    pipeline.create_pipeline(use_boxcox=True, use_fourier=True)  # Enable Fourier for seasonal data
    pipeline.fit(train_data)
    
    # Generate forecasts
    forecasts, conf_int = pipeline.predict(n_periods=len(test_data), return_conf_int=True)
    
    # Calculate performance
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_data, forecasts)
        rmse = np.sqrt(mean_squared_error(test_data, forecasts))
        
        print(f"\nSeasonal Pipeline Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    except ImportError:
        print("\nSkipping performance metrics (sklearn not available)")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label='Training', alpha=0.7)
    plt.plot(test_data.index, test_data.values, label='Actual', linewidth=2)
    plt.plot(test_data.index, forecasts, label='Pipeline Forecast', linewidth=2)
    plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], 
                     alpha=0.2, label='95% CI')
    plt.title('Seasonal Pipeline Forecasting Results (with Fourier Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return pipeline

def demonstrate_model_persistence():
    """
    Demonstrate model saving and loading capabilities.
    """
    print("\n=== Model Persistence Demonstration ===\n")
    
    # Create simple dataset
    y = pm.datasets.load_wineind()
    dates = pd.date_range(start='1980-01', periods=len(y), freq='M')
    ts_data = pd.Series(y, index=dates)
    
    # Train pipeline
    pipeline = ProductionForecastingPipeline(seasonal_period=12)
    pipeline.fit(ts_data)
    
    # Try to save model (will handle read-only filesystem gracefully)
    model_path = 'production_model.pkl'  # Use current directory
    pipeline.save_model(model_path, method='pickle')
    
    # Try to load model
    loaded_pipeline = ProductionForecastingPipeline.load_model(model_path, method='pickle')
    
    if loaded_pipeline is not None:
        # Compare predictions
        original_pred = pipeline.predict(n_periods=12, return_conf_int=False)
        loaded_pred = loaded_pipeline.predict(n_periods=12, return_conf_int=False)
        
        # Verify predictions match
        prediction_match = np.allclose(original_pred, loaded_pred)
        print(f"Predictions match after loading: {prediction_match}")
        
        # Display model info
        model_info = loaded_pipeline.get_model_info()
        print(f"\nLoaded Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
            
        return loaded_pipeline
    else:
        print("Model persistence demonstration skipped due to filesystem constraints.")
        print("In production environments, ensure write permissions are available.")
        
        # Show that the original pipeline still works
        print(f"\nOriginal Pipeline Information:")
        info = pipeline.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        return pipeline

def create_deployment_example():
    """
    Create example deployment script for production use.
    
    Returns:
        str: Deployment script code
    """
    deployment_code = '''
# Production Deployment Example
import pandas as pd
from datetime import datetime, timedelta

def production_forecast_service(model_path, data_path, forecast_periods=12):
    """
    Production forecasting service function.
    
    Args:
        model_path (str): Path to saved model
        data_path (str): Path to input data
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        dict: Forecast results with metadata
    """
    # Load the trained model
    pipeline = ProductionForecastingPipeline.load_model(model_path)
    
    # Load new data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Generate forecasts
    forecasts, conf_int = pipeline.predict(
        n_periods=forecast_periods, 
        return_conf_int=True
    )
    
    # Create forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_periods,
        freq='M'  # Adjust frequency as needed
    )
    
    # Format results
    results = {
        'forecasts': forecasts.tolist(),
        'lower_bound': conf_int[:, 0].tolist(),
        'upper_bound': conf_int[:, 1].tolist(),
        'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
        'model_info': pipeline.get_model_info(),
        'generation_time': datetime.now().isoformat()
    }
    
    return results

# Example usage:
# results = production_forecast_service('models/prod_model.pkl', 'data/latest.csv')
'''
    
    return deployment_code

def main():
    """
    Main function demonstrating complete pipeline integration workflow.
    """
    print("=== Pipeline Integration and Model Persistence (FIXED) ===\n")
    
    # Demonstrate non-seasonal pipeline workflow
    print("1. Non-seasonal (Annual) Data Pipeline:")
    trained_pipeline = demonstrate_pipeline_workflow()
    
    # Show pipeline info
    print(f"\nTrained Pipeline Information:")
    info = trained_pipeline.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Demonstrate seasonal pipeline workflow
    print("\n" + "="*50)
    print("2. Seasonal (Monthly) Data Pipeline:")
    seasonal_pipeline = demonstrate_seasonal_pipeline()
    
    # Show seasonal pipeline info
    print(f"\nSeasonal Pipeline Information:")
    seasonal_info = seasonal_pipeline.get_model_info()
    for key, value in seasonal_info.items():
        print(f"  {key}: {value}")
    
    # Demonstrate model persistence
    print("\n" + "="*50)
    loaded_pipeline = demonstrate_model_persistence()
    
    # Show deployment example
    print(f"\n=== Production Deployment Pattern ===")
    deployment_code = create_deployment_example()
    print("Generated deployment template for production use.")
    print("Key features:")
    print("- Model loading and validation")
    print("- Automated forecast generation")
    print("- Structured output format")
    print("- Error handling and metadata")
    
    plt.show()
    
    print(f"\n=== Pipeline Integration Complete ===")
    print("Production-ready forecasting pipeline successfully demonstrated!")
    print("\nKey fixes applied:")
    print("- Conditional Fourier features (only for seasonal_period > 2)")
    print("- Automatic k calculation (k = min(2, m//2))")
    print("- Separate seasonal and non-seasonal model configurations")
    print("- Enhanced error handling and validation")

if __name__ == "__main__":
    main()