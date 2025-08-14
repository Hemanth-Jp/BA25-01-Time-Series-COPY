"""
Comprehensive Error Handling for Pmdarima

This module demonstrates robust error handling patterns for production
pmdarima applications, including convergence failures, data validation,
and graceful degradation strategies.


@version: 1.0
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import ndiffs, nsdiffs
from pmdarima.arima.stationarity import ADFTest
import warnings
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustARIMAForecaster:
    """
    Production-ready ARIMA forecaster with comprehensive error handling.
    """
    
    def __init__(self, seasonal_period: int = 12, max_retries: int = 3):
        """
        Initialize robust ARIMA forecaster.
        
        Args:
            seasonal_period: Seasonal period for the data
            max_retries: Maximum number of retry attempts
        """
        self.seasonal_period = seasonal_period
        self.max_retries = max_retries
        self.model = None
        self.fitted = False
        self.fallback_model = None
        
    def validate_data(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Comprehensive data validation.
        
        Args:
            y: Endogenous time series
            X: Exogenous variables (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if data is empty
            if len(y) == 0:
                return False, "Time series data is empty"
            
            # Check for minimum data length
            if len(y) < 2 * self.seasonal_period:
                return False, f"Insufficient data: need at least {2 * self.seasonal_period} observations"
            
            # Check for missing values
            if y.isnull().any():
                return False, "Time series contains missing values"
            
            # Check for infinite values
            if np.isinf(y).any():
                return False, "Time series contains infinite values"
            
            # Check for constant series
            if y.nunique() <= 1:
                return False, "Time series is constant"
            
            # Check for negative values (if Box-Cox will be applied)
            if (y <= 0).any():
                logger.warning("Time series contains non-positive values - Box-Cox transformation unavailable")
            
            # Validate exogenous variables if provided
            if X is not None:
                if len(X) != len(y):
                    return False, "Exogenous variables length doesn't match endogenous series"
                
                if X.isnull().any().any():
                    return False, "Exogenous variables contain missing values"
                
                if np.isinf(X).any().any():
                    return False, "Exogenous variables contain infinite values"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
    
    def preprocess_data(self, y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Robust data preprocessing with error handling.
        
        Args:
            y: Input time series
            
        Returns:
            Tuple of (processed_series, preprocessing_info)
        """
        preprocessing_info = {}
        processed_y = y.copy()
        
        try:
            # Handle outliers using IQR method
            Q1 = processed_y.quantile(0.25)
            Q3 = processed_y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (processed_y < lower_bound) | (processed_y > upper_bound)
            n_outliers = outliers_mask.sum()
            
            if n_outliers > 0:
                logger.warning(f"Found {n_outliers} outliers, capping extreme values")
                processed_y = processed_y.clip(lower_bound, upper_bound)
                preprocessing_info['outliers_handled'] = n_outliers
            
            # Log transformation for highly skewed data
            if processed_y.skew() > 2 and (processed_y > 0).all():
                logger.info("Applying log transformation for skewed data")
                processed_y = np.log(processed_y)
                preprocessing_info['log_transformed'] = True
            
            preprocessing_info['original_length'] = len(y)
            preprocessing_info['final_length'] = len(processed_y)
            
            return processed_y, preprocessing_info
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return y, {'error': str(e)}
    
    def fit_with_fallback(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> bool:
        """
        Fit model with multiple fallback strategies.
        
        Args:
            y: Endogenous time series
            X: Exogenous variables (optional)
            
        Returns:
            bool: Success status
        """
        # Strategy 1: Full auto_arima with seasonal components
        strategies = [
            {
                'name': 'Full Auto-ARIMA',
                'params': {
                    'seasonal': True,
                    'm': self.seasonal_period,
                    'stepwise': True,
                    'max_p': 3, 'max_q': 3,
                    'max_P': 2, 'max_Q': 2,
                    'suppress_warnings': True,
                    'error_action': 'ignore'
                }
            },
            {
                'name': 'Simplified Auto-ARIMA',
                'params': {
                    'seasonal': True,
                    'm': self.seasonal_period,
                    'stepwise': True,
                    'max_p': 2, 'max_q': 2,
                    'max_P': 1, 'max_Q': 1,
                    'suppress_warnings': True,
                    'error_action': 'ignore'
                }
            },
            {
                'name': 'Non-seasonal ARIMA',
                'params': {
                    'seasonal': False,
                    'stepwise': True,
                    'max_p': 3, 'max_q': 3,
                    'suppress_warnings': True,
                    'error_action': 'ignore'
                }
            },
            {
                'name': 'Simple AR Model',
                'params': {
                    'seasonal': False,
                    'stepwise': False,
                    'start_p': 1, 'max_p': 2,
                    'start_q': 0, 'max_q': 0,
                    'suppress_warnings': True,
                    'error_action': 'ignore'
                }
            }
        ]
        
        for attempt, strategy in enumerate(strategies):
            try:
                logger.info(f"Attempting strategy {attempt + 1}: {strategy['name']}")
                
                # Attempt to fit model
                model = pm.auto_arima(y, X=X, **strategy['params'])
                
                # Validate model convergence
                if self._validate_model(model, y):
                    self.model = model
                    self.fitted = True
                    logger.info(f"Successfully fitted using {strategy['name']}")
                    return True
                else:
                    logger.warning(f"Model validation failed for {strategy['name']}")
                    
            except Exception as e:
                logger.error(f"Strategy {strategy['name']} failed: {str(e)}")
                continue
        
        # Final fallback: simple mean model
        logger.warning("All ARIMA strategies failed, using simple mean forecast")
        self.fallback_model = SimpleMeanModel(y)
        return False
    
    def _validate_model(self, model, y: pd.Series) -> bool:
        """
        Validate fitted model quality.
        
        Args:
            model: Fitted pmdarima model
            y: Training data
            
        Returns:
            bool: Model is valid
        """
        try:
            # Check if model parameters are reasonable
            if hasattr(model, 'aic') and np.isnan(model.aic()):
                logger.warning("Model AIC is NaN")
                return False
            
            # Check residuals
            residuals = model.resid()
            if len(residuals) == 0:
                logger.warning("No residuals available")
                return False
            
            # Check for excessive residual variance
            resid_std = np.std(residuals)
            data_std = np.std(y)
            if resid_std > data_std:
                logger.warning("Residual variance exceeds data variance")
                return False
            
            # Check for model stability (roots outside unit circle)
            if hasattr(model, 'arroots'):
                ar_roots = model.arroots()
                if len(ar_roots) > 0 and np.any(np.abs(ar_roots) <= 1.01):
                    logger.warning("Model may be non-stationary (AR roots near unit circle)")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation error: {str(e)}")
            return False
    
    def predict_robust(self, n_periods: int, X: Optional[pd.DataFrame] = None, 
                      alpha: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Generate robust forecasts with comprehensive error handling.
        
        Args:
            n_periods: Number of periods to forecast
            X: Future exogenous variables
            alpha: Significance level for confidence intervals
            
        Returns:
            Tuple of (forecasts, confidence_intervals, metadata)
        """
        metadata = {'method': 'unknown', 'warnings': []}
        
        try:
            if self.fitted and self.model is not None:
                # Use fitted ARIMA model
                forecasts, conf_int = self.model.predict(
                    n_periods=n_periods, 
                    X=X, 
                    return_conf_int=True, 
                    alpha=alpha
                )
                metadata['method'] = f"ARIMA{self.model.order}x{self.model.seasonal_order}"
                
                # Validate forecasts
                if np.isnan(forecasts).any() or np.isinf(forecasts).any():
                    raise ValueError("Generated forecasts contain invalid values")
                
                return forecasts, conf_int, metadata
                
            elif self.fallback_model is not None:
                # Use fallback model
                forecasts = self.fallback_model.predict(n_periods)
                metadata['method'] = 'Simple Mean'
                metadata['warnings'].append('Using fallback model due to ARIMA failure')
                return forecasts, None, metadata
                
            else:
                raise ValueError("No model available for forecasting")
                
        except Exception as e:
            logger.error(f"Forecasting error: {str(e)}")
            
            # Emergency fallback: naive forecast
            if hasattr(self, '_last_values'):
                last_value = self._last_values[-1]
                forecasts = np.full(n_periods, last_value)
                metadata['method'] = 'Naive (last value)'
                metadata['warnings'].append(f'Emergency fallback due to error: {str(e)}')
                return forecasts, None, metadata
            else:
                raise RuntimeError(f"Complete forecasting failure: {str(e)}")
    
    def fit_predict_safe(self, y: pd.Series, n_periods: int, 
                        X_train: Optional[pd.DataFrame] = None,
                        X_future: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Complete safe fit and predict workflow.
        
        Args:
            y: Time series data
            n_periods: Forecast horizon
            X_train: Training exogenous variables
            X_future: Future exogenous variables
            
        Returns:
            dict: Complete results with metadata
        """
        results = {
            'success': False,
            'forecasts': None,
            'confidence_intervals': None,
            'metadata': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Data validation
            is_valid, validation_msg = self.validate_data(y, X_train)
            if not is_valid:
                results['errors'].append(f"Data validation failed: {validation_msg}")
                return results
            
            # Data preprocessing
            processed_y, preprocess_info = self.preprocess_data(y)
            results['metadata']['preprocessing'] = preprocess_info
            
            # Store last values for emergency fallback
            self._last_values = processed_y.tail(min(10, len(processed_y))).values
            
            # Model fitting
            fit_success = self.fit_with_fallback(processed_y, X_train)
            
            # Generate forecasts
            forecasts, conf_int, forecast_metadata = self.predict_robust(
                n_periods, X_future
            )
            
            # Post-process forecasts if log transformation was applied
            if preprocess_info.get('log_transformed', False):
                forecasts = np.exp(forecasts)
                if conf_int is not None:
                    conf_int = np.exp(conf_int)
                results['warnings'].append('Forecasts back-transformed from log scale')
            
            results.update({
                'success': True,
                'forecasts': forecasts,
                'confidence_intervals': conf_int,
                'metadata': {**results['metadata'], **forecast_metadata},
                'model_fitted': fit_success
            })
            
        except Exception as e:
            results['errors'].append(f"Complete workflow failure: {str(e)}")
            logger.error(f"Fit-predict workflow failed: {str(e)}")
        
        return results

class SimpleMeanModel:
    """
    Simple fallback model using historical mean.
    """
    
    def __init__(self, y: pd.Series):
        self.mean_value = y.mean()
        self.std_value = y.std()
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Generate simple mean forecasts."""
        return np.full(n_periods, self.mean_value)

def demonstrate_error_scenarios():
    """
    Demonstrate various error scenarios and handling.
    """
    print("=== Error Handling Demonstrations ===\n")
    
    # Scenario 1: Insufficient data
    print("1. Testing insufficient data scenario...")
    short_data = pd.Series([1, 2, 3, 4, 5], name='short_series')
    forecaster = RobustARIMAForecaster()
    results = forecaster.fit_predict_safe(short_data, n_periods=5)
    print(f"   Result: {'Success' if results['success'] else 'Failed as expected'}")
    if results['errors']:
        print(f"   Error: {results['errors'][0]}")
    
    # Scenario 2: Data with missing values
    print("\n2. Testing missing values scenario...")
    data_with_nans = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 5)
    results = forecaster.fit_predict_safe(data_with_nans, n_periods=5)
    print(f"   Result: {'Success' if results['success'] else 'Failed as expected'}")
    if results['errors']:
        print(f"   Error: {results['errors'][0]}")
    
    # Scenario 3: Constant series
    print("\n3. Testing constant series scenario...")
    constant_data = pd.Series([5.0] * 50, name='constant_series')
    results = forecaster.fit_predict_safe(constant_data, n_periods=5)
    print(f"   Result: {'Success' if results['success'] else 'Failed as expected'}")
    if results['errors']:
        print(f"   Error: {results['errors'][0]}")
    
    # Scenario 4: Highly volatile data (should succeed with fallback)
    print("\n4. Testing highly volatile data...")
    np.random.seed(42)
    volatile_data = pd.Series(np.random.normal(0, 10, 100) + np.random.normal(0, 50, 100))
    results = forecaster.fit_predict_safe(volatile_data, n_periods=10)
    print(f"   Result: {'Success' if results['success'] else 'Failed'}")
    if results['success']:
        print(f"   Method used: {results['metadata'].get('method', 'Unknown')}")
        if results['warnings']:
            print(f"   Warnings: {len(results['warnings'])} issued")
    
    # Scenario 5: Normal data (should succeed)
    print("\n5. Testing normal seasonal data...")
    normal_data = pm.datasets.load_wineind()
    normal_series = pd.Series(normal_data, name='wine_sales')
    results = forecaster.fit_predict_safe(normal_series, n_periods=12)
    print(f"   Result: {'Success' if results['success'] else 'Failed'}")
    if results['success']:
        print(f"   Method: {results['metadata'].get('method', 'Unknown')}")
        print(f"   Forecast range: {results['forecasts'].min():.2f} to {results['forecasts'].max():.2f}")

def create_production_wrapper():
    """
    Create production-ready wrapper function.
    
    Returns:
        callable: Production forecasting function
    """
    def production_forecast(data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Production forecasting endpoint with comprehensive error handling.
        
        Args:
            data_path: Path to input data file
            config: Configuration parameters
            
        Returns:
            dict: Forecast results with status information
        """
        try:
            # Load data with error handling
            try:
                if data_path.endswith('.csv'):
                    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
                elif data_path.endswith('.json'):
                    data = pd.read_json(data_path)
                else:
                    raise ValueError(f"Unsupported file format: {data_path}")
                
                # Extract time series
                if isinstance(data, pd.DataFrame):
                    if len(data.columns) == 1:
                        y = data.iloc[:, 0]
                    else:
                        y = data['value']  # Assume 'value' column
                else:
                    y = data
                    
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Data loading failed: {str(e)}',
                    'forecasts': None
                }
            
            # Initialize forecaster with configuration
            forecaster = RobustARIMAForecaster(
                seasonal_period=config.get('seasonal_period', 12),
                max_retries=config.get('max_retries', 3)
            )
            
            # Generate forecasts
            results = forecaster.fit_predict_safe(
                y, 
                n_periods=config.get('forecast_horizon', 12)
            )
            
            # Format response
            if results['success']:
                return {
                    'status': 'success',
                    'forecasts': results['forecasts'].tolist(),
                    'confidence_intervals': results['confidence_intervals'].tolist() if results['confidence_intervals'] is not None else None,
                    'method': results['metadata'].get('method', 'Unknown'),
                    'warnings': results['warnings'],
                    'model_fitted': results['model_fitted']
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Forecasting failed',
                    'errors': results['errors'],
                    'forecasts': None
                }
                
        except Exception as e:
            return {
                'status': 'critical_error',
                'message': f'Unexpected error: {str(e)}',
                'forecasts': None
            }
    
    return production_forecast

def main():
    """
    Main function demonstrating comprehensive error handling.
    """
    print("=== Comprehensive Error Handling for Pmdarima ===\n")
    
    # Demonstrate error scenarios
    demonstrate_error_scenarios()
    
    # Show production wrapper
    print(f"\n=== Production Wrapper ===")
    prod_forecast = create_production_wrapper()
    
    # Test with sample configuration
    sample_config = {
        'seasonal_period': 12,
        'forecast_horizon': 6,
        'max_retries': 3
    }
    
    print("Production wrapper created successfully!")
    print("Key features:")
    print("- Comprehensive data validation")
    print("- Multiple fallback strategies")
    print("- Graceful error handling")
    print("- Structured response format")
    print("- Production logging")
    
    print(f"\n=== Error Handling Complete ===")

if __name__ == "__main__":
    main()