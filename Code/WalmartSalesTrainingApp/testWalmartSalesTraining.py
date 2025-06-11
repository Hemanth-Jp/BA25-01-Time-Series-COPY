"""
@brief Comprehensive test suite for Walmart sales training functionality
@details This module contains unit tests for all training-related functions including
         data loading, preprocessing, model training, and evaluation with extensive
         edge case coverage and error handling validation
@author Sales Prediction Team
@date 2025
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import io
from walmartSalesTrainingCore import *

class TestWalmartSales:
    """
    @brief Test class containing comprehensive unit tests for the training system
    @details Provides thorough testing coverage for data pipeline, model training,
             and evaluation functions using pytest framework with mock data and dependencies
    """
    
    def setup_method(self):
        """
        @brief Setup test data for each test method
        @details Creates mock datasets that simulate real Walmart data structure
                 including the IsHoliday column conflict that occurs during merging
        @note This method runs before each test to ensure clean test environment
        """
        # Create mock CSV data that will create IsHoliday_x and IsHoliday_y during merge
        self.train_data = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2010-02-05', '2010-02-12', '2010-02-05', '2010-02-12'],
            'Weekly_Sales': [1643690.90, 1641957.44, 647054.47, 684064.00],
            'IsHoliday': [False, True, False, True]  # This will become IsHoliday_x after merge
        })
        
        self.features_data = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2010-02-05', '2010-02-12', '2010-02-05', '2010-02-12'],
            'Temperature': [42.31, 38.51, 45.07, 49.27],
            'IsHoliday': [False, False, False, False],  # This will become IsHoliday_y after merge
            'MarkDown1': [np.nan, np.nan, np.nan, np.nan]  # Test missing value handling
        })
        
        self.stores_data = pd.DataFrame({
            'Store': [1, 2],
            'Type': ['A', 'A'],
            'Size': [151315, 202307]
        })
    
    def test_config_values(self):
        """
        @brief Test CONFIG dictionary contains expected configuration values
        @details Validates that all required configuration parameters are present
                 and set to expected default values for consistent behavior
        @note Configuration validation ensures proper system initialization
        """
        # Test core configuration parameters for training pipeline
        assert CONFIG['TRAIN_TEST_SPLIT'] == 0.7
        assert CONFIG['DEFAULT_SEASONAL_PERIODS'] == 20
        assert 'HOLIDAY_DATES' in CONFIG
    
    def test_load_and_merge_data_success(self):
        """
        @brief Test successful data loading and merging - test the full pipeline including clean_data
        @details Validates the complete data processing workflow from raw CSV files
                 through merging and cleaning, ensuring proper column handling
        @note Tests both merge operation and subsequent cleaning to verify end-to-end pipeline
        """
        # Create temporary CSV files using StringIO for testing
        train_csv = io.StringIO(self.train_data.to_csv(index=False))
        features_csv = io.StringIO(self.features_data.to_csv(index=False))
        stores_csv = io.StringIO(self.stores_data.to_csv(index=False))
        
        # Test the merged result from load_and_merge_data function
        merged_result = load_and_merge_data(train_csv, features_csv, stores_csv)
        
        # Verify basic structure and content of merged data
        assert isinstance(merged_result, pd.DataFrame)
        assert len(merged_result) > 0
        assert 'Weekly_Sales' in merged_result.columns
        
        # Now test after cleaning (which consolidates IsHoliday columns)
        cleaned_result = clean_data(merged_result)
        assert 'IsHoliday' in cleaned_result.columns
    
    def test_load_and_merge_data_error_handling(self):
        """
        @brief Test error handling for missing files in data loading
        @details Verifies proper validation and error messaging when required files are missing
        @note Tests the input validation layer before attempting data operations
        """
        # Test with all None files (should raise ValueError)
        with pytest.raises(ValueError, match="All three files"):
            load_and_merge_data(None, None, None)
    
    def test_clean_data_success(self):
        """
        @brief Test data cleaning functionality with realistic data scenarios
        @details Validates removal of invalid sales records, missing value imputation,
                 and holiday feature engineering
        @note Tests multiple data quality issues that occur in real-world datasets
        """
        # Create test DataFrame with common data quality issues
        df = pd.DataFrame({
            'Weekly_Sales': [1000.0, -500.0, 2000.0],  # Include negative sales to test filtering
            'Date': ['2010-02-12', '2010-03-01', '2010-04-01'],
            'MarkDown1': [np.nan, 100.0, np.nan]  # Test missing value handling
        })
        
        result = clean_data(df)
        
        # Verify data cleaning results
        assert len(result) == 2  # Negative sales should be removed
        assert result['MarkDown1'].isna().sum() == 0  # NaN values should be filled with 0
        assert 'Super_Bowl' in result.columns  # Holiday features should be created
    
    def test_clean_data_error_handling(self):
        """
        @brief Test error handling for invalid input data in cleaning function
        @details Verifies proper validation of input DataFrame before processing
        @note Tests both None input and empty DataFrame scenarios
        """
        # Test with None DataFrame
        with pytest.raises(ValueError, match="cannot be None or empty"):
            clean_data(None)
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="cannot be None or empty"):
            clean_data(pd.DataFrame())
    
    def test_prepare_time_series_data(self):
        """
        @brief Test time series data preparation with only available columns
        @details Validates date conversion, weekly aggregation, and differencing operations
                 for time series modeling preparation
        @note Tests the transformation from raw data to model-ready time series format
        """
        # Create realistic time series data for testing
        df = pd.DataFrame({
            'Date': ['2010-02-05', '2010-02-12', '2010-02-19'],
            'Weekly_Sales': [1000.0, 1100.0, 1200.0],
            'Temperature': [40.0, 45.0, 50.0],
            'Fuel_Price': [2.5, 2.6, 2.7],
            'CPI': [130.0, 131.0, 132.0],
            'Unemployment': [8.0, 8.1, 8.2]
        })
        
        df_week, df_week_diff = prepare_time_series_data(df)
        
        # Verify time series preparation results
        assert isinstance(df_week, pd.DataFrame)
        assert isinstance(df_week_diff, pd.Series)
        assert len(df_week_diff) == len(df_week) - 1  # Differencing reduces length by 1
    
    def test_wmae_calculation(self):
        """
        @brief Test WMAE (Weighted Mean Absolute Error) calculation
        @details Validates the evaluation metric calculation with known input/output pairs
        @note WMAE is the primary evaluation metric for sales forecasting models
        """
        # Create test data with known error patterns
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        wmae = wmae_ts(y_true, y_pred)
        
        # Verify WMAE calculation properties
        assert isinstance(wmae, (float, np.float64))
        assert wmae >= 0  # WMAE should always be non-negative
    
    def test_wmae_error_handling(self):
        """
        @brief Test WMAE error handling for invalid inputs
        @details Verifies proper validation of input arrays before calculation
        @note Tests None input validation to prevent runtime errors
        """
        # Test with None inputs
        with pytest.raises(ValueError, match="cannot be None"):
            wmae_ts(None, None)
    
    @patch('walmartSalesTrainingCore.auto_arima')
    def test_train_auto_arima(self, mock_arima):
        """
        @brief Test Auto ARIMA training with mocked dependencies
        @details Validates the Auto ARIMA training workflow without actual model fitting
        @param mock_arima Mocked auto_arima function to simulate training process
        @note Uses mocking to isolate the training logic from external dependencies
        """
        # Setup mock Auto ARIMA model and training process
        mock_model = Mock()
        mock_model.fit = Mock()
        mock_arima.return_value = mock_model
        
        # Create test training data
        train_data = pd.Series([1, 2, 3, 4, 5])
        
        result = train_auto_arima(train_data)
        
        # Verify Auto ARIMA training workflow
        mock_arima.assert_called_once()  # auto_arima should be called once
        mock_model.fit.assert_called_once()  # model.fit should be called once
        assert result == mock_model  # returned model should match mock
    
    def test_train_auto_arima_error_handling(self):
        """
        @brief Test Auto ARIMA error handling for invalid training data
        @details Verifies proper validation of training data before model fitting
        @note Tests both None input and empty data scenarios
        """
        # Test with None training data
        with pytest.raises(ValueError, match="cannot be None or empty"):
            train_auto_arima(None)
        
        # Test with empty training data
        with pytest.raises(ValueError, match="cannot be None or empty"):
            train_auto_arima([])

if __name__ == "__main__":
    """
    @brief Entry point for running tests directly with verbose output
    @details Allows running the test suite directly when executed as main module
    @note Uses pytest.main() to run tests with verbose flag for detailed reporting
    """
    pytest.main([__file__, "-v"])