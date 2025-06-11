import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import io
from walmartSalesTrainingCore import *

class TestWalmartSales:
    
    def setup_method(self):
        """Setup test data"""
        # Create mock CSV data that will create IsHoliday_x and IsHoliday_y during merge
        self.train_data = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2010-02-05', '2010-02-12', '2010-02-05', '2010-02-12'],
            'Weekly_Sales': [1643690.90, 1641957.44, 647054.47, 684064.00],
            'IsHoliday': [False, True, False, True]  # This will become IsHoliday_x
        })
        
        self.features_data = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Date': ['2010-02-05', '2010-02-12', '2010-02-05', '2010-02-12'],
            'Temperature': [42.31, 38.51, 45.07, 49.27],
            'IsHoliday': [False, False, False, False],  # This will become IsHoliday_y
            'MarkDown1': [np.nan, np.nan, np.nan, np.nan]
        })
        
        self.stores_data = pd.DataFrame({
            'Store': [1, 2],
            'Type': ['A', 'A'],
            'Size': [151315, 202307]
        })
    
    def test_config_values(self):
        """Test CONFIG dictionary contains expected values"""
        assert CONFIG['TRAIN_TEST_SPLIT'] == 0.7
        assert CONFIG['DEFAULT_SEASONAL_PERIODS'] == 20
        assert 'HOLIDAY_DATES' in CONFIG
    
    def test_load_and_merge_data_success(self):
        """Test successful data loading and merging - test the full pipeline including clean_data"""
        # Create temporary CSV files
        train_csv = io.StringIO(self.train_data.to_csv(index=False))
        features_csv = io.StringIO(self.features_data.to_csv(index=False))
        stores_csv = io.StringIO(self.stores_data.to_csv(index=False))
        
        # Test the merged result
        merged_result = load_and_merge_data(train_csv, features_csv, stores_csv)
        
        assert isinstance(merged_result, pd.DataFrame)
        assert len(merged_result) > 0
        assert 'Weekly_Sales' in merged_result.columns
        
        # Now test after cleaning (which consolidates IsHoliday columns)
        cleaned_result = clean_data(merged_result)
        assert 'IsHoliday' in cleaned_result.columns
    
    def test_load_and_merge_data_error_handling(self):
        """Test error handling for missing files"""
        with pytest.raises(ValueError, match="All three files"):
            load_and_merge_data(None, None, None)
    
    def test_clean_data_success(self):
        """Test data cleaning functionality"""
        df = pd.DataFrame({
            'Weekly_Sales': [1000.0, -500.0, 2000.0],
            'Date': ['2010-02-12', '2010-03-01', '2010-04-01'],
            'MarkDown1': [np.nan, 100.0, np.nan]
        })
        
        result = clean_data(df)
        
        assert len(result) == 2  # Negative sales removed
        assert result['MarkDown1'].isna().sum() == 0  # NaN filled with 0
        assert 'Super_Bowl' in result.columns
    
    def test_clean_data_error_handling(self):
        """Test error handling for empty dataframe"""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            clean_data(None)
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            clean_data(pd.DataFrame())
    
    def test_prepare_time_series_data(self):
        """Test time series data preparation with only available columns"""
        df = pd.DataFrame({
            'Date': ['2010-02-05', '2010-02-12', '2010-02-19'],
            'Weekly_Sales': [1000.0, 1100.0, 1200.0],
            'Temperature': [40.0, 45.0, 50.0],
            'Fuel_Price': [2.5, 2.6, 2.7],
            'CPI': [130.0, 131.0, 132.0],
            'Unemployment': [8.0, 8.1, 8.2]
        })
        
        df_week, df_week_diff = prepare_time_series_data(df)
        
        assert isinstance(df_week, pd.DataFrame)
        assert isinstance(df_week_diff, pd.Series)
        assert len(df_week_diff) == len(df_week) - 1
    
    def test_wmae_calculation(self):
        """Test WMAE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        wmae = wmae_ts(y_true, y_pred)
        
        assert isinstance(wmae, (float, np.float64))
        assert wmae >= 0
    
    def test_wmae_error_handling(self):
        """Test WMAE error handling"""
        with pytest.raises(ValueError, match="cannot be None"):
            wmae_ts(None, None)
    
    @patch('walmartSalesTrainingCore.auto_arima')
    def test_train_auto_arima(self, mock_arima):
        """Test Auto ARIMA training"""
        mock_model = Mock()
        mock_model.fit = Mock()
        mock_arima.return_value = mock_model
        
        train_data = pd.Series([1, 2, 3, 4, 5])
        
        result = train_auto_arima(train_data)
        
        mock_arima.assert_called_once()
        mock_model.fit.assert_called_once()
        assert result == mock_model
    
    def test_train_auto_arima_error_handling(self):
        """Test Auto ARIMA error handling"""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            train_auto_arima(None)
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            train_auto_arima([])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])