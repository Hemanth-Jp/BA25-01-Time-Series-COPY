"""
Time Series Analysis with Pandas Example 

This module demonstrates pandas time series capabilities including datetime
indexing, resampling, rolling windows, and temporal data transformations.

Version: 1.1 
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_timeseries_data():
    """
    Generate time series data with various frequencies and patterns.
    
    Returns:
        pd.DataFrame: Time series dataset with multiple metrics
    """
    # Create hourly data for 30 days
    date_range = pd.date_range(
        start='2023-01-01', 
        end='2023-01-31', 
        freq='H'
    )
    
    np.random.seed(42)
    n_points = len(date_range)
    
    # Generate synthetic time series with trends and seasonality
    trend = np.linspace(100, 150, n_points)
    daily_seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    weekly_seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
    noise = np.random.normal(0, 5, n_points)
    
    values = trend + daily_seasonal + weekly_seasonal + noise
    
    # Create DataFrame with time index
    df = pd.DataFrame({
        'value': values,
        'volume': np.random.poisson(50, n_points),
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365)) + np.random.normal(0, 2, n_points)
    }, index=date_range)  # Set index directly in constructor
    
    # Ensure index is properly named
    df.index.name = 'timestamp'
    
    return df


def datetime_operations(df):
    """
    Demonstrate datetime operations and time-based indexing.
    
    Args:
        df (pd.DataFrame): Time series dataframe with datetime index
    """
    print("=== DATETIME OPERATIONS ===")
    
    # Extract datetime components
    df_copy = df.copy()  # Work with copy to avoid modifying original
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['is_weekend'] = df_copy.index.dayofweek >= 5
    
    print("Datetime components added:")
    print(df_copy[['value', 'hour', 'day_of_week', 'is_weekend']].head())
    
    # Time-based selection - Fixed approach
    print("\nJanuary 15th data:")
    try:
        # Method 1: Using loc with date string (most reliable)
        jan_15 = df.loc['2023-01-15']
        print(f"Records for Jan 15: {len(jan_15)}")
        print(jan_15[['value', 'volume']].head())
    except KeyError:
        # Method 2: Alternative approach using boolean indexing
        jan_15_mask = df.index.date == pd.to_datetime('2023-01-15').date()
        jan_15 = df[jan_15_mask]
        print(f"Records for Jan 15: {len(jan_15)}")
        print(jan_15[['value', 'volume']].head())
    
    # Time range selection
    print("\nFirst week of January:")
    first_week = df.loc['2023-01-01':'2023-01-07']
    print(f"First week records: {len(first_week)}")
    print(f"Date range: {first_week.index[0]} to {first_week.index[-1]}")
    
    # Business day analysis
    print("\nWeekend vs Weekday analysis:")
    is_weekend = df.index.dayofweek >= 5
    
    # Simple approach - calculate each statistic individually
    weekday_data = df[~is_weekend]
    weekend_data = df[is_weekend]
    
    # Create summary table with individual calculations
    summary_data = {
        'Value_Mean': [weekday_data['value'].mean(), weekend_data['value'].mean()],
        'Value_Std': [weekday_data['value'].std(), weekend_data['value'].std()],
        'Volume_Mean': [weekday_data['volume'].mean(), weekend_data['volume'].mean()]
    }
    
    weekend_analysis = pd.DataFrame(summary_data, 
                                  index=['Weekday', 'Weekend']).round(2)
    print(weekend_analysis)


def resampling_operations(df):
    """
    Demonstrate resampling and frequency conversion operations.
    
    Args:
        df (pd.DataFrame): Time series dataframe
    """
    print("\n=== RESAMPLING OPERATIONS ===")
    
    # Daily resampling with different aggregations
    print("Daily aggregated statistics:")
    daily = df.resample('D').agg({
        'value': ['mean', 'min', 'max', 'std'],
        'volume': 'sum',
        'temperature': 'mean'
    }).round(2)
    
    # Flatten column names for better readability
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    print(daily.head())
    
    # Weekly resampling
    print("\nWeekly summary:")
    weekly = df.resample('W').agg({
        'value': 'mean',
        'volume': 'sum'
    }).round(2)
    print(weekly.head())
    
    # Custom resampling - every 6 hours
    print("\n6-hourly averages:")
    six_hourly = df.resample('6H')['value'].mean().round(2)
    print(six_hourly.head(10))


def rolling_window_analysis(df):
    """
    Demonstrate rolling window operations for time series analysis.
    
    Args:
        df (pd.DataFrame): Time series dataframe
    """
    print("\n=== ROLLING WINDOW ANALYSIS ===")
    
    # Create a copy to avoid modifying original
    df_rolling = df.copy()
    
    # Calculate rolling statistics
    df_rolling['value_rolling_24h'] = df_rolling['value'].rolling(window=24, min_periods=1).mean()
    df_rolling['value_rolling_std'] = df_rolling['value'].rolling(window=24, min_periods=1).std()
    
    # Calculate moving averages with time-based windows
    df_rolling['value_ma_7d'] = df_rolling['value'].rolling(window='7D', min_periods=1).mean()
    df_rolling['value_ma_3d'] = df_rolling['value'].rolling(window='3D', min_periods=1).mean()
    
    print("Rolling statistics (first 30 records):")
    rolling_cols = ['value', 'value_rolling_24h', 'value_rolling_std', 'value_ma_3d']
    print(df_rolling[rolling_cols].head(30).round(2))
    
    # Calculate percentage change
    df_rolling['value_pct_change'] = df_rolling['value'].pct_change()
    
    print("\nPercentage change statistics:")
    print(df_rolling['value_pct_change'].describe().round(4))
    
    # Exponential weighted moving average
    df_rolling['value_ewm'] = df_rolling['value'].ewm(span=24).mean()
    
    print("\nExponential weighted moving average (last 10 values):")
    print(df_rolling[['value', 'value_ewm']].tail(10).round(2))


def time_zone_operations():
    """
    Demonstrate time zone operations with pandas.
    """
    print("\n=== TIME ZONE OPERATIONS ===")
    
    # Create timestamp data
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    ts = pd.Series(range(5), index=dates)
    
    print("Original timestamps (naive):")
    print(ts)
    
    # Localize to UTC
    ts_utc = ts.tz_localize('UTC')
    print("\nLocalized to UTC:")
    print(ts_utc)
    
    # Convert to different time zones
    ts_est = ts_utc.tz_convert('US/Eastern')
    print("\nConverted to US/Eastern:")
    print(ts_est)
    
    ts_pst = ts_utc.tz_convert('US/Pacific')
    print("\nConverted to US/Pacific:")
    print(ts_pst)


def advanced_time_operations(df):
    """
    Demonstrate advanced time series operations.
    
    Args:
        df (pd.DataFrame): Time series dataframe
    """
    print("\n=== ADVANCED TIME OPERATIONS ===")
    
    # Shift operations for lag analysis
    df_advanced = df.copy()
    df_advanced['value_lag_1h'] = df_advanced['value'].shift(1)
    df_advanced['value_lead_1h'] = df_advanced['value'].shift(-1)
    
    print("Lag and lead operations:")
    print(df_advanced[['value', 'value_lag_1h', 'value_lead_1h']].head(10).round(2))
    
    # Calculate correlation between current and lagged values
    correlation = df_advanced['value'].corr(df_advanced['value_lag_1h'])
    print(f"\nCorrelation with 1-hour lag: {correlation:.4f}")
    
    # Seasonal decomposition simulation
    print("\nHourly patterns (average by hour):")
    hourly_pattern = df.groupby(df.index.hour)['value'].mean()
    print(hourly_pattern.round(2))
    
    # Find peak hours
    peak_hour = hourly_pattern.idxmax()
    low_hour = hourly_pattern.idxmin()
    print(f"\nPeak hour: {peak_hour}:00 (avg: {hourly_pattern[peak_hour]:.2f})")
    print(f"Low hour: {low_hour}:00 (avg: {hourly_pattern[low_hour]:.2f})")


if __name__ == "__main__":
    # Create and analyze time series data
    ts_df = create_timeseries_data()
    print(f"Created time series with {len(ts_df):,} hourly observations")
    print(f"Date range: {ts_df.index[0]} to {ts_df.index[-1]}")
    
    # Perform time series operations
    datetime_operations(ts_df)
    resampling_operations(ts_df)
    rolling_window_analysis(ts_df)
    time_zone_operations()
    advanced_time_operations(ts_df)
    
    print("\n" + "="*50)
    print("Time series analysis completed successfully!")
    print("="*50)