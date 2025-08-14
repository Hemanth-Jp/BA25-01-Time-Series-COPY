"""
Basic Pandas Data Manipulation Example

This module demonstrates fundamental pandas operations including data loading,
exploration, filtering, and basic transformations for data analysis workflows.


Version: 1.0
"""

import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_sample_data():
    """
    Create sample dataset for demonstration.
    
    Returns:
        pd.DataFrame: Sample sales data with multiple columns and data types
    """
    np.random.seed(42)
    
    # Generate sample sales data
    data = {
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], 100),
        'sales': np.random.normal(1000, 200, 100).round(2),
        'quantity': np.random.randint(1, 50, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'customer_satisfaction': np.random.uniform(1, 5, 100).round(1)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values for demonstration
    df.loc[np.random.choice(df.index, 5), 'customer_satisfaction'] = np.nan
    
    return df


def explore_data(df):
    """
    Perform basic data exploration and display key statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe to explore
    """
    print("=== DATA EXPLORATION ===")
    
    # Basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types and missing values
    print("\nData Types and Missing Values:")
    info_summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_summary)
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())


def basic_operations(df):
    """
    Demonstrate basic pandas operations including filtering, sorting, and selection.
    
    Args:
        df (pd.DataFrame): Input dataframe for operations
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("\n=== BASIC OPERATIONS ===")
    
    # Column selection
    print("Selected columns (product, sales, region):")
    selected = df[['product', 'sales', 'region']]
    print(selected.head())
    
    # Filtering data
    print("\nHigh sales transactions (>= 1200):")
    high_sales = df[df['sales'] >= 1200]
    print(f"Found {len(high_sales)} high-value transactions")
    print(high_sales[['date', 'product', 'sales']].head())
    
    # Sorting data
    print("\nTop 5 sales by value:")
    top_sales = df.nlargest(5, 'sales')[['date', 'product', 'sales', 'region']]
    print(top_sales)
    
    # Multiple conditions
    print("\nWidget A sales in North region:")
    filtered = df[(df['product'] == 'Widget A') & (df['region'] == 'North')]
    print(f"Found {len(filtered)} matching records")
    
    return df


def data_cleaning_demo(df):
    """
    Demonstrate data cleaning operations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n=== DATA CLEANING ===")
    
    # Handle missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Fill missing satisfaction scores with median
    median_satisfaction = df['customer_satisfaction'].median()
    df_clean = df.copy()
    df_clean['customer_satisfaction'].fillna(median_satisfaction, inplace=True)
    
    print(f"\nFilled missing satisfaction scores with median: {median_satisfaction}")
    print("Missing values after cleaning:")
    print(df_clean.isnull().sum())
    
    # Remove duplicates if any
    print(f"\nDuplicates found: {df_clean.duplicated().sum()}")
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def aggregation_demo(df):
    """
    Demonstrate aggregation and grouping operations.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n=== AGGREGATION OPERATIONS ===")
    
    # Group by product
    print("Sales summary by product:")
    product_summary = df.groupby('product').agg({
        'sales': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_satisfaction': 'mean'
    }).round(2)
    print(product_summary)
    
    # Group by region
    print("\nSales summary by region:")
    region_summary = df.groupby('region')['sales'].agg(['sum', 'mean']).round(2)
    print(region_summary)


if __name__ == "__main__":
    # Load and explore data
    df = load_sample_data()
    explore_data(df)
    
    # Perform basic operations
    processed_df = basic_operations(df)
    
    # Clean data
    clean_df = data_cleaning_demo(df)
    
    # Perform aggregations
    aggregation_demo(clean_df)
    
    print("\nBasic pandas operations completed successfully!")
    