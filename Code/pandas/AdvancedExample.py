"""
Advanced Pandas Data Analysis Example

This module demonstrates sophisticated pandas operations including grouping,
aggregation, pivoting, and complex data transformations for in-depth analysis.


Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_advanced_dataset():
    """
    Create a comprehensive dataset for advanced analysis.
    
    Returns:
        pd.DataFrame: Complex sales dataset with multiple dimensions
    """
    np.random.seed(42)
    
    # Generate comprehensive sales data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    regions = ['North', 'South', 'East', 'West']
    channels = ['Online', 'Retail', 'Wholesale']
    
    data = []
    for date in dates:
        for product in products:
            for region in regions:
                # Seasonal effects
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                sales = np.random.normal(1000 * seasonal_factor, 150)
                quantity = max(1, int(np.random.normal(25, 5)))
                channel = np.random.choice(channels, p=[0.4, 0.35, 0.25])
                
                data.append({
                    'date': date,
                    'product': product,
                    'region': region,
                    'channel': channel,
                    'sales': round(sales, 2),
                    'quantity': quantity,
                    'unit_price': round(sales / quantity, 2),
                    'cost': round(sales * np.random.uniform(0.6, 0.8), 2)
                })
    
    df = pd.DataFrame(data)
    df['profit'] = df['sales'] - df['cost']
    df['profit_margin'] = (df['profit'] / df['sales'] * 100).round(2)
    
    return df


def advanced_grouping_analysis(df):
    """
    Perform advanced grouping and aggregation operations.
    
    Args:
        df (pd.DataFrame): Input sales dataframe
    """
    print("=== ADVANCED GROUPING ANALYSIS ===")
    
    # Multi-level grouping with multiple aggregations
    print("Multi-dimensional analysis by Product and Region:")
    grouped = df.groupby(['product', 'region']).agg({
        'sales': ['sum', 'mean', 'std'],
        'quantity': 'sum',
        'profit': 'sum',
        'profit_margin': 'mean'
    }).round(2)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    print(grouped.head(10))
    
    # Custom aggregation functions
    print("\nCustom aggregation analysis:")
    def sales_range(x):
        return x.max() - x.min()
    
    custom_agg = df.groupby('product').agg({
        'sales': ['count', sales_range, lambda x: x.quantile(0.95)],
        'profit_margin': ['min', 'max', 'median']
    }).round(2)
    
    custom_agg.columns = ['transaction_count', 'sales_range', 'sales_95th_percentile',
                         'min_margin', 'max_margin', 'median_margin']
    print(custom_agg)


def pivot_and_reshape_analysis(df):
    """
    Demonstrate pivoting and data reshaping operations.
    
    Args:
        df (pd.DataFrame): Input sales dataframe
    """
    print("\n=== PIVOT AND RESHAPE ANALYSIS ===")
    
    # Create monthly aggregated data
    df['month'] = df['date'].dt.to_period('M')
    monthly_data = df.groupby(['month', 'product', 'region'])['sales'].sum().reset_index()
    
    # Pivot table for product performance across regions
    print("Product performance pivot table:")
    pivot_product = pd.pivot_table(
        monthly_data,
        values='sales',
        index='month',
        columns=['product', 'region'],
        aggfunc='sum',
        fill_value=0
    )
    print(pivot_product.head())
    
    # Cross-tabulation analysis
    print("\nChannel distribution by region (cross-tabulation):")
    crosstab = pd.crosstab(
        df['region'], 
        df['channel'], 
        values=df['sales'], 
        aggfunc='sum',
        margins=True
    ).round(0)
    print(crosstab)


def merging_and_joining_demo():
    """
    Demonstrate merging and joining operations.
    """
    print("\n=== MERGING AND JOINING DEMO ===")
    
    # Create sample datasets for merging
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'customer_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'region': ['North', 'South', 'East', 'West', 'North']
    })
    
    orders = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105, 106],
        'customer_id': [1, 2, 1, 3, 4, 6],
        'order_value': [500, 750, 300, 1200, 450, 800],
        'order_date': pd.date_range('2023-01-01', periods=6, freq='D')
    })
    
    print("Customers DataFrame:")
    print(customers)
    print("\nOrders DataFrame:")
    print(orders)
    
    # Inner join
    print("\nInner Join (customers with orders):")
    inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
    print(inner_join)
    
    # Left join
    print("\nLeft Join (all customers, with/without orders):")
    left_join = pd.merge(customers, orders, on='customer_id', how='left')
    print(left_join)


if __name__ == "__main__":
    # Create and analyze advanced dataset
    df = create_advanced_dataset()
    print(f"Created dataset with {len(df):,} records")
    
    # Perform advanced analyses
    advanced_grouping_analysis(df)
    pivot_and_reshape_analysis(df)
    merging_and_joining_demo()
    
    print("\nAdvanced pandas analysis completed successfully!")