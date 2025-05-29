import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_data():
    """Generate sample data for testing the training app"""
    
    # Generate dates
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2012, 12, 31)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=7)  # Weekly data
    
    # Generate store data
    stores = list(range(1, 6))  # 5 stores for testing
    store_data = []
    for store_id in stores:
        store_data.append({
            'Store': store_id,
            'Type': random.choice(['A', 'B', 'C']),
            'Size': random.randint(50000, 200000)
        })
    
    # Generate training data
    train_data = []
    for store_id in stores:
        for date in dates:
            # Generate realistic sales data with seasonality
            base_sales = 15000 + random.randint(-2000, 2000)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            weekly_sales = base_sales * seasonal_factor
            
            train_data.append({
                'Store': store_id,
                'Date': date.strftime('%Y-%m-%d'),
                'Weekly_Sales': weekly_sales,
                'IsHoliday': date.weekday() in [4, 5]  # Weekend as holiday
            })
    
    # Generate features data
    features_data = []
    for date in dates:
        features_data.append({
            'Store': random.choice(stores),
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature': random.uniform(20, 100),
            'Fuel_Price': random.uniform(2.5, 4.5),
            'MarkDown1': random.uniform(1000, 5000) if random.random() > 0.7 else np.nan,
            'MarkDown2': random.uniform(500, 3000) if random.random() > 0.8 else np.nan,
            'MarkDown3': random.uniform(200, 2000) if random.random() > 0.9 else np.nan,
            'MarkDown4': random.uniform(100, 1000) if random.random() > 0.9 else np.nan,
            'MarkDown5': random.uniform(50, 500) if random.random() > 0.95 else np.nan,
            'CPI': random.uniform(130, 150),
            'Unemployment': random.uniform(6, 10),
            'IsHoliday': date.weekday() in [4, 5]
        })
    
    # Create DataFrames
    df_train = pd.DataFrame(train_data)
    df_features = pd.DataFrame(features_data)
    df_stores = pd.DataFrame(store_data)
    
    # Save to CSV files
    df_train.to_csv('train.csv', index=False)
    df_features.to_csv('features.csv', index=False)
    df_stores.to_csv('stores.csv', index=False)
    
    print("Test data generated successfully!")
    print(f"Generated files: train.csv, features.csv, stores.csv")
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Number of stores: {len(stores)}")
    print(f"Number of weeks: {len(dates)}")

if __name__ == "__main__":
    generate_test_data()