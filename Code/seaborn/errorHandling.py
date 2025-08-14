"""
Comprehensive Error Handling with Seaborn

This script demonstrates robust error handling patterns for seaborn
visualizations, including data validation and graceful failure recovery.


"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union, Any

class SeabornErrorHandler:
    """
    Utility class for handling common seaborn visualization errors.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Validate DataFrame for visualization requirements.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if valid, False otherwise
        """
        if df is None or df.empty:
            print("Error: DataFrame is None or empty")
            return False
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                return False
        
        return True
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'warn') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'warn', 'drop', 'fill_mean', 'fill_median'
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        if df.isnull().sum().sum() == 0:
            return df
        
        if strategy == 'warn':
            missing_info = df.isnull().sum()
            print(f"Warning: Missing values detected:\n{missing_info[missing_info > 0]}")
            return df
        elif strategy == 'drop':
            return df.dropna()
        elif strategy == 'fill_mean':
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == 'fill_median':
            return df.fillna(df.median(numeric_only=True))
        else:
            print(f"Unknown strategy: {strategy}")
            return df

def safe_load_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Safely load seaborn dataset with error handling.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        pd.DataFrame or None: Loaded dataset or None if failed
    """
    try:
        data = sns.load_dataset(dataset_name)
        print(f"Successfully loaded '{dataset_name}' dataset")
        return data
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return None

def safe_plot_creation(plot_func, data: pd.DataFrame, **kwargs) -> bool:
    """
    Safely create seaborn plots with comprehensive error handling.
    
    Args:
        plot_func: Seaborn plotting function
        data: DataFrame for plotting
        **kwargs: Additional arguments for the plot function
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate data
        if not SeabornErrorHandler.validate_dataframe(data):
            return False
        
        # Check for required columns in kwargs
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        hue_col = kwargs.get('hue')
        
        required_cols = [col for col in [x_col, y_col, hue_col] if col is not None]
        
        if not SeabornErrorHandler.validate_dataframe(data, required_cols):
            return False
        
        # Handle missing values
        clean_data = SeabornErrorHandler.handle_missing_values(data, 'warn')
        
        # Create plot with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plot_func(data=clean_data, **kwargs)
        
        return True
        
    except ValueError as e:
        print(f"ValueError in plotting: {e}")
        print("This often occurs due to incompatible data types or column names")
        return False
    except KeyError as e:
        print(f"KeyError in plotting: {e}")
        print("Check that specified column names exist in the dataset")
        return False
    except Exception as e:
        print(f"Unexpected error in plotting: {e}")
        return False

def demonstrate_data_validation():
    """
    Demonstrate data validation and error handling.
    """
    print("=== Data Validation Demo ===")
    
    # Test with valid data
    tips = safe_load_dataset("tips")
    if tips is not None:
        print("Valid data test:")
        success = safe_plot_creation(sns.scatterplot, tips, 
                                   x="total_bill", y="tip")
        print(f"Plot creation: {'Success' if success else 'Failed'}")
    
    # Test with invalid column names
    print("\nInvalid column test:")
    if tips is not None:
        success = safe_plot_creation(sns.scatterplot, tips, 
                                   x="invalid_column", y="tip")
        print(f"Plot creation: {'Success' if success else 'Failed'}")
    
    # Test with missing values
    print("\nMissing values test:")
    if tips is not None:
        # Introduce missing values
        tips_with_na = tips.copy()
        tips_with_na.loc[0:10, 'tip'] = np.nan
        
        success = safe_plot_creation(sns.scatterplot, tips_with_na, 
                                   x="total_bill", y="tip")
        print(f"Plot creation: {'Success' if success else 'Failed'}")

def robust_visualization_pipeline():
    """
    Demonstrate a robust visualization pipeline with error handling.
    """
    print("\n=== Robust Visualization Pipeline ===")
    
    # Load data safely
    data = safe_load_dataset("tips")
    if data is None:
        print("Cannot proceed without data")
        return
    
    # Create figure with error handling
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Robust Seaborn Visualizations', fontsize=16)
        
        plots_config = [
            (sns.scatterplot, {"x": "total_bill", "y": "tip", "hue": "time"}, axes[0,0]),
            (sns.boxplot, {"x": "day", "y": "total_bill"}, axes[0,1]),
            (sns.histplot, {"x": "total_bill", "kde": True}, axes[1,0]),
            (sns.barplot, {"x": "day", "y": "tip"}, axes[1,1])
        ]
        
        for plot_func, kwargs, ax in plots_config:
            kwargs['ax'] = ax
            success = safe_plot_creation(plot_func, data, **kwargs)
            if not success:
                ax.text(0.5, 0.5, 'Plot Failed', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{plot_func.__name__} - Failed')
            else:
                ax.set_title(f'{plot_func.__name__} - Success')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization pipeline: {e}")

def handle_large_dataset_errors():
    """
    Demonstrate handling of large dataset visualization errors.
    """
    print("\n=== Large Dataset Handling ===")
    
    try:
        # Create synthetic large dataset
        large_data = pd.DataFrame({
            'x': np.random.randn(100000),
            'y': np.random.randn(100000),
            'category': np.random.choice(['A', 'B', 'C'], 100000)
        })
        
        print(f"Created dataset with {len(large_data)} rows")
        
        # Sample for plotting to avoid performance issues
        sample_size = min(1000, len(large_data))
        sampled_data = large_data.sample(n=sample_size, random_state=42)
        
        print(f"Sampling {sample_size} rows for visualization")
        
        # Create plot with sampled data
        plt.figure(figsize=(8, 6))
        success = safe_plot_creation(sns.scatterplot, sampled_data, 
                                   x="x", y="y", hue="category", alpha=0.6)
        
        if success:
            plt.title(f'Large Dataset Visualization (n={sample_size})')
            plt.show()
        
    except MemoryError:
        print("Memory error: Dataset too large for visualization")
    except Exception as e:
        print(f"Error handling large dataset: {e}")

def main():
    """
    Execute comprehensive error handling demonstrations.
    """
    print("=== Comprehensive Seaborn Error Handling ===\n")
    
    # Demonstrate data validation
    demonstrate_data_validation()
    
    # Show robust pipeline
    robust_visualization_pipeline()
    
    # Handle large datasets
    handle_large_dataset_errors()
    
    print("\nError handling demonstration completed!")
    print("Key takeaways:")
    print("- Always validate data before plotting")
    print("- Handle missing values appropriately")
    print("- Use try-except blocks for robust error handling")
    print("- Sample large datasets for better performance")

if __name__ == "__main__":
    main()