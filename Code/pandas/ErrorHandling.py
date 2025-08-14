"""
Comprehensive Error Handling with Pandas

This module demonstrates robust error handling patterns, data validation,
and exception management for pandas operations in production environments.


Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, List


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def safe_read_csv(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file with comprehensive error handling.
    
    Args:
        filepath (str): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pd.DataFrame or None: Loaded dataframe or None if error occurred
    """
    try:
        logger.info(f"Attempting to read CSV file: {filepath}")
        
        # Check file existence and read data
        df = pd.read_csv(filepath, **kwargs)
        
        if df.empty:
            logger.warning(f"CSV file {filepath} is empty")
            return None
            
        logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in file: {filepath}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Parser error reading {filepath}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {str(e)}")
        return None


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                      numeric_columns: List[str] = None) -> bool:
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): Required column names
        numeric_columns (List[str], optional): Columns that should be numeric
        
    Returns:
        bool: True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if df.empty:
            raise DataValidationError("Dataframe is empty")
        
        # Validate numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        raise DataValidationError(f"Column '{col}' is not numeric")
        
        logger.info("Dataframe validation passed")
        return True
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Validation error: {str(e)}")


def safe_data_operations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data operations with error handling and recovery.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Handle missing values safely
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if result_df[col].isnull().any():
                median_value = result_df[col].median()
                result_df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Safe mathematical operations
        if 'sales' in result_df.columns and 'quantity' in result_df.columns:
            try:
                # Avoid division by zero
                mask = result_df['quantity'] != 0
                result_df.loc[mask, 'unit_price'] = (
                    result_df.loc[mask, 'sales'] / result_df.loc[mask, 'quantity']
                )
                result_df.loc[~mask, 'unit_price'] = 0
                logger.info("Calculated unit prices safely")
            except Exception as e:
                logger.warning(f"Could not calculate unit prices: {str(e)}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in data operations: {str(e)}")
        return df  # Return original dataframe if processing fails


def safe_aggregation(df: pd.DataFrame, group_cols: List[str], 
                    agg_dict: dict) -> Optional[pd.DataFrame]:
    """
    Perform safe aggregation operations with error handling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_cols (List[str]): Columns to group by
        agg_dict (dict): Aggregation dictionary
        
    Returns:
        pd.DataFrame or None: Aggregated results or None if error
    """
    try:
        # Validate group columns exist
        missing_cols = set(group_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Group columns not found: {missing_cols}")
            return None
        
        # Check if aggregation columns exist
        for col in agg_dict.keys():
            if col not in df.columns:
                logger.warning(f"Aggregation column '{col}' not found, skipping")
                agg_dict.pop(col, None)
        
        if not agg_dict:
            logger.error("No valid aggregation columns found")
            return None
        
        # Perform aggregation
        result = df.groupby(group_cols).agg(agg_dict)
        logger.info(f"Aggregation completed: {len(result)} groups")
        return result
        
    except Exception as e:
        logger.error(f"Error in aggregation: {str(e)}")
        return None


def demonstrate_error_handling():
    """
    Demonstrate various error handling scenarios.
    """
    print("=== ERROR HANDLING DEMONSTRATION ===")
    
    # Create sample data with potential issues
    sample_data = {
        'sales': [100, 200, np.nan, 400, 500],
        'quantity': [1, 2, 3, 0, 5],  # Note: zero quantity for testing
        'product': ['A', 'B', 'C', 'D', 'E']
    }
    df = pd.DataFrame(sample_data)
    
    try:
        # Validate dataframe
        validate_dataframe(df, ['sales', 'quantity', 'product'], ['sales', 'quantity'])
        print("✓ Dataframe validation passed")
        
        # Process data safely
        processed_df = safe_data_operations(df)
        print("✓ Data operations completed")
        print(processed_df)
        
        # Safe aggregation
        agg_result = safe_aggregation(processed_df, ['product'], {'sales': 'mean'})
        if agg_result is not None:
            print("✓ Aggregation completed")
            print(agg_result)
        
    except DataValidationError as e:
        logger.error(f"Validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Demonstrate error handling patterns
    logger.info("Starting error handling demonstration")
    demonstrate_error_handling()
    logger.info("Error handling demonstration completed")