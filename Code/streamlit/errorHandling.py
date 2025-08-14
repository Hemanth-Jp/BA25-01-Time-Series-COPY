"""
@file ErrorHandling.py
@brief Comprehensive Error Handling Patterns for Streamlit Applications


This module demonstrates robust error handling patterns, input validation,
and user feedback mechanisms for production-ready Streamlit applications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import logging
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta
import io
import sqlite3
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validation utility class for Streamlit applications.
    """
    
    @staticmethod
    def validate_file_upload(file, allowed_extensions: list, max_size_mb: float = 10.0) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        @param file: Streamlit uploaded file object
        @param allowed_extensions: List of allowed file extensions
        @param max_size_mb: Maximum file size in MB
        @return: Tuple of (is_valid, error_message)
        """
        if file is None:
            return False, "No file uploaded"
        
        # Check file extension
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large. Maximum size: {max_size_mb}MB"
        
        return True, "File is valid"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None, min_rows: int = 1) -> Tuple[bool, str]:
        """
        Validate DataFrame structure and content.
        
        @param df: DataFrame to validate
        @param required_columns: List of required column names
        @param min_rows: Minimum number of rows required
        @return: Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        if len(df) < min_rows:
            return False, f"Insufficient data. Minimum {min_rows} rows required"
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        return True, "DataFrame is valid"
    
    @staticmethod
    def validate_numeric_input(value: Union[int, float], min_val: float = None, max_val: float = None) -> Tuple[bool, str]:
        """
        Validate numeric input values.
        
        @param value: Numeric value to validate
        @param min_val: Minimum allowed value
        @param max_val: Maximum allowed value
        @return: Tuple of (is_valid, error_message)
        """
        if value is None:
            return False, "Value cannot be empty"
        
        try:
            num_value = float(value)
            
            if min_val is not None and num_value < min_val:
                return False, f"Value must be at least {min_val}"
            
            if max_val is not None and num_value > max_val:
                return False, f"Value must be at most {max_val}"
            
            return True, "Value is valid"
        
        except (ValueError, TypeError):
            return False, "Value must be a valid number"


def safe_file_reader(file, file_type: str = 'csv') -> Optional[pd.DataFrame]:
    """
    Safely read uploaded files with comprehensive error handling.
    
    @param file: Streamlit uploaded file object
    @param file_type: Type of file to read ('csv', 'excel', 'json')
    @return: DataFrame or None if error occurred
    """
    try:
        if file_type.lower() == 'csv':
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    st.success(f"File loaded successfully with {encoding} encoding")
                    logger.info(f"CSV file loaded with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail
            st.error("Unable to read file with any supported encoding")
            return None
        
        elif file_type.lower() == 'excel':
            df = pd.read_excel(file)
            st.success("Excel file loaded successfully")
            logger.info("Excel file loaded successfully")
            return df
        
        elif file_type.lower() == 'json':
            df = pd.read_json(file)
            st.success("JSON file loaded successfully")
            logger.info("JSON file loaded successfully")
            return df
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty")
        logger.error("Attempted to load empty file")
        return None
    
    except pd.errors.ParserError as e:
        st.error(f"Error parsing file: {str(e)}")
        logger.error(f"Parser error: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"Unexpected error loading file: {str(e)}")
        logger.error(f"Unexpected file loading error: {str(e)}")
        return None


def safe_api_request(url: str, timeout: int = 10, retries: int = 3) -> Optional[dict]:
    """
    Make API requests with error handling and retries.
    
    @param url: API endpoint URL
    @param timeout: Request timeout in seconds
    @param retries: Number of retry attempts
    @return: JSON response or None if failed
    """
    for attempt in range(retries):
        try:
            with st.spinner(f"Making API request (attempt {attempt + 1}/{retries})..."):
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                
                return response.json()
        
        except requests.exceptions.Timeout:
            if attempt == retries - 1:
                st.error(f"Request timed out after {timeout} seconds")
                logger.error(f"API request timeout: {url}")
            else:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        except requests.exceptions.ConnectionError:
            if attempt == retries - 1:
                st.error("Unable to connect to the API")
                logger.error(f"Connection error: {url}")
            else:
                time.sleep(2 ** attempt)
        
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code}")
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            break
        
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {str(e)}")
            logger.error(f"Request exception: {str(e)}")
            break
        
        except ValueError as e:
            st.error(f"Invalid JSON response: {str(e)}")
            logger.error(f"JSON parsing error: {str(e)}")
            break
    
    return None


def safe_data_processing(df: pd.DataFrame, operation: str) -> Optional[pd.DataFrame]:
    """
    Safely perform data processing operations with error handling.
    
    @param df: Input DataFrame
    @param operation: Type of operation to perform
    @return: Processed DataFrame or None if error occurred
    """
    try:
        if operation == "clean_missing":
            # Handle missing values
            initial_rows = len(df)
            df_cleaned = df.dropna()
            removed_rows = initial_rows - len(df_cleaned)
            
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} rows with missing values")
            
            return df_cleaned
        
        elif operation == "normalize_numeric":
            # Normalize numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                st.warning("No numeric columns found for normalization")
                return df
            
            df_normalized = df.copy()
            for col in numeric_columns:
                df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
            
            st.success(f"Normalized {len(numeric_columns)} numeric columns")
            return df_normalized
        
        elif operation == "detect_outliers":
            # Detect outliers using IQR method
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            outlier_indices = set()
            
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outlier_indices.update(col_outliers)
            
            df_clean = df.drop(outlier_indices)
            removed_outliers = len(outlier_indices)
            
            if removed_outliers > 0:
                st.warning(f"Removed {removed_outliers} outlier rows")
            
            return df_clean
        
        else:
            st.error(f"Unknown operation: {operation}")
            return None
    
    except Exception as e:
        st.error(f"Error during {operation}: {str(e)}")
        logger.error(f"Data processing error in {operation}: {str(e)}")
        return None


def create_error_boundary(func):
    """
    Decorator to create error boundaries for Streamlit functions.
    
    @param func: Function to wrap with error handling
    @return: Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred in {func.__name__}: {str(e)}")
            logger.error(f"Error in {func.__name__}: {str(e)}")
            
            # Show error details in expander for debugging
            with st.expander("Error Details (for debugging)"):
                st.code(f"Function: {func.__name__}\nError: {str(e)}\nType: {type(e).__name__}")
            
            return None
    
    return wrapper


@create_error_boundary
def demonstration_function_with_errors():
    """
    Demonstration function that may raise various errors.
    """
    error_type = st.selectbox(
        "Select error type to demonstrate",
        ["No Error", "Division by Zero", "Key Error", "Type Error", "Value Error"]
    )
    
    if error_type == "Division by Zero":
        result = 10 / 0
    elif error_type == "Key Error":
        data = {"a": 1, "b": 2}
        result = data["nonexistent_key"]
    elif error_type == "Type Error":
        result = "string" + 123
    elif error_type == "Value Error":
        result = int("not_a_number")
    else:
        result = "No error occurred"
        st.success("Function executed successfully!")
    
    return result


def main():
    """
    Main function demonstrating comprehensive error handling patterns.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Error Handling Demo",
        page_icon=":shield:",
        layout="wide"
    )
    
    # Application header
    st.title("Streamlit Error Handling Demonstration")
    st.markdown("---")
    st.write("This application demonstrates robust error handling patterns for production Streamlit apps.")
    
    # Create tabs for different error handling scenarios
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "File Upload", "API Requests", "Data Processing", "Error Boundaries", "Validation Demo"
    ])
    
    with tab1:
        st.header("File Upload Error Handling")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files"
        )
        
        if uploaded_file is not None:
            # Validate file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            is_valid, message = DataValidator.validate_file_upload(
                uploaded_file, 
                ['csv', 'xlsx', 'json'], 
                max_size_mb=50.0
            )
            
            if is_valid:
                st.success(message)
                
                # Safely read the file
                df = safe_file_reader(uploaded_file, file_extension)
                
                if df is not None:
                    # Validate DataFrame
                    is_valid_df, df_message = DataValidator.validate_dataframe(df, min_rows=1)
                    
                    if is_valid_df:
                        st.success(df_message)
                        st.subheader("Data Preview")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Display basic statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(df))
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    else:
                        st.error(df_message)
            
            else:
                st.error(message)
    
    with tab2:
        st.header("API Request Error Handling")
        
        st.write("Test API requests with comprehensive error handling:")
        
        # Predefined API endpoints for testing
        api_options = {
            "JSONPlaceholder (Working)": "https://jsonplaceholder.typicode.com/posts/1",
            "Invalid URL": "https://nonexistent-api-endpoint-12345.com/data",
            "Timeout Simulation": "https://httpbin.org/delay/15"  # This will timeout
        }
        
        selected_api = st.selectbox("Select API endpoint", list(api_options.keys()))
        
        if st.button("Make API Request"):
            url = api_options[selected_api]
            result = safe_api_request(url, timeout=5, retries=2)
            
            if result:
                st.success("API request successful!")
                st.json(result)
            else:
                st.error("API request failed. Check the error messages above.")
    
    with tab3:
        st.header("Data Processing Error Handling")
        
        if 'df' in locals() and df is not None:
            st.write("Select a data processing operation to test error handling:")
            
            processing_operation = st.selectbox(
                "Select operation",
                ["clean_missing", "normalize_numeric", "detect_outliers"]
            )
            
            if st.button("Process Data"):
                processed_df = safe_data_processing(df, processing_operation)
                
                if processed_df is not None:
                    st.success("Data processing completed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Data")
                        st.dataframe(df.head(), use_container_width=True)
                    
                    with col2:
                        st.subheader("Processed Data")
                        st.dataframe(processed_df.head(), use_container_width=True)
        
        else:
            st.info("Please upload a file in the 'File Upload' tab first.")
    
    with tab4:
        st.header("Error Boundary Demonstration")
        
        st.write("Test the error boundary decorator with different error types:")
        
        if st.button("Execute Function with Error Boundary"):
            result = demonstration_function_with_errors()
    
    with tab5:
        st.header("Input Validation Demonstration")
        
        st.write("Test various input validation scenarios:")
        
        # Numeric input validation
        st.subheader("Numeric Input Validation")
        
        with st.form("numeric_validation"):
            age = st.number_input("Enter your age", min_value=0, max_value=120, value=25)
            salary = st.number_input("Enter your salary", min_value=0.0, value=50000.0)
            
            submit_numeric = st.form_submit_button("Validate Numeric Inputs")
        
        if submit_numeric:
            # Validate age
            age_valid, age_message = DataValidator.validate_numeric_input(age, min_val=0, max_val=120)
            
            # Validate salary
            salary_valid, salary_message = DataValidator.validate_numeric_input(salary, min_val=0)
            
            if age_valid and salary_valid:
                st.success("All inputs are valid!")
                st.write(f"Age: {age}, Salary: ${salary:,.2f}")
            else:
                if not age_valid:
                    st.error(f"Age validation failed: {age_message}")
                if not salary_valid:
                    st.error(f"Salary validation failed: {salary_message}")
        
        # Custom validation example
        st.subheader("Custom Validation Example")
        
        with st.form("custom_validation"):
            email = st.text_input("Enter your email")
            password = st.text_input("Enter your password", type="password")
            
            submit_custom = st.form_submit_button("Validate Custom Inputs")
        
        if submit_custom:
            # Email validation
            if "@" not in email or "." not in email:
                st.error("Please enter a valid email address")
            else:
                st.success("Email format is valid")
            
            # Password validation
            if len(password) < 8:
                st.error("Password must be at least 8 characters long")
            elif not any(c.isupper() for c in password):
                st.error("Password must contain at least one uppercase letter")
            elif not any(c.isdigit() for c in password):
                st.error("Password must contain at least one number")
            else:
                st.success("Password meets requirements")
    
    # Error logging demonstration
    st.markdown("---")
    st.subheader("Application Health Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session Duration", f"{time.time() - st.session_state.get('start_time', time.time()):.0f}s")
    with col2:
        if 'error_count' not in st.session_state:
            st.session_state.error_count = 0
        st.metric("Errors Handled", st.session_state.error_count)
    with col3:
        st.metric("Status", "Healthy")


# Initialize session state
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()


if __name__ == "__main__":
    main()