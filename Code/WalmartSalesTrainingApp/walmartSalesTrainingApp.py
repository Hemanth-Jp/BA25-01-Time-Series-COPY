"""
@brief Streamlit web application for Walmart sales model training
@details This module provides a comprehensive user interface for training time series models
         on Walmart sales data. Features include data upload, preprocessing, hyperparameter
         tuning, model training, evaluation, and model download capabilities.
@author Sales Prediction Team
@date 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from walmartSalesTrainingCore import *

# Set page config for optimal user experience
st.set_page_config(
    page_title="Walmart Sales Model Training",
    page_icon="📈",
    layout="wide"
)

def main():
    """
    @brief Main application function that orchestrates the training interface
    @details Manages the complete workflow from data upload through model training
             and evaluation, including hyperparameter customization and result visualization
    @note Handles all user interactions and maintains training session state
    """
    # App title and comprehensive description
    st.title("📈 Walmart Sales Model Training")
    st.markdown("""
    This app allows you to train time series models for Walmart sales forecasting.
    
    **Steps:**
    1. Upload the required CSV files (train.csv, features.csv, stores.csv)
    2. Select a model to train (Auto ARIMA or Exponential Smoothing)
    3. Optionally customize hyperparameters
    4. Train the model and view diagnostic plots
    5. Download the trained model
    """)
    
    # Create file upload section with organized layout
    st.header("📁 Upload Dataset Files")
    col1, col2, col3 = st.columns(3)
    
    # Individual file uploaders for each required dataset
    with col1:
        train_file = st.file_uploader("Upload train.csv", type="csv")
    with col2:
        features_file = st.file_uploader("Upload features.csv", type="csv")
    with col3:
        stores_file = st.file_uploader("Upload stores.csv", type="csv")
    
    # Check if all required files are uploaded before proceeding
    if train_file and features_file and stores_file:
        st.success("✅ All files uploaded successfully!")
        
        # Load and preprocess data with comprehensive error handling
        with st.spinner("Loading and preprocessing data..."):
            try:
                # Input validation before calling core functions
                if not train_file or not features_file or not stores_file:
                    st.error("All three CSV files are required")
                    return
                
                # Execute data pipeline: load, merge, clean, and prepare for modeling
                df = load_and_merge_data(train_file, features_file, stores_file)
                df = clean_data(df)
                df_week, df_week_diff = prepare_time_series_data(df)
                
                # Split into train/test using configured ratio
                train_size = int(CONFIG['TRAIN_TEST_SPLIT'] * len(df_week_diff))
                train_data_diff = df_week_diff[:train_size]
                test_data_diff = df_week_diff[train_size:]
                
                # Display data information for user awareness
                st.write(f"**Data Info:**")
                st.write(f"- Training samples: {len(train_data_diff)}")
                st.write(f"- Testing samples: {len(test_data_diff)}")
                
                # Model selection section
                st.header("🤖 Model Selection")
                model_type = st.selectbox(
                    "Choose a model to train:",
                    ["Auto ARIMA", "Exponential Smoothing (Holt-Winters)"]
                )
                
                # Hyperparameter customization section with model-specific parameters
                st.header("⚙️ Hyperparameter Settings")
                col_left, col_right = st.columns(2)
                
                hyperparams = {}
                
                # Configure hyperparameters based on selected model type
                if model_type == "Auto ARIMA":
                    with col_left:
                        st.subheader("ARIMA Parameters")
                        # Basic ARIMA order parameters
                        hyperparams['start_p'] = st.number_input("Start p", value=0, min_value=0, max_value=10)
                        hyperparams['start_q'] = st.number_input("Start q", value=0, min_value=0, max_value=10)
                        hyperparams['max_p'] = st.number_input("Max p", value=CONFIG['DEFAULT_MAX_P'], min_value=1, max_value=30)
                        hyperparams['max_q'] = st.number_input("Max q", value=CONFIG['DEFAULT_MAX_Q'], min_value=1, max_value=30)
                    
                    with col_right:
                        st.subheader("Seasonal Parameters")
                        # Seasonal ARIMA parameters for capturing yearly patterns
                        hyperparams['start_P'] = st.number_input("Start P", value=0, min_value=0, max_value=10)
                        hyperparams['start_Q'] = st.number_input("Start Q", value=0, min_value=0, max_value=10)
                        hyperparams['max_P'] = st.number_input("Max P", value=CONFIG['DEFAULT_MAX_P_SEASONAL'], min_value=1, max_value=30)
                        hyperparams['max_Q'] = st.number_input("Max Q", value=CONFIG['DEFAULT_MAX_Q_SEASONAL'], min_value=1, max_value=30)
                    
                else:  # Exponential Smoothing configuration
                    with col_left:
                        st.subheader("Smoothing Parameters")
                        # Seasonal configuration for Holt-Winters method
                        hyperparams['seasonal_periods'] = st.number_input("Seasonal periods", value=CONFIG['DEFAULT_SEASONAL_PERIODS'], min_value=1, max_value=52)
                        hyperparams['seasonal'] = st.selectbox("Seasonal component", ["additive", "multiplicative"], index=0)
                    
                    with col_right:
                        st.subheader("Trend Parameters")
                        # Trend configuration for capturing long-term patterns
                        hyperparams['trend'] = st.selectbox("Trend component", ["additive", "multiplicative", None], index=0)
                        hyperparams['damped'] = st.checkbox("Damped trend", value=True)
                
                # Training section with comprehensive workflow
                st.header("🚀 Train Model")
                if st.button("Start Training", type="primary"):
                    with st.spinner(f"Training {model_type} model..."):
                        try:
                            # Input validation before training
                            if len(train_data_diff) == 0:
                                st.error("Training data is empty")
                                return
                            
                            # Train the selected model with specified hyperparameters
                            if model_type == "Auto ARIMA":
                                model = train_auto_arima(train_data_diff, hyperparams)
                                model_filename = "AutoARIMA.pkl"
                            else:
                                model = train_exponential_smoothing(train_data_diff, hyperparams)
                                model_filename = "ExponentialSmoothingHoltWinters.pkl"
                            
                            # Generate predictions for model evaluation
                            if model_type == "Auto ARIMA":
                                # Use predict method for ARIMA models
                                predictions = model.predict(n_periods=len(test_data_diff))
                            else:
                                # Use forecast method for Exponential Smoothing models
                                predictions = model.forecast(len(test_data_diff))
                            
                            # Calculate evaluation metric (WMAE)
                            wmae = wmae_ts(test_data_diff, predictions)
                            
                            # Create comprehensive diagnostic plots for model assessment
                            fig = create_diagnostic_plots(train_data_diff, test_data_diff, predictions, model_type)
                            
                            # Display training results and performance metrics
                            st.success("✅ Model training completed!")
                            st.metric("Weighted Mean Absolute Error (WMAE)", f"{wmae:.4f}")
                            
                            # Show diagnostic plot for visual model assessment
                            st.pyplot(fig)
                            plt.close()  # Clean up matplotlib resources
                            
                            # Save model to models/default/ directory for later use
                            default_dir = "models/default/"
                            os.makedirs(default_dir, exist_ok=True)  # Create directory if needed
                            model_path = os.path.join(default_dir, model_filename)
                            
                            # Save both model types using joblib for consistency
                            joblib.dump(model, model_path)
                            
                            st.success(f"Model saved to: {model_path}")
                            
                            # Provide download button for trained model
                            with open(model_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {model_type} Model",
                                    data=f,
                                    file_name=model_filename,
                                    mime="application/octet-stream"
                                )
                        
                        except ValueError as ve:
                            # Handle validation errors with specific messaging
                            st.error(f"Validation Error: {str(ve)}")
                        except Exception as e:
                            # Handle unexpected errors with full exception details
                            st.error(f"Error during training: {str(e)}")
                            st.exception(e)
            
            except ValueError as ve:
                # Handle data processing validation errors
                st.error(f"Data Processing Error: {str(ve)}")
            except Exception as e:
                # Handle unexpected data processing errors
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)
                
    else:
        # Display instruction message when files are missing
        st.info("👆 Please upload all three CSV files to continue")

if __name__ == "__main__":
    """
    @brief Entry point for running the training application
    @details Executes the main function when script is run directly
    @note Ensures proper application initialization and execution
    """
    main()