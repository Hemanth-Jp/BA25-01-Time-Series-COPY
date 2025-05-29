import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Walmart Sales Model Training",
    page_icon="📈",
    layout="wide"
)

# Import the core modeling functions from the reference code
# Note: In a real deployment, you'd import from core_modeling_code.py
# For this example, I'll include the necessary functions directly

import joblib
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# ============================================================================
# CORE MODELING FUNCTIONS (from reference code)
# ============================================================================

def load_and_merge_data(train_file, features_file, stores_file):
    """Load and merge the three CSV files"""
    df_store = pd.read_csv(stores_file)
    df_train = pd.read_csv(train_file)
    df_features = pd.read_csv(features_file)
    
    # Merge datasets
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    df.drop(['IsHoliday_y'], axis=1, inplace=True)
    df.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)
    
    return df

def clean_data(df):
    """Clean the merged data"""
    # Remove non-positive sales
    df = df.loc[df['Weekly_Sales'] > 0]
    
    # Fill missing values in markdown columns with zeros
    df = df.fillna(0)
    
    # Create specific holiday indicators
    df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
    df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False
    
    df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
    df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False
    
    df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
    df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False
    
    df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
    df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False
    
    return df

def prepare_time_series_data(df):
    """Prepare data for time series modeling"""
    # Convert date and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index('Date', inplace=True)
    
    # Create weekly aggregated data
    df_week = df.select_dtypes(include='number').resample('W').mean()
    
    # Difference the data for stationarity
    df_week_diff = df_week['Weekly_Sales'].diff().dropna()
    
    return df_week, df_week_diff

def train_auto_arima(train_data_diff, hyperparams=None):
    """Train Auto ARIMA model"""
    default_params = {
        'start_p': 0,
        'start_q': 0,
        'start_P': 0,
        'start_Q': 0,
        'max_p': 20,
        'max_q': 20,
        'max_P': 20,
        'max_Q': 20,
        'seasonal': True,
        'maxiter': 200,
        'information_criterion': 'aic',
        'stepwise': False,
        'suppress_warnings': True,
        'D': 1,
        'max_D': 10,
        'error_action': 'ignore',
        'approximation': False
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    model_auto_arima = auto_arima(train_data_diff, trace=True, **default_params)
    model_auto_arima.fit(train_data_diff)
    
    return model_auto_arima

def train_exponential_smoothing(train_data_diff, hyperparams=None):
    """Train Exponential Smoothing model"""
    default_params = {
        'seasonal_periods': 20,
        'seasonal': 'additive',
        'trend': 'additive',
        'damped': True
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    model_holt_winters = ExponentialSmoothing(
        train_data_diff,
        **default_params
    ).fit()
    
    return model_holt_winters

def wmae_ts(y_true, y_pred):
    """Calculate weighted mean absolute error"""
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values
    
    weights = np.ones_like(y_true)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def create_diagnostic_plots(train_data, test_data, predictions, model_type):
    """Create diagnostic plots for model evaluation"""
    plt.figure(figsize=(15, 6))
    plt.title(f'Prediction using {model_type}', fontsize=15)
    plt.plot(train_data.index, train_data.values, label='Train')
    plt.plot(test_data.index, test_data.values, label='Test')
    plt.plot(test_data.index, predictions, label='Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales (Differenced)')
    plt.grid(True)
    
    return plt.gcf()

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

def main():
    # App title and description
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
    
    # Create file upload section
    st.header("📁 Upload Dataset Files")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_file = st.file_uploader("Upload train.csv", type="csv")
    with col2:
        features_file = st.file_uploader("Upload features.csv", type="csv")
    with col3:
        stores_file = st.file_uploader("Upload stores.csv", type="csv")
    
    # Check if all files are uploaded
    if train_file and features_file and stores_file:
        st.success("✅ All files uploaded successfully!")
        
        # Load and preprocess data
        with st.spinner("Loading and preprocessing data..."):
            try:
                df = load_and_merge_data(train_file, features_file, stores_file)
                df = clean_data(df)
                df_week, df_week_diff = prepare_time_series_data(df)
                
                # Split into train/test
                train_size = int(0.7 * len(df_week_diff))
                train_data_diff = df_week_diff[:train_size]
                test_data_diff = df_week_diff[train_size:]
                
                st.write(f"**Data Info:**")
                st.write(f"- Training samples: {len(train_data_diff)}")
                st.write(f"- Testing samples: {len(test_data_diff)}")
                
                # Model selection section
                st.header("🤖 Model Selection")
                model_type = st.selectbox(
                    "Choose a model to train:",
                    ["Auto ARIMA", "Exponential Smoothing (Holt-Winters)"]
                )
                
                # Hyperparameter customization section
                st.header("⚙️ Hyperparameter Settings")
                col_left, col_right = st.columns(2)
                
                hyperparams = {}
                
                if model_type == "Auto ARIMA":
                    with col_left:
                        st.subheader("ARIMA Parameters")
                        hyperparams['start_p'] = st.number_input("Start p", value=0, min_value=0, max_value=10)
                        hyperparams['start_q'] = st.number_input("Start q", value=0, min_value=0, max_value=10)
                        hyperparams['max_p'] = st.number_input("Max p", value=20, min_value=1, max_value=30)
                        hyperparams['max_q'] = st.number_input("Max q", value=20, min_value=1, max_value=30)
                    
                    with col_right:
                        st.subheader("Seasonal Parameters")
                        hyperparams['start_P'] = st.number_input("Start P", value=0, min_value=0, max_value=10)
                        hyperparams['start_Q'] = st.number_input("Start Q", value=0, min_value=0, max_value=10)
                        hyperparams['max_P'] = st.number_input("Max P", value=20, min_value=1, max_value=30)
                        hyperparams['max_Q'] = st.number_input("Max Q", value=20, min_value=1, max_value=30)
                    
                else:  # Exponential Smoothing
                    with col_left:
                        st.subheader("Smoothing Parameters")
                        hyperparams['seasonal_periods'] = st.number_input("Seasonal periods", value=20, min_value=1, max_value=52)
                        hyperparams['seasonal'] = st.selectbox("Seasonal component", ["additive", "multiplicative"], index=0)
                    
                    with col_right:
                        st.subheader("Trend Parameters")
                        hyperparams['trend'] = st.selectbox("Trend component", ["additive", "multiplicative", None], index=0)
                        hyperparams['damped'] = st.checkbox("Damped trend", value=True)
                
                # Training section
                st.header("🚀 Train Model")
                if st.button("Start Training", type="primary"):
                    with st.spinner(f"Training {model_type} model..."):
                        try:
                            # Train the selected model
                            if model_type == "Auto ARIMA":
                                model = train_auto_arima(train_data_diff, hyperparams)
                                model_filename = "auto_arima.pkl"
                            else:
                                model = train_exponential_smoothing(train_data_diff, hyperparams)
                                model_filename = "exponential_smoothing.pkl"
                            
                            # Make predictions for evaluation
                            if model_type == "Auto ARIMA":
                                predictions = model.predict(n_periods=len(test_data_diff))
                            else:
                                predictions = model.forecast(len(test_data_diff))
                            
                            # Calculate WMAE
                            wmae = wmae_ts(test_data_diff, predictions)
                            
                            # Create diagnostic plots
                            fig = create_diagnostic_plots(train_data_diff, test_data_diff, predictions, model_type)
                            
                            # Display results
                            st.success("✅ Model training completed!")
                            st.metric("Weighted Mean Absolute Error (WMAE)", f"{wmae:.4f}")
                            
                            # Show diagnostic plot
                            st.pyplot(fig)
                            plt.close()
                            
                            # Save model to models/default/ directory
                            default_dir = "models/default/"
                            os.makedirs(default_dir, exist_ok=True)
                            model_path = os.path.join(default_dir, model_filename)
                            
                            # Save both model types using joblib for consistency
                            joblib.dump(model, model_path)
                            
                            st.success(f"Model saved to: {model_path}")
                            
                            # Download button
                            with open(model_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {model_type} Model",
                                    data=f,
                                    file_name=model_filename,
                                    mime="application/octet-stream"
                                )
                        
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.exception(e)
                
    else:
        st.info("👆 Please upload all three CSV files to continue")

if __name__ == "__main__":
    main()