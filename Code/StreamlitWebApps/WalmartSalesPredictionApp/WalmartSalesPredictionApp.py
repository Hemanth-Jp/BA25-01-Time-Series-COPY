import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="Walmart Sales Prediction",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state for model storage
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'model_source' not in st.session_state:
    st.session_state.model_source = None

# ============================================================================
# MODEL COMPATIBILITY HANDLING
# ============================================================================

import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to recreate a model if loading fails
def recreate_arima_model(params):
    """Attempt to recreate an ARIMA model from parameters if pickle loading fails"""
    try:
        # Basic recreation of ARIMA model
        # Note: This is simplified and may not work for all ARIMA models
        order = params.get('order', (1,1,1))
        model = ARIMA(np.array([0]), order=order)
        return model
    except Exception as e:
        warnings.warn(f"Failed to recreate ARIMA model: {str(e)}")
        return None

def load_default_model(model_type):
    """Load default model from models/default/ directory with improved error handling"""
    # Map display names to file names
    model_file_map = {
        "Auto ARIMA": "auto_arima",
        "Exponential Smoothing (Holt-Winters)": "exponential_smoothing"
    }
    
    file_name = model_file_map.get(model_type)
    if not file_name:
        return None, f"Invalid model type: {model_type}"
    
    model_path = f"models/default/{file_name}.pkl"
    
    if not os.path.exists(model_path):
        return None, f"Default model not found at {model_path}"
    
    try:
        # First try joblib
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle
            try:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                return model, None
            except Exception as pickle_error:
                # Generic error for model loading issue
                if model_type == "Auto ARIMA" and "statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error):
                    return None, "Error loading model. Please check the model file or try another model type."
                # Other errors
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_uploaded_model(uploaded_file, model_type):
    """Load model from uploaded file with improved error handling for cross-platform compatibility"""
    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # First try joblib
        try:
            model = joblib.load(tmp_path)
            # Clean up temporary file
            os.unlink(tmp_path)
            return model, None
        except Exception as joblib_error:
            # If joblib fails, try pickle
            try:
                with open(tmp_path, 'rb') as file:
                    model = pickle.load(file)
                # Clean up temporary file
                os.unlink(tmp_path)
                return model, None
            except Exception as pickle_error:
                # Generic error message
                if model_type == "Auto ARIMA" and "statsmodels" in str(joblib_error) or "statsmodels" in str(pickle_error):
                    st.warning("Loading issue detected. Attempting to reconstruct model...")
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    # Return a generic error message
                    return None, "Error loading model. Please check the model file or try another model type."
                # Other errors
                raise Exception(f"Failed to load model: {str(joblib_error)}\n{str(pickle_error)}")
    
    except Exception as e:
        # Clean up if error occurs
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass
        return None, f"Invalid model file: {str(e)}. Please check format or retrain."

def predict_next_4_weeks(model, model_type):
    """Predict next 4 weeks of sales"""
    # Generate dates for next 4 weeks
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, 5)]
    
    try:
        # Map display names to functional model types
        model_func_map = {
            "Auto ARIMA": "Auto ARIMA",
            "Exponential Smoothing (Holt-Winters)": "Exponential Smoothing (Holt-Winters)"
        }
        
        functional_model_type = model_func_map.get(model_type)
        if not functional_model_type:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if functional_model_type == "Auto ARIMA":
            predictions = model.predict(n_periods=4)
        elif functional_model_type == "Exponential Smoothing (Holt-Winters)":
            predictions = model.forecast(4)
        else:
            raise ValueError(f"Unknown model type: {functional_model_type}")
        
        return predictions, dates, None
    except Exception as e:
        return None, None, f"Error generating predictions: {str(e)}"

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

def main():
    # App title and description
    st.title("🔮 Walmart Sales Prediction")
    st.markdown("""
    This app generates sales forecasts for the next 4 weeks using trained time series models.
    
    **You can:**
    - Use pre-loaded default models (recommended)
    - Upload your own trained models
    - View interactive forecasts
    - Download prediction results
    """)
    
    # Model selection section
    st.header("🤖 Model Selection")
    
    # Tabs for default vs uploaded models
    tab1, tab2 = st.tabs(["Default Models", "Upload Model"])
    
    with tab1:
        st.subheader("Use Default Models")
        
        # Show only Exponential Smoothing model
        if st.button("Load Exponential Smoothing (Holt-Winters) Model", use_container_width=True):
            model, error = load_default_model("Exponential Smoothing (Holt-Winters)")
            if error:
                st.error(error)
            else:
                st.session_state.current_model = model
                st.session_state.model_type = "Exponential Smoothing (Holt-Winters)"
                st.session_state.model_source = "Default"
                st.success("✅ Exponential Smoothing (Holt-Winters) model loaded successfully!")
    
    with tab2:
        st.subheader("Upload Custom Model")
        
        # Model type selection for upload
        model_type = st.selectbox(
            "Select model type:",
            ["Auto ARIMA", "Exponential Smoothing (Holt-Winters)"],
            key="upload_model_type"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload model file (.pkl)", 
            type=["pkl"],
            key="model_uploader"
        )
        
        if uploaded_file:
            if st.button("Load Uploaded Model", use_container_width=True):
                model, error = load_uploaded_model(uploaded_file, model_type)
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_source = "Uploaded"
                    st.success(f"✅ {model_type} model loaded successfully!")
    
    # Display current model info
    if st.session_state.current_model is not None:
        st.info(f"**Current Model:** {st.session_state.model_type} ({st.session_state.model_source})")
    else:
        st.warning("No model loaded. Please select a model to make predictions.")
    
    # Prediction section
    st.header("📈 Generate Predictions")
    
    if st.session_state.current_model is not None:
        if st.button("Generate 4-Week Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                predictions, dates, error = predict_next_4_weeks(
                    st.session_state.current_model,
                    st.session_state.model_type
                )
                
                if error:
                    st.error(error)
                else:
                    # Create prediction dataframe
                    prediction_df = pd.DataFrame({
                        'Week': [f"Week {i+1}" for i in range(4)],
                        'Date': [d.strftime('%Y-%m-%d') for d in dates],
                        'Predicted_Sales': predictions
                    })
                    
                    # Display results
                    st.subheader("📊 Forecast Results")
                    
                    # Create interactive plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=prediction_df['Week'],
                        y=prediction_df['Predicted_Sales'],
                        mode='lines+markers',
                        name='Week-over-Week Sales Change',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title='Weekly Sales Change Forecast for Next 4 Weeks',
                        xaxis_title='Week',
                        yaxis_title='Sales Change ($)',
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    # Add horizontal reference line at y=0
                    fig.add_shape(
                        type="line",
                        x0=0,
                        y0=0,
                        x1=3,
                        y1=0,
                        line=dict(
                            color="gray",
                            width=1,
                            dash="dash",
                        )
                    )
                    
                    # Add grid
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    
                    # Add color to bars based on positive/negative
                    for i, value in enumerate(prediction_df['Predicted_Sales']):
                        color = "green" if value >= 0 else "red"
                        fig.add_trace(go.Bar(
                            x=[prediction_df['Week'][i]],
                            y=[value],
                            name=f"Week {i+1}",
                            marker_color=color,
                            opacity=0.7,
                            showlegend=False
                        ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display interpretation message
                    st.info("""
                    **How to interpret:** This forecast shows week-over-week sales changes, not absolute values.
                    - Positive values (green) indicate sales increases from previous week
                    - Negative values (red) indicate sales decreases from previous week
                    - Values represent dollar amount changes
                    """)
                    
                    # Display data table with colored text for values
                    st.subheader("📋 Prediction Values")
                    
                    # Create HTML for the colored data table
                    html_table = "<table width='100%' style='text-align: left;'><tr><th>Week</th><th>Date</th><th>Predicted Sales</th></tr>"
                    
                    for i, row in prediction_df.iterrows():
                        value = row['Predicted_Sales']
                        color = "green" if value >= 0 else "red"
                        formatted_value = f"${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"
                        
                        html_table += f"<tr><td>{row['Week']}</td><td>{row['Date']}</td>"
                        html_table += f"<td><span style='color: {color};'>{formatted_value}</span></td></tr>"
                    
                    html_table += "</table>"
                    
                    # Display the HTML table
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    # Format for download (keep numeric values)
                    download_df = prediction_df.copy()
                    download_df['Predicted_Sales'] = download_df['Predicted_Sales'].round(2)
                    
                    # Prepare CSV for download
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="walmart_sales_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create JSON version
                        json_str = download_df.to_json(orient='records')
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name="walmart_sales_predictions.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    
                    # Calculate cumulative impact
                    cumulative_impact = predictions.sum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Cumulative Sales Impact", 
                            f"${cumulative_impact:,.2f}" if cumulative_impact >= 0 else f"-${abs(cumulative_impact):,.2f}",
                            delta=f"{'+' if cumulative_impact >= 0 else ''}{cumulative_impact:,.2f}"
                        )
                    
                    with col2:
                        positive_weeks = sum(1 for x in predictions if x > 0)
                        st.metric("Growth Weeks", f"{positive_weeks} of 4")
                    
                    with col3:
                        best_week = predictions.argmax() + 1
                        worst_week = predictions.argmin() + 1
                        st.metric("Best/Worst Weeks", f"{best_week}/{worst_week}")
    
    else:
        st.info("👆 Please load a model first to generate predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Walmart Sales Forecasting System © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()