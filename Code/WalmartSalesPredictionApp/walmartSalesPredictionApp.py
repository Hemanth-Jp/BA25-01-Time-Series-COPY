"""
@brief Streamlit web application for Walmart sales prediction
@details This module provides a user-friendly web interface for loading time series models
         and generating sales forecasts. Supports both default and uploaded models with
         interactive visualizations and downloadable results.
@author Sales Prediction Team
@date 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from walmartSalesPredictionCore import *

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

def validate_model_input(model, model_type):
    """
    @brief Validate inputs before calling core functions
    @details Performs basic validation to ensure model and model_type are properly set
    @param model The model object to validate
    @param model_type The model type string to validate
    @return True if validation passes
    @raises ValueError If model is None or model_type is empty
    @note This function helps catch validation errors early in the UI flow
    """
    if not model:
        raise ValueError("Model cannot be None")
    if not model_type:
        raise ValueError("Model type cannot be empty")
    return True

def main():
    """
    @brief Main application function that renders the Streamlit interface
    @details Orchestrates the entire user interface flow including model loading,
             prediction generation, and results visualization
    @note This function handles all UI state management and user interactions
    """
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
        
        # Show only Exponential Smoothing model (ARIMA removed due to compatibility issues)
        if st.button("Load Exponential Smoothing (Holt-Winters) Model", use_container_width=True):
            try:
                # Attempt to load the default Exponential Smoothing model
                model, error = load_default_model("Exponential Smoothing (Holt-Winters)")
                if error:
                    # Display error message if loading fails
                    st.error(error)
                else:
                    # Update session state with loaded model
                    st.session_state.current_model = model
                    st.session_state.model_type = "Exponential Smoothing (Holt-Winters)"
                    st.session_state.model_source = "Default"
                    st.success("✅ Exponential Smoothing (Holt-Winters) model loaded successfully!")
            except ValueError as e:
                # Handle validation errors
                st.error(f"Input validation error: {str(e)}")
            except Exception as e:
                # Handle unexpected errors
                st.error(f"Unexpected error: {str(e)}")
    
    with tab2:
        st.subheader("Upload Custom Model")
        
        # Model type selection for upload
        model_type = st.selectbox(
            "Select model type:",
            ["Auto ARIMA", "Exponential Smoothing (Holt-Winters)"],
            key="upload_model_type"
        )
        
        # File uploader widget
        uploaded_file = st.file_uploader(
            f"Upload model file (.{CONFIG['SUPPORTED_EXTENSIONS'][0]})", 
            type=CONFIG['SUPPORTED_EXTENSIONS'],
            key="model_uploader"
        )
        
        # Process uploaded file when available
        if uploaded_file:
            if st.button("Load Uploaded Model", use_container_width=True):
                try:
                    # Attempt to load the uploaded model
                    model, error = load_uploaded_model(uploaded_file, model_type)
                    if error:
                        # Display error message if loading fails
                        st.error(error)
                    else:
                        # Update session state with uploaded model
                        st.session_state.current_model = model
                        st.session_state.model_type = model_type
                        st.session_state.model_source = "Uploaded"
                        st.success(f"✅ {model_type} model loaded successfully!")
                except ValueError as e:
                    # Handle validation errors
                    st.error(f"Input validation error: {str(e)}")
                except Exception as e:
                    # Handle unexpected errors
                    st.error(f"Unexpected error: {str(e)}")
    
    # Display current model info
    if st.session_state.current_model is not None:
        st.info(f"**Current Model:** {st.session_state.model_type} ({st.session_state.model_source})")
    else:
        st.warning("No model loaded. Please select a model to make predictions.")
    
    # Prediction section
    st.header("📈 Generate Predictions")
    
    # Only show prediction interface if model is loaded
    if st.session_state.current_model is not None:
        if st.button(f"Generate {CONFIG['PREDICTION_PERIODS']}-Week Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                try:
                    # Validate model inputs before prediction
                    validate_model_input(st.session_state.current_model, st.session_state.model_type)
                    
                    # Generate predictions using the loaded model
                    predictions, dates, error = predict_next_4_weeks(
                        st.session_state.current_model,
                        st.session_state.model_type
                    )
                    
                    if error:
                        # Display prediction error
                        st.error(error)
                    else:
                        # Create prediction dataframe for display and download
                        prediction_df = pd.DataFrame({
                            'Week': [f"Week {i+1}" for i in range(CONFIG['PREDICTION_PERIODS'])],
                            'Date': [d.strftime('%Y-%m-%d') for d in dates],
                            'Predicted_Sales': predictions
                        })
                        
                        # Display results
                        st.subheader("📊 Forecast Results")
                        
                        # Create interactive plot using Plotly
                        fig = go.Figure()
                        
                        # Add main line plot for predictions
                        fig.add_trace(go.Scatter(
                            x=prediction_df['Week'],
                            y=prediction_df['Predicted_Sales'],
                            mode='lines+markers',
                            name='Week-over-Week Sales Change',
                            line=dict(color='blue', width=3),
                            marker=dict(size=10)
                        ))
                        
                        # Configure plot layout
                        fig.update_layout(
                            title=f'Weekly Sales Change Forecast for Next {CONFIG["PREDICTION_PERIODS"]} Weeks',
                            xaxis_title='Week',
                            yaxis_title='Sales Change ($)',
                            hovermode='x unified',
                            template='plotly_white',
                            height=500
                        )
                        
                        # Add horizontal reference line at y=0 for better interpretation
                        fig.add_shape(
                            type="line",
                            x0=0,
                            y0=0,
                            x1=CONFIG['PREDICTION_PERIODS']-1,
                            y1=0,
                            line=dict(
                                color="gray",
                                width=1,
                                dash="dash",
                            )
                        )
                        
                        # Add grid for better readability
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                        
                        # Add colored bars based on positive/negative values
                        for i, value in enumerate(prediction_df['Predicted_Sales']):
                            color = "green" if value >= 0 else "red"  # Green for growth, red for decline
                            fig.add_trace(go.Bar(
                                x=[prediction_df['Week'][i]],
                                y=[value],
                                name=f"Week {i+1}",
                                marker_color=color,
                                opacity=0.7,
                                showlegend=False
                            ))
                        
                        # Display the interactive chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display interpretation message for user understanding
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
                        
                        # Generate table rows with colored values
                        for i, row in prediction_df.iterrows():
                            value = row['Predicted_Sales']
                            color = "green" if value >= 0 else "red"  # Color coding for positive/negative
                            formatted_value = f"${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"
                            
                            html_table += f"<tr><td>{row['Week']}</td><td>{row['Date']}</td>"
                            html_table += f"<td><span style='color: {color};'>{formatted_value}</span></td></tr>"
                        
                        html_table += "</table>"
                        
                        # Display the HTML table
                        st.markdown(html_table, unsafe_allow_html=True)
                        
                        # Download section
                        st.subheader("💾 Download Results")
                        
                        # Format for download (keep numeric values, round to 2 decimals)
                        download_df = prediction_df.copy()
                        download_df['Predicted_Sales'] = download_df['Predicted_Sales'].round(2)
                        
                        # Prepare CSV for download
                        csv = download_df.to_csv(index=False).encode('utf-8')
                        
                        # Create download buttons in columns
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
                            # Create JSON version for download
                            json_str = download_df.to_json(orient='records')
                            st.download_button(
                                label="Download as JSON",
                                data=json_str,
                                file_name="walmart_sales_predictions.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        # Summary statistics section
                        st.subheader("📊 Summary Statistics")
                        
                        # Calculate cumulative impact across all predicted weeks
                        cumulative_impact = predictions.sum()
                        
                        # Display key metrics in columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Show total cumulative sales impact
                            st.metric(
                                "Cumulative Sales Impact", 
                                f"${cumulative_impact:,.2f}" if cumulative_impact >= 0 else f"-${abs(cumulative_impact):,.2f}",
                                delta=f"{'+' if cumulative_impact >= 0 else ''}{cumulative_impact:,.2f}"
                            )
                        
                        with col2:
                            # Count weeks with positive growth
                            positive_weeks = sum(1 for x in predictions if x > 0)
                            st.metric("Growth Weeks", f"{positive_weeks} of {CONFIG['PREDICTION_PERIODS']}")
                        
                        with col3:
                            # Identify best and worst performing weeks
                            best_week = predictions.argmax() + 1
                            worst_week = predictions.argmin() + 1
                            st.metric("Best/Worst Weeks", f"{best_week}/{worst_week}")
                
                except ValueError as e:
                    # Handle validation errors during prediction
                    st.error(f"Input validation error: {str(e)}")
                except Exception as e:
                    # Handle unexpected errors during prediction
                    st.error(f"Unexpected error: {str(e)}")
    
    else:
        # Display message when no model is loaded
        st.info("👆 Please load a model first to generate predictions")
    
    # Footer section
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Walmart Sales Forecasting System © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()