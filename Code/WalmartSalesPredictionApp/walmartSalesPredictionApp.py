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
    """Validate inputs before calling core functions"""
    if not model:
        raise ValueError("Model cannot be None")
    if not model_type:
        raise ValueError("Model type cannot be empty")
    return True

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
            try:
                model, error = load_default_model("Exponential Smoothing (Holt-Winters)")
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = "Exponential Smoothing (Holt-Winters)"
                    st.session_state.model_source = "Default"
                    st.success("✅ Exponential Smoothing (Holt-Winters) model loaded successfully!")
            except ValueError as e:
                st.error(f"Input validation error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    
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
            f"Upload model file (.{CONFIG['SUPPORTED_EXTENSIONS'][0]})", 
            type=CONFIG['SUPPORTED_EXTENSIONS'],
            key="model_uploader"
        )
        
        if uploaded_file:
            if st.button("Load Uploaded Model", use_container_width=True):
                try:
                    model, error = load_uploaded_model(uploaded_file, model_type)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.current_model = model
                        st.session_state.model_type = model_type
                        st.session_state.model_source = "Uploaded"
                        st.success(f"✅ {model_type} model loaded successfully!")
                except ValueError as e:
                    st.error(f"Input validation error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    # Display current model info
    if st.session_state.current_model is not None:
        st.info(f"**Current Model:** {st.session_state.model_type} ({st.session_state.model_source})")
    else:
        st.warning("No model loaded. Please select a model to make predictions.")
    
    # Prediction section
    st.header("📈 Generate Predictions")
    
    if st.session_state.current_model is not None:
        if st.button(f"Generate {CONFIG['PREDICTION_PERIODS']}-Week Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                try:
                    validate_model_input(st.session_state.current_model, st.session_state.model_type)
                    
                    predictions, dates, error = predict_next_4_weeks(
                        st.session_state.current_model,
                        st.session_state.model_type
                    )
                    
                    if error:
                        st.error(error)
                    else:
                        # Create prediction dataframe
                        prediction_df = pd.DataFrame({
                            'Week': [f"Week {i+1}" for i in range(CONFIG['PREDICTION_PERIODS'])],
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
                            title=f'Weekly Sales Change Forecast for Next {CONFIG["PREDICTION_PERIODS"]} Weeks',
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
                            x1=CONFIG['PREDICTION_PERIODS']-1,
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
                            st.metric("Growth Weeks", f"{positive_weeks} of {CONFIG['PREDICTION_PERIODS']}")
                        
                        with col3:
                            best_week = predictions.argmax() + 1
                            worst_week = predictions.argmin() + 1
                            st.metric("Best/Worst Weeks", f"{best_week}/{worst_week}")
                
                except ValueError as e:
                    st.error(f"Input validation error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
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