"""
@file InteractiveApp.py
@brief Interactive Streamlit Application with Advanced Widgets


This module demonstrates advanced Streamlit interactivity including session state,
form handling, and complex widget interactions for creating dynamic user experiences.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math


def initialize_session_state():
    """
    Initialize session state variables for the application.
    """
    if 'calculation_history' not in st.session_state:
        st.session_state.calculation_history = []
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'light',
            'chart_type': 'line',
            'show_grid': True
        }


def financial_calculator(principal, rate, time, compound_freq):
    """
    Calculate compound interest and return detailed breakdown.
    
    @param principal: Initial investment amount
    @param rate: Annual interest rate (as percentage)
    @param time: Investment period in years
    @param compound_freq: Compounding frequency per year
    @return dict: Calculation results and breakdown
    """
    # Convert rate to decimal
    r = rate / 100
    
    # Calculate compound interest
    amount = principal * (1 + r/compound_freq) ** (compound_freq * time)
    interest = amount - principal
    
    # Generate year-by-year breakdown
    years = []
    amounts = []
    interests = []
    
    for year in range(int(time) + 1):
        year_amount = principal * (1 + r/compound_freq) ** (compound_freq * year)
        year_interest = year_amount - principal
        
        years.append(year)
        amounts.append(year_amount)
        interests.append(year_interest)
    
    return {
        'final_amount': amount,
        'total_interest': interest,
        'years': years,
        'amounts': amounts,
        'interests': interests
    }


def data_analyzer(data, analysis_type):
    """
    Perform statistical analysis on uploaded data.
    
    @param data: DataFrame containing the data
    @param analysis_type: Type of analysis to perform
    @return dict: Analysis results
    """
    if data.empty:
        return None
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        return None
    
    results = {}
    
    if analysis_type == "Basic Statistics":
        results['statistics'] = data[numeric_columns].describe()
        results['correlation'] = data[numeric_columns].corr()
    
    elif analysis_type == "Distribution Analysis":
        results['skewness'] = data[numeric_columns].skew()
        results['kurtosis'] = data[numeric_columns].kurtosis()
    
    elif analysis_type == "Missing Data Analysis":
        results['missing_count'] = data.isnull().sum()
        results['missing_percentage'] = (data.isnull().sum() / len(data)) * 100
    
    return results


def main():
    """
    Main function to create the interactive Streamlit application.
    """
    # Initialize session state
    initialize_session_state()
    
    # Set page configuration
    st.set_page_config(
        page_title="Interactive Data App",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application header
    st.title("Interactive Data Application")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["Financial Calculator", "Data Analyzer", "Interactive Charts", "User Preferences"]
    )
    
    if app_mode == "Financial Calculator":
        financial_calculator_page()
    elif app_mode == "Data Analyzer":
        data_analyzer_page()
    elif app_mode == "Interactive Charts":
        interactive_charts_page()
    elif app_mode == "User Preferences":
        user_preferences_page()


def financial_calculator_page():
    """
    Create the financial calculator page with interactive widgets.
    """
    st.header("Financial Calculator")
    st.write("Calculate compound interest with interactive parameters.")
    
    # Create form for calculator inputs
    with st.form("financial_calculator"):
        col1, col2 = st.columns(2)
        
        with col1:
            principal = st.number_input(
                "Initial Investment ($)",
                min_value=1.0,
                max_value=1000000.0,
                value=10000.0,
                step=100.0
            )
            
            rate = st.slider(
                "Annual Interest Rate (%)",
                min_value=0.1,
                max_value=20.0,
                value=5.0,
                step=0.1
            )
        
        with col2:
            time = st.slider(
                "Investment Period (Years)",
                min_value=1,
                max_value=50,
                value=10
            )
            
            compound_freq = st.selectbox(
                "Compounding Frequency",
                [1, 2, 4, 12, 365],
                index=3,
                format_func=lambda x: {
                    1: "Annually",
                    2: "Semi-annually", 
                    4: "Quarterly",
                    12: "Monthly",
                    365: "Daily"
                }[x]
            )
        
        calculate_button = st.form_submit_button("Calculate", type="primary")
    
    if calculate_button:
        # Perform calculation
        results = financial_calculator(principal, rate, time, compound_freq)
        
        # Store in session state
        calculation = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'principal': principal,
            'rate': rate,
            'time': time,
            'compound_freq': compound_freq,
            'final_amount': results['final_amount'],
            'total_interest': results['total_interest']
        }
        st.session_state.calculation_history.append(calculation)
        
        # Display results
        st.success("Calculation completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Amount", f"${results['final_amount']:,.2f}")
        with col2:
            st.metric("Total Interest", f"${results['total_interest']:,.2f}")
        with col3:
            roi = (results['total_interest'] / principal) * 100
            st.metric("Return on Investment", f"{roi:.2f}%")
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['years'],
            y=results['amounts'],
            mode='lines+markers',
            name='Investment Growth',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.update_layout(
            title="Investment Growth Over Time",
            xaxis_title="Years",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display calculation history
    if st.session_state.calculation_history:
        st.subheader("Calculation History")
        history_df = pd.DataFrame(st.session_state.calculation_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.calculation_history = []
            st.rerun()


def data_analyzer_page():
    """
    Create the data analyzer page with file upload and analysis.
    """
    st.header("Data Analyzer")
    st.write("Upload your data file and perform statistical analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Shape: {data.shape}")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Basic Statistics", "Distribution Analysis", "Missing Data Analysis"]
            )
            
            if st.button("Perform Analysis", type="primary"):
                results = data_analyzer(data, analysis_type)
                
                if results:
                    st.subheader(f"Results: {analysis_type}")
                    
                    if analysis_type == "Basic Statistics":
                        st.write("**Descriptive Statistics:**")
                        st.dataframe(results['statistics'], use_container_width=True)
                        
                        st.write("**Correlation Matrix:**")
                        fig = px.imshow(
                            results['correlation'],
                            text_auto=True,
                            aspect="auto",
                            title="Correlation Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "Distribution Analysis":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Skewness:**")
                            st.dataframe(results['skewness'], use_container_width=True)
                        with col2:
                            st.write("**Kurtosis:**")
                            st.dataframe(results['kurtosis'], use_container_width=True)
                    
                    elif analysis_type == "Missing Data Analysis":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Missing Values Count:**")
                            st.dataframe(results['missing_count'], use_container_width=True)
                        with col2:
                            st.write("**Missing Values Percentage:**")
                            st.dataframe(results['missing_percentage'], use_container_width=True)
                
                else:
                    st.warning("No numeric columns found for analysis.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")


def interactive_charts_page():
    """
    Create interactive charts with customizable parameters.
    """
    st.header("Interactive Charts")
    st.write("Create and customize interactive visualizations.")
    
    # Generate sample data
    @st.cache_data
    def generate_chart_data():
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Value_A': np.cumsum(np.random.randn(100)) + 100,
            'Value_B': np.cumsum(np.random.randn(100)) + 50,
            'Category': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        return data
    
    chart_data = generate_chart_data()
    
    # Chart customization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"]
        )
    
    with col2:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["viridis", "plasma", "blues", "reds", "greens"]
        )
    
    with col3:
        show_grid = st.checkbox("Show Grid", value=True)
    
    # Create the selected chart
    if chart_type == "Line Chart":
        fig = px.line(
            chart_data, 
            x='Date', 
            y=['Value_A', 'Value_B'],
            title="Interactive Line Chart",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    
    elif chart_type == "Bar Chart":
        agg_data = chart_data.groupby('Category')[['Value_A', 'Value_B']].mean().reset_index()
        fig = px.bar(
            agg_data,
            x='Category',
            y=['Value_A', 'Value_B'],
            title="Interactive Bar Chart",
            color_discrete_sequence=px.colors.qualitative.Set2,
            barmode='group'
        )
    
    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            chart_data,
            x='Value_A',
            y='Value_B',
            color='Category',
            title="Interactive Scatter Plot",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    
    elif chart_type == "Histogram":
        fig = px.histogram(
            chart_data,
            x='Value_A',
            color='Category',
            title="Interactive Histogram",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    
    # Apply customizations
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        xaxis_showgrid=show_grid,
        yaxis_showgrid=show_grid
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart statistics
    st.subheader("Chart Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Points", len(chart_data))
    with col2:
        st.metric("Mean Value A", f"{chart_data['Value_A'].mean():.2f}")
    with col3:
        st.metric("Mean Value B", f"{chart_data['Value_B'].mean():.2f}")


def user_preferences_page():
    """
    Create user preferences page for application settings.
    """
    st.header("User Preferences")
    st.write("Customize your application experience.")
    
    with st.form("preferences"):
        theme = st.selectbox(
            "Application Theme",
            ["light", "dark"],
            index=0 if st.session_state.user_preferences['theme'] == 'light' else 1
        )
        
        default_chart = st.selectbox(
            "Default Chart Type",
            ["line", "bar", "scatter"],
            index=["line", "bar", "scatter"].index(st.session_state.user_preferences['chart_type'])
        )
        
        show_grid = st.checkbox(
            "Show Grid by Default",
            value=st.session_state.user_preferences['show_grid']
        )
        
        # Notification preferences
        st.subheader("Notifications")
        email_notifications = st.checkbox("Email Notifications", value=False)
        calculation_alerts = st.checkbox("Calculation Completion Alerts", value=True)
        
        submit_prefs = st.form_submit_button("Save Preferences", type="primary")
    
    if submit_prefs:
        # Update session state
        st.session_state.user_preferences.update({
            'theme': theme,
            'chart_type': default_chart,
            'show_grid': show_grid
        })
        
        st.success("Preferences saved successfully!")
        
        # Display current preferences
        st.subheader("Current Preferences")
        prefs_df = pd.DataFrame([st.session_state.user_preferences])
        st.dataframe(prefs_df, use_container_width=True)


if __name__ == "__main__":
    main()