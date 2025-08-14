"""

This module demonstrates the creation of a basic data dashboard using Streamlit.
It showcases fundamental Streamlit components including data display, charts,
and basic interactivity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta


def generate_sample_data():
    """
    Generate sample data for the dashboard demonstration.
    
    @return pandas.DataFrame: Sample dataset with sales data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate sample sales data
    data = {
        'Date': dates,
        'Sales': np.random.normal(1000, 200, len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates))
    }
    
    # Ensure positive sales values
    df = pd.DataFrame(data)
    df['Sales'] = np.abs(df['Sales'])
    
    return df


def main():
    """
    Main function to create the Streamlit dashboard.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Sales Dashboard",
        page_icon=":bar_chart:",
        layout="wide"
    )
    
    # Dashboard title and description
    st.title("Sales Dashboard")
    st.markdown("---")
    st.write("Welcome to the interactive sales dashboard. Explore your data below!")
    
    # Generate and cache data
    @st.cache_data
    def load_data():
        return generate_sample_data()
    
    # Load data
    df = load_data()
    
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    # Date range selector
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Product filter
    products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product'].unique(),
        default=df['Product'].unique()
    )
    
    # Filter data based on selections
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[
            (df['Date'].dt.date >= start_date) &
            (df['Date'].dt.date <= end_date) &
            (df['Region'].isin(regions)) &
            (df['Product'].isin(products))
        ]
    else:
        filtered_df = df[
            (df['Region'].isin(regions)) &
            (df['Product'].isin(products))
        ]
    
    # Main dashboard content
    if not filtered_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = filtered_df['Sales'].sum()
            st.metric("Total Sales", f"${total_sales:,.0f}")
        
        with col2:
            avg_sales = filtered_df['Sales'].mean()
            st.metric("Average Daily Sales", f"${avg_sales:,.0f}")
        
        with col3:
            max_sales = filtered_df['Sales'].max()
            st.metric("Highest Daily Sales", f"${max_sales:,.0f}")
        
        with col4:
            num_records = len(filtered_df)
            st.metric("Number of Records", f"{num_records:,}")
        
        st.markdown("---")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Trend Over Time")
            # Create line chart using Plotly
            fig_line = px.line(
                filtered_df, 
                x='Date', 
                y='Sales',
                title="Daily Sales Trend"
            )
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.subheader("Sales by Region")
            # Create bar chart for regional sales
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            fig_bar = px.bar(
                region_sales,
                x='Region',
                y='Sales',
                title="Total Sales by Region",
                color='Sales',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Additional visualizations
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Performance")
            # Create pie chart for product sales distribution
            product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
            fig_pie = px.pie(
                product_sales,
                values='Sales',
                names='Product',
                title="Sales Distribution by Product"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Sales Statistics")
            # Display descriptive statistics
            stats_df = filtered_df['Sales'].describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
            
            # Additional insights
            st.write("**Key Insights:**")
            best_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
            best_product = filtered_df.groupby('Product')['Sales'].sum().idxmax()
            
            st.write(f"* Best performing region: **{best_region}**")
            st.write(f"* Best performing product: **{best_product}**")
        
        # Data table section
        st.markdown("---")
        st.subheader("Raw Data")
        
        # Show/hide raw data
        if st.checkbox("Show raw data"):
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with Streamlit*")


if __name__ == "__main__":
    main()