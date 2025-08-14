"""
Basic Plotly Visualization Examples

This module demonstrates fundamental Plotly capabilities including:
- Express interface for quick plotting
- Graph Objects for detailed customization
- Multiple chart types and styling options
- Data handling and visualization best practices


Version: 1.0
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_sample_data():
    """
    Generate sample datasets for visualization examples.
    
    Returns:
        tuple: Contains sample DataFrames for different chart types
    """
    np.random.seed(42)
    
    # Sample dataset for scatter plots
    n_points = 100
    scatter_data = pd.DataFrame({
        'x': np.random.randn(n_points),
        'y': np.random.randn(n_points) * 2 + 3,
        'category': np.random.choice(['A', 'B', 'C'], n_points),
        'size': np.random.randint(10, 50, n_points)
    })
    
    # Time series data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    time_series = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(50)) + 100,
        'volume': np.random.randint(1000, 5000, 50)
    })
    
    # Categorical data for bar charts
    categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    bar_data = pd.DataFrame({
        'product': categories,
        'sales': np.random.randint(50, 200, len(categories)),
        'profit': np.random.randint(10, 50, len(categories))
    })
    
    return scatter_data, time_series, bar_data


def create_express_visualizations():
    """
    Demonstrate Plotly Express interface for rapid visualization creation.
    
    Creates various chart types using the high-level Express API.
    """
    print("Creating Plotly Express visualizations...")
    
    scatter_data, time_series, bar_data = generate_sample_data()
    
    # 1. Scatter Plot with Color and Size Mapping
    scatter_fig = px.scatter(
        scatter_data, 
        x='x', 
        y='y',
        color='category',
        size='size',
        title='Interactive Scatter Plot with Express',
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
        hover_data=['size']
    )
    
    # Customize the scatter plot
    scatter_fig.update_layout(
        font=dict(size=12),
        title_x=0.5,  # Center the title
        showlegend=True
    )
    
    # 2. Time Series Line Plot
    line_fig = px.line(
        time_series, 
        x='date', 
        y='value',
        title='Time Series Visualization',
        labels={'date': 'Date', 'value': 'Value ($)'}
    )
    
    # Add volume as a secondary trace
    line_fig.add_scatter(
        x=time_series['date'],
        y=time_series['volume'] / 50,  # Scale for visibility
        mode='lines',
        name='Volume (scaled)',
        yaxis='y2'
    )
    
    # Configure secondary y-axis
    line_fig.update_layout(
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        )
    )
    
    # 3. Bar Chart with Multiple Variables
    bar_fig = px.bar(
        bar_data, 
        x='product', 
        y='sales',
        title='Product Sales Performance',
        color='profit',
        color_continuous_scale='Viridis'
    )
    
    # Customize bar chart appearance
    bar_fig.update_traces(
        texttemplate='%{y}',
        textposition='outside'
    )
    
    # Display the figures
    scatter_fig.show()
    line_fig.show()
    bar_fig.show()
    
    return scatter_fig, line_fig, bar_fig


def create_graph_objects_visualizations():
    """
    Demonstrate Graph Objects interface for detailed plot customization.
    
    Creates complex visualizations using the low-level Graph Objects API.
    """
    print("Creating Graph Objects visualizations...")
    
    scatter_data, time_series, bar_data = generate_sample_data()
    
    # 1. Custom Scatter Plot with Graph Objects
    fig = go.Figure()
    
    # Add traces for each category
    for category in scatter_data['category'].unique():
        category_data = scatter_data[scatter_data['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['x'],
            y=category_data['y'],
            mode='markers',
            name=f'Category {category}',
            marker=dict(
                size=category_data['size'] / 2,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=category_data['size'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Size: %{text}<br>' +
                         '<extra></extra>'
        ))
    
    # Customize layout
    fig.update_layout(
        title='Custom Scatter Plot with Graph Objects',
        title_x=0.5,
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )
    
    # 2. Subplot Example with Mixed Chart Types
    subplot_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scatter Plot', 'Bar Chart', 'Line Plot', 'Histogram'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Add scatter plot
    subplot_fig.add_trace(
        go.Scatter(x=scatter_data['x'], y=scatter_data['y'], mode='markers'),
        row=1, col=1
    )
    
    # Add bar chart
    subplot_fig.add_trace(
        go.Bar(x=bar_data['product'], y=bar_data['sales']),
        row=1, col=2
    )
    
    # Add line plot with secondary y-axis
    subplot_fig.add_trace(
        go.Scatter(x=time_series['date'], y=time_series['value'], mode='lines'),
        row=2, col=1
    )
    
    subplot_fig.add_trace(
        go.Scatter(x=time_series['date'], y=time_series['volume'], 
                  mode='lines', name='Volume'),
        row=2, col=1, secondary_y=True
    )
    
    # Add histogram
    subplot_fig.add_trace(
        go.Histogram(x=scatter_data['x'], nbinsx=20),
        row=2, col=2
    )
    
    # Update subplot layout
    subplot_fig.update_layout(
        title_text="Multiple Chart Types in Subplots",
        title_x=0.5,
        showlegend=False,
        height=800
    )
    
    # Display the figures
    fig.show()
    subplot_fig.show()
    
    return fig, subplot_fig


def demonstrate_styling_options():
    """
    Demonstrate advanced styling and customization options.
    
    Shows how to apply themes, custom colors, and professional styling.
    """
    print("Demonstrating styling options...")
    
    scatter_data, _, _ = generate_sample_data()
    
    # Create a professionally styled scatter plot
    fig = px.scatter(
        scatter_data,
        x='x',
        y='y',
        color='category',
        size='size',
        title='Professional Styling Example'
    )
    
    # Apply custom styling
    fig.update_layout(
        # Theme and colors
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        
        # Typography
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2E2E2E"
        ),
        
        # Title styling
        title=dict(
            font=dict(size=16, color="#1F1F1F"),
            x=0.5,
            xanchor='center'
        ),
        
        # Axis styling
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=14)
        ),
        
        # Legend styling
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(128,128,128,0.5)",
            borderwidth=1
        )
    )
    
    # Update traces for better visual appeal
    fig.update_traces(
        marker=dict(
            opacity=0.8,
            line=dict(width=0.5, color='white')
        )
    )
    
    fig.show()
    
    return fig


def main():
    """
    Main function demonstrating basic Plotly visualization capabilities.
    
    Executes all example functions and provides a comprehensive overview
    of Plotly's core features.
    """
    print("=== Basic Plotly Visualization Examples ===\n")
    
    try:
        # Generate sample data
        print("Generating sample data...")
        sample_data = generate_sample_data()
        print(f"Generated {len(sample_data)} sample datasets\n")
        
        # Create Express visualizations
        express_figs = create_express_visualizations()
        print("Express visualizations created successfully\n")
        
        # Create Graph Objects visualizations
        go_figs = create_graph_objects_visualizations()
        print("Graph Objects visualizations created successfully\n")
        
        # Demonstrate styling
        styled_fig = demonstrate_styling_options()
        print("Styling demonstration completed successfully\n")
        
        print("=== All visualizations completed successfully ===")
        
        return {
            'express_figures': express_figs,
            'graph_objects_figures': go_figs,
            'styled_figure': styled_fig
        }
        
    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
