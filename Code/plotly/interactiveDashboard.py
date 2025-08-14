"""
Interactive Plotly Dashboard with Dash

This module demonstrates creating interactive web dashboards using Plotly Dash:
- Multi-component dashboard layout
- Interactive callbacks and state management
- Real-time data updates and filtering
- Advanced user interface elements


Version: 1.0
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_dashboard_data():
    """
    Generate comprehensive sample data for dashboard components.
    
    Returns:
        dict: Dictionary containing various datasets for dashboard use
    """
    np.random.seed(42)
    
    # Sales data over time
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.cumsum(np.random.randn(365) * 100) + 10000,
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'product': np.random.choice(['Product A', 'Product B', 'Product C'], 365),
        'customers': np.random.randint(50, 500, 365)
    })
    
    # Performance metrics
    metrics_data = pd.DataFrame({
        'metric': ['Revenue', 'Profit', 'Customers', 'Orders'],
        'current': [250000, 75000, 1250, 3500],
        'previous': [230000, 68000, 1180, 3200],
        'target': [280000, 85000, 1400, 4000]
    })
    
    # Geographic data
    geo_data = pd.DataFrame({
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
        'sales': np.random.randint(10000, 100000, 10),
        'customers': np.random.randint(100, 1000, 10)
    })
    
    return {
        'sales': sales_data,
        'metrics': metrics_data,
        'geography': geo_data
    }


class DashboardApp:
    """
    Main dashboard application class handling layout and callbacks.
    """
    
    def __init__(self):
        """Initialize the Dash application with data and layout."""
        self.app = dash.Dash(__name__)
        self.data = generate_dashboard_data()
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """
        Create the dashboard layout with multiple components.
        """
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Interactive Business Dashboard", 
                       style={'textAlign': 'center', 'color': '#2E86C1'}),
                html.Hr()
            ]),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Select Date Range:"),
                    dcc.DatePickerRange(
                        id='date-picker',
                        start_date=self.data['sales']['date'].min(),
                        end_date=self.data['sales']['date'].max(),
                        display_format='YYYY-MM-DD'
                    )
                ], className='six columns'),
                
                html.Div([
                    html.Label("Select Region:"),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': 'All Regions', 'value': 'all'}] +
                               [{'label': region, 'value': region} 
                                for region in self.data['sales']['region'].unique()],
                        value='all'
                    )
                ], className='six columns')
            ], className='row', style={'margin': '20px'}),
            
            # Key Metrics Cards
            html.Div([
                html.Div([
                    html.Div(id='metrics-cards')
                ], className='twelve columns')
            ], className='row', style={'margin': '20px'}),
            
            # Main Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='sales-timeline')
                ], className='eight columns'),
                
                html.Div([
                    dcc.Graph(id='product-breakdown')
                ], className='four columns')
            ], className='row', style={'margin': '20px'}),
            
            # Secondary Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='geographic-map')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='performance-comparison')
                ], className='six columns')
            ], className='row', style={'margin': '20px'}),
            
            # Data Table
            html.Div([
                html.H3("Detailed Data View"),
                html.Div(id='data-table')
            ], style={'margin': '20px'})
            
        ], style={'fontFamily': 'Arial, sans-serif'})
    
    def setup_callbacks(self):
        """
        Setup interactive callbacks for dashboard components.
        """
        
        @self.app.callback(
            [Output('metrics-cards', 'children'),
             Output('sales-timeline', 'figure'),
             Output('product-breakdown', 'figure'),
             Output('geographic-map', 'figure'),
             Output('performance-comparison', 'figure'),
             Output('data-table', 'children')],
            [Input('date-picker', 'start_date'),
             Input('date-picker', 'end_date'),
             Input('region-dropdown', 'value')]
        )
        def update_dashboard(start_date, end_date, selected_region):
            """
            Update all dashboard components based on user selections.
            
            Args:
                start_date (str): Start date for filtering
                end_date (str): End date for filtering
                selected_region (str): Selected region for filtering
            
            Returns:
                tuple: Updated components for all dashboard elements
            """
            # Filter data based on selections
            filtered_data = self.filter_data(start_date, end_date, selected_region)
            
            # Create updated components
            metrics_cards = self.create_metrics_cards(filtered_data)
            sales_fig = self.create_sales_timeline(filtered_data)
            product_fig = self.create_product_breakdown(filtered_data)
            geo_fig = self.create_geographic_map()
            comparison_fig = self.create_performance_comparison()
            data_table = self.create_data_table(filtered_data)
            
            return (metrics_cards, sales_fig, product_fig, 
                   geo_fig, comparison_fig, data_table)
    
    def filter_data(self, start_date, end_date, region):
        """
        Filter sales data based on user selections.
        
        Args:
            start_date (str): Start date for filtering
            end_date (str): End date for filtering
            region (str): Selected region ('all' for no filtering)
        
        Returns:
            pd.DataFrame: Filtered sales data
        """
        filtered = self.data['sales'].copy()
        
        # Date filtering
        if start_date and end_date:
            filtered = filtered[
                (filtered['date'] >= start_date) & 
                (filtered['date'] <= end_date)
            ]
        
        # Region filtering
        if region != 'all':
            filtered = filtered[filtered['region'] == region]
        
        return filtered
    
    def create_metrics_cards(self, data):
        """
        Create key performance indicator cards.
        
        Args:
            data (pd.DataFrame): Filtered sales data
        
        Returns:
            html.Div: Metrics cards layout
        """
        total_sales = data['sales'].sum()
        total_customers = data['customers'].sum()
        avg_order = total_sales / len(data) if len(data) > 0 else 0
        growth_rate = np.random.uniform(-5, 15)  # Simulated growth rate
        
        cards = html.Div([
            html.Div([
                html.H4(f"${total_sales:,.0f}", style={'color': '#2E86C1'}),
                html.P("Total Sales")
            ], className='three columns', style=self.card_style()),
            
            html.Div([
                html.H4(f"{total_customers:,}", style={'color': '#28B463'}),
                html.P("Total Customers")
            ], className='three columns', style=self.card_style()),
            
            html.Div([
                html.H4(f"${avg_order:.0f}", style={'color': '#F39C12'}),
                html.P("Average Order Value")
            ], className='three columns', style=self.card_style()),
            
            html.Div([
                html.H4(f"{growth_rate:.1f}%", style={'color': '#E74C3C'}),
                html.P("Growth Rate")
            ], className='three columns', style=self.card_style())
        ], className='row')
        
        return cards
    
    def create_sales_timeline(self, data):
        """
        Create sales timeline chart.
        
        Args:
            data (pd.DataFrame): Filtered sales data
        
        Returns:
            plotly.graph_objects.Figure: Sales timeline figure
        """
        # Aggregate daily sales
        daily_sales = data.groupby('date')['sales'].sum().reset_index()
        
        fig = px.line(daily_sales, x='date', y='sales',
                     title='Sales Timeline',
                     labels={'sales': 'Sales ($)', 'date': 'Date'})
        
        fig.update_layout(
            template='plotly_white',
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    
    def create_product_breakdown(self, data):
        """
        Create product sales breakdown chart.
        
        Args:
            data (pd.DataFrame): Filtered sales data
        
        Returns:
            plotly.graph_objects.Figure: Product breakdown figure
        """
        product_sales = data.groupby('product')['sales'].sum().reset_index()
        
        fig = px.pie(product_sales, values='sales', names='product',
                    title='Sales by Product')
        
        fig.update_layout(
            template='plotly_white',
            title_x=0.5
        )
        
        return fig
    
    def create_geographic_map(self):
        """
        Create geographic sales distribution map.
        
        Returns:
            plotly.graph_objects.Figure: Geographic map figure
        """
        geo_data = self.data['geography']
        
        fig = px.choropleth(
            geo_data,
            locations='state',
            color='sales',
            locationmode='USA-states',
            title='Sales by State',
            scope='usa',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            template='plotly_white',
            title_x=0.5,
            geo=dict(bgcolor='rgba(0,0,0,0)')
        )
        
        return fig
    
    def create_performance_comparison(self):
        """
        Create performance metrics comparison chart.
        
        Returns:
            plotly.graph_objects.Figure: Performance comparison figure
        """
        metrics = self.data['metrics']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics['metric'],
            y=metrics['current'],
            name='Current',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=metrics['metric'],
            y=metrics['target'],
            name='Target',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Current vs Target Performance',
            template='plotly_white',
            title_x=0.5,
            barmode='group'
        )
        
        return fig
    
    def create_data_table(self, data):
        """
        Create data table component.
        
        Args:
            data (pd.DataFrame): Filtered sales data
        
        Returns:
            html.Table: Data table component
        """
        # Show recent data sample
        recent_data = data.tail(10)
        
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th(col) for col in recent_data.columns
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(recent_data.iloc[i][col]) 
                    for col in recent_data.columns
                ]) for i in range(len(recent_data))
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
        
        return table
    
    def card_style(self):
        """
        Return consistent styling for metric cards.
        
        Returns:
            dict: CSS styling dictionary
        """
        return {
            'textAlign': 'center',
            'padding': '20px',
            'margin': '10px',
            'backgroundColor': '#F8F9FA',
            'border': '1px solid #E9ECEF',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    
    def run(self, debug=True, port=8050):
        """
        Run the dashboard application.
        
        Args:
            debug (bool): Enable debug mode
            port (int): Port number for the application
        """
        print(f"Starting dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)


def main():
    """
    Main function to create and run the interactive dashboard.
    """
    print("=== Interactive Plotly Dashboard Example ===\n")
    
    try:
        # Create dashboard instance
        dashboard = DashboardApp()
        
        print("Dashboard created successfully!")
        print("Features included:")
        print("- Interactive date and region filtering")
        print("- Real-time metric cards")
        print("- Multiple chart types (line, pie, choropleth, bar)")
        print("- Data table with filtered results")
        print("- Responsive layout with CSS styling\n")
        
        # Note: In a real application, you would call dashboard.run()
        # For documentation purposes, we'll just return the dashboard object
        print("To run the dashboard, call: dashboard.run()")
        
        return dashboard
        
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        return None


if __name__ == "__main__":
    # Create dashboard (but don't run in documentation mode)
    dashboard = main()
    
    # Uncomment the following line to actually run the dashboard
    # dashboard.run(debug=True, port=8050)
