"""
Plotly Error Handling and Best Practices

This module demonstrates comprehensive error handling patterns and best practices
for robust Plotly applications:
- Data validation and sanitization
- Performance optimization for large datasets
- Browser compatibility and rendering issues
- Export and deployment error handling


Version: 1.0
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Union, Optional, Dict, Any, List
from functools import wraps
import time

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PlotlyErrorHandler:
    """
    Comprehensive error handling class for Plotly applications.
    
    Provides methods for data validation, performance optimization,
    and robust visualization creation with fallback options.
    """
    
    def __init__(self):
        """Initialize the error handler with default configurations."""
        self.max_data_points = 50000
        self.performance_threshold = 2.0  # seconds
        self.supported_formats = ['html', 'png', 'pdf', 'svg', 'jpeg']
        self.fallback_colors = ['blue', 'red', 'green', 'orange', 'purple']
        
    def performance_monitor(self, func):
        """
        Decorator to monitor function performance and log slow operations.
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function with performance monitoring
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > self.performance_threshold:
                    logger.warning(f"Slow operation detected: {func.__name__} "
                                 f"took {execution_time:.2f} seconds")
                else:
                    logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Error in {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame structure and content for plotting.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_result['errors'].append("DataFrame is empty")
                validation_result['is_valid'] = False
                return validation_result
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                    validation_result['is_valid'] = False
            
            # Check for excessive data points
            if len(df) > self.max_data_points:
                validation_result['warnings'].append(
                    f"Large dataset detected ({len(df)} rows). "
                    f"Consider sampling to improve performance."
                )
                validation_result['recommendations'].append("Use data sampling or WebGL rendering")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                columns_with_missing = missing_counts[missing_counts > 0].to_dict()
                validation_result['warnings'].append(f"Missing values detected: {columns_with_missing}")
                validation_result['recommendations'].append("Consider imputation or filtering missing values")
            
            # Check for infinite values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if np.isinf(df[col]).any():
                    validation_result['warnings'].append(f"Infinite values detected in column: {col}")
                    validation_result['recommendations'].append(f"Filter infinite values in {col}")
            
            # Check data types
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.5 and len(df) > 100:
                        validation_result['warnings'].append(
                            f"High cardinality in column {col} ({df[col].nunique()} unique values)"
                        )
                        validation_result['recommendations'].append(
                            f"Consider grouping or filtering {col} for better visualization"
                        )
            
            logger.info(f"DataFrame validation completed: {len(validation_result['errors'])} errors, "
                       f"{len(validation_result['warnings'])} warnings")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
            logger.error(f"DataFrame validation failed: {str(e)}")
        
        return validation_result
    
    def safe_data_sampling(self, df: pd.DataFrame, max_points: Optional[int] = None) -> pd.DataFrame:
        """
        Safely sample large datasets while preserving data characteristics.
        
        Args:
            df: DataFrame to sample
            max_points: Maximum number of points to retain
            
        Returns:
            Sampled DataFrame
        """
        if max_points is None:
            max_points = self.max_data_points
        
        try:
            if len(df) <= max_points:
                return df
            
            # Stratified sampling for categorical data
            if 'category' in df.columns or any(df.dtypes == 'object'):
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    # Sample proportionally from each category
                    sampled_dfs = []
                    for cat_col in categorical_cols[:1]:  # Use first categorical column
                        for category in df[cat_col].unique():
                            cat_data = df[df[cat_col] == category]
                            n_samples = min(len(cat_data), max_points // df[cat_col].nunique())
                            if n_samples > 0:
                                sampled_dfs.append(cat_data.sample(n=n_samples, random_state=42))
                    
                    if sampled_dfs:
                        result = pd.concat(sampled_dfs, ignore_index=True)
                        logger.info(f"Stratified sampling: {len(df)} -> {len(result)} rows")
                        return result
            
            # Random sampling as fallback
            result = df.sample(n=max_points, random_state=42)
            logger.info(f"Random sampling: {len(df)} -> {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Sampling failed: {str(e)}. Returning original data.")
            return df
    
    def create_robust_figure(self, plot_func, *args, **kwargs) -> Optional[go.Figure]:
        """
        Create figure with comprehensive error handling and fallback options.
        
        Args:
            plot_func: Function to create the plot
            *args: Arguments for plot function
            **kwargs: Keyword arguments for plot function
            
        Returns:
            Plotly figure or None if creation fails
        """
        try:
            # Attempt to create the figure
            fig = plot_func(*args, **kwargs)
            
            # Validate the figure
            if not isinstance(fig, (go.Figure, type(px.scatter()))):
                raise ValueError("Plot function did not return a valid figure")
            
            # Apply standard optimizations
            fig = self.optimize_figure_performance(fig)
            
            logger.info(f"Figure created successfully using {plot_func.__name__}")
            return fig
            
        except MemoryError:
            logger.error("Memory error encountered. Attempting data reduction.")
            # Try with reduced data if possible
            if args and isinstance(args[0], pd.DataFrame):
                reduced_data = self.safe_data_sampling(args[0], max_points=10000)
                try:
                    fig = plot_func(reduced_data, *args[1:], **kwargs)
                    fig = self.optimize_figure_performance(fig)
                    logger.info("Figure created with reduced dataset")
                    return fig
                except Exception as e:
                    logger.error(f"Fallback attempt failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Figure creation failed: {str(e)}")
            # Create minimal fallback figure
            return self.create_fallback_figure(error_message=str(e))
        
        return None
    
    def optimize_figure_performance(self, fig: go.Figure) -> go.Figure:
        """
        Apply performance optimizations to figure.
        
        Args:
            fig: Plotly figure to optimize
            
        Returns:
            Optimized figure
        """
        try:
            # Check if WebGL should be used for large datasets
            total_points = 0
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    total_points += len(trace.x)
            
            if total_points > 10000:
                # Convert scatter plots to WebGL
                new_data = []
                for trace in fig.data:
                    if trace.type == 'scatter' and trace.mode and 'markers' in trace.mode:
                        # Convert to Scattergl for better performance
                        new_trace = go.Scattergl(
                            x=trace.x,
                            y=trace.y,
                            mode=trace.mode,
                            name=trace.name,
                            marker=trace.marker
                        )
                        new_data.append(new_trace)
                    else:
                        new_data.append(trace)
                
                fig.data = new_data
                logger.info("Applied WebGL optimization for large dataset")
            
            # Optimize layout for performance
            fig.update_layout(
                hovermode='closest',  # More efficient than 'x' or 'y'
                dragmode='pan',       # Disable selection for better performance
                showlegend=len(fig.data) <= 10  # Hide legend for many traces
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"Performance optimization failed: {str(e)}")
            return fig
    
    def create_fallback_figure(self, error_message: str = "Unknown error") -> go.Figure:
        """
        Create a simple fallback figure when primary visualization fails.
        
        Args:
            error_message: Error message to display
            
        Returns:
            Simple fallback figure
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Visualization Error<br>{error_message}<br><br>Please check your data and try again.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2
        )
        
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white'
        )
        
        logger.info("Created fallback error figure")
        return fig
    
    def safe_export(self, fig: go.Figure, filename: str, format: str = 'html', 
                   **export_kwargs) -> bool:
        """
        Safely export figure with comprehensive error handling.
        
        Args:
            fig: Figure to export
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg', 'jpeg')
            **export_kwargs: Additional export arguments
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Validate format
            if format.lower() not in self.supported_formats:
                logger.error(f"Unsupported format: {format}. Supported: {self.supported_formats}")
                return False
            
            # Validate figure
            if not isinstance(fig, go.Figure):
                logger.error("Invalid figure object for export")
                return False
            
            # Attempt export based on format
            if format.lower() == 'html':
                fig.write_html(filename, **export_kwargs)
            elif format.lower() in ['png', 'jpeg', 'pdf', 'svg']:
                # Requires kaleido or orca
                try:
                    if format.lower() == 'png':
                        fig.write_image(filename, format='png', **export_kwargs)
                    elif format.lower() == 'jpeg':
                        fig.write_image(filename, format='jpeg', **export_kwargs)
                    elif format.lower() == 'pdf':
                        fig.write_image(filename, format='pdf', **export_kwargs)
                    elif format.lower() == 'svg':
                        fig.write_image(filename, format='svg', **export_kwargs)
                except Exception as e:
                    if "kaleido" in str(e).lower() or "orca" in str(e).lower():
                        logger.error("Static image export requires kaleido. Install with: pip install kaleido")
                        # Fallback to HTML export
                        html_filename = filename.rsplit('.', 1)[0] + '.html'
                        fig.write_html(html_filename)
                        logger.info(f"Exported as HTML instead: {html_filename}")
                        return True
                    else:
                        raise
            
            logger.info(f"Successfully exported figure to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False


def demonstrate_error_handling():
    """
    Demonstrate various error handling scenarios and best practices.
    
    Shows how to handle common issues like missing data, large datasets,
    invalid inputs, and export problems.
    """
    print("=== Plotly Error Handling Demonstration ===\n")
    
    # Initialize error handler
    handler = PlotlyErrorHandler()
    
    # Test data validation
    print("1. Testing data validation...")
    
    # Create problematic test data
    problematic_data = pd.DataFrame({
        'x': [1, 2, np.inf, 4, 5],
        'y': [1, np.nan, 3, 4, 5],
        'category': ['A', 'B', 'C', 'D', 'E'],
        'large_text': ['text_' + str(i) * 100 for i in range(5)]
    })
    
    validation = handler.validate_dataframe(problematic_data, required_columns=['x', 'y'])
    print(f"Validation result: {validation['is_valid']}")
    print(f"Warnings: {len(validation['warnings'])}")
    print(f"Errors: {len(validation['errors'])}")
    
    # Test large dataset handling
    print("\n2. Testing large dataset handling...")
    large_data = pd.DataFrame({
        'x': np.random.randn(100000),
        'y': np.random.randn(100000),
        'category': np.random.choice(['A', 'B', 'C'], 100000)
    })
    
    sampled_data = handler.safe_data_sampling(large_data, max_points=5000)
    print(f"Original size: {len(large_data)}, Sampled size: {len(sampled_data)}")
    
    # Test robust figure creation
    print("\n3. Testing robust figure creation...")
    
    @handler.performance_monitor
    def create_test_plot(data):
        return px.scatter(data, x='x', y='y', color='category', title='Test Plot')
    
    # Test with good data
    good_data = pd.DataFrame({
        'x': np.random.randn(1000),
        'y': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    fig = handler.create_robust_figure(create_test_plot, good_data)
    if fig:
        print("Good data plot created successfully")
    
    # Test with bad data
    print("\n4. Testing fallback handling...")
    bad_data = pd.DataFrame({'invalid': [None, None, None]})
    
    fig_fallback = handler.create_robust_figure(create_test_plot, bad_data)
    if fig_fallback:
        print("Fallback figure created for invalid data")
    
    # Test export functionality
    print("\n5. Testing export functionality...")
    if fig:
        success = handler.safe_export(fig, 'test_plot.html', format='html')
        print(f"HTML export successful: {success}")
        
        # Test static export (may fail without kaleido)
        success_png = handler.safe_export(fig, 'test_plot.png', format='png')
        print(f"PNG export successful: {success_png}")
    
    print("\n=== Error handling demonstration completed ===")


def create_production_ready_plot(data: pd.DataFrame, x_col: str, y_col: str, 
                                color_col: Optional[str] = None, 
                                title: str = "Production Plot") -> Optional[go.Figure]:
    """
    Create a production-ready plot with comprehensive error handling.
    
    Args:
        data: Input DataFrame
        x_col: X-axis column name
        y_col: Y-axis column name
        color_col: Optional color column name
        title: Plot title
        
    Returns:
        Plotly figure or None if creation fails
    """
    handler = PlotlyErrorHandler()
    
    try:
        # Validate input data
        required_cols = [x_col, y_col]
        if color_col:
            required_cols.append(color_col)
        
        validation = handler.validate_dataframe(data, required_columns=required_cols)
        
        if not validation['is_valid']:
            logger.error(f"Data validation failed: {validation['errors']}")
            return handler.create_fallback_figure("Data validation failed")
        
        # Log warnings and recommendations
        for warning in validation['warnings']:
            logger.warning(warning)
        for rec in validation['recommendations']:
            logger.info(f"Recommendation: {rec}")
        
        # Sample data if too large
        processed_data = handler.safe_data_sampling(data)
        
        # Clean data
        processed_data = processed_data.dropna(subset=[x_col, y_col])
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if processed_data.empty:
            logger.error("No valid data remaining after cleaning")
            return handler.create_fallback_figure("No valid data after cleaning")
        
        # Create the plot
        plot_kwargs = {
            'x': x_col,
            'y': y_col,
            'title': title,
            'template': 'plotly_white'
        }
        
        if color_col:
            plot_kwargs['color'] = color_col
        
        fig = px.scatter(processed_data, **plot_kwargs)
        
        # Optimize for performance
        fig = handler.optimize_figure_performance(fig)
        
        # Add error information as subtitle if warnings exist
        if validation['warnings']:
            warning_text = f"Note: {len(validation['warnings'])} data quality issues detected"
            fig.add_annotation(
                text=warning_text,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color="orange")
            )
        
        logger.info("Production-ready plot created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Production plot creation failed: {str(e)}")
        return handler.create_fallback_figure(f"Plot creation error: {str(e)}")


def advanced_error_recovery_example():
    """
    Demonstrate advanced error recovery techniques for complex visualizations.
    
    Shows progressive degradation strategies and intelligent fallbacks
    for maintaining functionality under adverse conditions.
    """
    print("=== Advanced Error Recovery Example ===\n")
    
    handler = PlotlyErrorHandler()
    
    # Simulate complex multi-panel dashboard creation
    def create_complex_dashboard(data_sources: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
        """Create a complex dashboard with multiple data sources."""
        
        try:
            # Attempt full dashboard creation
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Sales Trend', 'Category Distribution', 
                               'Geographic Analysis', 'Performance Metrics'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "geo"}, {"secondary_y": True}]]
            )
            
            # Panel 1: Sales trend
            if 'sales' in data_sources and not data_sources['sales'].empty:
                sales_data = handler.safe_data_sampling(data_sources['sales'])
                fig.add_trace(
                    go.Scatter(x=sales_data.get('date', range(len(sales_data))),
                              y=sales_data.get('amount', sales_data.iloc[:, 0]),
                              mode='lines', name='Sales'),
                    row=1, col=1
                )
            else:
                # Fallback: Add placeholder
                fig.add_annotation(text="Sales data unavailable", 
                                 xref="x", yref="y", x=0.5, y=0.5,
                                 row=1, col=1)
            
            # Panel 2: Category distribution  
            if 'categories' in data_sources and not data_sources['categories'].empty:
                cat_data = data_sources['categories']
                fig.add_trace(
                    go.Pie(labels=cat_data.get('category', cat_data.iloc[:, 0]),
                          values=cat_data.get('value', cat_data.iloc[:, 1]),
                          name='Categories'),
                    row=1, col=2
                )
            else:
                # Fallback: Show message
                fig.add_annotation(text="Category data unavailable",
                                 xref="x2", yref="y2", x=0.5, y=0.5,
                                 row=1, col=2)
            
            # Panel 3: Geographic (may fail due to missing geo libraries)
            try:
                if 'geographic' in data_sources and not data_sources['geographic'].empty:
                    geo_data = data_sources['geographic']
                    fig.add_trace(
                        go.Scattergeo(
                            lon=geo_data.get('longitude', [-74, -87, -122]),
                            lat=geo_data.get('latitude', [40.7, 41.9, 37.8]),
                            text=geo_data.get('city', ['NYC', 'Chicago', 'SF']),
                            mode='markers',
                            name='Locations'
                        ),
                        row=2, col=1
                    )
                else:
                    raise ValueError("No geographic data")
            except Exception as geo_error:
                logger.warning(f"Geographic visualization failed: {geo_error}")
                # Fallback to bar chart
                fig.add_trace(
                    go.Bar(x=['East', 'Central', 'West'], 
                          y=[100, 85, 120], name='Regional Sales'),
                    row=2, col=1
                )
            
            # Panel 4: Performance metrics
            if 'performance' in data_sources and not data_sources['performance'].empty:
                perf_data = data_sources['performance']
                fig.add_trace(
                    go.Bar(x=perf_data.get('metric', perf_data.iloc[:, 0]),
                          y=perf_data.get('value', perf_data.iloc[:, 1]),
                          name='Current'),
                    row=2, col=2
                )
                # Add target line if available
                if 'target' in perf_data.columns:
                    fig.add_trace(
                        go.Scatter(x=perf_data['metric'], y=perf_data['target'],
                                  mode='markers', name='Target',
                                  marker=dict(symbol='diamond', size=10)),
                        row=2, col=2, secondary_y=True
                    )
            
            fig.update_layout(
                title="Multi-Panel Dashboard (with Error Recovery)",
                height=800,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Complex dashboard creation failed: {str(e)}")
            # Progressive degradation: create simpler version
            return create_simple_fallback_dashboard(data_sources)
    
    def create_simple_fallback_dashboard(data_sources: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create simplified dashboard when complex version fails."""
        
        fig = go.Figure()
        
        # Find any available data to display
        available_data = None
        data_name = "Unknown"
        
        for name, data in data_sources.items():
            if not data.empty:
                available_data = data
                data_name = name
                break
        
        if available_data is not None:
            # Create simple line plot with first two numeric columns
            numeric_cols = available_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig.add_trace(go.Scatter(
                    x=available_data[numeric_cols[0]],
                    y=available_data[numeric_cols[1]],
                    mode='markers',
                    name=f'{data_name} Data'
                ))
            else:
                # Fallback to bar chart of first column
                fig.add_trace(go.Bar(
                    x=list(range(len(available_data))),
                    y=available_data.iloc[:, 0] if not available_data.empty else [0],
                    name=f'{data_name} Values'
                ))
        else:
            # Ultimate fallback: empty plot with message
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16, color="gray")
            )
        
        fig.update_layout(
            title="Simplified Dashboard (Fallback Mode)",
            template='plotly_white'
        )
        
        logger.info("Created simplified fallback dashboard")
        return fig
    
    # Test with various data scenarios
    print("Testing with complete data...")
    complete_data = {
        'sales': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'amount': np.random.randint(1000, 5000, 100)
        }),
        'categories': pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [25, 35, 20, 20]
        }),
        'geographic': pd.DataFrame({
            'longitude': [-74, -87, -122],
            'latitude': [40.7, 41.9, 37.8],
            'city': ['NYC', 'Chicago', 'SF']
        }),
        'performance': pd.DataFrame({
            'metric': ['Revenue', 'Profit', 'Users'],
            'value': [100, 75, 150],
            'target': [120, 80, 140]
        })
    }
    
    fig_complete = create_complex_dashboard(complete_data)
    if fig_complete:
        print(" Complete dashboard created successfully")
    
    print("\nTesting with partial data...")
    partial_data = {
        'sales': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'amount': np.random.randint(1000, 3000, 50)
        }),
        'categories': pd.DataFrame(),  # Empty DataFrame
        'geographic': pd.DataFrame(),  # Empty DataFrame
        'performance': pd.DataFrame({
            'metric': ['Revenue', 'Profit'],
            'value': [80, 60]
        })
    }
    
    fig_partial = create_complex_dashboard(partial_data)
    if fig_partial:
        print(" Partial dashboard created with fallbacks")
    
    print("\nTesting with minimal data...")
    minimal_data = {
        'sales': pd.DataFrame({'value': [1, 2, 3, 4, 5]}),
        'categories': pd.DataFrame(),
        'geographic': pd.DataFrame(),
        'performance': pd.DataFrame()
    }
    
    fig_minimal = create_complex_dashboard(minimal_data)
    if fig_minimal:
        print(" Minimal dashboard created with maximum fallbacks")
    
    print("\n=== Advanced error recovery completed ===")
    
    return {
        'complete': fig_complete,
        'partial': fig_partial,
        'minimal': fig_minimal
    }


def memory_optimization_example():
    """
    Demonstrate memory optimization techniques for large datasets.
    
    Shows strategies for handling memory constraints and preventing
    browser crashes with large visualizations.
    """
    print("=== Memory Optimization Example ===\n")
    
    handler = PlotlyErrorHandler()
    
    # Simulate memory-intensive scenario
    print("Creating large dataset simulation...")
    
    # Generate progressively larger datasets
    dataset_sizes = [1000, 10000, 50000, 100000]
    memory_results = {}
    
    for size in dataset_sizes:
        print(f"\nTesting with {size:,} data points...")
        
        try:
            # Create large dataset
            large_data = pd.DataFrame({
                'x': np.random.randn(size),
                'y': np.random.randn(size),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
                'size': np.random.randint(1, 100, size),
                'text': [f'Point_{i}' for i in range(size)]
            })
            
            # Test memory-efficient plotting
            start_time = time.time()
            
            @handler.performance_monitor
            def create_memory_efficient_plot(data):
                # Use WebGL for large datasets
                if len(data) > 10000:
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=data['x'],
                        y=data['y'],
                        mode='markers',
                        marker=dict(
                            color=data['size'],
                            colorscale='Viridis',
                            size=3,  # Fixed small size for performance
                            opacity=0.6
                        ),
                        text=None,  # Remove text for memory efficiency
                        hoverinfo='x+y'
                    ))
                else:
                    fig = px.scatter(data, x='x', y='y', color='category', size='size')
                
                return fig
            
            fig = handler.create_robust_figure(create_memory_efficient_plot, large_data)
            
            creation_time = time.time() - start_time
            memory_results[size] = {
                'success': fig is not None,
                'time': creation_time,
                'webgl_used': len(large_data) > 10000
            }
            
            if fig:
                print(f"   Success in {creation_time:.2f}s "
                      f"(WebGL: {memory_results[size]['webgl_used']})")
            else:
                print(f"   Failed after {creation_time:.2f}s")
                
        except MemoryError:
            print(f"   Memory error at {size:,} points")
            memory_results[size] = {'success': False, 'error': 'MemoryError'}
        except Exception as e:
            print(f"   Error: {str(e)}")
            memory_results[size] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n=== Memory optimization results ===")
    for size, result in memory_results.items():
        status = "right" if result['success'] else "wrong"
        print(f"{size:>6,} points: {status} {result}")
    
    return memory_results


def main():
    """
    Main function demonstrating comprehensive error handling and best practices.
    
    Runs through various error scenarios and demonstrates robust
    visualization creation techniques.
    """
    print("=== Comprehensive Plotly Error Handling Examples ===\n")
    
    try:
        # Basic error handling demonstration
        demonstrate_error_handling()
        
        print("\n" + "="*60 + "\n")
        
        # Advanced error recovery
        recovery_results = advanced_error_recovery_example()
        
        print("\n" + "="*60 + "\n")
        
        # Memory optimization
        memory_results = memory_optimization_example()
        
        print("\n" + "="*60 + "\n")
        
        # Production-ready example
        print("=== Production-Ready Plot Example ===\n")
        
        # Create test data with various issues
        test_data = pd.DataFrame({
            'sales': [100, 150, 120, np.inf, 200, 180, 160],
            'profit': [20, 30, np.nan, 40, 45, 35, 32],
            'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East'],
            'quarter': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3']
        })
        
        production_fig = create_production_ready_plot(
            test_data, 'sales', 'profit', 'region', 
            'Sales vs Profit Analysis'
        )
        
        if production_fig:
            print("Production-ready plot created successfully")
            
            # Test export
            handler = PlotlyErrorHandler()
            export_success = handler.safe_export(
                production_fig, 'production_plot.html', 'html'
            )
            print(f"Export successful: {export_success}")
        
        print("\n=== All error handling examples completed successfully ===")
        
        return {
            'recovery_results': recovery_results,
            'memory_results': memory_results,
            'production_figure': production_fig
        }
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()