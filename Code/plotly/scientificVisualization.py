"""
Advanced Scientific Visualization with Plotly

This module demonstrates creating publication-quality scientific visualizations:
- 3D surface and scatter plots
- Statistical distributions and error analysis
- Multi-panel scientific figures
- Animation and interactive elements for research


Version: 1.0
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_scientific_data():
    """
    Generate realistic scientific datasets for visualization examples.
    
    Returns:
        dict: Dictionary containing various scientific datasets
    """
    np.random.seed(42)
    
    # 3D surface data (e.g., potential energy surface)
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))
    
    # Experimental data with error bars
    n_experiments = 50
    temperatures = np.linspace(200, 400, n_experiments)
    # Simulated Arrhenius relationship with noise
    rate_constants = 1e6 * np.exp(-8000 / temperatures) * (1 + np.random.normal(0, 0.1, n_experiments))
    errors = rate_constants * 0.15  # 15% experimental error
    
    # Statistical distribution data
    sample_sizes = [30, 100, 500, 1000]
    distributions = {}
    for n in sample_sizes:
        distributions[f'n_{n}'] = np.random.normal(100, 15, n)
    
    # Time series data (e.g., oscillation decay)
    time = np.linspace(0, 10, 1000)
    amplitude = np.exp(-0.2 * time) * np.sin(2 * np.pi * time)
    noise = np.random.normal(0, 0.05, 1000)
    signal = amplitude + noise
    
    # Correlation matrix data
    variables = ['Temp', 'Pressure', 'Density', 'Viscosity', 'pH']
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return {
        'surface': {'X': X, 'Y': Y, 'Z': Z},
        'kinetics': {
            'temperature': temperatures,
            'rate_constant': rate_constants,
            'error': errors
        },
        'distributions': distributions,
        'time_series': {'time': time, 'signal': signal, 'amplitude': amplitude},
        'correlation': {'matrix': correlation_matrix, 'variables': variables}
    }


def create_3d_surface_plot(data):
    """
    Create a 3D surface plot for scientific data visualization.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: 3D surface plot figure
    """
    surface_data = data['surface']
    
    fig = go.Figure(data=[
        go.Surface(
            x=surface_data['X'],
            y=surface_data['Y'],
            z=surface_data['Z'],
            colorscale='Viridis',
            opacity=0.9,
            showscale=True,
            colorbar=dict(
                title="Amplitude",
                titleside="right"
            )
        )
    ])
    
    # Add contour projections
    fig.add_trace(go.Contour(
        x=surface_data['X'][0],
        y=surface_data['Y'][:, 0],
        z=surface_data['Z'],
        colorscale='Viridis',
        opacity=0.3,
        showscale=False,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
    ))
    
    fig.update_layout(
        title='3D Potential Energy Surface',
        scene=dict(
            xaxis_title='X Coordinate (Angstrom)',
            yaxis_title='Y Coordinate (Angstrom)',
            zaxis_title='Energy (eV)',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.6)
            )
        ),
        font=dict(size=12),
        template='plotly_white'
    )
    
    return fig


def create_error_bar_analysis(data):
    """
    Create scientific plot with error bars and curve fitting.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Error bar analysis figure
    """
    kinetics = data['kinetics']
    
    fig = go.Figure()
    
    # Experimental data with error bars
    fig.add_trace(go.Scatter(
        x=1000/kinetics['temperature'],  # Arrhenius plot (1/T)
        y=np.log(kinetics['rate_constant']),
        error_y=dict(
            type='data',
            array=kinetics['error']/kinetics['rate_constant'],  # Relative error
            visible=True,
            color='red',
            thickness=1.5
        ),
        mode='markers',
        name='Experimental Data',
        marker=dict(
            size=8,
            color='blue',
            symbol='circle',
            line=dict(width=1, color='darkblue')
        )
    ))
    
    # Theoretical fit line
    x_fit = np.linspace(1000/kinetics['temperature'].max(), 1000/kinetics['temperature'].min(), 100)
    # Linear fit for Arrhenius equation: ln(k) = ln(A) - Ea/(RT)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        1000/kinetics['temperature'], np.log(kinetics['rate_constant'])
    )
    y_fit = slope * x_fit + intercept
    
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name=f'Linear Fit (R^2 = {r_value**2:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add confidence interval
    y_err = std_err * np.sqrt(1 + 1/len(kinetics['temperature']) + 
                             (x_fit - np.mean(1000/kinetics['temperature']))**2 / 
                             np.sum((1000/kinetics['temperature'] - np.mean(1000/kinetics['temperature']))**2))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fit, x_fit[::-1]]),
        y=np.concatenate([y_fit + 1.96*y_err, (y_fit - 1.96*y_err)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Arrhenius Plot with Error Analysis',
        xaxis_title='1000/T (K^-1)',
        yaxis_title='ln(k)',
        template='plotly_white',
        legend=dict(x=0.02, y=0.98),
        font=dict(size=12)
    )
    
    return fig


def create_statistical_distributions(data):
    """
    Create statistical distribution comparison plots.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Statistical distributions figure
    """
    distributions = data['distributions']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Sample Size: {n}' for n in [30, 100, 500, 1000]],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (key, sample) in enumerate(distributions.items()):
        row, col = positions[i]
        color = colors[i]
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=20,
                opacity=0.7,
                name=f'Histogram {key}',
                marker_color=color,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Normal distribution overlay
        x_norm = np.linspace(sample.min(), sample.max(), 100)
        y_norm = stats.norm.pdf(x_norm, sample.mean(), sample.std())
        # Scale to match histogram
        y_norm = y_norm * len(sample) * (sample.max() - sample.min()) / 20
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name=f'Normal Fit {key}',
                line=dict(color='black', width=2),
                showlegend=False
            ),
            row=row, col=col, secondary_y=True
        )
    
    fig.update_layout(
        title='Central Limit Theorem Demonstration',
        template='plotly_white',
        height=600,
        font=dict(size=10)
    )
    
    return fig


def create_time_series_analysis(data):
    """
    Create time series analysis with signal processing.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Time series analysis figure
    """
    ts_data = data['time_series']
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Original Signal', 'Frequency Domain', 'Phase Plot'],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Time domain plot
    fig.add_trace(
        go.Scatter(
            x=ts_data['time'],
            y=ts_data['signal'],
            mode='lines',
            name='Measured Signal',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=ts_data['time'],
            y=ts_data['amplitude'],
            mode='lines',
            name='True Amplitude',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Frequency domain (FFT)
    fft = np.fft.fft(ts_data['signal'])
    freqs = np.fft.fftfreq(len(ts_data['signal']), ts_data['time'][1] - ts_data['time'][0])
    
    fig.add_trace(
        go.Scatter(
            x=freqs[:len(freqs)//2],
            y=np.abs(fft)[:len(fft)//2],
            mode='lines',
            name='Power Spectrum',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Phase plot (derivative vs signal)
    dt = ts_data['time'][1] - ts_data['time'][0]
    derivative = np.gradient(ts_data['signal'], dt)
    
    fig.add_trace(
        go.Scatter(
            x=ts_data['signal'],
            y=derivative,
            mode='markers',
            name='Phase Space',
            marker=dict(
                size=3,
                color=ts_data['time'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time", y=0.2, len=0.3)
            )
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title='Time Series Analysis: Damped Oscillation',
        template='plotly_white',
        height=900,
        font=dict(size=10)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Power", row=2, col=1)
    fig.update_xaxes(title_text="Signal", row=3, col=1)
    fig.update_yaxes(title_text="dSignal/dt", row=3, col=1)
    
    return fig


def create_correlation_heatmap(data):
    """
    Create correlation matrix heatmap for multivariate analysis.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap figure
    """
    corr_data = data['correlation']
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data['matrix'],
        x=corr_data['variables'],
        y=corr_data['variables'],
        colorscale='RdBu',
        zmid=0,
        text=corr_data['matrix'],
        texttemplate="%{text:.2f}",
        textfont={"size": 12},
        colorbar=dict(
            title="Correlation Coefficient",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        title='Correlation Matrix: Process Variables',
        template='plotly_white',
        font=dict(size=12),
        width=500,
        height=500
    )
    
    return fig


def create_animated_visualization(data):
    """
    Create animated visualization for temporal data.
    
    Args:
        data (dict): Scientific data dictionary
    
    Returns:
        plotly.graph_objects.Figure: Animated visualization figure
    """
    # Create animation frames for 3D surface evolution
    surface_data = data['surface']
    
    frames = []
    for t in np.linspace(0, 2*np.pi, 20):
        # Animate the surface with time-dependent phase
        Z_animated = surface_data['Z'] * np.cos(t)
        
        frame = go.Frame(
            data=[go.Surface(
                x=surface_data['X'],
                y=surface_data['Y'],
                z=Z_animated,
                colorscale='Viridis',
                cmin=-1,
                cmax=1,
                showscale=True
            )],
            name=str(t)
        )
        frames.append(frame)
    
    fig = go.Figure(
        data=[go.Surface(
            x=surface_data['X'],
            y=surface_data['Y'],
            z=surface_data['Z'],
            colorscale='Viridis',
            showscale=True
        )],
        frames=frames
    )
    
    fig.update_layout(
        title='Animated Wave Propagation',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Amplitude',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"frame": {"duration": 500, "redraw": True},
                           "fromcurrent": True, "transition": {"duration": 300}}],
                    label="Play",
                    method="animate"
                ),
                dict(
                    args=[{"frame": {"duration": 0, "redraw": True},
                           "mode": "immediate", "transition": {"duration": 0}}],
                    label="Pause",
                    method="animate"
                )
            ]),
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.011,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        template='plotly_white'
    )
    
    return fig


def main():
    """
    Main function demonstrating advanced scientific visualization capabilities.
    
    Creates comprehensive scientific plots including 3D surfaces, error analysis,
    statistical distributions, time series analysis, and animations.
    """
    print("=== Advanced Scientific Visualization Examples ===\n")
    
    try:
        # Generate scientific data
        print("Generating scientific datasets...")
        data = generate_scientific_data()
        print("Scientific data generated successfully\n")
        
        # Create 3D surface plot
        print("Creating 3D surface visualization...")
        surface_fig = create_3d_surface_plot(data)
        surface_fig.show()
        print("3D surface plot created\n")
        
        # Create error bar analysis
        print("Creating error bar analysis...")
        error_fig = create_error_bar_analysis(data)
        error_fig.show()
        print("Error analysis plot created\n")
        
        # Create statistical distributions
        print("Creating statistical distribution analysis...")
        stats_fig = create_statistical_distributions(data)
        stats_fig.show()
        print("Statistical distributions created\n")
        
        # Create time series analysis
        print("Creating time series analysis...")
        ts_fig = create_time_series_analysis(data)
        ts_fig.show()
        print("Time series analysis created\n")
        
        # Create correlation heatmap
        print("Creating correlation matrix...")
        corr_fig = create_correlation_heatmap(data)
        corr_fig.show()
        print("Correlation heatmap created\n")
        
        # Create animated visualization
        print("Creating animated visualization...")
        anim_fig = create_animated_visualization(data)
        anim_fig.show()
        print("Animated visualization created\n")
        
        print("=== All scientific visualizations completed successfully ===")
        
        return {
            'surface': surface_fig,
            'error_analysis': error_fig,
            'statistics': stats_fig,
            'time_series': ts_fig,
            'correlation': corr_fig,
            'animation': anim_fig
        }
        
    except Exception as e:
        print(f"Error in scientific visualization creation: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()