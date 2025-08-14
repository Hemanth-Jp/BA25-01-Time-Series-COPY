"""
Advanced Matplotlib Visualization Examples

This module demonstrates sophisticated visualization techniques including:
- Statistical plots and distributions
- Advanced styling and themes
- Complex subplot layouts
- Professional publication-ready figures


Version: 1.0
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_statistical_plots():
    """
    Create comprehensive statistical visualization dashboard.
    
    Demonstrates:
    - Box plots and violin plots
    - Error bars and confidence intervals
    - Distribution overlays
    - Professional styling
    """
    # Generate sample data
    np.random.seed(42)
    groups = ['Group A', 'Group B', 'Group C', 'Group D']
    data = [np.random.normal(0, std, 100) for std in [1, 1.5, 0.5, 2]]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1])
    
    # Box plot
    ax1 = fig.add_subplot(gs[0, 0])
    bp = ax1.boxplot(data, labels=groups, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Values')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2 = fig.add_subplot(gs[0, 1])
    vp = ax2.violinplot(data, positions=range(1, len(groups)+1))
    ax2.set_xticks(range(1, len(groups)+1))
    ax2.set_xticklabels(groups)
    ax2.set_title('Violin Plot Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Error bar plot
    ax3 = fig.add_subplot(gs[1, 0])
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    x_pos = range(len(groups))
    ax3.errorbar(x_pos, means, yerr=stds, fmt='o-', 
                capsize=5, capthick=2, linewidth=2, markersize=8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(groups)
    ax3.set_title('Mean with Error Bars', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Value')
    ax3.grid(True, alpha=0.3)
    
    # Time series plot
    ax4 = fig.add_subplot(gs[1, 1])
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    values = np.cumsum(np.random.randn(30)) + 100
    ax4.plot(dates, values, 'b-', linewidth=2, marker='o', markersize=4)
    ax4.set_title('Time Series Data', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cumulative Value')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Distribution histogram
    ax5 = fig.add_subplot(gs[:, 2])
    combined_data = np.concatenate(data)
    ax5.hist(combined_data, bins=30, orientation='horizontal', 
             alpha=0.7, color='skyblue', density=True)
    ax5.set_title('Combined Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Density')
    
    plt.tight_layout()
    plt.show()

def create_heatmap_correlation():
    """
    Create correlation heatmap with advanced styling.
    
    Demonstrates:
    - Correlation matrix visualization
    - Custom colormaps
    - Annotations
    - Professional styling
    """
    # Generate sample correlation data
    np.random.seed(42)
    variables = ['Variable A', 'Variable B', 'Variable C', 'Variable D', 'Variable E']
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Set ticks and labels
    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_yticklabels(variables)
    
    # Add text annotations
    for i in range(len(variables)):
        for j in range(len(variables)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Advanced Matplotlib Visualization Examples...")
    
    try:
        create_statistical_plots()
        create_heatmap_correlation()
        print("Advanced visualization examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()