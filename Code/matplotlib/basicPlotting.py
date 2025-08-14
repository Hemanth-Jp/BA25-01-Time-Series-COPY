"""
Basic Matplotlib Plotting Examples

This module demonstrates fundamental plotting capabilities including:
- Line plots and scatter plots
- Basic customization and styling
- Multiple data series visualization
- Simple subplot layouts

Version: 1.0
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_basic_plots():
    """
    Create basic line and scatter plots with customization.
    
    Demonstrates:
    - Line plot creation
    - Scatter plot creation
    - Basic styling options
    - Legend and labels
    """
    # Generate sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-x/10)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Line plots with different styles
    ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    ax.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    ax.plot(x, y3, 'g:', linewidth=2, label='damped sin(x)')
    
    # Customization
    ax.set_xlabel('X values', fontsize=12)
    ax.set_ylabel('Y values', fontsize=12)
    ax.set_title('Basic Line Plots', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()

def create_scatter_plot():
    """
    Create scatter plot with different marker styles and colors.
    
    Demonstrates:
    - Scatter plot creation
    - Color mapping
    - Marker customization
    - Size variation
    """
    # Generate random data
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    colors = np.random.rand(n)
    sizes = 1000 * np.random.rand(n)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Color Scale')
    
    # Customization
    ax.set_xlabel('X values', fontsize=12)
    ax.set_ylabel('Y values', fontsize=12)
    ax.set_title('Scatter Plot with Color and Size Mapping', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_subplot_example():
    """
    Create multiple subplots in a single figure.
    
    Demonstrates:
    - Subplot creation
    - Different plot types in subplots
    - Shared axes
    - Subplot customization
    """
    # Generate data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Line plot
    ax1.plot(x, y1, 'b-', linewidth=2)
    ax1.set_title('Sine Function')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cosine plot
    ax2.plot(x, y2, 'r-', linewidth=2)
    ax2.set_title('Cosine Function')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    data = np.random.normal(0, 1, 1000)
    ax3.hist(data, bins=30, alpha=0.7, color='green')
    ax3.set_title('Random Normal Distribution')
    ax3.set_ylabel('Frequency')
    
    # Plot 4: Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    ax4.bar(categories, values, color='orange', alpha=0.7)
    ax4.set_title('Bar Chart')
    ax4.set_ylabel('Values')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Basic Matplotlib Examples...")
    
    try:
        create_basic_plots()
        create_scatter_plot()
        create_subplot_example()
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()