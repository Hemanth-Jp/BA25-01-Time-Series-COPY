"""
Custom Styling and Themes with Seaborn

This script demonstrates advanced styling techniques including
custom color palettes, themes, and branded visualizations.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_custom_palettes():
    """
    Create and demonstrate custom color palettes.
    """
    # Corporate brand colors
    brand_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Scientific publication palette
    science_colors = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC"]
    
    # Diverging palette for correlations
    diverging_colors = ["#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"]
    
    return {
        'brand': brand_colors,
        'science': science_colors,
        'diverging': diverging_colors
    }

def demonstrate_palette_usage():
    """
    Show different palette applications.
    """
    tips = sns.load_dataset("tips")
    palettes = create_custom_palettes()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Custom Color Palette Demonstrations', fontsize=16)
    
    # Brand palette
    sns.set_palette(palettes['brand'])
    sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0])
    axes[0].set_title('Brand Colors')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Science palette
    sns.set_palette(palettes['science'])
    sns.violinplot(data=tips, x="time", y="tip", hue="smoker", ax=axes[1])
    axes[1].set_title('Scientific Publication Colors')
    
    # Diverging palette for heatmap
    correlation = tips.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, cmap="RdBu_r", center=0, 
                annot=True, ax=axes[2])
    axes[2].set_title('Diverging Correlation Colors')
    
    plt.tight_layout()
    plt.show()

def create_themed_visualization():
    """
    Create a fully themed visualization with consistent styling.
    """
    # Load data
    flights = sns.load_dataset("flights")
    
    # Set dark theme
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    
    # Custom color map
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom", ["#000428", "#004e92", "#009ffd", "#00d2ff"]
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#2E2E2E')
    fig.suptitle('Dark Theme Aviation Data Analysis', 
                 color='white', fontsize=16, fontweight='bold')
    
    # Pivot data for heatmap - FIXED: using keyword arguments
    flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
    
    # Heatmap with custom colormap
    sns.heatmap(flights_pivot, cmap=custom_cmap, 
                cbar_kws={'label': 'Passengers (thousands)'}, ax=axes[0,0])
    axes[0,0].set_title('Passenger Heatmap', color='white')
    
    # Line plot with glow effect
    sns.lineplot(data=flights, x="year", y="passengers", 
                 linewidth=3, color='#00d2ff', ax=axes[0,1])
    axes[0,1].set_title('Passenger Trends', color='white')
    axes[0,1].grid(True, alpha=0.3)
    
    # Bar plot with gradient colors
    yearly_avg = flights.groupby('year')['passengers'].mean()
    bars = axes[1,0].bar(yearly_avg.index, yearly_avg.values, 
                        color='#009ffd', alpha=0.8)
    axes[1,0].set_title('Average Annual Passengers', color='white')
    axes[1,0].set_xlabel('Year', color='white')
    axes[1,0].set_ylabel('Passengers', color='white')
    
    # Distribution with custom styling
    sns.histplot(data=flights, x="passengers", kde=True, 
                 color='#00d2ff', alpha=0.7, ax=axes[1,1])
    axes[1,1].set_title('Passenger Distribution', color='white')
    
    # Style all axes
    for ax in axes.flat:
        ax.set_facecolor('#2E2E2E')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    plt.show()

def create_minimalist_style():
    """
    Demonstrate minimalist publication style.
    """
    # Reset to clean style
    sns.set_style("white")
    sns.despine()
    
    tips = sns.load_dataset("tips")
    
    # Minimalist color scheme
    minimal_colors = ["#2E2E2E", "#808080", "#CCCCCC"]
    sns.set_palette(minimal_colors)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Clean scatter plot
    sns.scatterplot(data=tips, x="total_bill", y="tip", 
                   hue="time", s=60, alpha=0.8, ax=ax)
    
    # Minimal styling
    ax.set_title('Minimalist Design: Tip Analysis', 
                 fontsize=16, fontweight='normal', pad=20)
    ax.set_xlabel('Total Bill ($)', fontsize=12)
    ax.set_ylabel('Tip ($)', fontsize=12)
    
    # Remove top and right spines
    sns.despine(top=True, right=True)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Time', 
             frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Execute custom styling demonstrations.
    """
    print("=== Custom Styling and Themes Demo ===\n")
    
    print("Creating custom color palettes...")
    palettes = create_custom_palettes()
    print(f"Created palettes: {list(palettes.keys())}")
    
    print("\nDemonstrating palette usage...")
    demonstrate_palette_usage()
    
    print("\nCreating dark themed visualization...")
    create_themed_visualization()
    
    print("\nDemonstrating minimalist style...")
    create_minimalist_style()
    
    print("\nCustom styling demo completed!")

if __name__ == "__main__":
    main()