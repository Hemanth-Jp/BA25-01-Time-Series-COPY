"""
Publication-Quality Visualizations with Seaborn (Fixed Version)

This script demonstrates creating publication-ready statistical graphics
with professional styling. Fixed to handle read-only file systems.


"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
import os
import tempfile

def configure_publication_style():
    """
    Configure matplotlib and seaborn for publication-quality output.
    """
    # Set publication-ready parameters
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    
    # Set seaborn style for publications
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # Custom color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    sns.set_palette(colors)

def create_publication_figure():
    """
    Create a comprehensive publication-quality figure.
    
    Returns:
        tuple: Figure and axes objects
    """
    # Load data
    tips = sns.load_dataset("tips")
    
    # Create figure with specific size for publication
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Statistical Analysis of Restaurant Tips', 
                 fontsize=16, fontweight='bold')
    
    # Panel A: Regression with confidence interval
    sns.regplot(data=tips, x="total_bill", y="tip", 
                scatter_kws={'alpha': 0.6}, ax=axes[0,0])
    axes[0,0].set_title('A) Tip vs Total Bill Relationship')
    axes[0,0].set_xlabel('Total Bill ($)')
    axes[0,0].set_ylabel('Tip ($)')
    
    # Panel B: Box plot with significance testing
    sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0,1])
    axes[0,1].set_title('B) Bill Distribution by Day')
    axes[0,1].set_xlabel('Day of Week')
    axes[0,1].set_ylabel('Total Bill ($)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Panel C: Violin plot with inner quartiles
    sns.violinplot(data=tips, x="time", y="tip", hue="smoker", 
                   inner="quartile", ax=axes[1,0])
    axes[1,0].set_title('C) Tip Distribution by Time and Smoking')
    axes[1,0].set_xlabel('Meal Time')
    axes[1,0].set_ylabel('Tip ($)')
    
    # Panel D: Correlation heatmap
    numeric_data = tips.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    sns.heatmap(correlation, annot=True, cmap="RdBu_r", center=0,
                square=True, ax=axes[1,1])
    axes[1,1].set_title('D) Variable Correlations')
    
    plt.tight_layout()
    return fig, axes

def create_statistical_summary():
    """
    Create a statistical summary visualization.
    """
    # Load data
    iris = sns.load_dataset("iris")
    
    # Create figure for pairplot alternative
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a more compact visualization instead of PairGrid
    # which can be problematic in some environments
    g = sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", 
                       hue="species", style="species", s=100)
    plt.title('Iris Dataset: Sepal Dimensions by Species', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def get_writable_directory():
    """
    Find a writable directory for saving files.
    
    Returns:
        str: Path to writable directory
    """
    # Try common writable directories
    writable_dirs = [
        os.path.expanduser("~"),  # Home directory
        "/tmp",                   # Temporary directory
        tempfile.gettempdir(),    # System temp directory
        "."                       # Current directory (last resort)
    ]
    
    for directory in writable_dirs:
        try:
            test_file = os.path.join(directory, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return directory
        except (OSError, PermissionError):
            continue
    
    return None

def save_publication_figures():
    """
    Save figures in multiple publication-ready formats if possible.
    """
    print("Creating publication figures...")
    
    # Configure style
    configure_publication_style()
    
    # Create main figure
    fig, axes = create_publication_figure()
    
    # Try to find writable directory
    save_dir = get_writable_directory()
    
    if save_dir:
        print(f"Saving files to: {save_dir}")
        
        # Save in multiple formats
        formats = {
            'png': {'dpi': 300, 'bbox_inches': 'tight'},
            'pdf': {'bbox_inches': 'tight'},
            'svg': {'bbox_inches': 'tight'}
        }
        
        saved_files = []
        for fmt, kwargs in formats.items():
            try:
                filename = os.path.join(save_dir, f"restaurant_analysis.{fmt}")
                fig.savefig(filename, **kwargs)
                saved_files.append(filename)
                print(f"✓ Saved: {filename}")
            except Exception as e:
                print(f"✗ Failed to save {fmt}: {e}")
        
        if saved_files:
            print(f"\nSuccessfully saved {len(saved_files)} files!")
        else:
            print("\nCould not save any files, but displaying visualization...")
    else:
        print("No writable directory found. Displaying visualization only...")
    
    plt.show()

def demonstrate_styling_options():
    """
    Demonstrate various styling options for publications.
    """
    tips = sns.load_dataset("tips")
    
    # Different style contexts
    contexts = ['paper', 'notebook', 'talk', 'poster']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seaborn Context Comparison for Publications', fontsize=16)
    
    for i, context in enumerate(contexts):
        row, col = i // 2, i % 2
        
        # Temporarily set context for this subplot
        with sns.plotting_context(context):
            sns.scatterplot(data=tips, x="total_bill", y="tip", 
                           hue="time", ax=axes[row, col])
            axes[row, col].set_title(f'{context.title()} Context')
            axes[row, col].set_xlabel('Total Bill ($)')
            axes[row, col].set_ylabel('Tip ($)')
    
    plt.tight_layout()
    plt.show()

def create_advanced_publication_plot():
    """
    Create an advanced publication-quality plot with multiple elements.
    """
    # Load data
    flights = sns.load_dataset("flights")
    
    # Create a comprehensive time series analysis
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Airline Passenger Analysis (1949-1960)', 
                 fontsize=16, fontweight='bold')
    
    # Top panel: Time series with trend
    pivot_flights = flights.pivot(index="month", columns="year", values="passengers")
    sns.lineplot(data=flights, x="year", y="passengers", 
                 estimator="mean", ci=95, ax=axes[0])
    axes[0].set_title('A) Average Monthly Passengers Over Time')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of Passengers (thousands)')
    axes[0].grid(True, alpha=0.3)
    
    # Bottom panel: Seasonal patterns
    sns.boxplot(data=flights, x="month", y="passengers", ax=axes[1])
    axes[1].set_title('B) Seasonal Passenger Distribution')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Number of Passengers (thousands)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Execute publication-quality visualization demonstrations.
    """
    print("=== Publication-Quality Seaborn Visualizations ===\n")
    
    print("1. Configuring publication style...")
    configure_publication_style()
    
    print("\n2. Creating and displaying main publication figure...")
    save_publication_figures()
    
    print("\n3. Creating statistical summary...")
    create_statistical_summary()
    
    print("\n4. Demonstrating styling options...")
    demonstrate_styling_options()
    
    print("\n5. Creating advanced publication plot...")
    create_advanced_publication_plot()
    
    print("\n✓ Publication demo completed!")
    print("\nTips for publication-ready plots:")
    print("- Use high DPI (300+) for print quality")
    print("- Choose serif fonts for academic publications")
    print("- Ensure sufficient contrast and readability")
    print("- Save in vector formats (PDF, SVG) when possible")
    print("- Always include clear titles and axis labels")

if __name__ == "__main__":
    main()