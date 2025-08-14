"""
Advanced Statistical Analysis with Seaborn

This script demonstrates sophisticated seaborn capabilities including
multi-plot grids, statistical functions, and complex visualizations.


"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def load_multiple_datasets():
    """
    Load multiple datasets for comprehensive analysis.
    
    Returns:
        dict: Dictionary containing multiple datasets
    """
    datasets = {}
    try:
        datasets['tips'] = sns.load_dataset("tips")
        datasets['flights'] = sns.load_dataset("flights")
        datasets['iris'] = sns.load_dataset("iris")
        print("Loaded datasets:", list(datasets.keys()))
        return datasets
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}

def create_regression_analysis(data):
    """
    Perform comprehensive regression analysis visualization.
    
    Args:
        data (pd.DataFrame): Tips dataset for regression analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Advanced Regression Analysis', fontsize=16)
    
    # Linear regression with confidence intervals
    sns.regplot(data=data, x="total_bill", y="tip", ax=axes[0,0])
    axes[0,0].set_title("Linear Regression with CI")
    
    # Regression by categorical variable
    sns.lmplot(data=data, x="total_bill", y="tip", hue="smoker", 
               col="time", height=4, aspect=0.8)
    plt.suptitle("Regression by Smoker Status and Time")
    
    # Residual plot for model diagnostics
    sns.residplot(data=data, x="total_bill", y="tip", ax=axes[0,1])
    axes[0,1].set_title("Residual Plot")
    
    # Joint plot with marginal distributions
    g = sns.jointplot(data=data, x="total_bill", y="tip", kind="reg")
    g.fig.suptitle("Joint Distribution with Regression")
    
    plt.show()

def create_facet_grids(datasets):
    """
    Demonstrate FacetGrid for multi-dimensional analysis.
    
    Args:
        datasets (dict): Dictionary of datasets
    """
    tips = datasets.get('tips')
    if tips is None:
        return
    
    # Create FacetGrid for multiple comparisons
    g = sns.FacetGrid(tips, col="time", row="smoker", 
                      margin_titles=True, height=4)
    g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
    g.add_legend()
    g.fig.suptitle("Multi-dimensional Facet Analysis")
    plt.show()
    
    # PairGrid for comprehensive variable relationships
    g = sns.PairGrid(tips, hue="time")
    g.map_diag(sns.histplot)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.regplot)
    g.add_legend()
    g.fig.suptitle("Pairwise Relationships Grid")
    plt.show()

def time_series_analysis(datasets):
    """
    Analyze time series data with advanced visualizations.
    
    Args:
        datasets (dict): Dictionary containing flights dataset
    """
    flights = datasets.get('flights')
    if flights is None:
        return
    
    # Pivot for heatmap - FIXED: Use keyword arguments
    flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Time Series Analysis', fontsize=16)
    
    # Heatmap of passenger data
    sns.heatmap(flights_pivot, annot=True, fmt="d", 
                cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Passenger Counts Heatmap")
    
    # Line plot with error estimation
    sns.lineplot(data=flights, x="year", y="passengers", 
                 estimator=np.mean, ci=95, ax=axes[1])
    axes[1].set_title("Passenger Trend with Confidence Interval")
    
    plt.tight_layout()
    plt.show()

def distribution_analysis(datasets):
    """
    Advanced distribution analysis and comparison.
    
    Args:
        datasets (dict): Dictionary of datasets
    """
    iris = datasets.get('iris')
    if iris is None:
        return
    
    # Multiple distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution Analysis', fontsize=16)
    
    # KDE plots by category
    for species in iris['species'].unique():
        subset = iris[iris['species'] == species]
        sns.kdeplot(data=subset, x="sepal_length", 
                   label=species, ax=axes[0,0])
    axes[0,0].set_title("Sepal Length Distribution by Species")
    axes[0,0].legend()
    
    # Strip plot with jitter
    sns.stripplot(data=iris, x="species", y="petal_width", 
                  size=8, jitter=True, ax=axes[0,1])
    axes[0,1].set_title("Petal Width by Species")
    
    # Swarm plot for detailed distribution
    sns.swarmplot(data=iris, x="species", y="petal_length", ax=axes[1,0])
    axes[1,0].set_title("Petal Length Distribution (Swarm)")
    
    # Box plot with statistical annotations
    sns.boxplot(data=iris, x="species", y="sepal_width", ax=axes[1,1])
    axes[1,1].set_title("Sepal Width with Outliers")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Execute advanced seaborn analysis demonstrations.
    """
    print("=== Advanced Seaborn Analysis Demo ===\n")
    
    # Load datasets
    datasets = load_multiple_datasets()
    
    if not datasets:
        print("Unable to proceed without datasets.")
        return
    
    print("\n=== Regression Analysis ===")
    create_regression_analysis(datasets['tips'])
    
    print("\n=== Facet Grid Analysis ===")
    create_facet_grids(datasets)
    
    print("\n=== Time Series Analysis ===")
    time_series_analysis(datasets)
    
    print("\n=== Distribution Analysis ===")
    distribution_analysis(datasets)
    
    print("Advanced analysis completed!")

if __name__ == "__main__":
    main()