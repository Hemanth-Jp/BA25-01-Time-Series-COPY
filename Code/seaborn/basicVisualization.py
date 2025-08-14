"""
Basic Statistical Visualization with Seaborn

This script demonstrates fundamental seaborn plotting capabilities
for exploratory data analysis and statistical visualization.


"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_explore_data():
    """
    Load sample dataset and perform basic exploration.
    
    Returns:
        pd.DataFrame: Loaded dataset for analysis
    """
    try:
        # Load built-in dataset
        tips = sns.load_dataset("tips")
        
        print("Dataset Overview:")
        print(f"Shape: {tips.shape}")
        print(f"Columns: {list(tips.columns)}")
        print("\nFirst few rows:")
        print(tips.head())
        
        return tips
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_basic_plots(data):
    """
    Create fundamental statistical visualizations.
    
    Args:
        data (pd.DataFrame): Dataset for visualization
    """
    if data is None:
        return
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Basic Statistical Visualizations with Seaborn', fontsize=16)
    
    # 1. Scatter plot with regression line
    sns.scatterplot(data=data, x="total_bill", y="tip", 
                   hue="time", style="smoker", ax=axes[0,0])
    axes[0,0].set_title("Tip vs Total Bill")
    
    # 2. Box plot for categorical analysis
    sns.boxplot(data=data, x="day", y="total_bill", ax=axes[0,1])
    axes[0,1].set_title("Total Bill Distribution by Day")
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Histogram with density curve
    sns.histplot(data=data, x="total_bill", kde=True, ax=axes[1,0])
    axes[1,0].set_title("Total Bill Distribution")
    
    # 4. Correlation heatmap
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", 
                center=0, ax=axes[1,1])
    axes[1,1].set_title("Correlation Matrix")
    
    plt.tight_layout()
    plt.show()

def demonstrate_categorical_plots(data):
    """
    Showcase categorical data visualization techniques.
    
    Args:
        data (pd.DataFrame): Dataset for visualization
    """
    if data is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Categorical Data Visualization', fontsize=16)
    
    # Count plot
    sns.countplot(data=data, x="day", hue="time", ax=axes[0])
    axes[0].set_title("Frequency by Day and Time")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=data, x="day", y="tip", ax=axes[1])
    axes[1].set_title("Tip Distribution by Day")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Bar plot with error bars
    sns.barplot(data=data, x="day", y="total_bill", 
                estimator=np.mean, ci=95, ax=axes[2])
    axes[2].set_title("Average Bill by Day (95% CI)")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function demonstrating basic seaborn capabilities.
    """
    print("=== Basic Seaborn Visualization Demo ===\n")
    
    # Load and explore data
    dataset = load_and_explore_data()
    
    if dataset is not None:
        print("\n=== Creating Basic Plots ===")
        create_basic_plots(dataset)
        
        print("\n=== Categorical Analysis ===")
        demonstrate_categorical_plots(dataset)
        
        print("\nDemo completed successfully!")
    else:
        print("Unable to proceed without data.")

if __name__ == "__main__":
    main()