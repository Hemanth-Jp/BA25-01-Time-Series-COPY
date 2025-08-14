"""
Linear Regression Analysis with Statsmodels

This module demonstrates comprehensive linear regression analysis using statsmodels,
including model fitting, assumption checking, and result interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera  # Fixed import
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=200, random_state=42):
    """
    Generate sample dataset for regression analysis.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate predictors
    age = np.random.normal(35, 10, n_samples)
    experience = age - 22 + np.random.normal(0, 2, n_samples)
    education = np.random.choice([12, 14, 16, 18, 20], n_samples, 
                                p=[0.2, 0.3, 0.3, 0.15, 0.05])
    
    # Generate response variable with some noise
    salary = (2000 + 500 * experience + 1000 * education + 
              100 * age + np.random.normal(0, 5000, n_samples))
    
    return pd.DataFrame({
        'salary': salary,
        'age': age,
        'experience': experience,
        'education': education
    })

def perform_linear_regression(data):
    """
    Perform comprehensive linear regression analysis.
    
    Args:
        data (pandas.DataFrame): Input dataset
        
    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Fitted model
    """
    print("=== Linear Regression Analysis ===")
    
    # Formula-based approach
    formula = 'salary ~ age + experience + education'
    model = smf.ols(formula, data=data)
    results = model.fit()
    
    # Display comprehensive results
    print("\nModel Summary:")
    print(results.summary())
    
    # Extract key statistics
    print(f"\nKey Statistics:")
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.4f}")
    print(f"F-statistic p-value: {results.f_pvalue:.4e}")
    
    return results

def check_assumptions(results, data):
    """
    Check linear regression assumptions.
    
    Args:
        results: Fitted regression results
        data (pandas.DataFrame): Original dataset
    """
    print("\n=== Assumption Checking ===")
    
    # 1. Linearity check (residuals vs fitted)
    fitted_values = results.fittedvalues
    residuals = results.resid
    
    # 2. Homoscedasticity test
    bp_test = het_breuschpagan(residuals, results.model.exog)
    print(f"\nBreusch-Pagan Test for Homoscedasticity:")
    print(f"Test statistic: {bp_test[0]:.4f}")
    print(f"p-value: {bp_test[1]:.4f}")
    print(f"Result: {'Homoscedastic' if bp_test[1] > 0.05 else 'Heteroscedastic'}")
    
    # 3. Normality of residuals
    jb_test = jarque_bera(residuals)
    print(f"\nJarque-Bera Test for Normality:")
    print(f"Test statistic: {jb_test[0]:.4f}")
    print(f"p-value: {jb_test[1]:.4f}")
    print(f"Result: {'Normal' if jb_test[1] > 0.05 else 'Non-normal'}")
    
    # 4. Multicollinearity check
    print(f"\nVariance Inflation Factors:")
    X = sm.add_constant(data[['age', 'experience', 'education']])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    print(vif_data)

def create_diagnostic_plots(results):
    """
    Create diagnostic plots for regression analysis.
    
    Args:
        results: Fitted regression results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(results.fittedvalues, results.resid, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot for normality
    sm.qqplot(results.resid, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Scale-Location plot
    standardized_resid = np.sqrt(np.abs(results.resid_pearson))
    axes[1, 0].scatter(results.fittedvalues, standardized_resid, alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    axes[1, 0].set_title('Scale-Location Plot')
    
    # Histogram of residuals
    axes[1, 1].hist(results.resid, bins=20, density=True, alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate linear regression analysis."""
    # Generate sample data
    print("Generating sample dataset...")
    data = generate_sample_data()
    
    print("\nDataset Overview:")
    print(data.describe())
    
    # Perform regression analysis
    results = perform_linear_regression(data)
    
    # Check assumptions
    check_assumptions(results, data)
    
    # Create diagnostic plots
    create_diagnostic_plots(results)
    
    # Confidence intervals
    print("\n=== Confidence Intervals ===")
    conf_int = results.conf_int()
    conf_int.columns = ['Lower 95%', 'Upper 95%']
    print(conf_int)
    
    # Predictions
    print("\n=== Sample Predictions ===")
    new_data = pd.DataFrame({
        'age': [30, 40, 50],
        'experience': [5, 15, 25],
        'education': [16, 18, 20]
    })
    
    predictions = results.predict(new_data)
    pred_intervals = results.get_prediction(new_data).conf_int()
    
    for i, (_, row) in enumerate(new_data.iterrows()):
        print(f"Age: {row['age']}, Experience: {row['experience']}, "
              f"Education: {row['education']}")
        print(f"Predicted Salary: ${predictions[i]:,.2f}")
        print(f"95% CI: [${pred_intervals[i, 0]:,.2f}, "
              f"${pred_intervals[i, 1]:,.2f}]\n")

if __name__ == "__main__":
    main()