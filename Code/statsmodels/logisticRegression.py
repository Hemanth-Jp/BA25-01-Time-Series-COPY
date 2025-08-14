"""
Logistic Regression Analysis with Statsmodels

This module demonstrates binary classification using logistic regression
with comprehensive statistical analysis and model evaluation.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def generate_binary_data(n_samples=500, random_state=42):
    """
    Generate sample dataset for binary classification.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate features
    age = np.random.normal(45, 15, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    
    # Generate binary outcome with logistic relationship
    linear_combination = (-3 + 0.05 * age + 0.00003 * income + 0.8 * education + 
                         np.random.normal(0, 0.5, n_samples))
    probabilities = 1 / (1 + np.exp(-linear_combination))
    purchased = np.random.binomial(1, probabilities)
    
    return pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'purchased': purchased
    })

def perform_logistic_regression(data):
    """
    Perform comprehensive logistic regression analysis.
    
    Args:
        data (pandas.DataFrame): Input dataset
        
    Returns:
        statsmodels.discrete.discrete_model.BinaryResultsWrapper: Fitted model
    """
    print("=== Logistic Regression Analysis ===")
    
    # Formula-based approach
    formula = 'purchased ~ age + income + education'
    model = smf.logit(formula, data=data)
    results = model.fit()
    
    # Display comprehensive results
    print("\nModel Summary:")
    print(results.summary())
    
    # Extract key statistics
    print(f"\nKey Statistics:")
    print(f"Log-Likelihood: {results.llf:.4f}")
    print(f"AIC: {results.aic:.4f}")
    print(f"BIC: {results.bic:.4f}")
    print(f"Pseudo R-squared: {results.prsquared:.4f}")
    
    return results

def calculate_odds_ratios(results):
    """
    Calculate and display odds ratios with confidence intervals.
    
    Args:
        results: Fitted logistic regression results
    """
    print("\n=== Odds Ratios ===")
    
    # Calculate odds ratios
    odds_ratios = np.exp(results.params)
    conf_int = np.exp(results.conf_int())
    
    # Create summary table
    or_summary = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'Lower CI': conf_int.iloc[:, 0],
        'Upper CI': conf_int.iloc[:, 1],
        'p-value': results.pvalues
    })
    
    print(or_summary)
    
    # Interpretation
    print("\nInterpretation:")
    for var in or_summary.index[1:]:  # Skip intercept
        or_val = or_summary.loc[var, 'Odds Ratio']
        p_val = or_summary.loc[var, 'p-value']
        
        if p_val < 0.05:
            if or_val > 1:
                print(f"- {var}: {((or_val - 1) * 100):.1f}% increase in odds per unit increase")
            else:
                print(f"- {var}: {((1 - or_val) * 100):.1f}% decrease in odds per unit increase")
        else:
            print(f"- {var}: No significant effect (p={p_val:.3f})")

def model_evaluation(results, data):
    """
    Evaluate model performance and create diagnostic plots.
    
    Args:
        results: Fitted logistic regression results
        data (pandas.DataFrame): Original dataset
    """
    print("\n=== Model Evaluation ===")
    
    # Get predictions
    predicted_probs = results.predict()
    predicted_classes = (predicted_probs > 0.5).astype(int)
    actual_classes = data['purchased']
    
    # Confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(actual_classes, predicted_classes))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(actual_classes, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    
    # Predicted probabilities distribution
    axes[0, 1].hist(predicted_probs[actual_classes == 0], bins=20, alpha=0.7, 
                    label='Class 0', density=True)
    axes[0, 1].hist(predicted_probs[actual_classes == 1], bins=20, alpha=0.7, 
                    label='Class 1', density=True)
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Predicted Probabilities')
    axes[0, 1].legend()
    
    # Residuals vs fitted
    residuals = results.resid_pearson
    axes[1, 0].scatter(predicted_probs, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Pearson Residuals')
    axes[1, 0].set_title('Residuals vs Fitted')
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

def predict_new_cases(results):
    """
    Demonstrate prediction on new cases.
    
    Args:
        results: Fitted logistic regression results
    """
    print("\n=== Sample Predictions ===")
    
    # Create new cases
    new_cases = pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'income': [30000, 45000, 60000, 75000, 90000],
        'education': [2, 3, 3, 4, 4]
    })
    
    # Generate predictions
    predictions = results.predict(new_cases)
    
    # Display results
    for i, (_, case) in enumerate(new_cases.iterrows()):
        prob = predictions.iloc[i]
        print(f"Case {i+1}:")
        print(f"  Age: {case['age']}, Income: ${case['income']:,}, Education: {case['education']}")
        print(f"  Probability of Purchase: {prob:.3f}")
        print(f"  Prediction: {'Purchase' if prob > 0.5 else 'No Purchase'}\n")

def main():
    """Main function to demonstrate logistic regression analysis."""
    # Generate sample data - FIXED: changed from generate_sample_data() to generate_binary_data()
    print("Generating sample dataset...")
    data = generate_binary_data()
    
    print("\nDataset Overview:")
    print(data.describe())
    print(f"\nClass Distribution:")
    print(data['purchased'].value_counts())
    print(f"Purchase Rate: {data['purchased'].mean():.3f}")
    
    # Perform logistic regression
    results = perform_logistic_regression(data)
    
    # Calculate odds ratios
    calculate_odds_ratios(results)
    
    # Model evaluation
    model_evaluation(results, data)
    
    # Prediction on new cases
    predict_new_cases(results)
    
    # Goodness of fit assessment
    print("=== Goodness of Fit ===")
    
    # Simple goodness of fit approximation
    predicted_probs = results.predict()
    predicted_classes = (predicted_probs > 0.5).astype(int)
    
    # Calculate accuracy for basic fit assessment
    accuracy = (predicted_classes == data['purchased']).mean()
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Model fit statistics (using available attributes)
    print(f"\nModel Fit Statistics:")
    print(f"AIC: {results.aic:.4f}")
    print(f"BIC: {results.bic:.4f}")
    print(f"Log-Likelihood: {results.llf:.4f}")
    print(f"Pseudo R-squared (McFadden): {results.prsquared:.4f}")
    
    # Additional model diagnostics
    print(f"\nModel Diagnostics:")
    print(f"Number of observations: {results.nobs}")
    print(f"Degrees of freedom (residuals): {results.df_resid}")
    print(f"Degrees of freedom (model): {results.df_model}")
    
    # Likelihood ratio test
    llr_pvalue = results.llr_pvalue
    print(f"Likelihood Ratio Test p-value: {llr_pvalue:.6f}")
    if llr_pvalue < 0.05:
        print("Model is significantly better than null model (p < 0.05)")
    else:
        print("Model is not significantly better than null model (p >= 0.05)")
    
    # Calculate deviance manually if needed
    # Deviance = -2 * log-likelihood
    deviance = -2 * results.llf
    print(f"Deviance: {deviance:.4f}")
    print(f"Null deviance: {results.llnull * -2:.4f}")
    print(f"Deviance explained: {(results.llnull * -2 - deviance):.4f}")

if __name__ == "__main__":
    main()