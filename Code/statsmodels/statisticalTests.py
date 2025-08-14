"""
Statistical Tests and Diagnostics with Statsmodels (Fixed Version)

This module demonstrates various statistical tests and diagnostic procedures
available in statsmodels for comprehensive data analysis.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera  # Moved here
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def generate_test_data(n_samples=200, random_state=42):
    """
    Generate sample dataset for statistical testing.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(random_state)
    
    # Generate variables
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)  # Correlated with x1
    
    # Generate dependent variable with heteroscedasticity
    error_var = 0.5 + 0.3 * x1**2  # Heteroscedastic errors
    errors = np.random.normal(0, np.sqrt(error_var))
    y = 2 + 1.5 * x1 + 0.8 * x2 + 0.3 * x3 + errors
    
    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3
    })

def test_normality(data):
    """
    Perform normality tests on data.
    
    Args:
        data (pandas.DataFrame): Input dataset
    """
    print("=== Normality Tests ===")
    
    for col in data.columns:
        print(f"\nTesting normality for {col}:")
        
        # Shapiro-Wilk test (limit to 5000 samples for performance)
        sample_data = data[col].sample(min(len(data[col]), 5000), random_state=42)
        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
        print(f"Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        
        # Jarque-Bera test
        jb_stat, jb_p, _, _ = jarque_bera(data[col])
        print(f"Jarque-Bera: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data[col], 'norm', 
                                     args=(data[col].mean(), data[col].std()))
        print(f"Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
        
        # Interpretation
        is_normal = (shapiro_p > 0.05 and jb_p > 0.05 and ks_p > 0.05)
        print(f"Result: {'Normal' if is_normal else 'Non-normal'}")

def test_regression_assumptions(data):
    """
    Test regression assumptions comprehensively.
    
    Args:
        data (pandas.DataFrame): Input dataset
    """
    print("\n=== Regression Assumption Tests ===")
    
    # Fit regression model
    X = sm.add_constant(data[['x1', 'x2', 'x3']])
    model = sm.OLS(data['y'], X)
    results = model.fit()
    
    print("Model Summary:")
    print(results.summary())
    
    # 1. Linearity (visual inspection through residual plots)
    print(f"\n1. Linearity Check:")
    print("Visual inspection required - see residual plots")
    
    # 2. Independence (Durbin-Watson test)
    print(f"\n2. Independence Test:")
    dw_stat = durbin_watson(results.resid)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print(f"Interpretation: {'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Possible autocorrelation'}")
    
    # 3. Homoscedasticity tests
    print(f"\n3. Homoscedasticity Tests:")
    
    # Breusch-Pagan test
    bp_stat, bp_p, _, _ = het_breuschpagan(results.resid, X)
    print(f"Breusch-Pagan: statistic={bp_stat:.4f}, p-value={bp_p:.4f}")
    
    # White test
    white_stat, white_p, _, _ = het_white(results.resid, X)
    print(f"White test: statistic={white_stat:.4f}, p-value={white_p:.4f}")
    
    homo_result = (bp_p > 0.05 and white_p > 0.05)
    print(f"Result: {'Homoscedastic' if homo_result else 'Heteroscedastic'}")
    
    # 4. Normality of residuals
    print(f"\n4. Normality of Residuals:")
    jb_stat, jb_p, _, _ = jarque_bera(results.resid)
    print(f"Jarque-Bera: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
    print(f"Result: {'Normal residuals' if jb_p > 0.05 else 'Non-normal residuals'}")
    
    return results

def outlier_detection(results, data):
    """
    Detect outliers and influential observations.
    
    Args:
        results: Fitted regression results
        data (pandas.DataFrame): Original dataset
    """
    print("\n=== Outlier and Influence Diagnostics ===")
    
    # Calculate influence measures
    influence = OLSInfluence(results)
    
    # Leverage values
    leverage = influence.hat_matrix_diag
    leverage_threshold = 2 * (results.model.exog.shape[1]) / len(data)
    high_leverage = np.where(leverage > leverage_threshold)[0]
    
    print(f"High leverage observations (threshold: {leverage_threshold:.3f}):")
    if len(high_leverage) > 0:
        print(f"Observations: {high_leverage}")
        print(f"Leverage values: {leverage[high_leverage]}")
    else:
        print("No high leverage observations found")
    
    # Cook's distance
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4 / len(data)
    influential = np.where(cooks_d > cooks_threshold)[0]
    
    print(f"\nInfluential observations (Cook's D > {cooks_threshold:.3f}):")
    if len(influential) > 0:
        print(f"Observations: {influential}")
        print(f"Cook's D values: {cooks_d[influential]}")
    else:
        print("No influential observations found")
    
    # Studentized residuals
    student_resid = influence.resid_studentized_external
    outliers = np.where(np.abs(student_resid) > 2)[0]
    
    print(f"\nOutliers (|studentized residual| > 2):")
    if len(outliers) > 0:
        print(f"Observations: {outliers}")
        print(f"Studentized residuals: {student_resid[outliers]}")
    else:
        print("No outliers found")

def perform_anova(data):
    """
    Perform ANOVA tests and model comparisons.
    
    Args:
        data (pandas.DataFrame): Input dataset
    """
    print("\n=== ANOVA and Model Comparison ===")
    
    # Fit nested models
    model_full = smf.ols('y ~ x1 + x2 + x3', data=data)
    results_full = model_full.fit()
    
    model_reduced = smf.ols('y ~ x1', data=data)
    results_reduced = model_reduced.fit()
    
    # ANOVA table for full model
    print("ANOVA Table (Full Model):")
    anova_table = anova_lm(results_full)
    print(anova_table)
    
    # F-test for model comparison
    print(f"\nModel Comparison (F-test):")
    print(f"Full model R-squared: {results_full.rsquared:.4f}")
    print(f"Reduced model R-squared: {results_reduced.rsquared:.4f}")
    
    # Manual F-test calculation
    rss_full = results_full.ssr
    rss_reduced = results_reduced.ssr
    df_full = results_full.df_resid
    df_reduced = results_reduced.df_resid
    
    f_stat = ((rss_reduced - rss_full) / (df_reduced - df_full)) / (rss_full / df_full)
    f_p = 1 - stats.f.cdf(f_stat, df_reduced - df_full, df_full)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {f_p:.4f}")
    print(f"Result: {'Significant improvement' if f_p < 0.05 else 'No significant improvement'}")

def time_series_tests(n_periods=200):
    """
    Demonstrate time series statistical tests.
    
    Args:
        n_periods (int): Number of time periods
    """
    print("\n=== Time Series Tests ===")
    
    # Generate time series data
    np.random.seed(42)
    trend = np.linspace(0, 10, n_periods)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    noise = np.random.normal(0, 1, n_periods)
    
    # Non-stationary series
    ts_nonstationary = trend + seasonal + noise
    
    # Stationary series (differenced)
    ts_stationary = np.diff(ts_nonstationary)
    
    # Augmented Dickey-Fuller test
    print("Augmented Dickey-Fuller Test:")
    
    print("\nNon-stationary series:")
    adf_result1 = adfuller(ts_nonstationary, autolag='AIC')
    adf_stat1, adf_p1 = adf_result1[0], adf_result1[1]
    adf_crit1 = adf_result1[4]
    print(f"ADF Statistic: {adf_stat1:.4f}")
    print(f"p-value: {adf_p1:.4f}")
    print(f"Critical Values: {adf_crit1}")
    print(f"Result: {'Stationary' if adf_p1 < 0.05 else 'Non-stationary'}")
    
    print("\nDifferenced series:")
    adf_result2 = adfuller(ts_stationary, autolag='AIC')
    adf_stat2, adf_p2 = adf_result2[0], adf_result2[1]
    adf_crit2 = adf_result2[4]
    print(f"ADF Statistic: {adf_stat2:.4f}")
    print(f"p-value: {adf_p2:.4f}")
    print(f"Critical Values: {adf_crit2}")
    print(f"Result: {'Stationary' if adf_p2 < 0.05 else 'Non-stationary'}")
    
    # KPSS test
    print("\nKPSS Test:")
    kpss_result = kpss(ts_nonstationary, regression='ct')
    kpss_stat1, kpss_p1 = kpss_result[0], kpss_result[1]
    kpss_crit1 = kpss_result[3]
    print(f"KPSS Statistic: {kpss_stat1:.4f}")
    print(f"p-value: {kpss_p1:.4f}")
    print(f"Critical Values: {kpss_crit1}")
    print(f"Result: {'Stationary' if kpss_p1 > 0.05 else 'Non-stationary'}")

def create_diagnostic_plots(results, data):
    """
    Create comprehensive diagnostic plots.
    
    Args:
        results: Fitted regression results
        data (pandas.DataFrame): Original dataset
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Residuals vs Fitted
    fitted_values = results.fittedvalues
    residuals = results.resid
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # 3. Scale-Location plot
    standardized_resid = np.sqrt(np.abs(results.resid_pearson))
    axes[1, 0].scatter(fitted_values, standardized_resid, alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    axes[1, 0].set_title('Scale-Location Plot')
    
    # 4. Cook's Distance
    influence = OLSInfluence(results)
    cooks_d = influence.cooks_distance[0]
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    axes[1, 1].axhline(y=4/len(data), color='red', linestyle='--', 
                       label='Threshold')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].legend()
    
    # 5. Leverage vs Residuals
    leverage = influence.hat_matrix_diag
    studentized_resid = influence.resid_studentized_external
    axes[2, 0].scatter(leverage, studentized_resid, alpha=0.6)
    axes[2, 0].axhline(y=0, color='red', linestyle='--')
    axes[2, 0].axvline(x=2*results.model.exog.shape[1]/len(data), 
                       color='red', linestyle='--')
    axes[2, 0].set_xlabel('Leverage')
    axes[2, 0].set_ylabel('Studentized Residuals')
    axes[2, 0].set_title('Leverage vs Studentized Residuals')
    
    # 6. Histogram of residuals
    axes[2, 1].hist(residuals, bins=20, density=True, alpha=0.7)
    axes[2, 1].set_xlabel('Residuals')
    axes[2, 1].set_ylabel('Density')
    axes[2, 1].set_title('Distribution of Residuals')
    
    # Add normal curve overlay
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[2, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                    'r-', label='Normal')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate statistical tests and diagnostics."""
    # Generate test data
    print("Generating sample dataset for statistical testing...")
    data = generate_test_data()
    
    print("Dataset Overview:")
    print(data.describe())
    
    # Test normality of variables
    test_normality(data)
    
    # Test regression assumptions
    results = test_regression_assumptions(data)
    
    # Outlier detection
    outlier_detection(results, data)
    
    # ANOVA tests
    perform_anova(data)
    
    # Time series tests
    time_series_tests()
    
    # Create diagnostic plots
    create_diagnostic_plots(results, data)
    
    # Additional statistical tests
    print("\n=== Additional Tests ===")
    
    # Correlation tests
    print("Correlation Analysis:")
    corr_matrix = data.corr()
    print(corr_matrix)
    
    # Test correlation significance
    for i in range(len(data.columns)):
        for j in range(i+1, len(data.columns)):
            col1, col2 = data.columns[i], data.columns[j]
            corr_coef, corr_p = stats.pearsonr(data[col1], data[col2])
            print(f"{col1} vs {col2}: r={corr_coef:.3f}, p={corr_p:.3f}")

if __name__ == "__main__":
    main()