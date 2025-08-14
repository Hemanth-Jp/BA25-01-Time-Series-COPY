"""
Advanced Model Diagnostics for Pmdarima

This module provides comprehensive diagnostic tools for ARIMA models,
including residual analysis, model validation, and performance assessment.


@version: 1.1 (Fixed imports)
"""

import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from pmdarima.arima.stationarity import ADFTest, KPSSTest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ARIMADiagnostics:
    """
    Comprehensive diagnostic toolkit for ARIMA models.
    """
    
    def __init__(self, model, y_train, y_test=None):
        """
        Initialize diagnostics with fitted model and data.
        
        Args:
            model: Fitted pmdarima ARIMA model
            y_train: Training data
            y_test: Test data (optional)
        """
        self.model = model
        self.y_train = y_train
        self.y_test = y_test
        self.residuals = model.resid()
        self.fitted_values = y_train - self.residuals
        
    def residual_analysis(self):
        """
        Perform comprehensive residual analysis.
        
        Returns:
            dict: Residual analysis results
        """
        print("=== Residual Analysis ===")
        
        # Basic statistics
        resid_mean = np.mean(self.residuals)
        resid_std = np.std(self.residuals)
        resid_skew = stats.skew(self.residuals)
        resid_kurtosis = stats.kurtosis(self.residuals)
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
        jarque_stat, jarque_p = stats.jarque_bera(self.residuals)
        
        # Autocorrelation test (Ljung-Box)
        ljung_stat, ljung_p = self._ljung_box_test(self.residuals, lags=10)
        
        results = {
            'mean': resid_mean,
            'std': resid_std,
            'skewness': resid_skew,
            'kurtosis': resid_kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'jarque_stat': jarque_stat,
            'jarque_p': jarque_p,
            'ljung_stat': ljung_stat,
            'ljung_p': ljung_p
        }
        
        # Print results
        print(f"Mean: {resid_mean:.4f}")
        print(f"Std Dev: {resid_std:.4f}")
        print(f"Skewness: {resid_skew:.4f}")
        print(f"Kurtosis: {resid_kurtosis:.4f}")
        print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
        print(f"Jarque-Bera p-value: {jarque_p:.4f}")
        print(f"Ljung-Box p-value: {ljung_p:.4f}")
        
        return results
    
    def _ljung_box_test(self, residuals, lags=10):
        """
        Perform Ljung-Box test for autocorrelation.
        
        Args:
            residuals: Model residuals
            lags: Number of lags to test
            
        Returns:
            tuple: Test statistic and p-value
        """
        n = len(residuals)
        acf_vals = acf(residuals, nlags=lags, fft=False)
        
        # Calculate Ljung-Box statistic
        ljung_stat = n * (n + 2) * np.sum([(acf_vals[i]**2) / (n - i) for i in range(1, lags + 1)])
        ljung_p = 1 - stats.chi2.cdf(ljung_stat, lags)
        
        return ljung_stat, ljung_p
    
    def model_fit_assessment(self):
        """
        Assess overall model fit quality.
        
        Returns:
            dict: Model fit assessment results
        """
        print("\n=== Model Fit Assessment ===")
        
        # Information criteria
        aic = self.model.aic()
        bic = self.model.bic()
        
        # Fitted vs actual correlation
        fit_corr = np.corrcoef(self.y_train, self.fitted_values)[0, 1]
        
        # Mean absolute percentage error on training
        train_mape = np.mean(np.abs((self.y_train - self.fitted_values) / self.y_train)) * 100
        
        results = {
            'aic': aic,
            'bic': bic,
            'fit_correlation': fit_corr,
            'train_mape': train_mape
        }
        
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        print(f"Fit Correlation: {fit_corr:.4f}")
        print(f"Training MAPE: {train_mape:.2f}%")
        
        # Test set performance if available
        if self.y_test is not None:
            test_pred = self.model.predict(n_periods=len(self.y_test))
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_mape = np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100
            
            results.update({
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape
            })
            
            print(f"Test MAE: {test_mae:.2f}")
            print(f"Test RMSE: {test_rmse:.2f}")
            print(f"Test MAPE: {test_mape:.2f}%")
        
        return results
    
    def parameter_significance(self):
        """
        Analyze parameter significance and stability.
        
        Returns:
            dict: Parameter analysis results
        """
        print("\n=== Parameter Significance ===")
        
        try:
            # Get parameter estimates and standard errors
            params = self.model.params()
            
            # Try to get p-values (may not be available in all pmdarima versions)
            try:
                pvalues = self.model.pvalues()
            except (AttributeError, TypeError):
                print("P-values not available for this model version")
                return {'parameters': params}
            
            # Print parameter significance
            param_names = []
            if hasattr(self.model, 'arparams') and len(self.model.arparams) > 0:
                param_names.extend(['AR' + str(i+1) for i in range(len(self.model.arparams))])
            if hasattr(self.model, 'maparams') and len(self.model.maparams) > 0:
                param_names.extend(['MA' + str(i+1) for i in range(len(self.model.maparams))])
            
            # If we don't have enough parameter names, create generic ones
            if len(param_names) < len(pvalues):
                param_names = [f'Param_{i+1}' for i in range(len(pvalues))]
            
            significant_params = []
            for i, (name, pval) in enumerate(zip(param_names, pvalues)):
                if i < len(pvalues):  # Safety check
                    is_significant = pval < 0.05
                    significant_params.append(is_significant)
                    print(f"{name}: p-value = {pval:.4f} {'*' if is_significant else ''}")
            
            return {
                'parameters': params,
                'p_values': pvalues,
                'significant': significant_params
            }
            
        except Exception as e:
            print(f"Parameter analysis unavailable: {e}")
            return {}
    
    def create_diagnostic_plots(self):
        """
        Create comprehensive diagnostic plots.
        
        Returns:
            matplotlib.figure.Figure: Diagnostic plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(self.fitted_values, self.residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        stats.probplot(self.residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Histogram
        axes[0, 2].hist(self.residuals, bins=20, density=True, alpha=0.7, color='skyblue')
        # Overlay normal distribution
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        axes[0, 2].plot(x, stats.norm.pdf(x, np.mean(self.residuals), np.std(self.residuals)), 
                       'r-', linewidth=2, label='Normal')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. ACF of Residuals
        max_lags = min(20, len(self.residuals) // 4)
        lags = np.arange(1, max_lags + 1)
        acf_vals = acf(self.residuals, nlags=max_lags, fft=False)
        
        axes[1, 0].stem(lags, acf_vals[1:], basefmt=" ")
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(self.residuals))
        axes[1, 0].axhline(y=conf_level, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=-conf_level, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('ACF of Residuals')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Time Series Plot of Residuals
        if hasattr(self.y_train, 'index'):
            axes[1, 1].plot(self.y_train.index, self.residuals, alpha=0.7)
        else:
            axes[1, 1].plot(self.residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Forecast vs Actual (if test data available)
        if self.y_test is not None:
            test_pred = self.model.predict(n_periods=len(self.y_test))
            axes[1, 2].scatter(self.y_test, test_pred, alpha=0.6)
            min_val = min(self.y_test.min(), test_pred.min())
            max_val = max(self.y_test.max(), test_pred.max())
            axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 2].set_xlabel('Actual')
            axes[1, 2].set_ylabel('Predicted')
            axes[1, 2].set_title('Forecast vs Actual')
        else:
            # Show fitted vs actual for training data
            axes[1, 2].scatter(self.y_train, self.fitted_values, alpha=0.6)
            min_val = min(self.y_train.min(), self.fitted_values.min())
            max_val = max(self.y_train.max(), self.fitted_values.max())
            axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 2].set_xlabel('Actual')
            axes[1, 2].set_ylabel('Fitted')
            axes[1, 2].set_title('Fitted vs Actual')
        
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def comprehensive_model_evaluation(y, test_size=0.2):
    """
    Perform comprehensive model evaluation workflow.
    
    Args:
        y: Time series data
        test_size: Proportion of data for testing
        
    Returns:
        dict: Complete evaluation results
    """
    print("=== Comprehensive Model Evaluation ===\n")
    
    # Split data
    split_idx = int(len(y) * (1 - test_size))
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Fit model
    model = pm.auto_arima(
        y_train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        random_state=42
    )
    
    print(f"Selected model: ARIMA{model.order} x {model.seasonal_order}")
    
    # Create diagnostics object
    diagnostics = ARIMADiagnostics(model, y_train, y_test)
    
    # Perform all diagnostic tests
    residual_results = diagnostics.residual_analysis()
    fit_results = diagnostics.model_fit_assessment()
    param_results = diagnostics.parameter_significance()
    
    # Create diagnostic plots
    diagnostic_fig = diagnostics.create_diagnostic_plots()
    diagnostic_fig.suptitle('ARIMA Model Diagnostics', fontsize=14, fontweight='bold')
    
    # Compile results
    evaluation_results = {
        'model_order': model.order,
        'seasonal_order': model.seasonal_order,
        'residual_analysis': residual_results,
        'fit_assessment': fit_results,
        'parameter_analysis': param_results
    }
    
    return evaluation_results, diagnostic_fig

def main():
    """
    Main function demonstrating comprehensive model diagnostics.
    """
    print("=== Advanced Model Diagnostics for Pmdarima ===\n")
    
    # Load sample data
    y = pm.datasets.load_wineind()
    dates = pd.date_range(start='1980-01', periods=len(y), freq='M')
    ts_data = pd.Series(y, index=dates, name='Wine_Sales')
    
    print(f"Dataset: {len(ts_data)} monthly observations")
    
    # Perform comprehensive evaluation
    results, fig = comprehensive_model_evaluation(ts_data, test_size=0.2)
    
    # Display summary
    print(f"\n=== Diagnostic Summary ===")
    print(f"Model: ARIMA{results['model_order']} x {results['seasonal_order']}")
    
    if 'test_mape' in results['fit_assessment']:
        print(f"Test MAPE: {results['fit_assessment']['test_mape']:.2f}%")
    
    print(f"Residual normality p-value: {results['residual_analysis']['shapiro_p']:.4f}")
    print(f"Ljung-Box p-value: {results['residual_analysis']['ljung_p']:.4f}")
    
    # Interpretation
    print(f"\n=== Model Assessment ===")
    if results['residual_analysis']['shapiro_p'] > 0.05:
        print("✓ Residuals appear normally distributed")
    else:
        print("✗ Residuals may not be normally distributed")
    
    if results['residual_analysis']['ljung_p'] > 0.05:
        print("✓ No significant autocorrelation in residuals")
    else:
        print("✗ Residuals show autocorrelation - model may be inadequate")
    
    plt.show()
    
    print(f"\n=== Model Diagnostics Complete ===")

if __name__ == "__main__":
    main()