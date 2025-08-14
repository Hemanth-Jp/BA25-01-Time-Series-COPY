"""
Statistical Best Practices and Error Handling with Statsmodels

This module demonstrates proper error handling, validation procedures,
and statistical best practices when using statsmodels.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera  # Correct import location
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score, KFold
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    A robust statistical analyzer with comprehensive error handling
    and validation procedures.
    """
    
    def __init__(self, validate_assumptions=True, handle_missing=True):
        """
        Initialize the statistical analyzer.
        
        Args:
            validate_assumptions (bool): Whether to validate model assumptions
            handle_missing (bool): Whether to handle missing data automatically
        """
        self.validate_assumptions = validate_assumptions
        self.handle_missing = handle_missing
        self.results = None
        self.validation_results = {}
        
    def validate_data(self, data, target_col, feature_cols):
        """
        Validate input data for statistical analysis.
        
        Args:
            data (pandas.DataFrame): Input dataset
            target_col (str): Target variable column name
            feature_cols (list): Feature column names
            
        Returns:
            pandas.DataFrame: Validated dataset
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            logger.info("Starting data validation...")
            
            # Check if data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            # Check if required columns exist
            required_cols = [target_col] + feature_cols
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Check for empty dataset
            if len(data) == 0:
                raise ValueError("Dataset is empty")
            
            # Check data types
            numeric_cols = [target_col] + feature_cols
            non_numeric = []
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    non_numeric.append(col)
            
            if non_numeric:
                logger.warning(f"Non-numeric columns detected: {non_numeric}")
                # Attempt conversion
                for col in non_numeric:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        logger.info(f"Converted {col} to numeric")
                    except:
                        raise ValueError(f"Cannot convert {col} to numeric")
            
            # Handle missing values
            if self.handle_missing:
                data = self._handle_missing_data(data, required_cols)
            else:
                missing_count = data[required_cols].isnull().sum().sum()
                if missing_count > 0:
                    raise ValueError(f"Dataset contains {missing_count} missing values")
            
            # Check for sufficient sample size
            min_sample_size = len(feature_cols) * 10  # Rule of thumb
            if len(data) < min_sample_size:
                logger.warning(f"Small sample size: {len(data)} (recommended: >{min_sample_size})")
            
            # Check for multicollinearity
            if len(feature_cols) > 1:
                self._check_multicollinearity(data, feature_cols)
            
            logger.info("Data validation completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _handle_missing_data(self, data, columns):
        """
        Handle missing data using appropriate strategies.
        
        Args:
            data (pandas.DataFrame): Input dataset
            columns (list): Columns to check for missing data
            
        Returns:
            pandas.DataFrame: Dataset with missing data handled
        """
        missing_summary = data[columns].isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing == 0:
            return data
        
        logger.info(f"Found {total_missing} missing values")
        
        # Strategy: Remove rows with any missing values if < 5% of data
        missing_pct = total_missing / (len(data) * len(columns))
        
        if missing_pct < 0.05:
            data_clean = data.dropna(subset=columns)
            logger.info(f"Removed {len(data) - len(data_clean)} rows with missing values")
            return data_clean
        else:
            # For higher missing rates, use imputation
            logger.warning("High missing data rate - using mean imputation")
            for col in columns:
                if data[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data[col].fillna(data[col].mean(), inplace=True)
                    else:
                        data[col].fillna(data[col].mode()[0], inplace=True)
            return data
    
    def _check_multicollinearity(self, data, feature_cols):
        """
        Check for multicollinearity using VIF.
        
        Args:
            data (pandas.DataFrame): Input dataset
            feature_cols (list): Feature column names
        """
        try:
            X = data[feature_cols]
            X = sm.add_constant(X)
            
            vif_values = []
            for i in range(1, X.shape[1]):  # Skip constant
                vif = variance_inflation_factor(X.values, i)
                vif_values.append((feature_cols[i-1], vif))
            
            high_vif = [(col, vif) for col, vif in vif_values if vif > 5]
            
            if high_vif:
                logger.warning("High multicollinearity detected:")
                for col, vif in high_vif:
                    logger.warning(f"  {col}: VIF = {vif:.2f}")
                    
        except Exception as e:
            logger.warning(f"Could not check multicollinearity: {str(e)}")
    
    def fit_linear_model(self, data, formula, robust=False):
        """
        Fit linear regression model with error handling.
        
        Args:
            data (pandas.DataFrame): Input dataset
            formula (str): Regression formula
            robust (bool): Whether to use robust standard errors
            
        Returns:
            statsmodels results object or None if fitting fails
        """
        try:
            logger.info(f"Fitting model: {formula}")
            
            # Fit model
            if robust:
                model = smf.ols(formula, data=data)
                self.results = model.fit(cov_type='HC3')  # Robust standard errors
                logger.info("Applied robust standard errors")
            else:
                model = smf.ols(formula, data=data)
                self.results = model.fit()
            
            # Validate assumptions if requested
            if self.validate_assumptions:
                self._validate_assumptions()
            
            # Log model statistics
            logger.info(f"Model fitted successfully:")
            logger.info(f"  R-squared: {self.results.rsquared:.4f}")
            logger.info(f"  AIC: {self.results.aic:.2f}")
            logger.info(f"  F-statistic p-value: {self.results.f_pvalue:.2e}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            return None
    
    def _validate_assumptions(self):
        """Validate linear regression assumptions."""
        if self.results is None:
            return
        
        logger.info("Validating model assumptions...")
        
        try:
            # 1. Normality of residuals
            jb_stat, jb_p = jarque_bera(self.results.resid)
            self.validation_results['normality'] = {
                'test': 'Jarque-Bera',
                'statistic': jb_stat,
                'p_value': jb_p,
                'assumption_met': jb_p > 0.05
            }
            
            # 2. Homoscedasticity
            bp_stat, bp_p, _, _ = het_breuschpagan(self.results.resid, 
                                                   self.results.model.exog)
            self.validation_results['homoscedasticity'] = {
                'test': 'Breusch-Pagan',
                'statistic': bp_stat,
                'p_value': bp_p,
                'assumption_met': bp_p > 0.05
            }
            
            # Log results
            for assumption, result in self.validation_results.items():
                status = "PASSED" if result['assumption_met'] else "FAILED"
                logger.info(f"  {assumption.title()}: {status} (p={result['p_value']:.4f})")
                
        except Exception as e:
            logger.warning(f"Assumption validation failed: {str(e)}")
    
    def cross_validate_model(self, data, formula, cv_folds=5):
        """
        Perform cross-validation for model evaluation.
        
        Args:
            data (pandas.DataFrame): Input dataset
            formula (str): Regression formula
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        try:
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            
            # Extract target and features from formula
            target = formula.split('~')[0].strip()
            features = [col.strip() for col in formula.split('~')[1].split('+')]
            
            # Prepare data
            X = data[features]
            y = data[target]
            
            # Perform cross-validation using sklearn wrapper
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            r2_scores = []
            rmse_scores = []
            
            for train_idx, test_idx in kfold.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Fit statsmodels on training data
                train_data = pd.concat([y_train, X_train], axis=1)
                model = smf.ols(formula, data=train_data)
                results = model.fit()
                
                # Predict on test data
                test_data = pd.concat([y_test, X_test], axis=1)
                y_pred = results.predict(test_data)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                r2_scores.append(r2)
                rmse_scores.append(rmse)
            
            cv_results = {
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'cv_folds': cv_folds
            }
            
            logger.info(f"Cross-validation completed:")
            logger.info(f"  R2 = {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
            logger.info(f"  RMSE = {cv_results['rmse_mean']:.4f} +/- {cv_results['rmse_std']:.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return None

def demonstrate_error_handling():
    """Demonstrate comprehensive error handling in statistical analysis."""
    
    print("=== Statistical Analysis with Error Handling ===")
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(validate_assumptions=True, handle_missing=True)
    
    try:
        # Generate problematic dataset
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.normal(10, 3, n),
            'x1': np.random.normal(5, 2, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(2, 1.5, n)
        })
        
        # Introduce missing values
        missing_indices = np.random.choice(n, size=5, replace=False)
        data.loc[missing_indices, 'x1'] = np.nan
        
        # Add correlation to create multicollinearity
        data['x4'] = 0.9 * data['x1'] + 0.1 * np.random.normal(0, 1, n)
        
        print(f"Generated dataset with {len(data)} observations")
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        # Validate data
        validated_data = analyzer.validate_data(
            data, 'y', ['x1', 'x2', 'x3', 'x4']
        )
        
        # Fit model
        formula = 'y ~ x1 + x2 + x3 + x4'
        results = analyzer.fit_linear_model(validated_data, formula, robust=True)
        
        if results:
            print("\nModel Summary:")
            print(results.summary().tables[1])
            
            # Cross-validation
            cv_results = analyzer.cross_validate_model(validated_data, formula)
            
            if cv_results:
                print(f"\nCross-Validation Results:")
                print(f"R2 = {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
                print(f"RMSE = {cv_results['rmse_mean']:.4f} +/- {cv_results['rmse_std']:.4f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    """Main function demonstrating error handling and best practices."""
    
    # Demonstrate comprehensive error handling
    demonstrate_error_handling()
    
    # Additional best practices
    print("\n=== Statistical Best Practices ===")
    print("1. Always validate your data before analysis")
    print("2. Check model assumptions systematically") 
    print("3. Use cross-validation for model evaluation")
    print("4. Handle missing data appropriately")
    print("5. Check for multicollinearity in multiple regression")
    print("6. Use robust standard errors when assumptions are violated")
    print("7. Report confidence intervals alongside point estimates")
    print("8. Document your analysis process and decisions")

if __name__ == "__main__":
    main()