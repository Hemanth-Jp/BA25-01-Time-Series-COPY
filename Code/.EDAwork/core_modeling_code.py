import numpy as np
import pandas as pd
import joblib
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# ============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

def load_and_merge_data(train_file, features_file, stores_file):
    """
    Load and merge the three CSV files
    Used by: App 1 (Training) to prepare data from user uploads
    """
    df_store = pd.read_csv(stores_file)
    df_train = pd.read_csv(train_file)
    df_features = pd.read_csv(features_file)
    
    # Merge datasets
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    df.drop(['IsHoliday_y'], axis=1, inplace=True)
    df.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)
    
    return df

def clean_data(df):
    """
    Clean the merged data
    Used by: App 1 (Training) before model training
    """
    # Remove non-positive sales
    df = df.loc[df['Weekly_Sales'] > 0]
    
    # Fill missing values in markdown columns with zeros
    df = df.fillna(0)
    
    # Create specific holiday indicators
    df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
    df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False
    
    df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
    df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False
    
    df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
    df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False
    
    df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
    df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False
    
    return df

def prepare_time_series_data(df):
    """
    Prepare data for time series modeling
    Used by: App 1 (Training) after data cleaning
    """
    # Convert date and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index('Date', inplace=True)
    
    # Create weekly aggregated data
    df_week = df.select_dtypes(include='number').resample('W').mean()
    
    # Difference the data for stationarity
    df_week_diff = df_week['Weekly_Sales'].diff().dropna()
    
    return df_week, df_week_diff

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_auto_arima(train_data_diff, hyperparams=None):
    """
    Train Auto ARIMA model
    Used by: App 1 (Training) when user selects Auto ARIMA
    
    Args:
        train_data_diff: Differenced training data
        hyperparams: Dict of custom hyperparameters (optional)
    
    Returns:
        Trained model
    """
    default_params = {
        'start_p': 0,
        'start_q': 0,
        'start_P': 0,
        'start_Q': 0,
        'max_p': 20,
        'max_q': 20,
        'max_P': 20,
        'max_Q': 20,
        'seasonal': True,
        'maxiter': 200,
        'information_criterion': 'aic',
        'stepwise': False,
        'suppress_warnings': True,
        'D': 1,
        'max_D': 10,
        'error_action': 'ignore',
        'approximation': False
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    model_auto_arima = auto_arima(train_data_diff, trace=True, **default_params)
    model_auto_arima.fit(train_data_diff)
    
    return model_auto_arima

def train_exponential_smoothing(train_data_diff, hyperparams=None):
    """
    Train Exponential Smoothing model
    Used by: App 1 (Training) when user selects Exponential Smoothing
    
    Args:
        train_data_diff: Differenced training data
        hyperparams: Dict of custom hyperparameters (optional)
    
    Returns:
        Trained model
    """
    default_params = {
        'seasonal_periods': 20,
        'seasonal': 'additive',
        'trend': 'additive',
        'damped': True
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    model_holt_winters = ExponentialSmoothing(
        train_data_diff,
        **default_params
    ).fit()
    
    return model_holt_winters

# ============================================================================
# MODEL SAVING AND LOADING FUNCTIONS
# ============================================================================

def save_model(model, filepath, model_type):
    """
    Save trained model
    Used by: App 1 (Training) to save models after training
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        model_type: 'auto_arima' or 'exponential_smoothing'
    """
    if model_type == 'auto_arima':
        joblib.dump(model, filepath)
    elif model_type == 'exponential_smoothing':
        model.save(filepath)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model(filepath, model_type):
    """
    Load a trained model
    Used by: App 2 (Prediction) to load default or uploaded models
    
    Args:
        filepath: Path to the model file
        model_type: 'auto_arima' or 'exponential_smoothing'
    
    Returns:
        Loaded model object
    """
    if model_type == 'auto_arima':
        model = joblib.load(filepath)
    elif model_type == 'exponential_smoothing':
        model = ExponentialSmoothing.load(filepath)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_next_4_weeks(model, model_type):
    """
    Predict next 4 weeks of sales
    Used by: App 2 (Prediction) to generate forecasts
    
    Args:
        model: Trained model object
        model_type: 'auto_arima' or 'exponential_smoothing'
    
    Returns:
        predictions: Array of predictions for next 4 weeks
        dates: Array of dates for the predictions
    """
    # Get the last date from model (assuming weekly data)
    # This would be adjusted based on how you store training metadata
    
    # For simplicity, we'll create the next 4 weeks of dates
    from datetime import datetime, timedelta
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, 5)]
    
    if model_type == 'auto_arima':
        predictions = model.predict(n_periods=4)
    elif model_type == 'exponential_smoothing':
        predictions = model.forecast(4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return predictions, dates

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def wmae_ts(y_true, y_pred):
    """
    Calculate weighted mean absolute error
    Used by: App 1 (Training) for model evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        WMAE score
    """
    # Convert to numpy arrays to avoid pandas alignment issues
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values
    
    weights = np.ones_like(y_true)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# ============================================================================
# DIAGNOSTIC PLOTTING FOR TRAINING APP
# ============================================================================

def create_diagnostic_plots(train_data, test_data, predictions, model_type):
    """
    Create diagnostic plots for model evaluation
    Used by: App 1 (Training) to display model performance
    
    Args:
        train_data: Training data
        test_data: Test data
        predictions: Model predictions
        model_type: Type of model used
    
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 6))
    plt.title(f'Prediction using {model_type}', fontsize=15)
    plt.plot(train_data.index, train_data.values, label='Train')
    plt.plot(test_data.index, test_data.values, label='Test')
    plt.plot(test_data.index, predictions, label='Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales (Differenced)')
    plt.grid(True)
    
    return plt.gcf()

# ============================================================================
# EXAMPLE USAGE FOR STREAMLIT APPS
# ============================================================================

"""
Example usage for App 1 (Training):

# Load user uploaded files
df = load_and_merge_data(train_file, features_file, stores_file)

# Clean data
df = clean_data(df)

# Prepare time series data
df_week, df_week_diff = prepare_time_series_data(df)

# Split into train/test
train_size = int(0.7 * len(df_week_diff))
train_data_diff = df_week_diff[:train_size]
test_data_diff = df_week_diff[train_size:]

# Train model (Auto ARIMA example)
model = train_auto_arima(train_data_diff)

# Make predictions for evaluation
predictions = model.predict(n_periods=len(test_data_diff))

# Calculate WMAE
wmae = wmae_ts(test_data_diff, predictions)

# Create diagnostic plots
fig = create_diagnostic_plots(train_data_diff, test_data_diff, predictions, 'Auto ARIMA')

# Save model
save_model(model, 'models/default/auto_arima.pkl', 'auto_arima')
"""

"""
Example usage for App 2 (Prediction):

# Load model (default or uploaded)
model = load_model('models/default/auto_arima.pkl', 'auto_arima')

# Or handle uploaded file
uploaded_file = st.file_uploader("Upload model", type=['pkl'])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        model = load_model(tmp_path, 'auto_arima')
    except:
        st.error("Invalid model file")
        model = None

# Make predictions if model loaded successfully
if model:
    predictions, dates = predict_next_4_weeks(model, 'auto_arima')
    
    # Create prediction DataFrame for download
    prediction_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Sales': predictions
    })
    
    # Create interactive plot with plotly
    import plotly.express as px
    fig = px.line(prediction_df, x='Date', y='Predicted_Sales',
                  title='Sales Forecast for Next 4 Weeks',
                  labels={'Predicted_Sales': 'Weekly Sales'})
    
    st.plotly_chart(fig)
    st.download_button("Download Predictions", 
                       prediction_df.to_csv(index=False),
                       "predictions.csv",
                       "text/csv")
"""
