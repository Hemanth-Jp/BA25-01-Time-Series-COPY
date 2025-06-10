# Walmart Sales Forecasting System

A comprehensive time series forecasting solution for predicting Walmart store sales.

## 📊 Project Overview

This project consists of two complementary Streamlit applications:

1. **Walmart Sales Training App**: Train time series models using historical Walmart sales data
2. **Walmart Sales Prediction App**: Generate weekly sales forecasts using the trained models

Both applications are deployed on Streamlit Community Cloud and can be accessed via the links below:

- 🔗 [Walmart Sales Training App](https://walmart-sales-training-app.streamlit.app)
- 🔗 [Walmart Sales Prediction App](https://walmart-sales-prediction-app.streamlit.app)

## 📋 Project Structure

```
.
├── WalmartSalesPredictionApp
│   ├── WalmartSalesPredictionApp.py
│   ├── models
│   │   └── default
│   │       ├── auto_arima.pkl
│   │       └── exponential_smoothing.pkl
│   └── requirements.txt
└── WalmartSalesTrainingApp
    ├── WalmartSalesTrainingApp.py
    ├── models
    │   └── default
    │       ├── auto_arima.pkl
    │       └── exponential_smoothing.pkl
    ├── requirements.txt
    └── utils
        └── walmart_test_data_generator.py
```

## 🚀 Getting Started

### Using the Deployed Apps

#### Walmart Sales Training App

1. Visit [https://walmart-sales-training-app.streamlit.app](https://walmart-sales-training-app.streamlit.app)
2. Upload the required CSV files:
   - `train.csv`: Historical weekly sales data
   - `features.csv`: Additional features like temperature, fuel price, etc.
   - `stores.csv`: Store-specific information (type, size)
3. Select a model type (Auto ARIMA or Exponential Smoothing)
4. Customize hyperparameters as needed
5. Click "Start Training" to train the model
6. View the results and download the trained model file

#### Walmart Sales Prediction App

1. Visit [https://walmart-sales-prediction-app.streamlit.app](https://walmart-sales-prediction-app.streamlit.app)
2. Choose one of two options:
   - Use a pre-loaded default model (recommended for beginners)
   - Upload your custom trained model (.pkl file from the Training App)
3. Click "Generate 4-Week Forecast" to see sales predictions
4. View the interactive forecast visualization and data table
5. Download the prediction results as CSV or JSON

### Self-Hosting Locally

#### Prerequisites

- Python 3.8+ installed
- Git (optional, for cloning the repository)

#### Installation Steps

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the desired app directory:
   ```bash
   # For the Training App
   cd WalmartSalesTrainingApp
   
   # OR
   
   # For the Prediction App
   cd WalmartSalesPredictionApp
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Ensure the required directory structure exists:
   ```bash
   # For both apps
   mkdir -p models/default
   ```

6. Run the Streamlit app:
   ```bash
   streamlit run WalmartSalesTrainingApp.py
   # OR
   streamlit run WalmartSalesPredictionApp.py
   ```

7. Access the app in your browser (typically at http://localhost:8501)

## 📊 Data Requirements

### Expected Data Formats

For the Walmart Sales Training App, you'll need three CSV files:

1. **train.csv**:
   - Columns: `Store`, `Date`, `Weekly_Sales`, `IsHoliday`
   - Each row represents weekly sales for a specific store on a specific date

2. **features.csv**:
   - Columns: `Store`, `Date`, `Temperature`, `Fuel_Price`, `MarkDown1`, `MarkDown2`, `MarkDown3`, `MarkDown4`, `MarkDown5`, `CPI`, `Unemployment`, `IsHoliday`
   - Contains additional features that might influence sales

3. **stores.csv**:
   - Columns: `Store`, `Type`, `Size`
   - Information about each store's type and size

## 🤖 Models

The system supports two types of time series models:

1. **Auto ARIMA**: Automatically identifies the best ARIMA (AutoRegressive Integrated Moving Average) model parameters
   - Good for capturing linear relationships in the data
   - Handles trends and seasonality

2. **Exponential Smoothing (Holt-Winters)**: 
   - Effectively captures seasonality and trends
   - Often performs well for retail sales forecasting
   - More robust to outliers

### Key Hyperparameters

#### Auto ARIMA
- `start_p`, `start_q`: Starting values for ARIMA parameters
- `max_p`, `max_q`: Maximum values for ARIMA parameters
- `start_P`, `start_Q`: Starting values for seasonal ARIMA parameters
- `max_P`, `max_Q`: Maximum values for seasonal ARIMA parameters

#### Exponential Smoothing
- `seasonal_periods`: Number of time periods in a season (e.g., 52 for weekly data with yearly patterns)
- `seasonal`: Type of seasonal component ('additive' or 'multiplicative')
- `trend`: Type of trend component ('additive', 'multiplicative', or None)
- `damped`: Whether to use damped trend (often improves forecasts)

## 📈 Forecast Interpretation

The prediction app generates forecasts showing **week-over-week sales changes** (not absolute values):

- **Green bars/values**: Indicate sales increases from previous week
- **Red bars/values**: Indicate sales decreases from previous week
- Values represent dollar amount changes

The app provides:
- Interactive visualizations
- Weekly prediction values
- Summary statistics (cumulative impact, growth weeks, best/worst weeks)
- Download options for further analysis

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure the model file (.pkl) was created with compatible versions of the libraries
   - Try using the default models if your custom model fails to load

2. **Missing Dependencies**:
   - If you encounter module not found errors when self-hosting, ensure you've installed all requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **File Upload Issues**:
   - Verify your CSV files match the expected format
   - Check for missing values or inconsistent data types

4. **Prediction Failures**:
   - The prediction app only works with models trained using the same feature structure
   - Try using the default models if custom models fail

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html) (for Exponential Smoothing)
- [pmdarima Documentation](https://alkaline-ml.com/pmdarima/) (for Auto ARIMA)

## 📝 License

This project is open-source and available under the MIT License.

## 📬 Contact

For questions, issues, or suggestions, please create an issue in this repository or contact the project maintainer.

---

Happy forecasting! 📈
