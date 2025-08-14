"""
@file MLModelApp.py
@brief Machine Learning Model Deployment with Streamlit


This module demonstrates deploying machine learning models using Streamlit,
including model training, prediction, and interactive model evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression
import joblib
import io


def generate_classification_data():
    """
    Generate sample classification dataset.
    
    @return tuple: Features, target, and feature names
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    return X, y, feature_names


def generate_regression_data():
    """
    Generate sample regression dataset.
    
    @return tuple: Features, target, and feature names
    """
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    return X, y, feature_names


def train_model(X, y, model_type, problem_type):
    """
    Train a machine learning model.
    
    @param X: Feature matrix
    @param y: Target variable
    @param model_type: Type of model to train
    @param problem_type: Classification or regression
    @return dict: Trained model and metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model based on type
    if problem_type == "Classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            X_test_final = X_test
        else:  # Logistic Regression
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            X_test_final = X_test_scaled
        
        # Make predictions
        y_pred = model.predict(X_test_final)
        y_pred_proba = model.predict_proba(X_test_final)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': model,
            'scaler': scaler if model_type == "Logistic Regression" else None,
            'X_test': X_test_final,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'report': report,
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }
    
    else:  # Regression
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            X_test_final = X_test
        else:  # Linear Regression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            X_test_final = X_test_scaled
        
        # Make predictions
        y_pred = model.predict(X_test_final)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler if model_type == "Linear Regression" else None,
            'X_test': X_test_final,
            'y_test': y_test,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        }


def main():
    """
    Main function to create the ML model deployment application.
    """
    # Set page configuration
    st.set_page_config(
        page_title="ML Model Deployment",
        page_icon=":robot_face:",
        layout="wide"
    )
    
    # Application header
    st.title("Machine Learning Model Deployment")
    st.markdown("---")
    st.write("Train, evaluate, and deploy machine learning models interactively.")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Problem type selection
    problem_type = st.sidebar.selectbox(
        "Problem Type",
        ["Classification", "Regression"]
    )
    
    # Model type selection
    if problem_type == "Classification":
        model_options = ["Random Forest", "Logistic Regression"]
    else:
        model_options = ["Random Forest", "Linear Regression"]
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        model_options
    )
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Generated Sample Data", "Upload Custom Data"]
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Data & Training", "Model Evaluation", "Make Predictions"])
    
    with tab1:
        st.header("Data Preparation and Model Training")
        
        if data_source == "Generated Sample Data":
            st.info("Using generated sample data for demonstration.")
            
            if problem_type == "Classification":
                X, y, feature_names = generate_classification_data()
                st.write(f"**Dataset:** {X.shape[0]} samples, {X.shape[1]} features, 2 classes")
            else:
                X, y, feature_names = generate_regression_data()
                st.write(f"**Dataset:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Display data preview
            df = pd.DataFrame(X, columns=feature_names)
            df['Target'] = y
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Train model button
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    results = train_model(X, y, model_type, problem_type)
                    st.session_state.model_results = results
                    st.session_state.feature_names = feature_names
                    st.session_state.problem_type = problem_type
                    st.session_state.model_type = model_type
                
                st.success("Model trained successfully!")
        
        else:
            st.subheader("Upload Custom Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with features and target variable"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"File uploaded successfully! Shape: {df.shape}")
                    
                    # Display data
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Target column selection
                    target_column = st.selectbox(
                        "Select Target Column",
                        df.columns.tolist()
                    )
                    
                    if st.button("Train Model with Uploaded Data", type="primary"):
                        # Prepare data
                        X = df.drop(columns=[target_column])
                        y = df[target_column]
                        
                        # Handle categorical variables
                        for col in X.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                        
                        feature_names = X.columns.tolist()
                        
                        with st.spinner("Training model..."):
                            results = train_model(X.values, y.values, model_type, problem_type)
                            st.session_state.model_results = results
                            st.session_state.feature_names = feature_names
                            st.session_state.problem_type = problem_type
                            st.session_state.model_type = model_type
                        
                        st.success("Model trained successfully!")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Model Evaluation")
        
        if 'model_results' in st.session_state:
            results = st.session_state.model_results
            problem_type = st.session_state.problem_type
            
            if problem_type == "Classification":
                # Classification metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(results['report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                
                with col2:
                    # Confusion matrix visualization
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(results['y_test'], results['y_pred'])
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if results['feature_importance'] is not None:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': results['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Regression metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("R2 Score", f"{results['r2']:.3f}")
                    st.metric("Mean Squared Error", f"{results['mse']:.3f}")
                
                with col2:
                    # Prediction vs Actual plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['y_test'],
                        y=results['y_pred'],
                        mode='markers',
                        name='Predictions',
                        opacity=0.7
                    ))
                    
                    # Perfect prediction line
                    min_val = min(results['y_test'].min(), results['y_pred'].min())
                    max_val = max(results['y_test'].max(), results['y_pred'].max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title="Predicted vs Actual Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if results['feature_importance'] is not None:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': results['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please train a model first in the 'Data & Training' tab.")
    
    with tab3:
        st.header("Make Predictions")
        
        if 'model_results' in st.session_state:
            results = st.session_state.model_results
            feature_names = st.session_state.feature_names
            problem_type = st.session_state.problem_type
            
            st.write("Enter feature values to make predictions:")
            
            # Create input widgets for each feature
            feature_values = {}
            cols = st.columns(min(3, len(feature_names)))
            
            for i, feature in enumerate(feature_names):
                with cols[i % len(cols)]:
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{feature}"
                    )
            
            if st.button("Make Prediction", type="primary"):
                # Prepare input data
                input_data = np.array([list(feature_values.values())])
                
                # Apply scaling if needed
                if results['scaler'] is not None:
                    input_data = results['scaler'].transform(input_data)
                
                # Make prediction
                prediction = results['model'].predict(input_data)[0]
                
                if problem_type == "Classification":
                    prediction_proba = results['model'].predict_proba(input_data)[0]
                    
                    st.success(f"Predicted Class: **{int(prediction)}**")
                    
                    # Show prediction probabilities
                    prob_df = pd.DataFrame({
                        'Class': range(len(prediction_proba)),
                        'Probability': prediction_proba
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.success(f"Predicted Value: **{prediction:.3f}**")
            
            # Batch prediction
            st.subheader("Batch Predictions")
            batch_file = st.file_uploader(
                "Upload CSV for batch predictions",
                type=['csv'],
                key="batch_upload"
            )
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    
                    if len(batch_df.columns) == len(feature_names):
                        # Apply scaling if needed
                        batch_data = batch_df.values
                        if results['scaler'] is not None:
                            batch_data = results['scaler'].transform(batch_data)
                        
                        # Make predictions
                        batch_predictions = results['model'].predict(batch_data)
                        
                        # Add predictions to dataframe
                        batch_df['Prediction'] = batch_predictions
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error(f"Expected {len(feature_names)} columns, got {len(batch_df.columns)}")
                
                except Exception as e:
                    st.error(f"Error processing batch file: {str(e)}")
        
        else:
            st.info("Please train a model first in the 'Data & Training' tab.")


if __name__ == "__main__":
    main()