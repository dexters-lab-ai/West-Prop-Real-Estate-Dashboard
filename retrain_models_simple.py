#!/usr/bin/env python3
"""
Simple model retraining script that works with current environment
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

def retrain_models():
    """Retrain models with current scikit-learn version"""
    print("Starting model retraining...")
    
    # Load data
    data_path = os.path.join("data", "real_estate_data_template.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Prepare features
    feature_columns = [
        'stand_size_sqm', 'building_size_sqm', 'bedrooms', 'bathrooms', 
        'sale_price_usd', 'has_solar', 'has_water_recycling', 'has_smart_locks',
        'has_smart_thermostats', 'has_integrated_security', 'has_ev_charging'
    ]
    
    # Add categorical features if they exist
    categorical_features = ['location_suburb', 'property_type']
    for cat_feature in categorical_features:
        if cat_feature in df.columns:
            feature_columns.append(cat_feature)
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using features: {available_features}")
    
    # Prepare data
    X = df[available_features].copy()
    
    # Handle categorical variables
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            X = pd.get_dummies(X, columns=[cat_feature], drop_first=True)
    
    # Target variable
    if 'roi_percentage' in df.columns:
        y = df['roi_percentage']
    elif 'ROI (%)' in df.columns:
        y = df['ROI (%)']
    else:
        print("Error: No ROI target column found")
        return
    
    # Remove any rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Training on {len(X)} clean records")
    print(f"Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform target variable
    pt = PowerTransformer(method='yeo-johnson')
    y_train_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_transformed)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/roi_prediction_model.pkl")
    joblib.dump(pt, "models/roi_transformer.pkl")
    joblib.dump(list(X.columns), "models/model_features.pkl")
    
    # Test prediction
    y_pred_transformed = model.predict(X_test)
    y_pred = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
    
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.3f}")
    print("Models saved successfully!")

if __name__ == "__main__":
    retrain_models()
