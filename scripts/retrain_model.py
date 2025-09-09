import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
import joblib
import os

print("Starting model retraining process...")

# --- 1. Load Data ---
data_path = os.path.join("..", "data", "real_estate_data_template.csv")
df = pd.read_csv(data_path)
print(f"Loaded dataset with shape: {df.shape}")

# --- 2. Preprocessing ---
# Handle missing values in target
df.dropna(subset=["ROI_percentage"], inplace=True)

# Apply PowerTransformer to the target variable
pt = PowerTransformer(method='yeo-johnson')
y_transformed = pt.fit_transform(df['ROI_percentage'].values.reshape(-1, 1))
df['ROI_percentage_transformed'] = y_transformed
print("Applied PowerTransformer to target variable 'ROI_percentage'.")

# Define features
numerical_features = ["stand_size_sqm", "building_size_sqm", "bedrooms", "bathrooms", "sale_price_usd"]
smart_features = ["has_solar", "has_water_recycling", "has_smart_locks", "has_smart_thermostats", "has_integrated_security", "has_ev_charging"]
categorical_features = ["location_suburb", "property_type"]

# Impute missing values in features
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

for col in smart_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(0, inplace=True)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna('Unknown', inplace=True)
print("Handled missing values in features.")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
encoded_categorical_features = [col for col in df_encoded.columns if col.startswith(tuple(categorical_features))]
all_features = numerical_features + smart_features + encoded_categorical_features
print("Applied one-hot encoding to categorical features.")

# --- 3. Prepare Data for Training ---
X = df_encoded[all_features]
y = df_encoded["ROI_percentage_transformed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 4. Train Model ---
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
print("Training Random Forest model...")
model.fit(X_train, y_train)
print("Model training completed.")

# --- 5. Save Artifacts ---
models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)

model_filename = os.path.join(models_dir, "roi_prediction_model.pkl")
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

transformer_filename = os.path.join(models_dir, "roi_transformer.pkl")
joblib.dump(pt, transformer_filename)
print(f"Transformer saved to {transformer_filename}")

features_filename = os.path.join(models_dir, "model_features.pkl")
joblib.dump(all_features, features_filename)
print(f"Feature list saved to {features_filename}")

print("Retraining process finished successfully.")
