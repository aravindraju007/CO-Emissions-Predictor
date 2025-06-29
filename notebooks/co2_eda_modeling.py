# CO₂ Emissions Prediction with XGBoost
# ======================================
# This script performs data loading, cleaning, feature engineering,
# model training, evaluation, and saving using XGBoost.

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import joblib
import os
# Load data directly
df = pd.read_csv('/Users/aravindraju/Documents/git/co2-emissions-predictor/data/raw/energy_buildings.csv')
# Drop unnecessary columns
df = df.drop(columns=['LMK_KEY', 'UPRN', 'UPRN_SOURCE', 'ADDRESS1', 'ADDRESS2', 'ADDRESS3', 
                      'POSTCODE', 'ADDRESS', 'BUILDING_REFERENCE_NUMBER'], errors='ignore')
# Drop datetime
df = df.drop(columns=['LODGEMENT_DATETIME'], errors='ignore')
# Drop rows missing target
df = df.dropna(subset=['CO2_EMISSIONS_CURRENT'])
# Drop columns with too many nulls
df = df.dropna(axis=1, thresh=len(df) * 0.9)
# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)
# Define features and target
features = [
    'PROPERTY_TYPE', 'BUILT_FORM', 'MAIN_FUEL',
    'TOTAL_FLOOR_AREA', 'ENERGY_CONSUMPTION_CURRENT',
    'CONSTRUCTION_AGE_BAND', 'WALLS_ENERGY_EFF',
    'WINDOWS_ENERGY_EFF', 
    'HOT_WATER_ENERGY_EFF',
    'LIGHTING_ENERGY_EFF', 'MAINS_GAS_FLAG',
    
]
X = df[features]
y = df['CO2_EMISSIONS_CURRENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build pipeline
categorical_cols = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')
pipeline = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100))
])
# Train model
pipeline.fit(X_train, y_train)
# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/xgb_pipeline.pkl')
# Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

