# src/model.py

"""
Model training script using XGBoost and pipeline.

This script handles:
- Data preprocessing with ColumnTransformer
- Model fitting using XGBoost
- Saving the trained pipeline
"""
import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Add root directory to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import load_processed_data
from src.clean import clean_data
from src.feature_engineering import select_features, get_target

MODEL_PATH = "model/xgb_pipeline.pkl"

def build_pipeline(categorical_cols):
    """
    Build the preprocessing and XGBoost pipeline.

    Parameters:
        categorical_cols (list): List of categorical feature names.

    Returns:
        Pipeline: scikit-learn Pipeline object
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100))
    ])
    return pipeline


def train_and_save_model():
    """
    Train the model pipeline and save it to disk.
    """
    print("[INFO] Loading and preparing data...")
    df = load_processed_data()
    df = clean_data(df)
    X = select_features(df)
    y = get_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and fit pipeline
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    pipeline = build_pipeline(categorical_cols)
    pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
