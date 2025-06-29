# src/evaluate.py

"""
Model evaluation module for CO₂ Emissions Predictor.

This script:
- Loads the saved model
- Evaluates it using standard regression metrics
- Displays results
"""

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.ingest import load_processed_data
from src.clean import clean_data
from src.feature_engineering import select_features, get_target
import numpy as np

MODEL_PATH = "model/xgb_pipeline.pkl"

def evaluate_model():
    """
    Load the trained model and evaluate it on test data.
    """
    print("[INFO] Loading and preparing data...")
    df = load_processed_data()
    df = clean_data(df)
    X = select_features(df)
    y = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("[INFO] Making predictions...")
    y_pred = model.predict(X_test)

    print("\n📊 Evaluation Metrics:")
    print(f"MAE  (Mean Absolute Error)     : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²   (R-squared)               : {r2_score(y_test, y_pred):.2f}")


if __name__ == "__main__":
    evaluate_model()
