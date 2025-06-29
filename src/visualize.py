# src/visualize.py

"""
Visualization utilities for CO₂ Emissions Predictor.

This script includes:
- Feature importance plot
- Prediction vs Actual scatter plot
- Residuals histogram
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from src.ingest import load_processed_data
from src.clean import clean_data
from src.feature_engineering import select_features, get_target

MODEL_PATH = "model/xgb_pipeline.pkl"

def plot_feature_importance():
    """
    Plot feature importances from trained XGBoost model.
    """
    df = load_processed_data()
    df = clean_data(df)
    X = select_features(df)
    y = get_target(df)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = joblib.load(MODEL_PATH)
    model = pipeline.named_steps['xgb']

    # Get feature names from OneHotEncoder
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    ohe_features = encoder.get_feature_names_out(categorical_cols)
    numeric_features = X.select_dtypes(exclude='object').columns.tolist()
    feature_names = list(ohe_features) + numeric_features

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()


def plot_predictions():
    """
    Plot actual vs predicted values.
    """
    df = load_processed_data()
    df = clean_data(df)
    X = select_features(df)
    y = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = joblib.load(MODEL_PATH)

    y_pred = pipeline.predict(X_test)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual CO₂ Emissions")
    plt.ylabel("Predicted CO₂ Emissions")
    plt.title("Actual vs Predicted CO₂ Emissions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()


def plot_residuals():
    """
    Plot histogram of residuals.
    """
    df = load_processed_data()
    df = clean_data(df)
    X = select_features(df)
    y = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Error (Actual - Predicted)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_feature_importance()
    plot_predictions()
    plot_residuals()
