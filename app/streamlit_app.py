# app/streamlit_app.py

# app/streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd
import joblib

# Add root directory to path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clean import clean_data
from src.feature_engineering import select_features


MODEL_PATH = "model/xgb_pipeline.pkl"

# Page configuration
st.set_page_config(page_title="CO₂ Emissions Predictor", layout="centered")
st.title("🌍 CO₂ Emissions Predictor")
st.write("Upload your building energy CSV data to predict current CO₂ emissions (kgCO₂/m²).")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Please train the model first.")
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if uploaded_file and pipeline:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(" File uploaded successfully.")

        st.write("### Raw Data Preview")
        st.dataframe(df_raw.head())

        df_clean = clean_data(df_raw)
        df_features = select_features(df_clean)

        st.write("### Cleaned & Transformed Data Preview")
        st.dataframe(df_features.head())

        predictions = pipeline.predict(df_features)
        df_result = df_features.copy()
        df_result["Predicted_CO2_Emissions"] = predictions

        st.write("###  Prediction Results")
        st.dataframe(df_result.head())

        # Download predictions
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predicted_emissions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
elif not uploaded_file:
    st.info("Please upload a CSV file to get started.")

