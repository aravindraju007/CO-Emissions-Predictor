# src/ingest.py

"""
Ingest module for loading building energy data.

This script contains functions to load raw and processed datasets 
from the `data/` directory.
"""

import pandas as pd
import os

def load_raw_data(file_path: str = "/Users/aravindraju/Documents/git/co2-emissions-predictor/data/raw/energy_buildings.csv") -> pd.DataFrame:
    """
    Load the raw dataset from the specified CSV file.

    Parameters:
        file_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded raw data with shape: {df.shape}")
    return df


def load_processed_data(file_path: str = "/Users/aravindraju/Documents/git/co2-emissions-predictor/data/processed/co2_cleaned.csv") -> pd.DataFrame:
    """
    Load the cleaned/processed dataset.

    Parameters:
        file_path (str): Path to the processed CSV file.

    Returns:
        pd.DataFrame: Loaded cleaned data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded processed data with shape: {df.shape}")
    return df


if __name__ == "__main__":
    # For standalone testing
    raw_df = load_raw_data()
    processed_df = load_processed_data()
