# src/feature_engineering.py

"""
Feature engineering module for CO₂ Emissions Predictor.

This script performs:
- Feature selection
- Type conversions
- Encoding preparation (if needed)
"""

import pandas as pd

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select a subset of relevant features for prediction.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: Subset with selected features.
    """
    selected_cols = [
        'PROPERTY_TYPE', 'BUILT_FORM', 'MAIN_FUEL',
        'TOTAL_FLOOR_AREA', 'ENERGY_CONSUMPTION_CURRENT',
        'CONSTRUCTION_AGE_BAND', 'WALLS_ENERGY_EFF',
        'WINDOWS_ENERGY_EFF','HOT_WATER_ENERGY_EFF',
        'LIGHTING_ENERGY_EFF', 'MAINS_GAS_FLAG',
        
    ]
    
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input DataFrame: {missing_cols}")
    
    return df[selected_cols]


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract the target variable (CO2 emissions).

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.Series: Target values.
    """
    if 'CO2_EMISSIONS_CURRENT' not in df.columns:
        raise KeyError("Target column 'CO2_EMISSIONS_CURRENT' not found in DataFrame")
    
    return df['CO2_EMISSIONS_CURRENT']


if __name__ == "__main__":
    from ingest import load_processed_data
    from clean import clean_data

    # Load and process
    df_raw = load_processed_data()
    df_clean = clean_data(df_raw)

    # Select features and target
    X = select_features(df_clean)
    y = get_target(df_clean)

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target variable shape: {y.shape}")
