# src/clean.py

"""
Data cleaning module for CO₂ Emissions Predictor.

This script handles:
- Dropping irrelevant columns
- Handling missing values
- Converting data types
- Outputting cleaned DataFrame
"""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw EPC dataset for modeling.

    Steps:
    - Drop high-cardinality or irrelevant columns
    - Handle missing values
    - Convert dates and categorical flags

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for feature engineering.
    """
    # Drop ID/address-related columns
    drop_cols = [
        'LMK_KEY', 'UPRN', 'UPRN_SOURCE', 'ADDRESS1', 'ADDRESS2', 'ADDRESS3',
        'POSTCODE', 'ADDRESS', 'BUILDING_REFERENCE_NUMBER'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Drop datetime duplicates
    df = df.drop(columns=['LODGEMENT_DATETIME'], errors='ignore')

    # Convert date columns to datetime (if useful for time analysis)
    if 'INSPECTION_DATE' in df.columns:
        df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'], errors='coerce')
    if 'LODGEMENT_DATE' in df.columns:
        df['LODGEMENT_DATE'] = pd.to_datetime(df['LODGEMENT_DATE'], errors='coerce')

    # Replace "NO DATA!" and similar invalids with NaN
    df.replace("NO DATA!", pd.NA, inplace=True)

    # Drop rows where target is missing
    df = df.dropna(subset=['CO2_EMISSIONS_CURRENT'])

    # Drop columns with too many missing values (>90%)
    threshold = len(df) * 0.9
    df = df.dropna(axis=1, thresh=threshold)

    # Fill remaining missing values with sensible defaults
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include='object').columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df


if __name__ == "__main__":
    from ingest import load_raw_data
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    print(f"[INFO] Cleaned data shape: {cleaned_df.shape}")
    cleaned_df.to_csv("/Users/aravindraju/Documents/git/co2-emissions-predictor/data/processed/co2_cleaned.csv", index=False)
    print("[INFO] Saved cleaned dataset to data/processed/co2_cleaned.csv")

