# CO₂ Emissions Predictor

This project predicts CO₂ emissions from UK building energy data using regression techniques.

## Folder Structure

```
co2-emissions-predictor/
│
├── README.md
├── data/
│   ├── raw/
│   │   └── energy_buildings.csv
│   └── processed/
│       └── co2_cleaned.csv
├── notebooks/
│   └── co2_eda_modeling.ipynb
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluate.py
│   └── visualize.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Objective
Predict CO₂ emissions using building energy data and evaluate model performance with XGBoost.
