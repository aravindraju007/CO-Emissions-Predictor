
# 🌍 CO₂ Emissions Predictor

A data science project that predicts **CO₂ emissions (kgCO₂/m²)** from UK building energy data using **XGBoost regression**. This project includes an end-to-end machine learning pipeline, from data ingestion and cleaning to modeling, evaluation, and deployment via a Streamlit dashboard.

---

## 🧠 Objective

To develop a regression-based machine learning model that predicts the **current CO₂ emissions** of buildings using publicly available energy performance data.



## 📦 Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/co2-emissions-predictor.git
cd co2-emissions-predictor
```
### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
## 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
Or 
conda env create -f environment.yml
conda activate co2-predictor-env
```
### 🚀 How to Use
🔧 Train the Model

```bash
python src/model.py
```
### 📊 Evaluate the Model

```bash
python src/evaluate.py
```
## 📈 Visualize Results

```bash
python src/visualize.py
```
### 🌐 Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```
### 📊 Sample Metrics
Metric Value (example)
MAE 4.58
RMSE 6.92
R² 0.82


### 🛠 Features Used
Building geometry: TOTAL_FLOOR_AREA, PROPERTY_TYPE, etc.

Energy systems: MAIN_FUEL, WALLS_ENERGY_EFF, WINDOWS_ENERGY_EFF, etc.

Heating and lighting systems

Energy consumption statistics


### 🧪 Sample Prediction Flow (Streamlit)
Upload your .csv file. 

App displays raw + cleaned data.

Predicts and displays CO₂ emissions.

Allows you to download the result.


### 👤 Author
Aravind Raju,
Data Analyst & Aspiring Data Scientist,
📧 aravindraju007@gmail.com,
🔗 LinkedIn: www.linkedin.com/in/arraju-wk796
