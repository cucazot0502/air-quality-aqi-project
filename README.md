# Air Quality AQI Analysis & Prediction

## 1. Project Overview

This project analyses air quality data and builds a Streamlit dashboard to visualise AQI patterns by country, city and month. It also includes an optional machine learning model to predict AQI from pollutant indicators, weather factors and location-related features.

Main objective:

> Create a clean, analysis-ready and machine-learning-ready dataset to analyse and approximately predict Air Quality Index (AQI) by country and by month.

---

## 2. Key Features

The dashboard includes:

- Dataset overview
- AQI by country
- Monthly average AQI
- AQI by country and month
- Top 10 cities by average AQI
- AQI distribution
- Air quality status distribution
- PM2.5 vs AQI scatter plot
- Correlation heatmap
- AQI prediction using a trained machine learning model
- Download filtered data as CSV

---

## 3. Project Structure

Recommended folder structure:

```text
Air_Quality_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── air_pollution_realistic_ml_dataset.csv
│   ├── air_pollution_cleaned.csv
│   ├── air_pollution_country_month_summary.csv
│   ├── cleaning_summary.csv
│   └── outlier_summary.csv
│
├── models/
│   └── aqi_model.pkl
│
├── charts/
│   ├── average_aqi_by_country.png
│   ├── monthly_average_aqi.png
│   ├── aqi_by_country_month.png
│   ├── aqi_distribution.png
│   ├── status_distribution.png
│   ├── correlation_heatmap.png
│   ├── pm25_vs_aqi.png
│   ├── top10_city_aqi.png
│   ├── boxplot_before_outlier_capping.png
│   └── boxplot_after_outlier_capping.png
│
├── scripts/
│   ├── cleaning_eda_air_pollution.py
│   └── import_cleaned_to_mysql.py
│
├── sql/
│   ├── create_air_quality_database.sql
│   └── analysis_queries.sql
│
└── database/
    └── air_quality_erd.jpg
```

For Streamlit deployment, the most important files are:

```text
app.py
requirements.txt
data/air_pollution_cleaned.csv
models/aqi_model.pkl
```

---

## 4. Dataset

The cleaned dataset should be placed in the `data/` folder.

The app will try to load one of the following files:

```text
data/air_pollution_cleaned.csv
data/cleaned_air_pollution_dataset.csv
data/air_pollution_realistic_ml_dataset.csv
```

Expected columns include:

```text
Country
City
PM2.5
PM10
NO2
SO2
CO
O3
AQI
Temperature
Humidity
WindSpeed
Date
Station_ID
Status
Year
Month
Month_Name
Quarter
High_AQI_Flag
Pollution_Load
```

---

## 5. Machine Learning Model

The machine learning model should be placed in:

```text
models/aqi_model.pkl
```

The project uses `RandomForestRegressor` to predict AQI.

The model is evaluated using:

- MAE
- RMSE
- R²

If the model file is missing, the dashboard will still run, but the prediction section will show a warning.

---

## 6. Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
streamlit
pandas
numpy
plotly
scikit-learn
joblib
matplotlib
seaborn
scipy
mysql-connector-python
```

---

## 7. Run Locally

From the project root folder, run:

```bash
streamlit run app.py
```

The app should open in your browser at a local Streamlit address.

---

## 8. Deploy to Streamlit Community Cloud

1. Push the full project to GitHub.
2. Go to Streamlit Community Cloud.
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path as:

```text
app.py
```

6. Click **Deploy**.

---

## 9. Important Notes

The MySQL database is used for the database design and SQL analysis part of the project. The deployed Streamlit app does not connect to local MySQL because Streamlit Cloud cannot access your local database at `127.0.0.1`.

The dashboard reads cleaned CSV files and the trained `.pkl` model directly from the project folder.

---

## 10. Team Responsibilities

Suggested division of work:

- Hai: Python data cleaning, EDA, outlier handling, charts and optional ML training
- Dung: MySQL database design, ERD and SQL queries
- Nam: Testing, Streamlit dashboard, report and presentation

---

## 11. How to Update the App

After making changes:

```bash
git add .
git commit -m "Update Streamlit dashboard"
git push
```

Streamlit Community Cloud will automatically rebuild the app after the GitHub repo is updated.
