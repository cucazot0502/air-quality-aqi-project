from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.interpolate import make_interp_spline

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CHARTS = BASE / "charts"
MODELS = BASE / "models"
DATA.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RAW = DATA / "air_pollution_realistic_ml_dataset.csv"
CLEAN = DATA / "air_pollution_cleaned.csv"
SUMMARY = DATA / "air_pollution_country_month_summary.csv"
CLEANING_SUMMARY = DATA / "cleaning_summary.csv"
OUTLIER_SUMMARY = DATA / "outlier_summary.csv"
FEATURE_IMPORTANCE = DATA / "feature_importance.csv"
MODEL_FILE = MODELS / "aqi_model.pkl"

NUM_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "Temperature", "Humidity", "WindSpeed"]
FEATURE_NUM_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
CAT_COLS = ["Country", "City", "Status", "Station_ID"]

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"Saved chart: charts/{name}")

def clean_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace("_err", "", regex=False).replace(["nan", "NaN", "None", ""], np.nan),
        errors="coerce"
    )

def cap_iqr(df, cols):
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[c] < lo) | (df[c] > hi)).sum()
        df[c] = df[c].clip(lo, hi)
        rows.append([c, round(lo, 4), round(hi, 4), int(outliers)])
    return pd.DataFrame(rows, columns=["column", "lower_bound", "upper_bound", "outliers_capped"])

# ============================================================
# 1. LOAD DATA
# ============================================================

section("1. LOAD DATA")
if not RAW.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW}")

df = pd.read_csv(RAW)

print(f"Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nUnique countries:", sorted(df["Country"].dropna().unique()) if "Country" in df else "N/A")
print("Unique cities:", df["City"].nunique(dropna=True) if "City" in df else "N/A")
print("Unique status values:", sorted(df["Status"].dropna().unique()) if "Status" in df else "N/A")

# ============================================================
# 2. RAW DATA QUALITY CHECK
# ============================================================

section("2. RAW DATA QUALITY CHECK")
before_rows = len(df)
before_missing = int(df.isna().sum().sum())
before_dup = int(df.duplicated().sum())

print(f"Rows before cleaning: {before_rows:,}")
print(f"Duplicate rows before cleaning: {before_dup:,}")
print(f"Total missing values before cleaning: {before_missing:,}")
print("\nMissing values by column:")
print(df.isna().sum().sort_values(ascending=False))

# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

section("3. REMOVE DUPLICATES")
df = df.drop_duplicates().copy()

print(f"Rows after removing duplicates: {len(df):,}")
print(f"Duplicate rows removed: {before_rows - len(df):,}")

# ============================================================
# 4. CONVERT NUMERIC + DATE COLUMNS
# ============================================================

section("4. CONVERT NUMERIC + DATE COLUMNS")
conversion_report = []

for c in NUM_COLS:
    before_na = df[c].isna().sum()
    df[c] = clean_num(df[c])
    after_na = df[c].isna().sum()
    conversion_report.append([c, int(before_na), int(after_na), int(after_na - before_na)])

conversion_report = pd.DataFrame(
    conversion_report,
    columns=["column", "missing_before_conversion", "missing_after_conversion", "new_invalid_values_found"]
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
invalid_dates = int(df["Date"].isna().sum())
df = df.dropna(subset=["Date"]).copy()

print("Numeric conversion report:")
print(conversion_report)
print(f"Invalid/unparseable dates removed: {invalid_dates}")

# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

section("5. HANDLE MISSING VALUES")

for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

for c in NUM_COLS:
    df[c] = df.groupby("City")[c].transform(lambda x: x.fillna(x.median()))
    df[c] = df[c].fillna(df[c].median())

after_missing_fill = int(df.isna().sum().sum())

print(f"Total missing values after filling: {after_missing_fill:,}")
print("\nMissing values after filling:")
print(df.isna().sum().sort_values(ascending=False))

# ============================================================
# 6. HANDLE OUTLIERS BY IQR CAPPING
# ============================================================

section("6. HANDLE OUTLIERS BY IQR CAPPING")

df_before_outlier_capping = df.copy()

outlier_report = cap_iqr(df, NUM_COLS)
outlier_report.to_csv(OUTLIER_SUMMARY, index=False)

print("Outlier capping summary:")
print(outlier_report)
print(f"Saved outlier summary: {OUTLIER_SUMMARY}")

boxplot_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_before_outlier_capping[boxplot_cols])
plt.title("Box Plot Before Outlier Capping")
plt.xticks(rotation=45)
savefig("boxplot_before_outlier_capping.png")

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[boxplot_cols])
plt.title("Box Plot After Outlier Capping")
plt.xticks(rotation=45)
savefig("boxplot_after_outlier_capping.png")

print("Saved boxplot charts:")
print("- charts/boxplot_before_outlier_capping.png")
print("- charts/boxplot_after_outlier_capping.png")

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

section("7. FEATURE ENGINEERING")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
df["Quarter"] = df["Date"].dt.quarter
df["High_AQI_Flag"] = (df["AQI"] > 100).astype(int)

pollution_norm = df[POLLUTANTS].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df["Pollution_Load"] = pollution_norm.mean(axis=1)

print("Created features:")
print(["Year", "Month", "Month_Name", "Quarter", "High_AQI_Flag", "Pollution_Load"])
print("\nFeature preview:")
print(df[["Country", "City", "Date", "AQI", "Year", "Month", "Month_Name", "Quarter", "High_AQI_Flag", "Pollution_Load"]].head())

# ============================================================
# 8. SAVE CLEANED DATASETS
# ============================================================

section("8. SAVE CLEANED DATASETS")

after_rows = len(df)
after_missing = int(df.isna().sum().sum())
after_dup = int(df.duplicated().sum())

cleaning_summary = pd.DataFrame({
    "metric": ["rows", "duplicates", "missing_values"],
    "before": [before_rows, before_dup, before_missing],
    "after": [after_rows, after_dup, after_missing]
})

country_month = (
    df.groupby(["Country", "Year", "Month", "Month_Name"], as_index=False)
      .agg(
          avg_aqi=("AQI", "mean"),
          avg_pm25=("PM2.5", "mean"),
          avg_pm10=("PM10", "mean"),
          record_count=("AQI", "size")
      )
      .sort_values(["Country", "Year", "Month"])
)

df.to_csv(CLEAN, index=False)
country_month.to_csv(SUMMARY, index=False)
cleaning_summary.to_csv(CLEANING_SUMMARY, index=False)

print("Cleaning summary:")
print(cleaning_summary)
print(f"\nSaved cleaned dataset: {CLEAN}")
print(f"Saved country-month summary: {SUMMARY}")
print(f"Saved cleaning summary: {CLEANING_SUMMARY}")

# ============================================================
# 9. DESCRIPTIVE RESULTS FOR REPORT
# ============================================================

section("9. DESCRIPTIVE RESULTS FOR REPORT")

print("Average AQI by country:")
print(df.groupby("Country")["AQI"].mean().sort_values(ascending=False).round(2))

print("\nAverage AQI by month:")
print(df.groupby("Month")["AQI"].mean().round(2))

print("\nTop 10 cities by average AQI:")
print(df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10).round(2))

print("\nStatus distribution:")
print(df["Status"].value_counts())

# ============================================================
# 10. SAVE CHARTS
# ============================================================

section("10. SAVE CHARTS")

sns.set_style("whitegrid")

plt.figure(figsize=(9, 5))
sns.barplot(
    data=df.groupby("Country", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False),
    x="Country", y="AQI"
)
plt.title("Average AQI by Country")
savefig("average_aqi_by_country.png")

plt.figure(figsize=(9, 5))
monthly = df.groupby("Month", as_index=False)["AQI"].mean()
sns.lineplot(data=monthly, x="Month", y="AQI", marker="o")
plt.title("Monthly Average AQI")
plt.xticks(range(1, 13))
savefig("monthly_average_aqi.png")

plt.figure(figsize=(11, 6))
for country in country_month["Country"].unique():
    temp = country_month[country_month["Country"] == country].sort_values("Month")
    x = temp["Month"].values
    y = temp["avg_aqi"].values
    if len(x) >= 4:
        x_smooth = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, label=country)
        plt.scatter(x, y, s=35)
    else:
        plt.plot(x, y, marker="o", label=country)
plt.title("Average AQI by Country and Month")
plt.xlabel("Month")
plt.ylabel("avg_aqi")
plt.xticks(range(1, 13))
plt.legend(title="Country")
plt.grid(True, alpha=0.3)
savefig("aqi_by_country_month.png")

plt.figure(figsize=(8, 5))
sns.histplot(df["AQI"], bins=30, kde=True)
plt.title("AQI Distribution")
savefig("aqi_distribution.png")

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Status", order=df["Status"].value_counts().index)
plt.title("Status Distribution")
plt.xticks(rotation=25, ha="right")
savefig("status_distribution.png")

plt.figure(figsize=(10, 7))
sns.heatmap(df[NUM_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
savefig("correlation_heatmap.png")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="PM2.5", y="AQI", hue="Country", alpha=0.55)
plt.title("PM2.5 vs AQI")
savefig("pm25_vs_aqi.png")

plt.figure(figsize=(11, 5))
top_city = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False).head(10)
sns.barplot(data=top_city, x="City", y="AQI")
plt.title("Top 10 Cities by Average AQI")
plt.xticks(rotation=25, ha="right")
savefig("top10_city_aqi.png")

plt.close("all")

# ============================================================
# 11. MACHINE LEARNING - AQI PREDICTION
# ============================================================

section("11. MACHINE LEARNING - AQI PREDICTION")

ml_features_num = FEATURE_NUM_COLS + ["Month", "Quarter", "High_AQI_Flag", "Pollution_Load"]
ml_features_cat = ["Country", "City", "Status"]
ml_target = "AQI"

ml_df = df[ml_features_num + ml_features_cat + [ml_target]].copy()
ml_df = ml_df.dropna()

X = ml_df[ml_features_num + ml_features_cat]
y = ml_df[ml_target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ml_features_cat),
        ("num", "passthrough", ml_features_num)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train/Test Shape:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nModel Evaluation:")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")

print("\nActual vs Predicted AQI sample:")
sample_result = pd.DataFrame({
    "Actual_AQI": y_test.head(10).values,
    "Predicted_AQI": y_test_pred[:10]
})
print(sample_result.round(2))

# Feature importance
rf = model.named_steps["model"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(FEATURE_IMPORTANCE, index=False)
print("\nTop 15 Feature Importances:")
print(importance.head(15))
print(f"Saved feature importance: {FEATURE_IMPORTANCE}")

# Save model bundle
model_bundle = {
    "model": model,
    "feature_cols": ml_features_num + ml_features_cat,
    "num_cols": ml_features_num,
    "cat_cols": ml_features_cat,
    "target": ml_target,
    "metrics": {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2)
    }
}

joblib.dump(model_bundle, MODEL_FILE)
print(f"Saved AQI model: {MODEL_FILE}")

# ============================================================
# 12. FINAL TERMINAL SUMMARY
# ============================================================

section("12. FINAL TERMINAL SUMMARY")

print("Training ML completed successfully.")
print(f"Input rows: {before_rows:,}")
print(f"Final rows: {after_rows:,}")
print(f"Duplicates removed: {before_rows - after_rows:,}")
print(f"Final missing values: {after_missing:,}")
print(f"Cleaned dataset: {CLEAN}")
print(f"Country-month summary: {SUMMARY}")
print(f"Charts folder: {CHARTS}")
print(f"AQI model: {MODEL_FILE}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2: {test_r2:.4f}")
