from pathlib import Path
import pandas as pd, numpy as np, streamlit as st, plotly.express as px, joblib

st.set_page_config(page_title="Air Quality AQI Dashboard", page_icon="🌫️", layout="wide")
BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; MODELS=BASE/"models"
DATA_FILES = [
    DATA / "air_pollution_cleaned.csv",
    DATA / "cleaned_air_pollution_dataset.csv",
    DATA / "air_pollution_realistic_ml_dataset.csv",
    BASE / "air_pollution_cleaned.csv",
    BASE / "cleaned_air_pollution_dataset.csv",
]
MODEL_FILES=[MODELS/"aqi_model.pkl",BASE/"aqi_model.pkl"]
NUM=["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed"]

def pick(files): return next((p for p in files if p.exists()), None)
def clean_num(s): return pd.to_numeric(s.astype(str).str.replace("_err","",regex=False).replace(["nan","NaN","None",""],np.nan), errors="coerce")

@st.cache_data
def load_data():
    p=pick(DATA_FILES)
    if not p: return None,None
    df=pd.read_csv(p).rename(columns={"PM25":"PM2.5","pm25":"PM2.5","pm10":"PM10","no2":"NO2","so2":"SO2","co":"CO","o3":"O3","aqi":"AQI","temperature":"Temperature","humidity":"Humidity","windspeed":"WindSpeed","country":"Country","city":"City","date":"Date","status":"Status"})
    for c in NUM+["AQI"]:
        if c in df: df[c]=clean_num(df[c])
    if "Date" in df:
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        if "Year" not in df: df["Year"]=df["Date"].dt.year
        if "Month" not in df: df["Month"]=df["Date"].dt.month
    return df,p

@st.cache_resource
def load_model():
    p=pick(MODEL_FILES)
    return (joblib.load(p),p) if p else (None,None)

df,path=load_data(); bundle,mpath=load_model()
st.title("🌫️ Air Quality Analysis & AQI Prediction")
if df is None:
    st.error("Put cleaned dataset in data/air_pollution_cleaned.csv or data/cleaned_air_pollution_dataset.csv"); st.stop()

with st.sidebar:
    st.write(f"Dataset: `{path.name}`")
    countries=sorted(df["Country"].dropna().astype(str).unique()) if "Country" in df else []
    sel=st.multiselect("Country", countries, default=countries)
    dff=df[df["Country"].astype(str).isin(sel)].copy() if sel and "Country" in df else df.copy()
    if "Month" in dff:
        months=sorted(dff["Month"].dropna().astype(int).unique())
        msel=st.multiselect("Month", months, default=months)
        dff=dff[dff["Month"].isin(msel)]

tab1,tab2,tab3,tab4=st.tabs(["Overview","AQI Analysis","Pollutants","Prediction"])
with tab1:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows", f"{len(dff):,}"); c2.metric("Countries", dff["Country"].nunique() if "Country" in dff else "N/A")
    c3.metric("Cities", dff["City"].nunique() if "City" in dff else "N/A"); c4.metric("Avg AQI", f"{dff['AQI'].mean():.2f}" if "AQI" in dff else "N/A")
    st.dataframe(dff.head(100), use_container_width=True)
    st.dataframe(pd.DataFrame({"Metric":["Missing values","Duplicate rows"],"Value":[int(dff.isna().sum().sum()),int(dff.duplicated().sum())]}), use_container_width=True)
with tab2:
    if {"Country","AQI"}.issubset(dff.columns):
        st.plotly_chart(px.bar(dff.groupby("Country",as_index=False)["AQI"].mean().sort_values("AQI",ascending=False),x="Country",y="AQI",text_auto=".2f",title="Average AQI by Country"),use_container_width=True)
    if {"Month","AQI"}.issubset(dff.columns):
        st.plotly_chart(px.line(dff.groupby("Month",as_index=False)["AQI"].mean(),x="Month",y="AQI",markers=True,title="Monthly Average AQI"),use_container_width=True)
    if {"Country","Month","AQI"}.issubset(dff.columns):
        st.plotly_chart(px.line(dff.groupby(["Country","Month"],as_index=False)["AQI"].mean(),x="Month",y="AQI",color="Country",markers=True,title="AQI by Country and Month"),use_container_width=True)
    if {"City","AQI"}.issubset(dff.columns):
        st.plotly_chart(px.bar(dff.groupby("City",as_index=False)["AQI"].mean().sort_values("AQI",ascending=False).head(10),x="City",y="AQI",text_auto=".2f",color="AQI",title="Top 10 Cities by Average AQI"),use_container_width=True)
with tab3:
    st.plotly_chart(px.histogram(dff,x="AQI",nbins=40,marginal="box",title="AQI Distribution"),use_container_width=True)
    if "Status" in dff:
        s=dff["Status"].value_counts().reset_index(); s.columns=["Status","Count"]
        st.plotly_chart(px.pie(s,names="Status",values="Count",title="Status Distribution"),use_container_width=True)
    if {"PM2.5","AQI","Country"}.issubset(dff.columns):
        st.plotly_chart(px.scatter(dff,x="PM2.5",y="AQI",color="Country",opacity=.55,title="PM2.5 vs AQI"),use_container_width=True)
    cols=[c for c in NUM+["AQI"] if c in dff]
    if len(cols)>1: st.plotly_chart(px.imshow(dff[cols].corr(),text_auto=".2f",aspect="auto",title="Correlation Heatmap"),use_container_width=True)
    st.subheader("Outlier Handling: Box Plot Before vs After Capping")
    boxplot_before = Path("charts/boxplot_before_outlier_capping.png")
    boxplot_after = Path("charts/boxplot_after_outlier_capping.png")
    col1, col2 = st.columns(2)
    with col1:
        if boxplot_before.exists():
            st.image(str(boxplot_before), caption="Before Outlier Capping", use_container_width=True)
    with col2:
        if boxplot_after.exists():
            st.image(str(boxplot_after), caption="After Outlier Capping", use_container_width=True)
with tab4:
    if not bundle: st.warning("Put aqi_model.pkl in models/ or project root.")
    else:
        m = bundle.get("metrics", {})

        mae = m.get("test_mae", m.get("MAE", 0))
        rmse = m.get("test_rmse", m.get("RMSE", 0))
        r2 = m.get("test_r2", m.get("R2", 0))

        st.write(
            f"Model: `{mpath.name}` | "
            f"MAE: **{mae:.2f}** | "
            f"RMSE: **{rmse:.2f}** | "
            f"R²: **{r2:.3f}**"
        )
        vals={}; cols=st.columns(3)
        for i,c in enumerate(bundle["num_cols"]): vals[c]=cols[i%3].number_input(c,value=float(dff[c].median() if c in dff else 0))
        for c in bundle["cat_cols"]:
            vals[c]=st.selectbox(c, list(range(1,13))) if c=="Month" else st.selectbox(c, sorted(dff[c].dropna().astype(str).unique()) if c in dff else ["Unknown"])
        if st.button("Predict AQI"):
            st.success(f"Predicted AQI: {bundle['model'].predict(pd.DataFrame([vals])[bundle['feature_cols']])[0]:.2f}")
st.download_button("Download filtered data", dff.to_csv(index=False).encode("utf-8"), "filtered_air_pollution_data.csv", "text/csv")
