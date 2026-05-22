from pathlib import Path
import pandas as pd
import mysql.connector as mysql

CFG = dict(host="127.0.0.1", port=3306, user="root", password="1234", database="air_quality_project")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CSV = next((p for p in [DATA/"air_pollution_cleaned.csv", DATA/"cleaned_air_pollution_dataset.csv"] if p.exists()), None)
if CSV is None:
    raise FileNotFoundError(f"Không tìm thấy file cleaned data trong: {DATA}")

df = pd.read_csv(CSV).rename(columns={
    "Country":"country", "City":"city", "PM2.5":"pm25", "PM10":"pm10", "NO2":"no2",
    "SO2":"so2", "CO":"co", "O3":"o3", "AQI":"aqi", "Temperature":"temperature",
    "Humidity":"humidity", "WindSpeed":"windspeed", "Date":"date", "Station_ID":"station_code",
    "Status":"status", "Year":"year", "Month":"month", "Day":"day"
})

for c in ["pm25","pm10","no2","so2","co","o3","aqi","temperature","humidity","windspeed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if "date" in df:
    df["record_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
elif {"year","month","day"}.issubset(df.columns):
    df["record_date"] = pd.to_datetime(df[["year","month","day"]], errors="coerce").dt.date
else:
    df["record_date"] = pd.to_datetime(df[["year","month"]].assign(day=1), errors="coerce").dt.date

df = df.dropna(subset=["country","city","station_code","status","record_date","aqi"]).copy()
df["station_code"] = df["station_code"].astype(str)
df["country"] = df["country"].astype(str).str.strip()
df["city"] = df["city"].astype(str).str.strip()
df["status"] = df["status"].astype(str).str.strip()

cn = mysql.connect(**CFG)
cur = cn.cursor(dictionary=True)

cur.execute("SET FOREIGN_KEY_CHECKS=0")
for t in ["country_month_aqi_summary","air_quality_records","stations","cities","countries","air_quality_status"]:
    cur.execute(f"TRUNCATE TABLE {t}")

# Nếu stations.station_code đang bị UNIQUE, bỏ UNIQUE đó vì cùng mã station có thể xuất hiện ở nhiều city trong dataset.
try:
    cur.execute("ALTER TABLE stations DROP INDEX station_code")
except Exception:
    pass

# Chỉ cần unique theo cặp station_code + city_id.
try:
    cur.execute("ALTER TABLE stations ADD UNIQUE KEY uq_station_code_city (station_code, city_id)")
except Exception:
    pass

cur.execute("SET FOREIGN_KEY_CHECKS=1")

status_rows = [
    ("Good",0,50), ("Moderate",51,100), ("Unhealthy for Sensitive Groups",101,150),
    ("Unhealthy",151,200), ("Very Unhealthy",201,300), ("Hazardous",301,500)
]
cur.executemany("INSERT INTO air_quality_status(status_name,aqi_min,aqi_max) VALUES(%s,%s,%s)", status_rows)

cur.executemany("INSERT INTO countries(country_name) VALUES(%s)", [(x,) for x in sorted(df.country.unique())])
cur.execute("SELECT country_id,country_name FROM countries")
cid = {r["country_name"]: r["country_id"] for r in cur.fetchall()}

cities = df[["country","city"]].drop_duplicates()
cur.executemany("INSERT INTO cities(city_name,country_id) VALUES(%s,%s)", [(r.city, cid[r.country]) for r in cities.itertuples()])
cur.execute("SELECT ci.city_id,ci.city_name,c.country_name FROM cities ci JOIN countries c USING(country_id)")
city_id = {(r["country_name"], r["city_name"]): r["city_id"] for r in cur.fetchall()}

stations = df[["country","city","station_code"]].drop_duplicates()
station_rows = [(r.station_code, city_id[(r.country, r.city)]) for r in stations.itertuples()]
cur.executemany("INSERT INTO stations(station_code,city_id) VALUES(%s,%s)", station_rows)

cur.execute("SELECT station_id,station_code,city_id FROM stations")
sid = {(r["station_code"], r["city_id"]): r["station_id"] for r in cur.fetchall()}

cur.execute("SELECT status_id,status_name FROM air_quality_status")
stid = {r["status_name"]: r["status_id"] for r in cur.fetchall()}

records = []
for r in df.itertuples():
    this_city_id = city_id[(r.country, r.city)]
    this_station_id = sid[(r.station_code, this_city_id)]
    records.append((
        this_station_id, stid[r.status], r.record_date, r.pm25, r.pm10, r.no2, r.so2,
        r.co, r.o3, r.aqi, r.temperature, r.humidity, r.windspeed
    ))

cur.executemany("""
    INSERT INTO air_quality_records
    (station_id,status_id,record_date,pm25,pm10,no2,so2,co,o3,aqi,temperature,humidity,windspeed)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", records)

summary = (
    df.groupby(["country","year","month"], as_index=False)
      .agg(avg_aqi=("aqi","mean"), avg_pm25=("pm25","mean"), avg_pm10=("pm10","mean"), record_count=("aqi","size"))
)
cur.executemany(
    "INSERT INTO country_month_aqi_summary(country_id,year,month,avg_aqi,avg_pm25,avg_pm10,record_count) VALUES(%s,%s,%s,%s,%s,%s,%s)",
    [(cid[r.country], int(r.year), int(r.month), r.avg_aqi, r.avg_pm25, r.avg_pm10, int(r.record_count)) for r in summary.itertuples()]
)

cn.commit()
cur.close()
cn.close()

print(f"Done: inserted {len(df)} air quality records from {CSV.name}.")
print("Station duplicate issue fixed by using unique key: (station_code, city_id).")
