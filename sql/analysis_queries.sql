USE air_quality_project;

-- 1. AQI trung bình theo quốc gia
SELECT 
    c.country_name,
    AVG(a.aqi) AS avg_aqi
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY c.country_name;

-- 2. AQI trung bình theo tháng
SELECT
    YEAR(record_date) AS year,
    MONTH(record_date) AS month,
    AVG(aqi) AS avg_aqi
FROM air_quality_records
GROUP BY YEAR(record_date), MONTH(record_date)
ORDER BY year, month;

-- 3. AQI trung bình theo quốc gia và tháng
SELECT 
    c.country_name,
    YEAR(a.record_date) AS year,
    MONTH(a.record_date) AS month,
    AVG(a.aqi) AS avg_aqi
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY 
    c.country_name,
    YEAR(a.record_date),
    MONTH(a.record_date)
ORDER BY 
    c.country_name,
    year,
    month;

-- 4. Top 10 thành phố có AQI cao nhất
SELECT
    ci.city_name,
    c.country_name,
    AVG(a.aqi) AS avg_aqi
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY ci.city_name, c.country_name
ORDER BY avg_aqi DESC
LIMIT 10;

-- 5. Số bản ghi theo từng status
SELECT
    st.status_name,
    COUNT(*) AS total_records
FROM air_quality_records a
JOIN air_quality_status st
    ON a.status_id = st.status_id
GROUP BY st.status_name;

-- 6. PM2.5 trung bình theo quốc gia
SELECT 
    c.country_name,
    AVG(a.pm25) AS avg_pm25
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY 
    c.country_name
ORDER BY 
    avg_pm25 DESC;

-- 7. Số bản ghi theo quốc gia
SELECT
    c.country_name,
    COUNT(*) AS total_records
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY 
    c.country_name
ORDER BY 
    total_records DESC;

-- 8. Số bản ghi theo thành phố và quốc gia
SELECT 
    ci.city_name,
    c.country_name,
    COUNT(*) AS total_records
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY 
    ci.city_name,
    c.country_name
ORDER BY 
    total_records DESC;

-- 9. Tháng có AQI cao nhất
SELECT
    YEAR(record_date) AS year,
    MONTH(record_date) AS month,
    AVG(aqi) AS avg_aqi
FROM air_quality_records
GROUP BY YEAR(record_date), MONTH(record_date)
ORDER BY avg_aqi DESC
LIMIT 1;

-- 10. Quốc gia có AQI trung bình cao nhất
SELECT
    c.country_name,
    AVG(a.aqi) AS avg_aqi
FROM air_quality_records a
JOIN stations s
    ON a.station_id = s.station_id
JOIN cities ci
    ON s.city_id = ci.city_id
JOIN countries c
    ON ci.country_id = c.country_id
GROUP BY
    c.country_name
ORDER BY
    avg_aqi DESC
LIMIT 1;