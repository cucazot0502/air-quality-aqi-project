CREATE DATABASE IF NOT EXISTS air_quality_project;
USE air_quality_project;

DROP TABLE IF EXISTS air_quality_records;
DROP TABLE IF EXISTS country_month_aqi_summary;
DROP TABLE IF EXISTS stations;
DROP TABLE IF EXISTS cities;
DROP TABLE IF EXISTS air_quality_status;
DROP TABLE IF EXISTS countries;

CREATE TABLE countries (
    country_id INT PRIMARY KEY AUTO_INCREMENT,
    country_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE cities (
    city_id INT PRIMARY KEY AUTO_INCREMENT,
    city_name VARCHAR(50) NOT NULL,
    country_id INT NOT NULL,
    CONSTRAINT fk_city_country
        FOREIGN KEY (country_id)
        REFERENCES countries(country_id)
);

CREATE TABLE stations (
    station_id INT PRIMARY KEY AUTO_INCREMENT,
    station_code VARCHAR(50) NOT NULL,
    city_id INT NOT NULL,
    CONSTRAINT fk_station_city
        FOREIGN KEY (city_id)
        REFERENCES cities(city_id),
    CONSTRAINT uq_station_code_city
        UNIQUE (station_code, city_id)
);

CREATE TABLE air_quality_status (
    status_id INT PRIMARY KEY AUTO_INCREMENT,
    status_name VARCHAR(50) NOT NULL UNIQUE,
    aqi_min INT,
    aqi_max INT
);

CREATE TABLE air_quality_records (
    record_id INT PRIMARY KEY AUTO_INCREMENT,
    station_id INT NOT NULL,
    status_id INT NOT NULL,
    record_date DATE NOT NULL,
    pm25 FLOAT,
    pm10 FLOAT,
    no2 FLOAT,
    so2 FLOAT,
    co FLOAT,
    o3 FLOAT,
    aqi FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    windspeed FLOAT,
    CONSTRAINT fk_record_station
        FOREIGN KEY (station_id)
        REFERENCES stations(station_id),
    CONSTRAINT fk_record_status
        FOREIGN KEY (status_id)
        REFERENCES air_quality_status(status_id)
);

CREATE TABLE country_month_aqi_summary (
    summary_id INT PRIMARY KEY AUTO_INCREMENT,
    country_id INT NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    avg_aqi FLOAT,
    avg_pm25 FLOAT,
    avg_pm10 FLOAT,
    record_count INT,
    CONSTRAINT fk_summary_country
        FOREIGN KEY (country_id)
        REFERENCES countries(country_id)
);