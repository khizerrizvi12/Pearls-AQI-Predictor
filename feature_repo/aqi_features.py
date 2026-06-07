"""Feast feature definitions for Karachi AQI forecasting."""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64


karachi = Entity(
    name="karachi",
    join_keys=["city_id"],
    value_type=ValueType.STRING,
    description="Karachi location entity used by the AQI forecasting system.",
)

aqi_hourly_source = FileSource(
    name="karachi_aqi_hourly_source",
    path="data/karachi_aqi_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

karachi_aqi_hourly = FeatureView(
    name="karachi_aqi_hourly",
    entities=[karachi],
    ttl=timedelta(days=365),
    schema=[
        Field(name="us_aqi", dtype=Float64),
        Field(name="pm2_5", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="temperature_2m", dtype=Float64),
        Field(name="relative_humidity_2m", dtype=Float64),
        Field(name="wind_speed_10m", dtype=Float64),
        Field(name="pressure_msl", dtype=Float64),
        Field(name="us_aqi_lag_24h", dtype=Float64),
        Field(name="us_aqi_lag_48h", dtype=Float64),
        Field(name="us_aqi_lag_72h", dtype=Float64),
        Field(name="us_aqi_rolling_mean_24h", dtype=Float64),
        Field(name="pm2_5_rolling_mean_24h", dtype=Float64),
        Field(name="aqi_change_24h", dtype=Float64),
    ],
    source=aqi_hourly_source,
    online=True,
    description="Hourly AQI, pollutant, weather, lag, and rolling features.",
    tags={"city": "karachi", "domain": "air-quality"},
)

aqi_forecast_service = FeatureService(
    name="aqi_forecast_service",
    features=[karachi_aqi_hourly],
    description="Feature service used by the 24h, 48h, and 72h AQI models.",
)
