"""Create model features and forecasting targets from raw AQI data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR


RAW_DATA_PATH = DATA_DIR / "raw" / "karachi_weather_air_quality_hourly.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "karachi_aqi_features.csv"

POLLUTANT_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
]

LAG_HOURS = [1, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOWS = [3, 6, 12, 24, 48, 72]
FORECAST_HORIZONS = [24, 48, 72]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day_of_month"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_base_columns = ["us_aqi", *POLLUTANT_COLUMNS, *WEATHER_COLUMNS]
    lag_features = {}
    for column in lag_base_columns:
        for hours in LAG_HOURS:
            lag_features[f"{column}_lag_{hours}h"] = df[column].shift(hours)
    return pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_base_columns = ["us_aqi", "pm2_5", "pm10", "temperature_2m", "wind_speed_10m"]
    rolling_features = {}
    for column in rolling_base_columns:
        for window in ROLLING_WINDOWS:
            shifted = df[column].shift(1)
            rolling_features[f"{column}_rolling_mean_{window}h"] = shifted.rolling(window=window).mean()
            rolling_features[f"{column}_rolling_std_{window}h"] = shifted.rolling(window=window).std()
    return pd.concat([df, pd.DataFrame(rolling_features, index=df.index)], axis=1)


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    change_features = {
        "aqi_change_1h": df["us_aqi"] - df["us_aqi_lag_1h"],
        "aqi_change_24h": df["us_aqi"] - df["us_aqi_lag_24h"],
        "pm2_5_change_1h": df["pm2_5"] - df["pm2_5_lag_1h"],
        "pm10_change_1h": df["pm10"] - df["pm10_lag_1h"],
    }
    return pd.concat([df, pd.DataFrame(change_features, index=df.index)], axis=1)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    targets = {}
    for horizon in FORECAST_HORIZONS:
        targets[f"target_aqi_{horizon}h"] = df["us_aqi"].shift(-horizon)
    return pd.concat([df, pd.DataFrame(targets, index=df.index)], axis=1)


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = add_targets(df)

    df = df.dropna().reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AQI model features.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(args.input)
    features_df = build_features(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)

    target_columns = [f"target_aqi_{horizon}h" for horizon in FORECAST_HORIZONS]
    print(f"Loaded {len(raw_df)} raw rows from {args.input}")
    print(f"Saved {len(features_df)} feature rows to {args.output}")
    print(f"Created {len(features_df.columns)} columns")
    print(f"Targets: {', '.join(target_columns)}")


if __name__ == "__main__":
    main()
