"""Fetch Karachi weather and air-quality data from Open-Meteo."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from config import DATA_DIR, DEFAULT_CITY, DEFAULT_LATITUDE, DEFAULT_LONGITUDE


WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_ARCHIVE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
]

AIR_QUALITY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


def fetch_open_meteo(url: str, params: dict[str, Any]) -> pd.DataFrame:
    """Fetch hourly Open-Meteo data and return it as a DataFrame."""
    query_params = params.copy()
    query_params["hourly"] = ",".join(query_params["hourly"])
    request_url = f"{url}?{urlencode(query_params)}"

    try:
        with urlopen(request_url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Open-Meteo request failed with HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc.reason}") from exc

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response did not contain hourly data: {payload}")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather(start_date: str, end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARIABLES,
        "timezone": "Asia/Karachi",
    }
    return fetch_open_meteo(WEATHER_ARCHIVE_URL, params)


def fetch_air_quality(start_date: str, end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_VARIABLES,
        "timezone": "Asia/Karachi",
    }
    return fetch_open_meteo(AIR_QUALITY_ARCHIVE_URL, params)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=90)

    parser = argparse.ArgumentParser(description="Fetch Karachi AQI project data.")
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--start-date", default=default_start.isoformat())
    parser.add_argument("--end-date", default=default_end.isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = DATA_DIR / "raw"
    weather = fetch_weather(args.start_date, args.end_date, args.latitude, args.longitude)
    air_quality = fetch_air_quality(args.start_date, args.end_date, args.latitude, args.longitude)

    merged = weather.merge(air_quality, on="time", how="inner")
    merged.insert(0, "city", args.city)
    merged.insert(1, "latitude", args.latitude)
    merged.insert(2, "longitude", args.longitude)

    save_csv(weather, raw_dir / "karachi_weather_hourly.csv")
    save_csv(air_quality, raw_dir / "karachi_air_quality_hourly.csv")
    save_csv(merged, raw_dir / "karachi_weather_air_quality_hourly.csv")

    print(f"Saved {len(weather)} weather rows")
    print(f"Saved {len(air_quality)} air-quality rows")
    print(f"Saved {len(merged)} merged rows to {raw_dir / 'karachi_weather_air_quality_hourly.csv'}")


if __name__ == "__main__":
    main()
