"""Build, materialize, and verify the local Feast feature store."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from config import DATA_DIR, PROJECT_ROOT


FEATURE_REPO = PROJECT_ROOT / "feature_repo"
FEATURE_DATA_DIR = FEATURE_REPO / "data"
RUNTIME_REPO = (
    Path(tempfile.gettempdir()) / "pearls_aqi_feature_repo"
    if os.name == "nt" and "OneDrive" in str(PROJECT_ROOT)
    else FEATURE_REPO
)
RUNTIME_DATA_DIR = RUNTIME_REPO / "data"
SOURCE_PATH = DATA_DIR / "processed" / "karachi_aqi_features.csv"
OFFLINE_STORE_PATH = FEATURE_DATA_DIR / "karachi_aqi_features.parquet"
REGISTRY_PATH = RUNTIME_DATA_DIR / "registry.db"
ONLINE_STORE_PATH = RUNTIME_DATA_DIR / "online_store.db"
DEMO_OUTPUT_PATH = DATA_DIR / "processed" / "feature_store_demo.json"

FEATURE_COLUMNS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "pressure_msl",
    "us_aqi_lag_24h",
    "us_aqi_lag_48h",
    "us_aqi_lag_72h",
    "us_aqi_rolling_mean_24h",
    "pm2_5_rolling_mean_24h",
    "aqi_change_24h",
]


def configure_feast_runtime() -> None:
    """Keep Feast's Prometheus files in a writable project directory."""
    metrics_dir = PROJECT_ROOT / ".feast_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))


def prepare_offline_store(source_path: Path = SOURCE_PATH) -> pd.DataFrame:
    """Convert model features into Feast's entity/timestamp format."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Run 'python src/features.py' first."
        )

    source_df = pd.read_csv(source_path)
    required = {"time", *FEATURE_COLUMNS}
    missing = sorted(required.difference(source_df.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {', '.join(missing)}")

    feast_df = source_df.loc[:, ["time", *FEATURE_COLUMNS]].copy()
    feast_df["city_id"] = "karachi"
    feast_df["event_timestamp"] = pd.to_datetime(feast_df.pop("time"), utc=True)
    feast_df["created_timestamp"] = feast_df["event_timestamp"]
    feast_df = feast_df[
        ["city_id", "event_timestamp", "created_timestamp", *FEATURE_COLUMNS]
    ].sort_values("event_timestamp")

    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    feast_df.to_parquet(OFFLINE_STORE_PATH, index=False)
    return feast_df


def build_and_verify_store(feast_df: pd.DataFrame) -> dict:
    """Register definitions, materialize features, and run sample lookups."""
    configure_feast_runtime()

    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    sys.path.insert(0, str(FEATURE_REPO))
    from aqi_features import (
        aqi_forecast_service,
        karachi,
        karachi_aqi_hourly,
    )

    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    runtime_offline_path = RUNTIME_DATA_DIR / OFFLINE_STORE_PATH.name
    if runtime_offline_path.resolve() != OFFLINE_STORE_PATH.resolve():
        shutil.copy2(OFFLINE_STORE_PATH, runtime_offline_path)

    config = RepoConfig(
        project="pearls_aqi_predictor",
        provider="local",
        registry=os.path.relpath(REGISTRY_PATH, PROJECT_ROOT),
        online_store={"type": "sqlite", "path": "data/online_store.db"},
        offline_store={"type": "file"},
        repo_path=RUNTIME_REPO,
        entity_key_serialization_version=3,
    )
    store = FeatureStore(config=config)
    store.apply([karachi, karachi_aqi_hourly, aqi_forecast_service])

    start_time = feast_df["event_timestamp"].min().to_pydatetime()
    end_time = (
        feast_df["event_timestamp"].max() + pd.Timedelta(hours=1)
    ).to_pydatetime()
    store.materialize(start_date=start_time, end_date=end_time)

    feature_refs = [f"karachi_aqi_hourly:{column}" for column in FEATURE_COLUMNS]
    latest_time = feast_df["event_timestamp"].max()

    historical_df = store.get_historical_features(
        entity_df=pd.DataFrame(
            {"city_id": ["karachi"], "event_timestamp": [latest_time]}
        ),
        features=feature_refs,
    ).to_df()

    online_result = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"city_id": "karachi"}],
    ).to_dict()

    return {
        "project": "pearls_aqi_predictor",
        "entity": "karachi",
        "rows_in_offline_store": len(feast_df),
        "feature_count": len(FEATURE_COLUMNS),
        "latest_event_timestamp": latest_time.isoformat(),
        "historical_lookup": historical_df.iloc[0].to_dict(),
        "online_lookup": {
            key: values[0] if isinstance(values, list) else values
            for key, values in online_result.items()
        },
    }


def publish_runtime_artifacts() -> None:
    """Copy Windows temp databases back into the ignored artifact directory."""
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source in [REGISTRY_PATH, ONLINE_STORE_PATH]:
        destination = FEATURE_DATA_DIR / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def make_json_serializable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and verify the local Feast AQI feature store."
    )
    parser.add_argument("--input", type=Path, default=SOURCE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feast_df = prepare_offline_store(args.input)
    result = build_and_verify_store(feast_df)
    publish_runtime_artifacts()

    DEMO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, default=make_json_serializable),
        encoding="utf-8",
    )

    print(f"Offline store: {OFFLINE_STORE_PATH}")
    print(f"Rows: {result['rows_in_offline_store']}")
    print(f"Registered features: {result['feature_count']}")
    print(f"Latest event: {result['latest_event_timestamp']}")
    print(f"Online AQI: {result['online_lookup']['us_aqi']}")
    print(f"Verification output: {DEMO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
