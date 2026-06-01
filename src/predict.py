"""Load trained models and generate AQI forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import DATA_DIR, MODELS_DIR
from features import add_change_features, add_lag_features, add_rolling_features, add_time_features


RAW_DATA_PATH = DATA_DIR / "raw" / "karachi_weather_air_quality_hourly.csv"
PREDICTIONS_PATH = DATA_DIR / "processed" / "latest_predictions.csv"
BEST_MODELS_PATH = MODELS_DIR / "best_models.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TARGET_TO_HORIZON = {
    "target_aqi_24h": 24,
    "target_aqi_48h": 48,
    "target_aqi_72h": 72,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def alert_message(aqi: float) -> str:
    if aqi > 300:
        return "Hazardous AQI alert"
    if aqi > 200:
        return "Very unhealthy AQI alert"
    if aqi > 150:
        return "Unhealthy AQI alert"
    return "No hazardous AQI alert"


def load_latest_feature_row(raw_data_path: Path) -> pd.Series:
    df = pd.read_csv(raw_data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_features(df)
    df = df.dropna().reset_index(drop=True)
    return df.iloc[-1]


def predict_target(
    latest_row: pd.Series,
    target_column: str,
    model_name: str,
    feature_columns: list[str],
) -> float:
    if model_name == "current_aqi_baseline":
        return float(latest_row["us_aqi"])

    model_path = MODELS_DIR / f"{target_column}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model file not found: {model_path}")

    model = joblib.load(model_path)
    features = pd.DataFrame([latest_row[feature_columns].to_dict()])
    return float(model.predict(features)[0])


def build_predictions(
    latest_row: pd.Series,
    best_models: dict[str, str],
    feature_columns: list[str],
) -> pd.DataFrame:
    prediction_rows = []
    latest_time = pd.to_datetime(latest_row["time"])

    for target_column, horizon_hours in TARGET_TO_HORIZON.items():
        model_name = best_models[target_column]
        predicted_aqi = round(predict_target(latest_row, target_column, model_name, feature_columns), 2)
        forecast_time = latest_time + pd.Timedelta(hours=horizon_hours)

        prediction_rows.append(
            {
                "generated_at": latest_time,
                "forecast_time": forecast_time,
                "horizon_hours": horizon_hours,
                "model": model_name,
                "current_aqi": latest_row["us_aqi"],
                "predicted_aqi": predicted_aqi,
                "aqi_category": aqi_category(predicted_aqi),
                "alert": alert_message(predicted_aqi),
            }
        )

    return pd.DataFrame(prediction_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest AQI forecasts.")
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    best_models = load_json(BEST_MODELS_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    latest_row = load_latest_feature_row(args.raw_data)

    predictions = build_predictions(latest_row, best_models, feature_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(f"Generated predictions from latest feature row: {latest_row['time']}")
    print(f"Saved predictions to {args.output}")
    print(predictions[["horizon_hours", "model", "predicted_aqi", "aqi_category", "alert"]].to_string(index=False))


if __name__ == "__main__":
    main()
