"""Generate feature-importance explanations for AQI forecast models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import DATA_DIR, MODELS_DIR


FEATURES_PATH = DATA_DIR / "processed" / "karachi_aqi_features.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "processed" / "feature_importance.csv"
BEST_MODELS_PATH = MODELS_DIR / "best_models.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_feature_name(feature: str) -> str:
    replacements = {
        "us_aqi": "US AQI",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "carbon_monoxide": "CO",
        "nitrogen_dioxide": "NO2",
        "sulphur_dioxide": "SO2",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "wind_speed_10m": "Wind speed",
        "wind_direction_10m": "Wind direction",
        "pressure_msl": "Pressure",
        "cloud_cover": "Cloud cover",
        "rolling_mean": "rolling mean",
        "rolling_std": "rolling std",
        "day_of_month": "Day of month",
        "day_of_week": "Day of week",
        "aqi_change": "AQI change",
    }
    label = feature
    for raw, clean in replacements.items():
        label = label.replace(raw, clean)
    return label.replace("_", " ").strip()


def model_path_for(target: str, model_name: str) -> Path:
    return MODELS_DIR / f"{target}_{model_name}.joblib"


def tree_importance(model: Any, feature_columns: list[str], target: str, model_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "target": target,
            "model": model_name,
            "feature": feature_columns,
            "display_feature": [clean_feature_name(feature) for feature in feature_columns],
            "importance": estimator.feature_importances_,
            "method": "random_forest_importance",
        }
    )


def shap_importance(model: Any, x_sample: pd.DataFrame, feature_columns: list[str], target: str, model_name: str) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    importance = pd.Series(abs(shap_values).mean(axis=0), index=feature_columns)
    return pd.DataFrame(
        {
            "target": target,
            "model": model_name,
            "feature": feature_columns,
            "display_feature": [clean_feature_name(feature) for feature in feature_columns],
            "importance": importance.values,
            "method": "mean_abs_shap",
        }
    )


def build_importance(features_path: Path, sample_rows: int) -> pd.DataFrame:
    best_models = load_json(BEST_MODELS_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    features = pd.read_csv(features_path).sort_values("time").reset_index(drop=True)
    x_sample = features[feature_columns].tail(sample_rows)

    frames = []
    for target, model_name in best_models.items():
        if model_name == "current_aqi_baseline":
            continue

        path = model_path_for(target, model_name)
        if not path.exists():
            print(f"Skipping {target}: missing model file {path}")
            continue

        model = joblib.load(path)
        explanation = shap_importance(model, x_sample, feature_columns, target, model_name)
        if explanation.empty:
            explanation = tree_importance(model, feature_columns, target, model_name)

        frames.append(explanation)

    if not frames:
        return pd.DataFrame(columns=["target", "model", "feature", "display_feature", "importance", "method", "rank"])

    importance = pd.concat(frames, ignore_index=True)
    importance = importance.sort_values(["target", "importance"], ascending=[True, False])
    importance["rank"] = importance.groupby("target").cumcount() + 1
    return importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AQI model feature importances.")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=FEATURE_IMPORTANCE_PATH)
    parser.add_argument("--sample-rows", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importance = build_importance(args.features, args.sample_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output, index=False)

    print(f"Saved {len(importance)} feature-importance rows to {args.output}")
    if not importance.empty:
        preview = importance[importance["rank"] <= 10]
        print(preview[["target", "rank", "display_feature", "importance", "method"]].to_string(index=False))


if __name__ == "__main__":
    main()
