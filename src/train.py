"""Train and evaluate AQI forecasting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, MODELS_DIR


FEATURES_PATH = DATA_DIR / "processed" / "karachi_aqi_features.csv"
METRICS_PATH = DATA_DIR / "processed" / "model_metrics.csv"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
NON_FEATURE_COLUMNS = {"city", "time", *TARGET_COLUMNS}


def split_by_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def root_mean_squared_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def baseline_predictions(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["us_aqi"].to_numpy()


def build_models() -> dict[str, Any]:
    return {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
    }


def train_target(
    target_column: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], str, Any]:
    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    results = []

    baseline_metrics = evaluate_predictions(y_test, baseline_predictions(test_df))
    results.append({"target": target_column, "model": "current_aqi_baseline", **baseline_metrics})

    best_model_name = "current_aqi_baseline"
    best_model = None
    best_rmse = baseline_metrics["rmse"]

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        results.append({"target": target_column, "model": model_name, **metrics})

        if metrics["rmse"] < best_rmse:
            best_model_name = model_name
            best_model = model
            best_rmse = metrics["rmse"]

    return results, best_model_name, best_model


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AQI forecasting models.")
    parser.add_argument("--input", type=Path, default=FEATURES_PATH)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    train_df, test_df = split_by_time(df, args.test_size)
    feature_columns = get_feature_columns(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_models = {}

    for target_column in TARGET_COLUMNS:
        target_results, best_model_name, best_model = train_target(
            target_column=target_column,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
        )
        all_results.extend(target_results)
        best_models[target_column] = best_model_name

        if best_model is not None:
            model_path = MODELS_DIR / f"{target_column}_{best_model_name}.joblib"
            joblib.dump(best_model, model_path)

    metrics_df = pd.DataFrame(all_results).sort_values(["target", "rmse"])
    metrics_df.to_csv(args.metrics_output, index=False)

    save_json(feature_columns, FEATURE_COLUMNS_PATH)
    save_json(best_models, MODELS_DIR / "best_models.json")

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Saved metrics to {args.metrics_output}")
    print("Best models:")
    for target_column, model_name in best_models.items():
        best_row = metrics_df[(metrics_df["target"] == target_column) & (metrics_df["model"] == model_name)].iloc[0]
        print(f"  {target_column}: {model_name} RMSE={best_row['rmse']:.2f} MAE={best_row['mae']:.2f} R2={best_row['r2']:.3f}")


if __name__ == "__main__":
    main()
