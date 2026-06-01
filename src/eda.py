"""Generate EDA charts and summaries for the AQI project report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, PROJECT_ROOT


RAW_DATA_PATH = DATA_DIR / "raw" / "karachi_weather_air_quality_hourly.csv"
METRICS_PATH = DATA_DIR / "processed" / "model_metrics.csv"
PREDICTIONS_PATH = DATA_DIR / "processed" / "latest_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_aqi_trend(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "aqi_trend.png"
    plt.figure(figsize=(11, 4.8))
    plt.plot(df["time"], df["us_aqi"], color="#2563eb", linewidth=1.8)
    plt.title("Karachi AQI Trend")
    plt.xlabel("Time")
    plt.ylabel("US AQI")
    plt.grid(alpha=0.25)
    save_figure(path)
    return path


def plot_hourly_pattern(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "hourly_aqi_pattern.png"
    hourly = df.assign(hour=df["time"].dt.hour).groupby("hour", as_index=False)["us_aqi"].mean()
    plt.figure(figsize=(9, 4.8))
    plt.bar(hourly["hour"], hourly["us_aqi"], color="#0f766e")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average US AQI")
    plt.xticks(range(0, 24, 2))
    plt.grid(axis="y", alpha=0.25)
    save_figure(path)
    return path


def plot_daily_pattern(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "daily_aqi_pattern.png"
    daily = df.assign(day=df["time"].dt.day_name()).groupby("day")["us_aqi"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = daily.reindex(order)
    plt.figure(figsize=(9, 4.8))
    plt.bar(daily.index, daily.values, color="#c69214")
    plt.title("Average AQI by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average US AQI")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_figure(path)
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "pollutant_weather_correlation.png"
    columns = ["us_aqi", *POLLUTANT_COLUMNS, "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl"]
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Between AQI, Pollutants, and Weather")

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    save_figure(path)
    return path


def plot_pollutants(df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "pollutant_trends.png"
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    for ax, column in zip(axes, POLLUTANT_COLUMNS):
        ax.plot(df["time"], df[column], linewidth=1.2)
        ax.set_title(column)
        ax.grid(alpha=0.2)
    fig.suptitle("Pollutant Trends", y=1.02)
    save_figure(path)
    return path


def build_summary(df: pd.DataFrame, figure_paths: list[Path], output_path: Path) -> None:
    missing_values = int(df.isna().sum().sum())
    summary = df["us_aqi"].describe()
    category_counts = pd.cut(
        df["us_aqi"],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
        include_lowest=True,
    ).value_counts().sort_index()

    lines = [
        "# EDA Summary",
        "",
        f"- Rows analyzed: {len(df):,}",
        f"- Date range: {df['time'].min():%d %b %Y, %I:%M %p} to {df['time'].max():%d %b %Y, %I:%M %p}",
        f"- Missing values: {missing_values}",
        f"- Mean AQI: {summary['mean']:.2f}",
        f"- Minimum AQI: {summary['min']:.0f}",
        f"- Maximum AQI: {summary['max']:.0f}",
        "",
        "## AQI Category Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]

    lines.extend(f"| {category} | {count:,} |" for category, count in category_counts.items())
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in figure_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AQI EDA report assets.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "eda_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw_data(args.input)

    figure_paths = [
        plot_aqi_trend(df),
        plot_hourly_pattern(df),
        plot_daily_pattern(df),
        plot_correlation_heatmap(df),
        plot_pollutants(df),
    ]
    build_summary(df, figure_paths, args.output)

    print(f"Saved EDA summary to {args.output}")
    for path in figure_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
