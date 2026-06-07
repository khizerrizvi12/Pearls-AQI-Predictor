"""Streamlit dashboard for Karachi AQI monitoring and forecasts."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "karachi_weather_air_quality_hourly.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "latest_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_importance.csv"

AQI_BANDS = [
    ("Good", 0, 50, "#2f8f46"),
    ("Moderate", 51, 100, "#c69214"),
    ("Unhealthy for Sensitive Groups", 101, 150, "#d97924"),
    ("Unhealthy", 151, 200, "#c93d3d"),
    ("Very Unhealthy", 201, 300, "#7b3fa1"),
    ("Hazardous", 301, 500, "#6b2737"),
]

POLLUTANTS = [
    ("pm2_5", "PM2.5", "ug/m3"),
    ("pm10", "PM10", "ug/m3"),
    ("ozone", "Ozone", "ug/m3"),
    ("nitrogen_dioxide", "NO2", "ug/m3"),
    ("sulphur_dioxide", "SO2", "ug/m3"),
    ("carbon_monoxide", "CO", "ug/m3"),
]


st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --surface: #ffffff;
            --surface-2: #f8fafc;
            --border: #d9e2ef;
            --ink: #172033;
            --muted: #657184;
            --accent: #0f766e;
            --accent-2: #2563eb;
        }

        .stApp {
            background: var(--bg);
        }

        .main .block-container {
            max-width: 1280px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            min-height: 112px;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-weight: 720;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
            font-weight: 820 !important;
        }

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            opacity: 1 !important;
        }

        .topbar {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }

        .brand-title {
            font-size: 2rem;
            font-weight: 820;
            line-height: 1.12;
            margin: 0;
            color: var(--ink);
        }

        .brand-subtitle {
            color: var(--muted);
            margin-top: 0.35rem;
            font-size: 0.98rem;
        }

        .run-status {
            border: 1px solid #b6d8d4;
            background: #ecfdf5;
            color: #0f513c;
            border-radius: 999px;
            padding: 0.45rem 0.7rem;
            font-weight: 760;
            white-space: nowrap;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--ink);
            margin: 1.25rem 0 0.6rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: minmax(280px, 0.95fr) minmax(360px, 1.4fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }

        .panel-muted {
            background: var(--surface-2);
        }

        .label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .aqi-main {
            display: flex;
            align-items: flex-end;
            gap: 0.75rem;
            margin: 0.35rem 0 0.75rem;
        }

        .aqi-number {
            color: var(--ink);
            font-size: 4rem;
            font-weight: 860;
            line-height: 0.95;
        }

        .aqi-category {
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.88rem;
            font-weight: 800;
            padding: 0.25rem 0.7rem;
            margin-bottom: 0.2rem;
        }

        .detail {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .scale {
            margin-top: 1rem;
        }

        .scale-track {
            position: relative;
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, #2f8f46 0 10%, #c69214 10% 20%, #d97924 20% 30%, #c93d3d 30% 40%, #7b3fa1 40% 60%, #6b2737 60% 100%);
        }

        .scale-marker {
            position: absolute;
            top: -5px;
            width: 4px;
            height: 22px;
            border-radius: 2px;
            background: #111827;
            box-shadow: 0 0 0 3px #ffffff;
        }

        .forecast-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .forecast-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent-2);
            border-radius: 8px;
            padding: 1rem;
        }

        .forecast-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
        }

        .forecast-horizon {
            color: var(--muted);
            font-weight: 800;
            font-size: 0.82rem;
        }

        .forecast-value {
            color: var(--ink);
            font-size: 2.5rem;
            font-weight: 860;
            margin: 0.25rem 0;
        }

        .pollutant-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .pollutant {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        .pollutant-name {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
        }

        .pollutant-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 820;
            margin-top: 0.2rem;
        }

        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .pipeline-step {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        .step-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 50%;
            background: #dbeafe;
            color: #1d4ed8;
            font-weight: 850;
            margin-bottom: 0.55rem;
        }

        .step-title {
            color: var(--ink);
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .topbar,
            .summary-grid {
                display: block;
            }

            .run-status {
                display: inline-flex;
                margin-top: 0.8rem;
            }

            .forecast-grid,
            .pollutant-grid,
            .pipeline-grid {
                grid-template-columns: 1fr;
            }

            .aqi-number {
                font-size: 3.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_raw_data(modified_at: float) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data
def load_predictions(modified_at: float) -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    return df.sort_values("horizon_hours").reset_index(drop=True)


@st.cache_data
def load_metrics(modified_at: float) -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_feature_importance(modified_at: float) -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def aqi_category(aqi: float) -> str:
    for name, low, high, _color in AQI_BANDS:
        if low <= aqi <= high:
            return name
    return "Hazardous"


def aqi_color(category: str) -> str:
    for name, _low, _high, color in AQI_BANDS:
        if name == category:
            return color
    return "#455a64"


def status_pill(category: str) -> str:
    return f'<span class="aqi-category" style="background:{aqi_color(category)};">{category}</span>'


def format_number(value: float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}"


def render_html(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def file_modified_at(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def delta_label(current: float, previous: float) -> tuple[str, str]:
    delta = current - previous
    if delta > 0:
        return f"+{delta:.0f}", "from previous hour"
    if delta < 0:
        return f"{delta:.0f}", "from previous hour"
    return "0", "from previous hour"


def best_metric(metrics: pd.DataFrame, target: str, metric: str) -> tuple[str, float] | None:
    rows = metrics[metrics["target"] == target].sort_values(metric)
    if rows.empty:
        return None
    best_row = rows.iloc[0]
    return str(best_row["model"]), float(best_row[metric])


def render_sidebar(raw_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    latest = raw_df.iloc[-1]
    first = raw_df.iloc[0]
    basis_time = predictions["generated_at"].max()

    with st.sidebar:
        st.markdown("## Pearls AQI")
        st.markdown("Karachi, Pakistan")
        st.divider()
        st.markdown("**Data window**")
        st.write(f"{first['time']:%d %b %Y} to {latest['time']:%d %b %Y}")
        st.markdown("**Rows loaded**")
        st.write(f"{len(raw_df):,} hourly observations")
        st.markdown("**Prediction basis**")
        st.write(f"{basis_time:%d %b %Y, %I:%M %p}")
        st.divider()
        st.markdown("**AQI categories**")
        for name, low, high, color in AQI_BANDS:
            high_label = f"{high}" if high < 500 else "500+"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.35rem 0;">'
                f'<span style="width:0.7rem;height:0.7rem;border-radius:50%;background:{color};display:inline-block;"></span>'
                f'<span>{name}: {low}-{high_label}</span></div>',
                unsafe_allow_html=True,
            )


def render_header(latest_time: pd.Timestamp) -> None:
    updated_at = pd.Timestamp.now(tz=ZoneInfo("Asia/Karachi"))
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="brand-title">Karachi Air Quality Forecasting</div>
                <div class="brand-subtitle">Operational dashboard for current AQI, pollutant conditions, and 3-day forecast signals. Latest observed AQI row: {latest_time:%d %b %Y, %I:%M %p}</div>
            </div>
            <div class="run-status">Pipeline active | Updated: {updated_at:%d %b, %I:%M %p}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(raw_df: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    latest = raw_df.iloc[-1]
    previous = raw_df.iloc[-2]
    current_aqi = float(latest["us_aqi"])
    category = aqi_category(current_aqi)
    aqi_delta, delta_caption = delta_label(current_aqi, float(previous["us_aqi"]))
    max_forecast = float(predictions["predicted_aqi"].max())
    best_24h = best_metric(metrics, "target_aqi_24h", "rmse")
    best_24h_model = best_24h[0].replace("_", " ") if best_24h else "N/A"
    best_24h_rmse = best_24h[1] if best_24h else None

    cols = st.columns(4, gap="medium")
    cols[0].metric("Current AQI", f"{current_aqi:.0f}", delta=aqi_delta, help=delta_caption)
    cols[1].metric("Condition", category)
    cols[2].metric("Peak Forecast", f"{max_forecast:.1f}")
    cols[3].metric("24h Best RMSE", f"{best_24h_rmse:.2f}" if best_24h_rmse is not None else "N/A", help=best_24h_model)


def render_summary(raw_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    latest = raw_df.iloc[-1]
    current_aqi = float(latest["us_aqi"])
    category = aqi_category(current_aqi)
    marker_position = min(max(current_aqi / 500 * 100, 0), 100)

    pollutant_cards = "".join(
        f"""
        <div class="pollutant">
            <div class="pollutant-name">{label}</div>
            <div class="pollutant-value">{format_number(latest[column])}</div>
            <div class="detail">{unit}</div>
        </div>
        """
        for column, label, unit in POLLUTANTS
    )

    left, right = st.columns([0.95, 1.35], gap="medium")

    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div class="label">Live air quality snapshot</div>
                <div class="aqi-main">
                    <div class="aqi-number">{current_aqi:.0f}</div>
                    <div>{status_pill(category)}</div>
                </div>
                <div class="detail">Temperature {format_number(latest["temperature_2m"])} C | Humidity {latest["relative_humidity_2m"]:.0f}% | Wind {format_number(latest["wind_speed_10m"])} km/h</div>
                <div class="scale">
                    <div class="scale-track">
                        <div class="scale-marker" style="left:{marker_position:.1f}%;"></div>
                    </div>
                    <div class="detail" style="display:flex;justify-content:space-between;margin-top:0.45rem;">
                        <span>0</span><span>100</span><span>200</span><span>300+</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="panel panel-muted">
                <div class="label">Pollutant profile</div>
                <div style="height:0.65rem;"></div>
                <div class="pollutant-grid">{pollutant_cards}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    forecast_cards = []
    for row in predictions.itertuples(index=False):
        category_color = aqi_color(row.aqi_category)
        forecast_cards.append(
            (
                f'<div class="forecast-card" style="border-left-color:{category_color};">'
                f'<div class="forecast-top">'
                f'<div class="forecast-horizon">{int(row.horizon_hours)} hour forecast</div>'
                f'{status_pill(row.aqi_category)}'
                f'</div>'
                f'<div class="forecast-value">{float(row.predicted_aqi):.1f}</div>'
                f'<div class="detail">Forecast time: {row.forecast_time:%d %b %Y, %I:%M %p}</div>'
                f'<div class="detail">Model: {str(row.model).replace("_", " ")}</div>'
                f'</div>'
            )
        )

    render_html('<div class="section-title">3-day forecast outlook</div>')
    render_html(f'<div class="forecast-grid">{"".join(forecast_cards)}</div>')


def render_charts(raw_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    left, right = st.columns([1.45, 0.9], gap="medium")

    with left:
        st.markdown('<div class="section-title">Observed and forecast AQI</div>', unsafe_allow_html=True)
        observed = raw_df.tail(168)[["time", "us_aqi"]].rename(columns={"time": "Time", "us_aqi": "AQI"})
        forecast = predictions[["forecast_time", "predicted_aqi"]].rename(
            columns={"forecast_time": "Time", "predicted_aqi": "AQI"}
        )
        observed["Series"] = "Observed"
        forecast["Series"] = "Forecast"
        chart_df = pd.concat([observed, forecast], ignore_index=True)
        fig, ax = plt.subplots(figsize=(9, 4.1))
        for series, color in [("Observed", "#2563eb"), ("Forecast", "#c69214")]:
            subset = chart_df[chart_df["Series"] == series]
            ax.plot(subset["Time"], subset["AQI"], marker="o", linewidth=2, markersize=3.5, label=series, color=color)
        ax.set_ylabel("AQI")
        ax.set_xlabel("")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="upper left")
        fig.autofmt_xdate()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown('<div class="section-title">Forecast by horizon</div>', unsafe_allow_html=True)
        bar_df = predictions[["horizon_hours", "predicted_aqi"]].copy()
        bar_df["horizon_hours"] = bar_df["horizon_hours"].astype(str) + "h"
        fig, ax = plt.subplots(figsize=(5.2, 4.1))
        bars = ax.bar(bar_df["horizon_hours"], bar_df["predicted_aqi"], color="#0f766e", width=0.58)
        ax.set_ylabel("Predicted AQI")
        ax.set_xlabel("Forecast horizon")
        ax.grid(True, axis="y", alpha=0.22)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}", ha="center", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_model_section(metrics: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)

    leaderboard = metrics.copy()
    leaderboard["forecast"] = leaderboard["target"].str.replace("target_aqi_", "", regex=False)
    leaderboard["model"] = leaderboard["model"].str.replace("_", " ")
    leaderboard = leaderboard.sort_values(["target", "rmse"])

    best = leaderboard.groupby("forecast", as_index=False).first()
    best_display = best[["forecast", "model", "mae", "rmse", "r2"]].rename(
        columns={"forecast": "Forecast", "model": "Best model", "mae": "MAE", "rmse": "RMSE", "r2": "R2"}
    )
    st.dataframe(
        best_display.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View full model comparison"):
        full_display = leaderboard[["forecast", "model", "mae", "rmse", "r2"]].rename(
            columns={"forecast": "Forecast", "model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R2"}
        )
        st.dataframe(
            full_display.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )


def render_explainability_section(feature_importance: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Forecast drivers</div>', unsafe_allow_html=True)

    if feature_importance.empty:
        st.info("Feature importance has not been generated yet. Run `python src/explain.py` to create it.")
        return

    available_targets = list(feature_importance["target"].drop_duplicates())
    target_labels = {target: target.replace("target_aqi_", "").replace("h", " hour") for target in available_targets}
    selected_label = st.radio(
        "Forecast horizon",
        options=[target_labels[target] for target in available_targets],
        index=0,
        horizontal=True,
    )
    selected_target = next(target for target, label in target_labels.items() if label == selected_label)

    top_features = (
        feature_importance[feature_importance["target"] == selected_target]
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance", ascending=True)
    )
    chart_df = top_features[["display_feature", "importance"]].rename(
        columns={"display_feature": "Feature", "importance": "Importance"}
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(chart_df["Feature"], chart_df["Importance"], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.22)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    method = top_features["method"].iloc[0].replace("_", " ")
    st.caption(f"Explanation method: {method}. Higher values indicate stronger influence on the selected forecast model.")


def render_pipeline_section() -> None:
    steps = [
        ("Fetch", "Open-Meteo weather and pollutant data"),
        ("Feature", "Lags, rolling windows, AQI change rate"),
        ("Train", "Baseline, Ridge, and Random Forest models"),
        ("Explain", "Feature importance for forecast drivers"),
        ("Predict", "24h, 48h, and 72h AQI forecasts"),
    ]
    cards = "".join(
        (
            f'<div class="pipeline-step">'
            f'<div class="step-index">{index}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="detail">{body}</div>'
            f'</div>'
        )
        for index, (title, body) in enumerate(steps, start=1)
    )
    render_html('<div class="section-title">Automated pipeline</div>')
    render_html(f'<div class="pipeline-grid">{cards}</div>')


def main() -> None:
    inject_styles()

    try:
        raw_df = load_raw_data(file_modified_at(RAW_DATA_PATH))
        predictions = load_predictions(file_modified_at(PREDICTIONS_PATH))
        metrics = load_metrics(file_modified_at(METRICS_PATH))
        feature_importance = load_feature_importance(file_modified_at(FEATURE_IMPORTANCE_PATH))
    except FileNotFoundError as exc:
        st.error(f"Missing project data file: {exc.filename}")
        st.stop()

    render_sidebar(raw_df, predictions)
    render_header(raw_df.iloc[-1]["time"])
    render_overview(raw_df, predictions, metrics)
    render_summary(raw_df, predictions)
    render_charts(raw_df, predictions)
    render_model_section(metrics)
    render_explainability_section(feature_importance)
    render_pipeline_section()


if __name__ == "__main__":
    main()
