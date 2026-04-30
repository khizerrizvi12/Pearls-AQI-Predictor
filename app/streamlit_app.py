"""Professional Streamlit dashboard for AQI forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "karachi_weather_air_quality_hourly.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "latest_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"

AQI_COLORS = {
    "Good": "#2e7d32",
    "Moderate": "#b88700",
    "Unhealthy for Sensitive Groups": "#c66a00",
    "Unhealthy": "#c62828",
    "Very Unhealthy": "#6a1b9a",
    "Hazardous": "#5d1f1f",
}

POLLUTANT_COLUMNS = [
    ("pm2_5", "PM2.5", "ug/m3"),
    ("pm10", "PM10", "ug/m3"),
    ("ozone", "Ozone", "ug/m3"),
    ("nitrogen_dioxide", "NO2", "ug/m3"),
    ("sulphur_dioxide", "SO2", "ug/m3"),
    ("carbon_monoxide", "CO", "ug/m3"),
]


st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --panel-border: #d8dee8;
            --muted-text: #5b6472;
            --surface: #ffffff;
            --soft-surface: #f6f8fb;
            --ink: #172033;
        }

        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        .app-header {
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 1rem;
            margin-bottom: 1.3rem;
        }

        .app-title {
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted-text);
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 700;
            min-height: 2rem;
            padding: 0.28rem 0.72rem;
            white-space: nowrap;
        }

        .panel {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--surface);
            padding: 1rem;
        }

        .panel-title {
            color: var(--muted-text);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .large-number {
            color: var(--ink);
            font-size: 2.65rem;
            font-weight: 780;
            line-height: 1;
        }

        .detail-text {
            color: var(--muted-text);
            font-size: 0.92rem;
            margin-top: 0.45rem;
        }

        .forecast-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .forecast-card {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--soft-surface);
            padding: 0.95rem;
        }

        .forecast-horizon {
            color: var(--muted-text);
            font-size: 0.82rem;
            font-weight: 720;
        }

        .forecast-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 780;
            margin: 0.25rem 0 0.45rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .mini-metric {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.8rem;
            background: #ffffff;
        }

        .mini-label {
            color: var(--muted-text);
            font-size: 0.76rem;
            font-weight: 720;
        }

        .mini-value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 760;
            margin-top: 0.2rem;
        }

        @media (max-width: 760px) {
            .forecast-grid,
            .metric-row {
                grid-template-columns: 1fr;
            }
            .app-title {
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    return df.sort_values("horizon_hours").reset_index(drop=True)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


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


def status_pill(category: str) -> str:
    color = AQI_COLORS.get(category, "#455a64")
    return f'<span class="status-pill" style="background:{color};">{category}</span>'


def format_number(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def render_header(latest_time: pd.Timestamp) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">Karachi AQI Predictor</div>
            <div class="app-subtitle">Latest observed hour: {latest_time:%d %b %Y, %I:%M %p}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_snapshot(latest_row: pd.Series) -> None:
    current_aqi = float(latest_row["us_aqi"])
    category = aqi_category(current_aqi)

    left, right = st.columns([0.95, 1.35], gap="medium")

    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">Current AQI</div>
                <div class="large-number">{current_aqi:.0f}</div>
                <div style="margin-top:0.65rem;">{status_pill(category)}</div>
                <div class="detail-text">Temperature {format_number(latest_row["temperature_2m"])} C | Wind {format_number(latest_row["wind_speed_10m"])} km/h | Humidity {latest_row["relative_humidity_2m"]:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        top_pollutants = "".join(
            f"""
                <div class="mini-metric">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{format_number(latest_row[column])}</div>
                    <div class="detail-text">{unit}</div>
                </div>
                """
            for column, label, unit in POLLUTANT_COLUMNS[:3]
        )
        bottom_pollutants = "".join(
            f"""
                <div class="mini-metric">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{format_number(latest_row[column])}</div>
                    <div class="detail-text">{unit}</div>
                </div>
                """
            for column, label, unit in POLLUTANT_COLUMNS[3:]
        )
        st.markdown(
            f"""
            <div class="metric-row">{top_pollutants}</div>
            <div style="height:0.75rem;"></div>
            <div class="metric-row">{bottom_pollutants}</div>
            """,
            unsafe_allow_html=True,
        )


def render_forecasts(predictions: pd.DataFrame) -> None:
    cards = []
    for row in predictions.itertuples(index=False):
        cards.append(
            f"""
            <div class="forecast-card">
                <div class="forecast-horizon">{int(row.horizon_hours)} hour forecast</div>
                <div class="forecast-value">{float(row.predicted_aqi):.1f}</div>
                {status_pill(row.aqi_category)}
                <div class="detail-text">{row.forecast_time:%d %b, %I:%M %p}</div>
                <div class="detail-text">Model: {str(row.model).replace("_", " ")}</div>
            </div>
            """
        )

    st.markdown("### Forecast")
    st.markdown(f'<div class="forecast-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_trend(raw_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    trend = raw_df.tail(168)[["time", "us_aqi"]].rename(columns={"time": "Time", "us_aqi": "AQI"})
    forecast = predictions[["forecast_time", "predicted_aqi"]].rename(
        columns={"forecast_time": "Time", "predicted_aqi": "AQI"}
    )

    trend["Series"] = "Observed"
    forecast["Series"] = "Forecast"
    chart_df = pd.concat([trend, forecast], ignore_index=True)

    st.markdown("### AQI Trend")
    st.line_chart(chart_df, x="Time", y="AQI", color="Series", height=330)


def render_metrics(metrics_df: pd.DataFrame) -> None:
    metrics = metrics_df.copy()
    metrics["forecast"] = metrics["target"].str.replace("target_aqi_", "", regex=False)
    metrics["model"] = metrics["model"].str.replace("_", " ")
    metrics = metrics[["forecast", "model", "mae", "rmse", "r2"]]

    st.markdown("### Model Evaluation")
    st.dataframe(
        metrics.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "r2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    inject_styles()

    try:
        raw_df = load_raw_data()
        predictions = load_predictions()
        metrics = load_metrics()
    except FileNotFoundError as exc:
        st.error(f"Missing project data file: {exc.filename}")
        st.stop()

    latest_row = raw_df.iloc[-1]
    render_header(latest_row["time"])
    render_current_snapshot(latest_row)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    render_forecasts(predictions)

    alert_rows = predictions[predictions["alert"] != "No hazardous AQI alert"]
    if not alert_rows.empty:
        st.error(alert_rows.iloc[0]["alert"])

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    render_trend(raw_df, predictions)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    render_metrics(metrics)


if __name__ == "__main__":
    main()
