# AQI Predictor

End-to-end Air Quality Index prediction system for forecasting AQI for the next 3 days.

## Project Goal

Build a serverless AQI prediction pipeline that:

- fetches weather and pollutant data,
- creates model-ready features,
- trains forecasting models,
- automates feature and training pipelines,
- shows real-time and forecasted AQI in a dashboard.

## Planned Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Hopsworks Feature Store
- GitHub Actions
- SHAP
- Random Forest feature importance / optional SHAP explanations

## Project Structure

```text
aqi-predictor/
  app/                  Streamlit dashboard
  data/                 Local raw/processed data, ignored by Git
  models/               Saved trained models, ignored by Git
  notebooks/            EDA and experiments
  reports/              Final report assets
  src/                  Data, feature, training, and prediction scripts
  .github/workflows/    Automation workflows
```

## First Milestone

Create a local pipeline that fetches AQI data for one city, builds features, trains a baseline model, and displays predictions in Streamlit.

## Local Pipeline

Run the project pipeline from the repository root:

```bash
python src/fetch_data.py
python src/features.py
python src/train.py
python src/explain.py
python src/predict.py
streamlit run app/streamlit_app.py
```

The explainability step generates `data/processed/feature_importance.csv`, which is shown in the dashboard as forecast drivers.

For Streamlit Community Cloud deployment, the repository includes a small dashboard snapshot:

- `data/raw/karachi_weather_air_quality_hourly.csv`
- `data/processed/latest_predictions.csv`
- `data/processed/model_metrics.csv`
- `data/processed/feature_importance.csv`

These files allow the public dashboard to load immediately. The scheduled GitHub Actions workflows still generate fresh artifacts for pipeline runs.

## Automation

GitHub Actions workflows are included for scheduled pipeline runs:

- `Hourly AQI Predictions`: fetches data, builds features, creates fallback models if needed, explains models, and generates predictions every hour.
- `Daily AQI Model Training`: fetches data, builds features, trains models, explains models, and generates predictions once per day.

Generated CSV and model files are uploaded as workflow artifacts. Because this starter version does not yet use a model registry, the hourly workflow trains fallback models if saved model artifacts are not present on the fresh runner. In a production version, these outputs should be written to a feature store and model registry.
