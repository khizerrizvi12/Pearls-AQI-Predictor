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

