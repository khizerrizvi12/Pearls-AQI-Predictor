# Submission Checklist

## Required Links

- GitHub repository: https://github.com/khizerrizvi12/Pearls-AQI-Predictor
- Live dashboard: https://pearls-aqi-predictor-8eluvoxqdhfnpt5iyzfogl.streamlit.app
- Final report: `reports/final_report.md`
- Final report PDF: `reports/Pearls_AQI_Predictor_Final_Report.pdf`
- EDA summary: `reports/eda_summary.md`

## Completed Deliverables

- [x] End-to-end AQI data pipeline
- [x] Historical 90-day data backfill
- [x] Time, lag, rolling, and change features
- [x] Feast offline and online feature store
- [x] Historical and online feature retrieval verification
- [x] 24h, 48h, and 72h forecasting targets
- [x] Baseline, Ridge, and Random Forest comparison
- [x] MAE, RMSE, and R2 evaluation
- [x] Forecast-driver feature importance
- [x] Hazardous AQI alert logic
- [x] Interactive Streamlit dashboard
- [x] Streamlit Community Cloud deployment
- [x] Hourly GitHub Actions prediction workflow
- [x] Daily GitHub Actions training workflow
- [x] Workflow artifacts verified
- [x] EDA figures and summary
- [x] Final report and PDF export

## Before Uploading

- [x] Open the live dashboard and confirm it loads.
- [ ] Open the GitHub repository in a private/incognito browser.
- [ ] Confirm both GitHub Actions workflows are visible.
- [ ] Download the latest successful workflow artifact as backup.
- [ ] Replace any internship portal placeholders such as name, organization, or supervisor.
- [x] Export `reports/final_report.md` to PDF if the portal requires a PDF.
- [ ] Keep the repository public until evaluation is complete.

## Honest Scope Statement

The submitted system includes a local Feast feature store using Parquet for
offline features and SQLite for online serving. A managed feature platform and
production model registry remain future work. The explanation output uses
Random Forest feature importance locally, with optional SHAP support in the
explanation script.

## Suggested Demo Flow

1. Open the live dashboard.
2. Explain the current AQI and three forecast horizons.
3. Show the observed/forecast chart and forecast drivers.
4. Open GitHub and show the pipeline scripts.
5. Show the Feast entity, feature view, feature service, and verification output.
6. Open the Actions tab and show successful workflow runs.
7. Show model metrics and explain why different horizons select different models.
8. End with limitations and future work.
