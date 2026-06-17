# E-Commerce Conversion Intelligence Platform

## Project Summary

This project is an end-to-end **E-Commerce Conversion Intelligence Platform** built on RetailRocket visitor behaviour data.

The goal is to help a business answer one simple question:

> Which visitors are most likely to buy, and how should the business act on that information?

The project goes beyond a basic machine learning notebook. It includes data engineering, model selection, final champion hardening, batch scoring, forecasting, anomaly review, monitoring, alerting design, and a production-style MLOps architecture.

---

## Business Problem

Most ecommerce visitors do not buy. Random marketing wastes budget because it targets too many low-intent visitors.

This platform improves targeting by scoring each visitor with a purchase-intent model.

Simple business interpretation:

- Random targeting finds fewer than 1 buyer per 100 visitors.
- Model-based targeting focuses on high-intent visitors.
- The final champion model helps marketing teams target fewer visitors with higher buyer concentration.

---

## Main Features

- Visitor-level feature engineering from raw event data
- Baseline Logistic Regression model
- Manual champion/challenger benchmark
- AutoML-style model benchmark
- Final champion hardening
- Tuned Random Forest final champion
- Threshold optimization for business targeting
- Batch scoring workflow
- KPI forecasting with model comparison
- Anomaly and outlier review layer
- Monitoring, drift, and health dashboard
- MVD/course method coverage proof
- Streamlit executive dashboard
- Docker, CI/CD, monitoring, and deployment path

---

## Final Champion Model

The final deployable champion is selected using business-focused metrics:

- PR-AUC
- F1 score
- Precision
- Recall
- ROC-AUC
- Business score

The final hardening workflow is run with:

```bash
python3 -m src.models.finalize_true_champion
```

Main outputs:

```text
models/trained/final_champion_model.joblib
models/metadata/final_champion_metadata.json
reports/tables/final_true_champion_comparison.csv
reports/tables/final_true_champion_summary.csv
reports/tables/final_true_champion_thresholds.csv
reports/tables/final_true_champion_stability.csv
reports/tables/final_true_champion_sensitivity.csv
data/processed/final_champion_visitor_scores.csv
```

---

## Streamlit App

Run the dashboard with:

```bash
python3 -m streamlit run app/Executive_Overview.py
```

Dashboard pages:

```text
Executive Overview
1 Visitor Intent Predictor
2 Batch Scoring
3 Model Benchmark Selection
4 Business KPI Forecasting
5 Anomaly Outlier
6 Monitoring Drift Health
7 MLOps Architecture
8 MVD Coverage Proof
```

---

## Recommended Run Order

Run commands from the project root.

```bash
python3 -m src.data.build_visitor_features
python3 -m src.models.train_baseline_model
python3 -m src.models.analyze_thresholds
python3 -m src.models.run_model_selection
python3 -m src.models.finalize_true_champion
python3 -m src.forecasting.build_business_forecasts
python3 -m src.anomaly.build_anomaly_signals
python3 -m src.models.build_mvd_method_coverage
python3 -m streamlit run app/Executive_Overview.py
```

---

## Project Structure

```text
app/
  Executive_Overview.py
  app_utils.py
  pages/

src/
  data/
  models/
  forecasting/
  anomaly/

data/
  raw/
  processed/

models/
  trained/
  metadata/

reports/
  tables/

outputs/
  charts/

monitoring/
  prediction_logs/
  prometheus/
  grafana/
  alertmanager/

tests/
```

---

## MLOps Story

This project is designed as a production-style ML system:

1. Raw ecommerce events are converted into visitor-level features.
2. Multiple models are benchmarked fairly.
3. The final champion is hardened through tuning, threshold testing, stability checks, and outlier sensitivity.
4. The Streamlit app supports single scoring, batch scoring, business forecasting, anomaly review, and monitoring.
5. Prediction logs support production monitoring.
6. Prometheus, Grafana, and Alertmanager are planned for system metrics, dashboards, and alerts.
7. Docker, Kubernetes, and Helm provide the deployment story.

---

## Notes

Raw data and large model artifacts may be excluded from the final submission ZIP if file size is too large. The project includes scripts and documentation to regenerate the important outputs.
