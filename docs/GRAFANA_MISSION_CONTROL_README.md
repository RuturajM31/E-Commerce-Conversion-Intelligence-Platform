# Grafana Mission-Control Upgrade

## What This Upgrade Adds

This package upgrades Grafana from a mostly infrastructure dashboard into a rich mission-control dashboard.

It adds:

```text
1. A metrics exporter service
2. Prometheus scrape config for the exporter
3. New alert rules based on real project metrics
4. A richer Grafana dashboard
5. Docker Compose updates
```

## Simple Flow

```text
CSV/JSON project outputs
        ↓
src/monitoring/metrics_exporter.py
        ↓
Prometheus
        ↓
Grafana mission-control dashboard
```

## New Service

```text
metrics-exporter
```

Local URL:

```text
http://localhost:8000/metrics
```

Docker service name:

```text
metrics-exporter:8000
```

## KPIs Fed Into Grafana

### Business KPIs

- Total visitors scored
- High-intent visitors
- High-intent rate
- Visitor segment counts
- Forecast totals by KPI
- Daily KPI latest values
- Converted visitors, only when actual outcome labels are available
- Conversion rate, only when actual outcome labels are available

The current production scoring table is unlabeled. Therefore, converted-visitor
and conversion-rate metrics are intentionally omitted instead of being reported
as fabricated zero values.

### Model KPIs

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1 score
- Final threshold
- Business score
- Champion model info

### Prediction KPIs

- Logged predictions
- Prediction errors
- Average prediction score
- Median prediction score
- P90 score
- P95 score
- Maximum score
- Score rows
- Batch scoring rows

### Data Quality KPIs

- Missing value count
- Missing value rate
- Duplicate visitor count
- Duplicate visitor rate
- Invalid score count
- Invalid score rate
- File availability

### Anomaly and Forecast KPIs

- Anomaly count
- Anomaly rate
- Forecast rows
- Forecasted totals by target

### System KPIs

- Streamlit app health
- Prometheus target health
- Active alerts
- Metrics exporter health
- Exporter scrape duration

## Files to Replace or Add

```text
src/monitoring/metrics_exporter.py
docker-compose.yml
monitoring/prometheus/prometheus.yml
monitoring/prometheus/alert_rules.yml
monitoring/grafana/dashboards/ecommerce_mlops_dashboard.json
GRAFANA_MISSION_CONTROL_QA_QC_STATUS.md
```

## Run Steps

After replacing the files:

```bash
docker compose down
docker compose up --build
```

Then open:

```text
Metrics exporter: http://localhost:8000/metrics
Prometheus:       http://localhost:9090
Grafana:          http://localhost:3000
```

Grafana login:

```text
Username: admin
Password: value supplied at runtime through GRAFANA_ADMIN_PASSWORD
```

## Prometheus Checks

Run these in Prometheus:

```promql
ecommerce_metrics_exporter_up
ecommerce_total_visitors
ecommerce_model_pr_auc
ecommerce_high_intent_visitors_total
ecommerce_anomaly_rate
ecommerce_file_available
```

## Grafana Dashboard

Open:

```text
Dashboards → E-Commerce MLOps → E-Commerce Mission Control - Conversion Intelligence
```

## Honest Scope

This upgrade makes many Grafana panels real by exposing project output metrics.

Some metrics may still show no data if the required source files do not exist yet. The exporter exposes file availability metrics so this is visible instead of hidden.
