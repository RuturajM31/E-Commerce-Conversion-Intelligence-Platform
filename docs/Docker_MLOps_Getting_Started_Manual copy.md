# Docker + Local MLOps Stack Getting Started Manual

## Project

**Project name:** E-Commerce Conversion Intelligence Platform  
**Project root:**

```text
/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/
```

This manual explains how to start, check, use, stop, and troubleshoot the Docker-based MLOps stack for this project.

---

# 1. What We Are Running

```text
Streamlit      = business dashboard
Prometheus     = metrics collector
Blackbox       = health checker
Grafana        = monitoring dashboard
Alertmanager   = alert routing service
Docker Compose = starts all services together
```

| Service | URL | Purpose |
|---|---|---|
| Streamlit app | http://localhost:8501 | Main business dashboard for visitor intent, model results, forecasting, anomalies, and MVD proof |
| Prometheus | http://localhost:9090 | Collects monitoring metrics and evaluates alert rules |
| Blackbox Exporter | http://localhost:9115 | Checks whether the Streamlit health endpoint is reachable |
| Grafana | http://localhost:3000 | Monitoring dashboard for app health, alerts, prediction, drift, anomaly, and data-quality KPIs |
| Alertmanager | http://localhost:9093 | Receives alerts from Prometheus and routes them |

---

# 2. Login Details

## Grafana Login

```text
URL:      http://localhost:3000
Username: admin
Password: admin
```

Important: `admin/admin` is only for this local project demo. Do not use this password in real production.

## Docker Account

Docker Desktop sign-in is optional for running locally. Sign-in is useful if you want to explore Docker Hub and Docker Desktop features.

```text
Use a free Docker account.
Do not enter paid plan details unless you intentionally want a paid plan.
```

---

# 3. Files Used by the Docker/MLOps Stack

## Root Files

| File | What it means | What it does |
|---|---|---|
| `Dockerfile` | Recipe for the Streamlit app container | Builds a Python environment, installs requirements, copies project files, and starts Streamlit |
| `.dockerignore` | List of files Docker should not pack | Keeps Docker image clean by excluding cache, secrets, raw data, notebooks, and temporary files |
| `docker-compose.yml` | Multi-service startup file | Starts Streamlit, Prometheus, Blackbox Exporter, Grafana, and Alertmanager together |

## Monitoring Files

| File | What it means | What it does |
|---|---|---|
| `monitoring/README.md` | Monitoring documentation | Explains the monitoring stack, KPIs, file purpose, and run commands |
| `monitoring/prometheus/prometheus.yml` | Prometheus config | Tells Prometheus what to monitor and where alert rules are |
| `monitoring/prometheus/alert_rules.yml` | Alert rules | Defines when to raise alerts for app down, drift, missing values, anomaly rate, etc. |
| `monitoring/blackbox/blackbox.yml` | Blackbox config | Defines HTTP health check behaviour |
| `monitoring/alertmanager/alertmanager.yml` | Alertmanager config | Defines how alerts are grouped and routed |
| `monitoring/grafana/provisioning/datasources/prometheus.yml` | Grafana datasource config | Automatically connects Grafana to Prometheus |
| `monitoring/grafana/provisioning/dashboards/dashboard.yml` | Grafana dashboard loader | Tells Grafana where to load dashboard JSON files from |
| `monitoring/grafana/dashboards/ecommerce_mlops_dashboard.json` | Grafana dashboard | Defines monitoring panels |

---

# 4. One-Time Docker Setup

Run:

```bash
docker --version
docker compose version
```

Expected example:

```text
Docker version 24.0.6, build ed223bc
Docker Compose version v2.22.0-desktop.2
```

If this works, Docker is installed and available in Terminal.

---

# 5. Validate Project Docker Files Before Starting

Run from the project root.

## Validate Docker Compose

```bash
docker compose config
```

This checks whether `docker-compose.yml` is valid. It does not start containers.

## Check Monitoring Files

```bash
find monitoring -maxdepth 5 -type f
```

Expected important files:

```text
monitoring/README.md
monitoring/prometheus/prometheus.yml
monitoring/prometheus/alert_rules.yml
monitoring/blackbox/blackbox.yml
monitoring/alertmanager/alertmanager.yml
monitoring/grafana/provisioning/datasources/prometheus.yml
monitoring/grafana/provisioning/dashboards/dashboard.yml
monitoring/grafana/dashboards/ecommerce_mlops_dashboard.json
```

## Validate Grafana Dashboard JSON

```bash
python3 -m json.tool monitoring/grafana/dashboards/ecommerce_mlops_dashboard.json > /dev/null
```

Good result: no output means the JSON is valid.

---

# 6. Start the Full Docker Stack

Run:

```bash
docker compose up --build
```

This builds and starts:

```text
Streamlit app
Prometheus
Blackbox Exporter
Grafana
Alertmanager
```

The first run can take time because Docker downloads images and installs Python packages. Keep this terminal open because it shows live logs.

---

# 7. Check Running Containers

Open a new Terminal tab and run:

```bash
docker compose ps
```

Expected containers:

```text
ecommerce_streamlit_app
ecommerce_prometheus
ecommerce_blackbox_exporter
ecommerce_grafana
ecommerce_alertmanager
```

---

# 8. Open the Services

Open these URLs:

```text
Streamlit:    http://localhost:8501
Prometheus:   http://localhost:9090
Blackbox:     http://localhost:9115
Grafana:      http://localhost:3000
Alertmanager: http://localhost:9093
```

Grafana login:

```text
Username: admin
Password: admin
```

In Grafana, check:

```text
Dashboards → E-Commerce MLOps → E-Commerce Conversion Intelligence - MLOps Monitoring
```

---

# 9. Quick Health Checks from Terminal

Run while Docker stack is running:

```bash
curl -I http://localhost:8501/_stcore/health
curl -I http://localhost:9090/-/ready
curl -I http://localhost:3000/api/health
curl -I http://localhost:9093/-/ready
```

Good signs:

```text
HTTP/1.1 200 OK
HTTP/1.1 204 No Content
```

---

# 10. Important KPIs for This Project

## Business KPIs

```text
Total visitors scored
High-intent visitors detected
Expected buyers in high-intent segment
Conversion rate
High-intent visitor rate
Estimated marketing efficiency gain
Visitor segment distribution
Marketing focus rate
Potential campaign target size
Expected buyer capture
Estimated wasted targeting reduction
```

## Model Performance KPIs

```text
PR-AUC
ROC-AUC
Precision
Recall
F1 score
Best threshold
Business score
False positive count
False negative count
True positive count
True negative count
Champion model name
Champion model version
```

## Prediction and Serving KPIs

```text
Prediction requests per minute
Successful predictions
Failed predictions
Prediction failure rate
Average prediction score
High-score prediction count
Batch scoring volume
Average scoring time
Active model version
Final threshold currently active
Visitors scored today
Visitors scored in latest batch
```

## Data Quality and Drift KPIs

```text
Missing value rate
Duplicate visitor count
Invalid score count
Feature drift score
Prediction drift score
Anomaly rate
Outlier visitor count
Data freshness
Last scoring timestamp
Input row count
Rejected row count
Schema mismatch count
Null visitor ID count
Invalid target value count
```

## System Health KPIs

```text
App up/down status
CPU usage
Memory usage
Container restart count
Error count
Request latency
Prometheus target health
Grafana dashboard health
Alertmanager health
Active alert count
Resolved alert count
Uptime percentage
Docker container status
```

---

# 11. Common Problems and Fixes

## Problem: `docker: command not found`

Meaning: Docker Desktop is not installed or Docker is not available in Terminal.

Fix: Open Docker Desktop and wait until it is running. Then run:

```bash
docker --version
```

## Problem: Docker Desktop Requires Newer macOS

For this project machine:

```text
macOS 11.7.11
Intel Mac x86_64
```

The compatible setup used here is Docker Desktop 4.24.2 for Mac Intel.

## Problem: Port 8501 Already in Use

Error:

```text
listen tcp 0.0.0.0:8501: bind: address already in use
```

Find the process:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
```

Kill it:

```bash
kill -9 <PID>
```

Then restart:

```bash
docker compose down
docker compose up
```

## Problem: Grafana Shows `database is locked`

This can appear during startup. If Grafana continues and shows `HTTP Server Listen address=[::]:3000`, it is usually fine.

## Problem: Grafana Panels Show No Data

Some custom metrics are placeholders:

```text
ecommerce_predictions_total
ecommerce_model_drift_detected
ecommerce_missing_value_rate
ecommerce_anomaly_rate
```

This means the monitoring architecture is prepared, but not every custom model metric is exported yet. Working metrics should include app health, service status, and alerts.

---

# 12. Stop and Restart the Stack

Stop from the logs terminal:

```text
CTRL + C
```

Then clean containers:

```bash
docker compose down
```

Restart without rebuilding:

```bash
docker compose up
```

Rebuild after Dockerfile or requirements changes:

```bash
docker compose up --build
```

---

# 13. Deployment Story

Deployment levels:

```text
Level 1: Local Python run
Level 2: Local Docker Compose deployment
Level 3: Kubernetes manifests
Level 4: Helm chart
Level 5: Cloud deployment path, such as AWS EKS, Azure AKS, or Google GKE
```

Correct project explanation:

```text
The project includes a local Docker Compose deployment for validation and a Kubernetes/Helm deployment path for production-style deployment.
```

---

# 14. Interview Explanation

```text
I containerized the Streamlit ML dashboard using Docker and created a local Docker Compose MLOps stack.

The stack runs Streamlit, Prometheus, Blackbox Exporter, Grafana, and Alertmanager.

Streamlit is used for business users, while Grafana and Prometheus support operational monitoring.

The monitoring design includes app health, prediction reliability, model-quality KPIs, data-quality checks, drift indicators, anomaly rate, and alerting rules.

This shows that the project is not only a machine learning dashboard, but also has a production-style deployment and monitoring path.
```

---

# 15. Honest Scope

This setup proves:

```text
Docker deployment works locally.
The app runs in a container.
Monitoring services run together.
Grafana is available.
Prometheus is available.
Alertmanager is available.
Health checks are configured.
Alert rules are defined.
```

This setup does not yet prove:

```text
The app is deployed to a real cloud cluster.
Real Slack or email alerts are active.
Every custom model metric is automatically exported.
Kubernetes/Helm deployment is already complete.
```

Those are next deployment steps.