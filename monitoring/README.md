# Monitoring, Alerting, and MLOps Observability

## Purpose of This Monitoring Layer

This folder contains the local monitoring and alerting setup for the **E-Commerce Conversion Intelligence Platform**.

The goal is to show how the machine learning application can be monitored like a production system.

The Streamlit app explains the **business value** of the model.

The monitoring stack explains the **operational health** of the system.

In simple words:

```text
Streamlit = business dashboard
Grafana = monitoring dashboard
Prometheus = metrics collector
Alertmanager = alarm handler
```

---

# Why Monitoring Is Important

A model is not finished after training.

In real business use, we must also check:

```text
Is the app running?
Are predictions being generated?
Are prediction errors increasing?
Are visitor patterns changing?
Is the model still reliable?
Is data quality still good?
Should someone be alerted?
```

This monitoring layer adds a professional MLOps story to the project.

It shows that the system is not only built for analysis, but also designed for ongoing operation.

---

# KPI Layers for This Project

The project can include monitoring KPIs in **5 layers**:

```text
1. Business KPIs
2. Model performance KPIs
3. Prediction and serving KPIs
4. Data quality and drift KPIs
5. System health KPIs
```

These layers help explain the complete production story:

```text
Business value + model quality + prediction reliability + data quality + system health
```

---

## 1. Business KPIs

Business KPIs explain the value of the model for ecommerce decision-making.

They answer:

```text
How many visitors are being scored?
How many high-intent visitors did we find?
How much better is model-based targeting than random targeting?
What is the business impact?
```

Recommended business KPIs:

```text
Total visitors scored
High-intent visitors detected
Expected buyers in high-intent segment
Conversion rate, only when actual outcome labels are available
High-intent visitor rate
Estimated marketing efficiency gain
Top visitor segment distribution
Marketing focus rate
Potential campaign target size
Expected buyer capture
Estimated wasted targeting reduction
```

Simple example:

```text
Random targeting:
Target 100 visitors and maybe find less than 1 buyer.

Model-based targeting:
Target 100 high-intent visitors and find a much higher buyer concentration.
```

Business meaning:

```text
Instead of targeting everyone, marketing can focus on the visitors most likely to buy.
This improves campaign efficiency and reduces wasted marketing budget.
```

---

## 2. Model Performance KPIs

Model performance KPIs explain whether the machine learning model is actually useful.

They answer:

```text
Is the model better than a basic baseline?
How well does it find rare buyers?
How good is the selected threshold?
How many real buyers does it catch?
How many predicted buyers are actually useful?
```

Recommended model performance KPIs:

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
Precision at final threshold
Recall at final threshold
F1 at final threshold
Model comparison rank
Champion model name
Champion model version
```

Important for this project:

```text
PR-AUC
Precision
Recall
F1 score
Final threshold
Business score
```

Simple explanation:

```text
Precision = when the model says “high intent”, how often is it correct?

Recall = out of all real buyers, how many did the model catch?

F1 score = balance between precision and recall.

PR-AUC = model quality for rare buyer detection.

ROC-AUC = general ranking ability of the model.

Threshold = the score cut-off used to decide who is high intent.
```

Why PR-AUC matters here:

```text
Buyer conversion is a rare event.
When buyers are rare, PR-AUC is more useful than only accuracy.
```

---

## 3. Prediction and Serving KPIs

Prediction and serving KPIs explain whether the model is being used correctly after training.

They answer:

```text
Are predictions being generated?
How many visitors are scored?
Are prediction requests failing?
How fast is scoring?
What is the average model score?
Is the active model version known?
```

Recommended prediction and serving KPIs:

```text
Prediction requests per minute
Successful predictions
Failed predictions
Prediction failure rate
Average prediction score
Median prediction score
High-score prediction count
Low-score prediction count
Batch scoring volume
Average scoring time
Maximum scoring time
Model version currently active
Final threshold currently active
Visitors scored today
Visitors scored this week
Visitors scored in latest batch
High-intent predictions per batch
```

Simple example:

```text
Prediction requests today = 5,000
Failed predictions = 3
Failure rate = 0.06%
Average prediction score = 0.18
Active threshold = 0.97
```

Business meaning:

```text
The model is not only trained.
It is being used, scored, and monitored.
```

---

## 4. Data Quality and Drift KPIs

Data quality and drift KPIs explain whether the input data is still reliable.

They answer:

```text
Is the incoming data clean?
Are important values missing?
Are duplicate visitors appearing?
Are visitor behaviour patterns changing?
Is the model seeing different data than it was trained on?
```

Recommended data quality and drift KPIs:

```text
Missing value rate
Duplicate visitor count
Duplicate visitor rate
Invalid score count
Invalid score rate
New visitor volume
Feature drift score
Prediction drift score
Target drift, if future actual conversions are available
Feature distribution change
Anomaly rate
Outlier visitor count
Data freshness
Last scoring timestamp
Input row count
Rejected row count
Schema mismatch count
Unexpected column count
Null visitor ID count
Invalid target value count
```

Simple example:

```text
Missing value rate = 0.00%
Duplicate visitor rows = 0
Invalid prediction scores = 0
Anomaly rate = 3.66%
Drift detected = No
```

Drift example:

```text
Before:
Visitors viewed 3 products before buying.

Now:
Visitors view 12 products before buying.

This means visitor behaviour changed.
The model may need to be reviewed.
```

Why this matters:

```text
Even a good model can become weak if customer behaviour changes.
Monitoring drift helps decide when retraining or review is needed.
```

---

## 5. System Health KPIs

System health KPIs explain whether the app and infrastructure are running correctly.

They answer:

```text
Is the app up?
Is Prometheus scraping metrics?
Is Grafana connected?
Are containers restarting?
Are alerts active?
Is the system stable?
```

Recommended system health KPIs:

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
Service response time
Failed scrape count
Last successful scrape timestamp
Docker container status
Disk usage
Network error count
```

Simple example:

```text
App status = UP
Memory usage = 420 MB
Container restarts = 0
Active alerts = 0
Prometheus target health = healthy
```

Why this matters:

```text
If the app is down, business users cannot score visitors.
If monitoring is down, problems may happen silently.
If alerts are not working, nobody knows when the model or app needs attention.
```

---

# Recommended Grafana Dashboard Structure

Grafana should not show too many weak metrics.

For this project, the dashboard should be grouped into clear sections.

---

## Section 1: Business Impact

Recommended panels:

```text
Total visitors scored
High-intent visitors detected
High-intent visitor rate
Expected buyers in high-intent segment
Marketing focus rate
Estimated marketing efficiency gain
```

Purpose:

```text
Shows how the model improves targeting and supports business action.
```

---

## Section 2: Model Quality

Recommended panels:

```text
PR-AUC
ROC-AUC
Precision
Recall
F1 score
Final threshold
Business score
Champion model name
```

Purpose:

```text
Shows whether the final champion model is useful and reliable.
```

---

## Section 3: Prediction Monitoring

Recommended panels:

```text
Prediction volume over time
Successful predictions
Failed predictions
Prediction failure rate
Average prediction score
High-score prediction count
Batch scoring volume
Active model version
```

Purpose:

```text
Shows whether the model is being used correctly after training.
```

---

## Section 4: Data Quality and Drift

Recommended panels:

```text
Missing value rate
Duplicate visitor count
Invalid score count
Feature drift score
Prediction drift score
Anomaly rate
Outlier visitor count
Data freshness
```

Purpose:

```text
Shows whether the model input data is still clean and stable.
```

---

## Section 5: System Health and Alerts

Recommended panels:

```text
App health
Prometheus target health
Grafana health
Alertmanager health
CPU usage
Memory usage
Container restart count
Active alerts
Resolved alerts
```

Purpose:

```text
Shows whether the technical system is healthy and whether action is needed.
```

---

# Best 12 KPIs for Final Project Story

For the final project, the strongest 12 KPIs are:

```text
1. App health
2. Total visitors scored
3. Prediction volume
4. High-intent visitors detected
5. Average prediction score
6. PR-AUC
7. Precision
8. Recall
9. Final threshold
10. Anomaly rate
11. Missing value rate
12. Active alerts
```

These 12 KPIs give a complete and clean story:

```text
Business value
Model quality
Data quality
Operational health
Alerting readiness
```

---

# File-by-File Explanation

## 1. `Dockerfile`

### What this file means

The `Dockerfile` is the recipe for building the application container.

It tells Docker:

```text
Which Python version to use
Which project files to copy
Which packages to install
Which command starts the Streamlit app
```

### What it does

It packages the Streamlit dashboard and project code into a repeatable environment.

This helps avoid the common problem:

```text
It works on my laptop, but not on another machine.
```

### Simple example

Without Docker:

```bash
python3 -m streamlit run app/Executive_Overview.py
```

With Docker:

```bash
docker build -t ecommerce-conversion-app .
docker run -p 8501:8501 ecommerce-conversion-app
```

### Why it matters

It shows that the project can be containerized and prepared for deployment.

---

## 2. `.dockerignore`

### What this file means

`.dockerignore` tells Docker which files should not be copied into the Docker image.

It works like `.gitignore`, but for Docker.

### What it does

It keeps the Docker image clean by excluding unnecessary files such as:

```text
__pycache__/
.git/
.env
large raw data
notebook checkpoints
logs
temporary files
```

### Why it matters

It prevents the Docker image from becoming too large, messy, or unsafe.

---

## 3. `docker-compose.yml`

### What this file means

`docker-compose.yml` starts multiple services together.

In this project, it can start:

```text
Streamlit app
Prometheus
Grafana
Alertmanager
```

### What it does

It allows the local MLOps stack to run with one command:

```bash
docker compose up --build
```

### Why it matters

It shows that the project can run as a connected system, not only as separate scripts.

Simple meaning:

```text
Dockerfile = builds one service
docker-compose.yml = starts multiple services together
```

---

## 4. `monitoring/prometheus/prometheus.yml`

### What this file means

This is the main Prometheus configuration file.

Prometheus is the metrics collector.

### What it does

It tells Prometheus:

```text
How often to collect metrics
Which services to monitor
Where alert rules are located
Where Alertmanager is running
```

### Simple example

Prometheus acts like a person with a clipboard:

```text
Every few seconds, it checks system and model-health numbers.
```

### Why it matters

It creates the monitoring foundation for the project.

---

## 5. `monitoring/prometheus/alert_rules.yml`

### What this file means

This file defines the alert conditions.

### What it does

It tells Prometheus when something should be treated as a problem.

Example alert ideas:

```text
App is down
Prediction error rate is high
Model drift is detected
No predictions are received
Data quality has dropped
```

### Simple example

```text
If app health = down for more than 2 minutes, raise an alert.
```

### Why it matters

Monitoring without alerts is weak.

Alerts show that the system can notify someone when action is needed.

---

## 6. `monitoring/alertmanager/alertmanager.yml`

### What this file means

Alertmanager receives alerts from Prometheus and decides how to handle them.

### What it does

It controls:

```text
How alerts are grouped
How often alerts are repeated
Where alerts should be sent
Which team or channel receives them
```

For this project, real email or Slack secrets should not be hardcoded.

Use safe placeholders only.

### Simple flow

```text
Prometheus detects a problem
Prometheus sends alert
Alertmanager receives alert
Alertmanager routes the alert
```

### Why it matters

It completes the alerting story.

---

## 7. `monitoring/grafana/provisioning/datasources/prometheus.yml`

### What this file means

This file tells Grafana where to get metrics from.

### What it does

It connects Grafana to Prometheus.

Inside Docker Compose, Grafana can reach Prometheus using:

```text
http://prometheus:9090
```

### Simple analogy

```text
Grafana = TV screen
Prometheus = data source
datasource config = cable connecting the screen to the data
```

### Why it matters

It allows Grafana dashboards to load metrics automatically.

---

## 8. `monitoring/grafana/provisioning/dashboards/dashboard.yml`

### What this file means

This file tells Grafana where dashboard JSON files are stored.

### What it does

It points Grafana to:

```text
monitoring/grafana/dashboards/
```

### Why it matters

Without this file, dashboards may need to be imported manually.

With it, Grafana can load dashboards automatically when the stack starts.

---

## 9. `monitoring/grafana/dashboards/ecommerce_mlops_dashboard.json`

### What this file means

This is the Grafana dashboard definition.

### What it does

It defines dashboard panels such as:

```text
App health
Prediction volume
Prediction error count
Model drift status
Average prediction score
High-intent visitor count
Active alerts
```

### Difference from Streamlit

```text
Streamlit dashboard = business dashboard
Grafana dashboard = system and monitoring dashboard
```

### Why it matters

It shows the project has an operations view, not only a business analytics view.

---

## 10. `monitoring/README.md`

### What this file means

This file explains the monitoring and alerting setup.

### What it does

It documents:

```text
What Prometheus does
What Grafana does
What Alertmanager does
Which KPIs are monitored
How to run the stack
Which files are included
What is implemented and validated locally
What remains a local project scaffold
```

### Why it matters

It makes the monitoring folder easy to understand for reviewers and interviewers.

---

## 11. `DOCKER_MONITORING_QA_QC_STATUS.md`

### What this file means

This is a quality-control summary for the Docker and monitoring package.

### What it does

It documents:

```text
Which files were created
What each file is for
Whether the file passed QA/QC review
How to run the stack
What is fully functional
What is a professional placeholder
```

### Why it matters

It makes the package easy to review and explain.

---

# How to Run the Local Stack

After adding the Docker and monitoring files, run:

```bash
docker compose up --build
```

Expected local services:

```text
Streamlit app:   http://localhost:8501
Prometheus:      http://localhost:9090
Grafana:         http://localhost:3000
Alertmanager:    http://localhost:9093
```

---

# Honest Scope

This monitoring layer is a local MLOps setup.

It proves the architecture and monitoring design.

It does not mean:

```text
The app is deployed to cloud
Real Slack/email alerts are active
Every Streamlit metric is automatically scraped
A full production Kubernetes cluster is running
```

Correct project explanation:

```text
This project includes a local Docker-based MLOps monitoring stack with Prometheus, Grafana, and Alertmanager configuration. It demonstrates the production monitoring and alerting design path for an ecommerce conversion intelligence system.
```

---

# Interview Explanation

A clean way to explain this part:

```text
I separated business analytics from operational monitoring.

The Streamlit app is used by business users to understand visitor intent, model results, forecasting, anomalies, and MVD coverage.

The monitoring stack uses Docker, Prometheus, Grafana, and Alertmanager to demonstrate how the system could be monitored in production. It includes app-health monitoring, model-quality KPIs, data-quality checks, drift indicators, anomaly rate, prediction reliability, and alerting rules.
```
