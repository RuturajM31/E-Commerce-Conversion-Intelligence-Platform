# Kubernetes + Helm Presentation Story

## Simple explanation

Docker Compose helped me run the full local stack quickly:
Streamlit, Prometheus, Grafana, Alertmanager, and exporters.

Kubernetes shows how the same system can move toward orchestration:
applications run as Deployments, networking happens through Services,
and health is managed with readiness and liveness probes.

Helm packages the Kubernetes YAML into a reusable deployment chart.

## Professional explanation

This project uses a layered MLOps deployment approach:

1. Docker Compose for local reproducibility.
2. Prometheus and Grafana for monitoring and observability.
3. A lightweight cached metrics exporter to avoid heavy scrape-time computation.
4. Kubernetes Deployments and Services for orchestration.
5. Helm for parameterized deployment packaging.

## Interview line

> I separated the heavy analytics workload from live monitoring by generating a small metrics snapshot. Then I exposed that snapshot through a lightweight Prometheus exporter. Finally, I packaged the app and monitoring stack for Kubernetes using Helm, showing how the project can move from local development toward production-style deployment.
