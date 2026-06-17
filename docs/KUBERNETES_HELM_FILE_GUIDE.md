# Kubernetes + Helm File Guide

## What each file does

| File | Purpose |
|---|---|
| `helm/ecommerce-conversion-platform/Chart.yaml` | Metadata for the Helm chart. |
| `helm/ecommerce-conversion-platform/values.yaml` | Configurable deployment values such as image names, ports, namespace, and local host path. |
| `templates/namespace.yaml` | Creates a separate Kubernetes namespace for the project. |
| `templates/streamlit.yaml` | Deploys the Streamlit app and creates an internal Kubernetes service. |
| `templates/metrics-exporter.yaml` | Deploys the lightweight metrics exporter and exposes `/metrics` for Prometheus. |
| `templates/blackbox.yaml` | Deploys Blackbox Exporter to check Streamlit HTTP health. |
| `templates/prometheus.yaml` | Deploys Prometheus and configures scrape jobs. |
| `templates/grafana.yaml` | Deploys Grafana, connects it to Prometheus, and loads the dashboard. |

## Why both `k8s/` and `helm/` exist

`k8s/` contains normal Kubernetes YAML files.  
`helm/` contains reusable templated YAML files.

For your project story:

> I first created raw Kubernetes manifests to show the deployment structure. Then I converted them into a Helm chart so the deployment can be parameterized and reused.
