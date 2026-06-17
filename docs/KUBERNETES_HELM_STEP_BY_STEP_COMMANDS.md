# Kubernetes + Helm Step-by-Step Commands

Run from the project root.

## 1. Make Helm available

```bash
export PATH="$HOME/.local/bin:$PATH"
helm version
```

## 2. Validate Helm chart

```bash
helm lint helm/ecommerce-conversion-platform
```

## 3. Render Helm templates

```bash
helm template ecommerce-conversion-platform helm/ecommerce-conversion-platform > /tmp/ecommerce_helm_rendered.yaml
```

## 4. Deploy with Helm

```bash
helm install ecommerce-conversion-platform helm/ecommerce-conversion-platform --namespace ecommerce-mlops --create-namespace
```

## 5. Check deployment

```bash
kubectl get pods -n ecommerce-mlops
kubectl get svc -n ecommerce-mlops
```

## 6. Access services

Run each port-forward command in a separate terminal.

```bash
kubectl port-forward -n ecommerce-mlops svc/streamlit-service 8502:8501
kubectl port-forward -n ecommerce-mlops svc/grafana-service 3001:3000
kubectl port-forward -n ecommerce-mlops svc/prometheus-service 9091:9090
```

Then open:

```text
Streamlit:  http://localhost:8502
Grafana:    http://localhost:3001
Prometheus: http://localhost:9091
```

Grafana login:

```text
admin
Techno#123
```

## 7. Clean up if needed

```bash
helm uninstall ecommerce-conversion-platform -n ecommerce-mlops
kubectl delete namespace ecommerce-mlops
```
