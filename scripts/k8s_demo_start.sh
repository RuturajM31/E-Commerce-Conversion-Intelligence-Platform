#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH"

NAMESPACE="ecommerce-mlops"
RELEASE="ecommerce-conversion-platform"
CHART="helm/ecommerce-conversion-platform"
LOG_DIR=".k8s-port-forward-logs"

echo "1) Checking Kubernetes..."
kubectl get nodes

echo ""
echo "2) Refreshing monitoring snapshot..."
python3 -m src.monitoring.build_monitoring_snapshot

echo ""
echo "3) Deploying/updating Helm release..."
helm upgrade --install "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" \
  --create-namespace

echo ""
echo "4) Waiting for pods to become ready..."
kubectl wait --for=condition=Ready pod --all -n "$NAMESPACE" --timeout=180s

echo ""
echo "5) Current Kubernetes status:"
kubectl get pods -n "$NAMESPACE"
kubectl get svc -n "$NAMESPACE"
helm list -n "$NAMESPACE"

echo ""
echo "6) Starting port-forwards..."
mkdir -p "$LOG_DIR"

# Stop old port-forwards started by this script.
if ls "$LOG_DIR"/*.pid >/dev/null 2>&1; then
  kill $(cat "$LOG_DIR"/*.pid) >/dev/null 2>&1 || true
  rm -f "$LOG_DIR"/*.pid
fi

kubectl port-forward -n "$NAMESPACE" svc/streamlit-service 8502:8501 > "$LOG_DIR/streamlit.log" 2>&1 &
echo $! > "$LOG_DIR/streamlit.pid"

kubectl port-forward -n "$NAMESPACE" svc/grafana-service 3001:3000 > "$LOG_DIR/grafana.log" 2>&1 &
echo $! > "$LOG_DIR/grafana.pid"

kubectl port-forward -n "$NAMESPACE" svc/prometheus-service 9091:9090 > "$LOG_DIR/prometheus.log" 2>&1 &
echo $! > "$LOG_DIR/prometheus.pid"

kubectl port-forward -n "$NAMESPACE" svc/metrics-exporter-service 8001:8000 > "$LOG_DIR/metrics-exporter.log" 2>&1 &
echo $! > "$LOG_DIR/metrics-exporter.pid"

kubectl port-forward -n "$NAMESPACE" svc/blackbox-exporter-service 9116:9115 > "$LOG_DIR/blackbox.log" 2>&1 &
echo $! > "$LOG_DIR/blackbox.pid"

sleep 5

echo ""
echo "✅ Kubernetes demo is ready:"
echo "Streamlit:          http://localhost:8502"
echo "Grafana:            http://localhost:3001"
echo "Prometheus:         http://localhost:9091"
echo "Prometheus Targets: http://localhost:9091/targets"
echo "Prometheus Alerts:  http://localhost:9091/alerts"
echo "Metrics Exporter:   http://localhost:8001/metrics"
echo "Blackbox Exporter:  http://localhost:9116"
echo ""
echo "Grafana login:"
echo "User: admin"
echo "Pass: Techno#123"
