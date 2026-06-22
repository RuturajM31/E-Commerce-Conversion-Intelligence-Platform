#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

NAMESPACE="ecommerce-mlops"
RELEASE="ecommerce-conversion-platform"
CHART="helm/ecommerce-conversion-platform"
LOG_DIR=".k8s-port-forward-logs"

GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"

if [[ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]]; then
  echo "ERROR: GRAFANA_ADMIN_PASSWORD is not set."
  echo "Run: export GRAFANA_ADMIN_PASSWORD='your-strong-local-password'"
  exit 1
fi

command -v kubectl >/dev/null 2>&1 || {
  echo "ERROR: kubectl is not installed."
  exit 1
}

command -v helm >/dev/null 2>&1 || {
  echo "ERROR: Helm is not installed."
  exit 1
}

echo "1) Checking Kubernetes..."
kubectl get nodes

echo ""
echo "2) Creating namespace and Grafana credential Secret..."

kubectl create namespace "$NAMESPACE" \
  --dry-run=client \
  -o yaml \
  | kubectl apply -f -

SECRET_FILE="$(mktemp)"
chmod 600 "$SECRET_FILE"

cleanup_secret_file() {
  rm -f "$SECRET_FILE"
}

trap cleanup_secret_file EXIT

printf '%s=%s\n' \
  "admin-user" \
  "$GRAFANA_ADMIN_USER" \
  > "$SECRET_FILE"

printf '%s=%s\n' \
  "admin-password" \
  "$GRAFANA_ADMIN_PASSWORD" \
  >> "$SECRET_FILE"

kubectl create secret generic grafana-admin-credentials \
  --namespace "$NAMESPACE" \
  --from-env-file="$SECRET_FILE" \
  --dry-run=client \
  -o yaml \
  | kubectl apply -f -

echo ""
echo "3) Refreshing monitoring snapshot..."
python3 -m src.monitoring.build_monitoring_snapshot

echo ""
echo "4) Deploying or updating Helm release..."

helm upgrade --install "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" \
  --create-namespace

echo ""
echo "5) Waiting for deployments to finish rolling out..."

DEPLOYMENTS=(
  "alert-webhook"
  "alertmanager"
  "blackbox-exporter"
  "grafana"
  "metrics-exporter"
  "prometheus"
  "streamlit-app"
)

for deployment_name in "${DEPLOYMENTS[@]}"; do
  echo "Waiting for deployment/$deployment_name..."

  kubectl rollout status     "deployment/$deployment_name"     --namespace "$NAMESPACE"     --timeout=240s
done

echo ""
echo "6) Current Kubernetes status..."

kubectl get pods \
  --namespace "$NAMESPACE"

kubectl get services \
  --namespace "$NAMESPACE"

helm list \
  --namespace "$NAMESPACE"

echo ""
echo "7) Starting port-forwards..."

mkdir -p "$LOG_DIR"

if find "$LOG_DIR" -name "*.pid" -type f | grep -q .; then
  cat "$LOG_DIR"/*.pid \
    | xargs kill \
    >/dev/null 2>&1 \
    || true

  rm -f "$LOG_DIR"/*.pid
fi

start_port_forward() {
  local service_name="$1"
  local ports="$2"
  local log_name="$3"

  kubectl port-forward \
    --namespace "$NAMESPACE" \
    "service/$service_name" \
    "$ports" \
    > "$LOG_DIR/$log_name.log" 2>&1 &

  echo $! > "$LOG_DIR/$log_name.pid"
}

start_port_forward \
  "streamlit-service" \
  "8502:8501" \
  "streamlit"

start_port_forward \
  "grafana-service" \
  "3001:3000" \
  "grafana"

start_port_forward \
  "prometheus-service" \
  "9091:9090" \
  "prometheus"

start_port_forward \
  "metrics-exporter-service" \
  "8001:8000" \
  "metrics-exporter"

start_port_forward \
  "blackbox-exporter-service" \
  "9116:9115" \
  "blackbox"

start_port_forward \
  "alertmanager-service" \
  "9094:9093" \
  "alertmanager"

start_port_forward \
  "alert-webhook-service" \
  "5002:5001" \
  "alert-webhook"

sleep 5

echo ""
echo "8) Verifying port-forward processes..."

FORWARD_NAMES=(
  "streamlit"
  "grafana"
  "prometheus"
  "metrics-exporter"
  "blackbox"
  "alertmanager"
  "alert-webhook"
)

for forward_name in "${FORWARD_NAMES[@]}"; do
  pid_file="$LOG_DIR/$forward_name.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "ERROR: Missing PID file: $pid_file"
    exit 1
  fi

  pid="$(cat "$pid_file")"

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "ERROR: Port-forward failed: $forward_name"
    echo "Log output:"

    tail -n 20 "$LOG_DIR/$forward_name.log" || true
    exit 1
  fi
done

echo "Verified ${#FORWARD_NAMES[@]} port-forward processes."

echo ""
echo "Kubernetes demo is ready:"
echo "Streamlit:          http://localhost:8502"
echo "Grafana:            http://localhost:3001"
echo "Prometheus:         http://localhost:9091"
echo "Prometheus Targets: http://localhost:9091/targets"
echo "Prometheus Alerts:  http://localhost:9091/alerts"
echo "Metrics Exporter:   http://localhost:8001/metrics"
echo "Blackbox Exporter:  http://localhost:9116"
echo "Alertmanager:       http://localhost:9094"
echo "Alert Webhook:      http://localhost:5002/health"
echo ""
echo "Grafana user: $GRAFANA_ADMIN_USER"
echo "Grafana password: supplied securely through GRAFANA_ADMIN_PASSWORD"
