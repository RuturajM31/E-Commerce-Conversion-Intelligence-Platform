#!/usr/bin/env bash

LOG_DIR=".k8s-port-forward-logs"

if ls "$LOG_DIR"/*.pid >/dev/null 2>&1; then
  kill $(cat "$LOG_DIR"/*.pid) >/dev/null 2>&1 || true
  rm -f "$LOG_DIR"/*.pid
  echo "✅ Kubernetes port-forwards stopped."
else
  echo "No port-forward processes found."
fi
