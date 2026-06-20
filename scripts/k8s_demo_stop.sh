#!/usr/bin/env bash

set -euo pipefail

LOG_DIR=".k8s-port-forward-logs"

# nullglob prevents an unmatched *.pid pattern from being treated as a file.
shopt -s nullglob
PID_FILES=("$LOG_DIR"/*.pid)

if (( ${#PID_FILES[@]} == 0 )); then
  echo "No Kubernetes port-forward processes found."
  exit 0
fi

stopped_count=0

for pid_file in "${PID_FILES[@]}"; do
  forward_name="$(basename "$pid_file" .pid)"
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    process_command="$(ps -p "$pid" -o command= 2>/dev/null || true)"

    # Do not kill a reused PID belonging to an unrelated process.
    if [[ "$process_command" == *"kubectl port-forward"* ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      stopped_count=$((stopped_count + 1))
      echo "Stopped: $forward_name"
    else
      echo "Skipped stale PID for $forward_name: $pid"
    fi
  else
    echo "Already stopped: $forward_name"
  fi

  rm -f "$pid_file"
done

echo "Kubernetes port-forwards stopped: $stopped_count"
