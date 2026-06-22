#!/usr/bin/env bash

# Stop immediately when a command fails.
set -euo pipefail

# Find the project folder from this script's location.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Always run MLflow from the project root.
cd "$PROJECT_ROOT"

# Use MLflow from the isolated environment.
MLFLOW_BIN="$PROJECT_ROOT/.venv-mlflow/bin/mlflow"

# Store runtime information in simple local files.
PID_FILE="$PROJECT_ROOT/.mlflow-server.pid"
LOG_FILE="$PROJECT_ROOT/logs/mlflow-server.log"

# Create folders used by MLflow.
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/mlflow_artifacts"

# Fail clearly when the isolated environment is missing.
if [[ ! -x "$MLFLOW_BIN" ]]; then
    echo "ERROR: MLflow is not installed in .venv-mlflow."
    exit 1
fi

# Do not start a second server when one is already running.
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "GOOD: MLflow is already running."
    echo "URL: http://127.0.0.1:5000"
    exit 0
fi

# Start the local tracking server in the background.
nohup "$MLFLOW_BIN" server \
    --backend-store-uri "sqlite:///mlflow.db" \
    --artifacts-destination "$PROJECT_ROOT/mlflow_artifacts" \
    --host "127.0.0.1" \
    --port "5000" \
    > "$LOG_FILE" 2>&1 &

# Save the process ID so the stop script can find it.
echo $! > "$PID_FILE"

# Wait up to 30 seconds for the health endpoint.
for attempt in {1..30}; do
    if curl -fsS "http://127.0.0.1:5000/health" >/dev/null; then
        echo "GOOD: MLflow server is healthy."
        echo "URL: http://127.0.0.1:5000"
        exit 0
    fi

    sleep 1
done

# Show recent logs when startup fails.
echo "ERROR: MLflow did not become healthy."
tail -40 "$LOG_FILE"
exit 1
