#!/usr/bin/env bash

# Stop immediately when a command fails.
set -euo pipefail

# Find the project folder from this script's location.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PID_FILE="$PROJECT_ROOT/.mlflow-server.pid"

# Nothing needs stopping when the PID file does not exist.
if [[ ! -f "$PID_FILE" ]]; then
    echo "GOOD: MLflow is already stopped."
    exit 0
fi

# Read the stored server process ID.
MLFLOW_PID="$(cat "$PID_FILE")"

# Stop the server when the process still exists.
if kill -0 "$MLFLOW_PID" 2>/dev/null; then
    kill "$MLFLOW_PID"
    echo "GOOD: MLflow server stopped."
else
    echo "GOOD: MLflow process was not running."
fi

# Remove the old runtime PID file.
rm -f "$PID_FILE"
