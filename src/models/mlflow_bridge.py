"""Run MLflow tracking from the isolated MLflow environment.

This module does not import MLflow. It safely calls the separate
`.venv-mlflow` environment only when tracking is explicitly enabled.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


# Locate the project root from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLflow is installed only inside this isolated environment.
MLFLOW_PYTHON = PROJECT_ROOT / ".venv-mlflow/bin/python"


def run_optional_mlflow_tracking() -> bool:
    """Track the final champion only when explicitly enabled.

    Returns:
        True when tracking completes successfully.
        False when tracking is disabled, unavailable, or unsuccessful.
    """

    # Normal training does not require MLflow.
    tracking_enabled = (
        os.getenv("RUN_MLFLOW_TRACKING", "0").strip() == "1"
    )

    if not tracking_enabled:
        print(
            "INFO: MLflow tracking skipped. "
            "Set RUN_MLFLOW_TRACKING=1 to enable it."
        )
        return False

    # Fail softly when the isolated environment is unavailable.
    if not MLFLOW_PYTHON.exists():
        print(
            "WARNING: MLflow tracking was requested, but "
            ".venv-mlflow is missing."
        )
        return False

    # Run the existing tracking module using the isolated environment.
    command = [
        str(MLFLOW_PYTHON),
        "-m",
        "src.models.mlflow_tracking",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Show the useful output produced by the tracking module.
    if result.stdout.strip():
        print(result.stdout.strip())

    # MLflow failure must not destroy successful model training.
    if result.returncode != 0:
        print("WARNING: Optional MLflow tracking failed.")

        if result.stderr.strip():
            print(result.stderr.strip())

        return False

    print("GOOD: Optional MLflow tracking completed.")
    return True
