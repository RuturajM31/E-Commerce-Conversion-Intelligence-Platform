"""Run Evidently monitoring from its isolated Python environment.

The normal application environment does not install Evidently.

Input:
    The environment variable ``RUN_EVIDENTLY_MONITORING``.

Output:
    True when monitoring completes successfully.
    False when monitoring is disabled, unavailable, or unsuccessful.

Used next:
    The final champion workflow will call this bridge after production
    scoring artifacts have been created successfully.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


# Find the repository root from src/monitoring/evidently_bridge.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Evidently is installed only inside this isolated environment.
EVIDENTLY_PYTHON = (
    PROJECT_ROOT
    / ".venv-evidently/bin/python"
)


def run_optional_evidently_monitoring() -> bool:
    """Generate drift reports only when explicitly enabled.

    Returns:
        True:
            Evidently ran successfully and generated monitoring outputs.

        False:
            Monitoring was disabled, the isolated environment was absent,
            or the Evidently process failed.

    Important:
        Evidently failure is intentionally non-fatal. A successful model
        training or production scoring run must not be destroyed because
        the optional monitoring service is unavailable.
    """

    # Monitoring is opt-in so normal workflows stay lightweight.
    monitoring_enabled = (
        os.getenv(
            "RUN_EVIDENTLY_MONITORING",
            "0",
        ).strip()
        == "1"
    )

    if not monitoring_enabled:
        print(
            "INFO: Evidently monitoring skipped. "
            "Set RUN_EVIDENTLY_MONITORING=1 to enable it."
        )
        return False

    # Fail softly when the isolated environment has not been created.
    if not EVIDENTLY_PYTHON.exists():
        print(
            "WARNING: Evidently monitoring was requested, "
            "but .venv-evidently is missing."
        )
        return False

    # Run the real monitoring module inside the isolated environment.
    command = [
        str(EVIDENTLY_PYTHON),
        "-m",
        "src.monitoring.evidently_drift",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Show the monitoring module's useful status output.
    if result.stdout.strip():
        print(result.stdout.strip())

    # Keep monitoring failure separate from model-production success.
    if result.returncode != 0:
        print(
            "WARNING: Optional Evidently monitoring failed."
        )

        if result.stderr.strip():
            print(result.stderr.strip())

        return False

    print(
        "GOOD: Optional Evidently monitoring completed."
    )

    return True
