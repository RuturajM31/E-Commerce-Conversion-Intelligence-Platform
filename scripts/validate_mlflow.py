"""Validate the local MLflow champion from end to end.

Run with:

    .venv-mlflow/bin/python scripts/validate_mlflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# Find the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow this directly executed script to import the project's src package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.models import get_model_info

from src.models.mlflow_tracking import (
    METADATA_PATH,
    LINEAGE_PATH,
    load_input_example,
    numeric_metrics,
    read_json,
)


def main() -> None:
    """Verify the registered champion and its supporting evidence."""

    # Load the production metadata and saved MLflow lineage.
    metadata = read_json(METADATA_PATH)
    lineage = json.loads(
        Path(LINEAGE_PATH).read_text(encoding="utf-8")
    )

    # Connect to the local MLflow tracking and registry server.
    mlflow.set_tracking_uri(lineage["tracking_uri"])
    mlflow.set_registry_uri(lineage["tracking_uri"])

    client = MlflowClient(
        tracking_uri=lineage["tracking_uri"],
        registry_uri=lineage["tracking_uri"],
    )

    # Resolve the model currently assigned to the champion alias.
    model_version = client.get_model_version_by_alias(
        lineage["registered_model_name"],
        lineage["registered_model_alias"],
    )

    # The champion alias must point to the same run and version
    # recorded in the saved MLflow lineage file.
    assert model_version.run_id == lineage["run_id"]
    assert str(model_version.version) == str(
        lineage["registered_model_version"]
    )

    # Read the MLflow run containing parameters and metrics.
    run = client.get_run(lineage["run_id"])

    required_parameters = {
        "final_model_name",
        "selection_metric",
        "selection_split",
        "final_evaluation_split",
        "selected_threshold",
        "feature_count",
    }

    missing_parameters = required_parameters - set(run.data.params)

    assert not missing_parameters, (
        f"Missing MLflow parameters: {sorted(missing_parameters)}"
    )

    # Build the exact validation and holdout metric names we logged.
    expected_metrics = {
        **numeric_metrics(
            metadata["validation_metrics"],
            prefix="validation_",
        ),
        **numeric_metrics(
            metadata["final_holdout_metrics"],
            prefix="holdout_",
        ),
    }

    missing_metrics = set(expected_metrics) - set(run.data.metrics)

    assert not missing_metrics, (
        f"Missing MLflow metrics: {sorted(missing_metrics)}"
    )

    # Verify that the model signature uses the canonical feature order.
    model_info = get_model_info(lineage["model_uri"])
    signature_features = model_info.signature.inputs.input_names()

    assert signature_features == metadata["feature_columns"], (
        "MLflow feature order does not match production metadata."
    )

    # Load the registered XGBoost model from the champion alias.
    registered_model = mlflow.xgboost.load_model(
        lineage["model_uri"]
    )

    # Use five real production rows in the exact feature order.
    input_example = load_input_example(metadata)

    # Confirm that the registered model produces valid probabilities.
    probabilities = registered_model.predict_proba(
        input_example
    )[:, 1]

    assert len(probabilities) == len(input_example)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    print("GOOD: MLflow champion validation passed")
    print("Run ID:", lineage["run_id"])
    print("Version:", lineage["registered_model_version"])
    print("Alias:", lineage["registered_model_alias"])
    print("Feature count:", len(signature_features))
    print("Logged parameters:", len(run.data.params))
    print("Logged metrics:", len(run.data.metrics))
    print(
        "Probability range:",
        f"{probabilities.min():.6f}",
        "to",
        f"{probabilities.max():.6f}",
    )


if __name__ == "__main__":
    main()
