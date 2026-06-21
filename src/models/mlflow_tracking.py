"""Log the existing production champion to local MLflow.

Run this module with the isolated MLflow environment:

    .venv-mlflow/bin/python -m src.models.mlflow_tracking

The script reads existing production artifacts. It does not retrain the model.
"""

from __future__ import annotations

import json
import os
from numbers import Real
from pathlib import Path
from typing import Any


# Do not store local environment-variable names in model artifacts.
os.environ.setdefault(
    "MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING",
    "false",
)

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


# ---------------------------------------------------------------------
# Project and MLflow settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "ecommerce-conversion-intelligence"
REGISTERED_MODEL_NAME = "ecommerce-conversion-champion"
MODEL_ALIAS = "champion"

MODEL_PATH = PROJECT_ROOT / "models/trained/final_champion_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models/metadata/final_champion_metadata.json"

# Store MLflow lineage separately so the existing metadata hash stays valid.
LINEAGE_PATH = PROJECT_ROOT / "models/metadata/mlflow_champion_lineage.json"


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file and return its dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_value: str) -> Path:
    """Convert a metadata path into a usable project path."""

    path = Path(path_value)

    # Relative paths are resolved from the project root.
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def numeric_metrics(
    values: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    """Keep only numeric metrics and add a clear name prefix."""

    metrics: dict[str, float] = {}

    for name, value in values.items():
        # Booleans are excluded because they are not model metrics.
        if isinstance(value, Real) and not isinstance(value, bool):
            metrics[f"{prefix}{name}"] = float(value)

    return metrics


def model_parameters(model: Any) -> dict[str, Any]:
    """Return simple XGBoost parameters that MLflow can store."""

    clean_parameters: dict[str, Any] = {}

    for name, value in model.get_params().items():
        # MLflow parameters should be simple scalar values.
        if value is None or isinstance(value, (str, int, float, bool)):
            clean_parameters[f"model__{name}"] = value

    return clean_parameters


def load_input_example(
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Load five production rows in the exact model-feature order."""

    feature_columns = metadata["feature_columns"]
    scoring_path = resolve_project_path(metadata["scoring_data_path"])

    if not scoring_path.exists():
        raise FileNotFoundError(
            f"Scoring dataset not found: {scoring_path}"
        )

    # Only five rows are needed for the model signature and example.
    input_example = pd.read_csv(
        scoring_path,
        usecols=feature_columns,
        nrows=5,
    )

    # Preserve the exact feature order saved in production metadata.
    ordered_example = input_example[feature_columns].copy()

    # Float columns can safely represent missing values during inference.
    return ordered_example.astype("float64")


def log_evidence_files() -> None:
    """Attach existing model-selection evidence when files exist."""

    evidence_files = [
        "final_true_champion_comparison.csv",
        "final_true_champion_holdout.csv",
        "final_true_champion_summary.csv",
        "final_true_champion_thresholds.csv",
        "final_true_champion_sensitivity.csv",
        "final_true_champion_stability.csv",
    ]

    tables_folder = PROJECT_ROOT / "reports/tables"

    for filename in evidence_files:
        file_path = tables_folder / filename

        # Optional evidence is logged only when it exists.
        if file_path.exists():
            mlflow.log_artifact(
                str(file_path),
                artifact_path="evaluation_evidence",
            )


def find_registered_version(
    client: MlflowClient,
    run_id: str,
) -> str:
    """Find the registry version created by the current run."""

    model_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    matching_versions = [
        version
        for version in model_versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "MLflow registered the model, but its version was not found."
        )

    newest_version = max(
        matching_versions,
        key=lambda version: int(version.version),
    )

    return str(newest_version.version)


def track_final_champion() -> dict[str, str]:
    """Log the champion model, metrics, evidence, and registry lineage."""

    metadata = read_json(METADATA_PATH)
    model = joblib.load(MODEL_PATH)
    input_example = load_input_example(metadata)

    # The output example lets MLflow infer the prediction schema.
    probability_example = model.predict_proba(input_example)[:, 1]
    signature = infer_signature(input_example, probability_example)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="final-champion-retrospective-registration"
    ) as run:
        # Tags explain the purpose and source of this MLflow run.
        mlflow.set_tags(
            {
                "project": "ecommerce-conversion-intelligence",
                "run_type": "production_champion_registration",
                "model_family": str(metadata["model_family"]),
                "cost_profile": "local_open_source",
            }
        )

        # Log clear project-level parameters.
        mlflow.log_params(
            {
                "final_model_name": metadata["final_model_name"],
                "selection_metric": metadata["selection_metric"],
                "selection_split": metadata["selection_split"],
                "final_evaluation_split": metadata[
                    "final_evaluation_split"
                ],
                "selected_threshold": metadata["best_threshold"],
                "feature_count": len(metadata["feature_columns"]),
            }
        )

        # Log the fitted XGBoost configuration.
        mlflow.log_params(model_parameters(model))

        # Keep validation and untouched holdout metrics separate.
        validation_metrics = numeric_metrics(
            metadata["validation_metrics"],
            prefix="validation_",
        )
        holdout_metrics = numeric_metrics(
            metadata["final_holdout_metrics"],
            prefix="holdout_",
        )

        mlflow.log_metrics(validation_metrics)
        mlflow.log_metrics(holdout_metrics)

        # Preserve the complete original metadata as run evidence.
        mlflow.log_dict(
            metadata,
            "metadata/final_champion_metadata.json",
        )

        log_evidence_files()

        # Log and register the real XGBoost champion.
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="champion_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=[
                "mlflow==3.11.1",
                "xgboost==3.2.0",
                "scikit-learn==1.4.2",
                "pandas==2.1.4",
                "numpy==1.26.4",
                "pyarrow==14.0.2",
                "joblib==1.3.2",
            ],
        )

        client = MlflowClient(
            tracking_uri=TRACKING_URI,
            registry_uri=TRACKING_URI,
        )

        model_version = find_registered_version(
            client=client,
            run_id=run.info.run_id,
        )

        # The alias provides a stable production-style model reference.
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
            version=model_version,
        )

        lineage = {
            "tracking_uri": TRACKING_URI,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run.info.run_id,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": model_version,
            "registered_model_alias": MODEL_ALIAS,
            "model_uri": (
                f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
            ),
        }

        # Save compact lineage without modifying production metadata.
        LINEAGE_PATH.write_text(
            json.dumps(lineage, indent=2) + "\n",
            encoding="utf-8",
        )

        # Attach the final lineage record to the same MLflow run.
        mlflow.log_dict(
            lineage,
            "metadata/mlflow_champion_lineage.json",
        )

        return lineage


if __name__ == "__main__":
    result = track_final_champion()

    print("GOOD: champion logged and registered")
    print("Run ID:", result["run_id"])
    print("Registered model:", result["registered_model_name"])
    print("Model version:", result["registered_model_version"])
    print("Alias:", result["registered_model_alias"])
    print("Model URI:", result["model_uri"])
