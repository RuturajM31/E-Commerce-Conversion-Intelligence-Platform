"""Generate local Evidently feature and prediction drift reports.

Run with:

    .venv-evidently/bin/python -m src.monitoring.evidently_drift

The script compares:

1. One latest development snapshot per historical visitor.
2. The current one-row-per-visitor scoring population.

The untouched final holdout is never used as the drift baseline.
"""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import evidently
import joblib
import numpy as np
import pandas as pd

from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.presets import DataSummaryPreset

from src.monitoring.drift_summary import (
    build_drift_monitoring_summary,
)
from src.monitoring.monitoring_provenance import (
    build_monitoring_provenance,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "models/metadata/final_champion_metadata.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models/trained/final_champion_model.joblib"
)

SCORES_PATH = (
    PROJECT_ROOT
    / "data/processed/final_champion_visitor_scores.csv"
)

REPORT_FOLDER = (
    PROJECT_ROOT
    / "reports/evidently/latest"
)

SUMMARY_FOLDER = (
    PROJECT_ROOT
    / "monitoring/snapshots"
)

SUMMARY_PATH = (
    SUMMARY_FOLDER
    / "evidently_monitoring_summary.json"
)


# ---------------------------------------------------------------------
# Monitoring settings
# ---------------------------------------------------------------------

VISITOR_ID_COLUMN = "visitorid"
SCORE_COLUMN = "purchase_intent_score"

DEVELOPMENT_SPLITS = {
    "train",
    "validation",
}

MONITORING_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# PSI values at or above 0.10 are treated as drift.
DRIFT_METHOD = "psi"
DRIFT_THRESHOLD = 0.10

# Dataset drift is triggered when at least half the columns drift.
DRIFT_SHARE = 0.50


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file was not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def resolve_project_path(path_value: str) -> Path:
    """Resolve a metadata path relative to the project root."""

    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def validate_required_files(
    training_path: Path,
    scoring_path: Path,
) -> None:
    """Confirm that all required input artifacts exist."""

    required_paths = [
        METADATA_PATH,
        MODEL_PATH,
        training_path,
        scoring_path,
        SCORES_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted_paths = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required monitoring files are missing:\n"
            f"{formatted_paths}"
        )


def validate_feature_values(
    data: pd.DataFrame,
    feature_columns: list[str],
    dataset_name: str,
) -> None:
    """Reject missing or infinite production feature values."""

    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing model features: "
            f"{missing_columns}"
        )

    missing_counts = (
        data[feature_columns]
        .isna()
        .sum()
    )

    columns_with_missing_values = (
        missing_counts[
            missing_counts > 0
        ]
        .index
        .tolist()
    )

    if columns_with_missing_values:
        raise ValueError(
            f"{dataset_name} contains missing feature values in: "
            f"{columns_with_missing_values}"
        )

    numeric_values = (
        data[feature_columns]
        .astype("float64")
        .to_numpy()
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"{dataset_name} contains infinite feature values."
        )


def load_reference_population(
    training_path: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create one latest development snapshot per visitor."""

    required_columns = [
        VISITOR_ID_COLUMN,
        "snapshot_time",
        "data_split",
        *feature_columns,
    ]

    training_data = pd.read_csv(
        training_path,
        usecols=required_columns,
    )

    training_data["snapshot_time"] = pd.to_datetime(
        training_data["snapshot_time"],
        errors="coerce",
        utc=True,
    )

    split_values = (
        training_data["data_split"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    development_data = training_data.loc[
        split_values.isin(DEVELOPMENT_SPLITS)
    ].copy()

    if development_data.empty:
        raise RuntimeError(
            "No train or validation rows were found."
        )

    if development_data["snapshot_time"].isna().any():
        raise ValueError(
            "Development data contains invalid snapshot times."
        )

    # A visitor may appear in several historical snapshots.
    # Keep only the latest eligible development snapshot.
    reference_data = (
        development_data
        .sort_values(
            [
                VISITOR_ID_COLUMN,
                "snapshot_time",
            ]
        )
        .drop_duplicates(
            subset=[VISITOR_ID_COLUMN],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if reference_data[VISITOR_ID_COLUMN].duplicated().any():
        raise RuntimeError(
            "Reference population still contains duplicate visitors."
        )

    validate_feature_values(
        data=reference_data,
        feature_columns=feature_columns,
        dataset_name="Reference population",
    )

    return reference_data


def load_current_population(
    scoring_path: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load current features and merge deployed production scores."""

    required_scoring_columns = [
        VISITOR_ID_COLUMN,
        "scoring_time",
        *feature_columns,
    ]

    scoring_data = pd.read_csv(
        scoring_path,
        usecols=required_scoring_columns,
    )

    scores = pd.read_csv(
        SCORES_PATH,
        usecols=[
            VISITOR_ID_COLUMN,
            SCORE_COLUMN,
            "predicted_conversion",
            "production_threshold",
        ],
    )

    if scoring_data[VISITOR_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Current scoring data contains duplicate visitors."
        )

    if scores[VISITOR_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Production score data contains duplicate visitors."
        )

    current_data = scoring_data.merge(
        scores,
        on=VISITOR_ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_scores = int(
        (current_data["_merge"] != "both").sum()
    )

    if missing_scores:
        raise ValueError(
            f"{missing_scores:,} current visitors have no score."
        )

    current_data = current_data.drop(
        columns="_merge"
    )

    if current_data[SCORE_COLUMN].isna().any():
        raise ValueError(
            "Current production scores contain missing values."
        )

    if not current_data[SCORE_COLUMN].between(0, 1).all():
        raise ValueError(
            "Current production scores must be between 0 and 1."
        )

    validate_feature_values(
        data=current_data,
        feature_columns=feature_columns,
        dataset_name="Current population",
    )

    return current_data


def verify_current_scores(
    model: Any,
    current_data: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """Confirm saved production scores match the champion model."""

    model_input = (
        current_data[feature_columns]
        .astype("float64")
    )

    recalculated_scores = (
        model.predict_proba(model_input)[:, 1]
    )

    saved_scores = (
        current_data[SCORE_COLUMN]
        .astype("float64")
        .to_numpy()
    )

    max_absolute_difference = float(
        np.max(
            np.abs(
                recalculated_scores
                - saved_scores
            )
        )
    )

    if max_absolute_difference > 0.000001:
        raise ValueError(
            "Saved production scores do not match the champion model. "
            f"Maximum difference: {max_absolute_difference:.10f}"
        )

    return max_absolute_difference


def add_reference_scores(
    model: Any,
    reference_data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate champion scores for the reference population."""

    scored_reference = reference_data.copy()

    model_input = (
        scored_reference[feature_columns]
        .astype("float64")
    )

    scored_reference[SCORE_COLUMN] = (
        model.predict_proba(model_input)[:, 1]
    )

    return scored_reference


def create_balanced_samples(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic and equally sized monitoring samples."""

    sample_size = min(
        MONITORING_SAMPLE_SIZE,
        len(reference_data),
        len(current_data),
    )

    if sample_size <= 0:
        raise RuntimeError(
            "Monitoring datasets contain no rows."
        )

    reference_sample = (
        reference_data
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    current_sample = (
        current_data
        .sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    return reference_sample, current_sample


def create_evidently_dataset(
    data: pd.DataFrame,
    numerical_columns: list[str],
) -> Dataset:
    """Create an Evidently dataset with an explicit definition."""

    report_data = (
        data[numerical_columns]
        .astype("float64")
        .copy()
    )

    data_definition = DataDefinition(
        numerical_columns=numerical_columns,
    )

    return Dataset.from_pandas(
        report_data,
        data_definition=data_definition,
    )


def run_feature_drift_report(
    reference_sample: pd.DataFrame,
    current_sample: pd.DataFrame,
    feature_columns: list[str],
) -> Any:
    """Generate feature quality and feature drift results."""

    reference_dataset = create_evidently_dataset(
        data=reference_sample,
        numerical_columns=feature_columns,
    )

    current_dataset = create_evidently_dataset(
        data=current_sample,
        numerical_columns=feature_columns,
    )

    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(
                columns=feature_columns,
                method=DRIFT_METHOD,
                threshold=DRIFT_THRESHOLD,
                drift_share=DRIFT_SHARE,
            ),
        ]
    )

    return report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )


def run_prediction_drift_report(
    reference_sample: pd.DataFrame,
    current_sample: pd.DataFrame,
) -> Any:
    """Generate champion prediction-score drift results."""

    score_columns = [
        SCORE_COLUMN,
    ]

    reference_dataset = create_evidently_dataset(
        data=reference_sample,
        numerical_columns=score_columns,
    )

    current_dataset = create_evidently_dataset(
        data=current_sample,
        numerical_columns=score_columns,
    )

    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(
                columns=score_columns,
                method=DRIFT_METHOD,
                threshold=DRIFT_THRESHOLD,
                drift_share=DRIFT_SHARE,
            ),
        ]
    )

    return report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )


def quality_summary(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Return compact data-quality measurements."""

    feature_values = (
        data[feature_columns]
        .astype("float64")
    )

    missing_by_feature = {
        column: int(value)
        for column, value in (
            feature_values
            .isna()
            .sum()
            .items()
        )
    }

    infinite_by_feature = {
        column: int(
            np.isinf(
                feature_values[column]
                .to_numpy()
            ).sum()
        )
        for column in feature_columns
    }

    return {
        "rows": int(len(data)),
        "unique_visitors": int(
            data[VISITOR_ID_COLUMN].nunique()
        ),
        "duplicate_visitor_rows": int(
            data[VISITOR_ID_COLUMN]
            .duplicated()
            .sum()
        ),
        "missing_values_by_feature": missing_by_feature,
        "infinite_values_by_feature": infinite_by_feature,
    }


def score_summary(
    data: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Return compact score-distribution measurements."""

    scores = (
        data[SCORE_COLUMN]
        .astype("float64")
    )

    predicted_positive = (
        scores >= threshold
    )

    return {
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "standard_deviation": float(scores.std()),
        "p90": float(scores.quantile(0.90)),
        "p95": float(scores.quantile(0.95)),
        "p99": float(scores.quantile(0.99)),
        "predicted_positive_count": int(
            predicted_positive.sum()
        ),
        "predicted_positive_rate": float(
            predicted_positive.mean()
        ),
    }


def relative_path(path: Path) -> str:
    """Return a project-relative path for documentation."""

    return str(
        path.relative_to(PROJECT_ROOT)
    )


def write_monitoring_summary(
    metadata: dict[str, Any],
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    reference_sample: pd.DataFrame,
    current_sample: pd.DataFrame,
    feature_report_paths: dict[str, Path],
    prediction_report_paths: dict[str, Path],
    provenance: dict[str, Any],
    drift_results: dict[str, Any],
    score_difference: float,
) -> dict[str, Any]:
    """Write a compact monitoring summary for later integrations."""

    threshold = float(
        metadata["best_threshold"]
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evidently_version": evidently.__version__,
        "model": {
            "name": metadata["final_model_name"],
            "family": metadata["model_family"],
            "production_threshold": threshold,
            "feature_columns": metadata["feature_columns"],
        },
        "provenance": provenance,
        "reference_population": {
            "design": (
                "Latest train or validation snapshot per visitor; "
                "final holdout excluded"
            ),
            "rows": int(len(reference_data)),
            "unique_visitors": int(
                reference_data[VISITOR_ID_COLUMN].nunique()
            ),
            "first_snapshot_time": (
                reference_data["snapshot_time"]
                .min()
                .isoformat()
            ),
            "last_snapshot_time": (
                reference_data["snapshot_time"]
                .max()
                .isoformat()
            ),
        },
        "current_population": {
            "design": "One latest scoring row per visitor",
            "rows": int(len(current_data)),
            "unique_visitors": int(
                current_data[VISITOR_ID_COLUMN].nunique()
            ),
            "scoring_time": str(
                current_data["scoring_time"].iloc[0]
            ),
        },
        "monitoring_sample": {
            "reference_rows": int(
                len(reference_sample)
            ),
            "current_rows": int(
                len(current_sample)
            ),
            "random_state": RANDOM_STATE,
        },
        "drift_configuration": {
            "method": DRIFT_METHOD,
            "column_threshold": DRIFT_THRESHOLD,
            "dataset_drift_share": DRIFT_SHARE,
        },
        "drift_results": drift_results,
        "data_quality": {
            "reference": quality_summary(
                reference_data,
                metadata["feature_columns"],
            ),
            "current": quality_summary(
                current_data,
                metadata["feature_columns"],
            ),
        },
        "prediction_scores": {
            "reference": score_summary(
                reference_data,
                threshold,
            ),
            "current": score_summary(
                current_data,
                threshold,
            ),
            "maximum_saved_score_difference": (
                score_difference
            ),
        },
        "label_status": {
            "current_labels_available": False,
            "performance_monitoring": (
                "Pending delayed-label workflow"
            ),
        },
        "reports": {
            "feature_drift_html": relative_path(
                feature_report_paths["html"]
            ),
            "feature_drift_json": relative_path(
                feature_report_paths["json"]
            ),
            "prediction_drift_html": relative_path(
                prediction_report_paths["html"]
            ),
            "prediction_drift_json": relative_path(
                prediction_report_paths["json"]
            ),
        },
    }

    SUMMARY_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def save_report(
    result: Any,
    report_name: str,
) -> dict[str, Path]:
    """Save one Evidently report as HTML and JSON."""

    REPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        REPORT_FOLDER
        / f"{report_name}.html"
    )

    json_path = (
        REPORT_FOLDER
        / f"{report_name}.json"
    )

    result.save_html(
        str(html_path)
    )

    result.save_json(
        str(json_path)
    )

    return {
        "html": html_path,
        "json": json_path,
    }


def generate_monitoring_reports() -> dict[str, Any]:
    """Run the complete local Evidently monitoring workflow."""

    metadata = read_json(
        METADATA_PATH
    )

    feature_columns = list(
        metadata["feature_columns"]
    )

    training_path = resolve_project_path(
        metadata["training_data_path"]
    )

    scoring_path = resolve_project_path(
        metadata["scoring_data_path"]
    )

    validate_required_files(
        training_path=training_path,
        scoring_path=scoring_path,
    )

    print("Step 1: Load production champion")
    model = joblib.load(
        MODEL_PATH
    )

    print("Step 2: Build visitor-level reference population")
    reference_data = load_reference_population(
        training_path=training_path,
        feature_columns=feature_columns,
    )

    print("Step 3: Load current visitor population and scores")
    current_data = load_current_population(
        scoring_path=scoring_path,
        feature_columns=feature_columns,
    )

    print("Step 4: Verify saved production scores")
    score_difference = verify_current_scores(
        model=model,
        current_data=current_data,
        feature_columns=feature_columns,
    )

    print("Step 5: Score the reference population")
    reference_data = add_reference_scores(
        model=model,
        reference_data=reference_data,
        feature_columns=feature_columns,
    )

    print("Step 6: Create deterministic balanced samples")
    reference_sample, current_sample = (
        create_balanced_samples(
            reference_data=reference_data,
            current_data=current_data,
        )
    )

    print("Step 7: Generate feature drift report")
    feature_result = run_feature_drift_report(
        reference_sample=reference_sample,
        current_sample=current_sample,
        feature_columns=feature_columns,
    )

    feature_report_paths = save_report(
        result=feature_result,
        report_name="feature_drift_report",
    )

    print("Step 8: Generate prediction drift report")
    prediction_result = run_prediction_drift_report(
        reference_sample=reference_sample,
        current_sample=current_sample,
    )

    prediction_report_paths = save_report(
        result=prediction_result,
        report_name="prediction_drift_report",
    )

    print("Step 9: Extract alert-ready drift results")
    drift_results = build_drift_monitoring_summary(
        feature_report=feature_result.dict(),
        prediction_report=prediction_result.dict(),
        feature_columns=feature_columns,
        score_column=SCORE_COLUMN,
        column_threshold=DRIFT_THRESHOLD,
        dataset_drift_share=DRIFT_SHARE,
    )

    print("Step 10: Build monitoring provenance")
    provenance = build_monitoring_provenance(
        metadata_path=METADATA_PATH,
        model_path=MODEL_PATH,
        training_path=training_path,
        scoring_path=scoring_path,
        scores_path=SCORES_PATH,
        feature_columns=feature_columns,
    )

    print("Step 11: Write compact monitoring summary")
    summary = write_monitoring_summary(
        metadata=metadata,
        reference_data=reference_data,
        current_data=current_data,
        reference_sample=reference_sample,
        current_sample=current_sample,
        feature_report_paths=feature_report_paths,
        prediction_report_paths=prediction_report_paths,
        provenance=provenance,
        drift_results=drift_results,
        score_difference=score_difference,
    )

    print()
    print("GOOD: Evidently monitoring reports created")
    print(
        "Reference visitors:",
        f"{len(reference_data):,}",
    )
    print(
        "Current visitors:",
        f"{len(current_data):,}",
    )
    print(
        "Monitoring sample per population:",
        f"{len(reference_sample):,}",
    )
    print(
        "Maximum saved-score difference:",
        f"{score_difference:.10f}",
    )
    print(
        "Feature report:",
        relative_path(
            feature_report_paths["html"]
        ),
    )
    print(
        "Prediction report:",
        relative_path(
            prediction_report_paths["html"]
        ),
    )
    print(
        "Monitoring summary:",
        relative_path(SUMMARY_PATH),
    )

    # Read the compact alert-ready values from the generated summary.
    feature_drift = summary["drift_results"]["feature_drift"]
    prediction_drift = summary["drift_results"]["prediction_drift"]
    alert = summary["drift_results"]["alert"]

    print(
        "Drifted features:",
        (
            f"{feature_drift['drifted_feature_count']}"
            f"/{feature_drift['feature_count']}"
        ),
    )
    print(
        "Prediction PSI:",
        f"{prediction_drift['psi']:.10f}",
    )
    print(
        "Monitoring alert:",
        alert["level"],
    )

    mlflow_lineage = summary["provenance"]["mlflow_champion"]

    print(
        "MLflow model version:",
        (
            mlflow_lineage["registered_model_version"]
            if mlflow_lineage
            else "not available"
        ),
    )

    return summary


if __name__ == "__main__":
    generate_monitoring_reports()
