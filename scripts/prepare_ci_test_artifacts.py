#!/usr/bin/env python3
"""Create small CI-only artifacts required by integration tests.

Simple idea:
    GitHub Actions downloads source code, but it does not receive large local
    model files or generated report CSV files because those files are ignored
    by Git.

    This script creates a tiny temporary model bundle and small report tables
    inside the disposable GitHub runner. The files exist only for automated
    tests and are never presented as production evidence.

Inputs:
    - data/sample/visitor_features_sample.csv
    - canonical feature logic from src.data.feature_engineering

Outputs:
    - models/trained/final_champion_model.joblib
    - models/metadata/final_champion_metadata.json
    - reports/tables/business_forecast_future.csv
    - reports/tables/business_forecast_comparison.csv
    - reports/tables/final_true_champion_comparison.csv
    - reports/tables/manual_model_comparison.csv
    - reports/tables/automl_benchmark_results.csv

Safety:
    The command-line entry point runs only when CI=true, unless --allow-local
    is explicitly supplied. This prevents accidental replacement of a real
    local production model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    prepare_model_features,
)
from src.models.model_config import TARGET_COLUMN  # noqa: E402


DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "visitor_features_sample.csv"


def load_ci_sample(sample_path: Path) -> pd.DataFrame:
    """Load the representative sample and validate required columns.

    Input:
        sample_path:
            Existing small sample CSV committed to Git.

    Output:
        A validated DataFrame containing base features and the binary target.

    Used next:
        The data is transformed with the same feature logic used by production.
    """

    if not sample_path.is_file():
        raise FileNotFoundError(f"CI sample data is missing: {sample_path}")

    data = pd.read_csv(sample_path)

    required_columns = {
        TARGET_COLUMN,
        *BASE_FEATURE_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(data.columns))

    if missing_columns:
        raise ValueError("CI sample data is missing columns: " + ", ".join(missing_columns))

    target_values = set(data[TARGET_COLUMN].dropna().astype(int).unique())

    if target_values != {0, 1}:
        raise ValueError("CI sample target must contain both classes 0 and 1.")

    return data


def train_ci_model(data: pd.DataFrame) -> Pipeline:
    """Train a small deterministic model using the canonical feature schema.

    This model proves that loading, feature order, and prediction code work.
    It is not a production champion and must not be used for business claims.
    """

    features = prepare_model_features(data[BASE_FEATURE_COLUMNS])
    target = data[TARGET_COLUMN].astype(int)

    if list(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("CI model features do not match the canonical feature schema.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, target)

    probabilities = model.predict_proba(features)[:, 1]

    if not ((probabilities >= 0).all() and (probabilities <= 1).all()):
        raise ValueError("CI model produced probabilities outside 0 to 1.")

    return model


def write_ci_model_bundle(
    project_root: Path,
    model: Pipeline,
) -> dict[str, Any]:
    """Write the temporary model and matching metadata pair.

    The filenames intentionally match the production resolver contract so the
    integration tests exercise the real loading path.
    """

    model_path = project_root / "models" / "trained" / "final_champion_model.joblib"
    metadata_path = project_root / "models" / "metadata" / "final_champion_metadata.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "artifact_scope": "ci_test_only",
        "purpose": ("Temporary GitHub Actions fixture. " "Not production performance evidence."),
        "final_model_name": "CI Logistic Regression Fixture",
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "best_threshold": 0.50,
        "model_family": "LogisticRegression",
        "training_data": "data/sample/visitor_features_sample.csv",
    }

    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def write_forecast_fixtures(project_root: Path) -> list[Path]:
    """Create tiny forecast tables with the real production schemas."""

    table_dir = project_root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    future_path = table_dir / "business_forecast_future.csv"
    comparison_path = table_dir / "business_forecast_comparison.csv"

    future = pd.DataFrame(
        [
            {
                "date": "2030-01-01",
                "target_name": "daily_conversions",
                "predicted_value": 12.0,
            },
            {
                "date": "2030-01-02",
                "target_name": "daily_conversions",
                "predicted_value": 13.0,
            },
        ]
    )

    comparison = pd.DataFrame(
        [
            {
                "target_name": "daily_conversions",
                "model_name": "Linear Regression",
                "is_best_model": True,
            },
            {
                "target_name": "daily_conversions",
                "model_name": "Naive Baseline",
                "is_best_model": False,
            },
        ]
    )

    future.to_csv(future_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    return [future_path, comparison_path]


def write_model_comparison_fixtures(
    project_root: Path,
) -> list[Path]:
    """Create small benchmark tables used by model-selection tests."""

    table_dir = project_root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    final_path = table_dir / "final_true_champion_comparison.csv"
    manual_path = table_dir / "manual_model_comparison.csv"
    automl_path = table_dir / "automl_benchmark_results.csv"

    comparison = pd.DataFrame(
        [
            {
                "model_name": "Logistic Regression",
                "business_score": 0.72,
                "pr_auc": 0.70,
                "best_f1_score": 0.68,
                "best_precision": 0.67,
                "is_selected": False,
            },
            {
                "model_name": "XGBoost",
                "business_score": 0.81,
                "pr_auc": 0.79,
                "best_f1_score": 0.75,
                "best_precision": 0.74,
                "is_selected": True,
            },
        ]
    )

    comparison.to_csv(final_path, index=False)
    comparison.iloc[[0]].to_csv(manual_path, index=False)
    comparison.iloc[[1]].to_csv(automl_path, index=False)

    return [final_path, manual_path, automl_path]


def validate_ci_outputs(
    project_root: Path,
    output_paths: list[Path],
) -> None:
    """Confirm every temporary artifact exists and contains data."""

    for output_path in output_paths:
        if not output_path.is_file():
            raise RuntimeError(f"Expected CI artifact is missing: {output_path}")

        if output_path.stat().st_size <= 0:
            raise RuntimeError(f"Expected CI artifact is empty: {output_path}")

    metadata_path = project_root / "models" / "metadata" / "final_champion_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("artifact_scope") != "ci_test_only":
        raise RuntimeError("CI metadata is not clearly marked as test-only.")


def prepare_ci_test_artifacts(
    project_root: Path = PROJECT_ROOT,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
) -> dict[str, Any]:
    """Build all temporary artifacts required by CI integration tests."""

    data = load_ci_sample(sample_path)
    model = train_ci_model(data)
    bundle = write_ci_model_bundle(project_root, model)

    output_paths = [
        bundle["model_path"],
        bundle["metadata_path"],
        *write_forecast_fixtures(project_root),
        *write_model_comparison_fixtures(project_root),
    ]

    validate_ci_outputs(project_root, output_paths)

    return {
        "artifact_scope": "ci_test_only",
        "row_count": int(len(data)),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "output_count": len(output_paths),
        "output_paths": [str(path.relative_to(project_root)) for path in output_paths],
    }


def parse_args() -> argparse.Namespace:
    """Read safe command-line options."""

    parser = argparse.ArgumentParser(description=("Create CI-only model and report fixtures."))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--sample-data",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
    )
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help=(
            "Allow execution outside GitHub Actions. "
            "Use only inside a disposable test directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Create fixtures only in CI or with explicit local permission."""

    args = parse_args()
    running_in_ci = os.environ.get("CI", "").strip().lower() == "true"

    if not running_in_ci and not args.allow_local:
        raise SystemExit(
            "Refusing to create CI fixtures outside CI. "
            "Use --allow-local only inside a disposable test copy."
        )

    project_root = args.project_root.resolve()
    sample_path = args.sample_data.resolve()

    result = prepare_ci_test_artifacts(
        project_root=project_root,
        sample_path=sample_path,
    )

    print("GOOD: CI-only test artifacts created")
    print(f"Sample rows: {result['row_count']}")
    print(f"Feature count: {result['feature_count']}")
    print(f"Generated artifacts: {result['output_count']}")
    print("Scope: ci_test_only")


if __name__ == "__main__":
    main()
