"""Tests for MLV-A01 to MLV-A04 champion-selection visuals.

Why this file exists:
    The first ML Visual Intelligence package must remain stable, use real
    source contracts, and produce readable files before MLflow logging.

Inputs:
    Small synthetic comparison and metadata files created in a temporary
    project folder.

Outputs:
    Test evidence for schema checks, model preparation, artifact creation,
    output resolution, and actual-number interpretation files.

Used next:
    Local validation and CI run these tests before the package is committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.visualization.champion_selection_visuals import (
    build_generalisation_table,
    generate_champion_visual_package,
    prepare_model_comparison,
    validate_comparison_schema,
)


def build_comparison_rows() -> pd.DataFrame:
    """Create a small realistic validation and holdout comparison table."""

    rows = [
        {
            "evaluation_split": "validation",
            "model_name": "Tuned XGBoost",
            "model_family": "Boosting",
            "training_rows": 100_000,
            "deployable": True,
            "pr_auc": 0.063,
            "roc_auc": 0.803,
            "best_threshold": 0.98,
            "best_precision": 0.175,
            "best_recall": 0.126,
            "best_f1_score": 0.146,
            "predicted_positive_rate_at_best_threshold": 0.0010,
            "notes": "Saved champion",
            "business_score": 0.145,
        },
        {
            "evaluation_split": "validation",
            "model_name": "Tuned Random Forest",
            "model_family": "Random Forest",
            "training_rows": 100_000,
            "deployable": True,
            "pr_auc": 0.065,
            "roc_auc": 0.747,
            "best_threshold": 0.85,
            "best_precision": 0.142,
            "best_recall": 0.143,
            "best_f1_score": 0.143,
            "predicted_positive_rate_at_best_threshold": 0.0014,
            "notes": "Challenger",
            "business_score": 0.138,
        },
        {
            "evaluation_split": "validation",
            "model_name": "Probability Average Ensemble",
            "model_family": "Ensemble check",
            "training_rows": 100_000,
            "deployable": False,
            "pr_auc": 0.066,
            "roc_auc": 0.793,
            "best_threshold": 0.90,
            "best_precision": 0.141,
            "best_recall": 0.141,
            "best_f1_score": 0.141,
            "predicted_positive_rate_at_best_threshold": 0.0014,
            "notes": "Comparison only",
            "business_score": 0.140,
        },
        {
            "evaluation_split": "final_holdout",
            "model_name": "Tuned XGBoost",
            "model_family": "Boosting",
            "training_rows": 100_000,
            "deployable": True,
            "pr_auc": 0.152,
            "roc_auc": 0.848,
            "best_threshold": 0.98,
            "best_precision": 0.286,
            "best_recall": 0.208,
            "best_f1_score": 0.240,
            "predicted_positive_rate_at_best_threshold": 0.0009,
            "notes": "Untouched holdout",
            "business_score": 0.210,
        },
        {
            "evaluation_split": "final_holdout",
            "model_name": "Tuned Random Forest",
            "model_family": "Random Forest",
            "training_rows": 100_000,
            "deployable": True,
            "pr_auc": 0.110,
            "roc_auc": 0.800,
            "best_threshold": 0.85,
            "best_precision": 0.220,
            "best_recall": 0.190,
            "best_f1_score": 0.204,
            "predicted_positive_rate_at_best_threshold": 0.0011,
            "notes": "Untouched holdout",
            "business_score": 0.180,
        },
    ]

    return pd.DataFrame(rows)


def build_metadata() -> dict:
    """Create champion metadata using the same contract as the project."""

    return {
        "final_model_name": "Tuned XGBoost",
        "model_family": "Boosting",
        "best_threshold": 0.98,
        "validation_metrics": {
            "business_score": 0.145,
            "pr_auc": 0.063,
            "roc_auc": 0.803,
            "precision": 0.175,
            "recall": 0.126,
            "f1_score": 0.146,
            "threshold": 0.98,
        },
        "final_holdout_metrics": {
            "evaluation_split": "final_holdout",
            "model_name": "Tuned XGBoost",
            "threshold": 0.98,
            "rows": 248_639,
            "positive_rows": 318,
            "predicted_positive_rows": 231,
            "pr_auc": 0.152,
            "roc_auc": 0.848,
            "precision": 0.286,
            "recall": 0.208,
            "f1_score": 0.240,
        },
    }


def write_test_sources(project_root: Path) -> None:
    """Write temporary source files consumed by the package generator."""

    # Create the same repository folders used by the production code.
    table_dir = project_root / "reports" / "tables"
    metadata_dir = project_root / "models" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    # Save the synthetic model-comparison table.
    build_comparison_rows().to_csv(
        table_dir / "final_true_champion_comparison.csv",
        index=False,
    )

    # Save champion identity and holdout metrics.
    (
        metadata_dir / "final_champion_metadata.json"
    ).write_text(
        json.dumps(build_metadata(), indent=2),
        encoding="utf-8",
    )

    # Save a minimal but realistic MLflow lineage record.
    (
        metadata_dir / "mlflow_champion_lineage.json"
    ).write_text(
        json.dumps(
            {
                "registered_model_name": (
                    "ecommerce-conversion-champion"
                ),
                "registered_model_alias": "champion",
                "registered_model_version": "3",
                "run_id": "e8a080c17e4541369a8dc905683e4791",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_prepare_model_comparison_marks_saved_champion() -> None:
    """The saved model identity should control the champion highlight."""

    # Arrange: create comparison rows and saved champion metadata.
    comparison = build_comparison_rows()
    metadata = build_metadata()

    # Act: prepare the fair same-split comparison table.
    prepared = prepare_model_comparison(
        comparison,
        metadata,
    )

    # Assert: only validation rows remain and the champion is explicit.
    assert set(prepared["evaluation_split"]) == {"validation"}
    assert prepared["is_champion"].sum() == 1
    assert (
        prepared.loc[
            prepared["is_champion"],
            "model_name",
        ].iloc[0]
        == "Tuned XGBoost"
    )


def test_generalisation_table_uses_authoritative_champion_metrics() -> None:
    """Champion validation and holdout values should come from metadata."""

    # Arrange: use the same realistic temporary source structures.
    comparison = build_comparison_rows()
    metadata = build_metadata()

    # Act: prepare validation-versus-holdout PR-AUC.
    result = build_generalisation_table(
        comparison,
        metadata,
    )

    champion = result.loc[
        result["model_name"] == "Tuned XGBoost"
    ].iloc[0]

    # Assert: metadata values replace any ambiguous duplicated source row.
    assert champion["validation_pr_auc"] == pytest.approx(0.063)
    assert champion["holdout_pr_auc"] == pytest.approx(0.152)
    assert champion["pr_auc_delta"] == pytest.approx(0.089)


def test_schema_validation_rejects_missing_metric() -> None:
    """A broken source contract must fail before visual generation."""

    # Arrange: remove one mandatory metric column.
    comparison = build_comparison_rows().drop(
        columns=["business_score"]
    )

    # Act and assert: the error identifies the missing contract.
    with pytest.raises(
        ValueError,
        match="business_score",
    ):
        validate_comparison_schema(comparison)


def test_generate_champion_visual_package(
    tmp_path: Path,
) -> None:
    """The complete package should create four clean visual artifacts."""

    # Arrange: create a temporary project with realistic source files.
    write_test_sources(tmp_path)

    # Act: generate all visuals and supporting evidence.
    manifest = generate_champion_visual_package(
        project_root=tmp_path,
    )

    # Assert: every visual passed the shared quality checks.
    assert set(manifest["qa"]) == {
        "MLV-A01",
        "MLV-A02",
        "MLV-A03",
        "MLV-A04",
    }

    for qa_result in manifest["qa"].values():
        assert qa_result["passed"] is True
        assert qa_result["width_px"] >= 1_600
        assert qa_result["height_px"] >= 850

    # Assert: every reported visual path exists and is a real PNG.
    for relative_path in manifest["artifacts"].values():
        output_file = tmp_path / relative_path
        assert output_file.exists()
        assert output_file.stat().st_size > 10_000

    # Assert: the interpretation and evidence files were also produced.
    support_files = manifest["supporting_files"]

    for key in (
        "ranking_csv",
        "generalisation_csv",
        "insights",
        "manifest",
    ):
        assert (tmp_path / support_files[key]).exists()

    insight_text = (
        tmp_path / support_files["insights"]
    ).read_text(encoding="utf-8")

    assert "248,639 holdout rows" in insight_text
    assert "318 positive rows" in insight_text
    assert "live conversion performance must not be invented" in insight_text
