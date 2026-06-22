"""Tests for MLV-B01 and MLV-B04 threshold-decision visuals.

Why this file exists:
    The threshold package must use the saved champion, reconstruct confusion
    counts exactly, and produce clean evidence files before MLflow logging.

Inputs:
    Small synthetic threshold and metadata sources in a temporary project.

Outputs:
    Test evidence for schema checks, selected-threshold logic, count
    reconstruction, package generation, and export quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.visualization.threshold_decision_visuals import (
    derive_confusion_counts,
    generate_threshold_visual_package,
    get_selected_threshold_row,
    prepare_champion_thresholds,
    validate_threshold_schema,
)


def build_threshold_rows() -> pd.DataFrame:
    """Create a realistic multi-model threshold table."""

    rows = []

    champion_values = [
        (0.50, 0.020, 0.800, 0.039, 0.0550),
        (0.80, 0.080, 0.500, 0.138, 0.0080),
        (0.95, 0.150, 0.220, 0.178, 0.0018),
        (0.98, 0.175, 0.126, 0.146, 0.0010),
        (0.99, 0.250, 0.025, 0.045, 0.0002),
    ]

    for threshold, precision, recall, f1, share in champion_values:
        rows.append(
            {
                "model_name": "Tuned XGBoost",
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predicted_positive_rate": share,
            }
        )

    rows.append(
        {
            "model_name": "Tuned Random Forest",
            "threshold": 0.98,
            "precision": 0.140,
            "recall": 0.100,
            "f1_score": 0.117,
            "predicted_positive_rate": 0.0011,
        }
    )

    return pd.DataFrame(rows)


def build_metadata() -> dict:
    """Create final champion metadata using the production contract."""

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
            "positive_rate": 0.0012789626727906,
            "predicted_positive_rows": 231,
            "predicted_positive_rate": 0.000929057790612,
            "pr_auc": 0.1518661918567523,
            "roc_auc": 0.848444891741996,
            "precision": 0.2857142857142857,
            "recall": 0.2075471698113207,
            "f1_score": 0.2404371584699453,
        },
    }


def write_test_sources(project_root: Path) -> None:
    """Write temporary source files used by the package generator."""

    table_dir = project_root / "reports" / "tables"
    metadata_dir = project_root / "models" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    build_threshold_rows().to_csv(
        table_dir / "final_true_champion_thresholds.csv",
        index=False,
    )

    (
        metadata_dir / "final_champion_metadata.json"
    ).write_text(
        json.dumps(build_metadata(), indent=2),
        encoding="utf-8",
    )


def test_prepare_champion_thresholds_filters_saved_model() -> None:
    """Only the saved champion should remain in the prepared curve."""

    prepared = prepare_champion_thresholds(
        build_threshold_rows(),
        build_metadata(),
    )

    assert set(prepared["model_name"]) == {"Tuned XGBoost"}
    assert prepared["threshold"].tolist() == sorted(
        prepared["threshold"].tolist()
    )


def test_selected_threshold_row_matches_metadata() -> None:
    """The selected row should be the exact saved threshold."""

    prepared = prepare_champion_thresholds(
        build_threshold_rows(),
        build_metadata(),
    )

    selected = get_selected_threshold_row(
        prepared,
        build_metadata(),
    )

    assert selected["threshold"] == pytest.approx(0.98)
    assert selected["precision"] == pytest.approx(0.175)


def test_confusion_counts_reconstruct_saved_metrics() -> None:
    """Aggregate holdout metrics should produce exact integer counts."""

    confusion = derive_confusion_counts(build_metadata())

    assert confusion["true_positive"] == 66
    assert confusion["false_positive"] == 165
    assert confusion["false_negative"] == 252
    assert confusion["true_negative"] == 248_156


def test_schema_validation_rejects_missing_column() -> None:
    """A broken threshold source must fail before rendering."""

    broken = build_threshold_rows().drop(
        columns=["predicted_positive_rate"]
    )

    with pytest.raises(
        ValueError,
        match="predicted_positive_rate",
    ):
        validate_threshold_schema(broken)


def test_generate_threshold_visual_package(
    tmp_path: Path,
) -> None:
    """The complete package should produce two clean visual artifacts."""

    write_test_sources(tmp_path)

    manifest = generate_threshold_visual_package(
        project_root=tmp_path,
    )

    assert set(manifest["qa"]) == {
        "MLV-B01",
        "MLV-B04",
    }

    for qa_result in manifest["qa"].values():
        assert qa_result["passed"] is True
        assert qa_result["width_px"] >= 1_600
        assert qa_result["height_px"] >= 850

    for relative_path in manifest["artifacts"].values():
        output_file = tmp_path / relative_path
        assert output_file.exists()
        assert output_file.stat().st_size > 10_000

    support = manifest["supporting_files"]

    for key in (
        "selected_threshold_evidence",
        "confusion_counts",
        "insights",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    insight_text = (
        tmp_path / support["insights"]
    ).read_text(encoding="utf-8")

    assert "66" in insight_text
    assert "165" in insight_text
    assert "252" in insight_text
    assert "248,156" in insight_text
    assert "production performance remains blocked" in insight_text
