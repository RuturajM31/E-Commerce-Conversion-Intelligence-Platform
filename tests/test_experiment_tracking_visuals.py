"""Tests for experiment-tracking visuals I01, I02, I04, and I05."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.experiment_tracking_data import (
    ExperimentTrackingBundle,
    build_parameter_density,
    select_metric_columns,
    select_primary_metric,
)
from src.visualization.experiment_tracking_visuals import (
    generate_experiment_tracking_visual_package,
)


def build_bundle() -> ExperimentTrackingBundle:
    """Create deterministic MLflow-like run and registry evidence."""

    start = pd.Timestamp(
        "2026-06-17T08:00:00Z"
    )
    run_rows = []

    for index in range(8):
        run_rows.append(
            {
                "run_id": f"run-{index}",
                "run_name": f"candidate-{index}",
                "experiment_id": str(index % 2),
                "experiment_name": (
                    "champion-selection"
                    if index % 2 == 0
                    else "production-registration"
                ),
                "start_time": (
                    start
                    + pd.Timedelta(
                        minutes=index * 18
                    )
                ),
                "end_time": (
                    start
                    + pd.Timedelta(
                        minutes=index * 18 + 5
                    )
                ),
                "status": "FINISHED",
                "lifecycle_stage": "active",
                "metric__pr_auc": 0.08 + index * 0.012,
                "metric__roc_auc": 0.74 + index * 0.018,
                "metric__f1_score": 0.12 + index * 0.014,
                "metric__business_score": 0.10 + index * 0.011,
                "param__max_depth": str(
                    [3, 4, 5, 3, 4, 5, 3, 4][index]
                ),
                "param__learning_rate": str(
                    [0.03, 0.05, 0.08, 0.03, 0.05, 0.08, 0.03, 0.05][index]
                ),
            }
        )

    runs = pd.DataFrame(run_rows)
    metric_columns = [
        "metric__pr_auc",
        "metric__roc_auc",
        "metric__f1_score",
        "metric__business_score",
    ]
    primary_metric = "metric__pr_auc"

    version_rows = []

    for index in range(6):
        version_rows.append(
            {
                "model_name": (
                    "ecommerce-conversion-champion"
                    if index >= 2
                    else "conversion-challenger"
                ),
                "version": index + 1,
                "run_id": f"run-{index}",
                "creation_time": (
                    start
                    + pd.Timedelta(
                        hours=index
                    )
                ),
                "current_stage": "",
                "status": "READY",
                "alias": (
                    "champion"
                    if index == 5
                    else (
                        "challenger"
                        if index == 1
                        else ""
                    )
                ),
                "run_name": f"candidate-{index}",
                "metric__pr_auc": 0.08 + index * 0.012,
            }
        )

    versions = pd.DataFrame(version_rows)
    parameter_density = build_parameter_density(
        runs
    )

    return ExperimentTrackingBundle(
        runs=runs,
        comparable_runs=runs,
        metric_columns=metric_columns,
        primary_metric=primary_metric,
        model_versions=versions,
        experiments=pd.DataFrame(
            {
                "experiment_id": ["0", "1"],
                "name": [
                    "champion-selection",
                    "production-registration",
                ],
            }
        ),
        lineage={
            "run_id": "run-5",
            "registered_model_name": (
                "ecommerce-conversion-champion"
            ),
            "registered_model_version": "6",
            "registered_model_alias": "champion",
        },
        counts={
            "experiments": 2,
            "runs": 8,
            "metrics": 32,
            "params": 16,
            "tags": 10,
            "registered_models": 2,
            "model_versions": 6,
            "registered_model_aliases": 2,
            "comparable_runs": 8,
        },
        parameter_density=parameter_density,
        i03_ready=True,
    )


def test_metric_selection_keeps_exact_shared_keys() -> None:
    """Only shared non-constant metric keys should be selected."""

    runs = build_bundle().runs.copy()
    runs["metric__constant"] = 1.0
    selected = select_metric_columns(runs)

    assert "metric__pr_auc" in selected
    assert "metric__roc_auc" in selected
    assert "metric__constant" not in selected


def test_primary_metric_prefers_pr_auc() -> None:
    """PR-AUC should be the preferred timeline metric when available."""

    selected = [
        "metric__roc_auc",
        "metric__pr_auc",
        "metric__f1_score",
    ]

    assert select_primary_metric(selected) == (
        "metric__pr_auc"
    )


def test_parameter_density_detects_two_dense_parameters() -> None:
    """Two repeated numeric parameter grids should be marked dense."""

    density = build_bundle().parameter_density

    assert density["dense_candidate"].sum() == 2
    assert set(
        density.loc[
            density["dense_candidate"],
            "parameter",
        ]
    ) == {
        "max_depth",
        "learning_rate",
    }


def test_parameter_density_handles_text_only_parameters() -> None:
    """Text-only MLflow parameters should return a valid empty schema."""

    runs = pd.DataFrame(
        {
            "run_id": ["run-1", "run-2"],
            "param__model_family": ["xgboost", "random_forest"],
            "param__dataset_name": ["retail", "retail"],
        }
    )

    density = build_parameter_density(runs)

    assert density.empty
    assert density.columns.tolist() == [
        "parameter",
        "non_null_runs",
        "unique_values",
        "minimum",
        "maximum",
        "dense_candidate",
    ]


def test_generate_experiment_tracking_package(
    tmp_path: Path,
) -> None:
    """The package should create four clean visuals and evidence."""

    manifest = generate_experiment_tracking_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    assert set(manifest["qa"]) == {
        "MLV-I01",
        "MLV-I02",
        "MLV-I04",
        "MLV-I05",
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
        "comparable_runs",
        "registry_versions",
        "parameter_density",
        "insights",
        "i03_status",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    insight_text = (
        tmp_path / support["insights"]
    ).read_text(encoding="utf-8")

    assert "exact shared metric keys" in insight_text
    assert "No sparse surface is fabricated" in insight_text
