"""Tests for one consolidated experiment readiness page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.experiment_tracking_data import (
    ExperimentTrackingBundle,
    build_parameter_density,
    filter_ecommerce_runs,
)
from src.visualization.experiment_tracking_visuals import (
    generate_experiment_tracking_visual_package,
)


def build_runs() -> pd.DataFrame:
    """Create mixed ecommerce and prompt/demo evidence."""

    start = pd.Timestamp("2026-06-17T08:00:00Z")
    return pd.DataFrame(
        [
            {
                "run_id": "ecom-1",
                "run_name": "ecommerce-candidate",
                "experiment_id": "1",
                "experiment_name": "ecommerce",
                "start_time": start,
                "end_time": start,
                "status": "FINISHED",
                "lifecycle_stage": "active",
                "metric__validation_pr_auc": 0.08,
                "metric__validation_roc_auc": 0.80,
                "tag__project": "ecommerce",
            },
            {
                "run_id": "demo-1",
                "run_name": "trace-level-evaluation",
                "experiment_id": "2",
                "experiment_name": "MLflow Demo",
                "start_time": start,
                "end_time": start,
                "status": "FINISHED",
                "lifecycle_stage": "active",
                "metric__correctness_mean": 0.90,
                "metric__groundedness_mean": 0.88,
                "tag__mlflow.runName": "mlflow-demo.prompts.trace",
            },
        ]
    )


def build_bundle() -> ExperimentTrackingBundle:
    """Create a deterministic readiness bundle."""

    runs = build_runs()
    project_runs = filter_ecommerce_runs(
        runs,
        lineage={
            "run_id": "ecom-1",
            "registered_model_name": "ecommerce-conversion-champion",
        },
        registry_run_ids={"ecom-1"},
    )
    versions = pd.DataFrame(
        {
            "model_name": ["ecommerce-conversion-champion"],
            "version": [3],
            "run_id": ["ecom-1"],
            "creation_time": [pd.Timestamp("2026-06-17T08:00:00Z")],
            "alias": ["champion"],
        }
    )
    return ExperimentTrackingBundle(
        runs=runs,
        project_runs=project_runs,
        comparable_runs=project_runs.iloc[0:0],
        metric_columns=[
            "metric__validation_pr_auc",
            "metric__validation_roc_auc",
        ],
        primary_metric="metric__validation_pr_auc",
        model_versions=versions,
        experiments=pd.DataFrame(),
        lineage={},
        counts={
            "runs_total": 2,
            "verified_ecommerce_runs": 1,
            "excluded_runs": 1,
            "verified_registry_versions": 1,
            "shared_metric_keys": 2,
        },
        parameter_density=build_parameter_density(
            project_runs.iloc[0:0]
        ),
        comparison_ready=False,
        comparison_reason=(
            "Clean comparable ecommerce evidence is insufficient."
        ),
        i03_ready=False,
    )


def test_demo_prompt_run_is_excluded() -> None:
    """Prompt/demo runs remain excluded even if present in registry IDs."""

    filtered = filter_ecommerce_runs(
        build_runs(),
        lineage={"run_id": "ecom-1"},
        registry_run_ids={"ecom-1", "demo-1"},
    )

    assert filtered["run_id"].tolist() == ["ecom-1"]


def test_one_readiness_page_replaces_four_placeholders(
    tmp_path: Path,
) -> None:
    """The final experiment package should contain only one readiness page."""

    manifest = generate_experiment_tracking_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    assert manifest["supported_visuals"] == []
    assert set(manifest["conditional_visuals"]) == {
        "MLV-I01",
        "MLV-I02",
        "MLV-I03",
        "MLV-I04",
        "MLV-I05",
    }
    assert "artifacts" not in manifest
    assert (
        tmp_path / manifest["readiness_artifact"]
    ).exists()

    qa = manifest["qa"][
        "experiment_tracking_readiness"
    ]
    assert qa["passed"] is True
    assert qa["width_px"] >= 1600
    assert qa["height_px"] >= 850


def test_exports_contain_no_demo_metrics(
    tmp_path: Path,
) -> None:
    """Supporting evidence must not contain prompt/demo tokens."""

    manifest = generate_experiment_tracking_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    combined = ""

    for relative_path in (
        manifest["supporting_files"].values()
    ):
        path = tmp_path / relative_path

        if path.suffix.lower() in {
            ".csv",
            ".md",
            ".json",
        }:
            combined += path.read_text(
                encoding="utf-8"
            ).lower()

    for token in (
        "correctness_mean",
        "groundedness_mean",
        "mlflow-demo.prompts",
        "trace-level-evaluation",
    ):
        assert token not in combined
