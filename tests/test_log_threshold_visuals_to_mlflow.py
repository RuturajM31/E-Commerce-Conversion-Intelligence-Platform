"""Tests for threshold visual MLflow package validation.

Why this file exists:
    The logger must reject incomplete or unapproved threshold packages before
    it contacts MLflow.

Inputs:
    Temporary package folders and manifest files.

Outputs:
    Test evidence for approval checks and expected MLflow artifact paths.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "log_threshold_visuals_to_mlflow.py"
)


def load_logger_module():
    """Load the logging script as a module for pure-function testing."""

    specification = importlib.util.spec_from_file_location(
        "threshold_visual_mlflow_logger",
        SCRIPT_PATH,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load threshold visual logger.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


def write_approved_package(
    project_root: Path,
    *,
    failed_visual: str | None = None,
) -> None:
    """Create a minimal approved package in a temporary project."""

    module = load_logger_module()

    package_dir = project_root / module.PACKAGE_DIR
    review_sheet = project_root / module.REVIEW_SHEET_PATH

    package_dir.mkdir(parents=True)
    review_sheet.parent.mkdir(parents=True)

    # Create every expected package file with non-empty content.
    for filename in module.EXPECTED_PACKAGE_FILES:
        path = package_dir / filename

        if path.suffix == ".json":
            continue

        path.write_bytes(b"validated threshold evidence")

    qa = {
        visual_id: {
            "passed": visual_id != failed_visual,
            "width_px": 2_200,
            "height_px": 1_350,
        }
        for visual_id in module.EXPECTED_VISUAL_IDS
    }

    manifest = {
        "category": "ML Visual Intelligence",
        "package": "Threshold decisions",
        "qa": qa,
    }

    (
        package_dir / "threshold_visual_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    review_sheet.write_bytes(b"x" * 20_000)


def test_validate_approved_package_passes(
    tmp_path: Path,
) -> None:
    """A complete approved package should pass validation."""

    module = load_logger_module()
    write_approved_package(tmp_path)

    manifest = module.validate_approved_package(tmp_path)

    assert manifest["package"] == "Threshold decisions"
    assert set(manifest["qa"]) == module.EXPECTED_VISUAL_IDS


def test_validate_approved_package_rejects_failed_visual(
    tmp_path: Path,
) -> None:
    """A failed visual must block MLflow logging."""

    module = load_logger_module()
    write_approved_package(
        tmp_path,
        failed_visual="MLV-B04",
    )

    with pytest.raises(
        ValueError,
        match="MLV-B04",
    ):
        module.validate_approved_package(tmp_path)


def test_validate_approved_package_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing visual artifact must block MLflow logging."""

    module = load_logger_module()
    write_approved_package(tmp_path)

    missing_file = (
        tmp_path
        / module.PACKAGE_DIR
        / "mlv-b01_threshold_decision_studio.png"
    )
    missing_file.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="mlv-b01",
    ):
        module.validate_approved_package(tmp_path)


def test_expected_mlflow_paths_include_review_sheet() -> None:
    """Upload verification must include package and QA evidence."""

    module = load_logger_module()

    paths = module.expected_mlflow_artifact_paths()

    assert len(paths) == 7
    assert (
        "visual_intelligence/threshold_decisions/"
        "mlv-b04_confusion_matrix_decision_map.png"
    ) in paths
    assert (
        "visual_intelligence/threshold_decisions/qa/"
        "threshold_visual_review_sheet.png"
    ) in paths
