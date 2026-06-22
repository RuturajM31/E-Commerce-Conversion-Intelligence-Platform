"""Tests for champion visual MLflow package validation.

Why this file exists:
    The logger must refuse incomplete or unapproved visual packages before it
    contacts MLflow. These tests cover the local safety checks independently
    of the running MLflow server.

Inputs:
    Temporary package folders and manifest files.

Outputs:
    Test evidence for approved-package validation and expected artifact paths.

Used next:
    Local validation runs these tests before the logger is committed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "log_champion_visuals_to_mlflow.py"
)


def load_logger_module():
    """Load the script as a Python module for pure-function testing."""

    specification = importlib.util.spec_from_file_location(
        "champion_visual_mlflow_logger",
        SCRIPT_PATH,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load MLflow visual logger.")

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

    # Write every required package file with non-empty content.
    for filename in module.EXPECTED_PACKAGE_FILES:
        path = package_dir / filename

        if path.suffix == ".json":
            continue

        path.write_bytes(b"validated visual evidence")

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
        "package": "Champion selection",
        "qa": qa,
    }

    (
        package_dir / "champion_visual_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # The manual review image must be large enough to count as real evidence.
    review_sheet.write_bytes(b"x" * 20_000)


def test_validate_approved_package_passes(
    tmp_path: Path,
) -> None:
    """A complete approved package should pass local validation."""

    module = load_logger_module()
    write_approved_package(tmp_path)

    manifest = module.validate_approved_package(tmp_path)

    assert manifest["package"] == "Champion selection"
    assert set(manifest["qa"]) == module.EXPECTED_VISUAL_IDS


def test_validate_approved_package_rejects_failed_visual(
    tmp_path: Path,
) -> None:
    """A failed visual must block MLflow logging."""

    module = load_logger_module()
    write_approved_package(
        tmp_path,
        failed_visual="MLV-A03",
    )

    with pytest.raises(
        ValueError,
        match="MLV-A03",
    ):
        module.validate_approved_package(tmp_path)


def test_validate_approved_package_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing approved artifact must block MLflow logging."""

    module = load_logger_module()
    write_approved_package(tmp_path)

    missing_file = (
        tmp_path
        / module.PACKAGE_DIR
        / "mlv-a04_champion_scorecard_identity.png"
    )
    missing_file.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="mlv-a04",
    ):
        module.validate_approved_package(tmp_path)


def test_expected_mlflow_paths_include_review_sheet() -> None:
    """Upload verification must include package files and manual QA evidence."""

    module = load_logger_module()

    paths = module.expected_mlflow_artifact_paths()

    assert len(paths) == 9
    assert (
        "visual_intelligence/champion_selection/"
        "mlv-a01_model_performance_frontier.png"
    ) in paths
    assert (
        "visual_intelligence/champion_selection/qa/"
        "champion_visual_review_sheet.png"
    ) in paths
