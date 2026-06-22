"""Tests for deterministic monitoring provenance records."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.monitoring.monitoring_provenance import (
    build_monitoring_provenance,
    feature_schema_sha256,
    file_sha256,
)


def test_file_sha256_matches_known_content(
    tmp_path: Path,
) -> None:
    """The streaming hash must match a direct SHA-256 calculation."""

    # Create a small known file that acts like a monitoring artifact.
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "monitoring provenance\n",
        encoding="utf-8",
    )

    # Calculate the expected value directly from the file bytes.
    expected_hash = hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()

    # The production streaming function must return the same result.
    assert file_sha256(file_path) == expected_hash


def test_feature_schema_hash_is_order_sensitive() -> None:
    """Changing feature order must change the schema fingerprint."""

    first_hash = feature_schema_sha256(
        [
            "total_events",
            "view_count",
        ]
    )

    second_hash = feature_schema_sha256(
        [
            "view_count",
            "total_events",
        ]
    )

    # The production model treats these as different input schemas.
    assert first_hash != second_hash


def test_missing_file_raises_clear_error(
    tmp_path: Path,
) -> None:
    """Missing required artifacts must fail instead of being ignored."""

    missing_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Provenance file was not found",
    ):
        file_sha256(missing_path)


def test_build_monitoring_provenance_records_all_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The final record must include every monitoring input artifact."""

    files: dict[str, Path] = {}

    # Create five small stand-in files for the real project artifacts.
    for name in [
        "metadata",
        "model",
        "training",
        "scoring",
        "scores",
    ]:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(
            f"{name}-content".encode("utf-8")
        )
        files[name] = path

    # Use predictable registry lineage without reading the real MLflow file.
    monkeypatch.setattr(
        "src.monitoring.monitoring_provenance."
        "load_optional_mlflow_lineage",
        lambda: {
            "run_id": "test-run",
            "registered_model_version": "3",
        },
    )

    provenance = build_monitoring_provenance(
        metadata_path=files["metadata"],
        model_path=files["model"],
        training_path=files["training"],
        scoring_path=files["scoring"],
        scores_path=files["scores"],
        feature_columns=[
            "total_events",
            "view_count",
        ],
    )

    # Confirm the ordered schema was recorded.
    assert provenance["feature_schema"][
        "column_count"
    ] == 2

    # Confirm every required monitoring input was fingerprinted.
    assert set(provenance["files"]) == {
        "champion_metadata",
        "champion_model",
        "training_reference_source",
        "current_scoring_source",
        "production_scores",
    }

    # Confirm the optional registry lineage is carried forward.
    assert provenance["mlflow_champion"][
        "registered_model_version"
    ] == "3"

    # Every file record must contain a valid hash and non-empty size.
    for record in provenance["files"].values():
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
