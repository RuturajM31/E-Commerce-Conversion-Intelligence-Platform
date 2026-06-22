# test_production_model.py
# Verify that all workflows can resolve one safe production model pair.

from pathlib import Path

import pytest

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.models.production_model import (
    load_production_bundle,
    resolve_production_artifacts,
    validate_metadata_features,
)


def create_empty_file(path: Path) -> None:
    """Create a small placeholder file for resolver tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_final_complete_pair_has_priority(tmp_path):
    """The final champion must win when both complete pairs exist."""

    final_model = tmp_path / "final_model.joblib"
    final_metadata = tmp_path / "final_metadata.json"
    old_model = tmp_path / "old_model.joblib"
    old_metadata = tmp_path / "old_metadata.json"

    for path in [
        final_model,
        final_metadata,
        old_model,
        old_metadata,
    ]:
        create_empty_file(path)

    selected = resolve_production_artifacts(
        [
            ("final_champion", final_model, final_metadata),
            ("initial_champion", old_model, old_metadata),
        ]
    )

    assert selected == (
        "final_champion",
        final_model,
        final_metadata,
    )


def test_incomplete_final_pair_uses_complete_fallback(tmp_path):
    """Never combine a final model with missing or older metadata."""

    final_model = tmp_path / "final_model.joblib"
    final_metadata = tmp_path / "missing_final_metadata.json"
    old_model = tmp_path / "old_model.joblib"
    old_metadata = tmp_path / "old_metadata.json"

    create_empty_file(final_model)
    create_empty_file(old_model)
    create_empty_file(old_metadata)

    selected = resolve_production_artifacts(
        [
            ("final_champion", final_model, final_metadata),
            ("initial_champion", old_model, old_metadata),
        ]
    )

    assert selected == (
        "initial_champion",
        old_model,
        old_metadata,
    )


def test_missing_complete_pair_raises_clear_error(tmp_path):
    """Production loading must fail clearly when no valid pair exists."""

    with pytest.raises(
        FileNotFoundError,
        match="No complete production model",
    ):
        resolve_production_artifacts(
            [
                (
                    "final_champion",
                    tmp_path / "missing_model.joblib",
                    tmp_path / "missing_metadata.json",
                )
            ]
        )


def test_metadata_must_use_canonical_feature_order():
    """Metadata cannot silently use another feature list or order."""

    metadata = {
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    result = validate_metadata_features(metadata)

    assert result == MODEL_FEATURE_COLUMNS


def test_local_final_champion_bundle_loads():
    """The checked project artifacts must form one valid final bundle."""

    bundle = load_production_bundle()

    assert bundle["generation"] == "final_champion"
    assert bundle["model_path"].name == "final_champion_model.joblib"
    assert bundle["metadata_path"].name == "final_champion_metadata.json"
    assert bundle["feature_columns"] == MODEL_FEATURE_COLUMNS