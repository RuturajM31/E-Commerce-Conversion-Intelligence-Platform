import json

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.models import score_export


class FakeFinalModel:
    """Return stable probabilities without running a real model."""

    def predict_proba(self, features):
        scores = np.array([0.20, 0.80])

        return np.column_stack([1 - scores, scores])


def build_feature_table():
    """Create two rows using the canonical production features."""

    return pd.DataFrame(
        [
            [5, 3, 1, 2, 100, 0.25, 5 / 3],
            [10, 7, 2, 4, 200, 0.25, 2],
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )


def test_final_scores_and_manifest_are_saved(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "final_champion_model.joblib"
    metadata_path = tmp_path / "final_champion_metadata.json"
    score_path = tmp_path / "final_champion_visitor_scores.csv"
    manifest_path = tmp_path / "final_champion_score_manifest.json"

    model_path.write_bytes(b"test-model-artifact")

    metadata_path.write_text(
        json.dumps(
            {
                "final_model_name": "Test Final Model",
                "feature_columns": MODEL_FEATURE_COLUMNS,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        score_export,
        "FINAL_MODEL_PATH",
        model_path,
    )
    monkeypatch.setattr(
        score_export,
        "FINAL_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        score_export,
        "FINAL_SCORE_PATH",
        score_path,
    )
    monkeypatch.setattr(
        score_export,
        "FINAL_SCORE_MANIFEST_PATH",
        manifest_path,
    )

    score_table, manifest = (
        score_export.save_final_champion_scores(
            final_model=FakeFinalModel(),
            X=build_feature_table(),
            visitor_ids=pd.Series([101, 202]),
            threshold=0.50,
        )
    )

    assert score_path.exists()
    assert manifest_path.exists()
    assert score_table["visitorid"].tolist() == [101, 202]
    assert score_table["predicted_conversion"].tolist() == [0, 1]
    assert manifest["model_generation"] == "final_champion"
    assert manifest["row_count"] == 2
    assert len(manifest["model_sha256"]) == 64
    assert len(manifest["score_sha256"]) == 64


def test_score_export_rejects_misaligned_visitor_ids():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        score_export.validate_score_inputs(
            X=build_feature_table(),
            visitor_ids=pd.Series([101]),
            threshold=0.50,
        )
