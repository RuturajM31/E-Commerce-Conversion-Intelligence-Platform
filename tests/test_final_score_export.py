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
    return pd.DataFrame(
        [
            [5, 3, 1, 2, 100, 0.25, 5 / 3],
            [10, 7, 2, 4, 200, 0.25, 2],
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )


def create_artifacts(tmp_path):
    model_path = tmp_path / "final_champion_model.joblib"
    metadata_path = tmp_path / "final_champion_metadata.json"
    score_path = tmp_path / "final_champion_visitor_scores.csv"
    manifest_path = tmp_path / "final_champion_score_manifest.json"

    model_path.write_bytes(b"test-model-artifact")

    metadata = {
        "final_model_name": "Test Final Model",
        "best_threshold": 0.50,
        "feature_columns": MODEL_FEATURE_COLUMNS,
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    bundle = {
        "model": FakeFinalModel(),
        "metadata": metadata,
        "generation": "final_champion",
        "model_path": model_path,
        "metadata_path": metadata_path,
        "feature_columns": MODEL_FEATURE_COLUMNS.copy(),
    }

    return bundle, score_path, manifest_path


def test_final_scores_and_manifest_are_saved(tmp_path):
    bundle, score_path, manifest_path = create_artifacts(tmp_path)
    visitor_ids = pd.Series([101, 202])

    score_table, manifest = score_export.save_final_champion_scores(
        final_model=bundle["model"],
        X=build_feature_table(),
        visitor_ids=visitor_ids,
        threshold=0.50,
        model_path=bundle["model_path"],
        metadata_path=bundle["metadata_path"],
        score_path=score_path,
        manifest_path=manifest_path,
        model_generation=bundle["generation"],
    )

    assert score_path.exists()
    assert manifest_path.exists()
    assert score_table["predicted_conversion"].tolist() == [0, 1]
    assert manifest["row_count"] == 2
    assert len(manifest["visitor_id_sha256"]) == 64


def test_valid_cached_scores_are_reused(tmp_path):
    bundle, score_path, manifest_path = create_artifacts(tmp_path)
    visitor_ids = pd.Series([101, 202])

    score_export.save_final_champion_scores(
        final_model=bundle["model"],
        X=build_feature_table(),
        visitor_ids=visitor_ids,
        threshold=0.50,
        model_path=bundle["model_path"],
        metadata_path=bundle["metadata_path"],
        score_path=score_path,
        manifest_path=manifest_path,
        model_generation=bundle["generation"],
    )

    cached_scores, message = score_export.load_valid_cached_scores(
        bundle,
        visitor_ids,
        score_path=score_path,
        manifest_path=manifest_path,
    )

    assert cached_scores is not None
    assert len(cached_scores) == 2
    assert "validated successfully" in message


def test_tampered_score_file_is_rejected(tmp_path):
    bundle, score_path, manifest_path = create_artifacts(tmp_path)
    visitor_ids = pd.Series([101, 202])

    score_export.save_final_champion_scores(
        final_model=bundle["model"],
        X=build_feature_table(),
        visitor_ids=visitor_ids,
        threshold=0.50,
        model_path=bundle["model_path"],
        metadata_path=bundle["metadata_path"],
        score_path=score_path,
        manifest_path=manifest_path,
        model_generation=bundle["generation"],
    )

    tampered = pd.read_csv(score_path)
    tampered.loc[0, "purchase_intent_score"] = 0.99
    tampered.to_csv(score_path, index=False)

    cached_scores, message = score_export.load_valid_cached_scores(
        bundle,
        visitor_ids,
        score_path=score_path,
        manifest_path=manifest_path,
    )

    assert cached_scores is None
    assert "score file hash does not match" in message


def test_score_export_rejects_misaligned_visitor_ids():
    with pytest.raises(ValueError, match="same length"):
        score_export.validate_score_inputs(
            X=build_feature_table(),
            visitor_ids=pd.Series([101]),
            threshold=0.50,
        )
