# score_export.py
# Save final production scores with evidence of the exact model that created them.
#
# Simple idea:
#   A score file is trustworthy only when we can identify its model,
#   metadata, features, threshold, and creation time.

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config.paths import (
    MODEL_METADATA_DIR,
    PROCESSED_DATA_DIR,
    TRAINED_MODELS_DIR,
)


FINAL_MODEL_PATH = TRAINED_MODELS_DIR / "final_champion_model.joblib"
FINAL_METADATA_PATH = MODEL_METADATA_DIR / "final_champion_metadata.json"

FINAL_SCORE_PATH = (
    PROCESSED_DATA_DIR / "final_champion_visitor_scores.csv"
)

FINAL_SCORE_MANIFEST_PATH = (
    MODEL_METADATA_DIR / "final_champion_score_manifest.json"
)


def calculate_file_sha256(path: Path) -> str:
    """Return a SHA-256 fingerprint for one saved artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> Dict:
    """Load required JSON metadata."""

    if not path.exists():
        raise FileNotFoundError(f"Required metadata is missing: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_score_inputs(
    X: pd.DataFrame,
    visitor_ids: pd.Series,
    threshold: float,
) -> None:
    """Validate data alignment and the production threshold."""

    if len(X) != len(visitor_ids):
        raise ValueError(
            "Feature rows and visitor IDs must have the same length."
        )

    if not 0 <= float(threshold) <= 1:
        raise ValueError(
            "Production threshold must be between 0 and 1."
        )


def save_final_champion_scores(
    final_model,
    X: pd.DataFrame,
    visitor_ids: pd.Series,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict]:
    """Score all visitors and save a traceable production artifact."""

    validate_score_inputs(X, visitor_ids, threshold)

    model_metadata = load_json(FINAL_METADATA_PATH)

    expected_features = model_metadata.get("feature_columns", [])

    if list(X.columns) != list(expected_features):
        raise ValueError(
            "Score-export features do not match final model metadata."
        )

    probabilities = final_model.predict_proba(X)[:, 1]
    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Production scores contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Production scores must be between 0 and 1.")

    score_table = pd.DataFrame(
        {
            "visitorid": pd.Series(visitor_ids).reset_index(drop=True),
            "purchase_intent_score": probabilities,
            "predicted_conversion": (
                probabilities >= float(threshold)
            ).astype(int),
            "production_threshold": float(threshold),
        }
    )

    FINAL_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SCORE_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_table.to_csv(FINAL_SCORE_PATH, index=False)

    manifest = {
        "model_generation": "final_champion",
        "model_name": (
            model_metadata.get("final_model_name")
            or model_metadata.get("champion_model_name")
            or "Final champion"
        ),
        "model_path": str(FINAL_MODEL_PATH),
        "metadata_path": str(FINAL_METADATA_PATH),
        "score_path": str(FINAL_SCORE_PATH),
        "model_sha256": calculate_file_sha256(FINAL_MODEL_PATH),
        "metadata_sha256": calculate_file_sha256(
            FINAL_METADATA_PATH
        ),
        "score_sha256": calculate_file_sha256(FINAL_SCORE_PATH),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(X.columns),
        "threshold": float(threshold),
        "row_count": int(len(score_table)),
        "score_min": float(probabilities.min()),
        "score_max": float(probabilities.max()),
        "score_mean": float(probabilities.mean()),
    }

    with open(
        FINAL_SCORE_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=4)

    print(f"Saved final visitor scores: {FINAL_SCORE_PATH}")
    print(f"Saved score manifest: {FINAL_SCORE_MANIFEST_PATH}")

    return score_table, manifest
