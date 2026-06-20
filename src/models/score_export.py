# score_export.py
# Save and validate production scores with evidence of the exact model.
#
# Simple idea:
#   A cached score file is trustworthy only when its model, metadata,
#   features, threshold, visitor IDs, row count, and file hashes match.

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.paths import (
    MODEL_METADATA_DIR,
    PROCESSED_DATA_DIR,
    TRAINED_MODELS_DIR,
)
from src.models.production_model import get_production_threshold


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


def calculate_visitor_id_sha256(visitor_ids: pd.Series) -> str:
    """Create a stable fingerprint for visitor identity and row order."""

    normalized_ids = (
        pd.Series(visitor_ids)
        .reset_index(drop=True)
        .astype("string")
        .fillna("<missing>")
    )

    payload = "\n".join(normalized_ids.tolist()).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


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
    """Validate feature rows, visitor IDs, and threshold."""

    if len(X) == 0:
        raise ValueError("Score export requires at least one feature row.")

    if len(X) != len(visitor_ids):
        raise ValueError(
            "Feature rows and visitor IDs must have the same length."
        )

    if not 0 <= float(threshold) <= 1:
        raise ValueError(
            "Production threshold must be between 0 and 1."
        )


def predict_probabilities(
    model,
    X: pd.DataFrame,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Create probabilities, optionally in memory-safe chunks."""

    if chunk_size is None:
        probabilities = model.predict_proba(X)[:, 1]
        return np.asarray(probabilities, dtype=float)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    score_chunks = []

    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        chunk_scores = model.predict_proba(X.iloc[start:end])[:, 1]
        score_chunks.append(np.asarray(chunk_scores, dtype=float))

    return np.concatenate(score_chunks)


def save_final_champion_scores(
    final_model,
    X: pd.DataFrame,
    visitor_ids: pd.Series,
    threshold: float,
    *,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    score_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    model_generation: str = "final_champion",
    chunk_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Score all visitors and save a traceable production artifact."""

    validate_score_inputs(X, visitor_ids, threshold)

    model_path = FINAL_MODEL_PATH if model_path is None else Path(model_path)
    metadata_path = (
        FINAL_METADATA_PATH
        if metadata_path is None
        else Path(metadata_path)
    )
    score_path = FINAL_SCORE_PATH if score_path is None else Path(score_path)
    manifest_path = (
        FINAL_SCORE_MANIFEST_PATH
        if manifest_path is None
        else Path(manifest_path)
    )

    model_metadata = load_json(metadata_path)
    expected_features = model_metadata.get("feature_columns", [])

    if list(X.columns) != list(expected_features):
        raise ValueError(
            "Score-export features do not match production metadata."
        )

    probabilities = predict_probabilities(
        final_model,
        X,
        chunk_size=chunk_size,
    )

    if not np.isfinite(probabilities).all():
        raise ValueError("Production scores contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Production scores must be between 0 and 1.")

    normalized_visitor_ids = pd.Series(visitor_ids).reset_index(drop=True)

    score_table = pd.DataFrame(
        {
            "visitorid": normalized_visitor_ids,
            "purchase_intent_score": probabilities,
            "predicted_conversion": (
                probabilities >= float(threshold)
            ).astype(int),
            "production_threshold": float(threshold),
        }
    )

    score_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    score_table.to_csv(score_path, index=False)

    manifest = {
        "model_generation": model_generation,
        "model_name": (
            model_metadata.get("final_model_name")
            or model_metadata.get("champion_model_name")
            or model_metadata.get("model_name")
            or "Production champion"
        ),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "score_path": str(score_path),
        "model_sha256": calculate_file_sha256(model_path),
        "metadata_sha256": calculate_file_sha256(metadata_path),
        "score_sha256": calculate_file_sha256(score_path),
        "visitor_id_sha256": calculate_visitor_id_sha256(
            normalized_visitor_ids
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(X.columns),
        "threshold": float(threshold),
        "row_count": int(len(score_table)),
        "score_min": float(probabilities.min()),
        "score_max": float(probabilities.max()),
        "score_mean": float(probabilities.mean()),
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print(f"Saved production visitor scores: {score_path}")
    print(f"Saved score manifest: {manifest_path}")

    return score_table, manifest


def load_valid_cached_scores(
    production_bundle: Dict,
    expected_visitor_ids: pd.Series,
    *,
    score_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Return cached scores only when every provenance check passes."""

    score_path = FINAL_SCORE_PATH if score_path is None else Path(score_path)
    manifest_path = (
        FINAL_SCORE_MANIFEST_PATH
        if manifest_path is None
        else Path(manifest_path)
    )

    if not score_path.exists():
        return None, f"score file is missing: {score_path}"

    if not manifest_path.exists():
        return None, f"score manifest is missing: {manifest_path}"

    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"score manifest cannot be read: {error}"

    required_manifest_fields = {
        "model_generation",
        "model_path",
        "metadata_path",
        "score_path",
        "model_sha256",
        "metadata_sha256",
        "score_sha256",
        "visitor_id_sha256",
        "feature_columns",
        "threshold",
        "row_count",
    }

    missing_fields = required_manifest_fields.difference(manifest)

    if missing_fields:
        return None, (
            "score manifest is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if manifest["model_generation"] != production_bundle["generation"]:
        return None, "model generation does not match"

    if list(manifest["feature_columns"]) != list(
        production_bundle["feature_columns"]
    ):
        return None, "feature names or order do not match"

    expected_threshold = get_production_threshold(
        production_bundle["metadata"]
    )

    if not math.isclose(
        float(manifest["threshold"]),
        expected_threshold,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        return None, "production threshold does not match"

    active_model_path = Path(production_bundle["model_path"])
    active_metadata_path = Path(production_bundle["metadata_path"])

    if Path(manifest["model_path"]).resolve() != active_model_path.resolve():
        return None, "model path does not match"

    if (
        Path(manifest["metadata_path"]).resolve()
        != active_metadata_path.resolve()
    ):
        return None, "metadata path does not match"

    if Path(manifest["score_path"]).resolve() != score_path.resolve():
        return None, "score path does not match"

    if manifest["model_sha256"] != calculate_file_sha256(
        active_model_path
    ):
        return None, "model hash does not match"

    if manifest["metadata_sha256"] != calculate_file_sha256(
        active_metadata_path
    ):
        return None, "metadata hash does not match"

    if manifest["score_sha256"] != calculate_file_sha256(score_path):
        return None, "score file hash does not match"

    try:
        score_table = pd.read_csv(score_path)
    except (OSError, pd.errors.ParserError) as error:
        return None, f"score file cannot be read: {error}"

    required_score_columns = {
        "visitorid",
        "purchase_intent_score",
        "predicted_conversion",
        "production_threshold",
    }

    missing_columns = required_score_columns.difference(score_table.columns)

    if missing_columns:
        return None, (
            "score file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if int(manifest["row_count"]) != len(score_table):
        return None, "score row count does not match manifest"

    expected_ids = pd.Series(expected_visitor_ids).reset_index(drop=True)

    if len(score_table) != len(expected_ids):
        return None, "score row count does not match current visitors"

    expected_id_hash = calculate_visitor_id_sha256(expected_ids)
    cached_id_hash = calculate_visitor_id_sha256(
        score_table["visitorid"]
    )

    if manifest["visitor_id_sha256"] != expected_id_hash:
        return None, "visitor ID fingerprint does not match current data"

    if cached_id_hash != expected_id_hash:
        return None, "cached visitor IDs or row order do not match"

    score_values = pd.to_numeric(
        score_table["purchase_intent_score"],
        errors="coerce",
    )

    if score_values.isna().any():
        return None, "cached scores contain non-numeric values"

    if not np.isfinite(score_values).all():
        return None, "cached scores contain non-finite values"

    if ((score_values < 0) | (score_values > 1)).any():
        return None, "cached scores fall outside zero to one"

    cached_thresholds = pd.to_numeric(
        score_table["production_threshold"],
        errors="coerce",
    )

    if cached_thresholds.isna().any():
        return None, "cached thresholds contain non-numeric values"

    if not np.allclose(
        cached_thresholds.to_numpy(),
        expected_threshold,
        rtol=0,
        atol=1e-12,
    ):
        return None, "cached threshold column does not match"

    expected_predictions = (
        score_values.to_numpy() >= expected_threshold
    ).astype(int)

    cached_predictions = pd.to_numeric(
        score_table["predicted_conversion"],
        errors="coerce",
    )

    if cached_predictions.isna().any():
        return None, "cached predictions contain non-numeric values"

    if not np.array_equal(
        cached_predictions.astype(int).to_numpy(),
        expected_predictions,
    ):
        return None, "cached class predictions do not match scores"

    return score_table, "cached score manifest validated successfully"
