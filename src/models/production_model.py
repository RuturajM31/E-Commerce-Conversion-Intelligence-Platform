# production_model.py
# One shared place for selecting and validating the production model.
#
# Simple idea:
#   Every production workflow must use the same model and metadata pair.
#
# Priority:
#   1. Final tuned champion
#   2. Earlier champion only when the complete final pair is unavailable

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS
from src.models.model_config import (
    INITIAL_CHAMPION_METADATA_PATH,
    INITIAL_CHAMPION_MODEL_PATH,
    TRUE_FINAL_METADATA_PATH,
    TRUE_FINAL_MODEL_PATH,
)


# One candidate contains:
# generation name, model path, and matching metadata path.
ArtifactCandidate = Tuple[str, Path, Path]


def get_artifact_candidates() -> List[ArtifactCandidate]:
    """Return production artifact pairs in priority order."""

    return [
        (
            "final_champion",
            TRUE_FINAL_MODEL_PATH,
            TRUE_FINAL_METADATA_PATH,
        ),
        (
            "initial_champion",
            INITIAL_CHAMPION_MODEL_PATH,
            INITIAL_CHAMPION_METADATA_PATH,
        ),
    ]


def resolve_production_artifacts(
    candidates: Optional[List[ArtifactCandidate]] = None,
) -> ArtifactCandidate:
    """Select the first complete model and metadata pair.

    A model is never combined with metadata from another generation.
    """

    candidate_pairs = (
        get_artifact_candidates()
        if candidates is None
        else candidates
    )

    for generation, model_path, metadata_path in candidate_pairs:
        if model_path.exists() and metadata_path.exists():
            return generation, model_path, metadata_path

    checked_paths = "\n".join(
        (
            f"- {generation}: "
            f"model={model_path}, metadata={metadata_path}"
        )
        for generation, model_path, metadata_path in candidate_pairs
    )

    raise FileNotFoundError(
        "No complete production model and metadata pair was found.\n"
        f"Checked:\n{checked_paths}"
    )


def load_metadata(metadata_path: Path) -> Dict:
    """Load one model metadata JSON file."""

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_metadata_features(metadata: Dict) -> List[str]:
    """Confirm metadata uses the approved feature names and order."""

    feature_columns = metadata.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "Production metadata does not contain feature_columns."
        )

    if list(feature_columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "Production metadata feature columns do not match the "
            "canonical feature schema."
        )

    return list(feature_columns)


def validate_model_features(model, feature_columns: List[str]) -> None:
    """Confirm the loaded model expects the approved feature schema."""

    expected_count = len(feature_columns)
    model_feature_count = getattr(model, "n_features_in_", None)

    if (
        model_feature_count is not None
        and int(model_feature_count) != expected_count
    ):
        raise ValueError(
            "Production model feature count does not match metadata."
        )

    model_feature_names = getattr(model, "feature_names_in_", None)

    if (
        model_feature_names is not None
        and list(model_feature_names) != feature_columns
    ):
        raise ValueError(
            "Production model feature names or order do not match metadata."
        )


def load_production_bundle(
    candidates: Optional[List[ArtifactCandidate]] = None,
) -> Dict:
    """Load and validate the active production model bundle.

    Output contains:
        model
        metadata
        generation
        model_path
        metadata_path
        feature_columns
    """

    generation, model_path, metadata_path = (
        resolve_production_artifacts(candidates)
    )

    metadata = load_metadata(metadata_path)
    feature_columns = validate_metadata_features(metadata)

    model = joblib.load(model_path)
    validate_model_features(model, feature_columns)

    return {
        "model": model,
        "metadata": metadata,
        "generation": generation,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "feature_columns": feature_columns,
    }


def get_production_threshold(metadata: Dict) -> float:
    """Return and validate the production decision threshold."""

    threshold = float(metadata.get("best_threshold", 0.50))

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Production threshold must be between 0 and 1."
        )

    return threshold


def get_production_model_name(metadata: Dict) -> str:
    """Return a readable model name across metadata generations."""

    return str(
        metadata.get("final_model_name")
        or metadata.get("champion_model_name")
        or metadata.get("model_name")
        or "Production champion"
    )