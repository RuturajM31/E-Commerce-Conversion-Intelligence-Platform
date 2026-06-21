"""Prepare native XGBoost explanation data for ML Visual Intelligence.

Why this file exists:
    Explainability visuals should use one controlled calculation layer instead
    of recomputing SHAP-style values differently in every chart.

Main inputs:
    - models/trained/final_champion_model.joblib
    - data/processed/visitor_features.csv
    - models/metadata/final_champion_metadata.json

Main outputs:
    - A deterministic explanation sample
    - Native XGBoost feature contributions
    - Native XGBoost interaction contributions
    - Model probabilities and representative visitors
    - Source-quality metadata such as missing-value counts

Used next:
    `explainability_visuals.py` renders MLV-E01 to MLV-E06 from the returned
    bundle without retraining the model.

Important limitation:
    Contributions explain the model's prediction logic. They do not prove
    that a feature causes a visitor to convert.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical feature contract
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
    "cart_to_view_ratio",
    "events_per_unique_item",
]

FEATURE_DISPLAY_NAMES = {
    "total_events": "Total events",
    "view_count": "Views",
    "addtocart_count": "Add-to-cart events",
    "unique_items": "Unique items",
    "activity_span_ms": "Activity span",
    "cart_to_view_ratio": "Cart-to-view ratio",
    "events_per_unique_item": "Events per unique item",
}

VISITOR_ID_CANDIDATES = [
    "visitorid",
    "visitor_id",
    "customer_id",
]

MODEL_PATH = Path("models/trained/final_champion_model.joblib")
FEATURE_PATH = Path("data/processed/visitor_features.csv")
METADATA_PATH = Path("models/metadata/final_champion_metadata.json")


@dataclass
class ExplainabilityBundle:
    """Container holding all data required by E01 to E06.

    Inputs stored:
        Sampled feature values, probabilities, contributions, interactions,
        visitor IDs, source positions, and model metadata.

    Outputs enabled:
        Global, dependence, local, representative, and interaction visuals.

    Used next:
        Rendering functions consume this object without reloading the model.
    """

    feature_frame: pd.DataFrame
    visitor_ids: pd.Series
    source_positions: np.ndarray
    probabilities: np.ndarray
    shap_values: np.ndarray
    base_values: np.ndarray
    interaction_values: np.ndarray
    interaction_sample_positions: np.ndarray
    representative_positions: dict[str, int]
    priority_position: int
    source_rows: int
    sample_rows: int
    model_name: str
    missing_counts: dict[str, int]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Convert log-odds margins into probabilities.

    Input:
        One scalar or NumPy array of margins.

    Output:
        Values between 0 and 1.

    Used next:
        Waterfall visuals translate baseline and final margins into readable
        probabilities.
    """

    # Clipping prevents numerical overflow for very large positive/negative
    # margins while preserving practical probability values.
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def find_visitor_id_column(frame: pd.DataFrame) -> str | None:
    """Find the first supported visitor identifier column."""

    for candidate in VISITOR_ID_CANDIDATES:
        if candidate in frame.columns:
            return candidate

    return None


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Validate that every canonical model feature is available.

    Input:
        Raw visitor feature table.

    Output:
        No value. A clear error is raised when the source contract is broken.

    Used next:
        Feature preparation only begins after this check passes.
    """

    missing = sorted(
        set(FEATURE_COLUMNS).difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Visitor feature source is missing required columns: "
            + ", ".join(missing)
        )


def prepare_feature_frame(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Prepare numeric model features and visitor IDs.

    Input:
        Raw visitor feature DataFrame.

    Output:
        Numeric feature frame, visitor ID series, and original missing counts.

    Used next:
        The model scores the full source before deterministic sampling.
    """

    validate_feature_schema(raw_frame)

    # Preserve a business identifier when available; otherwise use row index.
    visitor_column = find_visitor_id_column(raw_frame)

    if visitor_column is None:
        visitor_ids = pd.Series(
            raw_frame.index.astype(str),
            index=raw_frame.index,
            name="visitor_id",
        )
    else:
        visitor_ids = (
            raw_frame[visitor_column]
            .astype(str)
            .rename("visitor_id")
        )

    # Convert all model inputs to numeric values.
    feature_frame = raw_frame[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Record data-quality evidence before any missing-value treatment.
    missing_counts = {
        column: int(count)
        for column, count in feature_frame.isna().sum().items()
        if int(count) > 0
    }

    # Use feature medians only when the source unexpectedly contains missing
    # values. The counts remain visible in the manifest and insight report.
    if feature_frame.isna().any().any():
        medians = feature_frame.median(numeric_only=True)
        feature_frame = feature_frame.fillna(medians)

    # Refuse a feature that remains empty after median handling.
    if feature_frame.isna().any().any():
        unresolved = feature_frame.columns[
            feature_frame.isna().any()
        ].tolist()
        raise ValueError(
            "Feature values remain missing after median handling: "
            + ", ".join(unresolved)
        )

    return feature_frame, visitor_ids, missing_counts


def select_representative_source_positions(
    probabilities: np.ndarray,
) -> dict[str, int]:
    """Select low, medium, and high score visitors deterministically.

    Input:
        Probability for every source row.

    Output:
        Source-row positions nearest the 10th, 50th, and 90th percentiles.

    Used next:
        E05 compares representative model decision paths.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if probability_array.ndim != 1 or len(probability_array) == 0:
        raise ValueError(
            "Probabilities must be a non-empty one-dimensional array."
        )

    quantiles = {
        "Low intent": 0.10,
        "Medium intent": 0.50,
        "High intent": 0.90,
    }

    selected: dict[str, int] = {}

    for label, quantile in quantiles.items():
        target = float(
            np.quantile(probability_array, quantile)
        )
        selected[label] = int(
            np.argmin(np.abs(probability_array - target))
        )

    return selected


def build_sample_positions(
    *,
    row_count: int,
    sample_size: int,
    required_positions: list[int],
    random_state: int,
) -> np.ndarray:
    """Build a deterministic sample containing required visitors.

    Input:
        Source row count, sample limit, mandatory positions, and random seed.

    Output:
        Sorted unique source positions.

    Used next:
        Native contribution calculations run only on this controlled sample.
    """

    if row_count <= 0:
        raise ValueError("Source row count must be positive.")

    final_size = min(max(sample_size, 1), row_count)
    rng = np.random.default_rng(random_state)

    # Random sampling supports representative global distributions.
    selected = set(
        rng.choice(
            row_count,
            size=final_size,
            replace=False,
        ).tolist()
    )

    # Mandatory rows ensure local explanation visitors are always included.
    selected.update(
        int(position)
        for position in required_positions
        if 0 <= int(position) < row_count
    )

    return np.array(
        sorted(selected),
        dtype=int,
    )


def get_model_probabilities(
    model: Any,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Return positive-class probabilities from the saved classifier."""

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Saved champion model does not support `predict_proba`."
        )

    probabilities = np.asarray(
        model.predict_proba(feature_frame)
    )

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected a two-column probability matrix."
        )

    return probabilities[:, 1].astype(float)


def compute_native_xgboost_contributions(
    model: Any,
    sample_features: pd.DataFrame,
    *,
    interaction_sample_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute native XGBoost contributions and interactions.

    Input:
        Saved XGBClassifier, sampled features, and interaction sample limit.

    Output:
        Feature contributions, base margins, interaction tensor, and positions
        used for interaction calculations.

    Used next:
        E01 to E06 use the same native explanation source.
    """

    if not hasattr(model, "get_booster"):
        raise TypeError(
            "Saved champion model does not expose an XGBoost booster."
        )

    # Import only when production explanation calculation is requested.
    import xgboost as xgb

    booster = model.get_booster()

    # Preserve feature order explicitly in the DMatrix.
    explanation_matrix = xgb.DMatrix(
        sample_features,
        feature_names=list(sample_features.columns),
    )

    # Native contributions include one final bias column.
    raw_contributions = np.asarray(
        booster.predict(
            explanation_matrix,
            pred_contribs=True,
        )
    )

    expected_columns = sample_features.shape[1] + 1

    if (
        raw_contributions.ndim != 2
        or raw_contributions.shape[1] != expected_columns
    ):
        raise ValueError(
            "Unexpected native contribution matrix shape: "
            f"{raw_contributions.shape}"
        )

    shap_values = raw_contributions[:, :-1]
    base_values = raw_contributions[:, -1]

    # Interaction calculation is more expensive, so use a deterministic
    # evenly spaced subset that still covers the whole sample range.
    interaction_count = min(
        max(interaction_sample_size, 1),
        len(sample_features),
    )
    interaction_positions = np.unique(
        np.linspace(
            0,
            len(sample_features) - 1,
            num=interaction_count,
            dtype=int,
        )
    )
    interaction_features = sample_features.iloc[
        interaction_positions
    ]

    interaction_matrix = xgb.DMatrix(
        interaction_features,
        feature_names=list(sample_features.columns),
    )
    raw_interactions = np.asarray(
        booster.predict(
            interaction_matrix,
            pred_interactions=True,
        )
    )

    expected_shape = (
        len(interaction_features),
        expected_columns,
        expected_columns,
    )

    if raw_interactions.shape != expected_shape:
        raise ValueError(
            "Unexpected native interaction tensor shape: "
            f"{raw_interactions.shape}"
        )

    # Remove the bias row and column.
    interaction_values = raw_interactions[:, :-1, :-1]

    return (
        shap_values,
        base_values,
        interaction_values,
        interaction_positions,
    )


def build_explainability_bundle(
    project_root: str | Path = ".",
    *,
    sample_size: int = 2_500,
    interaction_sample_size: int = 600,
    random_state: int = 42,
) -> ExplainabilityBundle:
    """Load real sources and build the complete explanation bundle.

    Input:
        Project root plus deterministic sample settings.

    Output:
        `ExplainabilityBundle` used by all six visual renderers.

    Used next:
        The package generator renders E01 to E06 and supporting evidence.
    """

    root = Path(project_root)
    model_file = root / MODEL_PATH
    feature_file = root / FEATURE_PATH
    metadata_file = root / METADATA_PATH

    if not model_file.exists():
        raise FileNotFoundError(
            f"Saved champion model not found: {model_file}"
        )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Visitor feature source not found: {feature_file}"
        )

    # Load the existing model and source features.
    model = joblib.load(model_file)
    raw_frame = pd.read_csv(feature_file)
    metadata = read_json(metadata_file)

    full_features, full_visitor_ids, missing_counts = (
        prepare_feature_frame(raw_frame)
    )

    # Score all rows so local examples represent the complete source.
    full_probabilities = get_model_probabilities(
        model,
        full_features,
    )

    representative_source_positions = (
        select_representative_source_positions(
            full_probabilities
        )
    )
    priority_source_position = int(
        np.argmax(full_probabilities)
    )

    required_positions = list(
        representative_source_positions.values()
    ) + [priority_source_position]

    sample_source_positions = build_sample_positions(
        row_count=len(full_features),
        sample_size=sample_size,
        required_positions=required_positions,
        random_state=random_state,
    )

    sample_features = (
        full_features
        .iloc[sample_source_positions]
        .reset_index(drop=True)
    )
    sample_visitor_ids = (
        full_visitor_ids
        .iloc[sample_source_positions]
        .reset_index(drop=True)
    )
    sample_probabilities = full_probabilities[
        sample_source_positions
    ]

    (
        shap_values,
        base_values,
        interaction_values,
        interaction_positions,
    ) = compute_native_xgboost_contributions(
        model,
        sample_features,
        interaction_sample_size=interaction_sample_size,
    )

    # Map full-source positions into sampled-row positions.
    sample_position_lookup = {
        int(source_position): int(sample_position)
        for sample_position, source_position in enumerate(
            sample_source_positions
        )
    }

    representative_sample_positions = {
        label: sample_position_lookup[int(source_position)]
        for label, source_position
        in representative_source_positions.items()
    }
    priority_sample_position = sample_position_lookup[
        priority_source_position
    ]

    model_name = str(
        metadata.get("final_model_name", "Saved champion")
    )

    return ExplainabilityBundle(
        feature_frame=sample_features,
        visitor_ids=sample_visitor_ids,
        source_positions=sample_source_positions,
        probabilities=sample_probabilities,
        shap_values=shap_values,
        base_values=base_values,
        interaction_values=interaction_values,
        interaction_sample_positions=interaction_positions,
        representative_positions=representative_sample_positions,
        priority_position=priority_sample_position,
        source_rows=len(full_features),
        sample_rows=len(sample_features),
        model_name=model_name,
        missing_counts=missing_counts,
    )
