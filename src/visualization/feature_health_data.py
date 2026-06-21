"""Prepare data and feature-health evidence for ML Visual Intelligence.

Why this file exists:
    The processed visitor feature table is the common source for distribution,
    correlation, and validity diagnostics.

Main input:
    - data/processed/visitor_features.csv

Main outputs:
    - Clean canonical feature frame
    - Distribution summary statistics
    - Spearman correlation matrix and clustered order
    - Missingness and validity percentages

Used next:
    `feature_health_visuals.py` renders MLV-H01, MLV-H02, and MLV-H04.

Limitation:
    These visuals describe the current processed snapshot. Historical
    train-versus-holdout feature-shift evidence is not available yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_PATH = Path("data/processed/visitor_features.csv")

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

RATIO_FEATURES = {"cart_to_view_ratio"}
NON_NEGATIVE_FEATURES = set(FEATURE_COLUMNS)


@dataclass
class FeatureHealthBundle:
    """Container holding all evidence required by H01, H02, and H04."""

    feature_frame: pd.DataFrame
    distribution_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    clustered_order: list[str]
    validity_summary: pd.DataFrame
    source_rows: int


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Validate that every canonical model feature is available."""

    missing = sorted(
        set(FEATURE_COLUMNS).difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Visitor feature source is missing required columns: "
            + ", ".join(missing)
        )


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical model features to numeric form."""

    validate_feature_schema(frame)

    clean = frame[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if clean.empty:
        raise ValueError("Visitor feature source contains no rows.")

    return clean


def build_distribution_summary(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create robust distribution statistics for each feature."""

    records: list[dict[str, Any]] = []

    for feature_name in FEATURE_COLUMNS:
        values = feature_frame[feature_name]
        finite_values = values[
            np.isfinite(values)
        ].dropna()

        if finite_values.empty:
            records.append(
                {
                    "feature": feature_name,
                    "display_name": FEATURE_DISPLAY_NAMES[
                        feature_name
                    ],
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "minimum": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "maximum": np.nan,
                    "zero_rate": np.nan,
                }
            )
            continue

        records.append(
            {
                "feature": feature_name,
                "display_name": FEATURE_DISPLAY_NAMES[
                    feature_name
                ],
                "count": int(len(finite_values)),
                "mean": float(finite_values.mean()),
                "std": float(finite_values.std(ddof=1)),
                "minimum": float(finite_values.min()),
                "p05": float(finite_values.quantile(0.05)),
                "p25": float(finite_values.quantile(0.25)),
                "median": float(finite_values.median()),
                "p75": float(finite_values.quantile(0.75)),
                "p95": float(finite_values.quantile(0.95)),
                "p99": float(finite_values.quantile(0.99)),
                "maximum": float(finite_values.max()),
                "zero_rate": float(
                    (finite_values == 0).mean()
                ),
            }
        )

    return pd.DataFrame(records)


def build_correlation_matrix(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate a Spearman rank-correlation matrix."""

    return feature_frame.corr(
        method="spearman",
        min_periods=max(
            5,
            int(len(feature_frame) * 0.01),
        ),
    )


def build_clustered_order(
    correlation_matrix: pd.DataFrame,
) -> list[str]:
    """Order features by absolute-correlation similarity.

    SciPy hierarchical clustering is preferred. A deterministic
    absolute-correlation ordering is used if SciPy is unavailable.
    """

    clean = correlation_matrix.fillna(0.0).clip(
        -1.0,
        1.0,
    )

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        distance = 1.0 - np.abs(
            clean.to_numpy(dtype=float)
        )
        distance = np.clip(distance, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)

        condensed = squareform(
            distance,
            checks=False,
        )
        linkage_matrix = linkage(
            condensed,
            method="average",
        )
        positions = leaves_list(linkage_matrix).tolist()

        return [
            clean.index[position]
            for position in positions
        ]
    except Exception:
        return (
            clean.abs()
            .sum(axis=1)
            .sort_values(ascending=False)
            .index
            .tolist()
        )


def build_validity_summary(
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature-level quality checks and rates."""

    row_count = len(feature_frame)
    records: list[dict[str, Any]] = []

    for feature_name in FEATURE_COLUMNS:
        values = feature_frame[feature_name]

        missing_mask = values.isna()
        finite_mask = pd.Series(
            np.isfinite(values.fillna(0.0)),
            index=values.index,
        ) & ~missing_mask
        non_finite_mask = ~finite_mask & ~missing_mask
        negative_mask = (
            values < 0
        ).fillna(False)
        zero_mask = (
            values == 0
        ).fillna(False)

        ratio_above_one_mask = pd.Series(
            False,
            index=values.index,
        )

        if feature_name in RATIO_FEATURES:
            ratio_above_one_mask = (
                values > 1.0
            ).fillna(False)

        invalid_mask = (
            missing_mask
            | non_finite_mask
            | (
                negative_mask
                if feature_name in NON_NEGATIVE_FEATURES
                else False
            )
            | ratio_above_one_mask
        )
        valid_mask = ~invalid_mask

        records.append(
            {
                "feature": feature_name,
                "display_name": FEATURE_DISPLAY_NAMES[
                    feature_name
                ],
                "rows": row_count,
                "valid_count": int(valid_mask.sum()),
                "missing_count": int(missing_mask.sum()),
                "non_finite_count": int(
                    non_finite_mask.sum()
                ),
                "negative_count": int(
                    negative_mask.sum()
                ),
                "zero_count": int(zero_mask.sum()),
                "ratio_above_one_count": int(
                    ratio_above_one_mask.sum()
                ),
                "valid_rate": float(valid_mask.mean()),
                "missing_rate": float(
                    missing_mask.mean()
                ),
                "non_finite_rate": float(
                    non_finite_mask.mean()
                ),
                "negative_rate": float(
                    negative_mask.mean()
                ),
                "zero_rate": float(zero_mask.mean()),
                "ratio_above_one_rate": float(
                    ratio_above_one_mask.mean()
                ),
            }
        )

    return pd.DataFrame(records)


def build_feature_health_bundle(
    project_root: str | Path = ".",
) -> FeatureHealthBundle:
    """Load the real processed feature table and calculate all evidence."""

    root = Path(project_root)
    feature_file = root / FEATURE_PATH

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Visitor feature source not found: {feature_file}"
        )

    raw_frame = pd.read_csv(feature_file)
    feature_frame = prepare_feature_frame(raw_frame)
    distribution_summary = build_distribution_summary(
        feature_frame
    )
    correlation_matrix = build_correlation_matrix(
        feature_frame
    )
    clustered_order = build_clustered_order(
        correlation_matrix
    )
    validity_summary = build_validity_summary(
        feature_frame
    )

    return FeatureHealthBundle(
        feature_frame=feature_frame,
        distribution_summary=distribution_summary,
        correlation_matrix=correlation_matrix,
        clustered_order=clustered_order,
        validity_summary=validity_summary,
        source_rows=len(feature_frame),
    )
