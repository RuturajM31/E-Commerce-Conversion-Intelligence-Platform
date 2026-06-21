"""Prepare segment score evidence for MLV-G02.

Inputs:
    data/processed/visitor_scores.csv
    reports/tables/segment_summary.csv

Outputs:
    Clean visitor-level scores, segment summaries, business order, and an
    explicit flag showing whether matured outcomes are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VISITOR_SCORES_PATH = Path("data/processed/visitor_scores.csv")
SEGMENT_SUMMARY_PATH = Path("reports/tables/segment_summary.csv")

REQUIRED_SCORE_COLUMNS = {
    "visitorid",
    "purchase_intent_score",
    "intent_segment",
}
REQUIRED_SUMMARY_COLUMNS = {
    "intent_segment",
    "visitors",
    "avg_score",
    "visitor_share",
    "actual_converters",
    "conversion_rate",
}

SEGMENT_ORDER = [
    "High Intent",
    "Strong Intent",
    "Warm Intent",
    "Low Intent",
    "Very Low Intent",
]


@dataclass
class SegmentIntelligenceBundle:
    """Evidence consumed by the segment visual renderer."""

    visitor_scores: pd.DataFrame
    segment_summary: pd.DataFrame
    segment_order: list[str]
    source_rows: int
    outcome_columns_available: bool


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """Raise a clear error when a required source column is absent."""

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def prepare_visitor_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the visitor-level score and segment source."""

    _require_columns(
        frame,
        REQUIRED_SCORE_COLUMNS,
        "Visitor score source",
    )

    clean = frame[
        [
            "visitorid",
            "purchase_intent_score",
            "intent_segment",
        ]
    ].copy()

    clean["purchase_intent_score"] = pd.to_numeric(
        clean["purchase_intent_score"],
        errors="coerce",
    )
    clean["intent_segment"] = (
        clean["intent_segment"]
        .astype("string")
        .str.strip()
    )

    clean = clean.dropna(
        subset=[
            "visitorid",
            "purchase_intent_score",
            "intent_segment",
        ]
    ).reset_index(drop=True)

    if clean.empty:
        raise ValueError(
            "Visitor score source contains no usable rows."
        )

    if not clean["purchase_intent_score"].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "Purchase-intent scores must stay between 0 and 1."
        )

    if clean["visitorid"].duplicated().any():
        raise ValueError(
            "Visitor score source contains duplicate visitor IDs."
        )

    return clean


def prepare_segment_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the provided segment summary source."""

    _require_columns(
        frame,
        REQUIRED_SUMMARY_COLUMNS,
        "Segment summary",
    )

    clean = frame.copy()
    clean["intent_segment"] = (
        clean["intent_segment"]
        .astype("string")
        .str.strip()
    )

    for column in [
        "visitors",
        "avg_score",
        "visitor_share",
        "actual_converters",
        "conversion_rate",
    ]:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    if clean["intent_segment"].duplicated().any():
        raise ValueError(
            "Segment summary contains duplicate segments."
        )

    return clean


def build_score_summary(
    visitor_scores: pd.DataFrame,
    provided_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate score distribution evidence for each segment."""

    calculated = (
        visitor_scores
        .groupby("intent_segment")
        .agg(
            visitors=("visitorid", "nunique"),
            avg_score=("purchase_intent_score", "mean"),
            minimum=("purchase_intent_score", "min"),
            p10=(
                "purchase_intent_score",
                lambda values: values.quantile(0.10),
            ),
            p25=(
                "purchase_intent_score",
                lambda values: values.quantile(0.25),
            ),
            median=("purchase_intent_score", "median"),
            p75=(
                "purchase_intent_score",
                lambda values: values.quantile(0.75),
            ),
            p90=(
                "purchase_intent_score",
                lambda values: values.quantile(0.90),
            ),
            maximum=("purchase_intent_score", "max"),
            score_std=("purchase_intent_score", "std"),
        )
        .reset_index()
    )

    calculated["visitor_share"] = (
        calculated["visitors"]
        / calculated["visitors"].sum()
    )

    outcome_columns = provided_summary[
        [
            "intent_segment",
            "actual_converters",
            "conversion_rate",
        ]
    ]

    summary = calculated.merge(
        outcome_columns,
        on="intent_segment",
        how="left",
        validate="one_to_one",
    )
    summary["iqr"] = (
        summary["p75"] - summary["p25"]
    )
    summary["outcomes_available"] = (
        summary["actual_converters"].notna()
        & summary["conversion_rate"].notna()
    )

    return summary


def resolve_segment_order(
    available_segments: list[str],
) -> list[str]:
    """Use business priority order, then append unexpected segments."""

    ordered = [
        segment
        for segment in SEGMENT_ORDER
        if segment in available_segments
    ]
    unexpected = sorted(
        set(available_segments).difference(ordered)
    )

    return ordered + unexpected


def build_segment_intelligence_bundle(
    project_root: str | Path = ".",
) -> SegmentIntelligenceBundle:
    """Load real sources and build the complete segment bundle."""

    root = Path(project_root)
    score_path = root / VISITOR_SCORES_PATH
    summary_path = root / SEGMENT_SUMMARY_PATH

    if not score_path.exists():
        raise FileNotFoundError(
            f"Visitor score source not found: {score_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Segment summary source not found: {summary_path}"
        )

    visitor_scores = prepare_visitor_scores(
        pd.read_csv(score_path)
    )
    provided_summary = prepare_segment_summary(
        pd.read_csv(summary_path)
    )
    segment_summary = build_score_summary(
        visitor_scores,
        provided_summary,
    )
    segment_order = resolve_segment_order(
        segment_summary["intent_segment"]
        .astype(str)
        .tolist()
    )

    return SegmentIntelligenceBundle(
        visitor_scores=visitor_scores,
        segment_summary=segment_summary,
        segment_order=segment_order,
        source_rows=len(visitor_scores),
        outcome_columns_available=bool(
            segment_summary[
                "outcomes_available"
            ].all()
        ),
    )
