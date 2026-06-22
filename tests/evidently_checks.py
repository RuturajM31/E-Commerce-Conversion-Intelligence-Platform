"""Isolated tests for real Evidently report generation.

Run with:

    .venv-evidently/bin/python -m pytest \
        tests/evidently_checks.py -q

Why this file is not named ``test_*.py``:
    The normal application environment intentionally does not install
    Evidently. This file is executed only with ``.venv-evidently``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.monitoring import evidently_drift


# These are the seven canonical inputs expected by the champion model.
FEATURE_COLUMNS = [
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
    "cart_to_view_ratio",
    "events_per_unique_item",
]


def make_population(
    rows: int,
    count_shift: int = 0,
    score_shift: float = 0.0,
) -> pd.DataFrame:
    """Create a small but varied visitor-level monitoring population.

    Inputs:
        rows:
            Number of unique synthetic visitors to create.

        count_shift:
            Small change applied to activity counts. This allows tests
            to represent a newer population without using project data.

        score_shift:
            Small change applied to prediction scores.

    Output:
        One row per visitor with the seven production features and the
        champion score column.

    Used next:
        The returned frame is passed to the real report-generation,
        sampling, and validation functions.
    """

    # Row numbers create deterministic variation across all features.
    row_number = pd.Series(
        range(rows),
        dtype="int64",
    )

    total_events = (
        2
        + row_number % 9
        + count_shift
    )

    view_count = (
        1
        + row_number % 8
        + count_shift
    )

    addtocart_count = (
        row_number % 4
    )

    unique_items = (
        1
        + row_number % 6
    )

    activity_span_ms = (
        1_000
        + row_number * 1_000
        + count_shift * 100
    )

    # The denominator is always positive because view_count starts at 1.
    cart_to_view_ratio = (
        addtocart_count
        / view_count
    )

    events_per_unique_item = (
        total_events
        / unique_items
    )

    # Scores remain inside the valid production probability range.
    purchase_intent_score = (
        0.10
        + (row_number % 20) * 0.03
        + score_shift
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    return pd.DataFrame(
        {
            "visitorid": 100_000 + row_number,
            "total_events": total_events,
            "view_count": view_count,
            "addtocart_count": addtocart_count,
            "unique_items": unique_items,
            "activity_span_ms": activity_span_ms,
            "cart_to_view_ratio": cart_to_view_ratio,
            "events_per_unique_item": events_per_unique_item,
            "purchase_intent_score": purchase_intent_score,
        }
    )


def test_real_reports_create_html_and_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Feature and prediction reports must create both output formats."""

    # Reference represents historical visitor behaviour.
    reference = make_population(
        rows=60,
    )

    # Current represents a slightly shifted recent population.
    current = make_population(
        rows=60,
        count_shift=1,
        score_shift=0.04,
    )

    # Redirect generated test reports away from project runtime folders.
    monkeypatch.setattr(
        evidently_drift,
        "REPORT_FOLDER",
        tmp_path,
    )

    # Run the actual production feature-drift report function.
    feature_result = (
        evidently_drift
        .run_feature_drift_report(
            reference_sample=reference,
            current_sample=current,
            feature_columns=FEATURE_COLUMNS,
        )
    )

    # Run the actual production prediction-drift report function.
    prediction_result = (
        evidently_drift
        .run_prediction_drift_report(
            reference_sample=reference,
            current_sample=current,
        )
    )

    # Save both reports through the same function used in production.
    feature_paths = evidently_drift.save_report(
        result=feature_result,
        report_name="feature_test",
    )

    prediction_paths = evidently_drift.save_report(
        result=prediction_result,
        report_name="prediction_test",
    )

    # Both human-readable and machine-readable outputs are required.
    assert feature_paths["html"].exists()
    assert feature_paths["json"].exists()
    assert prediction_paths["html"].exists()
    assert prediction_paths["json"].exists()

    # The real Evidently result must contain calculated metrics.
    assert feature_result.dict()["metrics"]
    assert prediction_result.dict()["metrics"]


def test_missing_feature_fails_clearly() -> None:
    """A missing canonical feature must raise a clear validation error."""

    population = make_population(
        rows=20,
    ).drop(
        columns="total_events"
    )

    with pytest.raises(
        ValueError,
        match="missing model features",
    ):
        evidently_drift.validate_feature_values(
            data=population,
            feature_columns=FEATURE_COLUMNS,
            dataset_name="Missing-feature test",
        )


def test_extra_column_is_safely_ignored() -> None:
    """An unrelated extra column must not corrupt canonical alignment."""

    population = make_population(
        rows=20,
    )

    # This field is not part of the champion model input.
    population["unused_campaign_field"] = "test"

    # Validation reads only the seven canonical feature columns.
    evidently_drift.validate_feature_values(
        data=population,
        feature_columns=FEATURE_COLUMNS,
        dataset_name="Extra-column test",
    )


def test_empty_current_population_fails_honestly() -> None:
    """An empty current cohort must not produce a misleading report."""

    reference = make_population(
        rows=30,
    )

    empty_current = make_population(
        rows=0,
    )

    with pytest.raises(
        RuntimeError,
        match="contain no rows",
    ):
        evidently_drift.create_balanced_samples(
            reference_data=reference,
            current_data=empty_current,
        )


def test_small_current_population_generates_report() -> None:
    """A valid small cohort should use all available current rows."""

    reference = make_population(
        rows=50,
    )

    current = make_population(
        rows=20,
        count_shift=1,
        score_shift=0.02,
    )

    # The smaller current cohort controls the balanced sample size.
    reference_sample, current_sample = (
        evidently_drift.create_balanced_samples(
            reference_data=reference,
            current_data=current,
        )
    )

    assert len(reference_sample) == 20
    assert len(current_sample) == 20

    # Confirm that the actual report function handles the small cohort.
    result = evidently_drift.run_feature_drift_report(
        reference_sample=reference_sample,
        current_sample=current_sample,
        feature_columns=FEATURE_COLUMNS,
    )

    assert result.dict()["metrics"]
