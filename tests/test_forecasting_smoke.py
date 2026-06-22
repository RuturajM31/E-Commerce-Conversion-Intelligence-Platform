"""Validation tests for generated business forecasting outputs."""

from pathlib import Path

import pandas as pd


FUTURE_PATH = Path(
    "reports/tables/business_forecast_future.csv"
)

COMPARISON_PATH = Path(
    "reports/tables/business_forecast_comparison.csv"
)


def normalize_boolean_flags(values: pd.Series) -> pd.Series:
    """Convert supported CSV boolean values into True or False."""

    normalized = values.astype(str).str.strip().str.lower()

    valid_values = {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
    }

    invalid_values = normalized[~normalized.isin(valid_values)]

    assert invalid_values.empty, (
        "Invalid is_best_model values: "
        f"{sorted(invalid_values.unique())}"
    )

    return normalized.isin({"true", "1", "yes"})


def test_forecast_output_contains_valid_future_predictions():
    """Future KPI predictions must be complete, dated and non-negative."""

    assert FUTURE_PATH.exists(), (
        f"Missing forecast output: {FUTURE_PATH}"
    )

    data = pd.read_csv(FUTURE_PATH)

    required_columns = {
        "date",
        "target_name",
        "predicted_value",
    }

    missing_columns = required_columns - set(data.columns)

    assert not missing_columns, (
        f"Missing forecast columns: {missing_columns}"
    )

    assert not data.empty

    dates = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    predicted_values = pd.to_numeric(
        data["predicted_value"],
        errors="coerce",
    )

    assert dates.notna().all()
    assert predicted_values.notna().all()
    assert (predicted_values >= 0).all()

    assert (
        data["target_name"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )


def test_forecast_comparison_selects_one_winner_per_target():
    """Each forecast target must have exactly one selected best model.

    The test deliberately does not require Prophet or any other specific
    algorithm to win. The best model should be selected from measured
    performance for each target.
    """

    assert COMPARISON_PATH.exists(), (
        f"Missing forecast comparison: {COMPARISON_PATH}"
    )

    data = pd.read_csv(COMPARISON_PATH)

    required_columns = {
        "target_name",
        "model_name",
        "is_best_model",
    }

    missing_columns = required_columns - set(data.columns)

    assert not missing_columns, (
        "Missing forecast comparison columns: "
        f"{missing_columns}"
    )

    assert not data.empty

    assert (
        data["target_name"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    assert (
        data["model_name"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    data = data.copy()

    data["_is_best_model"] = normalize_boolean_flags(
        data["is_best_model"]
    )

    winner_count = data.groupby(
        "target_name"
    )["_is_best_model"].sum()

    assert (winner_count == 1).all(), (
        "Every forecast target must have exactly one best model. "
        f"Winner counts: {winner_count.to_dict()}"
    )
