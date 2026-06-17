from pathlib import Path

import pandas as pd
import pytest


def read_csv_or_skip(path: str) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise skip this test."""

    file_path = Path(path)

    if not file_path.exists():
        pytest.skip(f"File not available yet: {path}")

    return pd.read_csv(file_path)


def test_conversion_rate_is_valid():
    """Conversion rate should be a valid percentage between 0 and 1."""

    data = read_csv_or_skip("data/processed/visitor_features.csv")

    assert "converted" in data.columns

    conversion_rate = data["converted"].mean()

    assert 0 <= conversion_rate <= 1


def test_purchase_intent_scores_are_valid_probabilities_when_available():
    """Purchase intent scores must stay between 0 and 1."""

    score_paths = [
        "data/processed/final_champion_visitor_scores.csv",
        "data/processed/champion_visitor_scores.csv",
        "data/processed/visitor_scores.csv",
    ]

    available = [path for path in score_paths if Path(path).exists()]

    if not available:
        pytest.skip("No score file available yet.")

    for path in available:
        data = pd.read_csv(path)

        if "purchase_intent_score" not in data.columns:
            pytest.skip(f"{path} has no purchase_intent_score column.")

        invalid = ~data["purchase_intent_score"].between(0, 1)

        assert not invalid.any(), f"Invalid purchase intent scores found in {path}"


def test_forecast_predictions_are_non_negative_when_available():
    """Forecasts for visitors, events, conversions, and high-intent visitors cannot be negative."""

    data = read_csv_or_skip("reports/tables/business_forecast_future.csv")

    assert "predicted_value" in data.columns

    negative_rows = data[data["predicted_value"] < 0]

    assert negative_rows.empty, f"Negative forecast values found:\n{negative_rows.head()}"


def test_anomaly_rate_is_valid_when_available():
    """Anomaly rates should be valid percentages between 0 and 1."""

    summary = read_csv_or_skip("reports/tables/anomaly_summary.csv")

    if {"metric", "value"}.issubset(summary.columns):
        rows = summary[summary["metric"].astype(str).str.contains("rate", case=False, na=False)]

        if rows.empty:
            pytest.skip("No anomaly-rate metric found in anomaly summary.")

        invalid = ~rows["value"].astype(float).between(0, 1)

        assert not invalid.any(), f"Invalid anomaly rates found:\n{rows[invalid]}"

    elif "final_anomaly_rate" in summary.columns:
        assert summary["final_anomaly_rate"].astype(float).between(0, 1).all()

    else:
        pytest.skip("No anomaly-rate column found.")


def test_daily_business_kpi_formulas_when_available():
    """Daily KPI table should contain non-negative operational counts."""

    data = read_csv_or_skip("reports/tables/daily_business_kpis.csv")

    count_columns = [
        column
        for column in [
            "unique_visitors",
            "event_volume",
            "converted_visitor_count",
            "high_intent_visitor_count",
        ]
        if column in data.columns
    ]

    if not count_columns:
        pytest.skip("No known KPI count columns found.")

    for column in count_columns:
        assert (data[column] >= 0).all(), f"Negative values found in {column}"
