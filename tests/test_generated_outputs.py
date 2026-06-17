from pathlib import Path

import pandas as pd
import pytest


def read_csv_or_skip(path: str) -> pd.DataFrame:
    """Read a generated table if it exists, otherwise skip the test."""

    file_path = Path(path)

    if not file_path.exists():
        pytest.skip(f"Optional generated file not found yet: {path}")

    return pd.read_csv(file_path)


def test_final_champion_summary_has_metrics_when_available():
    """Final champion summary should contain the main model metrics.

    The exact model-name column can differ by pipeline version.
    Some project outputs store the model name in the metadata JSON,
    while the CSV focuses only on metrics. Therefore, this test checks
    the important metric columns and accepts model-name as optional.
    """

    data = read_csv_or_skip("reports/tables/final_true_champion_summary.csv")

    required_metric_columns = {
        "pr_auc",
        "roc_auc",
        "best_threshold",
        "best_precision",
        "best_recall",
        "best_f1_score",
    }

    missing_metrics = required_metric_columns - set(data.columns)

    assert not missing_metrics, f"Missing final champion metric columns: {missing_metrics}"
    assert len(data) >= 1

    # Optional model-name check.
    # We accept any common model-name column if it exists.
    possible_model_name_columns = {
        "model_name",
        "final_model_name",
        "champion_model",
        "champion_model_name",
        "selected_model",
    }

    existing_name_columns = possible_model_name_columns.intersection(data.columns)

    if existing_name_columns:
        column = sorted(existing_name_columns)[0]
        assert data[column].astype(str).str.len().gt(0).any()


def test_mvd_coverage_matrix_has_rows_when_available():
    """MVD coverage proof table should not be empty."""

    data = read_csv_or_skip("reports/tables/mvd_method_coverage_matrix.csv")

    assert len(data) > 0


def test_forecast_future_has_predictions_when_available():
    """Forecast output should contain future dates and predicted values."""

    data = read_csv_or_skip("reports/tables/business_forecast_future.csv")

    expected_columns = {"date", "target_name", "predicted_value"}
    missing = expected_columns - set(data.columns)

    assert not missing, f"Missing forecast columns: {missing}"
    assert len(data) > 0


def test_anomaly_summary_has_rows_when_available():
    """Anomaly summary should exist after anomaly pipeline runs."""

    data = read_csv_or_skip("reports/tables/anomaly_summary.csv")

    assert len(data) > 0