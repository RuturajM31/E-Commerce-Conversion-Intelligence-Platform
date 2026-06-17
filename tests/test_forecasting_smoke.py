import os
from pathlib import Path

import pandas as pd
import pytest


def test_forecast_output_is_valid_when_available():
    """Forecast output should contain usable future KPI predictions."""

    path = Path("reports/tables/business_forecast_future.csv")

    if not path.exists():
        pytest.skip("Forecast future table is not available.")

    data = pd.read_csv(path)

    required = {"date", "target_name", "predicted_value"}
    missing = required - set(data.columns)

    assert not missing, f"Missing forecast columns: {missing}"

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    assert data["date"].notna().all()
    assert data["predicted_value"].notna().all()
    assert (data["predicted_value"] >= 0).all()


def test_prophet_is_best_only_when_strict_mode_enabled():
    """Optional strict check that Prophet is selected as best model.

    Forecasting is never expected to be perfect.
    This test only checks Prophet winner status when the project owner wants that strict check.
    To run:
        REQUIRE_PROPHET_BEST=1 pytest tests/test_forecasting_smoke.py -q
    """

    if os.getenv("REQUIRE_PROPHET_BEST") != "1":
        pytest.skip("Strict Prophet-best test skipped by default.")

    path = Path("reports/tables/business_forecast_comparison.csv")

    if not path.exists():
        pytest.skip("Forecast comparison table is not available.")

    data = pd.read_csv(path)

    required = {"target_name", "model_name", "is_best_model"}
    missing = required - set(data.columns)

    assert not missing, f"Missing forecast comparison columns: {missing}"

    best_rows = data[data["is_best_model"].astype(str).str.lower().isin(["true", "1", "yes"])]

    assert not best_rows.empty
    assert best_rows["model_name"].astype(str).str.contains("prophet", case=False).all()
