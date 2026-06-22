# build_business_forecasts.py
# Build daily business KPIs and forecast future ecommerce demand.
#
# SIMPLE IDEA:
#   This script turns raw RetailRocket events into daily business KPIs,
#   then forecasts future operational demand.
#
# WHAT WE FORECAST:
#   1. daily unique visitors
#   2. daily event volume
#   3. daily converted visitor count
#   4. daily high-intent visitor count
#
# IMPORTANT BUSINESS HONESTY:
#   This is operational demand forecasting, not revenue forecasting.
#   RetailRocket gives behaviour events, but it does not provide reliable order value.
#
# WHY THIS SCRIPT EXISTS:
#   The Streamlit dashboard should not calculate forecasts live every time.
#   That would make the app slow.
#
#   Instead:
#       this script creates clean forecast tables
#       the Streamlit page reads those tables quickly
#
# RUN COMMAND:
#   python3 -m src.forecasting.build_business_forecasts
#
# OUTPUTS:
#   reports/tables/daily_business_kpis.csv
#   reports/tables/business_forecast_comparison.csv
#   reports/tables/business_forecast_future.csv
#   reports/tables/business_forecast_history_with_predictions.csv

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.production_model import (
    get_production_threshold,
    load_production_bundle,
)
from src.models.score_export import (
    FINAL_SCORE_MANIFEST_PATH,
    FINAL_SCORE_PATH,
    load_valid_cached_scores,
    save_final_champion_scores,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------
# These are relative paths from the project root.

EVENTS_PATH = Path("data/raw/retailrocket/events.csv")
VISITOR_FEATURES_PATH = Path("data/processed/visitor_features.csv")

# Explicit score output from the active production champion.
PRODUCTION_VISITOR_SCORES_PATH = FINAL_SCORE_PATH
PRODUCTION_SCORE_MANIFEST_PATH = FINAL_SCORE_MANIFEST_PATH

REPORT_TABLES_DIR = Path("reports/tables")

DAILY_KPI_PATH = REPORT_TABLES_DIR / "daily_business_kpis.csv"
FORECAST_COMPARISON_PATH = REPORT_TABLES_DIR / "business_forecast_comparison.csv"
FORECAST_FUTURE_PATH = REPORT_TABLES_DIR / "business_forecast_future.csv"
FORECAST_HISTORY_PATH = REPORT_TABLES_DIR / "business_forecast_history_with_predictions.csv"


# --------------------------------------------------
# Forecast configuration
# --------------------------------------------------

FORECAST_HORIZON_DAYS = 14
TEST_DAYS = 14
RANDOM_STATE = 42

FORECAST_TARGETS = [
    "unique_visitors",
    "event_volume",
    "converted_visitor_count",
    "high_intent_visitor_count",
]


# --------------------------------------------------
# Data class for forecast result
# --------------------------------------------------

@dataclass
class ForecastResult:
    """Container for one forecast model result."""

    target_name: str
    model_name: str
    status: str
    mae: float
    rmse: float
    smape: float
    test_predictions: pd.DataFrame
    future_predictions: pd.DataFrame
    error_message: str = ""


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def create_output_folders() -> None:
    """Create output folder for forecast reports."""

    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    """Load JSON safely."""

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_champion_threshold() -> float:
    """Read the validated threshold from active production metadata."""

    production_bundle = load_production_bundle()

    return get_production_threshold(
        production_bundle["metadata"]
    )


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate symmetric mean absolute percentage error."""

    # SMAPE is safer than MAPE when actual values can be zero.
    # Lower SMAPE is better.

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = (np.abs(actual) + np.abs(predicted)) / 2

    valid_mask = denominator != 0

    if not valid_mask.any():
        return 0.0

    smape = np.mean(
        np.abs(actual[valid_mask] - predicted[valid_mask])
        / denominator[valid_mask]
    )

    return float(smape)


def evaluate_forecast(
    target_name: str,
    model_name: str,
    test_dates: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    future_dates: pd.Series,
    future_pred: np.ndarray,
    status: str = "success",
    error_message: str = "",
) -> ForecastResult:
    """Create a standard forecast result object."""

    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Forecasts cannot be negative for business counts.
    y_pred = np.clip(y_pred, 0, None)
    future_pred = np.clip(np.asarray(future_pred, dtype=float), 0, None)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))
    smape = calculate_smape(y_test, y_pred)

    test_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "target_name": target_name,
            "model_name": model_name,
            "actual_value": y_test,
            "predicted_value": y_pred,
            "data_split": "test",
        }
    )

    future_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates),
            "target_name": target_name,
            "model_name": model_name,
            "actual_value": np.nan,
            "predicted_value": future_pred,
            "data_split": "future",
        }
    )

    return ForecastResult(
        target_name=target_name,
        model_name=model_name,
        status=status,
        mae=mae,
        rmse=rmse,
        smape=smape,
        test_predictions=test_predictions,
        future_predictions=future_predictions,
        error_message=error_message,
    )


def create_failed_result(
    target_name: str,
    model_name: str,
    error_message: str,
) -> ForecastResult:
    """Create a failed forecast result row without breaking the script."""

    return ForecastResult(
        target_name=target_name,
        model_name=model_name,
        status="failed",
        mae=np.nan,
        rmse=np.nan,
        smape=np.nan,
        test_predictions=pd.DataFrame(),
        future_predictions=pd.DataFrame(),
        error_message=error_message,
    )


# --------------------------------------------------
# High-intent visitor score preparation
# --------------------------------------------------

def ensure_visitor_scores_exist() -> pd.DataFrame:
    """Load validated cached scores or regenerate production scores."""

    if not VISITOR_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "visitor_features.csv is missing. "
            "Run feature engineering first."
        )

    visitor_features = pd.read_csv(VISITOR_FEATURES_PATH)
    production_bundle = load_production_bundle()

    cached_scores, validation_message = load_valid_cached_scores(
        production_bundle,
        visitor_features["visitorid"],
        score_path=PRODUCTION_VISITOR_SCORES_PATH,
        manifest_path=PRODUCTION_SCORE_MANIFEST_PATH,
    )

    required_columns = ["visitorid", "purchase_intent_score"]

    if cached_scores is not None:
        print(validation_message)
        return cached_scores[required_columns].copy()

    print(
        "Cached final champion scores are not reusable: "
        f"{validation_message}. Regenerating."
    )

    feature_columns = production_bundle["feature_columns"]
    required_feature_columns = ["visitorid"] + feature_columns

    missing_columns = [
        column
        for column in required_feature_columns
        if column not in visitor_features.columns
    ]

    if missing_columns:
        raise ValueError(
            "visitor_features.csv missing required columns: "
            f"{missing_columns}"
        )

    X = visitor_features[feature_columns].copy()

    score_table, _ = save_final_champion_scores(
        final_model=production_bundle["model"],
        X=X,
        visitor_ids=visitor_features["visitorid"],
        threshold=get_production_threshold(
            production_bundle["metadata"]
        ),
        model_path=production_bundle["model_path"],
        metadata_path=production_bundle["metadata_path"],
        score_path=PRODUCTION_VISITOR_SCORES_PATH,
        manifest_path=PRODUCTION_SCORE_MANIFEST_PATH,
        model_generation=production_bundle["generation"],
        chunk_size=100_000,
    )

    return score_table[required_columns].copy()


# --------------------------------------------------
# Daily KPI creation
# --------------------------------------------------

def load_events() -> pd.DataFrame:
    """Load RetailRocket events and create a date column."""

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Events file not found at {EVENTS_PATH}. "
            "Expected raw RetailRocket events.csv in data/raw/retailrocket/."
        )

    print("Loading RetailRocket events...")

    events = pd.read_csv(EVENTS_PATH)

    required_columns = ["timestamp", "visitorid", "event"]

    missing_columns = [
        column for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"events.csv missing required columns: {missing_columns}"
        )

    # RetailRocket timestamp is in milliseconds.
    events["event_datetime"] = pd.to_datetime(
        events["timestamp"],
        unit="ms",
        errors="coerce",
    )

    events = events.dropna(subset=["event_datetime"])

    events["date"] = events["event_datetime"].dt.date
    events["date"] = pd.to_datetime(events["date"])

    return events


def build_daily_business_kpis() -> pd.DataFrame:
    """Create daily business KPI table from raw events and visitor scores."""

    events = load_events()

    print("Preparing high-intent visitor mapping...")

    threshold = get_champion_threshold()

    visitor_scores = ensure_visitor_scores_exist()

    visitor_scores["visitorid"] = visitor_scores["visitorid"].astype(events["visitorid"].dtype, errors="ignore")

    high_intent_visitors = set(
        visitor_scores.loc[
            visitor_scores["purchase_intent_score"] >= threshold,
            "visitorid",
        ]
    )

    print("Aggregating daily KPIs...")

    # KPI 1: daily unique visitors.
    daily_unique_visitors = (
        events.groupby("date")["visitorid"]
        .nunique()
        .rename("unique_visitors")
        .reset_index()
    )

    # KPI 2: daily total event volume.
    daily_event_volume = (
        events.groupby("date")
        .size()
        .rename("event_volume")
        .reset_index()
    )

    # KPI 3: daily converted visitor count.
    # We count unique visitors with at least one transaction on that day.
    conversion_events = events[events["event"] == "transaction"].copy()

    daily_conversions = (
        conversion_events.groupby("date")["visitorid"]
        .nunique()
        .rename("converted_visitor_count")
        .reset_index()
    )

    # KPI 4: daily high-intent visitor count.
    # A high-intent visitor is a visitor whose model score is above the champion threshold.
    high_intent_events = events[events["visitorid"].isin(high_intent_visitors)].copy()

    daily_high_intent = (
        high_intent_events.groupby("date")["visitorid"]
        .nunique()
        .rename("high_intent_visitor_count")
        .reset_index()
    )

    # Combine all daily KPIs.
    daily_kpis = (
        daily_unique_visitors
        .merge(daily_event_volume, on="date", how="outer")
        .merge(daily_conversions, on="date", how="outer")
        .merge(daily_high_intent, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Fill missing daily counts with zero.
    for column in FORECAST_TARGETS:
        if column in daily_kpis.columns:
            daily_kpis[column] = daily_kpis[column].fillna(0)

    # Add useful derived KPIs for the dashboard.
    daily_kpis["conversion_rate"] = np.where(
        daily_kpis["unique_visitors"] > 0,
        daily_kpis["converted_visitor_count"] / daily_kpis["unique_visitors"],
        0,
    )

    daily_kpis["events_per_visitor"] = np.where(
        daily_kpis["unique_visitors"] > 0,
        daily_kpis["event_volume"] / daily_kpis["unique_visitors"],
        0,
    )

    daily_kpis["high_intent_share"] = np.where(
        daily_kpis["unique_visitors"] > 0,
        daily_kpis["high_intent_visitor_count"] / daily_kpis["unique_visitors"],
        0,
    )

    daily_kpis["date"] = pd.to_datetime(daily_kpis["date"])

    DAILY_KPI_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_kpis.to_csv(DAILY_KPI_PATH, index=False)

    print(f"Saved daily KPIs to: {DAILY_KPI_PATH}")

    return daily_kpis


# --------------------------------------------------
# Forecast feature engineering
# --------------------------------------------------

def split_series_for_forecast(daily_kpis: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one KPI series into train and test parts."""

    series_data = daily_kpis[["date", target_name]].copy()
    series_data = series_data.sort_values("date").reset_index(drop=True)

    if len(series_data) <= TEST_DAYS + 10:
        raise ValueError(
            f"Not enough days to forecast target: {target_name}"
        )

    train_data = series_data.iloc[:-TEST_DAYS].copy()
    test_data = series_data.iloc[-TEST_DAYS:].copy()

    return train_data, test_data


def create_lag_features(series_data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Create lag and calendar features for ML-style forecasting."""

    data = series_data.copy()
    data = data.sort_values("date").reset_index(drop=True)

    data["day_index"] = np.arange(len(data))
    data["day_of_week"] = data["date"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Lag features tell the model what happened before.
    data["lag_1"] = data[target_name].shift(1)
    data["lag_7"] = data[target_name].shift(7)
    data["lag_14"] = data[target_name].shift(14)

    # Rolling means smooth noisy daily behaviour.
    data["rolling_3"] = data[target_name].shift(1).rolling(3).mean()
    data["rolling_7"] = data[target_name].shift(1).rolling(7).mean()
    data["rolling_14"] = data[target_name].shift(1).rolling(14).mean()

    data = data.dropna().reset_index(drop=True)

    return data


def make_future_dates(last_date: pd.Timestamp, horizon_days: int) -> pd.Series:
    """Create future dates for forecast horizon."""

    return pd.Series(
        pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )
    )


# --------------------------------------------------
# Rolling baseline forecast
# --------------------------------------------------

def forecast_with_rolling_baseline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_name: str,
) -> ForecastResult:
    """Forecast using a simple rolling average baseline."""

    model_name = "Rolling Average Baseline"

    train_values = train_data[target_name].values
    test_values = test_data[target_name].values

    # Use the last 7 days as the simple baseline.
    window = min(7, len(train_values))

    history = list(train_values)
    test_predictions = []

    for _ in range(len(test_values)):
        prediction = np.mean(history[-window:])
        test_predictions.append(prediction)
        history.append(prediction)

    # Future forecast starts from all available real data.
    full_history = list(pd.concat([train_data[target_name], test_data[target_name]]).values)

    future_dates = make_future_dates(
        last_date=test_data["date"].max(),
        horizon_days=FORECAST_HORIZON_DAYS,
    )

    future_predictions = []

    for _ in range(FORECAST_HORIZON_DAYS):
        prediction = np.mean(full_history[-window:])
        future_predictions.append(prediction)
        full_history.append(prediction)

    return evaluate_forecast(
        target_name=target_name,
        model_name=model_name,
        test_dates=test_data["date"],
        y_test=test_values,
        y_pred=np.array(test_predictions),
        future_dates=future_dates,
        future_pred=np.array(future_predictions),
    )


# --------------------------------------------------
# ARIMA forecast
# --------------------------------------------------

def forecast_with_arima(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_name: str,
) -> ForecastResult:
    """Forecast using ARIMA if statsmodels is installed."""

    model_name = "ARIMA"

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=f"statsmodels ARIMA unavailable: {error}",
        )

    try:
        train_values = train_data[target_name].astype(float).values
        test_values = test_data[target_name].astype(float).values

        # Simple ARIMA order.
        # This keeps the script stable and fast.
        model = ARIMA(
            train_values,
            order=(1, 1, 1),
        )

        fitted_model = model.fit()

        test_predictions = fitted_model.forecast(steps=len(test_values))

        # Refit on full series for future forecast.
        full_values = pd.concat(
            [train_data[target_name], test_data[target_name]],
            ignore_index=True,
        ).astype(float).values

        full_model = ARIMA(
            full_values,
            order=(1, 1, 1),
        ).fit()

        future_predictions = full_model.forecast(steps=FORECAST_HORIZON_DAYS)

        future_dates = make_future_dates(
            last_date=test_data["date"].max(),
            horizon_days=FORECAST_HORIZON_DAYS,
        )

        return evaluate_forecast(
            target_name=target_name,
            model_name=model_name,
            test_dates=test_data["date"],
            y_test=test_values,
            y_pred=test_predictions,
            future_dates=future_dates,
            future_pred=future_predictions,
        )

    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=str(error),
        )


# --------------------------------------------------
# Prophet forecast
# --------------------------------------------------

def forecast_with_prophet(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_name: str,
) -> ForecastResult:
    """Forecast using Prophet if installed."""

    model_name = "Prophet"

    try:
        from prophet import Prophet
    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=f"Prophet unavailable: {error}",
        )

    try:
        prophet_train = train_data.rename(
            columns={
                "date": "ds",
                target_name: "y",
            }
        )[["ds", "y"]]

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        model.fit(prophet_train)

        test_future = pd.DataFrame(
            {
                "ds": test_data["date"],
            }
        )

        test_forecast = model.predict(test_future)
        test_predictions = test_forecast["yhat"].values

        # Refit on full data for future horizon.
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        prophet_full = full_data.rename(
            columns={
                "date": "ds",
                target_name: "y",
            }
        )[["ds", "y"]]

        full_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        full_model.fit(prophet_full)

        future_dates = make_future_dates(
            last_date=test_data["date"].max(),
            horizon_days=FORECAST_HORIZON_DAYS,
        )

        future_frame = pd.DataFrame({"ds": future_dates})
        future_forecast = full_model.predict(future_frame)
        future_predictions = future_forecast["yhat"].values

        return evaluate_forecast(
            target_name=target_name,
            model_name=model_name,
            test_dates=test_data["date"],
            y_test=test_data[target_name].values,
            y_pred=test_predictions,
            future_dates=future_dates,
            future_pred=future_predictions,
        )

    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=str(error),
        )


# --------------------------------------------------
# XGBoost-style ML forecast
# --------------------------------------------------

def get_xgboost_regressor():
    """Return XGBoost regressor if available, otherwise RandomForest fallback."""

    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ), "XGBoost Time-Series"

    except Exception:
        # Fallback keeps project stable even if xgboost is not installed.
        return RandomForestRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ), "RandomForest Time-Series Fallback"


def iterative_ml_forecast(
    full_series: pd.DataFrame,
    target_name: str,
    model,
    horizon_days: int,
) -> Tuple[pd.Series, np.ndarray]:
    """Create future ML forecasts one day at a time."""

    working_series = full_series[["date", target_name]].copy()
    working_series = working_series.sort_values("date").reset_index(drop=True)

    future_dates = make_future_dates(
        last_date=working_series["date"].max(),
        horizon_days=horizon_days,
    )

    predictions = []

    feature_columns = [
        "day_index",
        "day_of_week",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_3",
        "rolling_7",
        "rolling_14",
    ]

    for future_date in future_dates:
        extended = create_lag_features(
            working_series,
            target_name=target_name,
        )

        if extended.empty:
            prediction = working_series[target_name].tail(7).mean()
        else:
            last_feature_row = {
                "date": future_date,
                target_name: np.nan,
            }

            temp_series = pd.concat(
                [
                    working_series,
                    pd.DataFrame([last_feature_row]),
                ],
                ignore_index=True,
            )

            temp_features = create_lag_features(
                temp_series,
                target_name=target_name,
            )

            prediction_input = temp_features[feature_columns].tail(1)

            prediction = float(model.predict(prediction_input)[0])

        prediction = max(0, prediction)
        predictions.append(prediction)

        working_series = pd.concat(
            [
                working_series,
                pd.DataFrame(
                    [
                        {
                            "date": future_date,
                            target_name: prediction,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return future_dates, np.array(predictions)


def forecast_with_xgboost_style_ml(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_name: str,
) -> ForecastResult:
    """Forecast with lag features and XGBoost if available."""

    try:
        model, model_name = get_xgboost_regressor()

        combined_train = train_data[["date", target_name]].copy()
        feature_data = create_lag_features(
            combined_train,
            target_name=target_name,
        )

        feature_columns = [
            "day_index",
            "day_of_week",
            "is_weekend",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_3",
            "rolling_7",
            "rolling_14",
        ]

        if feature_data.empty:
            return create_failed_result(
                target_name=target_name,
                model_name=model_name,
                error_message="Not enough lag rows for ML forecast.",
            )

        X_train = feature_data[feature_columns]
        y_train = feature_data[target_name]

        model.fit(X_train, y_train)

        # For fair test prediction, forecast test days iteratively from train history.
        test_dates, test_predictions = iterative_ml_forecast(
            full_series=train_data,
            target_name=target_name,
            model=model,
            horizon_days=len(test_data),
        )

        # Refit on train + test for future forecast.
        full_series = pd.concat([train_data, test_data], ignore_index=True)
        full_feature_data = create_lag_features(
            full_series,
            target_name=target_name,
        )

        full_model, _ = get_xgboost_regressor()
        full_model.fit(
            full_feature_data[feature_columns],
            full_feature_data[target_name],
        )

        future_dates, future_predictions = iterative_ml_forecast(
            full_series=full_series,
            target_name=target_name,
            model=full_model,
            horizon_days=FORECAST_HORIZON_DAYS,
        )

        return evaluate_forecast(
            target_name=target_name,
            model_name=model_name,
            test_dates=test_data["date"],
            y_test=test_data[target_name].values,
            y_pred=test_predictions,
            future_dates=future_dates,
            future_pred=future_predictions,
        )

    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name="XGBoost Time-Series",
            error_message=str(error),
        )


# --------------------------------------------------
# Optional LSTM forecast
# --------------------------------------------------

def forecast_with_lstm_optional(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_name: str,
) -> ForecastResult:
    """Optional LSTM forecast, disabled by default for stability."""

    model_name = "LSTM Optional"

    # WHY DISABLED BY DEFAULT:
    #   TensorFlow can be heavy and environment-sensitive.
    #   This project should remain stable on your laptop.
    #
    # HOW TO ENABLE:
    #   ENABLE_LSTM_FORECAST=1 python3 -m src.forecasting.build_business_forecasts

    if os.getenv("ENABLE_LSTM_FORECAST", "0") != "1":
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message="Skipped by default. Set ENABLE_LSTM_FORECAST=1 to run.",
        )

    try:
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
        from sklearn.preprocessing import MinMaxScaler
    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=f"TensorFlow/Keras unavailable: {error}",
        )

    try:
        lookback = min(7, len(train_data) // 3)

        if lookback < 3:
            return create_failed_result(
                target_name=target_name,
                model_name=model_name,
                error_message="Not enough data for LSTM.",
            )

        scaler = MinMaxScaler()

        train_values = train_data[target_name].values.reshape(-1, 1)
        scaled_train = scaler.fit_transform(train_values)

        generator = TimeseriesGenerator(
            scaled_train,
            scaled_train,
            length=lookback,
            batch_size=8,
        )

        model = Sequential(
            [
                LSTM(16, activation="tanh", input_shape=(lookback, 1)),
                Dense(1),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        model.fit(
            generator,
            epochs=20,
            verbose=0,
        )

        history_scaled = list(scaled_train.flatten())
        test_predictions_scaled = []

        for _ in range(len(test_data)):
            input_sequence = np.array(history_scaled[-lookback:]).reshape(1, lookback, 1)
            prediction_scaled = float(model.predict(input_sequence, verbose=0)[0][0])
            test_predictions_scaled.append(prediction_scaled)
            history_scaled.append(prediction_scaled)

        test_predictions = scaler.inverse_transform(
            np.array(test_predictions_scaled).reshape(-1, 1)
        ).flatten()

        # Future from full available data.
        full_values = pd.concat([train_data[target_name], test_data[target_name]]).values.reshape(-1, 1)
        scaled_full = scaler.fit_transform(full_values)
        future_history_scaled = list(scaled_full.flatten())
        future_predictions_scaled = []

        for _ in range(FORECAST_HORIZON_DAYS):
            input_sequence = np.array(future_history_scaled[-lookback:]).reshape(1, lookback, 1)
            prediction_scaled = float(model.predict(input_sequence, verbose=0)[0][0])
            future_predictions_scaled.append(prediction_scaled)
            future_history_scaled.append(prediction_scaled)

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1)
        ).flatten()

        future_dates = make_future_dates(
            last_date=test_data["date"].max(),
            horizon_days=FORECAST_HORIZON_DAYS,
        )

        return evaluate_forecast(
            target_name=target_name,
            model_name=model_name,
            test_dates=test_data["date"],
            y_test=test_data[target_name].values,
            y_pred=test_predictions,
            future_dates=future_dates,
            future_pred=future_predictions,
        )

    except Exception as error:
        return create_failed_result(
            target_name=target_name,
            model_name=model_name,
            error_message=str(error),
        )


# --------------------------------------------------
# Run forecasts
# --------------------------------------------------

def run_forecasts_for_target(
    daily_kpis: pd.DataFrame,
    target_name: str,
) -> List[ForecastResult]:
    """Run all forecast models for one KPI target."""

    print(f"\nForecasting target: {target_name}")

    train_data, test_data = split_series_for_forecast(
        daily_kpis=daily_kpis,
        target_name=target_name,
    )

    forecast_functions = [
        forecast_with_rolling_baseline,
        forecast_with_arima,
        forecast_with_prophet,
        forecast_with_xgboost_style_ml,
        forecast_with_lstm_optional,
    ]

    results = []

    for forecast_function in forecast_functions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = forecast_function(
                train_data=train_data,
                test_data=test_data,
                target_name=target_name,
            )

        print(
            f"  {result.model_name}: "
            f"status={result.status}, "
            f"MAE={result.mae if not pd.isna(result.mae) else 'NA'}"
        )

        results.append(result)

    return results


def build_forecast_outputs(daily_kpis: pd.DataFrame) -> None:
    """Run forecasts for all targets and save output tables."""

    all_results = []

    for target_name in FORECAST_TARGETS:
        target_results = run_forecasts_for_target(
            daily_kpis=daily_kpis,
            target_name=target_name,
        )

        all_results.extend(target_results)

    comparison_rows = []
    prediction_tables = []

    for result in all_results:
        comparison_rows.append(
            {
                "target_name": result.target_name,
                "model_name": result.model_name,
                "status": result.status,
                "mae": result.mae,
                "rmse": result.rmse,
                "smape": result.smape,
                "error_message": result.error_message,
            }
        )

        if not result.test_predictions.empty:
            prediction_tables.append(result.test_predictions)

        if not result.future_predictions.empty:
            prediction_tables.append(result.future_predictions)

    comparison_table = pd.DataFrame(comparison_rows)

    # Select best model per target using MAE.
    successful_comparison = comparison_table[
        comparison_table["status"] == "success"
    ].copy()

    if not successful_comparison.empty:
        successful_comparison["rank_by_target"] = (
            successful_comparison
            .groupby("target_name")["mae"]
            .rank(method="first", ascending=True)
        )

        best_models = successful_comparison[
            successful_comparison["rank_by_target"] == 1
        ][["target_name", "model_name"]].copy()

        best_models["is_best_model"] = True

        comparison_table = comparison_table.merge(
            best_models,
            on=["target_name", "model_name"],
            how="left",
        )

        comparison_table["is_best_model"] = comparison_table["is_best_model"].fillna(False)

    else:
        comparison_table["is_best_model"] = False

    comparison_table.to_csv(
        FORECAST_COMPARISON_PATH,
        index=False,
    )

    if prediction_tables:
        prediction_table = pd.concat(prediction_tables, ignore_index=True)

        # Add best-model flag to every prediction row.
        best_lookup = comparison_table[
            comparison_table["is_best_model"] == True
        ][["target_name", "model_name"]].copy()

        best_lookup["is_best_model"] = True

        prediction_table = prediction_table.merge(
            best_lookup,
            on=["target_name", "model_name"],
            how="left",
        )

        prediction_table["is_best_model"] = prediction_table["is_best_model"].fillna(False)

        # Separate future forecast table and history/test prediction table.
        future_table = prediction_table[
            prediction_table["data_split"] == "future"
        ].copy()

        history_table = prediction_table[
            prediction_table["data_split"] == "test"
        ].copy()

        future_table.to_csv(
            FORECAST_FUTURE_PATH,
            index=False,
        )

        history_table.to_csv(
            FORECAST_HISTORY_PATH,
            index=False,
        )

    print(f"\nSaved forecast comparison to: {FORECAST_COMPARISON_PATH}")
    print(f"Saved future forecasts to: {FORECAST_FUTURE_PATH}")
    print(f"Saved history/test forecasts to: {FORECAST_HISTORY_PATH}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run the complete business KPI forecasting pipeline."""

    warnings.filterwarnings("ignore")

    print("Starting business KPI forecasting pipeline...")

    create_output_folders()

    daily_kpis = build_daily_business_kpis()

    print(f"Daily KPI rows: {len(daily_kpis):,}")
    print(f"Date range: {daily_kpis['date'].min().date()} to {daily_kpis['date'].max().date()}")

    build_forecast_outputs(daily_kpis)

    print("\nBusiness KPI forecasting pipeline complete.")


if __name__ == "__main__":
    main()
