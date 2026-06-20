# analyze_thresholds.py
# Select baseline marketing thresholds using validation data only.

from __future__ import annotations

import joblib
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.config.paths import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    TRAINED_MODELS_DIR,
)
from src.models.model_config import (
    FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    THRESHOLDS_TO_TEST,
    VALIDATION_SPLIT,
)


FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "visitor_training_snapshots.csv"
)

MODEL_FILE = (
    TRAINED_MODELS_DIR
    / "baseline_logistic_regression.joblib"
)

THRESHOLD_FILE = (
    TABLES_DIR
    / "threshold_analysis.csv"
)


def load_data_and_model():
    """Load training snapshots and the baseline model."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}"
        )

    return (
        pd.read_csv(FEATURE_FILE),
        joblib.load(MODEL_FILE),
    )


def create_validation_predictions(
    model,
    data: pd.DataFrame,
):
    """Create probabilities using validation rows only."""

    required = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    validation_data = data.loc[
        data[SPLIT_COLUMN] == VALIDATION_SPLIT
    ].copy()

    if validation_data.empty:
        raise ValueError(
            "Validation split contains no rows."
        )

    X_validation = validation_data[
        FEATURE_COLUMNS
    ]

    y_validation = validation_data[
        TARGET_COLUMN
    ].astype(int)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    return y_validation, probabilities


def analyze_thresholds(
    y_validation: pd.Series,
    probabilities,
) -> pd.DataFrame:
    """Calculate targeting metrics for each threshold."""

    natural_conversion_rate = (
        y_validation.mean()
    )

    rows = []

    for threshold in THRESHOLDS_TO_TEST:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        rows.append(
            {
                "evaluation_split": VALIDATION_SPLIT,
                "threshold": float(threshold),
                "targeted_visitors": int(
                    predictions.sum()
                ),
                "targeted_share": float(
                    predictions.mean()
                ),
                "precision": precision,
                "recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "lift_vs_random": (
                    precision
                    / natural_conversion_rate
                    if natural_conversion_rate > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run validation-only threshold analysis."""

    data, model = load_data_and_model()

    y_validation, probabilities = (
        create_validation_predictions(
            model,
            data,
        )
    )

    threshold_table = analyze_thresholds(
        y_validation,
        probabilities,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_table.to_csv(
        THRESHOLD_FILE,
        index=False,
    )

    print(
        threshold_table.sort_values(
            "f1_score",
            ascending=False,
        ).head(10)
    )


if __name__ == "__main__":
    main()
