# train_baseline_model.py
# Train the balanced Logistic Regression baseline using chronological splits.

from __future__ import annotations

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.paths import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    TRAINED_MODELS_DIR,
)
from src.models.model_config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_SPLIT,
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

METRICS_FILE = (
    TABLES_DIR
    / "baseline_model_metrics.csv"
)


def load_feature_data() -> pd.DataFrame:
    """Load leakage-safe visitor training snapshots."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_FILE}"
        )

    return pd.read_csv(FEATURE_FILE)


def create_train_validation_data(
    data: pd.DataFrame,
):
    """Return predefined train and validation matrices."""

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

    train_data = data.loc[
        data[SPLIT_COLUMN] == TRAIN_SPLIT
    ]

    validation_data = data.loc[
        data[SPLIT_COLUMN] == VALIDATION_SPLIT
    ]

    if train_data.empty or validation_data.empty:
        raise ValueError(
            "Train and validation splits are required."
        )

    return (
        train_data[FEATURE_COLUMNS].copy(),
        train_data[TARGET_COLUMN].astype(int),
        validation_data[FEATURE_COLUMNS].copy(),
        validation_data[TARGET_COLUMN].astype(int),
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a balanced Logistic Regression pipeline."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Evaluate the baseline on validation only."""

    y_pred = model.predict(X_validation)

    y_proba = model.predict_proba(
        X_validation
    )[:, 1]

    return {
        "evaluation_split": VALIDATION_SPLIT,
        "validation_rows": int(
            len(X_validation)
        ),
        "positive_rate": float(
            y_validation.mean()
        ),
        "accuracy": accuracy_score(
            y_validation,
            y_pred,
        ),
        "precision": precision_score(
            y_validation,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_validation,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            y_proba,
        ),
        "pr_auc": average_precision_score(
            y_validation,
            y_proba,
        ),
    }


def save_outputs(
    model: Pipeline,
    metrics: dict,
) -> pd.DataFrame:
    """Save the baseline artifact and validation metrics."""

    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    metrics_table = pd.DataFrame(
        [metrics]
    )

    metrics_table.to_csv(
        METRICS_FILE,
        index=False,
    )

    return metrics_table


def main() -> None:
    """Run baseline training and validation."""

    data = load_feature_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = create_train_validation_data(data)

    print(
        f"Training rows: {len(X_train):,} | "
        f"Validation rows: {len(X_validation):,}"
    )

    model = train_model(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    metrics_table = save_outputs(
        model,
        metrics,
    )

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(metrics_table)


if __name__ == "__main__":
    main()
