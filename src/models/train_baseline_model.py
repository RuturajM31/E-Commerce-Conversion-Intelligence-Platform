# train_baseline_model.py
# Train the first balanced Logistic Regression baseline for conversion prediction.

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.paths import PROCESSED_DATA_DIR, TABLES_DIR, TRAINED_MODELS_DIR
from src.models.model_config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN

FEATURE_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"
MODEL_FILE = TRAINED_MODELS_DIR / "baseline_logistic_regression.joblib"
METRICS_FILE = TABLES_DIR / "baseline_model_metrics.csv"


def load_feature_data() -> pd.DataFrame:
    """Load visitor-level feature data."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")
    return pd.read_csv(FEATURE_FILE)


def split_features_and_target(data: pd.DataFrame):
    """Create X/y using shared model config."""
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].astype(int)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a balanced Logistic Regression pipeline."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate baseline on unseen test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def save_outputs(model: Pipeline, metrics: dict) -> pd.DataFrame:
    """Save baseline model and metrics table."""
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    metrics_table = pd.DataFrame([metrics])
    metrics_table.to_csv(METRICS_FILE, index=False)
    return metrics_table


def main() -> None:
    """Run full baseline training workflow."""
    print("Loading feature data...")
    data = load_feature_data()
    X, y = split_features_and_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics_table = save_outputs(model, metrics)

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(metrics_table)


if __name__ == "__main__":
    main()
