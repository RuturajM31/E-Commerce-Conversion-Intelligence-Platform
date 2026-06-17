# analyze_thresholds.py
# Baseline threshold analysis for marketing targeting.

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config.paths import PROCESSED_DATA_DIR, TABLES_DIR, TRAINED_MODELS_DIR
from src.models.model_config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, THRESHOLDS_TO_TEST

FEATURE_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"
MODEL_FILE = TRAINED_MODELS_DIR / "baseline_logistic_regression.joblib"
THRESHOLD_FILE = TABLES_DIR / "threshold_analysis.csv"


def load_data_and_model():
    """Load visitor features and the trained baseline model."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
    return pd.read_csv(FEATURE_FILE), joblib.load(MODEL_FILE)


def split_features_and_target(data: pd.DataFrame):
    """Create X/y using the shared feature configuration."""
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].astype(int)


def create_test_predictions(model, X: pd.DataFrame, y: pd.Series):
    """Create probabilities on the same split style as baseline training."""
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return y_test, model.predict_proba(X_test)[:, 1]


def analyze_thresholds(y_test: pd.Series, y_proba) -> pd.DataFrame:
    """Calculate targeting metrics for each probability threshold."""
    natural_conversion_rate = y_test.mean()
    rows = []

    for threshold in THRESHOLDS_TO_TEST:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        rows.append({
            "threshold": float(threshold),
            "targeted_visitors": int(y_pred.sum()),
            "targeted_share": float(y_pred.mean()),
            "precision": precision,
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "lift_vs_random": precision / natural_conversion_rate if natural_conversion_rate > 0 else 0,
        })

    return pd.DataFrame(rows)


def save_threshold_table(threshold_table: pd.DataFrame) -> None:
    """Save threshold analysis table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(THRESHOLD_FILE, index=False)


def main() -> None:
    """Run baseline threshold analysis."""
    print("Loading data and baseline model...")
    data, model = load_data_and_model()
    X, y = split_features_and_target(data)
    y_test, y_proba = create_test_predictions(model, X, y)
    threshold_table = analyze_thresholds(y_test, y_proba)
    save_threshold_table(threshold_table)

    print(f"Saved threshold analysis: {THRESHOLD_FILE}")
    print(threshold_table.sort_values("f1_score", ascending=False).head(10))


if __name__ == "__main__":
    main()
