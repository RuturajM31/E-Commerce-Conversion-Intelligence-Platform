"""Run a lightweight training check with representative sample data.

This is not a replacement for full model training. It verifies that:
- the canonical feature schema can be built;
- the target contains both classes;
- a model can train and produce probabilities;
- metrics and evidence can be written without the full local dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Add the repository root before importing ``src``.
# Pytest adds this path automatically, but direct script execution does not.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.feature_engineering import (
    BASE_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    prepare_model_features,
)
from src.models.model_config import TARGET_COLUMN


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "visitor_features_sample.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "qa" / "smoke_training_result.json"
RANDOM_STATE = 42


def load_sample_data(path: Path) -> pd.DataFrame:
    """Load and validate the representative visitor sample."""

    if not path.exists():
        raise FileNotFoundError(f"Sample data not found: {path}")

    data = pd.read_csv(path)
    required_columns = {"visitorid", TARGET_COLUMN, *BASE_FEATURE_COLUMNS}
    missing = sorted(required_columns - set(data.columns))

    if missing:
        raise ValueError("Sample data is missing required columns: " + ", ".join(missing))

    if data["visitorid"].duplicated().any():
        raise ValueError("Sample data contains duplicate visitor IDs.")

    target_values = set(data[TARGET_COLUMN].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise ValueError("Sample target must contain both classes 0 and 1.")

    return data


def run_smoke_training(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Train a small canonical model and return evidence."""

    data = load_sample_data(data_path)
    features = prepare_model_features(data[BASE_FEATURE_COLUMNS])
    target = data[TARGET_COLUMN].astype(int)

    if list(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Smoke training feature order does not match the canonical schema.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    if not ((probabilities >= 0).all() and (probabilities <= 1).all()):
        raise ValueError("Smoke model probabilities are outside 0 to 1.")

    try:
        data_label = str(data_path.relative_to(PROJECT_ROOT))
    except ValueError:
        data_label = str(data_path)

    result = {
        "status": "PASS",
        "purpose": "Small-data CI validation only; not production evidence.",
        "data_path": data_label,
        "rows": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_columns": list(features.columns),
        "target_counts": {
            str(key): int(value) for key, value in target.value_counts().sort_index().items()
        },
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a small representative training check.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run smoke training and print a concise summary."""

    args = parse_args()
    result = run_smoke_training(data_path=args.data, output_path=args.output)

    print("GOOD: sample-data smoke training passed")
    print(f"Rows: {result['rows']}")
    print("Features: " + ", ".join(result["feature_columns"]))
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC: {result['pr_auc']:.4f}")
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
