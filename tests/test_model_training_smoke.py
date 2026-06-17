import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.models.model_config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def read_training_sample_or_skip(max_rows: int = 30_000) -> pd.DataFrame:
    """Load a safe sample of visitor features for fast model smoke tests."""

    path = Path("data/processed/visitor_features.csv")

    if not path.exists():
        pytest.skip("visitor_features.csv is not available.")

    data = pd.read_csv(path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]

    if missing:
        pytest.skip(f"Missing required model columns: {missing}")

    data = data[required].dropna().copy()

    if data[TARGET_COLUMN].nunique() < 2:
        pytest.skip("Need both converted and non-converted examples for model smoke test.")

    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=RANDOM_STATE)

    return data


def test_random_forest_retrains_on_safe_sample():
    """Random Forest should train and produce valid probability scores on a safe sample.

    This is a fast smoke test, not the full final champion hardening workflow.
    """

    data = read_training_sample_or_skip()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    assert len(scores) == len(X_test)
    assert ((scores >= 0) & (scores <= 1)).all()

    pr_auc = average_precision_score(y_test, scores)

    assert 0 <= pr_auc <= 1


def test_full_final_champion_retrain_command_optional():
    """Optionally run the full final champion retraining command.

    This is skipped by default because it can take time on the full RetailRocket data.
    To run it manually:
        RUN_FULL_PIPELINE_TESTS=1 pytest tests/test_model_training_smoke.py -q
    """

    if os.getenv("RUN_FULL_PIPELINE_TESTS") != "1":
        pytest.skip("Full retrain test skipped. Set RUN_FULL_PIPELINE_TESTS=1 to run it.")

    result = subprocess.run(
        [sys.executable, "-m", "src.models.finalize_true_champion"],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    assert result.returncode == 0, result.stderr
    assert Path("models/trained/final_champion_model.joblib").exists()
    assert Path("models/metadata/final_champion_metadata.json").exists()
