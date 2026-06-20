from pathlib import Path

import pandas as pd
import pytest

from src.models.model_config import (
    FEATURE_COLUMNS,
    FINAL_HOLDOUT_SPLIT,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


TRAINING_DATA_PATH = Path(
    "data/processed/visitor_training_snapshots.csv"
)

SCORING_DATA_PATH = Path(
    "data/processed/visitor_features.csv"
)


def read_csv_or_skip(path: Path) -> pd.DataFrame:
    """Load a generated CSV or skip until the pipeline creates it."""

    if not path.exists():
        pytest.skip(f"File not available yet: {path}")

    return pd.read_csv(path)


def test_production_scoring_features_have_expected_grain():
    """Production features must contain one current row per visitor."""

    data = read_csv_or_skip(SCORING_DATA_PATH)

    required_columns = [
        "visitorid",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    assert not missing
    assert not data.empty
    assert data["visitorid"].is_unique
    assert "converted" not in data.columns
    assert SPLIT_COLUMN not in data.columns


def test_training_snapshots_have_required_columns():
    """Training data must include features, target, and saved split."""

    data = read_csv_or_skip(TRAINING_DATA_PATH)

    required_columns = [
        "visitorid",
        "snapshot_time",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    assert not missing
    assert not data.empty


def test_training_target_is_binary():
    """Future-conversion target must contain only zero and one."""

    data = read_csv_or_skip(TRAINING_DATA_PATH)

    values = set(
        data[TARGET_COLUMN]
        .dropna()
        .astype(int)
        .unique()
    )

    assert values.issubset({0, 1})
    assert values == {0, 1}


def test_training_splits_are_complete():
    """All three chronological modelling splits must exist."""

    data = read_csv_or_skip(TRAINING_DATA_PATH)

    expected = {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        FINAL_HOLDOUT_SPLIT,
    }

    actual = set(
        data[SPLIT_COLUMN]
        .dropna()
        .astype(str)
        .unique()
    )

    assert actual == expected


def test_training_snapshot_grain_is_unique():
    """One visitor can repeat over time, but not within one snapshot."""

    data = read_csv_or_skip(TRAINING_DATA_PATH)

    duplicates = data.duplicated(
        subset=[
            "visitorid",
            "snapshot_time",
        ]
    ).sum()

    assert duplicates == 0


def test_model_features_are_complete():
    """Model input columns must not contain missing values."""

    data = read_csv_or_skip(TRAINING_DATA_PATH)

    assert not data[FEATURE_COLUMNS].isna().any().any()
