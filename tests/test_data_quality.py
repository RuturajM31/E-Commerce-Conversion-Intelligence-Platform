from pathlib import Path

import pandas as pd
import pytest

from src.models.model_config import FEATURE_COLUMNS, TARGET_COLUMN


VISITOR_FEATURES_PATH = Path("data/processed/visitor_features.csv")
CHAMPION_SCORES_PATHS = [
    Path("data/processed/final_champion_visitor_scores.csv"),
    Path("data/processed/champion_visitor_scores.csv"),
    Path("data/processed/visitor_scores.csv"),
]


def read_csv_or_skip(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise skip this test."""

    if not path.exists():
        pytest.skip(f"File not available yet: {path}")

    return pd.read_csv(path)


def test_visitor_features_have_one_row_per_visitor():
    """The visitor feature table must keep the project grain: one row = one visitor."""

    data = read_csv_or_skip(VISITOR_FEATURES_PATH)

    assert "visitorid" in data.columns, "visitorid column is required."
    duplicate_count = int(data["visitorid"].duplicated().sum())

    assert duplicate_count == 0, f"Duplicate visitor rows found: {duplicate_count}"


def test_visitor_features_have_no_missing_core_values():
    """Core modelling columns should not contain missing values."""

    data = read_csv_or_skip(VISITOR_FEATURES_PATH)

    required_columns = ["visitorid", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]

    assert not missing_columns, f"Missing required columns: {missing_columns}"

    missing_values = data[required_columns].isna().sum()
    missing_values = missing_values[missing_values > 0]

    assert missing_values.empty, f"Missing values found:\n{missing_values}"


def test_target_is_binary():
    """The conversion target must contain only 0 and 1."""

    data = read_csv_or_skip(VISITOR_FEATURES_PATH)

    values = set(data[TARGET_COLUMN].dropna().astype(int).unique())

    assert values.issubset({0, 1}), f"Target contains invalid values: {values}"


def test_model_feature_columns_do_not_include_target_leakage():
    """Model input features must not include obvious target leakage columns."""

    forbidden_terms = [
        "converted",
        "transaction",
        "purchase",
        "revenue",
        "buyer",
    ]

    leakage_features = [
        feature
        for feature in FEATURE_COLUMNS
        if any(term in feature.lower() for term in forbidden_terms)
    ]

    assert not leakage_features, f"Possible leakage features found: {leakage_features}"


def test_visitor_score_files_keep_one_row_per_visitor_when_available():
    """Scored visitor files should also keep one row per visitor."""

    available_paths = [path for path in CHAMPION_SCORES_PATHS if path.exists()]

    if not available_paths:
        pytest.skip("No visitor score file is available yet.")

    for path in available_paths:
        data = pd.read_csv(path)

        if "visitorid" not in data.columns:
            pytest.skip(f"{path} has no visitorid column.")

        duplicate_count = int(data["visitorid"].duplicated().sum())

        assert duplicate_count == 0, f"{path} has duplicate visitor rows: {duplicate_count}"
