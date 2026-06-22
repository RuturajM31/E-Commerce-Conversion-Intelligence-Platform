"""Validation tests for generated model-comparison results."""

from pathlib import Path

import numpy as np
import pandas as pd


FINAL_COMPARISON_PATH = Path(
    "reports/tables/final_true_champion_comparison.csv"
)

CANDIDATE_PATHS = [
    FINAL_COMPARISON_PATH,
    Path("reports/tables/manual_model_comparison.csv"),
    Path("reports/tables/automl_benchmark_results.csv"),
]


def normalize_boolean_flags(values: pd.Series) -> pd.Series:
    """Convert supported CSV boolean values into True or False."""

    normalized = values.astype(str).str.strip().str.lower()

    valid_values = {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
    }

    invalid_values = normalized[~normalized.isin(valid_values)]

    assert invalid_values.empty, (
        "Invalid model-selection flag values: "
        f"{sorted(invalid_values.unique())}"
    )

    return normalized.isin({"true", "1", "yes"})


def test_logistic_and_xgboost_are_in_comparison_results():
    """The benchmark must evaluate both baseline and boosting families."""

    available_paths = [
        path
        for path in CANDIDATE_PATHS
        if path.exists()
    ]

    assert available_paths, (
        "No model-comparison tables are available."
    )

    comparison_tables = []

    for path in available_paths:
        table = pd.read_csv(path)

        if "model_name" in table.columns:
            comparison_tables.append(
                table[["model_name"]].copy()
            )

    assert comparison_tables, (
        "Available comparison tables have no model_name column."
    )

    comparison = pd.concat(
        comparison_tables,
        ignore_index=True,
    )

    names = (
        comparison["model_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    assert names.str.contains("logistic").any(), (
        "Logistic Regression was not found in the benchmark."
    )

    assert names.str.contains(
        "xgboost|xgb",
        regex=True,
    ).any(), (
        "XGBoost was not found in the benchmark."
    )


def test_business_scores_are_valid_and_rankable():
    """Champion selection must use valid finite business scores.

    The test does not force a specific algorithm to win. It checks that
    every candidate has a usable score and that a highest-scoring model
    can be identified deterministically.
    """

    assert FINAL_COMPARISON_PATH.exists(), (
        f"Missing final comparison table: {FINAL_COMPARISON_PATH}"
    )

    comparison = pd.read_csv(
        FINAL_COMPARISON_PATH
    )

    required_columns = {
        "model_name",
        "business_score",
    }

    missing_columns = required_columns - set(
        comparison.columns
    )

    assert not missing_columns, (
        f"Missing comparison columns: {missing_columns}"
    )

    assert not comparison.empty

    model_names = (
        comparison["model_name"]
        .astype(str)
        .str.strip()
    )

    scores = pd.to_numeric(
        comparison["business_score"],
        errors="coerce",
    )

    assert model_names.ne("").all()
    assert scores.notna().all()
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()

    best_index = scores.idxmax()
    best_model_name = model_names.loc[best_index]

    assert best_model_name

    # Validate an explicit selection flag when the table contains one.
    flag_column = next(
        (
            column
            for column in [
                "is_selected",
                "is_best_model",
            ]
            if column in comparison.columns
        ),
        None,
    )

    if flag_column is not None:
        selected = normalize_boolean_flags(
            comparison[flag_column]
        )

        assert selected.sum() == 1, (
            "Exactly one final model must be selected."
        )

        selected_score = scores.loc[selected].iloc[0]

        assert np.isclose(
            selected_score,
            scores.max(),
        ), (
            "The selected model does not have the highest "
            "recorded business score."
        )
