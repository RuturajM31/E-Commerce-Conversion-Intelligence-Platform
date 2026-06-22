"""Prepare robustness and stability evidence for ML Visual Intelligence.

Why this file exists:
    The project already contains seed-stability and anomaly-sensitivity tables.
    This module turns those sources into one controlled calculation layer.

Main inputs:
    - reports/tables/final_true_champion_stability.csv
    - reports/tables/final_true_champion_sensitivity.csv
    - models/metadata/final_champion_metadata.json

Main outputs:
    - Clean seed-level stability rows
    - Per-model stability summary statistics
    - Paired sensitivity deltas
    - Saved champion identity and strongest challenger

Used next:
    `robustness_visuals.py` renders MLV-F01 to MLV-F05 without duplicating
    calculations across charts.

Important limitation:
    Seed stability currently contains only three seeds per model. This is
    useful evidence, but it is not a full statistical uncertainty study.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STABILITY_PATH = Path(
    "reports/tables/final_true_champion_stability.csv"
)
SENSITIVITY_PATH = Path(
    "reports/tables/final_true_champion_sensitivity.csv"
)
METADATA_PATH = Path(
    "models/metadata/final_champion_metadata.json"
)

STABILITY_REQUIRED_COLUMNS = {
    "model_name",
    "seed",
    "test_rows",
    "positive_rate",
    "pr_auc",
    "roc_auc",
}

SENSITIVITY_REQUIRED_COLUMNS = {
    "model_name",
    "evaluation_group",
    "validation_rows",
    "pr_auc",
    "roc_auc",
    "note",
}

ALL_GROUP = "all_validation_visitors"
NON_ANOMALOUS_GROUP = "non_anomalous_validation_visitors"


@dataclass
class RobustnessBundle:
    """Container holding all evidence required by F01 to F05.

    Inputs stored:
        Seed-level results, per-model summaries, paired sensitivity rows, and
        saved champion identity.

    Outputs enabled:
        Heatmap, distribution, profile, tornado, and waterfall visuals.
    """

    stability: pd.DataFrame
    stability_summary: pd.DataFrame
    sensitivity: pd.DataFrame
    sensitivity_pairs: pd.DataFrame
    champion_name: str
    challenger_name: str
    seed_count: int


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def validate_stability_schema(frame: pd.DataFrame) -> None:
    """Validate the seed-stability source contract."""

    missing = sorted(
        STABILITY_REQUIRED_COLUMNS.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Stability source is missing required columns: "
            + ", ".join(missing)
        )


def validate_sensitivity_schema(frame: pd.DataFrame) -> None:
    """Validate the anomaly-sensitivity source contract."""

    missing = sorted(
        SENSITIVITY_REQUIRED_COLUMNS.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Sensitivity source is missing required columns: "
            + ", ".join(missing)
        )


def prepare_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate seed-level model results.

    Input:
        Raw stability DataFrame.

    Output:
        Sorted numeric seed-level table.

    Used next:
        F01, F02, F03, and the stability evidence CSV.
    """

    validate_stability_schema(frame)

    clean = frame.copy()

    numeric_columns = [
        "seed",
        "test_rows",
        "positive_rate",
        "pr_auc",
        "roc_auc",
    ]

    for column in numeric_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="raise",
        )

    if clean.empty:
        raise ValueError("Stability source contains no rows.")

    duplicated = clean.duplicated(
        subset=["model_name", "seed"],
    )

    if duplicated.any():
        raise ValueError(
            "Stability source contains duplicate model/seed rows."
        )

    seed_counts = clean.groupby(
        "model_name"
    )["seed"].nunique()

    if int(seed_counts.min()) < 2:
        raise ValueError(
            "Each model needs at least two seeds for variability analysis."
        )

    return clean.sort_values(
        ["model_name", "seed"],
    ).reset_index(drop=True)


def build_stability_summary(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate per-model performance and variability statistics.

    Input:
        Clean seed-level stability table.

    Output:
        Means, standard deviations, ranges, coefficients of variation, and
        rank for PR-AUC and ROC-AUC.

    Used next:
        F02 and F03 compare performance with consistency.
    """

    summary = (
        stability
        .groupby("model_name")
        .agg(
            seed_count=("seed", "nunique"),
            test_rows=("test_rows", "median"),
            positive_rate=("positive_rate", "median"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_min=("pr_auc", "min"),
            pr_auc_max=("pr_auc", "max"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
        )
        .reset_index()
    )

    summary["pr_auc_range"] = (
        summary["pr_auc_max"]
        - summary["pr_auc_min"]
    )
    summary["roc_auc_range"] = (
        summary["roc_auc_max"]
        - summary["roc_auc_min"]
    )
    summary["pr_auc_cv"] = (
        summary["pr_auc_std"]
        / summary["pr_auc_mean"].replace(0, np.nan)
    ).fillna(0.0)
    summary["roc_auc_cv"] = (
        summary["roc_auc_std"]
        / summary["roc_auc_mean"].replace(0, np.nan)
    ).fillna(0.0)

    summary["performance_rank"] = (
        summary["pr_auc_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return summary.sort_values(
        ["performance_rank", "roc_auc_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)


def prepare_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the full-versus-non-anomalous sensitivity source."""

    validate_sensitivity_schema(frame)

    clean = frame.copy()

    for column in [
        "validation_rows",
        "pr_auc",
        "roc_auc",
    ]:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="raise",
        )

    if clean.empty:
        raise ValueError("Sensitivity source contains no rows.")

    required_groups = {
        ALL_GROUP,
        NON_ANOMALOUS_GROUP,
    }
    available_groups = set(
        clean["evaluation_group"].astype(str)
    )
    missing_groups = sorted(
        required_groups.difference(available_groups)
    )

    if missing_groups:
        raise ValueError(
            "Sensitivity source is missing evaluation groups: "
            + ", ".join(missing_groups)
        )

    duplicated = clean.duplicated(
        subset=["model_name", "evaluation_group"],
    )

    if duplicated.any():
        raise ValueError(
            "Sensitivity source contains duplicate model/group rows."
        )

    return clean.sort_values(
        ["model_name", "evaluation_group"],
    ).reset_index(drop=True)


def build_sensitivity_pairs(
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Pair full-validation and non-anomalous results per model.

    Input:
        Clean sensitivity table.

    Output:
        One row per model with start, end, absolute delta, and relative change.

    Used next:
        F04 and F05 use the same paired evidence.
    """

    metrics = [
        "validation_rows",
        "pr_auc",
        "roc_auc",
    ]

    pivot = sensitivity.pivot(
        index="model_name",
        columns="evaluation_group",
        values=metrics,
    )

    required_columns = []

    for metric in metrics:
        required_columns.extend(
            [
                (metric, ALL_GROUP),
                (metric, NON_ANOMALOUS_GROUP),
            ]
        )

    missing = [
        column
        for column in required_columns
        if column not in pivot.columns
    ]

    if missing:
        raise ValueError(
            "Sensitivity pairing is incomplete for one or more models."
        )

    records: list[dict[str, Any]] = []

    for model_name in pivot.index:
        all_rows = int(
            pivot.loc[
                model_name,
                ("validation_rows", ALL_GROUP),
            ]
        )
        non_anomalous_rows = int(
            pivot.loc[
                model_name,
                ("validation_rows", NON_ANOMALOUS_GROUP),
            ]
        )
        all_pr = float(
            pivot.loc[
                model_name,
                ("pr_auc", ALL_GROUP),
            ]
        )
        non_pr = float(
            pivot.loc[
                model_name,
                ("pr_auc", NON_ANOMALOUS_GROUP),
            ]
        )
        all_roc = float(
            pivot.loc[
                model_name,
                ("roc_auc", ALL_GROUP),
            ]
        )
        non_roc = float(
            pivot.loc[
                model_name,
                ("roc_auc", NON_ANOMALOUS_GROUP),
            ]
        )

        records.append(
            {
                "model_name": str(model_name),
                "all_validation_rows": all_rows,
                "non_anomalous_rows": non_anomalous_rows,
                "removed_rows": all_rows - non_anomalous_rows,
                "all_pr_auc": all_pr,
                "non_anomalous_pr_auc": non_pr,
                "pr_auc_delta": non_pr - all_pr,
                "pr_auc_relative_change": (
                    (non_pr - all_pr) / all_pr
                    if all_pr != 0
                    else 0.0
                ),
                "all_roc_auc": all_roc,
                "non_anomalous_roc_auc": non_roc,
                "roc_auc_delta": non_roc - all_roc,
                "roc_auc_relative_change": (
                    (non_roc - all_roc) / all_roc
                    if all_roc != 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "pr_auc_delta",
    ).reset_index(drop=True)


def choose_challenger(
    summary: pd.DataFrame,
    champion_name: str,
) -> str:
    """Choose the strongest non-champion by mean PR-AUC."""

    candidates = summary.loc[
        summary["model_name"] != champion_name
    ].sort_values(
        ["pr_auc_mean", "roc_auc_mean"],
        ascending=False,
    )

    if candidates.empty:
        raise ValueError(
            "No challenger model is available for stability comparison."
        )

    return str(candidates.iloc[0]["model_name"])


def build_robustness_bundle(
    project_root: str | Path = ".",
) -> RobustnessBundle:
    """Load real project sources and build the complete robustness bundle."""

    root = Path(project_root)
    stability_file = root / STABILITY_PATH
    sensitivity_file = root / SENSITIVITY_PATH
    metadata_file = root / METADATA_PATH

    if not stability_file.exists():
        raise FileNotFoundError(
            f"Stability source not found: {stability_file}"
        )

    if not sensitivity_file.exists():
        raise FileNotFoundError(
            f"Sensitivity source not found: {sensitivity_file}"
        )

    stability = prepare_stability(
        pd.read_csv(stability_file)
    )
    sensitivity = prepare_sensitivity(
        pd.read_csv(sensitivity_file)
    )
    metadata = read_json(metadata_file)

    champion_name = str(
        metadata.get("final_model_name", "")
    ).strip()

    if not champion_name:
        raise ValueError(
            "Champion metadata does not contain `final_model_name`."
        )

    stability_models = set(
        stability["model_name"].astype(str)
    )
    sensitivity_models = set(
        sensitivity["model_name"].astype(str)
    )

    if champion_name not in stability_models:
        raise ValueError(
            "Saved champion is missing from the stability source."
        )

    if champion_name not in sensitivity_models:
        raise ValueError(
            "Saved champion is missing from the sensitivity source."
        )

    summary = build_stability_summary(stability)
    pairs = build_sensitivity_pairs(sensitivity)
    challenger_name = choose_challenger(
        summary,
        champion_name,
    )

    seed_count = int(
        stability.groupby("model_name")["seed"]
        .nunique()
        .min()
    )

    return RobustnessBundle(
        stability=stability,
        stability_summary=summary,
        sensitivity=sensitivity,
        sensitivity_pairs=pairs,
        champion_name=champion_name,
        challenger_name=challenger_name,
        seed_count=seed_count,
    )
