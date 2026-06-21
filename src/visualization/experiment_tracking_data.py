"""Prepare MLflow experiment and registry evidence for visual intelligence.

Why this file exists:
    MLflow stores runs, metrics, parameters, tags, registered models, model
    versions, and aliases in `mlflow.db`. This module converts those tables
    into one controlled, comparable evidence layer.

Main inputs:
    - mlflow.db
    - models/metadata/mlflow_champion_lineage.json

Main outputs:
    - Flattened run table
    - Comparable metric selection
    - Registered model-version history with aliases
    - Current champion lineage
    - Hyperparameter-density assessment for conditional MLV-I03

Used next:
    `experiment_tracking_visuals.py` renders MLV-I01, I02, I04, and I05.

Important limitation:
    Runs are compared only on metric keys shared by at least two runs. A metric
    from one split is never silently renamed as a metric from another split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd


MLFLOW_DB_PATH = Path("mlflow.db")
LINEAGE_PATH = Path(
    "models/metadata/mlflow_champion_lineage.json"
)

PREFERRED_METRIC_KEYS = [
    "pr_auc",
    "roc_auc",
    "f1_score",
    "business_score",
    "champion_score",
    "best_f1_score",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_f1_score",
    "validation_business_score",
    "holdout_pr_auc",
    "holdout_roc_auc",
    "holdout_f1_score",
    "holdout_business_score",
    "precision",
    "recall",
    "validation_precision",
    "validation_recall",
    "holdout_precision",
    "holdout_recall",
]


@dataclass
class ExperimentTrackingBundle:
    """Container holding all evidence required by I01, I02, I04, and I05."""

    runs: pd.DataFrame
    comparable_runs: pd.DataFrame
    metric_columns: list[str]
    primary_metric: str
    model_versions: pd.DataFrame
    experiments: pd.DataFrame
    lineage: dict[str, Any]
    counts: dict[str, int]
    parameter_density: pd.DataFrame
    i03_ready: bool


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """Return whether one SQLite table exists."""

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
    """
    row = connection.execute(
        query,
        (table_name,),
    ).fetchone()

    return row is not None


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    """Read one required MLflow SQLite table."""

    if not table_exists(connection, table_name):
        raise ValueError(
            f"Required MLflow table is missing: {table_name}"
        )

    return pd.read_sql_query(
        f'SELECT * FROM "{table_name}"',
        connection,
    )


def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Resolve one schema field across compatible MLflow versions."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            "None of the expected columns exist: "
            + ", ".join(candidates)
        )

    return None


def epoch_to_datetime(
    values: pd.Series,
) -> pd.Series:
    """Convert MLflow epoch timestamps into UTC datetimes."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    return pd.to_datetime(
        numeric,
        unit="ms",
        utc=True,
        errors="coerce",
    )


def flatten_key_value_table(
    frame: pd.DataFrame,
    *,
    prefix: str,
    value_numeric: bool,
) -> pd.DataFrame:
    """Pivot one MLflow key/value table to one row per run.

    Input:
        Metrics, parameters, or tags table.

    Output:
        Wide run-indexed DataFrame with namespaced columns.
    """

    run_column = first_existing_column(
        frame,
        ["run_uuid", "run_id"],
    )
    key_column = first_existing_column(
        frame,
        ["key"],
    )
    value_column = first_existing_column(
        frame,
        ["value"],
    )

    clean = frame.copy()

    if value_numeric:
        clean[value_column] = pd.to_numeric(
            clean[value_column],
            errors="coerce",
        )

    sort_columns = [
        column
        for column in ["step", "timestamp"]
        if column in clean.columns
    ]

    if sort_columns:
        clean = clean.sort_values(sort_columns)

    clean = clean.drop_duplicates(
        subset=[run_column, key_column],
        keep="last",
    )

    pivot = clean.pivot(
        index=run_column,
        columns=key_column,
        values=value_column,
    )
    pivot.columns = [
        f"{prefix}{column}"
        for column in pivot.columns
    ]

    return pivot.reset_index().rename(
        columns={run_column: "run_id"}
    )


def metric_display_name(metric_column: str) -> str:
    """Convert a namespaced metric key into readable display copy."""

    key = metric_column.removeprefix("metric__")
    label = key.replace("_", " ").title()
    replacements = {
        "Pr Auc": "PR-AUC",
        "Roc Auc": "ROC-AUC",
        "F1 Score": "F1",
        "F1": "F1",
    }

    for source, replacement in replacements.items():
        label = label.replace(source, replacement)

    return label


def build_run_table(
    runs: pd.DataFrame,
    experiments: pd.DataFrame,
    metrics: pd.DataFrame,
    params: pd.DataFrame,
    tags: pd.DataFrame,
) -> pd.DataFrame:
    """Build one flattened MLflow run table."""

    run_id_column = first_existing_column(
        runs,
        ["run_uuid", "run_id"],
    )
    experiment_id_column = first_existing_column(
        runs,
        ["experiment_id"],
    )
    start_column = first_existing_column(
        runs,
        ["start_time"],
    )
    end_column = first_existing_column(
        runs,
        ["end_time"],
        required=False,
    )
    status_column = first_existing_column(
        runs,
        ["status"],
        required=False,
    )
    name_column = first_existing_column(
        runs,
        ["name", "run_name"],
        required=False,
    )
    lifecycle_column = first_existing_column(
        runs,
        ["lifecycle_stage"],
        required=False,
    )

    selected = pd.DataFrame(
        {
            "run_id": runs[run_id_column].astype(str),
            "experiment_id": runs[
                experiment_id_column
            ].astype(str),
            "start_time": epoch_to_datetime(
                runs[start_column]
            ),
        }
    )

    selected["end_time"] = (
        epoch_to_datetime(runs[end_column])
        if end_column
        else pd.NaT
    )
    selected["status"] = (
        runs[status_column].astype(str)
        if status_column
        else "UNKNOWN"
    )
    selected["run_name"] = (
        runs[name_column].astype(str)
        if name_column
        else selected["run_id"].str[:8]
    )
    selected["lifecycle_stage"] = (
        runs[lifecycle_column].astype(str)
        if lifecycle_column
        else "active"
    )

    experiment_id_source = first_existing_column(
        experiments,
        ["experiment_id"],
    )
    experiment_name_source = first_existing_column(
        experiments,
        ["name"],
    )
    experiment_lookup = experiments[
        [
            experiment_id_source,
            experiment_name_source,
        ]
    ].copy()
    experiment_lookup.columns = [
        "experiment_id",
        "experiment_name",
    ]
    experiment_lookup["experiment_id"] = (
        experiment_lookup["experiment_id"].astype(str)
    )

    selected = selected.merge(
        experiment_lookup,
        on="experiment_id",
        how="left",
        validate="many_to_one",
    )

    for flattened in [
        flatten_key_value_table(
            metrics,
            prefix="metric__",
            value_numeric=True,
        ),
        flatten_key_value_table(
            params,
            prefix="param__",
            value_numeric=False,
        ),
        flatten_key_value_table(
            tags,
            prefix="tag__",
            value_numeric=False,
        ),
    ]:
        selected = selected.merge(
            flattened,
            on="run_id",
            how="left",
            validate="one_to_one",
        )

    mlflow_name_column = "tag__mlflow.runName"

    if mlflow_name_column in selected.columns:
        tag_names = selected[
            mlflow_name_column
        ].replace(
            {"nan": np.nan, "None": np.nan}
        )
        selected["run_name"] = tag_names.fillna(
            selected["run_name"]
        )

    selected["run_name"] = (
        selected["run_name"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return selected.sort_values(
        ["start_time", "run_id"],
        na_position="last",
    ).reset_index(drop=True)


def select_metric_columns(
    runs: pd.DataFrame,
    *,
    minimum_runs: int = 2,
    maximum_metrics: int = 5,
) -> list[str]:
    """Select exact metric keys shared by multiple runs."""

    metric_columns = [
        column
        for column in runs.columns
        if column.startswith("metric__")
    ]

    coverage = {
        column: int(
            pd.to_numeric(
                runs[column],
                errors="coerce",
            ).notna().sum()
        )
        for column in metric_columns
    }

    eligible = [
        column
        for column in metric_columns
        if coverage[column] >= minimum_runs
        and pd.to_numeric(
            runs[column],
            errors="coerce",
        ).nunique(dropna=True)
        >= 2
    ]

    preferred_columns = [
        f"metric__{key}"
        for key in PREFERRED_METRIC_KEYS
    ]
    ordered = [
        column
        for column in preferred_columns
        if column in eligible
    ]

    remaining = sorted(
        set(eligible).difference(ordered),
        key=lambda column: (
            -coverage[column],
            column,
        ),
    )

    selected = (ordered + remaining)[:maximum_metrics]

    if len(selected) < 2:
        raise ValueError(
            "MLflow contains fewer than two comparable metric keys."
        )

    return selected


def select_primary_metric(
    metric_columns: list[str],
) -> str:
    """Choose the main timeline metric without changing its exact key."""

    priorities = [
        "pr_auc",
        "business_score",
        "champion_score",
        "f1",
        "roc_auc",
        "precision",
        "recall",
    ]

    for token in priorities:
        for column in metric_columns:
            if token in column.lower():
                return column

    return metric_columns[0]


def build_comparable_runs(
    runs: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Keep active runs with at least two selected comparable metrics."""

    clean = runs.copy()

    for column in metric_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    selected_count = clean[
        metric_columns
    ].notna().sum(axis=1)
    comparable = clean.loc[
        selected_count >= 2
    ].copy()

    active_mask = (
        comparable["lifecycle_stage"]
        .astype(str)
        .str.lower()
        != "deleted"
    )
    comparable = comparable.loc[active_mask]

    if comparable.empty:
        raise ValueError(
            "No MLflow runs contain at least two comparable metrics."
        )

    return comparable.sort_values(
        ["start_time", "run_name"],
        na_position="last",
    ).reset_index(drop=True)


def build_model_version_table(
    model_versions: pd.DataFrame,
    aliases: pd.DataFrame,
    runs: pd.DataFrame,
    primary_metric: str,
) -> pd.DataFrame:
    """Build registry version history with aliases and run metric evidence."""

    name_column = first_existing_column(
        model_versions,
        ["name"],
    )
    version_column = first_existing_column(
        model_versions,
        ["version"],
    )
    run_column = first_existing_column(
        model_versions,
        ["run_id", "run_uuid"],
        required=False,
    )
    creation_column = first_existing_column(
        model_versions,
        ["creation_time", "creation_timestamp"],
        required=False,
    )
    stage_column = first_existing_column(
        model_versions,
        ["current_stage"],
        required=False,
    )
    status_column = first_existing_column(
        model_versions,
        ["status"],
        required=False,
    )

    versions = pd.DataFrame(
        {
            "model_name": model_versions[
                name_column
            ].astype(str),
            "version": pd.to_numeric(
                model_versions[version_column],
                errors="coerce",
            ),
            "run_id": (
                model_versions[run_column].astype(str)
                if run_column
                else ""
            ),
            "creation_time": (
                epoch_to_datetime(
                    model_versions[creation_column]
                )
                if creation_column
                else pd.NaT
            ),
            "current_stage": (
                model_versions[stage_column].astype(str)
                if stage_column
                else ""
            ),
            "status": (
                model_versions[status_column].astype(str)
                if status_column
                else ""
            ),
        }
    )

    alias_name_column = first_existing_column(
        aliases,
        ["name"],
    )
    alias_column = first_existing_column(
        aliases,
        ["alias"],
    )
    alias_version_column = first_existing_column(
        aliases,
        ["version"],
    )

    alias_table = aliases[
        [
            alias_name_column,
            alias_version_column,
            alias_column,
        ]
    ].copy()
    alias_table.columns = [
        "model_name",
        "version",
        "alias",
    ]
    alias_table["version"] = pd.to_numeric(
        alias_table["version"],
        errors="coerce",
    )

    alias_summary = (
        alias_table
        .groupby(["model_name", "version"])["alias"]
        .apply(
            lambda values: ", ".join(
                sorted(
                    {
                        str(value)
                        for value in values
                        if pd.notna(value)
                    }
                )
            )
        )
        .reset_index()
    )

    versions = versions.merge(
        alias_summary,
        on=["model_name", "version"],
        how="left",
        validate="one_to_one",
    )
    versions["alias"] = versions[
        "alias"
    ].fillna("")

    metric_lookup = runs[
        ["run_id", "run_name", primary_metric]
    ].copy()
    metric_lookup[primary_metric] = pd.to_numeric(
        metric_lookup[primary_metric],
        errors="coerce",
    )

    versions = versions.merge(
        metric_lookup,
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    return versions.sort_values(
        ["creation_time", "model_name", "version"],
        na_position="last",
    ).reset_index(drop=True)


def build_parameter_density(
    comparable_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Assess whether numeric hyperparameters are dense enough for I03.

    Input:
        Comparable run table containing optional ``param__`` columns.

    Output:
        A stable parameter-density table. When no parameter can be converted
        to numeric values, an empty DataFrame with the full expected schema is
        returned instead of raising a ``KeyError``.

    Used next:
        The experiment package marks MLV-I03 as conditional when the returned
        table contains fewer than two dense numeric parameters.
    """

    output_columns = [
        "parameter",
        "non_null_runs",
        "unique_values",
        "minimum",
        "maximum",
        "dense_candidate",
    ]
    records: list[dict[str, Any]] = []

    for column in comparable_runs.columns:
        if not column.startswith("param__"):
            continue

        numeric = pd.to_numeric(
            comparable_runs[column],
            errors="coerce",
        )
        non_null = int(numeric.notna().sum())
        unique_values = int(
            numeric.nunique(dropna=True)
        )

        # Text-only parameters are valid MLflow metadata, but they cannot be
        # used to construct a numeric response surface.
        if non_null == 0:
            continue

        records.append(
            {
                "parameter": column.removeprefix(
                    "param__"
                ),
                "non_null_runs": non_null,
                "unique_values": unique_values,
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
                "dense_candidate": (
                    non_null >= 6
                    and unique_values >= 3
                ),
            }
        )

    density = pd.DataFrame(
        records,
        columns=output_columns,
    )

    # No numeric parameters is a valid conditional state for MLV-I03.
    if density.empty:
        return density

    return density.sort_values(
        ["dense_candidate", "unique_values", "non_null_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_experiment_tracking_bundle(
    project_root: str | Path = ".",
) -> ExperimentTrackingBundle:
    """Load the local MLflow database and build all experiment evidence."""

    root = Path(project_root)
    database_file = root / MLFLOW_DB_PATH
    lineage_file = root / LINEAGE_PATH

    if not database_file.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {database_file}"
        )

    lineage = read_json(lineage_file)

    with sqlite3.connect(database_file) as connection:
        experiments = read_table(
            connection,
            "experiments",
        )
        runs_source = read_table(
            connection,
            "runs",
        )
        metrics = read_table(
            connection,
            "metrics",
        )
        params = read_table(
            connection,
            "params",
        )
        tags = read_table(
            connection,
            "tags",
        )
        registered_models = read_table(
            connection,
            "registered_models",
        )
        model_versions = read_table(
            connection,
            "model_versions",
        )
        aliases = read_table(
            connection,
            "registered_model_aliases",
        )

    runs = build_run_table(
        runs_source,
        experiments,
        metrics,
        params,
        tags,
    )
    metric_columns = select_metric_columns(runs)
    primary_metric = select_primary_metric(
        metric_columns
    )
    comparable_runs = build_comparable_runs(
        runs,
        metric_columns,
    )
    versions = build_model_version_table(
        model_versions,
        aliases,
        runs,
        primary_metric,
    )
    parameter_density = build_parameter_density(
        comparable_runs
    )

    dense_parameter_count = (
        int(
            parameter_density[
                "dense_candidate"
            ].sum()
        )
        if not parameter_density.empty
        else 0
    )
    i03_ready = (
        dense_parameter_count >= 2
        and len(comparable_runs) >= 8
    )

    counts = {
        "experiments": len(experiments),
        "runs": len(runs_source),
        "metrics": len(metrics),
        "params": len(params),
        "tags": len(tags),
        "registered_models": len(
            registered_models
        ),
        "model_versions": len(model_versions),
        "registered_model_aliases": len(
            aliases
        ),
        "comparable_runs": len(
            comparable_runs
        ),
    }

    return ExperimentTrackingBundle(
        runs=runs,
        comparable_runs=comparable_runs,
        metric_columns=metric_columns,
        primary_metric=primary_metric,
        model_versions=versions,
        experiments=experiments,
        lineage=lineage,
        counts=counts,
        parameter_density=parameter_density,
        i03_ready=i03_ready,
    )
