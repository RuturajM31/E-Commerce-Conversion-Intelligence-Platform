"""Prepare strictly verified ecommerce MLflow evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd


MLFLOW_DB_PATH = Path("mlflow.db")
LINEAGE_PATH = Path("models/metadata/mlflow_champion_lineage.json")

FORBIDDEN_TEXT_TOKENS = (
    "mlflow demo",
    "mlflow-demo",
    "prompt",
    "groundedness",
    "correctness",
    "relevance",
    "safety",
    "toxicity",
    "fluency",
    "trace-level",
)

ECOMMERCE_METRIC_TOKENS = (
    "pr_auc",
    "roc_auc",
    "f1",
    "business_score",
    "champion_score",
    "precision",
    "recall",
    "accuracy",
    "threshold",
)


@dataclass
class ExperimentTrackingBundle:
    """Strict experiment evidence bundle used by conditional pages."""

    runs: pd.DataFrame
    project_runs: pd.DataFrame
    comparable_runs: pd.DataFrame
    metric_columns: list[str]
    primary_metric: str
    model_versions: pd.DataFrame
    experiments: pd.DataFrame
    lineage: dict[str, Any]
    counts: dict[str, int]
    parameter_density: pd.DataFrame
    comparison_ready: bool
    comparison_reason: str
    i03_ready: bool


def read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON source not found: {path}")

    content = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Return whether one SQLite table exists."""

    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Read one required MLflow table."""

    if not table_exists(connection, table_name):
        raise ValueError(f"Required MLflow table is missing: {table_name}")

    return pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)


def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Resolve one field across compatible MLflow schemas."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            "None of the expected columns exist: " + ", ".join(candidates)
        )

    return None


def epoch_to_datetime(values: pd.Series) -> pd.Series:
    """Convert MLflow millisecond timestamps to UTC."""

    return pd.to_datetime(
        pd.to_numeric(values, errors="coerce"),
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
    """Pivot metrics, parameters, or tags to one row per run."""

    run_column = first_existing_column(frame, ["run_uuid", "run_id"])
    key_column = first_existing_column(frame, ["key"])
    value_column = first_existing_column(frame, ["value"])

    clean = frame.copy()

    if value_numeric:
        clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")

    sort_columns = [
        column for column in ["step", "timestamp"] if column in clean.columns
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
    pivot.columns = [f"{prefix}{column}" for column in pivot.columns]

    return pivot.reset_index().rename(columns={run_column: "run_id"})


def build_run_table(
    runs: pd.DataFrame,
    experiments: pd.DataFrame,
    metrics: pd.DataFrame,
    params: pd.DataFrame,
    tags: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten MLflow run evidence without changing metric names."""

    run_id_column = first_existing_column(runs, ["run_uuid", "run_id"])
    experiment_id_column = first_existing_column(runs, ["experiment_id"])
    start_column = first_existing_column(runs, ["start_time"])
    end_column = first_existing_column(runs, ["end_time"], required=False)
    status_column = first_existing_column(runs, ["status"], required=False)
    name_column = first_existing_column(
        runs, ["name", "run_name"], required=False
    )
    lifecycle_column = first_existing_column(
        runs, ["lifecycle_stage"], required=False
    )

    selected = pd.DataFrame(
        {
            "run_id": runs[run_id_column].astype(str),
            "experiment_id": runs[experiment_id_column].astype(str),
            "start_time": epoch_to_datetime(runs[start_column]),
        }
    )
    selected["end_time"] = (
        epoch_to_datetime(runs[end_column]) if end_column else pd.NaT
    )
    selected["status"] = (
        runs[status_column].astype(str) if status_column else "UNKNOWN"
    )
    selected["run_name"] = (
        runs[name_column].astype(str)
        if name_column
        else selected["run_id"].str[:8]
    )
    selected["lifecycle_stage"] = (
        runs[lifecycle_column].astype(str) if lifecycle_column else "active"
    )

    exp_id_column = first_existing_column(experiments, ["experiment_id"])
    exp_name_column = first_existing_column(experiments, ["name"])
    experiment_lookup = experiments[
        [exp_id_column, exp_name_column]
    ].copy()
    experiment_lookup.columns = ["experiment_id", "experiment_name"]
    experiment_lookup["experiment_id"] = experiment_lookup[
        "experiment_id"
    ].astype(str)

    selected = selected.merge(
        experiment_lookup,
        on="experiment_id",
        how="left",
        validate="many_to_one",
    )

    for flattened in [
        flatten_key_value_table(metrics, prefix="metric__", value_numeric=True),
        flatten_key_value_table(params, prefix="param__", value_numeric=False),
        flatten_key_value_table(tags, prefix="tag__", value_numeric=False),
    ]:
        selected = selected.merge(
            flattened,
            on="run_id",
            how="left",
            validate="one_to_one",
        )

    mlflow_name = "tag__mlflow.runName"

    if mlflow_name in selected.columns:
        selected["run_name"] = (
            selected[mlflow_name]
            .replace({"nan": np.nan, "None": np.nan})
            .fillna(selected["run_name"])
        )

    return selected.sort_values(
        ["start_time", "run_id"],
        na_position="last",
    ).reset_index(drop=True)


def _normalise_text(value: Any) -> str:
    """Normalise text for strict evidence filtering."""

    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def contains_forbidden_text(value: Any) -> bool:
    """Return whether one value contains prompt/demo evidence."""

    text = _normalise_text(value)

    return any(token in text for token in FORBIDDEN_TEXT_TOKENS)


def extract_registry_run_ids(
    model_versions: pd.DataFrame,
    registered_model_name: str,
) -> set[str]:
    """Return run IDs for the exact saved registered model only."""

    if model_versions.empty or not registered_model_name:
        return set()

    name_column = first_existing_column(model_versions, ["name"])
    run_column = first_existing_column(
        model_versions,
        ["run_id", "run_uuid"],
        required=False,
    )

    if run_column is None:
        return set()

    target = _normalise_text(registered_model_name)
    selected = model_versions.loc[
        model_versions[name_column].astype(str).map(_normalise_text) == target,
        run_column,
    ]

    return {
        str(value)
        for value in selected
        if pd.notna(value) and str(value).strip()
    }


def _row_has_forbidden_evidence(
    row: pd.Series,
    *,
    text_columns: list[str],
    forbidden_metric_columns: list[str],
) -> bool:
    """Detect prompt/demo evidence anywhere in one run."""

    for column in text_columns:
        if column in row.index and contains_forbidden_text(row[column]):
            return True

    for column in forbidden_metric_columns:
        if column in row.index and pd.notna(row[column]):
            return True

    return False


def filter_ecommerce_runs(
    runs: pd.DataFrame,
    *,
    lineage: dict[str, Any],
    registry_run_ids: set[str],
) -> pd.DataFrame:
    """Keep exact champion/registry runs with no demo evidence."""

    if runs.empty:
        return runs.copy()

    champion_run_id = str(lineage.get("run_id", "")).strip()
    allowed_run_ids = set(registry_run_ids)

    if champion_run_id:
        allowed_run_ids.add(champion_run_id)

    if not allowed_run_ids:
        return runs.iloc[0:0].copy()

    candidates = runs.loc[
        runs["run_id"].astype(str).isin(allowed_run_ids)
    ].copy()

    text_columns = [
        "run_name",
        "experiment_name",
        *[
            column
            for column in candidates.columns
            if column.startswith("tag__")
        ],
    ]
    forbidden_metric_columns = [
        column
        for column in candidates.columns
        if column.startswith("metric__") and contains_forbidden_text(column)
    ]

    forbidden_mask = candidates.apply(
        _row_has_forbidden_evidence,
        axis=1,
        text_columns=text_columns,
        forbidden_metric_columns=forbidden_metric_columns,
    )

    return candidates.loc[~forbidden_mask].sort_values(
        ["start_time", "run_name"],
        na_position="last",
    ).reset_index(drop=True)


def is_ecommerce_metric_column(column: str) -> bool:
    """Return whether an exact metric key is model-evaluation evidence."""

    if not column.startswith("metric__") or contains_forbidden_text(column):
        return False

    key = column.removeprefix("metric__").lower()

    return any(token in key for token in ECOMMERCE_METRIC_TOKENS)


def select_metric_columns(
    runs: pd.DataFrame,
    *,
    minimum_runs: int = 2,
    maximum_metrics: int = 5,
) -> list[str]:
    """List exact ecommerce metric keys shared by multiple verified runs."""

    eligible: list[str] = []

    for column in runs.columns:
        if not is_ecommerce_metric_column(column):
            continue

        numeric = pd.to_numeric(runs[column], errors="coerce")

        if (
            int(numeric.notna().sum()) >= minimum_runs
            and int(numeric.nunique(dropna=True)) >= 2
        ):
            eligible.append(column)

    return sorted(eligible)[:maximum_metrics]


def select_primary_metric(
    metric_columns: list[str],
    project_runs: pd.DataFrame | None = None,
) -> str:
    """Return the preferred exact ecommerce metric key."""

    priorities = [
        "holdout_pr_auc",
        "validation_pr_auc",
        "pr_auc",
        "business_score",
        "champion_score",
        "f1",
        "roc_auc",
        "precision",
        "recall",
    ]
    candidates = list(metric_columns)

    if not candidates and project_runs is not None:
        candidates = [
            column
            for column in project_runs.columns
            if is_ecommerce_metric_column(column)
            and pd.to_numeric(
                project_runs[column],
                errors="coerce",
            ).notna().any()
        ]

    for token in priorities:
        for column in candidates:
            if token in column.lower():
                return column

    return candidates[0] if candidates else "metric__pr_auc"


def build_parameter_density(
    comparable_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Return a stable empty-or-populated numeric parameter table."""

    columns = [
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

        if non_null == 0:
            continue

        unique_values = int(numeric.nunique(dropna=True))
        records.append(
            {
                "parameter": column.removeprefix("param__"),
                "non_null_runs": non_null,
                "unique_values": unique_values,
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
                "dense_candidate": (
                    non_null >= 6 and unique_values >= 3
                ),
            }
        )

    return pd.DataFrame(records, columns=columns)


def build_model_version_table(
    model_versions: pd.DataFrame,
    aliases: pd.DataFrame,
    *,
    registered_model_name: str,
) -> pd.DataFrame:
    """Return safe registry metadata for the exact ecommerce model only."""

    columns = [
        "model_name",
        "version",
        "run_id",
        "creation_time",
        "alias",
    ]

    if model_versions.empty or not registered_model_name:
        return pd.DataFrame(columns=columns)

    name_column = first_existing_column(model_versions, ["name"])
    version_column = first_existing_column(model_versions, ["version"])
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

    target = _normalise_text(registered_model_name)

    if contains_forbidden_text(target):
        return pd.DataFrame(columns=columns)

    filtered = model_versions.loc[
        model_versions[name_column].astype(str).map(_normalise_text) == target
    ].copy()

    versions = pd.DataFrame(
        {
            "model_name": filtered[name_column].astype(str),
            "version": pd.to_numeric(
                filtered[version_column],
                errors="coerce",
            ),
            "run_id": (
                filtered[run_column].astype(str)
                if run_column
                else ""
            ),
            "creation_time": (
                epoch_to_datetime(filtered[creation_column])
                if creation_column
                else pd.NaT
            ),
        }
    )

    alias_output = pd.DataFrame(
        columns=["model_name", "version", "alias"]
    )

    if not aliases.empty:
        alias_name = first_existing_column(aliases, ["name"])
        alias_version = first_existing_column(aliases, ["version"])
        alias_value = first_existing_column(aliases, ["alias"])
        alias_filtered = aliases.loc[
            aliases[alias_name].astype(str).map(_normalise_text) == target,
            [alias_name, alias_version, alias_value],
        ].copy()
        alias_filtered.columns = [
            "model_name",
            "version",
            "alias",
        ]
        alias_filtered["version"] = pd.to_numeric(
            alias_filtered["version"],
            errors="coerce",
        )
        alias_output = (
            alias_filtered
            .groupby(["model_name", "version"])["alias"]
            .apply(
                lambda values: ", ".join(
                    sorted(
                        {
                            str(value)
                            for value in values
                            if pd.notna(value)
                            and not contains_forbidden_text(value)
                        }
                    )
                )
            )
            .reset_index()
        )

    versions = versions.merge(
        alias_output,
        on=["model_name", "version"],
        how="left",
        validate="one_to_one",
    )
    versions["alias"] = versions["alias"].fillna("")

    return versions[columns].sort_values(
        ["version", "creation_time"],
        na_position="last",
    ).reset_index(drop=True)


def metric_display_name(metric_column: str) -> str:
    """Convert an exact metric key into readable display copy."""

    key = metric_column.removeprefix("metric__")

    return key.replace("_", " ").title()


def build_experiment_tracking_bundle(
    project_root: str | Path = ".",
) -> ExperimentTrackingBundle:
    """Load MLflow, audit safe evidence, and enforce conditional status."""

    root = Path(project_root)
    database_file = root / MLFLOW_DB_PATH
    lineage_file = root / LINEAGE_PATH

    if not database_file.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {database_file}"
        )

    lineage = read_json(lineage_file)

    with sqlite3.connect(database_file) as connection:
        experiments = read_table(connection, "experiments")
        runs_source = read_table(connection, "runs")
        metrics = read_table(connection, "metrics")
        params = read_table(connection, "params")
        tags = read_table(connection, "tags")
        model_versions = read_table(connection, "model_versions")
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
    registered_model_name = str(
        lineage.get(
            "registered_model_name",
            "",
        )
    ).strip()
    registry_run_ids = extract_registry_run_ids(
        model_versions,
        registered_model_name,
    )
    project_runs = filter_ecommerce_runs(
        runs,
        lineage=lineage,
        registry_run_ids=registry_run_ids,
    )
    metric_columns = select_metric_columns(project_runs)
    primary_metric = select_primary_metric(
        metric_columns,
        project_runs,
    )
    versions = build_model_version_table(
        model_versions,
        aliases,
        registered_model_name=registered_model_name,
    )

    comparison_reason = (
        "The current MLflow database does not provide a fully verified, "
        "comparable ecommerce experiment set after prompt/demo evidence is "
        "excluded. This visual remains conditional until clean ecommerce-only "
        "runs are logged under a dedicated evaluation contract."
    )
    comparable_runs = project_runs.iloc[0:0].copy()
    parameter_density = build_parameter_density(comparable_runs)

    counts = {
        "runs_total": len(runs),
        "verified_ecommerce_runs": len(project_runs),
        "excluded_runs": len(runs) - len(project_runs),
        "verified_registry_versions": len(versions),
        "shared_metric_keys": len(metric_columns),
    }

    return ExperimentTrackingBundle(
        runs=runs,
        project_runs=project_runs,
        comparable_runs=comparable_runs,
        metric_columns=metric_columns,
        primary_metric=primary_metric,
        model_versions=versions,
        experiments=experiments,
        lineage=lineage,
        counts=counts,
        parameter_density=parameter_density,
        comparison_ready=False,
        comparison_reason=comparison_reason,
        i03_ready=False,
    )
