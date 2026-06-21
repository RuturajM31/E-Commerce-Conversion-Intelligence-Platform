"""Extract compact drift and alert results from Evidently reports.

This module contains pure Python logic. It does not import Evidently.

Inputs:
    Dictionaries returned by ``EvidentlyResult.dict()``.

Outputs:
    Compact feature drift, prediction drift, and alert information.

Used next:
    ``evidently_drift.py`` will place these results inside the monitoring
    summary consumed later by Grafana, Streamlit, and Alertmanager.
"""

from __future__ import annotations

from numbers import Real
from typing import Any


def extract_column_psi(
    report: dict[str, Any],
    expected_columns: list[str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Extract one PSI value and drift flag for every expected column.

    Args:
        report:
            Dictionary returned by an Evidently report result.

        expected_columns:
            Feature or score columns that must exist in the report.

        threshold:
            PSI value at or above which drift is detected.

    Returns:
        A dictionary keyed by column name.

        Each column contains:
        - PSI value
        - configured threshold
        - drift status

    Raises:
        ValueError:
            Raised when a required column has no PSI metric.
            Missing metrics must not silently disappear.
    """

    # Convert the required column names to a set for fast lookup.
    expected_set = set(expected_columns)

    # This dictionary will receive valid PSI results from the report.
    extracted: dict[str, dict[str, Any]] = {}

    # Evidently stores every result as one object inside "metrics".
    for metric in report.get("metrics", []):
        if not isinstance(metric, dict):
            continue

        # The metric configuration identifies the column and method.
        config = metric.get("config", {})

        if not isinstance(config, dict):
            continue

        column = config.get("column")
        method = str(
            config.get("method", "")
        ).lower()

        value = metric.get("value")

        # A column-level PSI metric has:
        # - one required column,
        # - method="psi",
        # - one numeric result value.
        if (
            column in expected_set
            and method == "psi"
            and isinstance(value, Real)
            and not isinstance(value, bool)
        ):
            psi_value = float(value)

            extracted[str(column)] = {
                "psi": psi_value,
                "threshold": float(threshold),
                "drift_detected": (
                    psi_value >= float(threshold)
                ),
            }

    # Every production feature must have a corresponding PSI result.
    missing_columns = [
        column
        for column in expected_columns
        if column not in extracted
    ]

    if missing_columns:
        raise ValueError(
            "PSI metrics were not found for columns: "
            f"{missing_columns}"
        )

    # Preserve the canonical model feature order in the JSON output.
    return {
        column: extracted[column]
        for column in expected_columns
    }


def summarize_feature_drift(
    feature_metrics: dict[str, dict[str, Any]],
    dataset_drift_share: float,
) -> dict[str, Any]:
    """Convert feature PSI results into dataset-level drift results.

    Args:
        feature_metrics:
            Per-feature PSI values produced by ``extract_column_psi``.

        dataset_drift_share:
            Minimum share of drifted features required to declare
            dataset-level drift.

    Returns:
        Counts, percentages, drifted feature names, and full results.
    """

    # Keep only features whose PSI reached the configured threshold.
    drifted_columns = [
        column
        for column, result in feature_metrics.items()
        if bool(result["drift_detected"])
    ]

    feature_count = len(feature_metrics)
    drifted_count = len(drifted_columns)

    # Avoid division by zero if no feature configuration is provided.
    drifted_share = (
        drifted_count / feature_count
        if feature_count
        else 0.0
    )

    return {
        "feature_count": feature_count,
        "drifted_feature_count": drifted_count,
        "drifted_feature_share": drifted_share,
        "dataset_drift_threshold": float(
            dataset_drift_share
        ),
        "dataset_drift_detected": (
            drifted_share
            >= float(dataset_drift_share)
        ),
        "drifted_features": drifted_columns,
        "features": feature_metrics,
    }


def build_alert_summary(
    feature_summary: dict[str, Any],
    prediction_metric: dict[str, Any],
) -> dict[str, Any]:
    """Create an operational alert level and supporting reasons.

    Alert rules:
        critical:
            Dataset-level feature drift is detected.

        warning:
            At least one feature drifts, or prediction-score drift occurs,
            but dataset-level feature drift is not reached.

        healthy:
            No feature or prediction-score drift is detected.
    """

    reasons: list[str] = []

    dataset_drift = bool(
        feature_summary[
            "dataset_drift_detected"
        ]
    )

    drifted_feature_count = int(
        feature_summary[
            "drifted_feature_count"
        ]
    )

    prediction_drift = bool(
        prediction_metric[
            "drift_detected"
        ]
    )

    if dataset_drift:
        alert_level = "critical"

        reasons.append(
            "Dataset-level feature drift exceeded "
            "the configured feature-share threshold."
        )

    elif (
        drifted_feature_count > 0
        or prediction_drift
    ):
        alert_level = "warning"

        if drifted_feature_count > 0:
            reasons.append(
                f"{drifted_feature_count} production feature(s) "
                "exceeded the PSI threshold."
            )

        if prediction_drift:
            reasons.append(
                "The purchase-intent score distribution "
                "exceeded the PSI threshold."
            )

    else:
        alert_level = "healthy"

        reasons.append(
            "No feature or prediction-score drift exceeded "
            "the configured PSI threshold."
        )

    return {
        "level": alert_level,
        "requires_attention": (
            alert_level != "healthy"
        ),
        "reasons": reasons,
    }


def build_drift_monitoring_summary(
    feature_report: dict[str, Any],
    prediction_report: dict[str, Any],
    feature_columns: list[str],
    score_column: str,
    column_threshold: float,
    dataset_drift_share: float,
) -> dict[str, Any]:
    """Build the complete compact drift-monitoring result.

    Inputs:
        Feature and prediction Evidently report dictionaries.

    Output:
        One dictionary containing:
        - feature PSI results,
        - prediction PSI result,
        - dataset drift status,
        - operational alert level.

    Used next:
        This dictionary will be stored under ``drift_results`` inside
        ``evidently_monitoring_summary.json``.
    """

    # Extract PSI for all seven canonical production features.
    feature_metrics = extract_column_psi(
        report=feature_report,
        expected_columns=feature_columns,
        threshold=column_threshold,
    )

    # Convert the feature results into dataset-level measurements.
    feature_summary = summarize_feature_drift(
        feature_metrics=feature_metrics,
        dataset_drift_share=dataset_drift_share,
    )

    # Extract PSI for the champion purchase-intent score.
    prediction_metrics = extract_column_psi(
        report=prediction_report,
        expected_columns=[score_column],
        threshold=column_threshold,
    )

    prediction_metric = (
        prediction_metrics[score_column]
    )

    # Include the score column name so downstream tools understand it.
    prediction_summary = {
        "column": score_column,
        **prediction_metric,
    }

    # Translate the technical results into an operational alert level.
    alert = build_alert_summary(
        feature_summary=feature_summary,
        prediction_metric=prediction_summary,
    )

    return {
        "feature_drift": feature_summary,
        "prediction_drift": prediction_summary,
        "alert": alert,
    }
