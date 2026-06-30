"""Executive Overview data, calculations, and chart builders.

Purpose
-------
Keep the Streamlit page focused on product layout while this module owns:

- evidence loading;
- scenario calculations;
- funnel and efficiency logic;
- composition, forecast, anomaly, and readiness evidence;
- Plotly figure construction;
- exportable executive summary creation.

All loaders use explicit source paths and honest fallback states. No metric is
invented when a source is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import json
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.ui.plotly_style import apply_product_layout


def _finish_figure(
    figure: go.Figure,
    *,
    title: str,
    subtitle: str,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply the existing shared Plotly API and add the chart heading."""

    # Preserve the height selected by the individual chart builder.
    current_height = int(figure.layout.height or 580)

    apply_product_layout(
        figure,
        height=current_height,
        legend_title=legend_title,
    )

    # Package 1's shared layout function does not accept title/subtitle
    # parameters, so Package 2 adds them through Plotly's supported layout API.
    figure.update_layout(
        title={
            "text": (
                f"<b>{title}</b><br>"
                f"<span style='font-size:12px'>{subtitle}</span>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        margin={
            "l": 52,
            "r": 34,
            "t": 96,
            "b": 56,
        },
    )

    return figure


TABLE_DIR = Path("reports/tables")
METADATA_DIR = Path("reports/metadata")

HOLDOUT_PATH = TABLE_DIR / "final_true_champion_holdout.csv"
PROJECTION_PATH = TABLE_DIR / "ml_validation_projection_sample.csv"
MANIFEST_PATH = METADATA_DIR / "ml_validation_manifest.json"

FORECAST_CANDIDATES = (
    # This is the approved business-forecast output currently produced by the
    # repository. Keep it first so the Executive Overview uses the same
    # evidence as the dedicated forecasting page.
    TABLE_DIR / "business_forecast_future.csv",
    TABLE_DIR / "forecast_summary.csv",
    TABLE_DIR / "business_kpi_forecast.csv",
    TABLE_DIR / "kpi_forecast_summary.csv",
    TABLE_DIR / "forecast_results.csv",
)

SCORE_CANDIDATES = (
    # The cached score file is approved generated evidence. Reading this CSV
    # does not load or execute the production model bundle.
    Path("data/processed/final_champion_visitor_scores.csv"),
)

MONITORING_CANDIDATES = {
    "Application": (
        Path("app/Executive_Overview.py"),
        Path("app/pages/1_Visitor_Intent_Predictor.py"),
    ),
    "Scoring": (
        TABLE_DIR / "final_true_champion_holdout.csv",
        Path("models"),
    ),
    "Drift": (
        Path("reports/evidently"),
        Path("monitoring"),
    ),
    "Alerts": (
        Path("monitoring/alertmanager"),
        Path("monitoring/prometheus"),
    ),
    "Labels": (
        # Delayed-label readiness is implemented in source code, exercised by
        # focused tests, and materialised as monitoring evidence. Any one of
        # these artifacts is sufficient to prove that the capability exists.
        Path("src/monitoring/delayed_labels.py"),
        Path("src/monitoring/run_delayed_label_evaluation.py"),
        Path("tests/test_delayed_labels.py"),
        Path("tests/test_delayed_label_evaluation_runner.py"),
        Path(
            "reports/visuals/ml_visual_intelligence/monitoring/"
            "delayed_label_funnel.csv"
        ),
        Path(
            "reports/visuals/ml_visual_intelligence/monitoring/"
            "delayed_label_rejections.csv"
        ),
        Path("docs/DELAYED_LABEL_MONITORING_RUNBOOK.md"),
        Path("docs/monitoring"),
    ),
}


@dataclass(frozen=True)
class ExecutiveEvidence:
    """Validated evidence used by the Executive Overview."""

    holdout: pd.DataFrame
    projection: pd.DataFrame
    scores: pd.DataFrame
    score_source: str | None
    manifest: dict[str, Any]
    forecast: pd.DataFrame
    forecast_source: str | None
    source_timestamp: str


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV when available; otherwise return an empty table."""

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object when available; otherwise return an empty dict."""

    if not path.is_file():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    """Return the first existing file from a candidate list."""

    for path in paths:
        if path.is_file():
            return path
    return None


def load_executive_evidence() -> ExecutiveEvidence:
    """Load all Executive Overview evidence from approved project outputs."""

    forecast_path = _first_existing(FORECAST_CANDIDATES)
    score_path = _first_existing(SCORE_CANDIDATES)

    # The displayed refresh time should represent every source that can change
    # what the page shows, not only the three original Package 1 artifacts.
    timestamp_paths = [HOLDOUT_PATH, PROJECTION_PATH, MANIFEST_PATH]
    timestamp_paths.extend(
        path for path in (forecast_path, score_path) if path is not None
    )
    source_times = [
        path.stat().st_mtime
        for path in timestamp_paths
        if path.exists()
    ]

    source_timestamp = "Unavailable"
    if source_times:
        latest = max(source_times)
        source_timestamp = datetime.fromtimestamp(
            latest,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC")

    return ExecutiveEvidence(
        holdout=_read_csv(HOLDOUT_PATH),
        projection=_read_csv(PROJECTION_PATH),
        scores=_read_csv(score_path) if score_path else pd.DataFrame(),
        score_source=str(score_path) if score_path else None,
        manifest=_read_json(MANIFEST_PATH),
        forecast=_read_csv(forecast_path) if forecast_path else pd.DataFrame(),
        forecast_source=str(forecast_path) if forecast_path else None,
        source_timestamp=source_timestamp,
    )


def controlling_holdout_row(
    holdout: pd.DataFrame,
    fallback: dict[str, float | str],
) -> dict[str, float | str]:
    """Return the single controlling holdout row or an explicit fallback."""

    if holdout.empty:
        return dict(fallback)

    row = holdout.iloc[0].to_dict()
    return {
        "model_name": str(row.get("model_name", fallback["model_name"])),
        "threshold": float(row.get("threshold", fallback["threshold"])),
        "rows": int(row.get("rows", fallback["rows"])),
        "positive_rows": int(
            row.get("positive_rows", fallback["positive_rows"])
        ),
        "positive_rate": float(
            row.get("positive_rate", fallback["positive_rate"])
        ),
        "predicted_positive_rows": int(
            row.get(
                "predicted_positive_rows",
                fallback["predicted_positive_rows"],
            )
        ),
        "predicted_positive_rate": float(
            row.get(
                "predicted_positive_rate",
                fallback["predicted_positive_rate"],
            )
        ),
        "pr_auc": float(row.get("pr_auc", fallback["pr_auc"])),
        "roc_auc": float(row.get("roc_auc", fallback["roc_auc"])),
        "precision": float(row.get("precision", fallback["precision"])),
        "recall": float(row.get("recall", fallback["recall"])),
        "f1_score": float(row.get("f1_score", fallback["f1_score"])),
    }


def source_visitor_count(manifest: dict[str, Any], holdout_rows: int) -> int:
    """Resolve the source visitor count without inventing a value."""

    candidates = [
        manifest.get("source_rows"),
        manifest.get("SOURCE_ROWS"),
        manifest.get("input_rows"),
        holdout_rows,
    ]

    for value in candidates:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue

        if number > 0:
            return number

    return max(int(holdout_rows), 0)


def calculate_scenario(
    *,
    available_targets: int,
    target_size: int,
    precision: float,
    baseline_rate: float,
    contact_cost: float,
    buyer_value: float,
) -> dict[str, float]:
    """Calculate one campaign scenario from validated model assumptions."""

    available_targets = max(int(available_targets), 0)
    target_size = min(max(int(target_size), 0), available_targets)

    precision = min(max(float(precision), 0.0), 1.0)
    baseline_rate = min(max(float(baseline_rate), 0.0), 1.0)
    contact_cost = max(float(contact_cost), 0.0)
    buyer_value = max(float(buyer_value), 0.0)

    expected_buyers = target_size * precision
    baseline_buyers = target_size * baseline_rate
    incremental_buyers = max(expected_buyers - baseline_buyers, 0.0)
    campaign_cost = target_size * contact_cost
    expected_value = expected_buyers * buyer_value
    baseline_value = baseline_buyers * buyer_value
    incremental_value = incremental_buyers * buyer_value
    net_value = expected_value - campaign_cost
    roi = (
        (expected_value - campaign_cost) / campaign_cost
        if campaign_cost > 0
        else 0.0
    )

    return {
        "target_size": float(target_size),
        "expected_buyers": expected_buyers,
        "baseline_buyers": baseline_buyers,
        "incremental_buyers": incremental_buyers,
        "campaign_cost": campaign_cost,
        "expected_value": expected_value,
        "baseline_value": baseline_value,
        "incremental_value": incremental_value,
        "net_value": net_value,
        "roi": roi,
    }


def build_funnel_table(
    *,
    source_visitors: int,
    holdout_rows: int,
    threshold_eligible: int,
    target_size: int,
    expected_buyers: float,
) -> pd.DataFrame:
    """Create the tested five-stage executive funnel."""

    stages = [
        ("Source visitors", max(int(source_visitors), 0)),
        ("Validated holdout", max(int(holdout_rows), 0)),
        ("Threshold eligible", max(int(threshold_eligible), 0)),
        ("Campaign target", max(int(target_size), 0)),
        ("Expected buyers", max(float(expected_buyers), 0.0)),
    ]

    frame = pd.DataFrame(stages, columns=["Stage", "Visitors"])

    previous = None
    retention = []
    for value in frame["Visitors"]:
        if previous is None or previous == 0:
            retention.append(1.0)
        else:
            retention.append(min(float(value) / float(previous), 1.0))
        previous = float(value)

    frame["Stage Retention"] = retention
    return frame


def build_funnel_figure(frame: pd.DataFrame) -> go.Figure:
    """Build the Executive Overview targeting funnel."""

    figure = go.Figure(
        go.Funnel(
            y=frame["Stage"],
            x=frame["Visitors"],
            textinfo="value+percent initial",
            marker={
                "line": {"width": 1, "color": "rgba(255,255,255,0.18)"}
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Count: %{x:,.1f}<br>"
                "Share of source: %{percentInitial:.2%}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(height=540)
    return _finish_figure(
        figure,
        title="Visitor-to-buyer decision funnel",
        subtitle=(
            "Validated evidence narrows broad traffic into an actionable "
            "campaign audience."
        ),
    )


def build_efficiency_figure(
    *,
    source_visitors: int,
    target_size: int,
    expected_buyers: float,
    baseline_rate: float,
) -> go.Figure:
    """Compare audience share, buyer yield, and random expectation."""

    audience_share = (
        target_size / source_visitors if source_visitors > 0 else 0.0
    )
    model_yield = (
        expected_buyers / target_size if target_size > 0 else 0.0
    )

    frame = pd.DataFrame(
        {
            "Metric": [
                "Audience selected",
                "Expected buyer yield",
                "Random buyer yield",
            ],
            "Rate": [
                audience_share * 100,
                model_yield * 100,
                baseline_rate * 100,
            ],
        }
    )

    figure = px.bar(
        frame,
        x="Metric",
        y="Rate",
        text="Rate",
        labels={"Rate": "Percent"},
    )
    figure.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
    )
    figure.update_layout(height=440, showlegend=False)
    return _finish_figure(
        figure,
        title="Audience reduction and campaign efficiency",
        subtitle=(
            "The selected audience is intentionally small while expected "
            "buyer concentration is materially higher."
        ),
    )


INTENT_TIER_THRESHOLDS = (
    (0.90, "High Intent"),
    (0.70, "Strong Intent"),
    (0.50, "Warm Intent"),
    (0.20, "Low Intent"),
)

COMPOSITION_DEFINITIONS = {
    "Segment": (
        ["kmeans_cluster", "cluster", "segment"],
        "Segment not covered by projection evidence",
    ),
    "Intent tier": (
        ["intent_tier", "intent_segment", "purchase_intent_segment"],
        "Intent tier unavailable",
    ),
    "Anomaly status": (
        ["lof_status", "anomaly_status", "is_anomaly", "lof_outlier"],
        "Anomaly status not covered by projection evidence",
    ),
}


def _intent_tier_from_score(score: float) -> str:
    """Convert one approved purchase-intent score into a business tier.

    The thresholds match ``src/segmentation/create_visitor_segments.py`` so
    the Executive Overview and the detailed segmentation workflow use the
    same business meaning.
    """

    for minimum_score, tier in INTENT_TIER_THRESHOLDS:
        if score >= minimum_score:
            return tier
    return "Cold Visitor"


def build_selected_campaign_audience(
    projection: pd.DataFrame,
    scores: pd.DataFrame,
    target_size: int,
) -> tuple[pd.DataFrame, str]:
    """Build the scored audience used by the composition visual.

    Inputs
    ------
    projection:
        Governed Package 1 sample containing segment and anomaly attributes.
    scores:
        Cached production score evidence containing visitor IDs, purchase
        intent scores, and the threshold decision.
    target_size:
        Number of highest-scored threshold-eligible visitors requested by the
        campaign scenario.

    Output
    ------
    The function returns the selected score rows, optionally enriched with
    projection attributes through ``visitorid``. A left join preserves every
    selected visitor. Missing segment or anomaly values remain explicit rather
    than silently replacing the audience with the full 8,000-row projection.
    """

    requested_size = max(int(target_size), 0)
    if requested_size == 0:
        return pd.DataFrame(), "Campaign target size is zero."

    if scores.empty:
        return (
            pd.DataFrame(),
            "Approved cached visitor-score evidence is unavailable.",
        )

    visitor_column = next(
        (
            column
            for column in ("visitorid", "visitor_id", "visitorId")
            if column in scores.columns
        ),
        None,
    )
    score_column = next(
        (
            column
            for column in (
                "purchase_intent_score",
                "conversion_probability",
                "score",
                "probability",
            )
            if column in scores.columns
        ),
        None,
    )

    if visitor_column is None or score_column is None:
        return (
            pd.DataFrame(),
            "Visitor-score evidence exists but required ID or score columns "
            "are missing.",
        )

    selected = scores.copy()
    selected[score_column] = pd.to_numeric(
        selected[score_column],
        errors="coerce",
    )
    selected = selected.dropna(subset=[visitor_column, score_column])

    # Use the saved threshold decision when it exists. If only the production
    # threshold is present, reconstruct the same decision from the saved score.
    if "predicted_conversion" in selected.columns:
        decision = pd.to_numeric(
            selected["predicted_conversion"],
            errors="coerce",
        ).fillna(0).astype(int)
        selected = selected.loc[decision.eq(1)].copy()
    elif "production_threshold" in selected.columns:
        threshold = pd.to_numeric(
            selected["production_threshold"],
            errors="coerce",
        )
        selected = selected.loc[selected[score_column].ge(threshold)].copy()
    else:
        return (
            pd.DataFrame(),
            "Visitor-score evidence has no threshold decision, so a campaign "
            "audience cannot be selected honestly.",
        )

    if selected.empty:
        return (
            pd.DataFrame(),
            "No threshold-eligible visitors exist in the cached score evidence.",
        )

    # Stable sorting makes repeated renders and tests deterministic when two
    # visitors have the same score. Duplicate IDs are removed before selection.
    selected[visitor_column] = (
        selected[visitor_column].astype("string").str.strip()
    )
    selected = (
        selected.sort_values(
            [score_column, visitor_column],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=[visitor_column], keep="first")
        .head(requested_size)
        .copy()
    )
    selected["intent_tier"] = selected[score_column].map(
        _intent_tier_from_score
    )

    projection_coverage = 0
    projection_visitor_column = next(
        (
            column
            for column in ("visitorid", "visitor_id", "visitorId")
            if column in projection.columns
        ),
        None,
    )

    if projection_visitor_column is not None and not projection.empty:
        attributes = projection.copy()
        attributes[projection_visitor_column] = (
            attributes[projection_visitor_column]
            .astype("string")
            .str.strip()
        )
        attributes = attributes.drop_duplicates(
            subset=[projection_visitor_column],
            keep="first",
        )

        if projection_visitor_column != visitor_column:
            attributes = attributes.rename(
                columns={projection_visitor_column: visitor_column}
            )

        # Score columns remain controlling when a name appears in both tables.
        # Projection-only fields such as cluster and LOF status are added.
        projection_fields = [
            column
            for column in attributes.columns
            if column == visitor_column or column not in selected.columns
        ]
        selected = selected.merge(
            attributes[projection_fields],
            on=visitor_column,
            how="left",
            validate="one_to_one",
        )

        coverage_columns = [
            column
            for column in (
                "kmeans_cluster",
                "cluster",
                "segment",
                "lof_status",
                "anomaly_status",
                "is_anomaly",
                "lof_outlier",
            )
            if column in selected.columns
        ]
        if coverage_columns:
            projection_coverage = int(
                selected[coverage_columns].notna().any(axis=1).sum()
            )

    message = (
        f"{len(selected):,} of {requested_size:,} requested visitors were "
        "selected from threshold-eligible score evidence. "
        f"{projection_coverage:,} selected visitors also have governed "
        "segment or anomaly attributes."
    )
    return selected, message


def available_composition_views(frame: pd.DataFrame) -> list[str]:
    """Return only composition views supported by the selected audience."""

    if frame.empty:
        return []

    views = []
    for view, (candidates, _) in COMPOSITION_DEFINITIONS.items():
        column = next(
            (candidate for candidate in candidates if candidate in frame),
            None,
        )
        if column is not None and frame[column].notna().any():
            views.append(view)
    return views


def _categorical_count(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    fallback_label: str,
) -> pd.DataFrame:
    """Build a category count table from the first available column."""

    column = next(
        (candidate for candidate in candidates if candidate in frame.columns),
        None,
    )

    if column is None or frame.empty:
        return pd.DataFrame(
            {"Category": [fallback_label], "Visitors": [0]}
        )

    counts = (
        frame[column]
        .fillna(fallback_label)
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Category")
        .reset_index(name="Visitors")
    )
    return counts


def build_composition_figure(
    selected_audience: pd.DataFrame,
    view: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build composition from the selected scored audience only."""

    if view not in COMPOSITION_DEFINITIONS:
        raise ValueError(f"Unsupported composition view: {view}")

    candidates, fallback = COMPOSITION_DEFINITIONS[view]
    counts = _categorical_count(selected_audience, candidates, fallback)

    figure = px.bar(
        counts.sort_values("Visitors", ascending=True),
        x="Visitors",
        y="Category",
        orientation="h",
        text="Visitors",
    )
    figure.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
    )
    figure.update_layout(height=max(390, 56 * len(counts)))
    return (
        _finish_figure(
            figure,
            title=f"Selected audience by {view.lower()}",
            subtitle=(
                "Composition is calculated only from the current "
                "threshold-selected campaign audience."
            ),
        ),
        counts,
    )


def anomaly_summary(projection: pd.DataFrame) -> dict[str, Any]:
    """Summarise LOF anomaly volume and severity from current evidence."""

    if projection.empty or "lof_outlier" not in projection.columns:
        return {
            "count": 0,
            "rate": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "key_segment": "Unavailable",
        }

    flags = pd.to_numeric(
        projection["lof_outlier"],
        errors="coerce",
    ).fillna(0).astype(bool)

    scores = pd.to_numeric(
        projection.get("lof_score", pd.Series(index=projection.index)),
        errors="coerce",
    )

    flagged = projection.loc[flags].copy()
    flagged_scores = scores.loc[flags].dropna()

    segment_column = next(
        (
            column
            for column in ("kmeans_cluster", "cluster", "segment")
            if column in flagged.columns
        ),
        None,
    )

    key_segment = "Unavailable"
    if segment_column and not flagged.empty:
        key_segment = str(
            flagged[segment_column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .index[0]
        )

    return {
        "count": int(flags.sum()),
        "rate": float(flags.mean()) if len(flags) else 0.0,
        "median_score": (
            float(flagged_scores.median())
            if not flagged_scores.empty
            else 0.0
        ),
        "max_score": (
            float(flagged_scores.max())
            if not flagged_scores.empty
            else 0.0
        ),
        "key_segment": key_segment,
    }


def build_anomaly_figure(projection: pd.DataFrame) -> go.Figure | None:
    """Build a compact executive anomaly-severity view."""

    if projection.empty or "lof_score" not in projection.columns:
        return None

    frame = projection.copy()
    frame["LOF score"] = pd.to_numeric(
        frame["lof_score"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["LOF score"])

    if frame.empty:
        return None

    cap = frame["LOF score"].quantile(0.99)
    frame["Display score"] = frame["LOF score"].clip(upper=cap)
    frame["Status"] = (
        pd.to_numeric(
            frame.get("lof_outlier", 0),
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
        .map({True: "Flagged", False: "Normal"})
    )

    figure = px.histogram(
        frame,
        x="Display score",
        color="Status",
        nbins=45,
        barmode="overlay",
        opacity=0.82,
    )
    figure.update_layout(height=420)
    return _finish_figure(
        figure,
        title="Current anomaly-severity distribution",
        subtitle=(
            "The display is capped at the 99th percentile to preserve shape; "
            "the uncapped maximum remains in the evidence summary."
        ),
    )


def build_forecast_figure(
    forecast: pd.DataFrame,
) -> tuple[go.Figure | None, str]:
    """Build a forward KPI outlook from the first supported forecast schema.

    The repository's approved business forecast contains several KPI and model
    combinations in one file. The Executive Overview deliberately chooses one
    business-facing KPI and its row marked ``is_best_model`` so unrelated
    series are never connected into one misleading line.
    """

    if forecast.empty:
        return None, "No approved forecast summary table is available."

    date_column = next(
        (
            column
            for column in (
                "date",
                "ds",
                "period",
                "forecast_date",
                "timestamp",
            )
            if column in forecast.columns
        ),
        None,
    )
    value_column = next(
        (
            column
            for column in (
                "forecast",
                "yhat",
                "predicted_value",
                "conversion_forecast",
                "expected_conversions",
            )
            if column in forecast.columns
        ),
        None,
    )

    if date_column is None or value_column is None:
        return None, "Forecast table exists but its schema is unsupported."

    frame = forecast.copy()
    selected_target = None
    selected_model = None

    # Prefer the conversion-count forecast because it is closest to the buyer
    # decision made on this page. The remaining priorities provide honest
    # fallbacks when a repository uses a smaller forecast scope.
    if "target_name" in frame.columns:
        available_targets = set(
            frame["target_name"].dropna().astype(str).tolist()
        )
        target_priority = (
            "converted_visitor_count",
            "high_intent_visitor_count",
            "unique_visitors",
            "event_volume",
        )
        selected_target = next(
            (target for target in target_priority if target in available_targets),
            None,
        )
        if selected_target is None and available_targets:
            selected_target = sorted(available_targets)[0]
        if selected_target is not None:
            frame = frame.loc[
                frame["target_name"].astype(str).eq(selected_target)
            ].copy()

    if "is_best_model" in frame.columns:
        raw_best = frame["is_best_model"]
        best_mask = (
            raw_best.eq(True)
            | raw_best.astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes"}
            )
        )
        if best_mask.any():
            frame = frame.loc[best_mask].copy()

    if "model_name" in frame.columns:
        models = frame["model_name"].dropna().astype(str).unique().tolist()
        if models:
            selected_model = models[0]
            # A malformed source could mark more than one model as best. Keep
            # one deterministic model rather than drawing a connected mixture.
            frame = frame.loc[
                frame["model_name"].astype(str).eq(selected_model)
            ].copy()

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )
    frame[value_column] = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )
    frame = frame.dropna(subset=[date_column, value_column]).sort_values(
        date_column
    )

    if frame.empty:
        return None, "Forecast table contains no usable dated values."

    target_label = (
        selected_target.replace("_", " ").title()
        if selected_target
        else "Forecast"
    )
    trace_name = (
        f"{target_label} · {selected_model}"
        if selected_model
        else target_label
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame[date_column],
            y=frame[value_column],
            mode="lines+markers",
            name=trace_name,
        )
    )

    lower_column = next(
        (
            column
            for column in ("lower", "yhat_lower", "lower_bound")
            if column in frame.columns
        ),
        None,
    )
    upper_column = next(
        (
            column
            for column in ("upper", "yhat_upper", "upper_bound")
            if column in frame.columns
        ),
        None,
    )

    if lower_column and upper_column:
        lower = pd.to_numeric(frame[lower_column], errors="coerce")
        upper = pd.to_numeric(frame[upper_column], errors="coerce")

        figure.add_trace(
            go.Scatter(
                x=list(frame[date_column]) + list(frame[date_column][::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(56,189,248,0.14)",
                line={"color": "rgba(255,255,255,0)"},
                name="Uncertainty",
                hoverinfo="skip",
            )
        )

    figure.update_layout(height=430)
    source_detail = target_label
    if selected_model:
        source_detail = f"{source_detail} using {selected_model}"

    return (
        _finish_figure(
            figure,
            title="Forward KPI outlook",
            subtitle=(
                "The approved best-model forecast is shown with uncertainty "
                "when interval columns are available."
            ),
        ),
        f"{len(frame):,} forecast periods loaded for {source_detail}.",
    )


def readiness_table() -> pd.DataFrame:
    """Create a compact operational readiness strip from real artifacts."""

    rows = []

    for area, candidates in MONITORING_CANDIDATES.items():
        available = any(path.exists() for path in candidates)
        rows.append(
            {
                "Area": area,
                "Status": "Ready" if available else "Evidence unavailable",
                "Evidence": next(
                    (
                        str(path)
                        for path in candidates
                        if path.exists()
                    ),
                    "No matching artifact",
                ),
            }
        )

    return pd.DataFrame(rows)


def build_readiness_figure(frame: pd.DataFrame) -> go.Figure:
    """Build an executive readiness strip."""

    chart = frame.copy()
    chart["Ready value"] = chart["Status"].eq("Ready").astype(int)

    figure = px.bar(
        chart,
        x="Area",
        y="Ready value",
        color="Status",
        text="Status",
        custom_data=["Evidence"],
    )
    figure.update_traces(
        textposition="inside",
        hovertemplate=(
            "<b>%{x}</b><br>Status: %{text}<br>"
            "Evidence: %{customdata[0]}<extra></extra>"
        ),
    )
    figure.update_yaxes(
        range=[0, 1.08],
        tickvals=[0, 1],
        ticktext=["Unavailable", "Ready"],
    )
    figure.update_layout(height=350, showlegend=False)
    return _finish_figure(
        figure,
        title="Production-readiness strip",
        subtitle=(
            "Application, scoring, drift, alerting, and delayed-label "
            "readiness are grounded in existing project artifacts."
        ),
    )


def decision_summary(
    *,
    target_size: int,
    expected_buyers: float,
    incremental_value: float,
    roi: float,
    anomaly_rate: float,
    forecast_available: bool,
) -> dict[str, str]:
    """Generate a concise finding, action, and limitation."""

    finding = (
        f"Targeting {target_size:,} validated high-intent visitors is "
        f"expected to reach {expected_buyers:.1f} buyers."
    )

    action = (
        f"Prioritise the filtered audience while monitoring a "
        f"{anomaly_rate:.1%} anomaly rate and an estimated ROI of {roi:.1%}."
    )

    limitation = (
        "Scenario value is assumption-based and should be replaced with "
        "observed campaign economics."
    )

    if not forecast_available:
        limitation += " No approved forecast summary was available."

    return {
        "finding": finding,
        "action": action,
        "limitation": limitation,
        "incremental_value": f"{incremental_value:,.2f}",
    }


def executive_brief_table(
    *,
    holdout: dict[str, float | str],
    scenario: dict[str, float],
    anomaly: dict[str, Any],
    readiness: pd.DataFrame,
    source_timestamp: str,
) -> pd.DataFrame:
    """Build the exportable filtered executive brief."""

    ready_count = int(readiness["Status"].eq("Ready").sum())

    rows = [
        ("Champion model", holdout["model_name"], "Validated holdout"),
        ("Threshold", holdout["threshold"], "Validated holdout"),
        ("PR-AUC", holdout["pr_auc"], "Validated holdout"),
        ("ROC-AUC", holdout["roc_auc"], "Validated holdout"),
        ("Precision", holdout["precision"], "Validated holdout"),
        ("Recall", holdout["recall"], "Validated holdout"),
        ("Target size", int(scenario["target_size"]), "Scenario"),
        ("Expected buyers", scenario["expected_buyers"], "Scenario"),
        ("Campaign cost", scenario["campaign_cost"], "Scenario"),
        ("Expected value", scenario["expected_value"], "Scenario"),
        ("Net value", scenario["net_value"], "Scenario"),
        ("ROI", scenario["roi"], "Scenario"),
        ("Anomaly count", anomaly["count"], "Validation evidence"),
        ("Anomaly rate", anomaly["rate"], "Validation evidence"),
        ("Readiness areas ready", f"{ready_count}/{len(readiness)}", "Artifacts"),
        ("Evidence timestamp", source_timestamp, "Source metadata"),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value", "Evidence type"])
