# 6_Monitoring_Drift_Health.py
# Consolidated Streamlit page for monitoring, drift, and operational health.
#
# Evidence boundaries:
# - Current session logs show only predictions recorded by the running app.
# - Saved Evidently JSON files show the latest governed offline drift snapshot.
# - Delayed-label evidence is shown only when matured labels exist.
# - Offline holdout metrics are never relabelled as live production results.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

while (
    PROJECT_ROOT != PROJECT_ROOT.parent
    and not (PROJECT_ROOT / "src").is_dir()
):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_utils import (  # noqa: E402
    get_best_threshold,
    get_champion_model_name,
    inject_global_css,
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
SINGLE_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring/prediction_logs/prediction_log.csv"
)
BATCH_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring/prediction_logs/batch_scoring_log.csv"
)

SCORE_PATHS = [
    PROJECT_ROOT
    / "data/processed/final_champion_visitor_scores.csv",
    PROJECT_ROOT
    / "data/processed/champion_visitor_scores.csv",
]

ANOMALY_SUMMARY_PATH = (
    PROJECT_ROOT / "reports/tables/anomaly_summary.csv"
)
FORECAST_FUTURE_PATH = (
    PROJECT_ROOT / "reports/tables/business_forecast_future.csv"
)
DAILY_KPI_PATH = (
    PROJECT_ROOT / "reports/tables/daily_business_kpis.csv"
)

FEATURE_DRIFT_PATH = (
    PROJECT_ROOT
    / "reports/evidently/latest/feature_drift_report.json"
)
PREDICTION_DRIFT_PATH = (
    PROJECT_ROOT
    / "reports/evidently/latest/prediction_drift_report.json"
)

SOURCE_STATUS_PATH = (
    PROJECT_ROOT
    / "reports/visuals/ml_visual_intelligence/monitoring/"
    "monitoring_source_status.csv"
)
DELAYED_FUNNEL_PATH = (
    PROJECT_ROOT
    / "reports/visuals/ml_visual_intelligence/monitoring/"
    "delayed_label_funnel.csv"
)
DELAYED_REJECTIONS_PATH = (
    PROJECT_ROOT
    / "reports/visuals/ml_visual_intelligence/monitoring/"
    "delayed_label_rejections.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models/trained/final_champion_model.joblib"
)
MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models/metadata/final_champion_metadata.json"
)
SCORE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "models/metadata/final_champion_score_manifest.json"
)
MLFLOW_LINEAGE_PATH = (
    PROJECT_ROOT
    / "models/metadata/mlflow_champion_lineage.json"
)
REGISTRY_PATH = (
    PROJECT_ROOT
    / "reports/visuals/ml_visual_intelligence/"
    "experiment_tracking/verified_ecommerce_registry_versions.csv"
)

MINIMUM_LIVE_SAMPLE = 30


# --------------------------------------------------
# LOADERS
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_if_exists(path_text: str) -> pd.DataFrame:
    """Read a CSV when present and return an empty frame otherwise."""

    path = Path(path_text)

    if not path.is_file():
        return pd.DataFrame()

    data = pd.read_csv(path)

    for column in (
        "timestamp_utc",
        "scored_at_utc",
        "date",
        "created_at_utc",
    ):
        if column in data.columns:
            data[column] = pd.to_datetime(
                data[column],
                errors="coerce",
                utc=True,
            )

    return data


@st.cache_data(show_spinner=False)
def read_json_if_exists(path_text: str) -> dict[str, Any]:
    """Read one JSON artifact when it exists."""

    path = Path(path_text)

    if not path.is_file():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_score_baseline(
    path_text: str,
    sample_size: int = 200_000,
) -> pd.DataFrame:
    """Load a deterministic baseline sample from the active score file."""

    path = Path(path_text)

    if not path.is_file():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if "purchase_intent_score" not in data.columns:
        return pd.DataFrame()

    data["purchase_intent_score"] = pd.to_numeric(
        data["purchase_intent_score"],
        errors="coerce",
    )
    data = data.dropna(subset=["purchase_intent_score"])

    if len(data) > sample_size:
        data = data.sample(
            n=sample_size,
            random_state=42,
        )

    return data


def active_score_file() -> Path:
    """Prefer the final champion score file."""

    for path in SCORE_PATHS:
        if path.is_file():
            return path

    return SCORE_PATHS[0]


def file_hash(path: Path) -> str:
    """Return a short SHA-256 hash for one local artifact."""

    if not path.is_file():
        return "Unavailable"

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()[:12]


# --------------------------------------------------
# SESSION LOG PREPARATION
# --------------------------------------------------
def extract_live_scores(log: pd.DataFrame) -> pd.DataFrame:
    """Normalise score values from the current app prediction log."""

    if log.empty:
        return pd.DataFrame()

    score_column = next(
        (
            column
            for column in (
                "purchase_intent_score",
                "score",
                "prediction_score",
                "intent_score",
            )
            if column in log.columns
        ),
        None,
    )

    if score_column is None:
        return pd.DataFrame()

    output = pd.DataFrame(
        {
            "purchase_intent_score": pd.to_numeric(
                log[score_column],
                errors="coerce",
            )
        }
    )

    if "timestamp_utc" in log.columns:
        output["timestamp_utc"] = pd.to_datetime(
            log["timestamp_utc"],
            errors="coerce",
            utc=True,
        )
    else:
        output["timestamp_utc"] = pd.NaT

    return output.dropna(subset=["purchase_intent_score"])


# --------------------------------------------------
# DRIFT HELPERS
# --------------------------------------------------
def calculate_psi(
    expected_scores: pd.Series,
    actual_scores: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI only after sample-size gating is satisfied."""

    expected = pd.to_numeric(
        expected_scores,
        errors="coerce",
    ).dropna()
    actual = pd.to_numeric(
        actual_scores,
        errors="coerce",
    ).dropna()

    if expected.empty or len(actual) < MINIMUM_LIVE_SAMPLE:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_share = (
        expected_counts / max(expected_counts.sum(), 1)
    )
    actual_share = actual_counts / max(actual_counts.sum(), 1)

    expected_share = np.where(
        expected_share == 0,
        0.0001,
        expected_share,
    )
    actual_share = np.where(
        actual_share == 0,
        0.0001,
        actual_share,
    )

    values = (
        (actual_share - expected_share)
        * np.log(actual_share / expected_share)
    )

    return float(values.sum())


def psi_status(value: float, live_count: int) -> str:
    """Translate PSI into a status without overstating tiny samples."""

    if live_count < MINIMUM_LIVE_SAMPLE:
        return "Insufficient live data"

    if pd.isna(value):
        return "Unavailable"

    if value < 0.10:
        return "Stable"

    if value < 0.25:
        return "Moderate drift"

    return "Major drift"


def score_bucket(score: float, threshold: float) -> str:
    """Convert a score into one readable operational bucket."""

    if score >= threshold:
        return "High intent"
    if score >= 0.80:
        return "Strong intent"
    if score >= 0.50:
        return "Warm intent"
    if score >= 0.20:
        return "Low intent"

    return "Cold visitor"


def score_bucket_table(
    scores: pd.Series,
    source: str,
    threshold: float,
) -> pd.DataFrame:
    """Create a comparable bucket distribution."""

    clean = pd.to_numeric(scores, errors="coerce").dropna()

    if clean.empty:
        return pd.DataFrame()

    order = [
        "Cold visitor",
        "Low intent",
        "Warm intent",
        "Strong intent",
        "High intent",
    ]

    counts = (
        clean.apply(
            lambda value: score_bucket(value, threshold)
        )
        .value_counts()
        .reindex(order)
        .fillna(0)
        .rename_axis("Score bucket")
        .reset_index(name="Visitor count")
    )
    counts["Visitor share"] = (
        counts["Visitor count"]
        / max(counts["Visitor count"].sum(), 1)
    )
    counts["Source"] = source

    return counts


def numeric_value(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    """Read the first finite numeric value from a dictionary."""

    for key in keys:
        if key not in mapping:
            continue

        try:
            value = float(mapping[key])
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            return value

    return None


def parse_drift_report(
    report: dict[str, Any],
    source_name: str,
) -> pd.DataFrame:
    """Extract drift evidence from old and new Evidently JSON layouts."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if not isinstance(node, dict):
            return

        config = (
            node.get("config")
            if isinstance(node.get("config"), dict)
            else {}
        )
        result = (
            node.get("result")
            if isinstance(node.get("result"), dict)
            else {}
        )

        context = " ".join(
            str(value)
            for value in (
                node.get("type"),
                node.get("metric"),
                node.get("metric_id"),
                node.get("id"),
                config.get("type"),
                config.get("metric"),
                result.get("metric"),
            )
            if value is not None
        ).lower()

        feature = next(
            (
                str(value)
                for value in (
                    config.get("column"),
                    config.get("column_name"),
                    node.get("column"),
                    node.get("column_name"),
                    result.get("column"),
                    result.get("column_name"),
                    result.get("feature"),
                )
                if value not in (None, "")
            ),
            None,
        )

        score = numeric_value(
            node,
            ("drift_score", "score"),
        )

        if score is None:
            score = numeric_value(
                result,
                ("drift_score", "score", "value"),
            )

        if score is None and "drift" in context:
            score = numeric_value(node, ("value",))

        threshold = (
            numeric_value(
                config,
                ("threshold", "drift_threshold"),
            )
            or numeric_value(
                node,
                ("threshold", "drift_threshold"),
            )
            or numeric_value(
                result,
                ("threshold", "drift_threshold"),
            )
            or 0.10
        )

        detected_value = next(
            (
                value
                for value in (
                    node.get("drift_detected"),
                    node.get("detected"),
                    result.get("drift_detected"),
                    result.get("detected"),
                )
                if value is not None
            ),
            None,
        )

        detected = (
            bool(detected_value)
            if isinstance(detected_value, bool)
            else (
                str(detected_value).strip().lower()
                in {"true", "1", "yes", "drift"}
                if detected_value is not None
                else (
                    score >= threshold
                    if score is not None
                    else False
                )
            )
        )

        if feature is not None and score is not None:
            key = (feature, round(score, 12), source_name)

            if key not in seen:
                rows.append(
                    {
                        "Feature": feature,
                        "Drift score": score,
                        "Threshold": threshold,
                        "Alert state": (
                            "Drift" if detected else "Stable"
                        ),
                        "Source": source_name,
                        "Method": str(
                            config.get("method")
                            or node.get("method")
                            or result.get("method")
                            or "Saved Evidently metric"
                        ),
                    }
                )
                seen.add(key)

        for value in node.values():
            if isinstance(value, (dict, list)):
                visit(value)

    visit(report)

    return pd.DataFrame(rows)


# --------------------------------------------------
# CHARTS
# --------------------------------------------------
def bucket_figure(data: pd.DataFrame) -> go.Figure:
    """Draw score-bucket shares."""

    figure = px.bar(
        data,
        x="Score bucket",
        y="Visitor share",
        color="Source",
        barmode="group",
        text="Visitor share",
        title="Score distribution by operational intent bucket",
    )
    figure.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )
    figure.update_yaxes(
        tickformat=".0%",
        title="Visitor share",
    )
    figure.update_xaxes(title="")
    figure.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title="Evidence",
    )

    return figure


def logging_volume_figure(
    single_scores: pd.DataFrame,
    batch_log: pd.DataFrame,
) -> go.Figure | None:
    """Draw available session-local scoring activity."""

    rows: list[pd.DataFrame] = []

    if (
        not single_scores.empty
        and "timestamp_utc" in single_scores.columns
    ):
        single = (
            single_scores.dropna(subset=["timestamp_utc"])
            .assign(
                date=lambda frame: (
                    frame["timestamp_utc"].dt.date
                )
            )
            .groupby("date")
            .size()
            .reset_index(name="Logged events")
        )
        single["Source"] = "Single visitor"
        rows.append(single)

    if (
        not batch_log.empty
        and "timestamp_utc" in batch_log.columns
    ):
        batch = (
            batch_log.dropna(subset=["timestamp_utc"])
            .assign(
                date=lambda frame: (
                    frame["timestamp_utc"].dt.date
                )
            )
            .groupby("date")
            .size()
            .reset_index(name="Logged events")
        )
        batch["Source"] = "Batch scoring"
        rows.append(batch)

    if not rows:
        return None

    data = pd.concat(rows, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])

    figure = px.line(
        data,
        x="date",
        y="Logged events",
        color="Source",
        markers=True,
        title="Available prediction-log activity",
    )
    figure.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title="Log type",
        xaxis_title="Date",
    )

    return figure


def forecast_health_figure(
    forecast: pd.DataFrame,
) -> go.Figure | None:
    """Show that approved forecast outputs are available."""

    required = {
        "date",
        "target_name",
        "predicted_value",
    }

    if forecast.empty or not required.issubset(
        forecast.columns
    ):
        return None

    data = forecast.copy()

    if "is_best_model" in data.columns:
        best = (
            data["is_best_model"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

        if best.any():
            data = data.loc[best].copy()

    figure = px.line(
        data,
        x="date",
        y="predicted_value",
        color="target_name",
        markers=True,
        title="Approved future KPI forecast availability",
    )
    figure.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title="KPI",
        xaxis_title="Date",
        yaxis_title="Predicted value",
    )

    return figure


def drift_figure(drift: pd.DataFrame) -> go.Figure:
    """Draw the saved offline drift snapshot."""

    ordered = drift.sort_values(
        "Drift score",
        ascending=True,
    )

    figure = px.bar(
        ordered,
        x="Drift score",
        y="Feature",
        orientation="h",
        color="Alert state",
        text="Drift score",
        hover_data={
            "Threshold": ":.4f",
            "Source": True,
            "Method": True,
        },
        title="Latest saved Evidently drift snapshot",
    )
    figure.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    threshold = float(
        pd.to_numeric(
            drift["Threshold"],
            errors="coerce",
        ).dropna().max()
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Saved alert threshold",
    )
    figure.update_layout(
        height=max(450, 45 * len(drift)),
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title="State",
    )

    return figure


def delayed_label_figure(
    funnel: pd.DataFrame,
) -> go.Figure | None:
    """Draw matured-label progression when the evidence exists."""

    if (
        funnel.empty
        or not {"stage", "count"}.issubset(funnel.columns)
    ):
        return None

    figure = go.Figure(
        go.Funnel(
            y=funnel["stage"],
            x=pd.to_numeric(
                funnel["count"],
                errors="coerce",
            ).fillna(0),
            textinfo="value+percent initial",
        )
    )
    figure.update_layout(
        title="Delayed-label maturity funnel",
        height=480,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


# --------------------------------------------------
# DISPLAY HELPERS
# --------------------------------------------------
def format_count(value: int | float) -> str:
    """Format one count for a KPI card."""

    number = float(value)

    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"

    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"

    return f"{number:,.0f}"


def format_rate(value: float) -> str:
    """Format a valid rate or show unavailable."""

    if pd.isna(value):
        return "Not assessed"

    return f"{value:.1%}"


def environment_summary(metadata: dict[str, Any]) -> str:
    """Turn training-environment metadata into one readable line."""

    environment = metadata.get("environment", {})

    if not isinstance(environment, dict):
        return str(environment or "Unavailable")

    python_version = environment.get(
        "python_version",
        "Unknown Python",
    )
    operating_system = environment.get(
        "operating_system",
        "Unknown OS",
    )
    architecture = environment.get(
        "machine_architecture",
        "Unknown architecture",
    )

    return (
        f"Python {python_version} · "
        f"{operating_system} · {architecture}"
    )


# --------------------------------------------------
# LOAD EVIDENCE
# --------------------------------------------------
score_file = active_score_file()
baseline_scores = load_score_baseline(str(score_file))
single_log = read_csv_if_exists(str(SINGLE_LOG_PATH))
batch_log = read_csv_if_exists(str(BATCH_LOG_PATH))
live_scores = extract_live_scores(single_log)

anomaly_summary = read_csv_if_exists(
    str(ANOMALY_SUMMARY_PATH)
)
forecast_future = read_csv_if_exists(
    str(FORECAST_FUTURE_PATH)
)
daily_kpis = read_csv_if_exists(str(DAILY_KPI_PATH))

feature_report = read_json_if_exists(
    str(FEATURE_DRIFT_PATH)
)
prediction_report = read_json_if_exists(
    str(PREDICTION_DRIFT_PATH)
)
feature_drift = parse_drift_report(
    feature_report,
    "Feature drift",
)
prediction_drift = parse_drift_report(
    prediction_report,
    "Prediction drift",
)
drift = pd.concat(
    [feature_drift, prediction_drift],
    ignore_index=True,
)

source_status = read_csv_if_exists(
    str(SOURCE_STATUS_PATH)
)
delayed_funnel = read_csv_if_exists(
    str(DELAYED_FUNNEL_PATH)
)
delayed_rejections = read_csv_if_exists(
    str(DELAYED_REJECTIONS_PATH)
)
registry = read_csv_if_exists(str(REGISTRY_PATH))

metadata = read_json_if_exists(
    str(MODEL_METADATA_PATH)
)
score_manifest = read_json_if_exists(
    str(SCORE_MANIFEST_PATH)
)
lineage = read_json_if_exists(
    str(MLFLOW_LINEAGE_PATH)
)

champion_name = get_champion_model_name()
threshold = float(get_best_threshold())

baseline_count = int(len(baseline_scores))
live_count = int(len(live_scores))
live_sample_ready = live_count >= MINIMUM_LIVE_SAMPLE

psi_value = calculate_psi(
    baseline_scores.get(
        "purchase_intent_score",
        pd.Series(dtype=float),
    ),
    live_scores.get(
        "purchase_intent_score",
        pd.Series(dtype=float),
    ),
)
current_psi_status = psi_status(
    psi_value,
    live_count,
)

baseline_high_intent_rate = (
    float(
        baseline_scores["purchase_intent_score"]
        .ge(threshold)
        .mean()
    )
    if baseline_count
    else float("nan")
)
live_high_intent_rate = (
    float(
        live_scores["purchase_intent_score"]
        .ge(threshold)
        .mean()
    )
    if live_sample_ready
    else float("nan")
)

drift_alerts = (
    int(drift["Alert state"].eq("Drift").sum())
    if not drift.empty
    else None
)

ready_sources = (
    int(
        source_status["exists"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .sum()
    )
    if (
        not source_status.empty
        and "exists" in source_status.columns
    )
    else None
)

matured_count = (
    int(
        pd.to_numeric(
            delayed_funnel.iloc[-1]["count"],
            errors="coerce",
        )
    )
    if (
        not delayed_funnel.empty
        and "count" in delayed_funnel.columns
    )
    else None
)


# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Monitoring, Drift & Health",
    page_icon="🩺",
    layout="wide",
)
inject_global_css()

st.caption("Production monitoring layer")
st.title("Monitoring, Drift & Health")
st.write(
    "This page separates session logging, saved offline drift evidence, "
    "artifact readiness, and genuine matured-label performance."
)

st.info(
    "A PSI result is not reported until at least "
    f"{MINIMUM_LIVE_SAMPLE} live prediction scores exist. "
    "One prediction cannot establish production drift."
)

header_1, header_2, header_3 = st.columns(3)
header_1.metric("Champion", champion_name)
header_2.metric("Production threshold", f"{threshold:.2f}")
header_3.metric("Live PSI status", current_psi_status)


# --------------------------------------------------
# LIVE SESSION HEALTH
# --------------------------------------------------
st.subheader("Current logging and score health")
st.caption(
    "Prediction-log counts describe the currently available app log file. "
    "They are not presented as durable multi-period production history."
)

card_1, card_2, card_3, card_4 = st.columns(4)

card_1.metric(
    "Available live predictions",
    format_count(live_count),
    help="Scores currently present in the single-prediction log.",
)
card_2.metric(
    "Baseline score sample",
    format_count(baseline_count),
    help="Saved champion scores used as the comparison baseline.",
)
card_3.metric(
    "PSI status",
    current_psi_status,
    help=(
        f"PSI is assessed only after {MINIMUM_LIVE_SAMPLE} "
        "live scores are available."
    ),
)
card_4.metric(
    "Live high-intent rate",
    format_rate(live_high_intent_rate),
    help=(
        f"Baseline high-intent rate: "
        f"{format_rate(baseline_high_intent_rate)}"
    ),
)

if not live_sample_ready:
    st.warning(
        f"Only {live_count} live prediction score(s) are available. "
        f"At least {MINIMUM_LIVE_SAMPLE} are required before PSI, "
        "live high-intent rate, or live-versus-baseline drift claims "
        "are shown."
    )
else:
    st.caption(f"Calculated PSI: {psi_value:.3f}")

baseline_buckets = (
    score_bucket_table(
        baseline_scores["purchase_intent_score"],
        "Saved champion baseline",
        threshold,
    )
    if "purchase_intent_score" in baseline_scores.columns
    else pd.DataFrame()
)

if live_sample_ready:
    live_buckets = score_bucket_table(
        live_scores["purchase_intent_score"],
        "Current live sample",
        threshold,
    )
else:
    live_buckets = pd.DataFrame()

bucket_data = pd.concat(
    [baseline_buckets, live_buckets],
    ignore_index=True,
)

if bucket_data.empty:
    st.info("Score-distribution evidence is unavailable.")
else:
    st.plotly_chart(
        bucket_figure(bucket_data),
        width="stretch",
        key="monitoring_score_buckets",
    )
    st.caption(
        "The live comparison is added only after the minimum sample gate "
        "is reached. Until then, the chart shows the saved baseline only."
    )

chart_left, chart_right = st.columns(2)

with chart_left:
    volume = logging_volume_figure(
        live_scores,
        batch_log,
    )

    if volume is None:
        st.info("No prediction-log timeline is available yet.")
    else:
        st.plotly_chart(
            volume,
            width="stretch",
            key="monitoring_log_volume",
        )
        st.caption(
            "This is available log activity, not a guaranteed persistent "
            "production-history store."
        )

with chart_right:
    forecast_chart = forecast_health_figure(
        forecast_future
    )

    if forecast_chart is None:
        st.info("Approved forecast output is unavailable.")
    else:
        st.plotly_chart(
            forecast_chart,
            width="stretch",
            key="monitoring_forecast_health",
        )
        st.caption(
            "This confirms that the approved forecast artifact is present."
        )


# --------------------------------------------------
# SAVED OFFLINE DRIFT SNAPSHOT
# --------------------------------------------------
st.divider()
st.subheader("Latest governed drift snapshot")
st.caption(
    "This section reads saved Evidently feature and prediction drift JSON. "
    "It is separate from the small live-session log above."
)

if drift.empty:
    st.error(
        "No compatible saved drift metrics are available. "
        "The page therefore does not claim that the system is healthy."
    )
else:
    st.plotly_chart(
        drift_figure(drift),
        width="stretch",
        key="monitoring_saved_drift",
    )

    highest = drift.sort_values(
        "Drift score",
        ascending=False,
    ).iloc[0]

    st.write(
        f"**Highest saved drift score:** "
        f"{highest['Drift score']:.4f} for "
        f"{highest['Feature']}."
    )
    st.write(
        f"**Active saved alerts:** {drift_alerts}."
    )
    st.dataframe(
        drift,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# OPERATIONAL READINESS
# --------------------------------------------------
st.divider()
st.subheader("Operational readiness")

readiness_1, readiness_2, readiness_3, readiness_4 = (
    st.columns(4)
)

readiness_1.metric(
    "Saved drift alerts",
    (
        format_count(drift_alerts)
        if drift_alerts is not None
        else "Unavailable"
    ),
)
readiness_2.metric(
    "Monitoring sources",
    (
        f"{ready_sources}/{len(source_status)}"
        if ready_sources is not None
        else "Unavailable"
    ),
)
readiness_3.metric(
    "Evaluated matured cohort",
    (
        format_count(matured_count)
        if matured_count is not None
        else "Not published"
    ),
)
readiness_4.metric(
    "Verified registry rows",
    (
        format_count(len(registry))
        if not registry.empty
        else "Not published"
    ),
)

if source_status.empty:
    st.warning(
        "Monitoring source-status evidence is unavailable."
    )
else:
    st.dataframe(
        source_status,
        width="stretch",
        hide_index=True,
    )

funnel_chart = delayed_label_figure(delayed_funnel)

if funnel_chart is None:
    st.info(
        "No matured delayed-label evaluation artifact is published. "
        "Offline holdout results remain separate and are not presented "
        "as live production performance."
    )
else:
    st.plotly_chart(
        funnel_chart,
        width="stretch",
        key="monitoring_delayed_label_funnel",
    )

    if not delayed_rejections.empty:
        with st.expander("Delayed-label rejection reasons"):
            st.dataframe(
                delayed_rejections,
                width="stretch",
                hide_index=True,
            )


# --------------------------------------------------
# LINEAGE
# --------------------------------------------------
st.divider()
st.subheader("Artifact lineage")

lineage_1, lineage_2 = st.columns(2)

with lineage_1:
    st.markdown("#### Model artifact")
    st.write(
        str(
            metadata.get("final_model_name")
            or metadata.get("model_name")
            or champion_name
        )
    )
    st.caption(f"SHA-256: {file_hash(MODEL_PATH)}")

    st.markdown("#### Training environment")
    st.write(environment_summary(metadata))
    st.caption(f"Metadata SHA-256: {file_hash(MODEL_METADATA_PATH)}")

with lineage_2:
    st.markdown("#### Score manifest")

    if score_manifest:
        st.write(
            str(
                score_manifest.get(
                    "model_generation",
                    "Final champion scores",
                )
            )
        )
        st.caption(
            f"Rows: "
            f"{score_manifest.get('row_count', 'Unavailable')} · "
            f"SHA-256: {file_hash(SCORE_MANIFEST_PATH)}"
        )
    else:
        st.write("Not published")

    st.markdown("#### MLflow lineage")

    if lineage:
        st.write(
            str(
                lineage.get("run_id")
                or lineage.get("model_name")
                or "Published lineage"
            )
        )
        st.caption(
            f"SHA-256: {file_hash(MLFLOW_LINEAGE_PATH)}"
        )
    else:
        st.write("Not published")


# --------------------------------------------------
# CURRENT HEALTH DECISION
# --------------------------------------------------
st.divider()
st.subheader("Current monitoring decision")

if drift.empty:
    health_state = "Evidence unavailable"
    action = (
        "Generate and publish compatible feature and prediction "
        "drift reports."
    )
elif drift_alerts and drift_alerts > 0:
    health_state = "Drift investigation required"
    action = (
        "Investigate the saved alerts before treating the snapshot "
        "as healthy."
    )
else:
    health_state = "Saved snapshot stable"
    action = (
        "Continue scheduled regeneration and accumulate durable "
        "multi-snapshot history."
    )

decision_1, decision_2, decision_3 = st.columns(3)
decision_1.metric("Current health", health_state)
decision_2.metric("Required action", action)
decision_3.metric(
    "Production limitation",
    "Matured labels only",
    help=(
        "Offline holdout metrics are not live production performance."
    ),
)

st.caption(
    "No multi-snapshot drift-history table is claimed. "
    "The page shows current available logs and the latest saved "
    "offline monitoring snapshot only."
)


# --------------------------------------------------
# RAW EVIDENCE
# --------------------------------------------------
st.divider()
st.subheader("Raw monitoring evidence")

tabs = st.tabs(
    [
        "Live scores",
        "Baseline scores",
        "Batch logs",
        "Anomaly summary",
        "Forecast output",
        "Registry evidence",
    ]
)

with tabs[0]:
    if live_scores.empty:
        st.info("No single-prediction scores are available.")
    else:
        st.dataframe(
            live_scores,
            width="stretch",
            hide_index=True,
        )

with tabs[1]:
    if baseline_scores.empty:
        st.info("The champion score baseline is unavailable.")
    else:
        st.dataframe(
            baseline_scores.head(1000),
            width="stretch",
            hide_index=True,
        )

with tabs[2]:
    if batch_log.empty:
        st.info("No batch-scoring log is available.")
    else:
        st.dataframe(
            batch_log,
            width="stretch",
            hide_index=True,
        )

with tabs[3]:
    if anomaly_summary.empty:
        st.info("The anomaly summary is unavailable.")
    else:
        st.dataframe(
            anomaly_summary,
            width="stretch",
            hide_index=True,
        )

with tabs[4]:
    if forecast_future.empty:
        st.info("The future forecast artifact is unavailable.")
    else:
        st.dataframe(
            forecast_future,
            width="stretch",
            hide_index=True,
        )

with tabs[5]:
    if registry.empty:
        st.info(
            "Verified ecommerce registry evidence is not published."
        )
    else:
        st.dataframe(
            registry,
            width="stretch",
            hide_index=True,
        )
