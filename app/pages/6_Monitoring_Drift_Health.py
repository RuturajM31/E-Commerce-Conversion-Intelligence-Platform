# 6_Monitoring_Drift_Health.py
# Streamlit page for model monitoring, drift, and health checks.
#
# Purpose:
#   Show whether the deployed scoring system is behaving normally.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Paths
#   4. Helper functions
#   5. Drift and health logic
#   6. Chart functions
#   7. Load monitoring data
#   8. Sidebar and header
#   9. Health KPI cards
#   10. Drift and score-health charts
#   11. Operational evidence tables

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from app.app_utils import (
    escape_text,
    format_percent,
    get_best_threshold,
    get_champion_model_name,
    inject_global_css,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Monitoring, Drift & Health",
    page_icon="📡",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .monitor-hero {
            padding: 34px 38px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.26), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(168, 85, 247, 0.22), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .monitor-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .monitor-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1050px;
        }

        .health-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 184px;
            margin-bottom: 18px;
        }

        .health-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .health-value {
            color: #F8FAFC;
            font-size: 2.20rem;
            line-height: 1;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .health-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.48;
        }

        .section-kicker {
            color: #93C5FD;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-top: 16px;
            margin-bottom: 4px;
        }

        .section-title {
            color: #F8FAFC;
            font-size: 1.45rem;
            font-weight: 950;
            margin-bottom: 12px;
        }

        .story-card {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(2, 6, 23, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.34);
            margin-bottom: 18px;
        }

        .story-title {
            color: #F8FAFC;
            font-size: 1.25rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .story-text {
            color: #CBD5E1;
            font-size: 0.94rem;
            line-height: 1.58;
        }
    </style>
    """
)


# --------------------------------------------------
# 3. INPUT PATHS
# --------------------------------------------------

SINGLE_PREDICTION_LOG_PATH = Path("monitoring/prediction_logs/prediction_log.csv")
BATCH_SCORING_LOG_PATH = Path("monitoring/prediction_logs/batch_scoring_log.csv")

FINAL_CHAMPION_VISITOR_SCORES_PATH = Path("data/processed/final_champion_visitor_scores.csv")
CHAMPION_VISITOR_SCORES_PATH = Path("data/processed/champion_visitor_scores.csv")

ANOMALY_SUMMARY_PATH = Path("reports/tables/anomaly_summary.csv")
FORECAST_FUTURE_PATH = Path("reports/tables/business_forecast_future.csv")
DAILY_KPI_PATH = Path("reports/tables/daily_business_kpis.csv")


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists."""

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    for column in ["timestamp_utc", "date", "scored_at_utc"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    return data


@st.cache_data(show_spinner=False)
def load_score_baseline(path_text: str, sample_size: int = 200_000) -> pd.DataFrame:
    """Load a sample of champion visitor scores for baseline monitoring."""

    path = Path(path_text)

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if "purchase_intent_score" not in data.columns:
        return pd.DataFrame()

    if len(data) > sample_size:
        data = data.sample(
            n=sample_size,
            random_state=42,
        )

    return data


def get_active_score_file() -> Path:
    """Prefer final champion score file, then fallback to current champion scores."""

    if FINAL_CHAMPION_VISITOR_SCORES_PATH.exists():
        return FINAL_CHAMPION_VISITOR_SCORES_PATH

    return CHAMPION_VISITOR_SCORES_PATH


def format_count(value: float) -> str:
    """Format count values for cards."""

    if pd.isna(value):
        return "NA"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_score(value: float) -> str:
    """Format a probability score as a percent."""

    if pd.isna(value):
        return "NA"

    return f"{float(value) * 100:.1f}%"


def get_summary_value(summary: pd.DataFrame, metric: str) -> float:
    """Read one metric from anomaly_summary.csv."""

    if summary.empty:
        return np.nan

    if {"metric", "value"}.issubset(summary.columns):
        row = summary[summary["metric"].astype(str) == metric]

        if not row.empty:
            return float(row.iloc[0]["value"])

    if metric in summary.columns:
        return float(summary.iloc[0][metric])

    return np.nan


def extract_live_prediction_scores(single_log: pd.DataFrame) -> pd.DataFrame:
    """Standardise live prediction scores from prediction logs."""

    if single_log.empty:
        return pd.DataFrame()

    data = single_log.copy()

    score_candidates = [
        "purchase_intent_score",
        "score",
        "prediction_score",
        "intent_score",
    ]

    score_column = None

    for candidate in score_candidates:
        if candidate in data.columns:
            score_column = candidate
            break

    if score_column is None:
        return pd.DataFrame()

    output = pd.DataFrame()

    output["timestamp_utc"] = (
        pd.to_datetime(data["timestamp_utc"], errors="coerce")
        if "timestamp_utc" in data.columns
        else pd.NaT
    )

    output["purchase_intent_score"] = pd.to_numeric(
        data[score_column],
        errors="coerce",
    )

    output["intent_segment"] = (
        data["intent_segment"].astype(str)
        if "intent_segment" in data.columns
        else "Unknown"
    )

    output["recommended_action"] = (
        data["recommended_action"].astype(str)
        if "recommended_action" in data.columns
        else "Unknown"
    )

    return output.dropna(subset=["purchase_intent_score"])


def create_score_bucket(score: float, threshold: float) -> str:
    """Convert score into business-friendly bucket."""

    if score >= threshold:
        return "High Intent"
    if score >= 0.80:
        return "Strong Intent"
    if score >= 0.50:
        return "Warm Intent"
    if score >= 0.20:
        return "Low Intent"

    return "Cold Visitor"


def build_score_bucket_table(
    scores: pd.Series,
    source_label: str,
    threshold: float,
) -> pd.DataFrame:
    """Build score-bucket distribution for baseline or live scores."""

    clean_scores = pd.to_numeric(scores, errors="coerce").dropna()

    if clean_scores.empty:
        return pd.DataFrame()

    buckets = clean_scores.apply(lambda score: create_score_bucket(score, threshold))

    order = [
        "Cold Visitor",
        "Low Intent",
        "Warm Intent",
        "Strong Intent",
        "High Intent",
    ]

    counts = buckets.value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["Score Bucket", "Visitor Count"]
    counts["Source"] = source_label
    counts["Visitor Share"] = counts["Visitor Count"] / counts["Visitor Count"].sum()

    return counts


def calculate_population_stability_index(
    expected_scores: pd.Series,
    actual_scores: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate simple PSI between baseline and live score distributions."""

    expected = pd.to_numeric(expected_scores, errors="coerce").dropna()
    actual = pd.to_numeric(actual_scores, errors="coerce").dropna()

    if expected.empty or actual.empty:
        return np.nan

    bin_edges = np.linspace(0, 1, bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    return float(np.sum(psi_values))


def interpret_psi(psi_value: float) -> str:
    """Convert PSI number into health status."""

    if pd.isna(psi_value):
        return "Not enough live data"

    if psi_value < 0.10:
        return "Stable"

    if psi_value < 0.25:
        return "Moderate drift"

    return "Major drift"


def get_numeric_summary(data: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate simple score health statistics."""

    if data.empty or column not in data.columns:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    values = pd.to_numeric(data[column], errors="coerce").dropna()

    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def check_file_status(path: Path) -> str:
    """Return OK or Missing for a project file."""

    return "OK" if path.exists() else "Missing"



def format_score_value(value: float) -> str:
    """Format a numeric health value."""

    if pd.isna(value):
        return "NA"

    return f"{float(value):.3f}"




def render_chart_explanation(title: str, shows: str, conclusion: str) -> None:
    """Add a compact explanation and conclusion under a chart."""

    render_html(
        (
            f'<div class="story-card">'
            f'<div class="story-title">{escape_text(title)}</div>'
            f'<div class="story-text">'
            f'<b>What this chart shows:</b> {escape_text(shows)}<br><br>'
            f'<b>Conclusion:</b> {escape_text(conclusion)}'
            f'</div></div>'
        )
    )

# --------------------------------------------------
# 5. CHART FUNCTIONS
# --------------------------------------------------

def build_score_bucket_chart(bucket_table: pd.DataFrame):
    """Create readable baseline vs live score-bucket chart.

    This replaces the old raw histogram because the raw baseline score distribution
    is too dominated by cold visitors.
    """

    if not PLOTLY_AVAILABLE or bucket_table.empty:
        return None

    fig = px.bar(
        bucket_table,
        x="Score Bucket",
        y="Visitor Share",
        color="Source",
        barmode="group",
        text="Visitor Share",
        title="Score health: baseline vs live visitor intent buckets",
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Intent bucket",
        yaxis_title="Visitor share",
        legend_title="Source",
        title_font=dict(size=20),
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def build_prediction_volume_chart(live_scores: pd.DataFrame, batch_log: pd.DataFrame):
    """Create prediction logging volume chart."""

    if not PLOTLY_AVAILABLE:
        return None

    rows = []

    if not live_scores.empty and "timestamp_utc" in live_scores.columns:
        single_daily = (
            live_scores
            .dropna(subset=["timestamp_utc"])
            .assign(date=lambda data: data["timestamp_utc"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="prediction_count")
        )
        single_daily["source"] = "Single visitor"

        rows.append(single_daily)

    if not batch_log.empty:
        date_column = "timestamp_utc" if "timestamp_utc" in batch_log.columns else None

        if date_column is not None:
            batch_daily = (
                batch_log
                .dropna(subset=[date_column])
                .assign(date=lambda data: data[date_column].dt.date)
                .groupby("date")
                .size()
                .reset_index(name="prediction_count")
            )
            batch_daily["source"] = "Batch scoring"

            rows.append(batch_daily)

    if not rows:
        return None

    chart_data = pd.concat(rows, ignore_index=True)
    chart_data["date"] = pd.to_datetime(chart_data["date"])

    fig = px.line(
        chart_data,
        x="date",
        y="prediction_count",
        color="source",
        markers=True,
        title="Prediction logging volume over time",
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Logged prediction events",
        legend_title="Log source",
        title_font=dict(size=20),
    )

    return fig


def build_forecast_health_chart(forecast_future: pd.DataFrame):
    """Create future forecast availability chart."""

    if not PLOTLY_AVAILABLE or forecast_future.empty:
        return None

    required = {"date", "target_name", "predicted_value"}

    if not required.issubset(forecast_future.columns):
        return None

    data = forecast_future.copy()

    if "is_best_model" in data.columns:
        best_rows = data[data["is_best_model"].astype(str).str.lower().isin(["true", "1", "yes"])]

        if not best_rows.empty:
            data = best_rows.copy()

    fig = px.line(
        data,
        x="date",
        y="predicted_value",
        color="target_name",
        markers=True,
        title="Forecast output health: future KPI predictions are available",
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Predicted value",
        legend_title="KPI",
        title_font=dict(size=20),
    )

    return fig


# --------------------------------------------------
# 6. LOAD MONITORING DATA
# --------------------------------------------------

single_log = load_optional_csv(SINGLE_PREDICTION_LOG_PATH)
batch_log = load_optional_csv(BATCH_SCORING_LOG_PATH)

score_file = get_active_score_file()
baseline_scores = load_score_baseline(str(score_file))

anomaly_summary = load_optional_csv(ANOMALY_SUMMARY_PATH)
forecast_future = load_optional_csv(FORECAST_FUTURE_PATH)
daily_kpis = load_optional_csv(DAILY_KPI_PATH)

live_scores = extract_live_prediction_scores(single_log)

champion_model_name = get_champion_model_name()
best_threshold = get_best_threshold()


# --------------------------------------------------
# 7. HEALTH CALCULATIONS
# --------------------------------------------------

baseline_score_summary = get_numeric_summary(
    baseline_scores,
    "purchase_intent_score",
)

live_score_summary = get_numeric_summary(
    live_scores,
    "purchase_intent_score",
)

psi_value = calculate_population_stability_index(
    baseline_scores["purchase_intent_score"] if "purchase_intent_score" in baseline_scores.columns else pd.Series(dtype=float),
    live_scores["purchase_intent_score"] if "purchase_intent_score" in live_scores.columns else pd.Series(dtype=float),
)

psi_status = interpret_psi(psi_value)

live_high_intent_rate = (
    float((live_scores["purchase_intent_score"] >= best_threshold).mean())
    if not live_scores.empty
    else np.nan
)

baseline_high_intent_rate = (
    float((baseline_scores["purchase_intent_score"] >= best_threshold).mean())
    if not baseline_scores.empty
    else np.nan
)

anomaly_rate = get_summary_value(anomaly_summary, "final_anomaly_rate")

baseline_bucket_table = (
    build_score_bucket_table(
        baseline_scores["purchase_intent_score"],
        "Champion baseline",
        best_threshold,
    )
    if "purchase_intent_score" in baseline_scores.columns
    else pd.DataFrame()
)

live_bucket_table = (
    build_score_bucket_table(
        live_scores["purchase_intent_score"],
        "Live predictions",
        best_threshold,
    )
    if "purchase_intent_score" in live_scores.columns
    else pd.DataFrame()
)

bucket_table = pd.concat(
    [baseline_bucket_table, live_bucket_table],
    ignore_index=True,
)


# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 📡 Monitoring")
st.sidebar.caption("Drift and production health")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Baseline score file:** {score_file.name if score_file.exists() else 'Missing'}")
st.sidebar.markdown(f"**PSI status:** {psi_status}")
st.sidebar.markdown("---")
st.sidebar.caption("Tracks prediction logs, score health, drift, anomaly outputs, and forecast availability.")


# --------------------------------------------------
# 9. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="monitor-hero">'
        f'<div class="eyebrow">Production monitoring layer</div>'
        f'<div class="monitor-title">Monitoring, Drift & Health</div>'
        f'<div class="monitor-subtitle">'
        f'This page checks whether the scoring system is being logged, whether score distributions look healthy, '
        f'whether live predictions drift from the champion baseline, and whether supporting anomaly/forecast outputs exist.'
        f'</div><br>'
        f'<span class="success-pill">Champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">PSI status: {escape_text(psi_status)}</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 10. HEALTH KPI CARDS
# --------------------------------------------------

card_1, card_2, card_3, card_4 = st.columns(4)

with card_1:
    render_html(
        (
            f'<div class="health-card">'
            f'<div class="health-label">Live predictions</div>'
            f'<div class="health-value">{format_count(live_score_summary["count"])}</div>'
            f'<div class="health-help">Single-visitor predictions logged in monitoring folder.</div>'
            f'</div>'
        )
    )

with card_2:
    render_html(
        (
            f'<div class="health-card">'
            f'<div class="health-label">Baseline scores</div>'
            f'<div class="health-value">{format_count(baseline_score_summary["count"])}</div>'
            f'<div class="health-help">Sample used to compare production score distribution.</div>'
            f'</div>'
        )
    )

with card_3:
    render_html(
        (
            f'<div class="health-card">'
            f'<div class="health-label">PSI drift status</div>'
            f'<div class="health-value">{escape_text(psi_status)}</div>'
            f'<div class="health-help">PSI value: {format_score_value(psi_value)}</div>'
            f'</div>'
        )
    )

with card_4:
    render_html(
        (
            f'<div class="health-card">'
            f'<div class="health-label">Live high-intent rate</div>'
            f'<div class="health-value">{format_score(live_high_intent_rate)}</div>'
            f'<div class="health-help">Baseline high-intent rate: {format_score(baseline_high_intent_rate)}</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 11. HEALTH STORY
# --------------------------------------------------

render_html(
    '<div class="story-card">'
    '<div class="story-title">How to read this page</div>'
    '<div class="story-text">'
    'A model is not finished when it is trained. A production-style model must be watched. '
    'Here we check logs, score buckets, drift, anomaly output, forecast output, and data availability. '
    'The bucket chart is more useful than a raw histogram because ecommerce data has many cold visitors.'
    '</div>'
    '</div>'
)


# --------------------------------------------------
# 12. DRIFT AND SCORE HEALTH CHARTS
# --------------------------------------------------

render_html('<div class="section-kicker">Score health</div><div class="section-title">Baseline vs live prediction distribution</div>')

if PLOTLY_AVAILABLE:
    bucket_chart = build_score_bucket_chart(bucket_table)

    if bucket_chart is not None:
        st.plotly_chart(bucket_chart, use_container_width=True)
        render_chart_explanation(
            "Chart explanation: baseline versus live score buckets",
            "It compares the normal champion score distribution with live prediction scores across intent buckets.",
            "If live scores move strongly away from baseline, the model may be seeing different visitor behaviour and should be checked for drift.",
        )
    else:
        st.info("Score-bucket chart needs baseline scores or live prediction logs.")

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        volume_chart = build_prediction_volume_chart(
            live_scores=live_scores,
            batch_log=batch_log,
        )

        if volume_chart is not None:
            st.plotly_chart(volume_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: prediction logging volume",
                "It shows how many single and batch predictions are being logged over time.",
                "Healthy production systems should produce consistent logs. Missing or sudden drops can mean the app, logging, or scoring workflow needs attention.",
            )
        else:
            st.info("Prediction volume chart appears after prediction logs exist.")

    with chart_col_2:
        forecast_chart = build_forecast_health_chart(forecast_future)

        if forecast_chart is not None:
            st.plotly_chart(forecast_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: forecast output health",
                "It confirms that future KPI forecasts are available for the monitoring layer.",
                "If forecast outputs are missing or outdated, the planning dashboard is not fully production-ready.",
            )
        else:
            st.info("Forecast health chart appears after forecast output exists.")

else:
    st.info("Plotly is not installed. Charts will appear after installing plotly.")


# --------------------------------------------------
# 13. OPERATIONAL CHECKLIST
# --------------------------------------------------

render_html('<div class="section-kicker">Operational checklist</div><div class="section-title">Files needed for production-style monitoring</div>')

checklist_rows = [
    {
        "Component": "Single prediction log",
        "Path": str(SINGLE_PREDICTION_LOG_PATH),
        "Status": check_file_status(SINGLE_PREDICTION_LOG_PATH),
    },
    {
        "Component": "Batch scoring log",
        "Path": str(BATCH_SCORING_LOG_PATH),
        "Status": check_file_status(BATCH_SCORING_LOG_PATH),
    },
    {
        "Component": "Champion visitor scores",
        "Path": str(score_file),
        "Status": check_file_status(score_file),
    },
    {
        "Component": "Anomaly summary",
        "Path": str(ANOMALY_SUMMARY_PATH),
        "Status": check_file_status(ANOMALY_SUMMARY_PATH),
    },
    {
        "Component": "Forecast future output",
        "Path": str(FORECAST_FUTURE_PATH),
        "Status": check_file_status(FORECAST_FUTURE_PATH),
    },
    {
        "Component": "Daily KPI table",
        "Path": str(DAILY_KPI_PATH),
        "Status": check_file_status(DAILY_KPI_PATH),
    },
]

checklist = pd.DataFrame(checklist_rows)

st.dataframe(
    checklist,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 14. RAW EVIDENCE TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Raw evidence</div><div class="section-title">Monitoring data used by this page</div>')

tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    [
        "Live scores",
        "Baseline scores",
        "Batch logs",
        "Anomaly summary",
        "Forecast output",
    ]
)

with tab_1:
    if live_scores.empty:
        st.info("No live prediction logs yet. Use the Visitor Intent Predictor page to create logs.")
    else:
        st.dataframe(
            live_scores,
            use_container_width=True,
            hide_index=True,
        )

with tab_2:
    if baseline_scores.empty:
        st.info("Champion score file is missing or does not contain purchase_intent_score.")
    else:
        st.dataframe(
            baseline_scores.head(1000),
            use_container_width=True,
            hide_index=True,
        )

with tab_3:
    if batch_log.empty:
        st.info("No batch scoring log yet. Use the Batch Scoring page to create logs.")
    else:
        st.dataframe(
            batch_log,
            use_container_width=True,
            hide_index=True,
        )

with tab_4:
    if anomaly_summary.empty:
        st.info("Anomaly summary is missing.")
    else:
        st.dataframe(
            anomaly_summary,
            use_container_width=True,
            hide_index=True,
        )

with tab_5:
    if forecast_future.empty:
        st.info("Forecast future output is missing.")
    else:
        st.dataframe(
            forecast_future,
            use_container_width=True,
            hide_index=True,
        )
