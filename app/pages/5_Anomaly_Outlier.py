# 5_Anomaly_Outlier.py
# Streamlit page for anomaly and outlier review.
#
# Purpose:
#   Show which visitors behave unusually and need review before campaign activation.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Paths and labels
#   4. Helper functions
#   5. Chart functions
#   6. Load anomaly outputs
#   7. Sidebar and header
#   8. KPI cards
#   9. Charts
#   10. Filterable review table
#   11. Evidence tables

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
    get_best_threshold,
    get_champion_model_name,
    inject_global_css,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Anomaly & Outlier Detection",
    page_icon="🚨",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .anomaly-hero {
            padding: 34px 38px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(248, 113, 113, 0.24), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(56, 189, 248, 0.20), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .anomaly-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .anomaly-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1050px;
        }

        .anomaly-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(248, 113, 113, 0.15), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 184px;
            margin-bottom: 18px;
        }

        .anomaly-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .anomaly-value {
            color: #F8FAFC;
            font-size: 2.20rem;
            line-height: 1;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .anomaly-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.48;
        }

        .section-kicker {
            color: #FCA5A5;
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
# 3. PATHS AND LABELS
# --------------------------------------------------

ANOMALY_SUMMARY_PATH = Path("reports/tables/anomaly_summary.csv")
ANOMALY_RULE_SUMMARY_PATH = Path("reports/tables/anomaly_rule_summary.csv")
ANOMALY_SEGMENT_SUMMARY_PATH = Path("reports/tables/anomaly_segment_summary.csv")
TOP_ANOMALOUS_VISITORS_PATH = Path("reports/tables/top_anomalous_visitors.csv")
MVD_OUTLIER_COMPARISON_PATH = Path("reports/tables/mvd_outlier_method_comparison.csv")

RULE_LABELS: Dict[str, str] = {
    "rule_extreme_total_events": "Extreme activity volume",
    "rule_extreme_cart_activity": "Extreme cart activity",
    "rule_extreme_product_browsing": "Extreme product browsing",
    "rule_extreme_session_span": "Extreme session span",
    "rule_high_intent_extreme_activity": "High intent + extreme behaviour",
    "rule_unusual_cart_to_view_ratio": "Unusual cart-to-view ratio",
}

SEGMENT_ORDER = [
    "High Intent",
    "Strong Intent",
    "Warm Intent",
    "Low Intent",
    "Cold Visitor",
]

RISK_BAND_ORDER = [
    "Critical Review",
    "High Review",
    "Medium Review",
    "Normal",
]


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_summary_value(summary: pd.DataFrame, metric: str) -> float:
    """Read one metric from anomaly_summary.csv."""

    if summary.empty:
        return np.nan

    if {"metric", "value"}.issubset(summary.columns):
        metric_row = summary[summary["metric"].astype(str) == metric]

        if not metric_row.empty:
            return float(metric_row.iloc[0]["value"])

    if metric in summary.columns:
        return float(summary.iloc[0][metric])

    return np.nan


def format_count(value: float) -> str:
    """Format visitor counts for dashboard cards."""

    if pd.isna(value):
        return "NA"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_rate(value: float) -> str:
    """Format a decimal rate as a percentage."""

    if pd.isna(value):
        return "NA"

    return f"{float(value) * 100:.2f}%"


def prepare_rule_summary(rule_summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare readable anomaly rule summary."""

    if rule_summary.empty:
        return rule_summary

    data = rule_summary.copy()

    if "rule_name" not in data.columns:
        return pd.DataFrame()

    data["Rule"] = data["rule_name"].map(RULE_LABELS).fillna(data["rule_name"])

    if "flagged_count" in data.columns:
        data["Flagged Visitors"] = data["flagged_count"]

    if "flagged_rate" in data.columns:
        data["Flagged Rate"] = data["flagged_rate"]

    return data


def prepare_segment_summary(segment_summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare intent-segment anomaly summary."""

    if segment_summary.empty:
        return segment_summary

    data = segment_summary.copy()

    segment_column = get_first_existing_column(
        data,
        ["purchase_intent_segment", "intent_segment", "segment"],
    )

    if segment_column is None:
        return data

    data[segment_column] = pd.Categorical(
        data[segment_column],
        categories=SEGMENT_ORDER,
        ordered=True,
    )

    data = data.sort_values(segment_column).reset_index(drop=True)

    return data


def get_first_existing_column(data: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return first column that exists in a table."""

    for column in candidates:
        if column in data.columns:
            return column

    return None


def build_hover_data(data: pd.DataFrame, columns: List[str]) -> Dict:
    """Build Plotly hover_data only with columns that exist."""

    return {
        column: True
        for column in columns
        if column in data.columns
    }




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

def build_rule_chart(rule_summary: pd.DataFrame):
    """Create anomaly-rule bar chart."""

    if not PLOTLY_AVAILABLE or rule_summary.empty:
        return None

    data = prepare_rule_summary(rule_summary)

    if data.empty or "Flagged Visitors" not in data.columns:
        return None

    data = data.sort_values("Flagged Visitors", ascending=True)

    hover_data = {}

    if "Flagged Rate" in data.columns:
        hover_data["Flagged Rate"] = ":.3%"

    fig = px.bar(
        data,
        x="Flagged Visitors",
        y="Rule",
        orientation="h",
        text="Flagged Visitors",
        title="Explainable anomaly rules: visitors flagged",
        hover_data=hover_data,
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Flagged visitors",
        yaxis_title="",
        showlegend=False,
        title_font=dict(size=20),
    )

    return fig


def build_segment_anomaly_chart(segment_summary: pd.DataFrame):
    """Create anomaly-rate chart by intent segment."""

    if not PLOTLY_AVAILABLE or segment_summary.empty:
        return None

    data = prepare_segment_summary(segment_summary)

    segment_column = get_first_existing_column(
        data,
        ["purchase_intent_segment", "intent_segment", "segment"],
    )

    anomaly_rate_column = get_first_existing_column(
        data,
        ["anomaly_rate", "final_anomaly_rate"],
    )

    if segment_column is None or anomaly_rate_column is None:
        return None

    hover_data = build_hover_data(
        data,
        [
            "visitors",
            "final_anomalies",
            "avg_purchase_intent_score",
            "avg_anomaly_risk_score",
        ],
    )

    fig = px.bar(
        data,
        x=segment_column,
        y=anomaly_rate_column,
        text=anomaly_rate_column,
        title="Anomaly rate by purchase-intent segment",
        hover_data=hover_data,
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Purchase-intent segment",
        yaxis_title="Anomaly rate",
        showlegend=False,
        title_font=dict(size=20),
    )

    fig.update_yaxes(tickformat=".1%")

    return fig


def build_risk_score_scatter(top_anomalies: pd.DataFrame):
    """Create purchase-intent vs anomaly-risk scatter chart."""

    if not PLOTLY_AVAILABLE or top_anomalies.empty:
        return None

    required = {"purchase_intent_score", "anomaly_risk_score"}

    if not required.issubset(top_anomalies.columns):
        return None

    data = top_anomalies.head(2500).copy()

    color_column = (
        "anomaly_risk_band"
        if "anomaly_risk_band" in data.columns
        else None
    )

    size_column = (
        "rule_anomaly_count"
        if "rule_anomaly_count" in data.columns
        else None
    )

    hover_data = build_hover_data(
        data,
        [
            "purchase_intent_segment",
            "total_events",
            "addtocart_count",
            "unique_items",
            "business_interpretation",
        ],
    )

    fig = px.scatter(
        data,
        x="purchase_intent_score",
        y="anomaly_risk_score",
        color=color_column,
        size=size_column,
        hover_name="visitorid" if "visitorid" in data.columns else None,
        hover_data=hover_data,
        title="Visitor risk map: purchase intent vs anomaly risk",
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Purchase intent score",
        yaxis_title="Anomaly risk score",
        legend_title="Risk band",
        title_font=dict(size=20),
    )

    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")

    return fig


def build_top_anomaly_bar(top_anomalies: pd.DataFrame, limit: int = 20):
    """Create chart for highest-risk anomalous visitors."""

    if not PLOTLY_AVAILABLE or top_anomalies.empty:
        return None

    required = {"visitorid", "anomaly_risk_score"}

    if not required.issubset(top_anomalies.columns):
        return None

    data = top_anomalies.head(limit).copy()
    data["visitorid"] = data["visitorid"].astype(str)

    color_column = (
        "anomaly_risk_band"
        if "anomaly_risk_band" in data.columns
        else None
    )

    hover_data = build_hover_data(
        data,
        [
            "purchase_intent_score",
            "purchase_intent_segment",
            "rule_anomaly_count",
            "business_interpretation",
        ],
    )

    fig = px.bar(
        data.sort_values("anomaly_risk_score", ascending=True),
        x="anomaly_risk_score",
        y="visitorid",
        orientation="h",
        color=color_column,
        hover_data=hover_data,
        title=f"Top {min(limit, len(data))} anomalous visitors",
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Anomaly risk score",
        yaxis_title="Visitor ID",
        legend_title="Risk band",
        title_font=dict(size=20),
    )

    fig.update_xaxes(tickformat=".0%")

    return fig


# --------------------------------------------------
# 6. LOAD ANOMALY OUTPUTS
# --------------------------------------------------

summary = load_csv(ANOMALY_SUMMARY_PATH)
rule_summary = load_csv(ANOMALY_RULE_SUMMARY_PATH)
segment_summary = load_csv(ANOMALY_SEGMENT_SUMMARY_PATH)
top_anomalies = load_csv(TOP_ANOMALOUS_VISITORS_PATH)
mvd_outlier_comparison = load_csv(MVD_OUTLIER_COMPARISON_PATH)

champion_model_name = get_champion_model_name()
best_threshold = get_best_threshold()


# --------------------------------------------------
# 7. MISSING FILE CHECK
# --------------------------------------------------

required_paths = [
    ANOMALY_SUMMARY_PATH,
    ANOMALY_RULE_SUMMARY_PATH,
    ANOMALY_SEGMENT_SUMMARY_PATH,
    TOP_ANOMALOUS_VISITORS_PATH,
]

missing_files = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing_files:
    render_html(
        '<div class="anomaly-hero">'
        '<div class="eyebrow">Anomaly & Outlier Detection</div>'
        '<div class="anomaly-title">Anomaly files are missing</div>'
        '<div class="anomaly-subtitle">Run the anomaly pipeline first, then refresh this page.</div>'
        '</div>'
    )

    st.code("python3 -m src.anomaly.build_anomaly_signals", language="bash")

    st.warning("Missing files:")
    for file_path in missing_files:
        st.write(f"- {file_path}")

    st.stop()


# --------------------------------------------------
# 8. SUMMARY METRICS
# --------------------------------------------------

total_visitors = get_summary_value(summary, "total_visitors")
final_anomalies = get_summary_value(summary, "final_anomalies")
final_anomaly_rate = get_summary_value(summary, "final_anomaly_rate")
isolation_anomalies = get_summary_value(summary, "isolation_forest_anomalies")
rule_based_anomalies = get_summary_value(summary, "rule_based_anomalies")
high_intent_anomalies = get_summary_value(summary, "high_intent_anomalies")


# --------------------------------------------------
# 9. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🚨 Anomaly Detection")
st.sidebar.caption("Unusual visitor behaviour")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Intent threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Visitors analysed:** {format_count(total_visitors)}")
st.sidebar.markdown(f"**Final anomalies:** {format_count(final_anomalies)}")
st.sidebar.markdown(f"**Anomaly rate:** {format_rate(final_anomaly_rate)}")
st.sidebar.markdown("---")
st.sidebar.caption("Rules + Isolation Forest anomaly detector.")


# --------------------------------------------------
# 10. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="anomaly-hero">'
        f'<div class="eyebrow">Anomaly & Outlier Detection</div>'
        f'<div class="anomaly-title">Detect unusual visitors before campaign activation</div>'
        f'<div class="anomaly-subtitle">'
        f'The purchase model says who is likely to buy. The anomaly layer checks whether the behaviour looks unusual. '
        f'This is a review layer, not a blind deletion rule.'
        f'</div><br>'
        f'<span class="success-pill">Visitors analysed: {format_count(total_visitors)}</span>'
        f'&nbsp;<span class="info-pill">Final anomaly rate: {format_rate(final_anomaly_rate)}</span>'
        f'&nbsp;<span class="warning-pill">High-intent anomalies: {format_count(high_intent_anomalies)}</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 11. KPI CARDS
# --------------------------------------------------

card_1, card_2, card_3, card_4 = st.columns(4)

with card_1:
    render_html(
        (
            f'<div class="anomaly-card">'
            f'<div class="anomaly-label">Visitors analysed</div>'
            f'<div class="anomaly-value">{format_count(total_visitors)}</div>'
            f'<div class="anomaly-help">Visitor-level records checked for unusual behaviour.</div>'
            f'</div>'
        )
    )

with card_2:
    render_html(
        (
            f'<div class="anomaly-card">'
            f'<div class="anomaly-label">Final anomalies</div>'
            f'<div class="anomaly-value">{format_count(final_anomalies)}</div>'
            f'<div class="anomaly-help">{format_rate(final_anomaly_rate)} of visitors flagged for review.</div>'
            f'</div>'
        )
    )

with card_3:
    render_html(
        (
            f'<div class="anomaly-card">'
            f'<div class="anomaly-label">Rule-based anomalies</div>'
            f'<div class="anomaly-value">{format_count(rule_based_anomalies)}</div>'
            f'<div class="anomaly-help">Explainable business rules triggered.</div>'
            f'</div>'
        )
    )

with card_4:
    render_html(
        (
            f'<div class="anomaly-card">'
            f'<div class="anomaly-label">High-intent anomalies</div>'
            f'<div class="anomaly-value">{format_count(high_intent_anomalies)}</div>'
            f'<div class="anomaly-help">Strong purchase intent but unusual behaviour.</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 12. BUSINESS EXPLANATION
# --------------------------------------------------

render_html(
    '<div class="story-card">'
    '<div class="story-title">Why this page matters</div>'
    '<div class="story-text">'
    'A high purchase score does not always mean “spend money immediately.” '
    'If behaviour is extreme or unusual, the visitor should be reviewed before campaign activation. '
    'This protects marketing budget and makes the model safer for production use.'
    '</div>'
    '</div>'
)


# --------------------------------------------------
# 13. CHARTS
# --------------------------------------------------

render_html('<div class="section-kicker">Anomaly overview</div><div class="section-title">Rules, segments, and visitor risk map</div>')

if not PLOTLY_AVAILABLE:
    st.warning("Plotly is not installed. Install plotly to show anomaly charts.")

else:
    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        rule_chart = build_rule_chart(rule_summary)

        if rule_chart is not None:
            st.plotly_chart(rule_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: anomaly rules",
                "It shows how many visitors were flagged by each explainable anomaly rule.",
                "Rules with many flagged visitors point to common unusual behaviours. These visitors should be reviewed, not automatically deleted.",
            )
        else:
            st.info("Rule chart is not available for the current table format.")

    with chart_col_2:
        segment_chart = build_segment_anomaly_chart(segment_summary)

        if segment_chart is not None:
            st.plotly_chart(segment_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: anomaly rate by intent segment",
                "It shows which purchase-intent segments contain more unusual visitors.",
                "High-intent visitors with high anomaly rates are important because they may be valuable buyers, suspicious activity, or edge cases needing business review.",
            )
        else:
            st.info("Segment anomaly chart is not available for the current table format.")

    risk_col_1, risk_col_2 = st.columns([1.1, 0.9])

    with risk_col_1:
        risk_scatter = build_risk_score_scatter(top_anomalies)

        if risk_scatter is not None:
            st.plotly_chart(risk_scatter, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: visitor risk map",
                "It plots purchase-intent score against anomaly risk so we can see valuable visitors and risky visitors together.",
                "The best business action is not simply target or remove. High-intent and high-risk visitors should be reviewed before campaign activation.",
            )
        else:
            st.info("Risk scatter chart is not available for the current table format.")

    with risk_col_2:
        top_anomaly_chart = build_top_anomaly_bar(top_anomalies, limit=20)

        if top_anomaly_chart is not None:
            st.plotly_chart(top_anomaly_chart, use_container_width=True)
            render_chart_explanation(
                "Chart explanation: top anomalous visitors",
                "It ranks the most unusual visitors by anomaly risk or anomaly score.",
                "These are priority review cases. They help analysts understand extreme behaviour and protect campaign quality.",
            )
        else:
            st.info("Top anomaly chart is not available for the current table format.")


# --------------------------------------------------
# 14. FILTERABLE REVIEW LIST
# --------------------------------------------------

render_html('<div class="section-kicker">Review list</div><div class="section-title">Top anomalous visitors</div>')

if top_anomalies.empty:
    st.info("No top anomaly table found.")

else:
    filter_col_1, filter_col_2 = st.columns([0.5, 0.5])

    with filter_col_1:
        risk_band_column = get_first_existing_column(
            top_anomalies,
            ["anomaly_risk_band", "risk_band"],
        )

        if risk_band_column:
            risk_band_options = [
                band
                for band in RISK_BAND_ORDER
                if band in top_anomalies[risk_band_column].dropna().unique().tolist()
            ]

            if not risk_band_options:
                risk_band_options = sorted(top_anomalies[risk_band_column].dropna().astype(str).unique().tolist())

            selected_risk_bands = st.multiselect(
                "Filter by risk band",
                options=risk_band_options,
                default=risk_band_options,
            )
        else:
            selected_risk_bands = []

    with filter_col_2:
        segment_column = get_first_existing_column(
            top_anomalies,
            ["purchase_intent_segment", "intent_segment", "segment"],
        )

        if segment_column:
            segment_options = [
                segment
                for segment in SEGMENT_ORDER
                if segment in top_anomalies[segment_column].dropna().unique().tolist()
            ]

            if not segment_options:
                segment_options = sorted(top_anomalies[segment_column].dropna().astype(str).unique().tolist())

            selected_segments = st.multiselect(
                "Filter by purchase-intent segment",
                options=segment_options,
                default=segment_options,
            )
        else:
            selected_segments = []

    filtered_anomalies = top_anomalies.copy()

    if risk_band_column and selected_risk_bands:
        filtered_anomalies = filtered_anomalies[
            filtered_anomalies[risk_band_column].isin(selected_risk_bands)
        ]

    if segment_column and selected_segments:
        filtered_anomalies = filtered_anomalies[
            filtered_anomalies[segment_column].isin(selected_segments)
        ]

    display_columns = [
        "visitorid",
        "purchase_intent_score",
        "purchase_intent_segment",
        "anomaly_risk_score",
        "anomaly_risk_band",
        "rule_anomaly_count",
        "total_events",
        "addtocart_count",
        "unique_items",
        "cart_to_view_ratio",
        "business_interpretation",
    ]

    available_columns = [
        column for column in display_columns
        if column in filtered_anomalies.columns
    ]

    st.dataframe(
        filtered_anomalies[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download filtered anomaly review list",
        data=filtered_anomalies.to_csv(index=False),
        file_name="filtered_anomaly_review_list.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------
# 15. EVIDENCE TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Evidence tables</div><div class="section-title">Anomaly summaries</div>')

tab_labels = [
    "Summary",
    "Rule summary",
    "Segment summary",
    "Top anomalies",
    "MVD outlier comparison",
]

tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(tab_labels)

with tab_1:
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

with tab_2:
    readable_rules = prepare_rule_summary(rule_summary)

    if readable_rules.empty:
        st.info("No rule summary available.")
    else:
        st.dataframe(
            readable_rules,
            use_container_width=True,
            hide_index=True,
        )

with tab_3:
    readable_segments = prepare_segment_summary(segment_summary)

    if readable_segments.empty:
        st.info("No segment summary available.")
    else:
        st.dataframe(
            readable_segments,
            use_container_width=True,
            hide_index=True,
        )

with tab_4:
    st.dataframe(
        top_anomalies,
        use_container_width=True,
        hide_index=True,
    )

with tab_5:
    if mvd_outlier_comparison.empty:
        st.info("MVD outlier comparison table is not available yet.")
    else:
        st.dataframe(
            mvd_outlier_comparison,
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------
# FINAL CLOSURE EXTENSION: ANOMALY INVESTIGATION INTELLIGENCE
# --------------------------------------------------

from app.ui.operations_intelligence import (
    render_anomaly_investigation_intelligence,
)

render_anomaly_investigation_intelligence()
