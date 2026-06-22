# Executive_Overview.py
# Landing page for the E-Commerce Conversion Intelligence Platform.
#
# Purpose:
#   Give an executive-level overview of the project.
#
# This page shows:
#   1. Final champion model impact
#   2. Random targeting vs model targeting
#   3. Final champion proof
#   4. Platform modules
#   5. Honest project scope

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from app.app_utils import (
    create_master_model_comparison_table,
    escape_text,
    format_lift,
    format_percent,
    get_best_threshold,
    get_champion_metrics,
    get_champion_model_name,
    get_champion_track,
    inject_global_css,
    load_model_selection_tables,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Conversion Intelligence",
    page_icon="🧠",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------
# Keep page-specific CSS here.
# Full visual polish will be handled at the final dashboard polish stage.

render_html(
    """
    <style>
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.2rem;
        }

        .dashboard-hero {
            padding: 34px 38px;
            border-radius: 34px;
            background:
                radial-gradient(circle at 12% 12%, rgba(56, 189, 248, 0.30), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(34, 197, 94, 0.22), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.50);
            margin-bottom: 20px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.35fr 0.65fr;
            gap: 22px;
            align-items: center;
        }

        .hero-eyebrow {
            color: #38BDF8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero-main-title {
            color: #F8FAFC;
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 950;
            margin-bottom: 12px;
        }

        .hero-one-liner {
            color: #CBD5E1;
            font-size: 1.02rem;
            line-height: 1.55;
            max-width: 980px;
        }

        .hero-score-box {
            padding: 24px 24px;
            border-radius: 28px;
            background: rgba(2, 6, 23, 0.48);
            border: 1px solid rgba(125, 211, 252, 0.22);
            text-align: center;
        }

        .hero-score-label {
            color: #94A3B8;
            font-size: 0.75rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-score-value {
            color: #F8FAFC;
            font-size: 3.25rem;
            font-weight: 950;
            line-height: 1;
            margin-bottom: 8px;
        }

        .hero-score-help {
            color: #A7F3D0;
            font-size: 0.86rem;
            font-weight: 800;
        }

        .pill-row {
            margin-top: 18px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .glass-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #E2E8F0;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .visual-card {
            padding: 24px 24px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.88));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 176px;
            margin-bottom: 18px;
        }

        .visual-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .visual-value {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .visual-caption {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .section-kicker {
            color: #7DD3FC;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-top: 18px;
            margin-bottom: 4px;
        }

        .section-title {
            color: #F8FAFC;
            font-size: 1.55rem;
            font-weight: 950;
            margin-bottom: 14px;
        }

        .story-card {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.14), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            margin-bottom: 18px;
        }

        .story-big {
            color: #F8FAFC;
            font-size: 1.25rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .story-small {
            color: #CBD5E1;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .flow-card {
            padding: 22px 22px;
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 16px 46px rgba(0, 0, 0, 0.24);
            min-height: 152px;
            margin-bottom: 16px;
        }

        .flow-icon {
            font-size: 1.58rem;
            margin-bottom: 8px;
        }

        .flow-title {
            color: #F8FAFC;
            font-size: 1.02rem;
            font-weight: 900;
            margin-bottom: 7px;
        }

        .flow-text {
            color: #CBD5E1;
            font-size: 0.85rem;
            line-height: 1.45;
        }

        @media (max-width: 980px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-main-title {
                font-size: 2rem;
            }
        }
    </style>
    """
)


# --------------------------------------------------
# 3. SMALL HELPER FUNCTIONS
# --------------------------------------------------

def load_optional_csv(path: str) -> pd.DataFrame:
    """Load a CSV if it exists."""

    csv_path = Path(path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def readable_track_name(track_name: str) -> str:
    """Convert internal track names into readable labels."""

    track_lower = str(track_name).lower()

    if "final" in track_lower:
        return "Final True Champion"

    if "manual" in track_lower:
        return "Manual Track"

    if "automl" in track_lower:
        return "AutoML-style Track"

    return str(track_name)


def get_natural_conversion_rate() -> float:
    """Return the labeled final-holdout buyer rate used for lift.

    The production visitor table intentionally has no outcome label.
    Therefore, lift must use the untouched labeled evaluation holdout,
    not the current production scoring population.
    """

    holdout = load_optional_csv(
        "reports/tables/final_true_champion_holdout.csv"
    )

    if holdout.empty:
        raise RuntimeError(
            "Final holdout results are unavailable. "
            "Run: python3 -m src.models.finalize_true_champion"
        )

    if "positive_rate" in holdout.columns:
        rates = pd.to_numeric(
            holdout["positive_rate"],
            errors="coerce",
        ).dropna()

        if not rates.empty and 0 < float(rates.iloc[0]) <= 1:
            return float(rates.iloc[0])

    required = {"rows", "positive_rows"}

    if required.issubset(holdout.columns):
        rows = float(holdout.iloc[0]["rows"])
        positives = float(holdout.iloc[0]["positive_rows"])

        if rows > 0 and 0 <= positives <= rows:
            return positives / rows

    raise RuntimeError(
        "Final holdout results do not contain a valid labeled buyer rate."
    )

def get_final_champion_summary_row() -> pd.Series:
    """Read final champion summary if available."""

    tables = load_model_selection_tables()
    final_summary = tables.get("final_true_champion_summary", pd.DataFrame())

    if final_summary.empty:
        return pd.Series(dtype="object")

    return final_summary.iloc[0]


def get_final_threshold_row(model_name: str) -> pd.Series:
    """Read threshold row for the active final champion."""

    tables = load_model_selection_tables()
    thresholds = tables.get("final_true_champion_thresholds", pd.DataFrame())

    if thresholds.empty:
        return pd.Series(dtype="object")

    if "model_name" in thresholds.columns:
        thresholds = thresholds[thresholds["model_name"] == model_name].copy()

    if thresholds.empty:
        return pd.Series(dtype="object")

    threshold = get_best_threshold()

    threshold_rows = thresholds[
        thresholds["threshold"].round(4) == round(threshold, 4)
    ]

    if not threshold_rows.empty:
        return threshold_rows.iloc[0]

    return thresholds.sort_values("f1_score", ascending=False).iloc[0]


def metric_value(
    row: pd.Series,
    column: str,
    fallback: float,
) -> float:
    """Safely read numeric metric value."""

    if row.empty or column not in row.index:
        return float(fallback)

    value = row[column]

    if pd.isna(value):
        return float(fallback)

    return float(value)


def get_top_models(limit: int = 8) -> pd.DataFrame:
    """Return best model rows for benchmark visuals."""

    comparison = create_master_model_comparison_table()

    if comparison.empty:
        return comparison

    comparison = comparison.copy()

    if "track" not in comparison.columns:
        comparison["track"] = "final_true_champion"

    comparison["Track Label"] = comparison["track"].apply(readable_track_name)

    top_models = comparison.head(limit).copy()

    top_models["Model Display"] = (
        top_models["model_name"].astype(str)
        + " · "
        + top_models["Track Label"].astype(str)
    )

    top_models["Champion"] = top_models["is_final_champion"].map(
        {
            True: "Final champion",
            False: "Compared model",
        }
    )

    return top_models


# --------------------------------------------------
# 4. CHART FUNCTIONS
# --------------------------------------------------

def build_bar_chart(
    random_buyers_per_100: float,
    model_buyers_per_100: float,
):
    """Create random vs model targeting buyer-yield chart."""

    chart_data = pd.DataFrame(
        {
            "Targeting Method": [
                "Random targeting",
                "Final champion targeting",
            ],
            "Expected Buyers per 100 Targeted": [
                random_buyers_per_100,
                model_buyers_per_100,
            ],
        }
    )

    fig = px.bar(
        chart_data,
        x="Targeting Method",
        y="Expected Buyers per 100 Targeted",
        text="Expected Buyers per 100 Targeted",
        title="Buyer yield: random targeting vs final champion targeting",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=58, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=20),
        xaxis_title="",
        yaxis_title="Expected buyers",
        showlegend=False,
    )

    return fig


def build_threshold_gauge(best_threshold: float):
    """Create production threshold gauge."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=best_threshold * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Final production score gate", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.24},
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.85,
                    "value": best_threshold * 100,
                },
                "steps": [
                    {"range": [0, 50], "color": "rgba(30, 41, 59, 0.75)"},
                    {"range": [50, 80], "color": "rgba(14, 165, 233, 0.18)"},
                    {"range": [80, 95], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [95, 100], "color": "rgba(34, 197, 94, 0.25)"},
                ],
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=58, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def build_model_benchmark_chart(top_models: pd.DataFrame):
    """Create model benchmark PR-AUC chart."""

    if top_models.empty:
        return None

    chart_data = top_models.sort_values("pr_auc", ascending=True).copy()

    fig = px.bar(
        chart_data,
        x="pr_auc",
        y="Model Display",
        color="Champion",
        orientation="h",
        hover_data={
            "best_f1_score": ":.3f",
            "best_precision": ":.3f",
            "best_recall": ":.3f",
            "business_score": ":.3f" if "business_score" in chart_data.columns else False,
            "Model Display": False,
        },
        title="Final champion benchmark: rare-buyer ranking strength",
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="PR-AUC",
        yaxis_title="",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


def build_precision_recall_chart(top_models: pd.DataFrame):
    """Create precision vs recall chart for campaign trade-off."""

    if top_models.empty:
        return None

    size_column = "business_score" if "business_score" in top_models.columns else "best_f1_score"

    fig = px.scatter(
        top_models,
        x="best_recall",
        y="best_precision",
        size=size_column,
        color="Champion",
        hover_name="Model Display",
        hover_data={
            "pr_auc": ":.3f",
            "best_f1_score": ":.3f",
            "best_threshold": ":.2f",
        },
        title="Campaign trade-off: buyer capture vs target quality",
    )

    fig.update_traces(
        marker=dict(
            opacity=0.88,
            line=dict(width=1),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Recall: buyers captured",
        yaxis_title="Precision: target quality",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


# --------------------------------------------------
# 5. LOAD PROJECT METRICS
# --------------------------------------------------

metadata_metrics = get_champion_metrics()
final_summary_row = get_final_champion_summary_row()

champion_model_name = get_champion_model_name()
champion_track = get_champion_track()
best_threshold = get_best_threshold()

natural_conversion_rate = get_natural_conversion_rate()

pr_auc = metric_value(final_summary_row, "pr_auc", float(metadata_metrics.get("pr_auc", 0)))
roc_auc = metric_value(final_summary_row, "roc_auc", float(metadata_metrics.get("roc_auc", 0)))
best_precision = metric_value(final_summary_row, "best_precision", float(metadata_metrics.get("best_precision", 0)))
best_recall = metric_value(final_summary_row, "best_recall", float(metadata_metrics.get("best_recall", 0)))
best_f1 = metric_value(final_summary_row, "best_f1_score", float(metadata_metrics.get("best_f1_score", 0)))

threshold_row = get_final_threshold_row(champion_model_name)

targeted_share = metric_value(
    threshold_row,
    "predicted_positive_rate",
    0.0176,
)

random_buyers_per_100 = natural_conversion_rate * 100
model_buyers_per_100_targeted = best_precision * 100
best_lift = best_precision / natural_conversion_rate if natural_conversion_rate > 0 else 0

targeted_visitors_per_10000 = targeted_share * 10_000
captured_buyers_per_10000 = targeted_visitors_per_10000 * best_precision

top_models = get_top_models(limit=8)


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🧠 Conversion Intelligence")
st.sidebar.caption("Executive dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("**Owner:** Ruturaj Mokashi")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**Lift:** {format_lift(best_lift)}")
st.sidebar.markdown("---")
st.sidebar.caption("Powered by final champion model metadata.")


# --------------------------------------------------
# 7. HERO SECTION
# --------------------------------------------------

render_html(
    '<div class="dashboard-hero">'
    '<div class="hero-grid">'
    '<div>'
    '<div class="hero-eyebrow">E-Commerce Conversion Intelligence Platform</div>'
    '<div class="hero-main-title">Find the buyers before marketing money is wasted.</div>'
    '<div class="hero-one-liner">'
    'A production-style ML + BI + MLOps platform for ranking visitors, scoring campaign audiences, '
    'forecasting demand, detecting anomalies, and monitoring prediction health.'
    '</div>'
    '<div class="pill-row">'
    f'<span class="glass-pill">Final champion: {escape_text(champion_model_name)}</span>'
    f'<span class="glass-pill">Threshold: {best_threshold:.2f}</span>'
    f'<span class="glass-pill">Lift: {format_lift(best_lift)} vs random</span>'
    f'<span class="glass-pill">Track: {escape_text(readable_track_name(champion_track))}</span>'
    '</div>'
    '</div>'
    '<div class="hero-score-box">'
    '<div class="hero-score-label">Model-targeted buyers</div>'
    f'<div class="hero-score-value">{model_buyers_per_100_targeted:.1f}</div>'
    '<div class="hero-score-help">buyers per 100 targeted visitors</div>'
    '</div>'
    '</div>'
    '</div>'
)


# --------------------------------------------------
# 8. IMPACT KPI CARDS
# --------------------------------------------------

impact_1, impact_2, impact_3, impact_4 = st.columns(4)

with impact_1:
    render_html(
        '<div class="visual-card">'
        '<div class="visual-label">Random targeting</div>'
        f'<div class="visual-value">{random_buyers_per_100:.1f}</div>'
        '<div class="visual-caption">buyers per 100 random visitors</div>'
        '</div>'
    )

with impact_2:
    render_html(
        '<div class="visual-card">'
        '<div class="visual-label">Final champion targeting</div>'
        f'<div class="visual-value">{model_buyers_per_100_targeted:.1f}</div>'
        '<div class="visual-caption">buyers per 100 model-targeted visitors</div>'
        '</div>'
    )

with impact_3:
    render_html(
        '<div class="visual-card">'
        '<div class="visual-label">Audience shortlist</div>'
        f'<div class="visual-value">{targeted_share * 100:.1f}%</div>'
        '<div class="visual-caption">visitors pass the final threshold</div>'
        '</div>'
    )

with impact_4:
    render_html(
        '<div class="visual-card">'
        '<div class="visual-label">Business lift</div>'
        f'<div class="visual-value">{format_lift(best_lift)}</div>'
        '<div class="visual-caption">target quality versus random marketing</div>'
        '</div>'
    )


# --------------------------------------------------
# 9. VISUAL IMPACT CHARTS
# --------------------------------------------------

render_html('<div class="section-kicker">Visual impact story</div><div class="section-title">From random traffic to high-intent audience</div>')

chart_col_1, chart_col_2 = st.columns([1.1, 0.9])

if PLOTLY_AVAILABLE:
    with chart_col_1:
        st.plotly_chart(
            build_bar_chart(
                random_buyers_per_100=random_buyers_per_100,
                model_buyers_per_100=model_buyers_per_100_targeted,
            ),
            use_container_width=True,
        )

    with chart_col_2:
        st.plotly_chart(
            build_threshold_gauge(best_threshold=best_threshold),
            use_container_width=True,
        )
else:
    with chart_col_1:
        st.info("Plotly is not installed. Install plotly to show executive charts.")
    with chart_col_2:
        st.metric("Production threshold", f"{best_threshold:.2f}")


# --------------------------------------------------
# 10. BUSINESS INTERPRETATION
# --------------------------------------------------

story_left, story_right = st.columns([0.58, 0.42])

with story_left:
    render_html(
        '<div class="story-card">'
        '<div class="story-big">Decision rule: target only the strongest intent signals.</div>'
        '<div class="story-small">'
        f'At a <b>{best_threshold:.2f}</b> threshold, the platform selects about '
        f'<b>{targeted_visitors_per_10000:.0f}</b> visitors from every 10,000 visitors. '
        f'That selected group contains about <b>{captured_buyers_per_10000:.0f}</b> expected buyers. '
        'This turns noisy traffic into a focused campaign audience.'
        '</div>'
        '</div>'
    )

with story_right:
    render_html(
        '<div class="story-card">'
        '<div class="story-big">Why the final champion won</div>'
        '<div class="story-small">'
        f'<b>PR-AUC {pr_auc:.3f}</b> · <b>F1 {best_f1:.3f}</b> · '
        f'<b>Precision {format_percent(best_precision)}</b> · '
        f'<b>Recall {format_percent(best_recall)}</b>. '
        'Best deployable balance after tuning, boosting comparison, ensemble check, and stability testing.'
        '</div>'
        '</div>'
    )


# --------------------------------------------------
# 11. FINAL MODEL PROOF
# --------------------------------------------------

render_html('<div class="section-kicker">Final champion proof</div><div class="section-title">Tuning and stability moved the model from current champion to true champion</div>')

proof_1, proof_2, proof_3 = st.columns(3)

with proof_1:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🧪</div>'
        '<div class="flow-title">Model hardening</div>'
        '<div class="flow-text">Checked class weights, threshold optimisation, tuned Random Forest, XGBoost, SMOTE option, and ensemble comparison.</div>'
        '</div>'
    )

with proof_2:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">📈</div>'
        '<div class="flow-title">Stability check</div>'
        '<div class="flow-text">Repeated split testing confirmed the tuned Random Forest stayed the strongest deployable model.</div>'
        '</div>'
    )

with proof_3:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🚨</div>'
        '<div class="flow-title">Outlier sensitivity</div>'
        '<div class="flow-text">Anomaly checks show extreme visitors carry useful conversion signal, so anomalies are reviewed instead of blindly removed.</div>'
        '</div>'
    )

if PLOTLY_AVAILABLE and not top_models.empty:
    benchmark_col_1, benchmark_col_2 = st.columns(2)

    with benchmark_col_1:
        st.plotly_chart(
            build_model_benchmark_chart(top_models),
            use_container_width=True,
        )

    with benchmark_col_2:
        st.plotly_chart(
            build_precision_recall_chart(top_models),
            use_container_width=True,
        )
else:
    st.caption("Model benchmark charts appear after final champion outputs exist and Plotly is available.")

with st.expander("Show final champion comparison table"):
    if not top_models.empty:
        display_columns = [
            "model_name",
            "model_family",
            "deployable",
            "pr_auc",
            "roc_auc",
            "best_threshold",
            "best_precision",
            "best_recall",
            "best_f1_score",
            "business_score",
            "Champion",
        ]

        existing_columns = [
            column for column in display_columns
            if column in top_models.columns
        ]

        st.dataframe(
            top_models[existing_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run final champion script first to generate final comparison tables.")


# --------------------------------------------------
# 12. PLATFORM CAPABILITY MAP
# --------------------------------------------------

render_html('<div class="section-kicker">Platform modules</div><div class="section-title">Full project story in one product</div>')

module_1, module_2, module_3, module_4 = st.columns(4)

with module_1:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🎯</div>'
        '<div class="flow-title">Visitor scoring</div>'
        '<div class="flow-text">Score one visitor, assign intent segment, recommend action, log prediction.</div>'
        '</div>'
    )

with module_2:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">📦</div>'
        '<div class="flow-title">Batch campaign scoring</div>'
        '<div class="flow-text">Upload CSV, score many visitors, rank campaign priority, download results.</div>'
        '</div>'
    )

with module_3:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">📈</div>'
        '<div class="flow-title">KPI forecasting</div>'
        '<div class="flow-text">Forecast traffic, event volume, conversions, and high-intent visitor demand.</div>'
        '</div>'
    )

with module_4:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🚨</div>'
        '<div class="flow-title">Monitoring + anomalies</div>'
        '<div class="flow-text">Track prediction logs, drift, abnormal behaviour, alert rules, and Grafana path.</div>'
        '</div>'
    )

module_5, module_6, module_7, module_8 = st.columns(4)

with module_5:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🏁</div>'
        '<div class="flow-title">Model benchmark</div>'
        '<div class="flow-text">Manual, AutoML-style, and final hardening comparison with proof tables.</div>'
        '</div>'
    )

with module_6:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🧩</div>'
        '<div class="flow-title">ML validation and evidence</div>'
        '<div class="flow-text">PCA, K-Means, DBSCAN, LOF, imbalance, tuning, boosting, and full Day 1–15 matrix.</div>'
        '</div>'
    )

with module_7:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">🐳</div>'
        '<div class="flow-title">MLOps stack</div>'
        '<div class="flow-text">Docker, tests, CI/CD, Prometheus, Grafana, alerts, and deployment path.</div>'
        '</div>'
    )

with module_8:
    render_html(
        '<div class="flow-card">'
        '<div class="flow-icon">☸️</div>'
        '<div class="flow-title">Kubernetes path</div>'
        '<div class="flow-text">Local app first, then Kubernetes + Helm architecture for production story.</div>'
        '</div>'
    )


# --------------------------------------------------
# 13. HONEST SCOPE NOTE
# --------------------------------------------------

render_html(
    '<div class="story-card">'
    '<div class="story-big">Honest scope note</div>'
    '<div class="story-small">'
    'The platform predicts conversion intent and forecasts operational demand. '
    'It does not claim direct revenue forecasting because the RetailRocket dataset contains behaviour events but no reliable transaction value.'
    '</div>'
    '</div>'
)
