# 3_Model_Benchmark_Selection.py
# Streamlit page for model benchmark and final champion selection.
#
# Purpose:
#   Prove why the final production model was selected.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Load final champion context
#   4. Helper functions
#   5. Charts
#   6. Sidebar and header
#   7. Champion KPI proof
#   8. Final benchmark comparison
#   9. Threshold and stability proof
#   10. Raw evidence tables

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
    format_score,
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
    page_title="Model Benchmark & Selection",
    page_icon="🏁",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .benchmark-hero {
            padding: 34px 38px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.28), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(34, 197, 94, 0.20), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .benchmark-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .benchmark-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1050px;
        }

        .winner-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 188px;
            margin-bottom: 18px;
        }

        .winner-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .winner-value {
            color: #F8FAFC;
            font-size: 2.05rem;
            line-height: 1;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .winner-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.48;
        }

        .logic-card {
            padding: 24px 26px;
            border-radius: 30px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(2, 6, 23, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.34);
            margin-bottom: 18px;
        }

        .logic-title {
            color: #F8FAFC;
            font-size: 1.28rem;
            font-weight: 950;
            margin-bottom: 8px;
        }

        .logic-text {
            color: #CBD5E1;
            font-size: 0.94rem;
            line-height: 1.58;
        }

        .section-kicker {
            color: #7DD3FC;
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

        .decision-step {
            padding: 22px 22px;
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 16px 46px rgba(0, 0, 0, 0.24);
            min-height: 154px;
            margin-bottom: 16px;
        }

        .step-number {
            display: inline-block;
            width: 32px;
            height: 32px;
            border-radius: 11px;
            background: linear-gradient(135deg, #0EA5E9, #22C55E);
            color: #020617;
            font-weight: 950;
            text-align: center;
            line-height: 32px;
            margin-bottom: 12px;
        }

        .step-title {
            color: #F8FAFC;
            font-size: 1.02rem;
            font-weight: 900;
            margin-bottom: 7px;
        }

        .step-text {
            color: #CBD5E1;
            font-size: 0.86rem;
            line-height: 1.48;
        }

        .table-note {
            color: #94A3B8;
            font-size: 0.82rem;
            line-height: 1.48;
            margin-top: 8px;
        }
    </style>
    """
)


# --------------------------------------------------
# 3. LOAD FINAL CHAMPION CONTEXT
# --------------------------------------------------

tables = load_model_selection_tables()
champion_metrics = get_champion_metrics()
master_table = create_master_model_comparison_table()

champion_model_name = get_champion_model_name()
champion_track = get_champion_track()
best_threshold = get_best_threshold()

pr_auc = float(champion_metrics.get("pr_auc", 0))
roc_auc = float(champion_metrics.get("roc_auc", 0))
best_precision = float(champion_metrics.get("best_precision", 0))
best_recall = float(champion_metrics.get("best_recall", 0))
best_f1 = float(champion_metrics.get("best_f1_score", 0))


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def load_optional_csv(path: str) -> pd.DataFrame:
    """Load optional CSV if it exists."""

    csv_path = Path(path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def get_natural_conversion_rate() -> float:
    """Calculate natural conversion rate from visitor feature table."""

    visitor_features = load_optional_csv("data/processed/visitor_features.csv")

    if visitor_features.empty or "converted" not in visitor_features.columns:
        return 0.008269

    return float(visitor_features["converted"].mean())


def calculate_lift(precision: float) -> float:
    """Calculate lift versus random targeting."""

    natural_rate = get_natural_conversion_rate()

    if natural_rate <= 0:
        return 0.0

    return precision / natural_rate


def readable_track_name(track_name: str) -> str:
    """Convert internal track names into readable dashboard labels."""

    track_lower = str(track_name).lower()

    if "final" in track_lower:
        return "Final True Champion"

    if "manual" in track_lower:
        return "Manual Champion/Challenger"

    if "automl" in track_lower:
        return "AutoML-style Benchmark"

    return str(track_name)


def safe_numeric(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column or a default series."""

    if data.empty or column not in data.columns:
        return pd.Series([default] * len(data), index=data.index)

    return pd.to_numeric(data[column], errors="coerce").fillna(default)


def get_final_champion_row(comparison: pd.DataFrame) -> pd.Series:
    """Find the selected final champion row from comparison table."""

    if comparison.empty:
        return pd.Series(dtype="object")

    data = comparison.copy()

    if "is_final_champion" in data.columns:
        selected = data[data["is_final_champion"] == True]
        if not selected.empty:
            return selected.iloc[0]

    if "model_name" in data.columns:
        selected = data[data["model_name"].astype(str) == champion_model_name]
        if not selected.empty:
            return selected.iloc[0]

    return data.iloc[0]


def get_metric_from_row(row: pd.Series, column: str, fallback: float) -> float:
    """Read one metric from a pandas row safely."""

    if row.empty or column not in row.index:
        return fallback

    value = row[column]

    if pd.isna(value):
        return fallback

    return float(value)


def prepare_display_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Prepare final model comparison table for display."""

    if comparison.empty:
        return comparison

    data = comparison.copy()

    if "track" not in data.columns:
        data["track"] = "final_true_champion"

    if "deployable" not in data.columns:
        data["deployable"] = True

    if "business_score" not in data.columns and "champion_score" in data.columns:
        data["business_score"] = data["champion_score"]

    if "best_lift_vs_random" not in data.columns:
        data["best_lift_vs_random"] = safe_numeric(data, "best_precision").apply(calculate_lift)

    data["Track"] = data["track"].apply(readable_track_name)
    data["Status"] = data["is_final_champion"].map(
        {
            True: "Selected final champion",
            False: "Compared model",
        }
    )

    columns = [
        "model_rank",
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
        "best_lift_vs_random",
        "Status",
    ]

    available_columns = [column for column in columns if column in data.columns]
    display = data[available_columns].copy()

    display = display.rename(
        columns={
            "model_rank": "Rank",
            "model_name": "Model",
            "model_family": "Family",
            "deployable": "Deployable",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "best_threshold": "Threshold",
            "best_precision": "Precision",
            "best_recall": "Recall",
            "best_f1_score": "F1",
            "business_score": "Business Score",
            "best_lift_vs_random": "Lift vs Random",
        }
    )

    return display


def get_threshold_table() -> pd.DataFrame:
    """Return the best available threshold table."""

    final_thresholds = tables.get("final_true_champion_thresholds", pd.DataFrame())

    if not final_thresholds.empty:
        data = final_thresholds.copy()

        if "model_name" in data.columns:
            matching = data[data["model_name"].astype(str) == champion_model_name]
            if not matching.empty:
                data = matching.copy()

        return data

    old_thresholds = tables.get("old_champion_threshold", pd.DataFrame())

    if not old_thresholds.empty:
        return old_thresholds.copy()

    return pd.DataFrame()


def get_stability_table() -> pd.DataFrame:
    """Return final champion stability table."""

    return tables.get("final_true_champion_stability", pd.DataFrame())


def get_sensitivity_table() -> pd.DataFrame:
    """Return final champion outlier sensitivity table."""

    return tables.get("final_true_champion_sensitivity", pd.DataFrame())




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

def build_pr_auc_chart(comparison: pd.DataFrame):
    """Create PR-AUC ranking chart."""

    if not PLOTLY_AVAILABLE or comparison.empty:
        return None

    data = comparison.head(10).copy()

    if "track" not in data.columns:
        data["track"] = "final_true_champion"

    data["model_display"] = (
        data["model_name"].astype(str)
        + " · "
        + data["track"].apply(readable_track_name)
    )

    data["champion_status"] = data["is_final_champion"].map(
        {
            True: "Final champion",
            False: "Compared model",
        }
    )

    data = data.sort_values("pr_auc", ascending=True)

    fig = px.bar(
        data,
        x="pr_auc",
        y="model_display",
        orientation="h",
        color="champion_status",
        hover_data={
            "best_f1_score": ":.3f" if "best_f1_score" in data.columns else False,
            "best_precision": ":.3f" if "best_precision" in data.columns else False,
            "best_recall": ":.3f" if "best_recall" in data.columns else False,
            "business_score": ":.3f" if "business_score" in data.columns else False,
        },
        title="Final model comparison: rare-buyer ranking strength",
    )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="PR-AUC",
        yaxis_title="",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


def build_precision_recall_chart(comparison: pd.DataFrame):
    """Create precision vs recall chart."""

    if not PLOTLY_AVAILABLE or comparison.empty:
        return None

    data = comparison.copy()

    if "model_name" not in data.columns:
        return None

    size_column = "business_score" if "business_score" in data.columns else "best_f1_score"

    data["champion_status"] = data["is_final_champion"].map(
        {
            True: "Final champion",
            False: "Compared model",
        }
    )

    fig = px.scatter(
        data,
        x="best_recall",
        y="best_precision",
        size=size_column,
        color="champion_status",
        hover_name="model_name",
        hover_data={
            "pr_auc": ":.3f" if "pr_auc" in data.columns else False,
            "best_f1_score": ":.3f" if "best_f1_score" in data.columns else False,
            "best_threshold": ":.2f" if "best_threshold" in data.columns else False,
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
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Recall: buyers captured",
        yaxis_title="Precision: target quality",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


def build_threshold_chart(thresholds: pd.DataFrame):
    """Create threshold trade-off chart."""

    if not PLOTLY_AVAILABLE or thresholds.empty:
        return None

    data = thresholds.copy()

    if "threshold" not in data.columns:
        return None

    rename_map = {
        "precision": "Precision",
        "best_precision": "Precision",
        "recall": "Recall",
        "best_recall": "Recall",
        "f1_score": "F1",
        "best_f1_score": "F1",
        "predicted_positive_rate": "Targeted Share",
        "targeted_share": "Targeted Share",
    }

    available_metric_columns = [
        column for column in rename_map
        if column in data.columns
    ]

    if not available_metric_columns:
        return None

    chart_data = data[["threshold", *available_metric_columns]].copy()
    chart_data = chart_data.rename(columns=rename_map)

    long_data = chart_data.melt(
        id_vars="threshold",
        var_name="Metric",
        value_name="Value",
    )

    fig = px.line(
        long_data,
        x="threshold",
        y="Value",
        color="Metric",
        markers=True,
        title="Threshold trade-off: precision, recall, F1, and target share",
    )

    fig.add_vline(
        x=best_threshold,
        line_dash="dash",
        annotation_text=f"Selected {best_threshold:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Decision threshold",
        yaxis_title="Metric value",
        legend_title="",
        title_font=dict(size=20),
    )

    return fig


def build_stability_chart(stability: pd.DataFrame):
    """Create model stability chart if stability output exists."""

    if not PLOTLY_AVAILABLE or stability.empty:
        return None

    data = stability.copy()

    if "model_name" not in data.columns:
        return None

    metric_column = None

    for candidate in ["mean_pr_auc", "pr_auc_mean", "average_pr_auc"]:
        if candidate in data.columns:
            metric_column = candidate
            break

    if metric_column is None:
        return None

    error_column = None

    for candidate in ["std_pr_auc", "pr_auc_std", "stdev_pr_auc"]:
        if candidate in data.columns:
            error_column = candidate
            break

    fig = px.bar(
        data.sort_values(metric_column, ascending=True),
        x=metric_column,
        y="model_name",
        orientation="h",
        error_x=error_column if error_column else None,
        title="Stability check: mean PR-AUC across repeated splits",
    )

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mean PR-AUC",
        yaxis_title="Model",
        title_font=dict(size=20),
    )

    return fig


# --------------------------------------------------
# 6. DERIVED VALUES
# --------------------------------------------------

champion_row = get_final_champion_row(master_table)

# Prefer row values because final comparison files are the strongest evidence.
pr_auc = get_metric_from_row(champion_row, "pr_auc", pr_auc)
roc_auc = get_metric_from_row(champion_row, "roc_auc", roc_auc)
best_precision = get_metric_from_row(champion_row, "best_precision", best_precision)
best_recall = get_metric_from_row(champion_row, "best_recall", best_recall)
best_f1 = get_metric_from_row(champion_row, "best_f1_score", best_f1)
business_score = get_metric_from_row(champion_row, "business_score", get_metric_from_row(champion_row, "champion_score", 0))
best_lift = calculate_lift(best_precision)

thresholds = get_threshold_table()
stability = get_stability_table()
sensitivity = get_sensitivity_table()
display_table = prepare_display_table(master_table)


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🏁 Model Benchmark")
st.sidebar.caption("Final champion selection proof")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Final champion:** {champion_model_name}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**PR-AUC:** {pr_auc:.3f}")
st.sidebar.markdown(f"**Precision:** {format_percent(best_precision)}")
st.sidebar.markdown(f"**Recall:** {format_percent(best_recall)}")
st.sidebar.markdown("---")
st.sidebar.caption("Shows why the final deployable model was selected.")


# --------------------------------------------------
# 8. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="benchmark-hero">'
        f'<div class="eyebrow">Model selection evidence</div>'
        f'<div class="benchmark-title">Model Benchmark & Final Champion Selection</div>'
        f'<div class="benchmark-subtitle">'
        f'This page proves why the final production model was selected. '
        f'It compares baseline, tuned models, boosting, imbalance handling, ensemble checks, '
        f'threshold trade-offs, and stability evidence.'
        f'</div><br>'
        f'<span class="success-pill">Final champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Lift: {format_lift(best_lift)} vs random</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 9. CHAMPION KPI PROOF
# --------------------------------------------------

winner_1, winner_2, winner_3, winner_4 = st.columns(4)

with winner_1:
    render_html(
        (
            f'<div class="winner-card">'
            f'<div class="winner-label">Final champion</div>'
            f'<div class="winner-value">{escape_text(champion_model_name)}</div>'
            f'<div class="winner-help">Selected as the strongest deployable model after final hardening.</div>'
            f'</div>'
        )
    )

with winner_2:
    render_html(
        (
            f'<div class="winner-card">'
            f'<div class="winner-label">PR-AUC</div>'
            f'<div class="winner-value">{pr_auc:.3f}</div>'
            f'<div class="winner-help">Ranking quality for rare-buyer detection.</div>'
            f'</div>'
        )
    )

with winner_3:
    render_html(
        (
            f'<div class="winner-card">'
            f'<div class="winner-label">Precision / Recall</div>'
            f'<div class="winner-value">{format_percent(best_precision)} / {format_percent(best_recall)}</div>'
            f'<div class="winner-help">Target quality balanced with buyer capture.</div>'
            f'</div>'
        )
    )

with winner_4:
    render_html(
        (
            f'<div class="winner-card">'
            f'<div class="winner-label">F1 / Business score</div>'
            f'<div class="winner-value">{best_f1:.3f} / {business_score:.3f}</div>'
            f'<div class="winner-help">Final decision score used for business-ready selection.</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 10. SELECTION LOGIC
# --------------------------------------------------

render_html('<div class="section-kicker">Selection logic</div><div class="section-title">How the true champion was selected</div>')

step_1, step_2, step_3, step_4 = st.columns(4)

with step_1:
    render_html(
        '<div class="decision-step">'
        '<div class="step-number">1</div>'
        '<div class="step-title">Baseline and benchmark</div>'
        '<div class="step-text">Started with Logistic Regression and manual/AutoML-style benchmark comparison.</div>'
        '</div>'
    )

with step_2:
    render_html(
        '<div class="decision-step">'
        '<div class="step-number">2</div>'
        '<div class="step-title">Imbalance handling</div>'
        '<div class="step-text">Checked class weights, threshold optimisation, and optional SMOTE-style logic.</div>'
        '</div>'
    )

with step_3:
    render_html(
        '<div class="decision-step">'
        '<div class="step-number">3</div>'
        '<div class="step-title">Model hardening</div>'
        '<div class="step-text">Compared tuned Random Forest, XGBoost, and ensemble output against deployable criteria.</div>'
        '</div>'
    )

with step_4:
    render_html(
        '<div class="decision-step">'
        '<div class="step-number">4</div>'
        '<div class="step-title">Production decision</div>'
        '<div class="step-text">Selected the best deployable model with strong PR-AUC, F1, precision, and stability.</div>'
        '</div>'
    )


# --------------------------------------------------
# 11. FINAL BENCHMARK VISUALS
# --------------------------------------------------

render_html('<div class="section-kicker">Final benchmark</div><div class="section-title">Model comparison after final hardening</div>')

if PLOTLY_AVAILABLE and not master_table.empty:
    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.plotly_chart(
            build_pr_auc_chart(master_table),
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: PR-AUC model ranking",
            "It compares models by PR-AUC. PR-AUC is important here because buyers are rare, so it shows how well each model ranks likely buyers above non-buyers.",
            "The strongest model is the one with the highest PR-AUC and acceptable business metrics. This supports why the final champion was selected instead of relying on accuracy alone.",
        )

    with chart_col_2:
        st.plotly_chart(
            build_precision_recall_chart(master_table),
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: precision versus recall",
            "It compares targeting quality. Precision means how many targeted visitors actually converted. Recall means how many real buyers the model captured.",
            "A useful campaign model needs balance. Very high recall with poor precision wastes budget, while very high precision with poor recall misses many buyers.",
        )
else:
    st.info("Run final champion workflow and install Plotly to see benchmark charts.")


# --------------------------------------------------
# 12. THRESHOLD AND STABILITY PROOF
# --------------------------------------------------

render_html('<div class="section-kicker">Threshold and stability</div><div class="section-title">Why the production score gate is strict</div>')

threshold_col, stability_col = st.columns(2)

with threshold_col:
    threshold_chart = build_threshold_chart(thresholds)

    if threshold_chart is not None:
        st.plotly_chart(
            threshold_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: threshold performance",
            "It shows how precision, recall, and F1 change when the purchase-intent cutoff changes.",
            "The threshold is a business decision, not only a technical setting. The selected threshold should target visitors with strong buying intent while avoiding too much wasted outreach.",
        )
    else:
        st.info("Threshold chart will appear when threshold table is available.")

with stability_col:
    stability_chart = build_stability_chart(stability)

    if stability_chart is not None:
        st.plotly_chart(
            stability_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: stability check",
            "It shows whether top models remain strong across repeated train/test splits or different random seeds.",
            "A stable model is safer for production because its performance is not based on one lucky split.",
        )
    else:
        render_html(
            '<div class="logic-card">'
            '<div class="logic-title">Stability proof</div>'
            '<div class="logic-text">'
            'The final champion workflow includes repeated split checks. '
            'If the stability table is available, it will appear here as a chart.'
            '</div>'
            '</div>'
        )


# --------------------------------------------------
# 13. OUTLIER SENSITIVITY STORY
# --------------------------------------------------

render_html('<div class="section-kicker">Outlier sensitivity</div><div class="section-title">Anomalies are reviewed, not blindly removed</div>')

if not sensitivity.empty:
    st.dataframe(
        sensitivity,
        use_container_width=True,
        hide_index=True,
    )
else:
    render_html(
        '<div class="logic-card">'
        '<div class="logic-title">Sensitivity check included</div>'
        '<div class="logic-text">'
        'The final champion workflow checks how model quality changes when anomaly-flagged visitors are removed. '
        'This supports the design choice: keep anomalies as a review layer instead of deleting useful conversion signals.'
        '</div>'
        '</div>'
    )


# --------------------------------------------------
# 14. RAW EVIDENCE TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Raw evidence</div><div class="section-title">Tables used by this page</div>')

with st.expander("Final model comparison table", expanded=True):
    if not display_table.empty:
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("This table is the main evidence for the final champion decision.")
    else:
        st.info("Final comparison table not available yet.")

with st.expander("Threshold table"):
    if not thresholds.empty:
        st.dataframe(
            thresholds,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Threshold table not available yet.")

with st.expander("Stability table"):
    if not stability.empty:
        st.dataframe(
            stability,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Stability table not available yet.")

with st.expander("Manual and AutoML-style benchmark tables"):
    manual_results = tables.get("manual_results", pd.DataFrame())
    automl_results = tables.get("automl_results", pd.DataFrame())

    st.markdown("#### Manual benchmark")
    if not manual_results.empty:
        st.dataframe(manual_results, use_container_width=True, hide_index=True)
    else:
        st.info("Manual benchmark table not available.")

    st.markdown("#### AutoML-style benchmark")
    if not automl_results.empty:
        st.dataframe(automl_results, use_container_width=True, hide_index=True)
    else:
        st.info("AutoML-style benchmark table not available.")
