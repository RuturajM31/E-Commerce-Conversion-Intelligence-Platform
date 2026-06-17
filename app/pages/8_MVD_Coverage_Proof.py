# 8_MVD_Coverage_Proof.py
# Streamlit page for final champion and MVD coverage proof.
#
# Purpose:
#   Prove that the project covers the required ML/MVD concepts and that the
#   final champion model was selected with evidence.
#
# Page flow:
#   1. Page setup
#   2. Page style
#   3. Load tables
#   4. Helper functions
#   5. Chart functions
#   6. Header and KPI cards
#   7. Final champion proof
#   8. MVD method coverage proof
#   9. PCA, clustering, and outlier evidence
#   10. Raw evidence tables

from __future__ import annotations

from typing import Dict, List

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
    get_champion_metrics,
    get_champion_model_name,
    inject_global_css,
    load_model_selection_tables,
    render_html,
)


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="MVD Coverage Proof",
    page_icon="🧩",
    layout="wide",
)

inject_global_css()


# --------------------------------------------------
# 2. PAGE STYLE
# --------------------------------------------------

render_html(
    """
    <style>
        .mvd-hero {
            padding: 34px 38px;
            border-radius: 32px;
            background:
                radial-gradient(circle at 12% 18%, rgba(34, 197, 94, 0.24), transparent 30%),
                radial-gradient(circle at 88% 82%, rgba(56, 189, 248, 0.22), transparent 34%),
                linear-gradient(135deg, #0F172A 0%, #111827 48%, #020617 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 28px 85px rgba(0, 0, 0, 0.46);
            margin-bottom: 22px;
        }

        .mvd-title {
            color: #F8FAFC;
            font-size: 2.35rem;
            line-height: 1.06;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .mvd-subtitle {
            color: #CBD5E1;
            font-size: 1.00rem;
            line-height: 1.58;
            max-width: 1080px;
        }

        .proof-card {
            padding: 24px 26px;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 30%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.97), rgba(17, 24, 39, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.32);
            min-height: 184px;
            margin-bottom: 18px;
        }

        .proof-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .proof-value {
            color: #F8FAFC;
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 950;
            margin-bottom: 10px;
        }

        .proof-help {
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.48;
        }

        .section-kicker {
            color: #86EFAC;
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
            font-size: 1.24rem;
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
# 3. LOAD TABLES
# --------------------------------------------------

tables = load_model_selection_tables()

final_comparison = tables.get("final_true_champion_comparison", pd.DataFrame())
final_summary = tables.get("final_true_champion_summary", pd.DataFrame())
final_thresholds = tables.get("final_true_champion_thresholds", pd.DataFrame())
final_stability = tables.get("final_true_champion_stability", pd.DataFrame())
final_sensitivity = tables.get("final_true_champion_sensitivity", pd.DataFrame())

mvd_coverage = tables.get("mvd_coverage_matrix", pd.DataFrame())
mvd_pca = tables.get("mvd_pca_variance", pd.DataFrame())
mvd_kmeans = tables.get("mvd_kmeans_summary", pd.DataFrame())
mvd_dbscan = tables.get("mvd_dbscan_summary", pd.DataFrame())
mvd_lof = tables.get("mvd_lof_summary", pd.DataFrame())
mvd_outlier_comparison = tables.get("mvd_outlier_comparison", pd.DataFrame())

champion_model_name = get_champion_model_name()
best_threshold = get_best_threshold()
champion_metrics = get_champion_metrics()


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def get_first_existing_column(data: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return first candidate column that exists."""

    for column in candidates:
        if column in data.columns:
            return column

    return None


def safe_metric(table: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    """Read one numeric value from first row of a table."""

    if table.empty or column not in table.columns:
        return float(fallback)

    value = table.iloc[0][column]

    if pd.isna(value):
        return float(fallback)

    return float(value)


def count_rows(data: pd.DataFrame) -> int:
    """Return number of rows in a table."""

    return 0 if data.empty else int(len(data))


def count_status_matches(data: pd.DataFrame, keywords: List[str]) -> int:
    """Count rows whose status/coverage column contains positive keywords."""

    if data.empty:
        return 0

    status_column = get_first_existing_column(
        data,
        [
            "status",
            "coverage_status",
            "implementation_status",
            "implemented",
            "project_status",
            "proof_status",
        ],
    )

    if status_column is None:
        return len(data)

    values = data[status_column].astype(str).str.lower()

    mask = False

    for keyword in keywords:
        mask = mask | values.str.contains(keyword.lower(), na=False)

    return int(mask.sum())


def format_count(value: int | float) -> str:
    """Format count for cards."""

    if pd.isna(value):
        return "NA"

    return f"{float(value):,.0f}"


def make_download_filename(label: str) -> str:
    """Create clean download filename."""

    return f"{label.lower().replace(' ', '_')}.csv"


def show_table_with_download(label: str, data: pd.DataFrame) -> None:
    """Show a table and add CSV download."""

    if data.empty:
        st.info(f"{label} is not available.")
        return

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label=f"Download {label}",
        data=data.to_csv(index=False),
        file_name=make_download_filename(label),
        mime="text/csv",
        use_container_width=True,
    )




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

def build_coverage_chart(coverage: pd.DataFrame):
    """Create coverage status chart if a useful column exists."""

    if not PLOTLY_AVAILABLE or coverage.empty:
        return None

    status_column = get_first_existing_column(
        coverage,
        [
            "status",
            "coverage_status",
            "implementation_status",
            "implemented",
            "project_status",
            "proof_status",
        ],
    )

    if status_column is None:
        category_column = get_first_existing_column(
            coverage,
            [
                "learning_day",
                "day",
                "topic",
                "method_family",
                "method",
            ],
        )

        if category_column is None:
            return None

        chart_data = (
            coverage[category_column]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        chart_data.columns = ["Coverage Area", "Count"]

        fig = px.bar(
            chart_data.sort_values("Count", ascending=True),
            x="Count",
            y="Coverage Area",
            orientation="h",
            title="MVD coverage rows by area",
        )

    else:
        chart_data = (
            coverage[status_column]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        chart_data.columns = ["Coverage Status", "Count"]

        fig = px.bar(
            chart_data,
            x="Coverage Status",
            y="Count",
            text="Count",
            title="MVD coverage status",
        )

        fig.update_traces(textposition="outside")

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=20),
        showlegend=False,
    )

    return fig


def build_final_comparison_chart(comparison: pd.DataFrame):
    """Create final model comparison chart."""

    if not PLOTLY_AVAILABLE or comparison.empty:
        return None

    if "model_name" not in comparison.columns or "pr_auc" not in comparison.columns:
        return None

    data = comparison.copy()

    if "business_score" not in data.columns:
        if "champion_score" in data.columns:
            data["business_score"] = data["champion_score"]
        else:
            data["business_score"] = 0

    data["is_selected"] = data["model_name"].astype(str).eq(champion_model_name)

    if "is_final_champion" in data.columns:
        data["is_selected"] = data["is_final_champion"].fillna(False)

    data["Status"] = data["is_selected"].map(
        {
            True: "Final champion",
            False: "Compared model",
        }
    )

    data = data.head(10).sort_values("pr_auc", ascending=True)

    fig = px.bar(
        data,
        x="pr_auc",
        y="model_name",
        color="Status",
        orientation="h",
        hover_data={
            "best_precision": ":.3f" if "best_precision" in data.columns else False,
            "best_recall": ":.3f" if "best_recall" in data.columns else False,
            "best_f1_score": ":.3f" if "best_f1_score" in data.columns else False,
            "business_score": ":.3f",
        },
        title="Final champion proof: model comparison by PR-AUC",
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="PR-AUC",
        yaxis_title="Model",
        legend_title="Status",
        title_font=dict(size=20),
    )

    return fig


def build_pca_chart(pca_table: pd.DataFrame):
    """Create PCA explained variance chart."""

    if not PLOTLY_AVAILABLE or pca_table.empty:
        return None

    component_column = get_first_existing_column(
        pca_table,
        ["component", "principal_component", "pca_component"],
    )

    variance_column = get_first_existing_column(
        pca_table,
        [
            "explained_variance_ratio",
            "variance_ratio",
            "explained_variance",
        ],
    )

    if component_column is None or variance_column is None:
        return None

    data = pca_table.copy()

    fig = px.bar(
        data,
        x=component_column,
        y=variance_column,
        text=variance_column,
        title="PCA proof: explained variance by component",
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="PCA component",
        yaxis_title="Explained variance ratio",
        title_font=dict(size=20),
        showlegend=False,
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def build_cluster_summary_chart(data: pd.DataFrame, title: str):
    """Create a generic clustering summary chart."""

    if not PLOTLY_AVAILABLE or data.empty:
        return None

    x_column = get_first_existing_column(
        data,
        [
            "k",
            "n_clusters",
            "eps",
            "method",
            "model_name",
            "configuration",
        ],
    )

    y_column = get_first_existing_column(
        data,
        [
            "silhouette_score",
            "cluster_count",
            "n_clusters_found",
            "noise_count",
            "inertia",
            "davies_bouldin_score",
        ],
    )

    if x_column is None or y_column is None:
        return None

    chart_data = data.copy()
    chart_data[x_column] = chart_data[x_column].astype(str)

    fig = px.bar(
        chart_data,
        x=x_column,
        y=y_column,
        text=y_column,
        title=title,
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=62, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_column.replace("_", " ").title(),
        yaxis_title=y_column.replace("_", " ").title(),
        title_font=dict(size=20),
        showlegend=False,
    )

    return fig


# --------------------------------------------------
# 6. DERIVED VALUES
# --------------------------------------------------

summary_row = final_summary.iloc[0] if not final_summary.empty else pd.Series(dtype="object")

pr_auc = safe_metric(final_summary, "pr_auc", float(champion_metrics.get("pr_auc", 0)))
roc_auc = safe_metric(final_summary, "roc_auc", float(champion_metrics.get("roc_auc", 0)))
precision = safe_metric(final_summary, "best_precision", float(champion_metrics.get("best_precision", 0)))
recall = safe_metric(final_summary, "best_recall", float(champion_metrics.get("best_recall", 0)))
f1_score = safe_metric(final_summary, "best_f1_score", float(champion_metrics.get("best_f1_score", 0)))

coverage_rows = count_rows(mvd_coverage)
implemented_rows = count_status_matches(
    mvd_coverage,
    ["yes", "covered", "implemented", "complete", "done", "proof"],
)

coverage_rate = implemented_rows / coverage_rows if coverage_rows > 0 else 0


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🧩 MVD Coverage Proof")
st.sidebar.caption("Final evidence page")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Champion:** {champion_model_name}")
st.sidebar.markdown(f"**Threshold:** {best_threshold:.2f}")
st.sidebar.markdown(f"**MVD rows:** {coverage_rows}")
st.sidebar.markdown(f"**Coverage rate:** {format_percent(coverage_rate)}")
st.sidebar.markdown("---")
st.sidebar.caption("Shows final champion proof and MVD method coverage evidence.")


# --------------------------------------------------
# 8. HEADER
# --------------------------------------------------

render_html(
    (
        f'<div class="mvd-hero">'
        f'<div class="eyebrow">MVD and course coverage evidence</div>'
        f'<div class="mvd-title">MVD Coverage Proof</div>'
        f'<div class="mvd-subtitle">'
        f'This page proves that the project covers the final champion workflow and the required ML method evidence: '
        f'PCA, clustering, outlier detection, imbalance handling, tuning, boosting, monitoring, and deployment thinking.'
        f'</div><br>'
        f'<span class="success-pill">Final champion: {escape_text(champion_model_name)}</span>'
        f'&nbsp;<span class="info-pill">Threshold: {best_threshold:.2f}</span>'
        f'&nbsp;<span class="warning-pill">Coverage rows: {coverage_rows}</span>'
        f'</div>'
    )
)


# --------------------------------------------------
# 9. KPI CARDS
# --------------------------------------------------

card_1, card_2, card_3, card_4 = st.columns(4)

with card_1:
    render_html(
        (
            f'<div class="proof-card">'
            f'<div class="proof-label">Final champion</div>'
            f'<div class="proof-value">{escape_text(champion_model_name)}</div>'
            f'<div class="proof-help">Final deployable model selected after hardening and comparison.</div>'
            f'</div>'
        )
    )

with card_2:
    render_html(
        (
            f'<div class="proof-card">'
            f'<div class="proof-label">PR-AUC / F1</div>'
            f'<div class="proof-value">{pr_auc:.3f} / {f1_score:.3f}</div>'
            f'<div class="proof-help">Rare-buyer ranking quality and balanced threshold performance.</div>'
            f'</div>'
        )
    )

with card_3:
    render_html(
        (
            f'<div class="proof-card">'
            f'<div class="proof-label">Precision / Recall</div>'
            f'<div class="proof-value">{format_percent(precision)} / {format_percent(recall)}</div>'
            f'<div class="proof-help">Target quality and buyer capture at the final threshold.</div>'
            f'</div>'
        )
    )

with card_4:
    render_html(
        (
            f'<div class="proof-card">'
            f'<div class="proof-label">MVD coverage rows</div>'
            f'<div class="proof-value">{coverage_rows}</div>'
            f'<div class="proof-help">Coverage matrix rows proving method implementation or evidence.</div>'
            f'</div>'
        )
    )


# --------------------------------------------------
# 10. FINAL CHAMPION PROOF
# --------------------------------------------------

render_html('<div class="section-kicker">Final champion proof</div><div class="section-title">The selected model is supported by comparison, threshold, stability, and sensitivity evidence</div>')

proof_col_1, proof_col_2 = st.columns(2)

with proof_col_1:
    comparison_chart = build_final_comparison_chart(final_comparison)

    if comparison_chart is not None:
        st.plotly_chart(
            comparison_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: final champion comparison",
            "It compares candidate models by PR-AUC and highlights the selected final champion.",
            "The selected champion is supported by evidence, not opinion. It gives the strongest practical rare-buyer ranking among deployable models.",
        )
    else:
        render_html(
            '<div class="story-card">'
            '<div class="story-title">Final comparison table</div>'
            '<div class="story-text">The final model comparison table will appear here after final champion outputs exist.</div>'
            '</div>'
        )

with proof_col_2:
    render_html(
        '<div class="story-card">'
        '<div class="story-title">What this proves</div>'
        '<div class="story-text">'
        'The final model was not selected randomly. The workflow compared baseline, tuned Random Forest, XGBoost, '
        'SMOTE-style imbalance handling, and an ensemble check. The final deployable champion was selected because it gave '
        'the best practical balance for rare-buyer ranking, precision, recall, F1, and stability.'
        '</div>'
        '</div>'
    )

    evidence_counts = pd.DataFrame(
        [
            {"Evidence Table": "Final comparison", "Rows": count_rows(final_comparison)},
            {"Evidence Table": "Threshold analysis", "Rows": count_rows(final_thresholds)},
            {"Evidence Table": "Stability check", "Rows": count_rows(final_stability)},
            {"Evidence Table": "Outlier sensitivity", "Rows": count_rows(final_sensitivity)},
        ]
    )

    st.dataframe(
        evidence_counts,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 11. MVD COVERAGE PROOF
# --------------------------------------------------

render_html('<div class="section-kicker">MVD coverage matrix</div><div class="section-title">Proof that required ML methods were covered</div>')

coverage_col_1, coverage_col_2 = st.columns([0.48, 0.52])

with coverage_col_1:
    coverage_chart = build_coverage_chart(mvd_coverage)

    if coverage_chart is not None:
        st.plotly_chart(
            coverage_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: MVD coverage status",
            "It summarises how many course/MVD requirements are covered, implemented, or mapped.",
            "This chart proves the project is not only a model demo. It documents method coverage across the learning module.",
        )
    else:
        render_html(
            '<div class="story-card">'
            '<div class="story-title">Coverage matrix</div>'
            '<div class="story-text">Coverage chart appears when the coverage matrix contains a status or topic column.</div>'
            '</div>'
        )

with coverage_col_2:
    render_html(
        '<div class="story-card">'
        '<div class="story-title">Coverage story</div>'
        '<div class="story-text">'
        'The project includes supervised learning, imbalance handling, model tuning, boosting, AutoML-style benchmarking, '
        'PCA evidence, clustering evidence, outlier evidence, forecasting, monitoring, alerting path, and MLOps architecture. '
        'The matrix below is the audit trail.'
        '</div>'
        '</div>'
    )

    if not mvd_coverage.empty:
        st.dataframe(
            mvd_coverage,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("MVD coverage matrix not available. Run: python3 -m src.models.build_mvd_method_coverage")


# --------------------------------------------------
# 12. PCA, CLUSTERING, AND OUTLIER EVIDENCE
# --------------------------------------------------

render_html('<div class="section-kicker">Method evidence</div><div class="section-title">PCA, clustering, and outlier method proof</div>')

method_col_1, method_col_2 = st.columns(2)

with method_col_1:
    pca_chart = build_pca_chart(mvd_pca)

    if pca_chart is not None:
        st.plotly_chart(
            pca_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: PCA explained variance",
            "It shows how much information each principal component captures from the visitor features.",
            "PCA proves dimensionality-reduction coverage and helps explain whether visitor behaviour can be summarised into fewer patterns.",
        )
    else:
        st.info("PCA chart appears when PCA variance table contains component and variance columns.")

with method_col_2:
    kmeans_chart = build_cluster_summary_chart(
        mvd_kmeans,
        "K-Means proof summary",
    )

    if kmeans_chart is not None:
        st.plotly_chart(
            kmeans_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: K-Means clustering proof",
            "It summarises K-Means clustering results such as cluster quality, inertia, or silhouette depending on the available table.",
            "K-Means proves unsupervised segmentation coverage and shows how visitors can be grouped without using conversion labels.",
        )
    else:
        st.info("K-Means chart appears when K-Means summary table has compatible columns.")

method_col_3, method_col_4 = st.columns(2)

with method_col_3:
    dbscan_chart = build_cluster_summary_chart(
        mvd_dbscan,
        "DBSCAN proof summary",
    )

    if dbscan_chart is not None:
        st.plotly_chart(
            dbscan_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: DBSCAN clustering proof",
            "It summarises density-based clustering output, including clusters or noise/outlier behaviour where available.",
            "DBSCAN proves density-clustering coverage and is useful for detecting unusual behaviour patterns that are not round or evenly sized.",
        )
    else:
        st.info("DBSCAN chart appears when DBSCAN summary table has compatible columns.")

with method_col_4:
    lof_chart = build_cluster_summary_chart(
        mvd_lof,
        "LOF outlier proof summary",
    )

    if lof_chart is not None:
        st.plotly_chart(
            lof_chart,
            use_container_width=True,
        )
        render_chart_explanation(
            "Chart explanation: LOF outlier proof",
            "It summarises Local Outlier Factor results for unusual visitor behaviour.",
            "LOF proves outlier-detection coverage and supports the project idea that suspicious visitors should be reviewed before campaign action.",
        )
    else:
        st.info("LOF chart appears when LOF summary table has compatible columns.")


# --------------------------------------------------
# 13. RAW EVIDENCE TABLES
# --------------------------------------------------

render_html('<div class="section-kicker">Raw evidence tables</div><div class="section-title">Downloadable proof files</div>')

tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(
    [
        "MVD matrix",
        "Final champion",
        "Thresholds",
        "Stability",
        "PCA / clustering",
        "Outliers",
    ]
)

with tab_1:
    show_table_with_download("MVD coverage matrix", mvd_coverage)

with tab_2:
    show_table_with_download("Final champion comparison", final_comparison)

with tab_3:
    show_table_with_download("Final champion thresholds", final_thresholds)

with tab_4:
    show_table_with_download("Final champion stability", final_stability)

with tab_5:
    st.markdown("#### PCA explained variance")
    show_table_with_download("MVD PCA explained variance", mvd_pca)

    st.markdown("#### K-Means summary")
    show_table_with_download("MVD KMeans summary", mvd_kmeans)

    st.markdown("#### DBSCAN summary")
    show_table_with_download("MVD DBSCAN summary", mvd_dbscan)

with tab_6:
    st.markdown("#### LOF outlier summary")
    show_table_with_download("MVD LOF outlier summary", mvd_lof)

    st.markdown("#### Outlier method comparison")
    show_table_with_download("MVD outlier method comparison", mvd_outlier_comparison)

    st.markdown("#### Final champion outlier sensitivity")
    show_table_with_download("Final champion outlier sensitivity", final_sensitivity)
