# 3_Model_Benchmark_Selection.py
# Governed Streamlit page for model selection and final champion evidence.
#
# Important evidence rule:
# - Validation evidence compares candidate models.
# - Untouched-holdout evidence reports final production performance.
# - Metrics from the two evidence splits are never mixed.

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file from inside the app folder. Find the repository root
# so imports and evidence paths work locally and on Streamlit Community Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent

while (
    PROJECT_ROOT != PROJECT_ROOT.parent
    and not (PROJECT_ROOT / "src").is_dir()
):
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.app_utils import inject_global_css


HOLDOUT_PATH = (
    PROJECT_ROOT / "reports/tables/final_true_champion_holdout.csv"
)
COMPARISON_PATH = (
    PROJECT_ROOT / "reports/tables/final_true_champion_comparison.csv"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "reports/tables/final_true_champion_thresholds.csv"
)
STABILITY_PATH = (
    PROJECT_ROOT / "reports/tables/final_true_champion_stability.csv"
)
SENSITIVITY_PATH = (
    PROJECT_ROOT / "reports/tables/final_true_champion_sensitivity.csv"
)


def read_csv_if_available(path: Path) -> pd.DataFrame:
    # Missing optional evidence should show a clear page message, not crash.
    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def numeric_value(
    row: pd.Series,
    candidates: tuple[str, ...],
    default: float = 0.0,
) -> float:
    # Different project generations used slightly different column names.
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                return float(value)

    return float(default)


def text_value(
    row: pd.Series,
    candidates: tuple[str, ...],
    default: str = "Unavailable",
) -> str:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value

    return default


def boolean_column(
    data: pd.DataFrame,
    column: str,
    default: bool = False,
) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=bool)

    return (
        data[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "selected"])
    )


def numeric_column(
    data: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series:
    for column in candidates:
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce")

    return pd.Series(float("nan"), index=data.index, dtype=float)


def percent(value: float) -> str:
    return f"{value:.1%}"


def score(value: float) -> str:
    return f"{value:.3f}"


def lift_text(value: float | None) -> str:
    if value is None:
        return "Unavailable"

    return f"{value:.1f}×"


def selected_candidate(
    comparison: pd.DataFrame,
    holdout_model_name: str,
) -> pd.Series | None:
    if comparison.empty:
        return None

    selected_mask = boolean_column(
        comparison,
        "is_final_champion",
    )
    selected_rows = comparison[selected_mask]

    if not selected_rows.empty:
        return selected_rows.iloc[0]

    if "model_name" in comparison.columns:
        name_mask = (
            comparison["model_name"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == holdout_model_name.strip().casefold()
        )
        matching_rows = comparison[name_mask]

        if not matching_rows.empty:
            return matching_rows.iloc[0]

    deployable_mask = boolean_column(comparison, "deployable")
    deployable_rows = comparison[deployable_mask].copy()

    if not deployable_rows.empty:
        deployable_rows["_business_score"] = numeric_column(
            deployable_rows,
            ("business_score", "champion_score"),
        )
        deployable_rows = deployable_rows.dropna(
            subset=["_business_score"]
        )

        if not deployable_rows.empty:
            index = deployable_rows["_business_score"].idxmax()
            return deployable_rows.loc[index]

    return comparison.iloc[0]


def highest_pr_auc_candidate(
    comparison: pd.DataFrame,
) -> pd.Series | None:
    if comparison.empty:
        return None

    candidate_data = comparison.copy()
    candidate_data["_pr_auc"] = numeric_column(
        candidate_data,
        ("pr_auc",),
    )
    candidate_data = candidate_data.dropna(subset=["_pr_auc"])

    if candidate_data.empty:
        return None

    return candidate_data.loc[candidate_data["_pr_auc"].idxmax()]


def build_model_comparison_chart(
    comparison: pd.DataFrame,
) -> object | None:
    required = {"model_name", "pr_auc"}

    if comparison.empty or not required.issubset(comparison.columns):
        return None

    chart_data = comparison.copy()
    chart_data["PR-AUC"] = numeric_column(
        chart_data,
        ("pr_auc",),
    )
    chart_data["ROC-AUC"] = numeric_column(
        chart_data,
        ("roc_auc",),
    )
    chart_data["Precision"] = numeric_column(
        chart_data,
        ("best_precision", "precision"),
    )
    chart_data["Recall"] = numeric_column(
        chart_data,
        ("best_recall", "recall"),
    )
    chart_data["Business score"] = numeric_column(
        chart_data,
        ("business_score", "champion_score"),
    )
    chart_data["Deployable"] = boolean_column(
        chart_data,
        "deployable",
    ).map(
        {
            True: "Deployable",
            False: "Not deployable",
        }
    )
    chart_data["Selected"] = boolean_column(
        chart_data,
        "is_final_champion",
    )
    chart_data["Model"] = chart_data["model_name"].astype(str)
    chart_data.loc[
        chart_data["Selected"],
        "Model",
    ] = chart_data.loc[
        chart_data["Selected"],
        "Model",
    ] + " ★ selected"

    chart_data = chart_data.sort_values("PR-AUC", ascending=True)

    figure = px.bar(
        chart_data,
        x="PR-AUC",
        y="Model",
        orientation="h",
        color="Deployable",
        hover_data={
            "ROC-AUC": ":.3f",
            "Precision": ":.1%",
            "Recall": ":.1%",
            "Business score": ":.3f",
            "Selected": True,
        },
        title="Validation candidate comparison by PR-AUC",
    )
    figure.update_layout(
        height=520,
        xaxis_title="Validation PR-AUC",
        yaxis_title="",
        legend_title="Deployment status",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_precision_recall_chart(
    comparison: pd.DataFrame,
) -> object | None:
    if comparison.empty or "model_name" not in comparison.columns:
        return None

    chart_data = comparison.copy()
    chart_data["Recall"] = numeric_column(
        chart_data,
        ("best_recall", "recall"),
    )
    chart_data["Precision"] = numeric_column(
        chart_data,
        ("best_precision", "precision"),
    )
    chart_data["Business score"] = numeric_column(
        chart_data,
        ("business_score", "champion_score"),
    ).fillna(0.0)
    chart_data["PR-AUC"] = numeric_column(
        chart_data,
        ("pr_auc",),
    )
    chart_data["Deployable"] = boolean_column(
        chart_data,
        "deployable",
    ).map(
        {
            True: "Deployable",
            False: "Not deployable",
        }
    )

    chart_data = chart_data.dropna(
        subset=["Recall", "Precision"]
    )

    if chart_data.empty:
        return None

    # Plotly requires positive marker sizes. Add a tiny display-only floor.
    chart_data["Marker size"] = (
        chart_data["Business score"].clip(lower=0.001)
    )

    figure = px.scatter(
        chart_data,
        x="Recall",
        y="Precision",
        size="Marker size",
        color="Deployable",
        hover_name="model_name",
        hover_data={
            "PR-AUC": ":.3f",
            "Business score": ":.3f",
            "Marker size": False,
        },
        title="Validation targeting trade-off",
    )
    figure.update_xaxes(
        title="Validation recall",
        tickformat=".0%",
    )
    figure.update_yaxes(
        title="Validation precision",
        tickformat=".0%",
    )
    figure.update_layout(
        height=520,
        legend_title="Deployment status",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_threshold_chart(
    thresholds: pd.DataFrame,
    selected_threshold: float,
) -> object | None:
    if thresholds.empty:
        return None

    threshold_column = None

    for candidate in ("threshold", "best_threshold"):
        if candidate in thresholds.columns:
            threshold_column = candidate
            break

    if threshold_column is None:
        return None

    metric_candidates = {
        "Precision": ("precision", "best_precision"),
        "Recall": ("recall", "best_recall"),
        "F1": ("f1_score", "best_f1_score"),
        "Target share": (
            "predicted_positive_rate",
            "targeted_share",
        ),
    }

    chart_data = pd.DataFrame(
        {
            "Threshold": pd.to_numeric(
                thresholds[threshold_column],
                errors="coerce",
            )
        }
    )

    for label, candidates in metric_candidates.items():
        values = numeric_column(thresholds, candidates)
        if values.notna().any():
            chart_data[label] = values

    value_columns = [
        column
        for column in chart_data.columns
        if column != "Threshold"
    ]

    if not value_columns:
        return None

    long_data = chart_data.melt(
        id_vars="Threshold",
        value_vars=value_columns,
        var_name="Metric",
        value_name="Value",
    ).dropna()

    figure = px.line(
        long_data,
        x="Threshold",
        y="Value",
        color="Metric",
        markers=True,
        title="Saved threshold operating trade-offs",
    )
    figure.add_vline(
        x=selected_threshold,
        line_dash="dash",
        annotation_text=f"Production {selected_threshold:.2f}",
        annotation_position="top right",
    )
    figure.update_yaxes(
        title="Metric value",
        tickformat=".0%",
    )
    figure.update_layout(
        height=520,
        legend_title="Metric",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_stability_chart(
    stability: pd.DataFrame,
) -> object | None:
    """Build repeated-split stability from raw seed rows or summary rows.

    The saved evidence currently contains one row per model and random seed:
    model_name, seed, pr_auc, and roc_auc. The chart therefore calculates the
    model-level mean and standard deviation before drawing the bars.

    If a future pipeline saves pre-aggregated mean/std columns, those columns
    are also supported.
    """

    if stability.empty or "model_name" not in stability.columns:
        return None

    data = stability.copy()

    summary_mean_column = next(
        (
            column
            for column in (
                "mean_pr_auc",
                "pr_auc_mean",
                "average_pr_auc",
            )
            if column in data.columns
        ),
        None,
    )

    summary_std_column = next(
        (
            column
            for column in (
                "std_pr_auc",
                "pr_auc_std",
                "stdev_pr_auc",
            )
            if column in data.columns
        ),
        None,
    )

    if summary_mean_column is not None:
        summary = pd.DataFrame(
            {
                "model_name": data["model_name"].astype(str),
                "mean_pr_auc": pd.to_numeric(
                    data[summary_mean_column],
                    errors="coerce",
                ),
            }
        )

        if summary_std_column is not None:
            summary["std_pr_auc"] = pd.to_numeric(
                data[summary_std_column],
                errors="coerce",
            )
        else:
            summary["std_pr_auc"] = 0.0

    elif "pr_auc" in data.columns:
        # Raw repeated-split evidence: aggregate one row per model.
        data["pr_auc"] = pd.to_numeric(
            data["pr_auc"],
            errors="coerce",
        )
        data = data.dropna(
            subset=["model_name", "pr_auc"],
        )

        if data.empty:
            return None

        summary = (
            data.groupby(
                "model_name",
                as_index=False,
                dropna=False,
            )
            .agg(
                mean_pr_auc=("pr_auc", "mean"),
                std_pr_auc=("pr_auc", "std"),
                repeated_splits=("pr_auc", "count"),
            )
        )
        summary["std_pr_auc"] = (
            summary["std_pr_auc"].fillna(0.0)
        )

    else:
        return None

    summary = summary.dropna(
        subset=["model_name", "mean_pr_auc"],
    )

    if summary.empty:
        return None

    summary = summary.sort_values(
        "mean_pr_auc",
        ascending=True,
    )

    hover_data = {
        "mean_pr_auc": ":.3f",
        "std_pr_auc": ":.3f",
    }

    if "repeated_splits" in summary.columns:
        hover_data["repeated_splits"] = True

    figure = px.bar(
        summary,
        x="mean_pr_auc",
        y="model_name",
        orientation="h",
        error_x="std_pr_auc",
        hover_data=hover_data,
        title="Repeated-split stability: mean PR-AUC by model",
    )
    figure.update_layout(
        height=430,
        xaxis_title="Mean validation PR-AUC",
        yaxis_title="",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Model Benchmark & Selection",
    page_icon="📊",
    layout="wide",
)

inject_global_css()

st.caption("Model selection evidence")
st.title("Model Benchmark & Final Champion Selection")
st.write(
    "This page explains the production decision without mixing validation "
    "metrics with untouched-holdout performance."
)

st.info(
    "Evidence boundary: validation compares candidate models. "
    "The untouched holdout reports final champion performance. "
    "Every final KPI and the lift calculation below use the untouched "
    "holdout only."
)


# --------------------------------------------------
# LOAD CONTROLLING EVIDENCE
# --------------------------------------------------
holdout = read_csv_if_available(HOLDOUT_PATH)
comparison = read_csv_if_available(COMPARISON_PATH)
thresholds = read_csv_if_available(THRESHOLD_PATH)
stability = read_csv_if_available(STABILITY_PATH)
sensitivity = read_csv_if_available(SENSITIVITY_PATH)

if holdout.empty:
    st.error(
        "Untouched-holdout evidence is unavailable. "
        "The page cannot make final performance claims safely."
    )
    st.code(
        "python3 -m src.models.finalize_true_champion",
        language="bash",
    )
    st.stop()

holdout_row = holdout.iloc[0]

champion_name = text_value(
    holdout_row,
    ("model_name", "champion_model_name"),
)
production_threshold = numeric_value(
    holdout_row,
    ("threshold", "best_threshold", "production_threshold"),
)
holdout_pr_auc = numeric_value(
    holdout_row,
    ("pr_auc", "holdout_pr_auc"),
)
holdout_roc_auc = numeric_value(
    holdout_row,
    ("roc_auc", "holdout_roc_auc"),
)
holdout_precision = numeric_value(
    holdout_row,
    ("precision", "best_precision"),
)
holdout_recall = numeric_value(
    holdout_row,
    ("recall", "best_recall"),
)
holdout_f1 = numeric_value(
    holdout_row,
    ("f1_score", "best_f1_score", "f1"),
)

holdout_rows = numeric_value(
    holdout_row,
    ("rows", "holdout_rows"),
)
positive_rows = numeric_value(
    holdout_row,
    ("positive_rows", "holdout_positive_rows"),
)
positive_rate = numeric_value(
    holdout_row,
    ("positive_rate", "buyer_rate"),
)

if positive_rate <= 0 and holdout_rows > 0:
    positive_rate = positive_rows / holdout_rows

holdout_lift = (
    holdout_precision / positive_rate
    if positive_rate > 0
    else None
)

selected_row = selected_candidate(
    comparison,
    champion_name,
)
highest_row = highest_pr_auc_candidate(comparison)


# --------------------------------------------------
# FINAL PERFORMANCE CLAIM
# --------------------------------------------------
st.subheader("Final production performance")
st.caption(
    "All cards in this section come from the untouched final holdout."
)

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Champion",
    champion_name,
    help="The saved production model.",
)
metric_columns[1].metric(
    "Production threshold",
    score(production_threshold),
    help="The saved operating rule.",
)
metric_columns[2].metric(
    "Holdout PR-AUC",
    score(holdout_pr_auc),
    help="Rare-buyer ranking quality on untouched data.",
)
metric_columns[3].metric(
    "Holdout ROC-AUC",
    score(holdout_roc_auc),
    help="General discrimination on untouched data.",
)

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Holdout precision",
    percent(holdout_precision),
    help="How many targeted visitors were actual buyers.",
)
metric_columns[1].metric(
    "Holdout recall",
    percent(holdout_recall),
    help="How many actual buyers were captured.",
)
metric_columns[2].metric(
    "Holdout F1",
    score(holdout_f1),
    help="Precision-recall balance on untouched data.",
)
metric_columns[3].metric(
    "Lift vs random",
    lift_text(holdout_lift),
    help=(
        "Untouched-holdout precision divided by the untouched-holdout "
        "buyer rate."
    ),
)

st.caption(
    "Lift formula: holdout precision ÷ holdout buyer rate. "
    f"Baseline buyer rate: {positive_rate:.3%}."
)


# --------------------------------------------------
# CANDIDATE-SELECTION DECISION
# --------------------------------------------------
st.divider()
st.subheader("Why this deployable champion was selected")
st.caption(
    "This section uses model-selection validation evidence. "
    "These numbers are not presented as final holdout performance."
)

decision_left, decision_right = st.columns(2)

with decision_left:
    st.markdown("#### Highest raw validation PR-AUC")

    if highest_row is None:
        st.warning("Candidate-comparison evidence is unavailable.")
    else:
        highest_name = text_value(
            highest_row,
            ("model_name",),
        )
        highest_pr_auc = numeric_value(
            highest_row,
            ("pr_auc",),
        )
        highest_deployable = bool(
            boolean_column(
                pd.DataFrame([highest_row]),
                "deployable",
            ).iloc[0]
        )
        status = (
            "deployable"
            if highest_deployable
            else "not deployable"
        )

        st.metric(
            highest_name,
            score(highest_pr_auc),
        )
        st.write(f"Deployment status: **{status}**.")

with decision_right:
    st.markdown("#### Selected deployable champion")

    if selected_row is None:
        st.warning("Selected candidate evidence is unavailable.")
    else:
        selected_name = text_value(
            selected_row,
            ("model_name",),
            champion_name,
        )
        selected_pr_auc = numeric_value(
            selected_row,
            ("pr_auc",),
        )
        selected_business_score = numeric_value(
            selected_row,
            ("business_score", "champion_score"),
        )

        st.metric(
            selected_name,
            score(selected_pr_auc),
        )
        st.write(
            "Selected because it had the strongest governed business "
            "score among deployable candidates."
        )
        st.caption(
            f"Saved validation business score: "
            f"{selected_business_score:.3f}."
        )

st.success(
    "Decision meaning: the model with the highest raw validation PR-AUC "
    "does not automatically become production champion. Deployability, "
    "business score, targeting trade-offs, and stability also control "
    "the final selection."
)


# --------------------------------------------------
# VALIDATION COMPARISON CHARTS
# --------------------------------------------------
st.divider()
st.subheader("Model-selection validation evidence")

comparison_chart = build_model_comparison_chart(comparison)
tradeoff_chart = build_precision_recall_chart(comparison)

if comparison_chart is None:
    st.warning(
        "Model-comparison chart cannot be drawn from the saved evidence."
    )
else:
    st.plotly_chart(
        comparison_chart,
        width="stretch",
        key="validation_pr_auc_comparison",
    )
    st.caption(
        "Longer bars mean stronger validation PR-AUC. "
        "Colour shows whether the candidate was deployable."
    )

if tradeoff_chart is None:
    st.warning(
        "Precision-recall comparison cannot be drawn from the saved evidence."
    )
else:
    st.plotly_chart(
        tradeoff_chart,
        width="stretch",
        key="validation_precision_recall",
    )
    st.caption(
        "This chart uses validation precision and recall only. "
        "It does not replace the untouched-holdout KPI cards."
    )


# --------------------------------------------------
# THRESHOLD AND ROBUSTNESS EVIDENCE
# --------------------------------------------------
st.divider()
st.subheader("Threshold and robustness evidence")

threshold_chart = build_threshold_chart(
    thresholds,
    production_threshold,
)
stability_chart = build_stability_chart(stability)

if threshold_chart is None:
    st.info("Saved threshold operating points are unavailable.")
else:
    st.plotly_chart(
        threshold_chart,
        width="stretch",
        key="threshold_operating_tradeoffs",
    )
    st.caption(
        "The dashed line marks the saved production threshold. "
        "The surrounding points show the targeting trade-off."
    )

if stability_chart is None:
    st.info("Repeated-split stability chart is unavailable.")
else:
    st.plotly_chart(
        stability_chart,
        width="stretch",
        key="repeated_split_stability",
    )
    st.caption(
        "Higher mean PR-AUC with lower variation indicates more stable "
        "validation performance."
    )

if not sensitivity.empty:
    with st.expander("Outlier-sensitivity evidence"):
        st.dataframe(
            sensitivity,
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------
# RAW GOVERNED EVIDENCE
# --------------------------------------------------
st.divider()
st.subheader("Raw governed evidence")

with st.expander(
    "Untouched-holdout result",
    expanded=True,
):
    st.dataframe(
        holdout,
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "This table controls final production-performance claims."
    )

with st.expander("Model-selection validation comparison"):
    if comparison.empty:
        st.info("Validation comparison table is unavailable.")
    else:
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "This table controls candidate comparison and selection logic."
        )

with st.expander("Threshold operating points"):
    if thresholds.empty:
        st.info("Threshold table is unavailable.")
    else:
        st.dataframe(
            thresholds,
            width="stretch",
            hide_index=True,
        )

with st.expander("Repeated-split stability"):
    if stability.empty:
        st.info("Stability table is unavailable.")
    else:
        st.dataframe(
            stability,
            width="stretch",
            hide_index=True,
        )

st.caption(
    "Sources: final_true_champion_holdout.csv, "
    "final_true_champion_comparison.csv, "
    "final_true_champion_thresholds.csv, and "
    "final_true_champion_stability.csv."
)
