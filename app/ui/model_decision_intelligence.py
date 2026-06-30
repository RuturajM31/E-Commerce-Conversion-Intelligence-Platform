"""Model-performance and threshold-decision extension.

This module converts existing model-selection artifacts into an interactive,
portfolio-ready decision layer without retraining any model. Unsupported
curve or calibration claims are shown as conditional evidence instead of being
invented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui.closure_core import (
    compact_count,
    currency,
    export_metadata,
    file_timestamp,
    finish_figure,
    percent,
    read_csv_if_exists,
    safe_divide,
)
from app.ui.components import (
    inject_product_css,
    render_detail_cards,
    render_empty_state,
    render_evidence_chart,
    render_kpi_grid,
    render_section_header,
    render_source_note,
    show_table_with_download,
)


COMPARISON_PATH = Path("reports/tables/final_true_champion_comparison.csv")
HOLDOUT_PATH = Path("reports/tables/final_true_champion_holdout.csv")
THRESHOLD_PATH = Path("reports/tables/final_true_champion_thresholds.csv")
STABILITY_PATH = Path("reports/tables/final_true_champion_stability.csv")
SENSITIVITY_PATH = Path("reports/tables/final_true_champion_sensitivity.csv")
SUMMARY_PATH = Path("reports/tables/final_true_champion_summary.csv")
CONFUSION_PATH = Path(
    "reports/visuals/ml_visual_intelligence/threshold_decisions/"
    "confusion_matrix_counts.csv"
)
GENERALISATION_PATH = Path(
    "reports/visuals/ml_visual_intelligence/champion_selection/"
    "champion_generalisation_metrics.csv"
)
REGISTRY_PATH = Path(
    "reports/visuals/ml_visual_intelligence/experiment_tracking/"
    "verified_ecommerce_registry_versions.csv"
)


def selected_holdout_row(holdout: pd.DataFrame) -> pd.Series | None:
    """Return the first controlling holdout row when it is available."""

    if holdout.empty:
        return None
    return holdout.iloc[0]


def build_model_comparison_figure(comparison: pd.DataFrame) -> go.Figure:
    """Compare candidate models using PR-AUC, ROC-AUC, and business score."""

    frame = comparison.copy()
    for column in ("pr_auc", "roc_auc", "business_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["Deployable"] = frame.get("deployable", False).astype(str)
    figure = px.scatter(
        frame,
        x="roc_auc",
        y="pr_auc",
        size="business_score",
        color="evaluation_split",
        symbol="Deployable",
        hover_name="model_name",
        hover_data={
            "model_family": True,
            "best_threshold": ":.3f",
            "best_precision": ":.2%",
            "best_recall": ":.2%",
            "business_score": ":.3f",
        },
    )
    figure.update_xaxes(title="ROC-AUC")
    figure.update_yaxes(title="PR-AUC")
    return finish_figure(
        figure,
        title="Candidate model evidence map",
        subtitle="PR-AUC, ROC-AUC, deployability, and business score are compared without mixing evidence splits.",
        height=560,
        legend_title="Evaluation split",
    )


def build_precision_recall_frontier(thresholds: pd.DataFrame) -> go.Figure:
    """Build the genuine saved precision-recall frontier across thresholds."""

    frame = thresholds.copy()
    for column in (
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["precision", "recall"])

    figure = px.scatter(
        frame,
        x="recall",
        y="precision",
        color="threshold",
        size="predicted_positive_rate",
        hover_data={
            "threshold": ":.3f",
            "f1_score": ":.3f",
            "predicted_positive_rate": ":.2%",
        },
        color_continuous_scale="Viridis",
    )
    figure.update_xaxes(title="Recall", tickformat=".0%")
    figure.update_yaxes(title="Precision", tickformat=".0%")
    return finish_figure(
        figure,
        title="Precision-recall decision frontier",
        subtitle="Each point is a saved threshold operating point; point size represents the targeted share.",
        height=540,
    )


def build_threshold_tradeoff(thresholds: pd.DataFrame) -> go.Figure:
    """Show threshold, precision, recall, F1, and selected share together."""

    frame = thresholds.copy()
    for column in (
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    figure = go.Figure()
    for column, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 score"),
        ("predicted_positive_rate", "Target share"),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame["threshold"],
                y=frame[column],
                mode="lines",
                name=label,
                hovertemplate=(
                    "Threshold %{x:.3f}<br>" + label + " %{y:.2%}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(title="Decision threshold")
    figure.update_yaxes(title="Rate", tickformat=".0%")
    return finish_figure(
        figure,
        title="Threshold operating trade-offs",
        subtitle="The saved production rule can be compared with nearby planning thresholds.",
        height=520,
        legend_title="Metric",
    )


def build_confusion_matrix_figure(confusion: pd.DataFrame) -> go.Figure:
    """Build a count and rate confusion matrix from saved holdout evidence."""

    row = confusion.iloc[0]
    values = np.array(
        [
            [float(row["true_negative"]), float(row["false_positive"])],
            [float(row["false_negative"]), float(row["true_positive"])],
        ]
    )
    total = max(values.sum(), 1.0)
    labels = np.vectorize(lambda value: f"{int(value):,}<br>{value / total:.2%}")(values)

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=["Predicted non-buyer", "Predicted buyer"],
            y=["Actual non-buyer", "Actual buyer"],
            text=labels,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Rows %{z:,.0f}<extra></extra>",
        )
    )
    return finish_figure(
        figure,
        title="Untouched-holdout confusion matrix",
        subtitle="Counts and whole-holdout rates show the operational consequences of the selected threshold.",
        height=470,
    )


def build_generalisation_figure(generalisation: pd.DataFrame) -> go.Figure:
    """Compare validation and untouched holdout performance."""

    row = generalisation.iloc[0]
    frame = pd.DataFrame(
        {
            "Evidence split": ["Validation", "Untouched holdout"],
            "PR-AUC": [row["validation_pr_auc"], row["holdout_pr_auc"]],
        }
    )
    figure = px.bar(
        frame,
        x="Evidence split",
        y="PR-AUC",
        text="PR-AUC",
    )
    figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    figure.update_yaxes(range=[0, max(float(frame["PR-AUC"].max()) * 1.25, 0.1)])
    return finish_figure(
        figure,
        title="Validation versus untouched holdout",
        subtitle="Generalisation evidence is kept separate from model-selection evidence.",
        height=430,
    )


def build_stability_figure(stability: pd.DataFrame) -> go.Figure:
    """Show PR-AUC across available random seeds."""

    frame = stability.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame["pr_auc"] = pd.to_numeric(frame["pr_auc"], errors="coerce")
    frame["roc_auc"] = pd.to_numeric(frame["roc_auc"], errors="coerce")

    melted = frame.melt(
        id_vars=["model_name", "seed"],
        value_vars=["pr_auc", "roc_auc"],
        var_name="Metric",
        value_name="Score",
    )
    figure = px.line(
        melted,
        x="seed",
        y="Score",
        color="model_name",
        line_dash="Metric",
        markers=True,
    )
    return finish_figure(
        figure,
        title="Seed-stability evidence",
        subtitle="Available repeated evaluations show whether rankings depend on one random split seed.",
        height=500,
        legend_title="Model and metric",
    )


def build_business_scenarios(
    thresholds: pd.DataFrame,
    *,
    holdout_rows: int,
    positive_rows: int,
    contact_cost: float,
    buyer_value: float,
) -> pd.DataFrame:
    """Translate threshold rows into transparent campaign economics."""

    frame = thresholds.copy()
    for column in (
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["threshold"]).copy()
    frame["Targeted visitors"] = (
        frame["predicted_positive_rate"].fillna(0) * holdout_rows
    ).round()
    frame["Expected buyers"] = (
        frame["Targeted visitors"] * frame["precision"].fillna(0)
    )
    frame["Buyers captured"] = positive_rows * frame["recall"].fillna(0)
    frame["Campaign cost"] = frame["Targeted visitors"] * contact_cost
    frame["Expected buyer value"] = frame["Expected buyers"] * buyer_value
    frame["Net scenario value"] = (
        frame["Expected buyer value"] - frame["Campaign cost"]
    )
    frame["Scenario ROI"] = np.where(
        frame["Campaign cost"] > 0,
        frame["Net scenario value"] / frame["Campaign cost"],
        0.0,
    )

    return frame[
        [
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "predicted_positive_rate",
            "Targeted visitors",
            "Expected buyers",
            "Buyers captured",
            "Campaign cost",
            "Expected buyer value",
            "Net scenario value",
            "Scenario ROI",
        ]
    ]


def render_model_decision_intelligence() -> None:
    """Render the final model and decision-intelligence extension."""

    inject_product_css()

    render_section_header(
        "Decision intelligence",
        "Compare model evidence, operating points, stability, and campaign economics",
        (
            "Every chart below uses saved project artifacts. Full ROC and calibration curves "
            "remain conditional because threshold-level false-positive and probability-bin tables were not saved."
        ),
    )

    comparison = read_csv_if_exists(str(COMPARISON_PATH))
    holdout = read_csv_if_exists(str(HOLDOUT_PATH))
    thresholds = read_csv_if_exists(str(THRESHOLD_PATH))
    stability = read_csv_if_exists(str(STABILITY_PATH))
    sensitivity = read_csv_if_exists(str(SENSITIVITY_PATH))
    confusion = read_csv_if_exists(str(CONFUSION_PATH))
    generalisation = read_csv_if_exists(str(GENERALISATION_PATH))
    registry = read_csv_if_exists(str(REGISTRY_PATH))
    summary = read_csv_if_exists(str(SUMMARY_PATH))

    holdout_row = selected_holdout_row(holdout)
    if holdout_row is None:
        render_empty_state(
            title="Final model evidence is unavailable",
            message="The untouched-holdout table is missing or empty.",
            next_action="Regenerate final champion artifacts before model closure.",
        )
        return

    model_name = str(holdout_row.get("model_name", "Unavailable"))
    threshold = float(holdout_row.get("threshold", 0))
    holdout_rows = int(holdout_row.get("rows", 0))
    positive_rows = int(holdout_row.get("positive_rows", 0))

    feature_schema = "Unavailable"
    split_strategy = "Unavailable"
    if not summary.empty:
        feature_schema = str(summary.iloc[0].get("feature_columns", "Unavailable"))
        split_strategy = str(summary.iloc[0].get("split_strategy", "Unavailable"))

    render_kpi_grid(
        [
            ("Champion model", model_name, "Final saved decision artifact"),
            ("Production threshold", f"{threshold:.3f}", "Saved operating rule"),
            ("Holdout PR-AUC", f"{float(holdout_row.get('pr_auc', 0)):.3f}", "Rare-buyer ranking evidence"),
            ("Holdout precision", percent(float(holdout_row.get("precision", 0))), "Target quality at the saved threshold"),
        ]
    )
    render_detail_cards(
        [
            ("Feature schema", "Approved model inputs", feature_schema),
            ("Split strategy", "Leakage-safe evaluation", split_strategy),
            ("Evidence date", "Final model artifacts", file_timestamp(HOLDOUT_PATH)),
        ]
    )

    tabs = st.tabs(
        [
            "Model comparison",
            "Threshold decisions",
            "Holdout consequences",
            "Robustness and registry",
        ]
    )

    with tabs[0]:
        if comparison.empty:
            render_empty_state(
                "Model comparison unavailable",
                "No final candidate comparison table was found.",
                "Regenerate final model selection evidence.",
            )
        else:
            figure = build_model_comparison_figure(comparison)
            best_pr = pd.to_numeric(comparison["pr_auc"], errors="coerce").max()
            render_evidence_chart(
                figure=figure,
                key="closure_model_comparison",
                what_it_shows="Candidate models by PR-AUC, ROC-AUC, business score, deployability, and evidence split.",
                how_to_read="Stronger candidates appear higher and farther right; larger points have a higher saved business score.",
                actual_finding=f"The strongest saved candidate PR-AUC is {best_pr:.3f}; the deployed champion is {model_name}.",
                conclusion="Model selection combines rare-buyer ranking, general discrimination, deployability, and business trade-offs.",
                recommended_action="Keep the saved champion until a challenger improves untouched-holdout evidence under the same split rules.",
                limitation="Candidate points may come from different evidence groups and are labelled rather than pooled silently.",
                source=str(COMPARISON_PATH),
                evidence_type="Offline validation and untouched-holdout comparison",
                refreshed_at=file_timestamp(COMPARISON_PATH),
            )
            show_table_with_download(
                label="model comparison evidence",
                data=comparison,
                file_name="model_comparison_evidence.csv",
                metadata=export_metadata(
                    source=str(COMPARISON_PATH),
                    evidence_type="Model comparison",
                    threshold=threshold,
                ),
            )

    with tabs[1]:
        if thresholds.empty:
            render_empty_state(
                "Threshold evidence unavailable",
                "No saved threshold frontier was found.",
                "Regenerate final champion threshold analysis.",
            )
        else:
            pr_figure = build_precision_recall_frontier(thresholds)
            render_evidence_chart(
                figure=pr_figure,
                key="closure_precision_recall_frontier",
                what_it_shows="Saved precision and recall operating points across model thresholds.",
                how_to_read="Move toward higher precision for smaller cleaner audiences or higher recall for broader buyer capture.",
                actual_finding=f"The saved production threshold is {threshold:.3f}.",
                conclusion="There is no single universally best threshold; capacity and contact economics determine the preferred operating point.",
                recommended_action="Use the production threshold by default and document any scenario override.",
                limitation="This is a threshold frontier, not a continuous probability calibration curve.",
                source=str(THRESHOLD_PATH),
                evidence_type="Saved validation threshold evidence",
                refreshed_at=file_timestamp(THRESHOLD_PATH),
            )

            tradeoff = build_threshold_tradeoff(thresholds)
            render_evidence_chart(
                figure=tradeoff,
                key="closure_threshold_tradeoff",
                what_it_shows="Precision, recall, F1, and targeted share against threshold.",
                how_to_read="Higher thresholds usually improve selectivity while reducing reach and captured buyers.",
                actual_finding=(
                    f"At the saved rule, {float(holdout_row.get('predicted_positive_rate', 0)):.2%} of the holdout is selected."
                ),
                conclusion="Threshold governance is a campaign decision, not merely a technical metric choice.",
                recommended_action="Review target share and buyer capture together before changing the rule.",
                limitation="Validation frontier values should not be described as live production performance.",
                source=str(THRESHOLD_PATH),
                evidence_type="Offline threshold validation",
                refreshed_at=file_timestamp(THRESHOLD_PATH),
            )

            scenario_1, scenario_2 = st.columns(2)
            with scenario_1:
                contact_cost = st.number_input(
                    "Scenario contact cost",
                    min_value=0.0,
                    value=1.0,
                    step=0.10,
                    key="model_contact_cost",
                )
            with scenario_2:
                buyer_value = st.number_input(
                    "Scenario buyer value",
                    min_value=0.0,
                    value=50.0,
                    step=5.0,
                    key="model_buyer_value",
                )

            scenarios = build_business_scenarios(
                thresholds,
                holdout_rows=holdout_rows,
                positive_rows=positive_rows,
                contact_cost=float(contact_cost),
                buyer_value=float(buyer_value),
            )
            show_table_with_download(
                label="threshold business scenarios",
                data=scenarios,
                file_name="threshold_business_scenarios.csv",
                metadata=export_metadata(
                    source=str(THRESHOLD_PATH),
                    evidence_type="Assumption-based threshold economics",
                    threshold=threshold,
                    filters={
                        "contact_cost": float(contact_cost),
                        "buyer_value": float(buyer_value),
                    },
                ),
                description="Scenario money values are assumptions; model metrics remain saved validation evidence.",
            )

    with tabs[2]:
        if confusion.empty:
            render_empty_state(
                "Confusion matrix unavailable",
                "Saved threshold confusion counts were not found.",
                "Regenerate the threshold decision visual package.",
            )
        else:
            confusion_figure = build_confusion_matrix_figure(confusion)
            false_positive = int(confusion.iloc[0].get("false_positive", 0))
            false_negative = int(confusion.iloc[0].get("false_negative", 0))
            render_evidence_chart(
                figure=confusion_figure,
                key="closure_confusion_matrix",
                what_it_shows="Correct and incorrect buyer classifications on the untouched holdout.",
                how_to_read="Rows are actual outcomes and columns are threshold decisions.",
                actual_finding=f"The saved rule produces {false_positive:,} false positives and {false_negative:,} false negatives on the holdout.",
                conclusion="False positives create contact cost; false negatives represent missed buyers.",
                recommended_action="Set the threshold using the relative business cost of those two errors.",
                limitation="These counts describe one untouched historical holdout, not current production traffic.",
                source=str(CONFUSION_PATH),
                evidence_type="Untouched-holdout threshold evidence",
                refreshed_at=file_timestamp(CONFUSION_PATH),
            )

        if not generalisation.empty:
            generalisation_figure = build_generalisation_figure(generalisation)
            delta = float(generalisation.iloc[0].get("pr_auc_delta", 0))
            render_evidence_chart(
                figure=generalisation_figure,
                key="closure_generalisation",
                what_it_shows="Champion PR-AUC on model-selection validation and untouched final holdout.",
                how_to_read="A small change indicates more stable generalisation; a large decline requires investigation.",
                actual_finding=f"The saved PR-AUC difference is {delta:+.3f}.",
                conclusion="Untouched-holdout performance is the controlling final evidence.",
                recommended_action="Use the holdout value in portfolio claims and keep validation results labelled separately.",
                limitation="One holdout does not replace future matured-label monitoring.",
                source=str(GENERALISATION_PATH),
                evidence_type="Validation-to-holdout generalisation",
                refreshed_at=file_timestamp(GENERALISATION_PATH),
            )

        render_source_note(
            source="No saved threshold-level false-positive curve or probability-bin table",
            evidence_type="Conditional ROC-curve and calibration evidence",
            refreshed_at=file_timestamp(HOLDOUT_PATH),
            extra=(
                "A single ROC operating point is available through the confusion matrix. "
                "A full ROC curve and observed-frequency calibration plot are not claimed."
            ),
        )

    with tabs[3]:
        if not stability.empty:
            stability_figure = build_stability_figure(stability)
            pr_std = pd.to_numeric(stability["pr_auc"], errors="coerce").std()
            render_evidence_chart(
                figure=stability_figure,
                key="closure_model_stability",
                what_it_shows="PR-AUC and ROC-AUC across the saved stability seeds.",
                how_to_read="Tighter lines and smaller ranges indicate less seed sensitivity.",
                actual_finding=f"PR-AUC standard deviation across saved rows is {pr_std:.4f}.",
                conclusion="Repeated evidence reduces the risk of selecting a model from one lucky split.",
                recommended_action="Retain stability evidence with every future champion decision.",
                limitation="The available seed count is finite and does not cover every data-shift scenario.",
                source=str(STABILITY_PATH),
                evidence_type="Offline repeated-seed robustness",
                refreshed_at=file_timestamp(STABILITY_PATH),
            )

        if not sensitivity.empty:
            show_table_with_download(
                label="model sensitivity evidence",
                data=sensitivity,
                file_name="model_sensitivity_evidence.csv",
                metadata=export_metadata(
                    source=str(SENSITIVITY_PATH),
                    evidence_type="Offline sensitivity analysis",
                    threshold=threshold,
                ),
            )

        if registry.empty:
            render_empty_state(
                "Verified ecommerce registry evidence unavailable",
                "No verified ecommerce MLflow registry rows were found.",
                "Run the existing MLflow lineage validation and regenerate the verified registry table.",
            )
        else:
            show_table_with_download(
                label="verified ecommerce registry versions",
                data=registry,
                file_name="verified_ecommerce_registry_versions.csv",
                metadata=export_metadata(
                    source=str(REGISTRY_PATH),
                    evidence_type="Verified ecommerce MLflow registry evidence",
                ),
                description="Only verified ecommerce registry rows are shown; unrelated evaluation runs are excluded.",
            )
