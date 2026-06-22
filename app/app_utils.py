# app_utils.py
# Shared helper functions for all Streamlit pages.
#
# Simple purpose:
#   Keep common app logic in one place.
#
# Main jobs:
#   1. Load the final champion model.
#   2. Load the final champion metadata.
#   3. Prepare prediction input features.
#   4. Assign visitor segments and business actions.
#   5. Load benchmark / MVD proof tables.
#   6. Provide shared formatting and HTML helpers.

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.data.feature_engineering import (
    build_single_visitor_features,
    prepare_model_features,
)
from src.models.production_model import (
    get_production_model_name,
    get_production_threshold,
    load_production_bundle,
)


# --------------------------------------------------
# 1. PRODUCTION MODEL SOURCE
# --------------------------------------------------
# Model and metadata selection is centralised in
# src.models.production_model.
#
# This prevents the app from selecting a model from one generation
# and metadata from another generation.


# --------------------------------------------------
# 2. RESULT TABLE PATHS
# --------------------------------------------------
# These tables are used by benchmark, monitoring, and MVD proof pages.

MANUAL_RESULTS_PATH = Path("reports/tables/manual_model_comparison.csv")
AUTOML_RESULTS_PATH = Path("reports/tables/automl_benchmark_results.csv")
INITIAL_FINAL_SELECTION_PATH = Path("reports/tables/final_model_selection.csv")
OLD_CHAMPION_METRICS_PATH = Path("reports/tables/champion_model_metrics.csv")
OLD_CHAMPION_THRESHOLD_PATH = Path("reports/tables/champion_threshold_analysis.csv")

FINAL_TRUE_CHAMPION_COMPARISON_PATH = Path("reports/tables/final_true_champion_comparison.csv")
FINAL_TRUE_CHAMPION_SUMMARY_PATH = Path("reports/tables/final_true_champion_summary.csv")
FINAL_TRUE_CHAMPION_THRESHOLDS_PATH = Path("reports/tables/final_true_champion_thresholds.csv")
FINAL_TRUE_CHAMPION_STABILITY_PATH = Path("reports/tables/final_true_champion_stability.csv")
FINAL_TRUE_CHAMPION_SENSITIVITY_PATH = Path("reports/tables/final_true_champion_sensitivity.csv")

MVD_COVERAGE_MATRIX_PATH = Path("reports/tables/mvd_method_coverage_matrix.csv")
MVD_PCA_VARIANCE_PATH = Path("reports/tables/mvd_pca_explained_variance.csv")
MVD_KMEANS_SUMMARY_PATH = Path("reports/tables/mvd_kmeans_summary.csv")
MVD_DBSCAN_SUMMARY_PATH = Path("reports/tables/mvd_dbscan_summary.csv")
MVD_LOF_SUMMARY_PATH = Path("reports/tables/mvd_lof_outlier_summary.csv")
MVD_OUTLIER_COMPARISON_PATH = Path("reports/tables/mvd_outlier_method_comparison.csv")

# Package 1 interactive ML validation evidence.
ML_VALIDATION_PROJECTION_PATH = Path("reports/tables/ml_validation_projection_sample.csv")
ML_VALIDATION_PCA_VARIANCE_PATH = Path("reports/tables/ml_validation_pca_variance.csv")
ML_VALIDATION_PCA_LOADINGS_PATH = Path("reports/tables/ml_validation_pca_loadings.csv")
ML_VALIDATION_CLUSTER_PROFILE_PATH = Path("reports/tables/ml_validation_cluster_profile.csv")
ML_VALIDATION_METHOD_SUMMARY_PATH = Path("reports/tables/ml_validation_method_summary.csv")


# --------------------------------------------------
# 3. SAFE HTML HELPERS
# --------------------------------------------------

def clean_html(raw_html: str) -> str:
    """Remove extra indentation before rendering custom HTML."""

    return textwrap.dedent(raw_html).strip()


def render_html(raw_html: str) -> None:
    """Render cleaned HTML inside the main Streamlit page."""

    st.markdown(
        clean_html(raw_html),
        unsafe_allow_html=True,
    )


def render_sidebar_html(raw_html: str) -> None:
    """Render cleaned HTML inside the Streamlit sidebar."""

    st.sidebar.markdown(
        clean_html(raw_html),
        unsafe_allow_html=True,
    )


def escape_text(value) -> str:
    """Escape text before placing it inside HTML."""

    return html.escape(str(value))


# --------------------------------------------------
# 4. MODEL AND METADATA PATH SELECTION
# --------------------------------------------------

@st.cache_resource
def load_active_production_bundle() -> Dict:
    """Load one validated production model and metadata pair.

    The shared resolver prefers the final tuned champion.
    It uses the earlier champion only when the complete final pair
    is unavailable.
    """

    return load_production_bundle()


def get_active_model_path() -> Path:
    """Return the model path selected by the shared resolver."""

    bundle = load_active_production_bundle()

    return Path(bundle["model_path"])


def get_active_metadata_path() -> Path:
    """Return the matching metadata path selected by the resolver."""

    bundle = load_active_production_bundle()

    return Path(bundle["metadata_path"])


@st.cache_resource
def load_champion_model():
    """Return the validated model used by prediction pages."""

    bundle = load_active_production_bundle()

    return bundle["model"]


@st.cache_data
def load_champion_metadata() -> Dict:
    """Return metadata belonging to the same active model generation."""

    bundle = load_active_production_bundle()

    # Copy the metadata so app-only display fields do not change the
    # original dictionary stored inside the cached production bundle.
    metadata = dict(bundle["metadata"])

    metadata["active_generation"] = bundle["generation"]
    metadata["active_model_path"] = str(bundle["model_path"])
    metadata["active_metadata_path"] = str(bundle["metadata_path"])

    return metadata


# --------------------------------------------------
# 5. CHAMPION MODEL INFORMATION
# --------------------------------------------------

def get_feature_columns() -> List[str]:
    """Return the exact feature columns expected by the active model."""

    metadata = load_champion_metadata()

    return metadata["feature_columns"]


def get_best_threshold() -> float:
    """Return the validated active production threshold."""

    metadata = load_champion_metadata()

    return get_production_threshold(metadata)


def get_champion_model_name() -> str:
    """Return a readable name for the active production model.

    The shared helper supports both final and earlier metadata formats.
    """

    metadata = load_champion_metadata()

    return get_production_model_name(metadata)


def get_champion_track() -> str:
    """Return the model-selection track name."""

    metadata = load_champion_metadata()

    if "final_model_name" in metadata:
        return "Final true champion"

    return metadata.get("champion_track", "Initial champion")


def get_champion_metrics() -> Dict:
    """Return champion metrics in a consistent dictionary.

    Old metadata stored metrics inside:
        metadata["metrics"]

    Final metadata stores metrics at the top level.
    This function supports both.
    """

    metadata = load_champion_metadata()

    if "metrics" in metadata and isinstance(metadata["metrics"], dict):
        return metadata["metrics"]

    metric_names = [
        "pr_auc",
        "roc_auc",
        "best_precision",
        "best_recall",
        "best_f1_score",
        "best_threshold",
    ]

    return {
        metric_name: metadata.get(metric_name)
        for metric_name in metric_names
        if metric_name in metadata
    }


def get_active_model_label() -> str:
    """Return a compact label for sidebar/dashboard use."""

    model_name = get_champion_model_name()
    threshold = get_best_threshold()

    return f"{model_name} | threshold {threshold:.2f}"


# --------------------------------------------------
# 6. PREDICTION INPUT PREPARATION
# --------------------------------------------------

def build_model_input(
    total_events: float,
    view_count: float,
    addtocart_count: float,
    unique_items: float,
    activity_span_ms: float,
) -> pd.DataFrame:
    """Create one visitor input row for model scoring."""

    # Use the same feature formulas as the training pipeline.
    # This prevents the same visitor receiving different offline and app scores.
    input_data = build_single_visitor_features(
        total_events=total_events,
        view_count=view_count,
        addtocart_count=addtocart_count,
        unique_items=unique_items,
        activity_span_ms=activity_span_ms,
    )

    # Keep the exact feature order expected by the active model metadata.
    feature_columns = get_feature_columns()

    return input_data[feature_columns]


def validate_batch_input(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a CSV uploaded for batch scoring."""

    errors: List[str] = []

    required_columns = [
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors

    for column in required_columns:
        converted_column = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if converted_column.isna().any():
            errors.append(f"Column '{column}' contains missing or non-numeric values.")

        if (converted_column < 0).any():
            errors.append(f"Column '{column}' contains negative values.")

    if (data["addtocart_count"] > data["total_events"]).any():
        errors.append("addtocart_count cannot be greater than total_events.")

    return len(errors) == 0, errors


def prepare_batch_model_input(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare many visitor rows for model scoring."""

    # The shared function validates numeric values and creates the same
    # derived features used by both training and single prediction.
    batch_data = prepare_model_features(data)

    # Keep the exact feature order expected by the active model metadata.
    feature_columns = get_feature_columns()

    return batch_data[feature_columns]


# --------------------------------------------------
# 7. PREDICTION LOGIC
# --------------------------------------------------

def predict_purchase_intent(model_input: pd.DataFrame) -> float:
    """Predict purchase probability for one visitor."""

    model = load_champion_model()
    probability = model.predict_proba(model_input)[:, 1][0]

    return float(probability)


def predict_batch_purchase_intent(model_input: pd.DataFrame) -> pd.Series:
    """Predict purchase probability for many visitors."""

    model = load_champion_model()
    probabilities = model.predict_proba(model_input)[:, 1]

    return pd.Series(probabilities)


# --------------------------------------------------
# 8. BUSINESS SEGMENTATION AND ACTIONS
# --------------------------------------------------

def assign_intent_segment(score: float) -> str:
    """Assign a business-friendly visitor segment."""

    threshold = get_best_threshold()

    if score >= threshold:
        return "High Intent"
    if score >= 0.80:
        return "Strong Intent"
    if score >= 0.50:
        return "Warm Intent"
    if score >= 0.20:
        return "Low Intent"

    return "Cold Visitor"


def assign_recommended_action(score: float) -> str:
    """Recommend the next business action for a visitor."""

    threshold = get_best_threshold()

    if score >= threshold:
        return "Prioritise immediately with conversion offer"
    if score >= 0.80:
        return "Retarget with urgency and cart reminder"
    if score >= 0.50:
        return "Nurture with product recommendation"
    if score >= 0.20:
        return "Keep in low-cost remarketing audience"

    return "Do not spend paid budget now"


def explain_prediction(score: float) -> str:
    """Explain the model score in simple business language."""

    threshold = get_best_threshold()

    if score >= threshold:
        return (
            "This visitor is above the final champion threshold. "
            "They should be treated as a high-priority conversion opportunity."
        )

    if score >= 0.80:
        return (
            "This visitor shows strong commercial intent, but they are below "
            "the strict final champion targeting threshold."
        )

    if score >= 0.50:
        return (
            "This visitor has some purchase signals. A lighter nurture action "
            "is safer than an expensive paid campaign."
        )

    if score >= 0.20:
        return "This visitor has weak intent. Use only low-cost remarketing."

    return "This visitor looks cold. Avoid spending campaign budget on them now."


# --------------------------------------------------
# 9. TABLE LOADING HELPERS
# --------------------------------------------------

def read_table_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV table if it exists, otherwise return an empty table."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_model_selection_tables() -> Dict[str, pd.DataFrame]:
    """Load all model-selection and MVD proof tables used by app pages."""

    return {
        "manual_results": read_table_if_exists(MANUAL_RESULTS_PATH),
        "automl_results": read_table_if_exists(AUTOML_RESULTS_PATH),
        "initial_final_selection": read_table_if_exists(INITIAL_FINAL_SELECTION_PATH),
        "old_champion_metrics": read_table_if_exists(OLD_CHAMPION_METRICS_PATH),
        "old_champion_threshold": read_table_if_exists(OLD_CHAMPION_THRESHOLD_PATH),
        "final_true_champion_comparison": read_table_if_exists(FINAL_TRUE_CHAMPION_COMPARISON_PATH),
        "final_true_champion_summary": read_table_if_exists(FINAL_TRUE_CHAMPION_SUMMARY_PATH),
        "final_true_champion_thresholds": read_table_if_exists(FINAL_TRUE_CHAMPION_THRESHOLDS_PATH),
        "final_true_champion_stability": read_table_if_exists(FINAL_TRUE_CHAMPION_STABILITY_PATH),
        "final_true_champion_sensitivity": read_table_if_exists(FINAL_TRUE_CHAMPION_SENSITIVITY_PATH),
        "mvd_coverage_matrix": read_table_if_exists(MVD_COVERAGE_MATRIX_PATH),
        "mvd_pca_variance": read_table_if_exists(MVD_PCA_VARIANCE_PATH),
        "mvd_kmeans_summary": read_table_if_exists(MVD_KMEANS_SUMMARY_PATH),
        "mvd_dbscan_summary": read_table_if_exists(MVD_DBSCAN_SUMMARY_PATH),
        "mvd_lof_summary": read_table_if_exists(MVD_LOF_SUMMARY_PATH),
        "mvd_outlier_comparison": read_table_if_exists(MVD_OUTLIER_COMPARISON_PATH),
        "ml_validation_projection": read_table_if_exists(ML_VALIDATION_PROJECTION_PATH),
        "ml_validation_pca_variance": read_table_if_exists(ML_VALIDATION_PCA_VARIANCE_PATH),
        "ml_validation_pca_loadings": read_table_if_exists(ML_VALIDATION_PCA_LOADINGS_PATH),
        "ml_validation_cluster_profile": read_table_if_exists(ML_VALIDATION_CLUSTER_PROFILE_PATH),
        "ml_validation_method_summary": read_table_if_exists(ML_VALIDATION_METHOD_SUMMARY_PATH),
    }


def create_master_model_comparison_table() -> pd.DataFrame:
    """Return the best available model comparison table.

    Priority:
        1. final_true_champion_comparison.csv
        2. manual + AutoML benchmark tables
    """

    tables = load_model_selection_tables()

    final_comparison = tables["final_true_champion_comparison"]

    if not final_comparison.empty:
        comparison = final_comparison.copy()

        # Normalize columns so older benchmark pages can still rank models.
        if "business_score" in comparison.columns:
            comparison["champion_score"] = comparison["business_score"]

        if "status" not in comparison.columns:
            comparison["status"] = "success"

        if "track" not in comparison.columns:
            comparison["track"] = "final_true_champion"

        final_model_name = get_champion_model_name()

        comparison["is_final_champion"] = (
            comparison["model_name"] == final_model_name
        )

        comparison = comparison.sort_values(
            ["champion_score", "pr_auc", "best_f1_score", "best_precision"],
            ascending=False,
        ).reset_index(drop=True)

        comparison["model_rank"] = comparison.index + 1

        return comparison

    manual_results = tables["manual_results"]
    automl_results = tables["automl_results"]

    all_results = pd.concat(
        [manual_results, automl_results],
        ignore_index=True,
    )

    if all_results.empty:
        return all_results

    if "status" in all_results.columns:
        all_results = all_results[all_results["status"] == "success"].copy()

    all_results = all_results.sort_values(
        ["champion_score", "pr_auc", "best_f1_score", "best_precision"],
        ascending=False,
    ).reset_index(drop=True)

    all_results["model_rank"] = all_results.index + 1

    metadata = load_champion_metadata()
    champion_model_name = (
        metadata.get("final_model_name")
        or metadata.get("champion_model_name")
    )
    champion_track = metadata.get("champion_track", "final_true_champion")

    if "track" in all_results.columns:
        all_results["is_final_champion"] = (
            (all_results["model_name"] == champion_model_name)
            & (all_results["track"] == champion_track)
        )
    else:
        all_results["is_final_champion"] = (
            all_results["model_name"] == champion_model_name
        )

    return all_results


# --------------------------------------------------
# 10. FORMATTING HELPERS
# --------------------------------------------------

def is_missing_number(value) -> bool:
    """Check whether a value is missing or not numeric."""

    try:
        return value is None or math.isnan(float(value))
    except Exception:
        return True


def format_percent(value: float, decimals: int = 1) -> str:
    """Format decimal value as percentage text."""

    if is_missing_number(value):
        return "NA"

    return f"{float(value) * 100:.{decimals}f}%"


def format_score(value: float, decimals: int = 3) -> str:
    """Format model score text."""

    if is_missing_number(value):
        return "NA"

    return f"{float(value):.{decimals}f}"


def format_lift(value: float, decimals: int = 1) -> str:
    """Format lift value."""

    if is_missing_number(value):
        return "NA"

    return f"{float(value):.{decimals}f}x"


def format_seconds(value: float, decimals: int = 2) -> str:
    """Format training time."""

    if is_missing_number(value):
        return "NA"

    return f"{float(value):.{decimals}f}s"


# --------------------------------------------------
# 11. SHARED DASHBOARD CSS
# --------------------------------------------------

def inject_global_css() -> None:
    """Inject reusable dark-mode dashboard styling."""

    render_html(
        """
        <style>
            .hero-card {
                padding: 34px 36px;
                border-radius: 28px;
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.24), transparent 34%),
                    radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.18), transparent 30%),
                    linear-gradient(135deg, #0f172a 0%, #111827 52%, #020617 100%);
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.40);
                margin-bottom: 24px;
            }

            .eyebrow {
                color: #38BDF8;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }

            .hero-title {
                color: #F8FAFC;
                font-size: 2.35rem;
                line-height: 1.08;
                font-weight: 900;
                margin-bottom: 12px;
            }

            .hero-subtitle {
                color: #CBD5E1;
                font-size: 1.02rem;
                line-height: 1.65;
                max-width: 980px;
            }

            .metric-card {
                padding: 22px 22px;
                border-radius: 22px;
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.94));
                border: 1px solid rgba(148, 163, 184, 0.20);
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
                min-height: 138px;
            }

            .metric-label {
                color: #94A3B8;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .metric-value {
                color: #F8FAFC;
                font-size: 1.92rem;
                font-weight: 900;
                margin-bottom: 6px;
            }

            .metric-help {
                color: #CBD5E1;
                font-size: 0.88rem;
                line-height: 1.45;
            }

            .insight-card {
                padding: 24px 26px;
                border-radius: 24px;
                background: rgba(15, 23, 42, 0.86);
                border: 1px solid rgba(148, 163, 184, 0.20);
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
                margin-top: 16px;
                margin-bottom: 16px;
            }

            .insight-title {
                color: #F8FAFC;
                font-size: 1.22rem;
                font-weight: 900;
                margin-bottom: 8px;
            }

            .insight-text {
                color: #CBD5E1;
                font-size: 0.96rem;
                line-height: 1.65;
            }

            .success-pill {
                display: inline-block;
                padding: 7px 11px;
                border-radius: 999px;
                background: rgba(34, 197, 94, 0.14);
                border: 1px solid rgba(34, 197, 94, 0.28);
                color: #86EFAC;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            .info-pill {
                display: inline-block;
                padding: 7px 11px;
                border-radius: 999px;
                background: rgba(56, 189, 248, 0.13);
                border: 1px solid rgba(56, 189, 248, 0.28);
                color: #7DD3FC;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            .warning-pill {
                display: inline-block;
                padding: 7px 11px;
                border-radius: 999px;
                background: rgba(245, 158, 11, 0.13);
                border: 1px solid rgba(245, 158, 11, 0.28);
                color: #FCD34D;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
        </style>
        """
    )
