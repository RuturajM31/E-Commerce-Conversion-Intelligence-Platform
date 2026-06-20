# build_anomaly_signals.py
# Build anomaly and outlier signals for visitor behaviour.
#
# SIMPLE IDEA:
#   The purchase-intent model answers:
#       "Who is likely to buy?"
#
#   The anomaly layer answers:
#       "Who behaves unusually?"
#
# WHY THIS MATTERS:
#   Some visitors can have strange behaviour:
#       - too many events
#       - too many add-to-cart actions
#       - very long sessions
#       - unusually high intent score with strange activity
#
#   These are not always bad.
#   They may be:
#       - very strong buyers
#       - bots
#       - crawlers
#       - broken tracking
#       - unusual campaign opportunities
#
# RUN COMMAND:
#   python3 -m src.anomaly.build_anomaly_signals
#
# OUTPUTS:
#   data/processed/final_champion_visitor_scores.csv
#   data/processed/visitor_anomaly_scores.csv
#   reports/tables/anomaly_summary.csv
#   reports/tables/anomaly_rule_summary.csv
#   reports/tables/anomaly_segment_summary.csv
#   reports/tables/top_anomalous_visitors.csv

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models.production_model import (
    get_production_threshold,
    load_production_bundle,
)
from src.models.score_export import (
    FINAL_SCORE_MANIFEST_PATH,
    FINAL_SCORE_PATH,
    load_valid_cached_scores,
    save_final_champion_scores,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

VISITOR_FEATURES_PATH = Path("data/processed/visitor_features.csv")

# Scores created from the shared active production model.
PRODUCTION_VISITOR_SCORES_PATH = FINAL_SCORE_PATH
PRODUCTION_SCORE_MANIFEST_PATH = FINAL_SCORE_MANIFEST_PATH
VISITOR_ANOMALY_SCORES_PATH = Path("data/processed/visitor_anomaly_scores.csv")

REPORT_TABLES_DIR = Path("reports/tables")

ANOMALY_SUMMARY_PATH = REPORT_TABLES_DIR / "anomaly_summary.csv"
ANOMALY_RULE_SUMMARY_PATH = REPORT_TABLES_DIR / "anomaly_rule_summary.csv"
ANOMALY_SEGMENT_SUMMARY_PATH = REPORT_TABLES_DIR / "anomaly_segment_summary.csv"
TOP_ANOMALOUS_VISITORS_PATH = REPORT_TABLES_DIR / "top_anomalous_visitors.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42

# We do not want to mark too many users as anomalies.
# 2% is a realistic starting point for unusual ecommerce behaviour.
ISOLATION_FOREST_CONTAMINATION = 0.02

# Use a sample for Isolation Forest training so the script stays fast.
# Then score all visitors.
ISOLATION_FOREST_TRAIN_SAMPLE_SIZE = 200_000

# Save top rows for dashboard performance.
TOP_ANOMALY_ROWS = 10_000


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def create_output_folders() -> None:
    """Create output folders."""

    PRODUCTION_VISITOR_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VISITOR_ANOMALY_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_columns(
    visitor_features: pd.DataFrame,
    production_bundle: Dict,
) -> List[str]:
    """Validate the features required by the active production model."""

    feature_columns = production_bundle["feature_columns"]

    missing_columns = [
        column for column in feature_columns
        if column not in visitor_features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"visitor_features.csv missing model feature columns: {missing_columns}"
        )

    return feature_columns


def load_visitor_features() -> pd.DataFrame:
    """Load visitor-level feature table."""

    if not VISITOR_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "visitor_features.csv not found. Run feature engineering first."
        )

    visitor_features = pd.read_csv(VISITOR_FEATURES_PATH)

    if "visitorid" not in visitor_features.columns:
        raise ValueError("visitor_features.csv must contain visitorid.")

    return visitor_features


def create_champion_visitor_scores(visitor_features: pd.DataFrame) -> pd.DataFrame:
    """Load validated cached scores or regenerate production scores."""

    production_bundle = load_production_bundle()

    cached_scores, validation_message = load_valid_cached_scores(
        production_bundle,
        visitor_features["visitorid"],
        score_path=PRODUCTION_VISITOR_SCORES_PATH,
        manifest_path=PRODUCTION_SCORE_MANIFEST_PATH,
    )

    required_columns = ["visitorid", "purchase_intent_score"]

    if cached_scores is not None:
        print(validation_message)
        return cached_scores[required_columns].copy()

    print(
        "Cached final champion scores are not reusable: "
        f"{validation_message}. Regenerating."
    )

    feature_columns = get_feature_columns(
        visitor_features,
        production_bundle,
    )

    X = visitor_features[feature_columns].copy()

    score_table, _ = save_final_champion_scores(
        final_model=production_bundle["model"],
        X=X,
        visitor_ids=visitor_features["visitorid"],
        threshold=get_production_threshold(
            production_bundle["metadata"]
        ),
        model_path=production_bundle["model_path"],
        metadata_path=production_bundle["metadata_path"],
        score_path=PRODUCTION_VISITOR_SCORES_PATH,
        manifest_path=PRODUCTION_SCORE_MANIFEST_PATH,
        model_generation=production_bundle["generation"],
        chunk_size=100_000,
    )

    return score_table[required_columns].copy()


def add_percentile_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Add percentile rank columns for important behavioural variables."""

    output = data.copy()

    for column in columns:
        percentile_column = f"{column}_percentile"

        output[percentile_column] = output[column].rank(
            method="average",
            pct=True,
        )

    return output


def assign_purchase_segment(score: float) -> str:
    """Assign purchase-intent segment from champion score."""

    if score >= 0.95:
        return "High Intent"

    if score >= 0.80:
        return "Strong Intent"

    if score >= 0.50:
        return "Warm Intent"

    if score >= 0.20:
        return "Low Intent"

    return "Cold Visitor"


# --------------------------------------------------
# Rule-based anomaly logic
# --------------------------------------------------

def add_rule_based_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable anomaly flags using business rules."""

    output = data.copy()

    # Percentile columns make the rules easy to explain.
    # Example:
    #   total_events_percentile >= 0.99 means top 1% by total events.
    percentile_columns = [
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "cart_to_view_ratio",
        "events_per_unique_item",
        "purchase_intent_score",
    ]

    output = add_percentile_columns(output, percentile_columns)

    # Rule 1:
    # Visitor has extremely high total activity.
    output["rule_extreme_total_events"] = output["total_events_percentile"] >= 0.99

    # Rule 2:
    # Visitor added to cart unusually often.
    output["rule_extreme_cart_activity"] = (
        (output["addtocart_count_percentile"] >= 0.99)
        & (output["addtocart_count"] > 0)
    )

    # Rule 3:
    # Visitor viewed or interacted with unusually many unique items.
    output["rule_extreme_product_browsing"] = output["unique_items_percentile"] >= 0.99

    # Rule 4:
    # Visitor has unusually long activity span.
    output["rule_extreme_session_span"] = output["activity_span_ms_percentile"] >= 0.99

    # Rule 5:
    # Visitor is high intent but behaviour is also extreme.
    # This is important for business review:
    #   Could be a very hot buyer.
    #   Could also be suspicious or abnormal.
    output["rule_high_intent_extreme_activity"] = (
        (output["purchase_intent_score"] >= 0.95)
        & (
            output["rule_extreme_total_events"]
            | output["rule_extreme_cart_activity"]
            | output["rule_extreme_product_browsing"]
            | output["rule_extreme_session_span"]
        )
    )

    # Rule 6:
    # Very high cart-to-view ratio.
    # Example:
    #   many add-to-cart actions compared with views.
    output["rule_unusual_cart_to_view_ratio"] = (
        (output["cart_to_view_ratio_percentile"] >= 0.99)
        & (output["cart_to_view_ratio"] > 0)
    )

    rule_columns = [
        "rule_extreme_total_events",
        "rule_extreme_cart_activity",
        "rule_extreme_product_browsing",
        "rule_extreme_session_span",
        "rule_high_intent_extreme_activity",
        "rule_unusual_cart_to_view_ratio",
    ]

    output["rule_anomaly_count"] = output[rule_columns].sum(axis=1)

    output["has_rule_based_anomaly"] = output["rule_anomaly_count"] > 0

    return output


# --------------------------------------------------
# Isolation Forest anomaly detection
# --------------------------------------------------

def add_isolation_forest_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Add Isolation Forest anomaly score."""

    output = data.copy()

    model_features = [
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "cart_to_view_ratio",
        "events_per_unique_item",
        "purchase_intent_score",
    ]

    X = output[model_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train on a sample for speed.
    if len(output) > ISOLATION_FOREST_TRAIN_SAMPLE_SIZE:
        sample_indices = output.sample(
            n=ISOLATION_FOREST_TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).index

        X_train = X_scaled[sample_indices]
    else:
        X_train = X_scaled

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Isolation Forest anomaly detector...")
    isolation_forest.fit(X_train)

    # sklearn returns:
    #   1 = normal
    #  -1 = anomaly
    predicted_labels = isolation_forest.predict(X_scaled)

    # decision_function:
    #   lower values = more abnormal.
    raw_scores = isolation_forest.decision_function(X_scaled)

    output["isolation_forest_label"] = predicted_labels
    output["isolation_forest_anomaly"] = predicted_labels == -1

    # Convert raw score into easier dashboard score:
    #   higher = more anomalous.
    output["isolation_anomaly_score"] = -raw_scores

    output["isolation_anomaly_percentile"] = output["isolation_anomaly_score"].rank(
        method="average",
        pct=True,
    )

    return output


# --------------------------------------------------
# Final risk score and labels
# --------------------------------------------------

def add_final_anomaly_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Create final anomaly risk score and business labels."""

    output = data.copy()

    # Combine machine anomaly percentile with rule count.
    # This gives both:
    #   - model-based anomaly signal
    #   - business-readable rule signal
    output["anomaly_risk_score"] = (
        0.70 * output["isolation_anomaly_percentile"]
        + 0.30 * np.minimum(output["rule_anomaly_count"] / 3, 1)
    )

    output["final_anomaly_flag"] = (
        output["isolation_forest_anomaly"]
        | output["has_rule_based_anomaly"]
        | (output["anomaly_risk_score"] >= 0.98)
    )

    def assign_risk_band(score: float) -> str:
        if score >= 0.98:
            return "Critical Review"

        if score >= 0.95:
            return "High Review"

        if score >= 0.90:
            return "Medium Review"

        return "Normal"

    output["anomaly_risk_band"] = output["anomaly_risk_score"].apply(assign_risk_band)

    output["purchase_intent_segment"] = output["purchase_intent_score"].apply(
        assign_purchase_segment
    )

    def assign_business_interpretation(row: pd.Series) -> str:
        if row["rule_high_intent_extreme_activity"]:
            return "High-value but unusual visitor. Review before campaign activation."

        if row["rule_extreme_cart_activity"]:
            return "Unusually high cart behaviour. Possible hot buyer or abnormal tracking."

        if row["rule_extreme_total_events"]:
            return "Extreme activity volume. Possible bot, crawler, or power user."

        if row["rule_extreme_product_browsing"]:
            return "Unusually broad product browsing. Review for intent or automation."

        if row["rule_extreme_session_span"]:
            return "Unusually long activity span. Review session quality."

        if row["isolation_forest_anomaly"]:
            return "Machine anomaly detector marked this visitor as unusual."

        return "No major anomaly signal."

    output["business_interpretation"] = output.apply(
        assign_business_interpretation,
        axis=1,
    )

    return output


# --------------------------------------------------
# Reporting tables
# --------------------------------------------------

def create_summary_tables(anomaly_data: pd.DataFrame) -> None:
    """Create summary tables for Streamlit dashboard."""

    total_visitors = len(anomaly_data)
    final_anomalies = int(anomaly_data["final_anomaly_flag"].sum())
    isolation_anomalies = int(anomaly_data["isolation_forest_anomaly"].sum())
    rule_anomalies = int(anomaly_data["has_rule_based_anomaly"].sum())
    high_intent_anomalies = int(
        (
            anomaly_data["final_anomaly_flag"]
            & (anomaly_data["purchase_intent_segment"] == "High Intent")
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "total_visitors",
                "value": total_visitors,
                "description": "Total visitors analysed.",
            },
            {
                "metric": "final_anomalies",
                "value": final_anomalies,
                "description": "Visitors flagged by final anomaly logic.",
            },
            {
                "metric": "final_anomaly_rate",
                "value": final_anomalies / total_visitors if total_visitors else 0,
                "description": "Share of visitors flagged as anomalous.",
            },
            {
                "metric": "isolation_forest_anomalies",
                "value": isolation_anomalies,
                "description": "Visitors flagged by Isolation Forest.",
            },
            {
                "metric": "rule_based_anomalies",
                "value": rule_anomalies,
                "description": "Visitors flagged by interpretable business rules.",
            },
            {
                "metric": "high_intent_anomalies",
                "value": high_intent_anomalies,
                "description": "High-intent visitors that also look unusual.",
            },
        ]
    )

    rule_columns = [
        "rule_extreme_total_events",
        "rule_extreme_cart_activity",
        "rule_extreme_product_browsing",
        "rule_extreme_session_span",
        "rule_high_intent_extreme_activity",
        "rule_unusual_cart_to_view_ratio",
    ]

    rule_summary_rows = []

    for rule_column in rule_columns:
        flagged_count = int(anomaly_data[rule_column].sum())

        rule_summary_rows.append(
            {
                "rule_name": rule_column,
                "flagged_count": flagged_count,
                "flagged_rate": flagged_count / total_visitors if total_visitors else 0,
            }
        )

    rule_summary = pd.DataFrame(rule_summary_rows)

    segment_summary = (
        anomaly_data
        .groupby("purchase_intent_segment")
        .agg(
            visitors=("visitorid", "count"),
            final_anomalies=("final_anomaly_flag", "sum"),
            avg_purchase_intent_score=("purchase_intent_score", "mean"),
            avg_anomaly_risk_score=("anomaly_risk_score", "mean"),
            avg_total_events=("total_events", "mean"),
            avg_addtocart_count=("addtocart_count", "mean"),
        )
        .reset_index()
    )

    segment_summary["anomaly_rate"] = (
        segment_summary["final_anomalies"] / segment_summary["visitors"]
    )

    top_anomalies = (
        anomaly_data
        .sort_values("anomaly_risk_score", ascending=False)
        .head(TOP_ANOMALY_ROWS)
        .copy()
    )

    top_columns = [
        "visitorid",
        "purchase_intent_score",
        "purchase_intent_segment",
        "anomaly_risk_score",
        "anomaly_risk_band",
        "final_anomaly_flag",
        "isolation_forest_anomaly",
        "rule_anomaly_count",
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "cart_to_view_ratio",
        "events_per_unique_item",
        "business_interpretation",
        "rule_extreme_total_events",
        "rule_extreme_cart_activity",
        "rule_extreme_product_browsing",
        "rule_extreme_session_span",
        "rule_high_intent_extreme_activity",
        "rule_unusual_cart_to_view_ratio",
    ]

    available_top_columns = [
        column for column in top_columns
        if column in top_anomalies.columns
    ]

    top_anomalies = top_anomalies[available_top_columns]

    summary.to_csv(ANOMALY_SUMMARY_PATH, index=False)
    rule_summary.to_csv(ANOMALY_RULE_SUMMARY_PATH, index=False)
    segment_summary.to_csv(ANOMALY_SEGMENT_SUMMARY_PATH, index=False)
    top_anomalies.to_csv(TOP_ANOMALOUS_VISITORS_PATH, index=False)

    print(f"Saved anomaly summary to: {ANOMALY_SUMMARY_PATH}")
    print(f"Saved anomaly rule summary to: {ANOMALY_RULE_SUMMARY_PATH}")
    print(f"Saved anomaly segment summary to: {ANOMALY_SEGMENT_SUMMARY_PATH}")
    print(f"Saved top anomalies to: {TOP_ANOMALOUS_VISITORS_PATH}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run complete anomaly detection pipeline."""

    print("Starting anomaly and outlier signal pipeline...")

    create_output_folders()

    visitor_features = load_visitor_features()

    print(f"Loaded visitor features: {visitor_features.shape}")

    champion_scores = create_champion_visitor_scores(visitor_features)

    anomaly_data = visitor_features.merge(
        champion_scores,
        on="visitorid",
        how="left",
    )

    anomaly_data["purchase_intent_score"] = anomaly_data["purchase_intent_score"].fillna(0)

    print("Creating rule-based anomaly flags...")
    anomaly_data = add_rule_based_anomalies(anomaly_data)

    anomaly_data = add_isolation_forest_scores(anomaly_data)

    print("Creating final anomaly labels...")
    anomaly_data = add_final_anomaly_labels(anomaly_data)

    print("Saving full visitor anomaly score table...")
    anomaly_data.to_csv(VISITOR_ANOMALY_SCORES_PATH, index=False)

    create_summary_tables(anomaly_data)

    final_anomaly_count = int(anomaly_data["final_anomaly_flag"].sum())
    final_anomaly_rate = final_anomaly_count / len(anomaly_data)

    print("\nAnomaly pipeline complete.")
    print(f"Visitors analysed: {len(anomaly_data):,}")
    print(f"Final anomalies: {final_anomaly_count:,}")
    print(f"Final anomaly rate: {final_anomaly_rate:.2%}")


if __name__ == "__main__":
    main()
