# create_visitor_segments.py
# This script scores visitors and converts model probabilities into business segments.
#
# Simple idea:
#   The model gives every visitor a purchase intent score.
#   We convert that score into readable business segments.

import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR, TABLES_DIR
from src.models.production_model import load_production_bundle


# Input file
FEATURE_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"

# Output files
SCORES_FILE = PROCESSED_DATA_DIR / "visitor_scores.csv"
SAMPLE_FILE = PROCESSED_DATA_DIR / "visitor_scores_sample.csv"
SEGMENT_SUMMARY_FILE = TABLES_DIR / "segment_summary.csv"


def load_data_and_model():
    """Load visitor features and the active production model bundle."""

    # Load visitor-level feature data.
    data = pd.read_csv(FEATURE_FILE)

    # Use the same validated model and metadata pair as the app,
    # anomaly workflow, and forecasting workflow.
    production_bundle = load_production_bundle()

    return data, production_bundle


def create_features(data, production_bundle):
    """Select and validate features required by the production model."""

    feature_columns = production_bundle["feature_columns"]

    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Visitor feature data is missing columns: {missing_columns}"
        )

    return data[feature_columns].copy()


def assign_segment(score):
    """Convert purchase intent score into a business segment."""

    if score >= 0.90:
        return "High Intent"
    elif score >= 0.70:
        return "Strong Intent"
    elif score >= 0.50:
        return "Warm Intent"
    elif score >= 0.20:
        return "Low Intent"
    else:
        return "Cold Visitor"


def assign_action(segment):
    """Recommend a marketing action for each segment."""

    actions = {
        "High Intent": "Target immediately with personalised offer",
        "Strong Intent": "Retarget with product reminder",
        "Warm Intent": "Nurture with email or recommendation",
        "Low Intent": "Low-cost awareness campaign only",
        "Cold Visitor": "Do not prioritise paid marketing",
    }

    return actions[segment]


def score_visitors(data, production_bundle):
    """Create purchase intent score and segment for each visitor."""

    # Select the exact features expected by the production model.
    X = create_features(data, production_bundle)

    # Predict probability of conversion.
    model = production_bundle["model"]
    data["purchase_intent_score"] = model.predict_proba(X)[:, 1]

    # Convert probability into readable segment.
    data["intent_segment"] = data["purchase_intent_score"].apply(assign_segment)

    # Add recommended business action.
    data["recommended_action"] = data["intent_segment"].apply(assign_action)

    return data


def create_segment_summary(
    scored_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a production-safe segment summary.

    The current scoring table contains only information available now.
    Future conversion labels are therefore optional and normally absent.
    """

    summary = (
        scored_data.groupby("intent_segment")
        .agg(
            visitors=("visitorid", "count"),
            avg_score=(
                "purchase_intent_score",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["visitor_share"] = (
        summary["visitors"]
        / summary["visitors"].sum()
    )

    if "converted" in scored_data.columns:
        actuals = (
            scored_data.groupby("intent_segment")
            .agg(
                actual_converters=(
                    "converted",
                    "sum",
                ),
                conversion_rate=(
                    "converted",
                    "mean",
                ),
            )
            .reset_index()
        )

        summary = summary.merge(
            actuals,
            on="intent_segment",
            how="left",
        )
    else:
        summary["actual_converters"] = pd.NA
        summary["conversion_rate"] = pd.NA

    summary = summary.sort_values(
        "avg_score",
        ascending=False,
    )

    return summary


def save_outputs(scored_data, segment_summary):
    """Save scored visitor data and segment summary."""

    # Save full scored visitor file.
    scored_data.to_csv(SCORES_FILE, index=False)

    # Save smaller sample for Streamlit app and quick demo.
    scored_data.sample(n=20000, random_state=42).to_csv(SAMPLE_FILE, index=False)

    # Save segment summary table.
    segment_summary.to_csv(SEGMENT_SUMMARY_FILE, index=False)


def main():
    """Run full visitor scoring and segmentation workflow."""

    # Load data and the shared production model bundle.
    data, production_bundle = load_data_and_model()

    # Score visitors and assign segments.
    scored_data = score_visitors(data, production_bundle)

    # Create business summary by segment.
    segment_summary = create_segment_summary(scored_data)

    # Save outputs.
    save_outputs(scored_data, segment_summary)

    # Print clean output.
    print("Visitor scoring and segmentation completed.")
    print("\nSegment summary:")
    print(segment_summary.round(4))
    print("\nSaved full scores to:", SCORES_FILE)
    print("Saved sample scores to:", SAMPLE_FILE)
    print("Saved segment summary to:", SEGMENT_SUMMARY_FILE)


if __name__ == "__main__":
    main()