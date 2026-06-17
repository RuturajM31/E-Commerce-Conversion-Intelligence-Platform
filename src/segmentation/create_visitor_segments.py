# create_visitor_segments.py
# This script scores visitors and converts model probabilities into business segments.
#
# Simple idea:
#   The model gives every visitor a purchase intent score.
#   We convert that score into readable business segments.

import pandas as pd
import joblib

from src.config.paths import PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, TABLES_DIR


# Input files
FEATURE_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"
MODEL_FILE = TRAINED_MODELS_DIR / "baseline_logistic_regression.joblib"

# Output files
SCORES_FILE = PROCESSED_DATA_DIR / "visitor_scores.csv"
SAMPLE_FILE = PROCESSED_DATA_DIR / "visitor_scores_sample.csv"
SEGMENT_SUMMARY_FILE = TABLES_DIR / "segment_summary.csv"


def load_data_and_model():
    """Load visitor features and trained model."""

    # Load visitor-level feature data.
    data = pd.read_csv(FEATURE_FILE)

    # Load trained model pipeline.
    model = joblib.load(MODEL_FILE)

    return data, model


def create_features(data):
    """Select the same feature columns used during training."""

    feature_columns = [
        "total_events",
        "view_count",
        "addtocart_count",
        "unique_items",
        "activity_span_ms",
        "cart_to_view_ratio",
        "events_per_unique_item",
    ]

    X = data[feature_columns]

    return X


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


def score_visitors(data, model):
    """Create purchase intent score and segment for each visitor."""

    # Select model input features.
    X = create_features(data)

    # Predict probability of conversion.
    data["purchase_intent_score"] = model.predict_proba(X)[:, 1]

    # Convert probability into readable segment.
    data["intent_segment"] = data["purchase_intent_score"].apply(assign_segment)

    # Add recommended business action.
    data["recommended_action"] = data["intent_segment"].apply(assign_action)

    return data


def create_segment_summary(scored_data):
    """Create segment-level business summary."""

    summary = (
        scored_data.groupby("intent_segment")
        .agg(
            visitors=("visitorid", "count"),
            actual_converters=("converted", "sum"),
            conversion_rate=("converted", "mean"),
            avg_score=("purchase_intent_score", "mean"),
        )
        .reset_index()
    )

    # Add visitor share.
    summary["visitor_share"] = summary["visitors"] / summary["visitors"].sum()

    # Sort segments by average score.
    summary = summary.sort_values("avg_score", ascending=False)

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

    # Load data and trained model.
    data, model = load_data_and_model()

    # Score visitors and assign segments.
    scored_data = score_visitors(data, model)

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