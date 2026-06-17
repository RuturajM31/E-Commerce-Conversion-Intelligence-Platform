# prediction_logger.py
# Logging helper for the E-Commerce Conversion Intelligence Platform.
#
# PURPOSE:
#   Save prediction activity from the Streamlit app.
#
# WHY THIS MATTERS:
#   A normal ML demo only shows predictions.
#   A production-style ML system also records what happened:
#       - when prediction happened
#       - which page made the prediction
#       - what score was produced
#       - what segment was assigned
#       - what action was recommended
#
# BUSINESS VALUE:
#   These logs help us monitor model usage, campaign decisions,
#   prediction patterns, and future data drift.

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Prediction logs are saved here.
# This folder already exists in our project structure.

PREDICTION_LOG_DIR = Path("monitoring/prediction_logs")

# Main prediction log file.

PREDICTION_LOG_FILE = PREDICTION_LOG_DIR / "prediction_log.csv"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def ensure_log_folder_exists():
    
    """Create the prediction log folder if it does not exist."""

    # WHAT:
    #   Make sure monitoring/prediction_logs exists.
    #
    # WHY:
    #   If the folder is missing, saving the CSV would fail.

    PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_current_utc_timestamp():
    """Return current timestamp in UTC format."""

    # WHAT:
    #   Create a standard timestamp.
    #
    # WHY:
    #   Monitoring systems usually prefer UTC timestamps.

    return datetime.now(timezone.utc).isoformat()


def append_prediction_log(log_row):
    
    """Append one prediction log row to the prediction log CSV."""

    # WHAT:
    #   Add one new row to prediction_log.csv.
    #
    # INPUT:
    #   log_row = dictionary containing prediction details.
    #
    # OUTPUT:
    #   monitoring/prediction_logs/prediction_log.csv
    #
    # WHY:
    #   This creates a simple prediction history that later monitoring tools can read.

    ensure_log_folder_exists()

    new_log_data = pd.DataFrame([log_row])

    if PREDICTION_LOG_FILE.exists():
        old_log_data = pd.read_csv(PREDICTION_LOG_FILE)
        full_log_data = pd.concat(
            [old_log_data, new_log_data],
            ignore_index=True,
        )
    else:
        full_log_data = new_log_data

    full_log_data.to_csv(PREDICTION_LOG_FILE, index=False)


def log_single_prediction(
    page_name,
    visitor_id,
    purchase_intent_score,
    intent_segment,
    recommended_action,
    input_features,
):
    """Log one single-visitor prediction."""

    # WHAT:
    
    #   Save one prediction from the Visitor Intent Predictor page.
    #
    # INPUTS:
    #   page_name              = where the prediction came from
    #   visitor_id             = visitor identifier or manual visitor label
    #   purchase_intent_score  = model probability between 0 and 1
    #   intent_segment         = High Intent / Warm Intent / Cold Visitor etc.
    #   recommended_action     = marketing action created by the app
    #   input_features         = dictionary of model input values
    #
    # OUTPUT:
    #   One new row in prediction_log.csv

    log_row = {
        "timestamp_utc": get_current_utc_timestamp(),
        "page_name": page_name,
        "prediction_type": "single_prediction",
        "visitor_id": str(visitor_id),
        "purchase_intent_score": round(float(purchase_intent_score), 6),
        "purchase_intent_score_pct": round(float(purchase_intent_score) * 100, 2),
        "intent_segment": intent_segment,
        "recommended_action": recommended_action,
    }

    # Add input features into the same log row.
    # This helps later with drift checks and debugging.
    
    for feature_name, feature_value in input_features.items():
        log_row[feature_name] = feature_value

    append_prediction_log(log_row)


def log_batch_summary(
    page_name,
    total_visitors,
    high_intent_count,
    average_score,
    source_label,
):
    """Log one batch scoring summary."""

    # WHAT:
    #   Save summary information from the Batch Scoring page.
    #
    # WHY:
    #   For batch scoring, logging every single row may become large.
    #   A summary log is enough for app monitoring and demo dashboards.

    high_intent_share = (
        high_intent_count / total_visitors
        if total_visitors > 0
        else 0
    )

    log_row = {
        "timestamp_utc": get_current_utc_timestamp(),
        "page_name": page_name,
        "prediction_type": "batch_summary",
        "visitor_id": "batch",
        "source_label": source_label,
        "total_visitors": int(total_visitors),
        "high_intent_count": int(high_intent_count),
        "high_intent_share": round(float(high_intent_share), 6),
        "high_intent_share_pct": round(float(high_intent_share) * 100, 2),
        "purchase_intent_score": round(float(average_score), 6),
        "purchase_intent_score_pct": round(float(average_score) * 100, 2),
        "intent_segment": "batch_summary",
        "recommended_action": "Review High Intent visitors first",
    }

    append_prediction_log(log_row)


def read_prediction_log():
    """Read the prediction log file."""

    # WHAT:
    #   Load prediction_log.csv if it exists.
    #
    # WHY:
    #   Later dashboards can use this function to show recent predictions.

    if PREDICTION_LOG_FILE.exists():
        return pd.read_csv(PREDICTION_LOG_FILE)

    return pd.DataFrame()