# paths.py
# This file stores the main folder paths used in the project.
# We keep paths here so the rest of the code stays clean and portable.

from pathlib import Path


# PROJECT_ROOT means the main project folder.
# This file is inside: src/config/paths.py
# parents[2] moves two levels up:
# paths.py -> config -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "retailrocket"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Model folders
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"


# Report output folders
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


# Log and monitoring folders
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "monitoring"

# Prediction monitoring folders
# prediction_logs stores every production scoring record.
PREDICTION_LOGS_DIR = MONITORING_DIR / "prediction_logs"

# delayed_labels stores future conversion outcomes and evaluation outputs.
DELAYED_LABELS_DIR = MONITORING_DIR / "delayed_labels"


# Delayed-label monitoring files
# The ledger links each prediction to its model, threshold, and scoring window.
PREDICTION_LEDGER_PATH = PREDICTION_LOGS_DIR / "prediction_ledger.csv"

# This is the controlled input file for future conversion outcomes.
DELAYED_LABEL_INPUT_PATH = DELAYED_LABELS_DIR / "delayed_labels.csv"

# This file stores predictions joined safely to matured outcomes.
MATURED_OUTCOMES_PATH = DELAYED_LABELS_DIR / "matured_prediction_outcomes.csv"

# This JSON file stores performance metrics for matured predictions only.
PRODUCTION_PERFORMANCE_PATH = (
    DELAYED_LABELS_DIR / "production_performance_snapshot.json"
)

# This JSON file records rejected, duplicate, premature, or invalid labels.
DELAYED_LABEL_VALIDATION_PATH = (
    DELAYED_LABELS_DIR / "delayed_label_validation_report.json"
)
