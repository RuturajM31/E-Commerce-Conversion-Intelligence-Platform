# model_config.py
# Shared settings for the model-selection workflow.

from pathlib import Path

import numpy as np

from src.data.feature_engineering import MODEL_FEATURE_COLUMNS

# 1. Input data
DATA_PATH = Path("data/processed/visitor_training_snapshots.csv")

# 2. Output folders
MODEL_DIR = Path("models/trained")
METADATA_DIR = Path("models/metadata")
REPORT_TABLES_DIR = Path("reports/tables")

# 3. Initial champion outputs from run_model_selection.py
# These are kept for backward compatibility with the first benchmark workflow.
FINAL_CHAMPION_MODEL_PATH = MODEL_DIR / "champion_model.joblib"
FINAL_CHAMPION_METADATA_PATH = METADATA_DIR / "champion_model_metadata.json"

# Clear aliases so the meaning is obvious in new code.
INITIAL_CHAMPION_MODEL_PATH = FINAL_CHAMPION_MODEL_PATH
INITIAL_CHAMPION_METADATA_PATH = FINAL_CHAMPION_METADATA_PATH

MANUAL_RESULTS_PATH = REPORT_TABLES_DIR / "manual_model_comparison.csv"
AUTOML_RESULTS_PATH = REPORT_TABLES_DIR / "automl_benchmark_results.csv"
FINAL_SELECTION_PATH = REPORT_TABLES_DIR / "final_model_selection.csv"
CHAMPION_METRICS_PATH = REPORT_TABLES_DIR / "champion_model_metrics.csv"
CHAMPION_THRESHOLD_PATH = REPORT_TABLES_DIR / "champion_threshold_analysis.csv"

# 4. Final true champion outputs from finalize_true_champion.py
TRUE_FINAL_MODEL_PATH = MODEL_DIR / "final_champion_model.joblib"
TRUE_FINAL_METADATA_PATH = METADATA_DIR / "final_champion_metadata.json"
TRUE_FINAL_COMPARISON_PATH = REPORT_TABLES_DIR / "final_true_champion_comparison.csv"
TRUE_FINAL_SUMMARY_PATH = REPORT_TABLES_DIR / "final_true_champion_summary.csv"
TRUE_FINAL_THRESHOLDS_PATH = REPORT_TABLES_DIR / "final_true_champion_thresholds.csv"
TRUE_FINAL_STABILITY_PATH = REPORT_TABLES_DIR / "final_true_champion_stability.csv"
TRUE_FINAL_SENSITIVITY_PATH = REPORT_TABLES_DIR / "final_true_champion_sensitivity.csv"

# 5. Features and target
# Use the canonical feature schema shared by dataset generation,
# model training, Streamlit prediction, and batch scoring.
#
# A separate list copy preserves the existing model-config interface while
# preventing accidental changes to the canonical source definition.
FEATURE_COLUMNS = list(MODEL_FEATURE_COLUMNS)
TARGET_COLUMN = "converted"
SPLIT_COLUMN = "data_split"

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
FINAL_HOLDOUT_SPLIT = "final_holdout"

# 6. Reproducibility and benchmark size limits
RANDOM_STATE = 42
MAX_BENCHMARK_ROWS = 180_000
HEAVY_MODEL_MAX_ROWS = 60_000

# 7. Business thresholds
# Include strict thresholds used by the final champion workflow.
THRESHOLDS_TO_TEST = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 0.97,
    0.98, 0.99,
])

# 8. Champion scoring weights
CHAMPION_SCORE_WEIGHTS = {
    "pr_auc": 0.45,
    "best_f1_score": 0.30,
    "best_precision": 0.15,
    "best_recall": 0.05,
    "roc_auc": 0.05,
}

# 9. Benchmark track names
MANUAL_TRACK_NAME = "manual_champion_challenger"
AUTOML_TRACK_NAME = "automl_style_benchmark"


def create_model_output_folders() -> None:
    """Create folders required by the model-selection workflow."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
