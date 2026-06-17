# run_model_selection.py
# Main runner for the full model selection workflow.
#
# SIMPLE IDEA:
#   This is the file you run from the terminal.
#
# WHAT THIS FILE DOES:
#   1. Loads visitor-level training data.
#   2. Runs manual champion/challenger models.
#   3. Runs AutoML-style benchmark models.
#   4. Compares both winners.
#   5. Saves the initial champion model used before final hardening.
#   6. Saves model comparison tables and metadata.
#
# IMPORTANT:
#   This file does not define models.
#   It only connects the other model-selection files together.
#
# RUN COMMAND:
#   python3 -m src.models.run_model_selection

import warnings

import pandas as pd

from src.models.automl_model_registry import get_automl_model_configs
from src.models.manual_model_registry import get_manual_model_configs
from src.models.model_config import (
    AUTOML_RESULTS_PATH,
    CHAMPION_METRICS_PATH,
    CHAMPION_THRESHOLD_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    FINAL_CHAMPION_METADATA_PATH,
    FINAL_CHAMPION_MODEL_PATH,
    FINAL_SELECTION_PATH,
    MANUAL_RESULTS_PATH,
    TARGET_COLUMN,
    create_model_output_folders,
)
from src.models.model_evaluation import run_benchmark_track
from src.models.model_selection import select_and_save_final_champion


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_training_data():
    """Load and validate the visitor-level training dataset."""

    print("\nLoading visitor-level training data...")

    # Read the feature table created earlier by build_visitor_features.py.
    data = pd.read_csv(DATA_PATH)

    # The model needs these columns.
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    # Check if any required columns are missing.
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in training data: {missing_columns}"
        )

    # Keep only the columns needed for model training.
    # This prevents accidental leakage from extra columns.
    training_data = data[required_columns].copy()

    # Remove rows with missing values in model columns.
    # Our current feature engineering should already avoid this,
    # but this is an extra safety check.
    before_drop = len(training_data)

    training_data = training_data.dropna(
        subset=required_columns,
    )

    after_drop = len(training_data)

    dropped_rows = before_drop - after_drop

    if dropped_rows > 0:
        print(f"Dropped rows with missing values: {dropped_rows}")

    # Make sure the target is integer 0/1.
    training_data[TARGET_COLUMN] = training_data[TARGET_COLUMN].astype(int)

    print(f"Training rows available: {len(training_data):,}")
    print(f"Buyer rate: {training_data[TARGET_COLUMN].mean():.4%}")

    return training_data


# --------------------------------------------------
# Print summary helper
# --------------------------------------------------

def print_final_summary(manual_winner, automl_winner, final_winner):
    """Print a simple final summary in the terminal."""

    print("\n" + "=" * 80)
    print("MODEL SELECTION COMPLETE")
    print("=" * 80)

    print("\nManual champion:")
    print(f"  Model: {manual_winner['model_name']}")
    print(f"  PR-AUC: {manual_winner['pr_auc']:.4f}")
    print(f"  ROC-AUC: {manual_winner['roc_auc']:.4f}")
    print(f"  Best threshold: {manual_winner['best_threshold']:.2f}")
    print(f"  Best F1: {manual_winner['best_f1_score']:.4f}")
    print(f"  Lift vs random: {manual_winner['best_lift_vs_random']:.2f}x")

    print("\nAutoML-style champion:")
    print(f"  Model: {automl_winner['model_name']}")
    print(f"  PR-AUC: {automl_winner['pr_auc']:.4f}")
    print(f"  ROC-AUC: {automl_winner['roc_auc']:.4f}")
    print(f"  Best threshold: {automl_winner['best_threshold']:.2f}")
    print(f"  Best F1: {automl_winner['best_f1_score']:.4f}")
    print(f"  Lift vs random: {automl_winner['best_lift_vs_random']:.2f}x")

    print("\nFinal production champion:")
    print(f"  Model: {final_winner['model_name']}")
    print(f"  Track: {final_winner['track']}")
    print(f"  PR-AUC: {final_winner['pr_auc']:.4f}")
    print(f"  ROC-AUC: {final_winner['roc_auc']:.4f}")
    print(f"  Best threshold: {final_winner['best_threshold']:.2f}")
    print(f"  Best precision: {final_winner['best_precision']:.4f}")
    print(f"  Best recall: {final_winner['best_recall']:.4f}")
    print(f"  Best F1: {final_winner['best_f1_score']:.4f}")
    print(f"  Lift vs random: {final_winner['best_lift_vs_random']:.2f}x")

    print("\nSaved files:")
    print(f"  Manual results: {MANUAL_RESULTS_PATH}")
    print(f"  AutoML results: {AUTOML_RESULTS_PATH}")
    print(f"  Final selection: {FINAL_SELECTION_PATH}")
    print(f"  Champion metrics: {CHAMPION_METRICS_PATH}")
    print(f"  Champion threshold table: {CHAMPION_THRESHOLD_PATH}")
    print(f"  Champion model: {FINAL_CHAMPION_MODEL_PATH}")
    print(f"  Champion metadata: {FINAL_CHAMPION_METADATA_PATH}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main():
    """Run the complete model selection workflow."""

    # Some sklearn models may show small numerical warnings.
    # We hide non-critical warnings to keep terminal output readable.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\nStarting model selection workflow...")

    # Create folders for model and report outputs.
    create_model_output_folders()

    # Load visitor-level training data.
    training_data = load_training_data()

    # --------------------------------------------------
    # Manual benchmark
    # --------------------------------------------------
    print("\n" + "-" * 80)
    print("Running manual champion/challenger benchmark")
    print("-" * 80)

    manual_model_configs = get_manual_model_configs()

    manual_results, manual_trained_models, manual_threshold_tables = (
        run_benchmark_track(
            model_configs=manual_model_configs,
            full_training_data=training_data,
        )
    )

    # --------------------------------------------------
    # AutoML-style benchmark
    # --------------------------------------------------
    print("\n" + "-" * 80)
    print("Running AutoML-style benchmark")
    print("-" * 80)

    automl_model_configs = get_automl_model_configs()

    automl_results, automl_trained_models, automl_threshold_tables = (
        run_benchmark_track(
            model_configs=automl_model_configs,
            full_training_data=training_data,
        )
    )

    # --------------------------------------------------
    # Combine trained models and threshold tables
    # --------------------------------------------------
    # Manual and AutoML models are trained separately.
    # For final selection, we combine both dictionaries.

    all_trained_models = {
        **manual_trained_models,
        **automl_trained_models,
    }

    all_threshold_tables = {
        **manual_threshold_tables,
        **automl_threshold_tables,
    }

    # --------------------------------------------------
    # Select and save final champion
    # --------------------------------------------------
    manual_winner, automl_winner, final_winner = (
        select_and_save_final_champion(
            manual_results=manual_results,
            automl_results=automl_results,
            trained_models=all_trained_models,
            threshold_tables=all_threshold_tables,
        )
    )

    # Print readable terminal summary.
    print_final_summary(
        manual_winner=manual_winner,
        automl_winner=automl_winner,
        final_winner=final_winner,
    )


# --------------------------------------------------
# Script entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()