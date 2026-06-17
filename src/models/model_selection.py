# model_selection.py
# Select manual champion, AutoML-style champion, and final production champion.
#
# SIMPLE IDEA:
#   After all models are trained and evaluated, this file decides:
#
#   1. Which model won the manual benchmark?
#   2. Which model won the AutoML-style benchmark?
#   3. Which of those two should become the final production champion?
#
# WHY THIS FILE EXISTS:
#   Model training and model selection are different jobs.
#
#   model_evaluation.py trains and scores models.
#   model_selection.py decides which model should be saved and used by the app.

from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

from src.models.model_config import (
    AUTOML_RESULTS_PATH,
    AUTOML_TRACK_NAME,
    CHAMPION_METRICS_PATH,
    CHAMPION_THRESHOLD_PATH,
    FEATURE_COLUMNS,
    FINAL_CHAMPION_METADATA_PATH,
    FINAL_CHAMPION_MODEL_PATH,
    FINAL_SELECTION_PATH,
    MANUAL_RESULTS_PATH,
    MANUAL_TRACK_NAME,
    TARGET_COLUMN,
)


# --------------------------------------------------
# Timestamp helper
# --------------------------------------------------

def get_current_utc_timestamp():
    """Return current UTC timestamp for model metadata."""

    # WHY UTC:
    #   Monitoring and production systems usually prefer UTC time.
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------
# Select best model from one track
# --------------------------------------------------

def select_best_model_from_track(results_data, track_name):
    """Select best successful model from one benchmark track."""

    # results_data contains many rows:
    #   one row = one trained model result.
    #
    # We first keep only:
    #   1. models from the selected track
    #   2. models that trained successfully

    successful_results = results_data[
        (results_data["track"] == track_name)
        & (results_data["status"] == "success")
    ].copy()

    # If all models failed, we cannot select a champion.
    if successful_results.empty:
        raise ValueError(
            f"No successful models found for track: {track_name}"
        )

    # Sort models by the most important ranking columns.
    #
    # champion_score:
    #   Main combined score.
    #
    # pr_auc:
    #   Important because buyers are rare.
    #
    # best_f1_score:
    #   Best threshold balance between precision and recall.
    #
    # roc_auc:
    #   General ranking quality.
    best_model_row = (
        successful_results.sort_values(
            ["champion_score", "pr_auc", "best_f1_score", "roc_auc"],
            ascending=False,
        )
        .iloc[0]
    )

    return best_model_row


# --------------------------------------------------
# Compare manual winner vs AutoML winner
# --------------------------------------------------

def create_final_selection_table(manual_winner, automl_winner):
    """Compare the manual champion and AutoML-style champion."""

    # manual_winner:
    #   Best model from the manual champion/challenger track.
    #
    # automl_winner:
    #   Best model from the broad AutoML-style benchmark.

    final_candidates = pd.DataFrame(
        [
            manual_winner.to_dict(),
            automl_winner.to_dict(),
        ]
    )

    # This label makes the final comparison easier to read.
    final_candidates["final_candidate_type"] = [
        "Manual champion",
        "AutoML-style champion",
    ]

    # Sort both winners again using the same selection logic.
    # The best of the two becomes the final production champion.
    final_candidates = final_candidates.sort_values(
        ["champion_score", "pr_auc", "best_f1_score", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    # Rank 1 = final winner.
    final_candidates["final_decision_rank"] = final_candidates.index + 1

    # Add a readable decision column.
    final_candidates["final_decision"] = np.where(
        final_candidates["final_decision_rank"] == 1,
        "Selected as production champion",
        "Not selected",
    )

    return final_candidates


# --------------------------------------------------
# Get final model object and threshold table
# --------------------------------------------------

def get_final_model_key(final_winner):
    """Create dictionary key used to find the trained final model."""

    # During training, each model is stored using this key pattern:
    #
    #   track_name__model_name
    #
    # Example:
    #   manual_champion_challenger__Random Forest
    #
    # This function recreates that key for the selected winner.

    final_model_key = (
        f"{final_winner['track']}__{final_winner['model_name']}"
    )

    return final_model_key


def get_final_model_and_threshold_table(
    final_winner,
    trained_models,
    threshold_tables,
):
    """Return final fitted model and its threshold analysis table."""

    final_model_key = get_final_model_key(final_winner)

    # trained_models contains fitted sklearn pipelines.
    final_model = trained_models[final_model_key]

    # threshold_tables contains threshold results for each fitted model.
    final_threshold_table = threshold_tables[final_model_key]

    return final_model, final_threshold_table


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def save_model_selection_outputs(
    manual_results,
    automl_results,
    final_selection,
    final_model,
    final_threshold_table,
):
    """Save model comparison tables, final champion model, and metadata."""

    # Save manual benchmark results.
    manual_results.to_csv(
        MANUAL_RESULTS_PATH,
        index=False,
    )

    # Save AutoML-style benchmark results.
    automl_results.to_csv(
        AUTOML_RESULTS_PATH,
        index=False,
    )

    # Save final comparison table:
    #   manual champion vs AutoML-style champion.
    final_selection.to_csv(
        FINAL_SELECTION_PATH,
        index=False,
    )

    # The first row is the selected production champion.
    final_champion_row = final_selection.iloc[0]

    # Save champion metrics as a one-row table.
    pd.DataFrame(
        [final_champion_row.to_dict()]
    ).to_csv(
        CHAMPION_METRICS_PATH,
        index=False,
    )

    # Save threshold table for final champion model.
    final_threshold_table.to_csv(
        CHAMPION_THRESHOLD_PATH,
        index=False,
    )

    # Save final trained model pipeline.
    #
    # This is the model the Streamlit app will load later.
    joblib.dump(
        final_model,
        FINAL_CHAMPION_MODEL_PATH,
    )

    # Save model metadata as JSON.
    #
    # Metadata explains:
    #   which model was selected
    #   which track it came from
    #   which threshold was best
    #   which features are required
    #   why the model was selected
    metadata = {
        "selected_at_utc": get_current_utc_timestamp(),
        "selection_stage": "initial_champion_from_manual_vs_automl_benchmark",
        "champion_model_name": final_champion_row["model_name"],
        "champion_track": final_champion_row["track"],
        "champion_model_family": final_champion_row["model_family"],
        "champion_model_path": str(FINAL_CHAMPION_MODEL_PATH),
        "best_threshold": float(final_champion_row["best_threshold"]),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "selection_logic": (
            "Initial champion selected by comparing the manual champion "
            "against the AutoML-style champion using champion_score, PR-AUC, "
            "best F1, recall, ROC-AUC, lift vs random targeting, and "
            "deployment suitability."
        ),
        "metrics": {
            "roc_auc": float(final_champion_row["roc_auc"]),
            "pr_auc": float(final_champion_row["pr_auc"]),
            "best_precision": float(final_champion_row["best_precision"]),
            "best_recall": float(final_champion_row["best_recall"]),
            "best_f1_score": float(final_champion_row["best_f1_score"]),
            "best_lift_vs_random": float(
                final_champion_row["best_lift_vs_random"]
            ),
            "champion_score": float(final_champion_row["champion_score"]),
        },
    }

    with open(
        FINAL_CHAMPION_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


# --------------------------------------------------
# Full selection workflow
# --------------------------------------------------

def select_and_save_final_champion(
    manual_results,
    automl_results,
    trained_models,
    threshold_tables,
):
    """Select final champion and save all model-selection outputs."""

    # Step 1:
    # Select best model from manual track.
    manual_winner = select_best_model_from_track(
        results_data=manual_results,
        track_name=MANUAL_TRACK_NAME,
    )

    # Step 2:
    # Select best model from AutoML-style track.
    automl_winner = select_best_model_from_track(
        results_data=automl_results,
        track_name=AUTOML_TRACK_NAME,
    )

    # Step 3:
    # Compare manual winner vs AutoML-style winner.
    final_selection = create_final_selection_table(
        manual_winner=manual_winner,
        automl_winner=automl_winner,
    )

    # Step 4:
    # The first row is the final production champion.
    final_winner = final_selection.iloc[0]

    # Step 5:
    # Get final fitted model and threshold table.
    final_model, final_threshold_table = get_final_model_and_threshold_table(
        final_winner=final_winner,
        trained_models=trained_models,
        threshold_tables=threshold_tables,
    )

    # Step 6:
    # Save comparison tables, champion model, threshold table, and metadata.
    save_model_selection_outputs(
        manual_results=manual_results,
        automl_results=automl_results,
        final_selection=final_selection,
        final_model=final_model,
        final_threshold_table=final_threshold_table,
    )

    return manual_winner, automl_winner, final_winner