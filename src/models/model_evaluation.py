# model_evaluation.py
# Shared model training and evaluation utilities.
#
# SIMPLE IDEA:
#   This file is the "model testing engine".
#
# WHAT THIS FILE DOES:
#   1. Takes one model recipe from the registry.
#   2. Creates a safe sample of the data if needed.
#   3. Splits the data into train and test sets.
#   4. Trains the model.
#   5. Predicts purchase intent probabilities.
#   6. Calculates model metrics.
#   7. Tests multiple business thresholds.
#   8. Creates one result row for comparison tables.
#
# WHY THIS FILE EXISTS:
#   Manual models and AutoML-style models should use the same evaluation logic.
#   That keeps the comparison fair.

import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.models.model_config import (
    CHAMPION_SCORE_WEIGHTS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    THRESHOLDS_TO_TEST,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


# --------------------------------------------------
# Safe sampling
# --------------------------------------------------

def create_stratified_sample(data, max_rows):
    """Create a smaller dataset sample while keeping buyer/non-buyer ratio similar."""

    # WHY WE SAMPLE:
    #   RetailRocket has many visitor rows.
    #   Some models can become slow.
    #
    # WHY STRATIFIED:
    #   Buyers are rare.
    #   We must keep the same buyer/non-buyer ratio in the sample.

    if len(data) <= max_rows:
        return data.copy()

    sample_fraction = max_rows / len(data)
    sampled_parts = []

    for _, group in data.groupby(TARGET_COLUMN):
        sampled_parts.append(
            group.sample(
                frac=sample_fraction,
                random_state=RANDOM_STATE,
            )
        )

    return pd.concat(sampled_parts, ignore_index=True)


# --------------------------------------------------
# Probability helper
# --------------------------------------------------

def get_prediction_probability(model, X_test):
    """Return probability-like score for class 1 = converted."""

    # Most classifiers provide predict_proba().
    # This gives direct probability values.
    #
    # Example:
    #   0.93 means 93% purchase intent score.

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    # Some classifiers do not give probabilities.
    # They give decision scores instead.
    # We convert those scores into a 0-1 range.

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_test)

        min_score = decision_scores.min()
        max_score = decision_scores.max()

        # Avoid division by zero if all scores are identical.
        if max_score == min_score:
            return np.zeros_like(decision_scores)

        probability_like_scores = (
            (decision_scores - min_score)
            / (max_score - min_score)
        )

        return probability_like_scores

    # If the model gives neither probability nor decision score,
    # it cannot be used for our threshold/lift analysis.
    raise ValueError(
        "Model does not support predict_proba() or decision_function()."
    )


# --------------------------------------------------
# Threshold evaluation
# --------------------------------------------------

def evaluate_thresholds(y_true, y_probability):
    """Test different decision thresholds and calculate business metrics."""

    # SIMPLE MEANING:
    #   The model gives a probability score.
    #   Threshold decides who we target.
    #
    # Example:
    #   threshold = 0.90
    #   score >= 0.90 -> target visitor
    #   score < 0.90  -> do not target visitor

    threshold_rows = []

    # Natural conversion rate in test data.
    # This is our random targeting baseline.
    #
    # Example:
    #   If buyer rate is 0.83%, random targeting gets around 0.83 buyers per 100 visitors.
    base_conversion_rate = y_true.mean()

    for threshold in THRESHOLDS_TO_TEST:
        # Convert probabilities into 0/1 predictions using current threshold.
        y_pred = (y_probability >= threshold).astype(int)

        # Count how many visitors would be targeted.
        targeted_visitors = int(y_pred.sum())

        # Calculate share of visitors selected for campaign targeting.
        targeted_share = targeted_visitors / len(y_pred)

        # Precision:
        #   Out of visitors we targeted, how many were real buyers?
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # Recall:
        #   Out of all real buyers, how many did we catch?
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # F1:
        #   Balance between precision and recall.
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # Lift:
        #   How much better model targeting is than random targeting.
        #
        # Example:
        #   precision = 25%
        #   random conversion = 0.83%
        #   lift = 25 / 0.83 = about 30x
        if base_conversion_rate > 0:
            lift_vs_random = precision / base_conversion_rate
        else:
            lift_vs_random = 0

        threshold_rows.append(
            {
                "threshold": threshold,
                "targeted_visitors": targeted_visitors,
                "targeted_share": targeted_share,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "lift_vs_random": lift_vs_random,
            }
        )

    threshold_data = pd.DataFrame(threshold_rows)

    # Pick best threshold.
    #
    # First priority: F1
    # Second priority: recall
    # Third priority: precision
    #
    # This gives a balanced campaign decision.
    best_threshold_row = (
        threshold_data.sort_values(
            ["f1_score", "precision", "recall"],
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )

    return threshold_data, best_threshold_row


# --------------------------------------------------
# Champion score
# --------------------------------------------------

def calculate_champion_score(pr_auc, roc_auc, best_threshold_row):
    """Calculate one business-focused score used to rank models."""

    # WHY NOT ACCURACY:
    
    #   Buyers are rare.
    #   A model can look accurate by predicting almost everyone as non-buyer.
    #
    # WHY THIS SCORE:
    #   This project is about ecommerce campaign targeting.
    #   We want a model that finds likely buyers and avoids wasting targets.
    #
    # PR-AUC:
    #   Rewards good ranking of rare buyers.
    #
    # Best F1:
    #   Rewards balance between precision and recall.
    #
    # Best precision:
    #   Rewards efficient targeting.
    #
    # Best recall:
    #   Still useful, but lower weight so it does not dominate.
    #
    # ROC-AUC:
    #   Useful general ranking metric, but less important than PR-AUC here.

    champion_score = (
        CHAMPION_SCORE_WEIGHTS["pr_auc"] * pr_auc
        + CHAMPION_SCORE_WEIGHTS["best_f1_score"] * best_threshold_row["f1_score"]
        + CHAMPION_SCORE_WEIGHTS["best_precision"] * best_threshold_row["precision"]
        + CHAMPION_SCORE_WEIGHTS["best_recall"] * best_threshold_row["recall"]
        + CHAMPION_SCORE_WEIGHTS["roc_auc"] * roc_auc
    )

    return champion_score

# --------------------------------------------------
# Train and evaluate one model
# --------------------------------------------------

def train_and_evaluate_model(
    model_config,
    full_training_data,
):
    """Train on historical train rows and evaluate on validation rows."""

    model_name = model_config["model_name"]
    track = model_config["track"]
    model_family = model_config["model_family"]
    reason = model_config["reason"]
    uses_sample_weight = model_config["uses_sample_weight"]
    max_rows = model_config["max_rows"]

    print(f"\nTraining model: {track} | {model_name}")

    if SPLIT_COLUMN not in full_training_data.columns:
        raise ValueError(
            f"Training data is missing '{SPLIT_COLUMN}'."
        )

    train_data = full_training_data.loc[
        full_training_data[SPLIT_COLUMN] == TRAIN_SPLIT
    ].copy()

    validation_data = full_training_data.loc[
        full_training_data[SPLIT_COLUMN] == VALIDATION_SPLIT
    ].copy()

    if train_data.empty:
        raise ValueError("Training split contains no rows.")

    if validation_data.empty:
        raise ValueError("Validation split contains no rows.")

    model_data = create_stratified_sample(
        data=train_data,
        max_rows=max_rows,
    )

    X_train = model_data[FEATURE_COLUMNS]
    y_train = model_data[TARGET_COLUMN].astype(int)

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training split must contain both target classes."
        )

    if y_validation.nunique() < 2:
        raise ValueError(
            "Validation split must contain both target classes."
        )

    pipeline = clone(model_config["pipeline"])

    start_time = time.time()

    if uses_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weight,
        )
    else:
        pipeline.fit(
            X_train,
            y_train,
        )

    training_seconds = time.time() - start_time

    y_probability = get_prediction_probability(
        model=pipeline,
        X_test=X_validation,
    )

    y_pred_default = (
        y_probability >= 0.50
    ).astype(int)

    threshold_data, best_threshold_row = evaluate_thresholds(
        y_true=y_validation,
        y_probability=y_probability,
    )

    threshold_data.insert(
        0,
        "evaluation_split",
        VALIDATION_SPLIT,
    )

    threshold_data.insert(
        0,
        "model_name",
        model_name,
    )

    roc_auc = roc_auc_score(
        y_validation,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_validation,
        y_probability,
    )

    accuracy_default = accuracy_score(
        y_validation,
        y_pred_default,
    )

    precision_default = precision_score(
        y_validation,
        y_pred_default,
        zero_division=0,
    )

    recall_default = recall_score(
        y_validation,
        y_pred_default,
        zero_division=0,
    )

    f1_default = f1_score(
        y_validation,
        y_pred_default,
        zero_division=0,
    )

    champion_score = calculate_champion_score(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        best_threshold_row=best_threshold_row,
    )

    result_row = {
        "evaluation_split": VALIDATION_SPLIT,
        "track": track,
        "model_name": model_name,
        "model_family": model_family,
        "reason": reason,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "buyer_rate_validation": float(
            y_validation.mean()
        ),
        "training_seconds": round(
            training_seconds,
            2,
        ),
        "accuracy_at_0_50": accuracy_default,
        "precision_at_0_50": precision_default,
        "recall_at_0_50": recall_default,
        "f1_at_0_50": f1_default,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_threshold_row[
            "threshold"
        ],
        "best_precision": best_threshold_row[
            "precision"
        ],
        "best_recall": best_threshold_row[
            "recall"
        ],
        "best_f1_score": best_threshold_row[
            "f1_score"
        ],
        "best_targeted_share": best_threshold_row[
            "targeted_share"
        ],
        "best_lift_vs_random": best_threshold_row[
            "lift_vs_random"
        ],
        "champion_score": champion_score,
        "uses_sample_weight": bool(
            uses_sample_weight
        ),
        "configured_max_training_rows": int(
            max_rows
        ),
        "status": "success",
        "error_message": "",
    }

    print(
        f"Finished {model_name} | "
        f"Validation PR-AUC={pr_auc:.4f} | "
        f"ROC-AUC={roc_auc:.4f} | "
        f"Best F1={best_threshold_row['f1_score']:.4f}"
    )

    return pipeline, result_row, threshold_data


# --------------------------------------------------
# Failed model result helper
# --------------------------------------------------

def create_failed_result_row(
    model_config,
    error,
):
    """Create a validation-result row when model training fails."""

    return {
        "evaluation_split": VALIDATION_SPLIT,
        "track": model_config["track"],
        "model_name": model_config["model_name"],
        "model_family": model_config["model_family"],
        "reason": model_config["reason"],
        "training_rows": np.nan,
        "validation_rows": np.nan,
        "buyer_rate_validation": np.nan,
        "training_seconds": np.nan,
        "accuracy_at_0_50": np.nan,
        "precision_at_0_50": np.nan,
        "recall_at_0_50": np.nan,
        "f1_at_0_50": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "best_threshold": np.nan,
        "best_precision": np.nan,
        "best_recall": np.nan,
        "best_f1_score": np.nan,
        "best_targeted_share": np.nan,
        "best_lift_vs_random": np.nan,
        "champion_score": np.nan,
        "uses_sample_weight": bool(
            model_config["uses_sample_weight"]
        ),
        "configured_max_training_rows": int(
            model_config["max_rows"]
        ),
        "status": "failed",
        "error_message": str(error),
    }


# --------------------------------------------------
# Run a full benchmark track
# --------------------------------------------------

def run_benchmark_track(model_configs, full_training_data):
    """Train and evaluate all models in one benchmark track."""

    # result_rows:
    #   List of model result dictionaries.
    #
    # trained_models:
    #   Stores fitted model objects so we can save the winner later.
    #
    # threshold_tables:
    #   Stores threshold analysis for each model.

    result_rows = []
    trained_models = {}
    threshold_tables = {}

    for model_config in model_configs:
        model_key = f"{model_config['track']}__{model_config['model_name']}"

        try:
            trained_model, result_row, threshold_data = train_and_evaluate_model(
                model_config=model_config,
                full_training_data=full_training_data,
            )

            trained_models[model_key] = trained_model
            threshold_tables[model_key] = threshold_data
            result_rows.append(result_row)

        except Exception as error:
            print(f"\nModel failed: {model_config['model_name']}")
            print(f"Reason: {error}")

            failed_row = create_failed_result_row(
                model_config=model_config,
                error=error,
            )

            result_rows.append(failed_row)

    results_data = pd.DataFrame(result_rows)

    return results_data, trained_models, threshold_tables