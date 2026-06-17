# manual_model_registry.py
# Manual Champion/Challenger model definitions.
#
# SIMPLE IDEA:
#   This file contains the models we intentionally choose and compare.
#
# WHY THIS FILE EXISTS:
#   Manual model selection shows that we understand different model families.
#   We are not blindly relying on AutoML.
#
# MANUAL TRACK STORY:
#   Logistic Regression = explainable baseline
#   Random Forest = non-linear tree ensemble
#   Extra Trees = stronger randomized tree ensemble
#   Gradient Boosting = boosting challenger
#   HistGradientBoosting = fast sklearn-native boosting model

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.model_config import (
    MANUAL_TRACK_NAME,
    MAX_BENCHMARK_ROWS,
    RANDOM_STATE,
)


# --------------------------------------------------
# Manual model registry
# --------------------------------------------------

def get_manual_model_configs():
    """Return manually selected champion/challenger models."""

    # Each dictionary describes one model experiment.
    #
    # track:
    #   Identifies this model as part of the manual benchmark.
    #
    # model_name:
    #   Human-readable model name for reports and tables.
    #
    # model_family:
    #   Broader model category for explanation.
    #
    # pipeline:
    #   The actual sklearn pipeline that will be trained.
    #
    # uses_sample_weight:
    #   Some models accept sample weights during training.
    #   This helps with imbalanced data when class_weight is not available.
    #
    # max_rows:
    #   Number of rows allowed for this model benchmark.
    #
    # reason:
    #   Short business/interview explanation for why we included this model.

    manual_models = [
        {
            "track": MANUAL_TRACK_NAME,
            "model_name": "Logistic Regression",
            "model_family": "Linear baseline",
            "pipeline": Pipeline(
                steps=[
                    # Logistic Regression is sensitive to feature scale.
                    # StandardScaler puts features on a comparable scale.
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Explainable production-safe baseline.",
        },
        {
            "track": MANUAL_TRACK_NAME,
            "model_name": "Random Forest",
            "model_family": "Bagging ensemble",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(
                            # Number of trees in the forest.
                            n_estimators=120,

                            # Limit tree depth to reduce overfitting and runtime.
                            max_depth=12,

                            # Require enough samples per leaf for more stable rules.
                            min_samples_leaf=25,

                            # Balance rare buyers inside bootstrap samples.
                            class_weight="balanced_subsample",

                            # Repeatable results.
                            random_state=RANDOM_STATE,

                            # Use available CPU cores.
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Non-linear challenger with feature importance.",
        },
        {
            "track": MANUAL_TRACK_NAME,
            "model_name": "Extra Trees",
            "model_family": "Bagging ensemble",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        ExtraTreesClassifier(
                            # Extra Trees uses more random split logic than Random Forest.
                            n_estimators=150,
                            max_depth=14,
                            min_samples_leaf=25,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Stronger randomized tree ensemble challenger.",
        },
        {
            "track": MANUAL_TRACK_NAME,
            "model_name": "Gradient Boosting",
            "model_family": "Boosting",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        GradientBoostingClassifier(
                            # Controlled number of boosting rounds for safe runtime.
                            n_estimators=80,

                            # Smaller learning rate makes boosting more stable.
                            learning_rate=0.06,

                            # Shallow trees reduce overfitting.
                            max_depth=3,

                            # Subsample adds randomness and can improve generalisation.
                            subsample=0.75,

                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            # GradientBoostingClassifier does not use class_weight directly.
            # We will pass sample_weight during fit().
            "uses_sample_weight": True,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Boosting challenger for structured tabular data.",
        },
        {
            "track": MANUAL_TRACK_NAME,
            "model_name": "HistGradientBoosting",
            "model_family": "Boosting",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            # Number of boosting iterations.
                            max_iter=150,

                            # Learning speed.
                            learning_rate=0.08,

                            # Limit tree complexity.
                            max_leaf_nodes=31,

                            # Regularisation for stability.
                            l2_regularization=0.10,

                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            # HistGradientBoosting supports sample weights.
            "uses_sample_weight": True,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Fast sklearn-native boosting model.",
        },
    ]

    return manual_models