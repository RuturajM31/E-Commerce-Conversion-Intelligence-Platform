# automl_model_registry.py
# AutoML-style model definitions for the ecommerce conversion project.
#
# SIMPLE IDEA:
#   This file creates a list of models that we want to test automatically.
#
# IMPORTANT:

#   This file does NOT train the models.
#   This file only says:
#       "These are the models we want to compare."
#
# WHY THIS IS CALLED AUTO ML STYLE:
#   Real AutoML tools automatically test many model families.
#   Here, we are building a lightweight version using sklearn.
#
# WHY WE DO THIS:
#   We do not want to depend on heavy AutoML packages like PyCaret or AutoGluon
#   inside the production app.
#
# BUSINESS STORY:
#   The system tries many model families and later selects the best champion model
#   using business metrics like PR-AUC, recall, precision, F1, and lift.

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.model_config import (
    AUTOML_TRACK_NAME,
    HEAVY_MODEL_MAX_ROWS,
    MAX_BENCHMARK_ROWS,
    RANDOM_STATE,
)


# --------------------------------------------------
# AutoML-style model registry
# --------------------------------------------------

def get_automl_model_configs():
    """Return a broad list of models for the AutoML-style benchmark."""

    # This list contains many model "recipes".
    #
    # Each recipe is a dictionary.
    #
    # The later benchmark runner will do this:
    #   1. Take one dictionary from this list.
    #   2. Train the model inside that dictionary.
    #   3. Calculate metrics.
    #   4. Save the result.
    #   5. Move to the next model.

    automl_models = [

        # --------------------------------------------------
        # 1. Logistic Regression
        # --------------------------------------------------
        # Simple linear model.
        #
        # WHY INCLUDED:
        #   It is the explainable baseline.
        #   It is fast, stable, and easy to explain in interviews.
        #
        # WHY SCALER:
        #   Logistic Regression is sensitive to feature scale.
        #   Example:
        #       activity_span_ms can be a very large number.
        #       cart_to_view_ratio is a small decimal.
        #   StandardScaler puts them on a comparable scale.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Logistic Regression",
            "model_family": "Linear",
            "pipeline": Pipeline(
                steps=[
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
            "reason": "Linear baseline included in AutoML-style comparison.",
        },

        # --------------------------------------------------
        # 2. SGD Logistic Classifier
        # --------------------------------------------------
        # Fast linear model trained with stochastic gradient descent.
        #
        # SIMPLE MEANING:
        #   Similar idea to Logistic Regression, but trained in a way that can be faster
        #   for large datasets.
        #
        # WHY INCLUDED:
        #   Good benchmark for large data.
        #
        # WHY loss='log_loss':
        #   This makes SGDClassifier output logistic-style probabilities.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "SGD Logistic Classifier",
            "model_family": "Linear",
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SGDClassifier(
                            loss="log_loss",
                            class_weight="balanced",
                            max_iter=1000,
                            tol=1e-3,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Fast linear model for large data.",
        },

        # --------------------------------------------------
        # 3. LDA
        # --------------------------------------------------
        # Linear Discriminant Analysis.
        #
        # SIMPLE MEANING:
        #   LDA tries to separate buyers and non-buyers using a linear boundary.
        #
        # WHY INCLUDED:
        #   Classical ML model.
        #   Useful to show that the AutoML benchmark tests more than trees and boosting.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "LDA",
            "model_family": "Discriminant analysis",
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LinearDiscriminantAnalysis(),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Classical discriminant model.",
        },

        # --------------------------------------------------
        # 4. QDA
        # --------------------------------------------------
        # Quadratic Discriminant Analysis.
        #
        # SIMPLE MEANING:
        #   QDA is like LDA, but it allows a more curved decision boundary.
        #
        # WHY SMALLER SAMPLE:
        #   QDA can be less stable and heavier.
        #   We use HEAVY_MODEL_MAX_ROWS to keep runtime safe.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "QDA",
            "model_family": "Discriminant analysis",
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        QuadraticDiscriminantAnalysis(
                            reg_param=0.05,
                        ),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": HEAVY_MODEL_MAX_ROWS,
            "reason": "Flexible discriminant model tested on smaller safe sample.",
        },

        # --------------------------------------------------
        # 5. Gaussian Naive Bayes
        # --------------------------------------------------
        # Fast probabilistic model.
        #
        # SIMPLE MEANING:
        #   It estimates the probability that a visitor belongs to buyer/non-buyer class
        #   using probability assumptions.
        #
        # WHY INCLUDED:
        #   Very fast baseline.
        #   Useful for AutoML comparison.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Gaussian Naive Bayes",
            "model_family": "Probabilistic",
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianNB(),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Fast probabilistic baseline.",
        },

        # --------------------------------------------------
        # 6. KNN
        # --------------------------------------------------
        # K-Nearest Neighbours.
        #
        # SIMPLE MEANING:
        #   KNN checks similar visitors and predicts based on nearby examples.
        #
        # WHY SCALER:
        #   KNN uses distance.
        #   If one feature has huge numbers, it can dominate the distance.
        #
        # WHY SMALLER SAMPLE:
        #   KNN can be slow with many rows.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "KNN",
            "model_family": "Distance-based",
            "pipeline": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KNeighborsClassifier(
                            n_neighbors=25,
                            weights="distance",
                        ),
                    ),
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": HEAVY_MODEL_MAX_ROWS,
            "reason": "Distance-based benchmark using smaller safe sample.",
        },

        # --------------------------------------------------
        # 7. Decision Tree
        # --------------------------------------------------
        # Single tree model.
        #
        # SIMPLE MEANING:
        #   A tree creates if/else rules.
        #   Example:
        #       if addtocart_count is high -> more likely buyer
        #
        # WHY INCLUDED:
        #   Easy to explain.
        #   Useful baseline before Random Forest / Extra Trees.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Decision Tree",
            "model_family": "Tree",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        DecisionTreeClassifier(
                            max_depth=10,
                            min_samples_leaf=30,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Simple interpretable tree benchmark.",
        },

        # --------------------------------------------------
        # 8. Random Forest
        # --------------------------------------------------
        # Many decision trees combined together.
        #
        # SIMPLE MEANING:
        #   One tree may overfit.
        #   Many trees together are usually more stable.
        #
        # WHY INCLUDED:
        #   Strong classical model for tabular data.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Random Forest",
            "model_family": "Bagging ensemble",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=120,
                            max_depth=12,
                            min_samples_leaf=25,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": False,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Bagging ensemble benchmark.",
        },

        # --------------------------------------------------
        # 9. Extra Trees
        # --------------------------------------------------
        # Similar to Random Forest, but more random.
        #
        # SIMPLE MEANING:
        #   Extra Trees adds more randomness in how trees split.
        #   This can sometimes improve generalisation.
        #
        # WHY INCLUDED:
        #   Strong tree ensemble challenger.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Extra Trees",
            "model_family": "Bagging ensemble",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        ExtraTreesClassifier(
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
            "reason": "Randomized tree ensemble benchmark.",
        },

        # --------------------------------------------------
        # 10. AdaBoost
        # --------------------------------------------------
        # Boosting model.
        #
        # SIMPLE MEANING:
        #   AdaBoost trains weak models one after another.
        #   Each new model focuses more on mistakes from earlier models.
        #
        # WHY uses_sample_weight=True:
        #   AdaBoost can receive sample weights during training.
        #   We use this later to give rare buyers more importance.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "AdaBoost",
            "model_family": "Boosting",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        AdaBoostClassifier(
                            n_estimators=80,
                            learning_rate=0.08,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": True,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Classical boosting benchmark.",
        },

        # --------------------------------------------------
        # 11. Gradient Boosting
        # --------------------------------------------------
        # Strong boosting model for tabular data.
        #
        # SIMPLE MEANING:
        #   It builds small trees sequentially.
        #   Each new tree tries to correct previous mistakes.
        #
        # WHY INCLUDED:
        #   Good challenger for structured ecommerce behaviour features.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "Gradient Boosting",
            "model_family": "Boosting",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=80,
                            learning_rate=0.06,
                            max_depth=3,
                            subsample=0.75,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": True,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Gradient boosting benchmark.",
        },

        # --------------------------------------------------
        # 12. HistGradientBoosting
        # --------------------------------------------------
        # Fast sklearn-native boosting model.
        #
        # SIMPLE MEANING:
        #   Similar boosting idea, but optimized for faster training.
        #
        # WHY INCLUDED:
        #   Strong production-friendly sklearn model.

        {
            "track": AUTOML_TRACK_NAME,
            "model_name": "HistGradientBoosting",
            "model_family": "Boosting",
            "pipeline": Pipeline(
                steps=[
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=150,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            l2_regularization=0.10,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "uses_sample_weight": True,
            "max_rows": MAX_BENCHMARK_ROWS,
            "reason": "Fast sklearn-native boosting benchmark.",
        },
    ]

    return automl_models