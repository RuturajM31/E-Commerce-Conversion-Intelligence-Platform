# finalize_true_champion.py
# Final model-hardening script for the E-Commerce Conversion Intelligence Platform.
#
# SIMPLE PURPOSE:
#   Until now, Random Forest was our "current champion".
#   This script checks whether it is truly the best model after stronger testing.
#
# WHAT THIS SCRIPT DOES:
#   1. Loads visitor-level ecommerce features.
#   2. Creates a clean supervised ML train/test split.
#   3. Evaluates the existing champion model if it already exists.
#   4. Tests imbalance-aware models.
#   5. Tunes Random Forest.
#   6. Tunes XGBoost if XGBoost is installed.
#   7. Optionally tests SMOTE if imbalanced-learn is installed.
#   8. Compares models using business-focused metrics.
#   9. Runs an outlier/anomaly sensitivity check if anomaly outputs exist.
#   10. Runs a small stability check across repeated splits.
#   11. Saves the final true champion model and metadata.
#
# WHY THIS IS IMPORTANT:
#   The first champion was selected from an initial benchmark.
#   A production-ready champion should also be checked for:
#       - imbalanced data
#       - threshold quality
#       - tuning
#       - boosting comparison
#       - outlier sensitivity
#       - stability
#
# HOW TO RUN FROM PROJECT ROOT:
#   python3 -m src.models.finalize_true_champion
#
# MAIN INPUT:
#   data/processed/visitor_features.csv
#
# MAIN OUTPUTS:
#   models/trained/final_champion_model.joblib
#   models/metadata/final_champion_metadata.json
#   reports/tables/final_true_champion_comparison.csv
#   reports/tables/final_true_champion_summary.csv
#   reports/tables/final_true_champion_stability.csv
#   reports/tables/final_true_champion_sensitivity.csv

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. BASIC SETTINGS
# --------------------------------------------------
# These settings keep the script understandable and controllable.

RANDOM_STATE = 42

# The target column says whether a visitor converted or not.
TARGET_COLUMN = "converted"

# Visitor ID is useful for joining with anomaly files, but not as a model feature.
ID_COLUMN = "visitorid"

# These timestamp columns are not used directly as model features.
# We already use activity_span_ms as a usable numeric feature.
TIME_COLUMNS = [
    "first_event_time",
    "last_event_time",
]

# We tune on a sample so the script does not become too slow.
# The final selected model is still trained on the full training data.
TUNING_SAMPLE_SIZE = 150_000

# The stability check also uses a sample so repeated training stays fast.
STABILITY_SAMPLE_SIZE = 200_000

# Threshold grid for finding the best classification threshold.
# Example:
#   0.95 means "predict buyer only if score >= 0.95".
THRESHOLDS = np.array(
    [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 0.97,
        0.98, 0.99,
    ]
)

# Business-focused model selection weights.
# This matches our project logic:
#   PR-AUC matters most because conversion is rare.
#   F1 balances precision and recall.
#   Precision matters because campaigns cost money.
#   Recall matters because we do not want to miss too many buyers.
#   ROC-AUC is useful but less important for rare conversion.
CHAMPION_SCORE_WEIGHTS = {
    "pr_auc": 0.45,
    "best_f1_score": 0.30,
    "best_precision": 0.15,
    "best_recall": 0.05,
    "roc_auc": 0.05,
}


# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------
# Keep all paths relative to the project root.
# This avoids personal local paths inside the project.

FEATURES_PATH = Path("data/processed/visitor_features.csv")
ANOMALY_SCORES_PATH = Path("data/processed/visitor_anomaly_scores.csv")

CURRENT_CHAMPION_MODEL_PATH = Path("models/trained/champion_model.joblib")
CURRENT_CHAMPION_METADATA_PATH = Path("models/metadata/champion_model_metadata.json")

FINAL_MODEL_PATH = Path("models/trained/final_champion_model.joblib")
FINAL_METADATA_PATH = Path("models/metadata/final_champion_metadata.json")

FINAL_COMPARISON_PATH = Path("reports/tables/final_true_champion_comparison.csv")
FINAL_SUMMARY_PATH = Path("reports/tables/final_true_champion_summary.csv")
FINAL_STABILITY_PATH = Path("reports/tables/final_true_champion_stability.csv")
FINAL_SENSITIVITY_PATH = Path("reports/tables/final_true_champion_sensitivity.csv")
FINAL_THRESHOLDS_PATH = Path("reports/tables/final_true_champion_thresholds.csv")
FINAL_CHAMPION_SCORES_PATH = Path("data/processed/final_champion_visitor_scores.csv")


# --------------------------------------------------
# 3. OPTIONAL IMPORTS
# --------------------------------------------------
# Some advanced libraries may or may not be installed.
# We keep them optional so the script stays stable.

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    IMBLEARN_AVAILABLE = True
except Exception:
    SMOTE = None
    ImbalancedPipeline = None
    IMBLEARN_AVAILABLE = False


# --------------------------------------------------
# 4. SMALL UTILITY FUNCTIONS
# --------------------------------------------------

def create_output_folders() -> None:
    """Create output folders if they do not exist.

    Why:
        Saving files fails if the folders are missing.
    """

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_CHAMPION_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    """Print a readable section title in the terminal."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate XGBoost imbalance weight.

    Simple meaning:
        If buyers are rare, the model should pay more attention to buyers.

    Formula:
        negative examples / positive examples

    Example:
        1000 non-buyers and 10 buyers
        scale_pos_weight = 100
    """

    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def calculate_business_score(row: Dict[str, float]) -> float:
    """Calculate one final score used to rank model candidates."""

    score = 0.0

    for metric_name, metric_weight in CHAMPION_SCORE_WEIGHTS.items():
        score += float(row.get(metric_name, 0.0)) * metric_weight

    return score


# --------------------------------------------------
# 5. DATA LOADING AND PREPARATION
# --------------------------------------------------

def load_visitor_features() -> pd.DataFrame:
    """Load the visitor-level feature table.

    Input file:
        data/processed/visitor_features.csv

    Expected grain:
        one row = one visitor

    Required target:
        converted
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURES_PATH}. "
            "Run the visitor feature pipeline first."
        )

    data = pd.read_csv(FEATURES_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    if ID_COLUMN not in data.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' was not found.")

    print(f"Loaded visitor features: {data.shape[0]:,} rows and {data.shape[1]:,} columns")

    return data


def build_model_matrix(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Create X, y, visitor_id, and feature column list.

    X:
        model input features

    y:
        target column, converted or not converted

    visitor_ids:
        used later for anomaly sensitivity check

    feature_columns:
        saved in metadata so the app knows which columns the model expects
    """

    columns_to_drop = [
        TARGET_COLUMN,
        ID_COLUMN,
        *TIME_COLUMNS,
    ]

    # Keep only columns that really exist in the dataframe.
    columns_to_drop = [
        column for column in columns_to_drop
        if column in data.columns
    ]

    X = data.drop(columns=columns_to_drop)

    # Keep numeric features only.
    # Current visitor feature table is numeric after feature engineering.
    X = X.select_dtypes(include=[np.number]).copy()

    # Fill rare missing values with 0.
    # This keeps the model training stable.
    X = X.fillna(0)

    y = data[TARGET_COLUMN].astype(int)

    visitor_ids = data[ID_COLUMN]

    feature_columns = X.columns.tolist()

    print(f"Model feature matrix shape: {X.shape}")
    print(f"Target positive rate: {y.mean():.4%}")
    print(f"Feature columns used: {feature_columns}")

    return X, y, visitor_ids, feature_columns


def create_split(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create one consistent train/test split.

    Stratify means:
        Keep the buyer/non-buyer ratio similar in train and test.

    Why:
        Conversion is rare, so random splitting without stratify can be unstable.
    """

    return train_test_split(
        X,
        y,
        visitor_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a smaller training sample for tuning.

    Why:
        Tuning on 1M+ rows can be slow.
        We tune on a representative sample, then train final models on full train data.
    """

    if len(X_train) <= sample_size:
        return X_train.copy(), y_train.copy()

    sample_index = y_train.sample(
        n=sample_size,
        random_state=random_state,
        replace=False,
    ).index

    return X_train.loc[sample_index].copy(), y_train.loc[sample_index].copy()


# --------------------------------------------------
# 6. EVALUATION FUNCTIONS
# --------------------------------------------------

def evaluate_thresholds(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate one model across many probability thresholds.

    Why:
        For imbalanced conversion prediction, default threshold 0.50 is not always best.

    Output:
        table with precision, recall, and F1 for each threshold.
    """

    rows = []

    for threshold in THRESHOLDS:
        y_pred = (y_score >= threshold).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        predicted_positive_rate = y_pred.mean()

        rows.append(
            {
                "model_name": model_name,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predicted_positive_rate": predicted_positive_rate,
            }
        )

    return pd.DataFrame(rows)


def summarize_model_performance(
    model_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    training_rows: int,
    model_family: str,
    notes: str,
    deployable: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Create one model summary row and one threshold table.

    Important metrics:
        PR-AUC:
            Best for rare buyer prediction.

        ROC-AUC:
            General ranking quality.

        best_precision / best_recall / best_f1_score:
            Taken from the best threshold row.
    """

    threshold_table = evaluate_thresholds(
        y_true=y_true,
        y_score=y_score,
        model_name=model_name,
    )

    best_threshold_row = threshold_table.sort_values(
        ["f1_score", "precision", "recall"],
        ascending=False,
    ).iloc[0]

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    summary = {
        "model_name": model_name,
        "model_family": model_family,
        "training_rows": training_rows,
        "deployable": deployable,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_threshold": float(best_threshold_row["threshold"]),
        "best_precision": float(best_threshold_row["precision"]),
        "best_recall": float(best_threshold_row["recall"]),
        "best_f1_score": float(best_threshold_row["f1_score"]),
        "predicted_positive_rate_at_best_threshold": float(best_threshold_row["predicted_positive_rate"]),
        "notes": notes,
    }

    summary["business_score"] = calculate_business_score(summary)

    return summary, threshold_table


def predict_scores(model, X_test: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""

    probabilities = model.predict_proba(X_test)

    return probabilities[:, 1]


# --------------------------------------------------
# 7. MODEL TRAINING FUNCTIONS
# --------------------------------------------------

def evaluate_existing_champion(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Optional[object], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """Load and evaluate the already-existing champion model.

    Why:
        This lets us compare the old champion against newly tuned models.
    """

    if not CURRENT_CHAMPION_MODEL_PATH.exists():
        print("Existing champion model not found. Skipping old champion evaluation.")
        return None, None, None

    print("Loading existing champion model...")

    model = joblib.load(CURRENT_CHAMPION_MODEL_PATH)

    y_score = predict_scores(model, X_test)

    summary, threshold_table = summarize_model_performance(
        model_name="Existing Champion Model",
        y_true=y_test,
        y_score=y_score,
        training_rows=0,
        model_family="Existing artifact",
        notes="Previously selected champion model loaded from models/trained/champion_model.joblib.",
        deployable=True,
    )

    return model, summary, threshold_table


def train_logistic_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train a simple imbalance-aware Logistic Regression model.

    Why:
        This is a strong baseline for imbalanced classification.

    class_weight='balanced':
        Gives more importance to rare buyer examples.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a simple class-weighted Random Forest.

    Why:
        Random Forest handles nonlinear behaviour well.
        class_weight='balanced_subsample' helps with rare buyers.
    """

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Tune Random Forest on a representative sample.

    Steps:
        1. Sample training data.
        2. Run RandomizedSearchCV.
        3. Train final best Random Forest on full training data.

    Why not tune on full data:
        Full tuning can be very slow with 1M+ rows.
    """

    X_sample, y_sample = sample_training_data(
        X_train=X_train,
        y_train=y_train,
        sample_size=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    parameter_grid = {
        "n_estimators": [250, 350, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [5, 10, 20, 50],
        "max_features": ["sqrt", 0.7],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_grid,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_sample, y_sample)

    print(f"Best Random Forest parameters: {search.best_params_}")

    final_model = RandomForestClassifier(
        **search.best_params_,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(X_train, y_train)

    return final_model


def train_xgboost_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Optional[object]:
    """Train an imbalance-aware XGBoost baseline.

    XGBoost is a boosting model.
    Boosting often performs well on tabular business data.
    """

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost baseline.")
        return None

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Optional[object]:
    """Tune XGBoost on a representative sample.

    We use average_precision scoring because PR-AUC is important for rare conversion.
    """

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost tuning.")
        return None

    X_sample, y_sample = sample_training_data(
        X_train=X_train,
        y_train=y_train,
        sample_size=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    scale_pos_weight = calculate_scale_pos_weight(y_sample)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    parameter_grid = {
        "n_estimators": [250, 350, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_grid,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_sample, y_sample)

    print(f"Best XGBoost parameters: {search.best_params_}")

    full_scale_pos_weight = calculate_scale_pos_weight(y_train)

    final_model = XGBClassifier(
        **search.best_params_,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=full_scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(X_train, y_train)

    return final_model


def train_smote_logistic_sample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Optional[object]:
    """Optionally train SMOTE + Logistic Regression on a sample.

    Why optional:
        SMOTE requires imbalanced-learn.
        It can also be expensive on very large data.

    Why included:
        This gives practical coverage for imbalance handling.
    """

    if not IMBLEARN_AVAILABLE:
        print("imbalanced-learn is not installed. Skipping SMOTE model.")
        return None

    X_sample, y_sample = sample_training_data(
        X_train=X_train,
        y_train=y_train,
        sample_size=80_000,
        random_state=RANDOM_STATE,
    )

    model = ImbalancedPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_sample, y_sample)

    return model


# --------------------------------------------------
# 8. OUTLIER / ANOMALY SENSITIVITY CHECK
# --------------------------------------------------

def load_test_anomaly_flags(test_visitor_ids: pd.Series) -> pd.Series:
    """Load anomaly flags for test visitors if anomaly file exists.

    Output:
        Boolean Series aligned with test_visitor_ids.

    Meaning:
        True  = visitor is anomalous
        False = visitor is normal / not flagged
    """

    if not ANOMALY_SCORES_PATH.exists():
        print("Anomaly score file not found. Skipping anomaly sensitivity check.")
        return pd.Series(False, index=test_visitor_ids.index)

    anomaly_columns = [
        ID_COLUMN,
        "final_anomaly_flag",
    ]

    anomaly_data = pd.read_csv(
        ANOMALY_SCORES_PATH,
        usecols=[
            column for column in anomaly_columns
            if column in pd.read_csv(ANOMALY_SCORES_PATH, nrows=0).columns
        ],
    )

    if "final_anomaly_flag" not in anomaly_data.columns:
        print("final_anomaly_flag column not found. Skipping anomaly sensitivity check.")
        return pd.Series(False, index=test_visitor_ids.index)

    anomaly_map = anomaly_data.set_index(ID_COLUMN)["final_anomaly_flag"]

    flags = test_visitor_ids.map(anomaly_map).fillna(False).astype(bool)

    return flags


def run_anomaly_sensitivity_check(
    model_scores: Dict[str, np.ndarray],
    y_test: pd.Series,
    test_visitor_ids: pd.Series,
) -> pd.DataFrame:
    """Compare model performance on all visitors vs non-anomalous visitors.

    Why:
        We want to know if a model only performs well because of extreme visitors.
    """

    anomaly_flags = load_test_anomaly_flags(test_visitor_ids)

    normal_mask = ~anomaly_flags

    rows = []

    if normal_mask.sum() == len(normal_mask):
        return pd.DataFrame(
            [
                {
                    "model_name": "not_available",
                    "evaluation_group": "anomaly_file_missing_or_no_flags",
                    "test_rows": len(y_test),
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "note": "Anomaly file was missing or no anomaly flags were available.",
                }
            ]
        )

    for model_name, y_score in model_scores.items():
        all_pr_auc = average_precision_score(y_test, y_score)
        normal_pr_auc = average_precision_score(y_test[normal_mask], y_score[normal_mask])

        all_roc_auc = roc_auc_score(y_test, y_score)
        normal_roc_auc = roc_auc_score(y_test[normal_mask], y_score[normal_mask])

        rows.append(
            {
                "model_name": model_name,
                "evaluation_group": "all_test_visitors",
                "test_rows": len(y_test),
                "pr_auc": all_pr_auc,
                "roc_auc": all_roc_auc,
                "note": "Performance on full test set.",
            }
        )

        rows.append(
            {
                "model_name": model_name,
                "evaluation_group": "non_anomalous_test_visitors",
                "test_rows": int(normal_mask.sum()),
                "pr_auc": normal_pr_auc,
                "roc_auc": normal_roc_auc,
                "note": "Performance after removing anomaly-flagged visitors.",
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# 9. STABILITY CHECK
# --------------------------------------------------

def run_stability_check(
    candidate_models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Run a small repeated split stability check.

    Why:
        We do not want a model that wins only because of one lucky split.

    Note:
        This is intentionally small and simple.
        It is a project-hardening check, not a heavy research experiment.
    """

    if not candidate_models:
        return pd.DataFrame()

    X_sample, y_sample = sample_training_data(
        X_train=X,
        y_train=y,
        sample_size=STABILITY_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []

    seeds = [11, 22, 33]

    for seed in seeds:
        X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
            X_sample,
            y_sample,
            test_size=0.25,
            random_state=seed,
            stratify=y_sample,
        )

        for model_name, model in candidate_models.items():
            print(f"Stability check: seed={seed}, model={model_name}")

            model_copy = clone(model)

            model_copy.fit(X_train_small, y_train_small)

            y_score = predict_scores(model_copy, X_test_small)

            pr_auc = average_precision_score(y_test_small, y_score)
            roc_auc = roc_auc_score(y_test_small, y_score)

            rows.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "test_rows": len(y_test_small),
                    "positive_rate": y_test_small.mean(),
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                }
            )

    stability = pd.DataFrame(rows)

    return stability


# --------------------------------------------------
# 10. FINAL MODEL SELECTION
# --------------------------------------------------

def choose_final_champion(comparison: pd.DataFrame) -> pd.Series:
    """Choose the best deployable model by business score.

    deployable=True means:
        The model can be saved and used by the Streamlit app.
    """

    deployable_models = comparison[comparison["deployable"] == True].copy()

    if deployable_models.empty:
        raise ValueError("No deployable models available for final champion selection.")

    deployable_models = deployable_models.sort_values(
        "business_score",
        ascending=False,
    )

    return deployable_models.iloc[0]


def save_final_outputs(
    final_model,
    final_summary: pd.Series,
    feature_columns: List[str],
    comparison: pd.DataFrame,
    threshold_tables: List[pd.DataFrame],
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    """Save final model, metadata, and result tables."""

    joblib.dump(final_model, FINAL_MODEL_PATH)

    comparison.to_csv(FINAL_COMPARISON_PATH, index=False)

    if threshold_tables:
        all_thresholds = pd.concat(threshold_tables, ignore_index=True)
        all_thresholds.to_csv(FINAL_THRESHOLDS_PATH, index=False)

    if not stability.empty:
        stability.to_csv(FINAL_STABILITY_PATH, index=False)

    if not sensitivity.empty:
        sensitivity.to_csv(FINAL_SENSITIVITY_PATH, index=False)

    metadata = {
        "final_model_name": final_summary["model_name"],
        "model_family": final_summary["model_family"],
        "selection_metric": "business_score",
        "business_score_weights": CHAMPION_SCORE_WEIGHTS,
        "best_threshold": float(final_summary["best_threshold"]),
        "pr_auc": float(final_summary["pr_auc"]),
        "roc_auc": float(final_summary["roc_auc"]),
        "best_precision": float(final_summary["best_precision"]),
        "best_recall": float(final_summary["best_recall"]),
        "best_f1_score": float(final_summary["best_f1_score"]),
        "feature_columns": feature_columns,
        "notes": str(final_summary["notes"]),
    }

    with open(FINAL_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    summary_df = pd.DataFrame([metadata])
    summary_df.to_csv(FINAL_SUMMARY_PATH, index=False)

    print("\nSaved final champion model and outputs:")
    print(f"- {FINAL_MODEL_PATH}")
    print(f"- {FINAL_METADATA_PATH}")
    print(f"- {FINAL_COMPARISON_PATH}")
    print(f"- {FINAL_SUMMARY_PATH}")


# --------------------------------------------------
# 11. MAIN SCRIPT
# --------------------------------------------------

def main() -> None:
    """Run the full true champion finalisation workflow."""

    warnings.filterwarnings("ignore")

    create_output_folders()

    print_section("Step 1: Load data")

    data = load_visitor_features()

    X, y, visitor_ids, feature_columns = build_model_matrix(data)

    print_section("Step 2: Create train/test split")

    X_train, X_test, y_train, y_test, train_visitor_ids, test_visitor_ids = create_split(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
    )

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Train conversion rate: {y_train.mean():.4%}")
    print(f"Test conversion rate: {y_test.mean():.4%}")

    model_objects: Dict[str, object] = {}
    model_scores: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, float]] = []
    threshold_tables: List[pd.DataFrame] = []

    print_section("Step 3: Evaluate existing champion")

    existing_model, existing_summary, existing_thresholds = evaluate_existing_champion(
        X_test=X_test,
        y_test=y_test,
    )

    if existing_model is not None:
        model_objects["Existing Champion Model"] = existing_model
        model_scores["Existing Champion Model"] = predict_scores(existing_model, X_test)
        summary_rows.append(existing_summary)
        threshold_tables.append(existing_thresholds)

    print_section("Step 4: Train Logistic Regression with class weights")

    logistic_model = train_logistic_class_weight(
        X_train=X_train,
        y_train=y_train,
    )

    logistic_scores = predict_scores(logistic_model, X_test)

    logistic_summary, logistic_thresholds = summarize_model_performance(
        model_name="Logistic Regression Class Weight",
        y_true=y_test,
        y_score=logistic_scores,
        training_rows=len(X_train),
        model_family="Linear baseline",
        notes="Imbalance-aware Logistic Regression using class_weight='balanced'.",
        deployable=True,
    )

    model_objects["Logistic Regression Class Weight"] = logistic_model
    model_scores["Logistic Regression Class Weight"] = logistic_scores
    summary_rows.append(logistic_summary)
    threshold_tables.append(logistic_thresholds)

    print_section("Step 5: Train class-weighted Random Forest")

    rf_class_weight_model = train_random_forest_class_weight(
        X_train=X_train,
        y_train=y_train,
    )

    rf_class_weight_scores = predict_scores(rf_class_weight_model, X_test)

    rf_class_weight_summary, rf_class_weight_thresholds = summarize_model_performance(
        model_name="Random Forest Class Weight",
        y_true=y_test,
        y_score=rf_class_weight_scores,
        training_rows=len(X_train),
        model_family="Random Forest",
        notes="Random Forest with class_weight='balanced_subsample'.",
        deployable=True,
    )

    model_objects["Random Forest Class Weight"] = rf_class_weight_model
    model_scores["Random Forest Class Weight"] = rf_class_weight_scores
    summary_rows.append(rf_class_weight_summary)
    threshold_tables.append(rf_class_weight_thresholds)

    print_section("Step 6: Tune Random Forest")

    tuned_rf_model = tune_random_forest(
        X_train=X_train,
        y_train=y_train,
    )

    tuned_rf_scores = predict_scores(tuned_rf_model, X_test)

    tuned_rf_summary, tuned_rf_thresholds = summarize_model_performance(
        model_name="Tuned Random Forest",
        y_true=y_test,
        y_score=tuned_rf_scores,
        training_rows=len(X_train),
        model_family="Random Forest",
        notes="RandomizedSearchCV tuned Random Forest, retrained on full training data.",
        deployable=True,
    )

    model_objects["Tuned Random Forest"] = tuned_rf_model
    model_scores["Tuned Random Forest"] = tuned_rf_scores
    summary_rows.append(tuned_rf_summary)
    threshold_tables.append(tuned_rf_thresholds)

    print_section("Step 7: Train XGBoost baseline")

    xgb_baseline_model = train_xgboost_baseline(
        X_train=X_train,
        y_train=y_train,
    )

    if xgb_baseline_model is not None:
        xgb_baseline_scores = predict_scores(xgb_baseline_model, X_test)

        xgb_baseline_summary, xgb_baseline_thresholds = summarize_model_performance(
            model_name="XGBoost Baseline",
            y_true=y_test,
            y_score=xgb_baseline_scores,
            training_rows=len(X_train),
            model_family="Boosting",
            notes="XGBoost baseline with scale_pos_weight for imbalance.",
            deployable=True,
        )

        model_objects["XGBoost Baseline"] = xgb_baseline_model
        model_scores["XGBoost Baseline"] = xgb_baseline_scores
        summary_rows.append(xgb_baseline_summary)
        threshold_tables.append(xgb_baseline_thresholds)

    print_section("Step 8: Tune XGBoost")

    tuned_xgb_model = tune_xgboost(
        X_train=X_train,
        y_train=y_train,
    )

    if tuned_xgb_model is not None:
        tuned_xgb_scores = predict_scores(tuned_xgb_model, X_test)

        tuned_xgb_summary, tuned_xgb_thresholds = summarize_model_performance(
            model_name="Tuned XGBoost",
            y_true=y_test,
            y_score=tuned_xgb_scores,
            training_rows=len(X_train),
            model_family="Boosting",
            notes="RandomizedSearchCV tuned XGBoost, retrained on full training data.",
            deployable=True,
        )

        model_objects["Tuned XGBoost"] = tuned_xgb_model
        model_scores["Tuned XGBoost"] = tuned_xgb_scores
        summary_rows.append(tuned_xgb_summary)
        threshold_tables.append(tuned_xgb_thresholds)

    print_section("Step 9: Optional SMOTE model")

    smote_model = train_smote_logistic_sample(
        X_train=X_train,
        y_train=y_train,
    )

    if smote_model is not None:
        smote_scores = predict_scores(smote_model, X_test)

        smote_summary, smote_thresholds = summarize_model_performance(
            model_name="SMOTE Logistic Regression Sample",
            y_true=y_test,
            y_score=smote_scores,
            training_rows=min(len(X_train), 80_000),
            model_family="Imbalance handling",
            notes="SMOTE + Logistic Regression trained on a sample for imbalance coverage.",
            deployable=True,
        )

        model_objects["SMOTE Logistic Regression Sample"] = smote_model
        model_scores["SMOTE Logistic Regression Sample"] = smote_scores
        summary_rows.append(smote_summary)
        threshold_tables.append(smote_thresholds)

    print_section("Step 10: Simple probability-average ensemble check")

    ensemble_candidates = [
        model_name for model_name in ["Tuned Random Forest", "Tuned XGBoost"]
        if model_name in model_scores
    ]

    if len(ensemble_candidates) == 2:
        ensemble_scores = (
            model_scores["Tuned Random Forest"] +
            model_scores["Tuned XGBoost"]
        ) / 2

        ensemble_summary, ensemble_thresholds = summarize_model_performance(
            model_name="Probability Average Ensemble",
            y_true=y_test,
            y_score=ensemble_scores,
            training_rows=len(X_train),
            model_family="Ensemble check",
            notes="Average of Tuned Random Forest and Tuned XGBoost probabilities. Used as comparison only.",
            deployable=False,
        )

        model_scores["Probability Average Ensemble"] = ensemble_scores
        summary_rows.append(ensemble_summary)
        threshold_tables.append(ensemble_thresholds)

    print_section("Step 11: Compare all candidate models")

    comparison = pd.DataFrame(summary_rows)

    comparison = comparison.sort_values(
        "business_score",
        ascending=False,
    ).reset_index(drop=True)

    print(
        comparison[
            [
                "model_name",
                "model_family",
                "deployable",
                "pr_auc",
                "roc_auc",
                "best_threshold",
                "best_precision",
                "best_recall",
                "best_f1_score",
                "business_score",
            ]
        ]
    )

    print_section("Step 12: Run anomaly / outlier sensitivity check")

    sensitivity = run_anomaly_sensitivity_check(
        model_scores=model_scores,
        y_test=y_test,
        test_visitor_ids=test_visitor_ids,
    )

    print(sensitivity.head(20))

    print_section("Step 13: Run stability check on top deployable models")

    top_deployable_names = (
        comparison[comparison["deployable"] == True]
        .head(3)["model_name"]
        .tolist()
    )

    stability_models = {
        model_name: model_objects[model_name]
        for model_name in top_deployable_names
        if model_name in model_objects
    }

    stability = run_stability_check(
        candidate_models=stability_models,
        X=X,
        y=y,
    )

    if not stability.empty:
        stability_summary = (
            stability.groupby("model_name")
            .agg(
                mean_pr_auc=("pr_auc", "mean"),
                std_pr_auc=("pr_auc", "std"),
                mean_roc_auc=("roc_auc", "mean"),
                std_roc_auc=("roc_auc", "std"),
            )
            .reset_index()
        )

        print(stability_summary)

    print_section("Step 14: Select and save final true champion")

    final_summary = choose_final_champion(comparison)

    final_model_name = final_summary["model_name"]
    final_model = model_objects[final_model_name]

    print(f"Final true champion selected: {final_model_name}")
    print(f"Business score: {final_summary['business_score']:.4f}")
    print(f"Best threshold: {final_summary['best_threshold']:.2f}")
    print(f"PR-AUC: {final_summary['pr_auc']:.4f}")
    print(f"Best precision: {final_summary['best_precision']:.4f}")
    print(f"Best recall: {final_summary['best_recall']:.4f}")
    print(f"Best F1: {final_summary['best_f1_score']:.4f}")

    save_final_outputs(
        final_model=final_model,
        final_summary=final_summary,
        feature_columns=feature_columns,
        comparison=comparison,
        threshold_tables=threshold_tables,
        stability=stability,
        sensitivity=sensitivity,
    )

    save_final_champion_scores(
        final_model=final_model,
        X=X,
        visitor_ids=visitor_ids,
        threshold=float(final_summary["best_threshold"]),
    )

    print_section("Done")

    print("Final true champion workflow completed successfully.")


if __name__ == "__main__":
    main()
