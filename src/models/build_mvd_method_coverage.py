# build_mvd_method_coverage.py
# MVD method coverage script for the E-Commerce Conversion Intelligence Platform.
#
# SIMPLE PURPOSE:
#   This script proves that important MVD/course methods were applied or clearly mapped.
#
# WHAT THIS SCRIPT COVERS:
#   Step 1: Load visitor features
#   Step 2: Create a safe modelling sample
#   Step 3: PCA explained variance + 2D map data
#   Step 4: K-Means clustering proof
#   Step 5: DBSCAN clustering proof
#   Step 6: LOF outlier proof
#   Step 7: Compare outlier methods
#   Step 8: Create MVD coverage matrix
#   Step 9: Save outputs
#
# HOW TO RUN:
#   python3 -m src.models.build_mvd_method_coverage
#
# MAIN INPUTS:
#   data/processed/visitor_features.csv
#   data/processed/visitor_anomaly_scores.csv          optional
#   reports/tables/final_true_champion_comparison.csv  optional
#
# MAIN OUTPUTS:
#   reports/tables/mvd_pca_explained_variance.csv
#   reports/tables/mvd_pca_projection_sample.csv
#   reports/tables/mvd_kmeans_summary.csv
#   reports/tables/mvd_dbscan_summary.csv
#   reports/tables/mvd_lof_outlier_summary.csv
#   reports/tables/mvd_outlier_method_comparison.csv
#   reports/tables/mvd_method_coverage_matrix.csv
#   outputs/charts/mvd_pca_scree_plot.png
#   outputs/charts/mvd_pca_2d_map.png
#   outputs/charts/mvd_kmeans_elbow.png
#   outputs/charts/mvd_kmeans_clusters.png
#   outputs/charts/mvd_dbscan_clusters.png

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. BASIC SETTINGS
# --------------------------------------------------

RANDOM_STATE = 42

TARGET_COLUMN = "converted"
ID_COLUMN = "visitorid"

TIME_COLUMNS = [
    "first_event_time",
    "last_event_time",
    "snapshot_time",
    "observation_start",
    "target_end",
    "data_split",
]

# We sample because some methods like DBSCAN and LOF can be slow on 1.4M rows.
PCA_SAMPLE_SIZE = 60_000
KMEANS_SAMPLE_SIZE = 60_000
DBSCAN_SAMPLE_SIZE = 12_000
LOF_SAMPLE_SIZE = 40_000


# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

FEATURES_PATH = Path("data/processed/visitor_training_snapshots.csv")
ANOMALY_SCORES_PATH = Path("data/processed/visitor_anomaly_scores.csv")
FINAL_CHAMPION_COMPARISON_PATH = Path("reports/tables/final_true_champion_comparison.csv")

TABLE_DIR = Path("reports/tables")
CHART_DIR = Path("outputs/charts")

PCA_VARIANCE_PATH = TABLE_DIR / "mvd_pca_explained_variance.csv"
PCA_PROJECTION_PATH = TABLE_DIR / "mvd_pca_projection_sample.csv"

KMEANS_SUMMARY_PATH = TABLE_DIR / "mvd_kmeans_summary.csv"
DBSCAN_SUMMARY_PATH = TABLE_DIR / "mvd_dbscan_summary.csv"
LOF_SUMMARY_PATH = TABLE_DIR / "mvd_lof_outlier_summary.csv"
OUTLIER_COMPARISON_PATH = TABLE_DIR / "mvd_outlier_method_comparison.csv"
COVERAGE_MATRIX_PATH = TABLE_DIR / "mvd_method_coverage_matrix.csv"

PCA_SCREE_CHART_PATH = CHART_DIR / "mvd_pca_scree_plot.png"
PCA_MAP_CHART_PATH = CHART_DIR / "mvd_pca_2d_map.png"
KMEANS_ELBOW_CHART_PATH = CHART_DIR / "mvd_kmeans_elbow.png"
KMEANS_CLUSTER_CHART_PATH = CHART_DIR / "mvd_kmeans_clusters.png"
DBSCAN_CLUSTER_CHART_PATH = CHART_DIR / "mvd_dbscan_clusters.png"


# --------------------------------------------------
# 3. SMALL HELPER FUNCTIONS
# --------------------------------------------------

def create_output_folders() -> None:
    """Create folders before saving tables and charts."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def print_step(title: str) -> None:
    """Print clear terminal progress."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_visitor_features() -> pd.DataFrame:
    """Load visitor-level feature table."""

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {FEATURES_PATH}. "
            "Run the visitor feature pipeline first."
        )

    data = pd.read_csv(FEATURES_PATH)

    print(f"Loaded visitor features: {data.shape[0]:,} rows, {data.shape[1]:,} columns")

    return data


def build_numeric_feature_matrix(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create numeric feature matrix for PCA, clustering, and outlier methods."""

    drop_columns = [
        TARGET_COLUMN,
        ID_COLUMN,
        *TIME_COLUMNS,
    ]

    drop_columns = [
        column for column in drop_columns
        if column in data.columns
    ]

    X = data.drop(columns=drop_columns)
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0)

    y = data[TARGET_COLUMN].astype(int) if TARGET_COLUMN in data.columns else pd.Series(0, index=data.index)
    visitor_ids = data[ID_COLUMN] if ID_COLUMN in data.columns else pd.Series(data.index, index=data.index)

    print(f"Numeric feature matrix: {X.shape[0]:,} rows, {X.shape[1]:,} features")
    print(f"Features used: {X.columns.tolist()}")

    return X, y, visitor_ids


def sample_rows(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
    sample_size: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create a repeatable sample for heavy unsupervised methods."""

    if len(X) <= sample_size:
        return X.copy(), y.copy(), visitor_ids.copy()

    sample_index = X.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).index

    return (
        X.loc[sample_index].copy(),
        y.loc[sample_index].copy(),
        visitor_ids.loc[sample_index].copy(),
    )


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """Standardize numeric features before PCA, clustering, and LOF."""

    scaler = StandardScaler()

    return scaler.fit_transform(X)


def save_chart(path: Path) -> None:
    """Save current matplotlib chart with clean spacing."""

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_simple_scatter_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str,
    path: Path,
) -> None:
    """Create a simple PCA scatter plot.

    We keep this basic because the final dashboard polish happens later.
    """

    plt.figure(figsize=(10, 7))

    groups = data[color_column].astype(str).unique()

    for group in groups:
        group_data = data[data[color_column].astype(str) == group]

        plt.scatter(
            group_data[x_column],
            group_data[y_column],
            s=10,
            alpha=0.45,
            label=str(group),
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=color_column, fontsize=8)
    plt.grid(alpha=0.25)

    save_chart(path)


# --------------------------------------------------
# 4. PCA COVERAGE
# --------------------------------------------------

def run_pca_coverage(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run PCA explained variance and 2D projection.

    Why:
        PCA shows whether many behaviour features can be compressed into fewer components.
        It also gives a 2D map for visualising visitor behaviour patterns.
    """

    print_step("Step 3: PCA explained variance and 2D map")

    X_sample, y_sample, visitor_sample = sample_rows(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
        sample_size=PCA_SAMPLE_SIZE,
    )

    X_scaled = scale_features(X_sample)

    pca = PCA(
        n_components=min(X_scaled.shape[1], 7),
        random_state=RANDOM_STATE,
    )

    pca_values = pca.fit_transform(X_scaled)

    variance_table = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    projection_table = pd.DataFrame(
        {
            ID_COLUMN: visitor_sample.values,
            "PC1": pca_values[:, 0],
            "PC2": pca_values[:, 1],
            TARGET_COLUMN: y_sample.values,
        }
    )

    variance_table.to_csv(PCA_VARIANCE_PATH, index=False)
    projection_table.to_csv(PCA_PROJECTION_PATH, index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(
        variance_table["component"],
        variance_table["cumulative_explained_variance"],
        marker="o",
        linewidth=2,
    )
    plt.title("PCA scree plot: cumulative explained variance", fontsize=14, fontweight="bold")
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    save_chart(PCA_SCREE_CHART_PATH)

    pca_plot_data = projection_table.copy()
    pca_plot_data["converted_label"] = np.where(
        pca_plot_data[TARGET_COLUMN] == 1,
        "Converted",
        "Not converted",
    )

    create_simple_scatter_chart(
        data=pca_plot_data.sample(
            n=min(10_000, len(pca_plot_data)),
            random_state=RANDOM_STATE,
        ),
        x_column="PC1",
        y_column="PC2",
        color_column="converted_label",
        title="PCA 2D map of visitor behaviour",
        path=PCA_MAP_CHART_PATH,
    )

    print(f"Saved PCA variance table: {PCA_VARIANCE_PATH}")
    print(f"Saved PCA projection sample: {PCA_PROJECTION_PATH}")
    print(f"Saved PCA charts: {PCA_SCREE_CHART_PATH}, {PCA_MAP_CHART_PATH}")

    return variance_table, projection_table, X_scaled


# --------------------------------------------------
# 5. K-MEANS CLUSTERING COVERAGE
# --------------------------------------------------

def run_kmeans_coverage(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
) -> pd.DataFrame:
    """Run K-Means clustering for selected k values.

    Why:
        K-Means gives unsupervised visitor groups.
        This supports segmentation beyond supervised purchase prediction.
    """

    print_step("Step 4: K-Means clustering proof")

    X_sample, y_sample, visitor_sample = sample_rows(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
        sample_size=KMEANS_SAMPLE_SIZE,
    )

    X_scaled = scale_features(X_sample)

    rows = []
    labels_for_chart = None

    for k in [3, 5, 7]:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(10_000, len(labels)),
            random_state=RANDOM_STATE,
        )

        davies_bouldin = davies_bouldin_score(
            X_scaled,
            labels,
        )

        rows.append(
            {
                "method": "K-Means",
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "cluster_count": len(np.unique(labels)),
                "sample_rows": len(X_sample),
            }
        )

        if k == 5:
            labels_for_chart = labels

    summary = pd.DataFrame(rows)
    summary.to_csv(KMEANS_SUMMARY_PATH, index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(summary["k"], summary["inertia"], marker="o", linewidth=2)
    plt.title("K-Means elbow check", fontsize=14, fontweight="bold")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.25)
    save_chart(KMEANS_ELBOW_CHART_PATH)

    # Use PCA projection for a readable 2D cluster chart.
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    cluster_plot = pd.DataFrame(
        {
            "PC1": pca_2d[:, 0],
            "PC2": pca_2d[:, 1],
            "cluster": labels_for_chart.astype(str),
        }
    )

    create_simple_scatter_chart(
        data=cluster_plot.sample(
            n=min(10_000, len(cluster_plot)),
            random_state=RANDOM_STATE,
        ),
        x_column="PC1",
        y_column="PC2",
        color_column="cluster",
        title="K-Means visitor clusters shown on PCA map",
        path=KMEANS_CLUSTER_CHART_PATH,
    )

    print(f"Saved K-Means summary: {KMEANS_SUMMARY_PATH}")
    print(f"Saved K-Means charts: {KMEANS_ELBOW_CHART_PATH}, {KMEANS_CLUSTER_CHART_PATH}")

    return summary


# --------------------------------------------------
# 6. DBSCAN CLUSTERING COVERAGE
# --------------------------------------------------

def run_dbscan_coverage(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
) -> pd.DataFrame:
    """Run a small DBSCAN grid.

    Why:
        DBSCAN finds dense behaviour groups and noise points.
        Noise points are visitors that do not fit dense clusters.
    """

    print_step("Step 5: DBSCAN clustering proof")

    X_sample, y_sample, visitor_sample = sample_rows(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
        sample_size=DBSCAN_SAMPLE_SIZE,
    )

    X_scaled = scale_features(X_sample)

    rows = []
    labels_for_chart = None

    grid = [
        {"eps": 0.5, "min_samples": 10},
        {"eps": 1.0, "min_samples": 10},
        {"eps": 1.5, "min_samples": 20},
        {"eps": 2.0, "min_samples": 20},
    ]

    for params in grid:
        model = DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
            n_jobs=-1,
        )

        labels = model.fit_predict(X_scaled)

        unique_labels = set(labels)
        cluster_count = len([label for label in unique_labels if label != -1])
        noise_count = int((labels == -1).sum())
        noise_rate = noise_count / len(labels)

        silhouette = np.nan

        # Silhouette is only valid if there are at least 2 non-noise clusters.
        if cluster_count >= 2:
            valid_mask = labels != -1

            if valid_mask.sum() > 100:
                silhouette = silhouette_score(
                    X_scaled[valid_mask],
                    labels[valid_mask],
                    sample_size=min(5_000, valid_mask.sum()),
                    random_state=RANDOM_STATE,
                )

        rows.append(
            {
                "method": "DBSCAN",
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "cluster_count": cluster_count,
                "noise_count": noise_count,
                "noise_rate": noise_rate,
                "silhouette_score_non_noise": silhouette,
                "sample_rows": len(X_sample),
            }
        )

        if labels_for_chart is None:
            labels_for_chart = labels

    summary = pd.DataFrame(rows)
    summary.to_csv(DBSCAN_SUMMARY_PATH, index=False)

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    dbscan_plot = pd.DataFrame(
        {
            "PC1": pca_2d[:, 0],
            "PC2": pca_2d[:, 1],
            "cluster": labels_for_chart.astype(str),
        }
    )

    create_simple_scatter_chart(
        data=dbscan_plot.sample(
            n=min(8_000, len(dbscan_plot)),
            random_state=RANDOM_STATE,
        ),
        x_column="PC1",
        y_column="PC2",
        color_column="cluster",
        title="DBSCAN clusters and noise shown on PCA map",
        path=DBSCAN_CLUSTER_CHART_PATH,
    )

    print(f"Saved DBSCAN summary: {DBSCAN_SUMMARY_PATH}")
    print(f"Saved DBSCAN chart: {DBSCAN_CLUSTER_CHART_PATH}")

    return summary


# --------------------------------------------------
# 7. LOF OUTLIER COVERAGE
# --------------------------------------------------

def run_lof_coverage(
    X: pd.DataFrame,
    y: pd.Series,
    visitor_ids: pd.Series,
) -> pd.DataFrame:
    """Run Local Outlier Factor on a sample.

    Why:
        LOF checks whether a visitor looks unusual compared with nearby visitors.
        This complements Isolation Forest from the anomaly pipeline.
    """

    print_step("Step 6: LOF outlier proof")

    X_sample, y_sample, visitor_sample = sample_rows(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
        sample_size=LOF_SAMPLE_SIZE,
    )

    X_scaled = scale_features(X_sample)

    model = LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.04,
        n_jobs=-1,
    )

    lof_labels = model.fit_predict(X_scaled)

    lof_flag = lof_labels == -1

    summary = pd.DataFrame(
        [
            {
                "method": "Local Outlier Factor",
                "sample_rows": len(X_sample),
                "outlier_count": int(lof_flag.sum()),
                "outlier_rate": float(lof_flag.mean()),
                "converted_rate_all": float(y_sample.mean()),
                "converted_rate_outliers": float(y_sample[lof_flag].mean()) if lof_flag.sum() > 0 else np.nan,
                "converted_rate_normal": float(y_sample[~lof_flag].mean()) if (~lof_flag).sum() > 0 else np.nan,
            }
        ]
    )

    summary.to_csv(LOF_SUMMARY_PATH, index=False)

    print(f"Saved LOF summary: {LOF_SUMMARY_PATH}")

    return summary


# --------------------------------------------------
# 8. OUTLIER METHOD COMPARISON
# --------------------------------------------------

def compare_outlier_methods(lof_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare LOF result with existing Isolation Forest pipeline if available."""

    print_step("Step 7: Compare outlier methods")

    rows = []

    if not lof_summary.empty:
        rows.append(
            {
                "method": "Local Outlier Factor",
                "source": "This script",
                "rows_checked": int(lof_summary.iloc[0]["sample_rows"]),
                "outlier_rate": float(lof_summary.iloc[0]["outlier_rate"]),
                "business_meaning": "Detects visitors unusual compared with nearby visitors.",
            }
        )

    if ANOMALY_SCORES_PATH.exists():
        anomaly_data = pd.read_csv(
            ANOMALY_SCORES_PATH,
            usecols=lambda column: column in ["final_anomaly_flag"],
        )

        if "final_anomaly_flag" in anomaly_data.columns:
            rows.append(
                {
                    "method": "Rule + Isolation Forest anomaly layer",
                    "source": "src/anomaly/build_anomaly_signals.py",
                    "rows_checked": len(anomaly_data),
                    "outlier_rate": float(anomaly_data["final_anomaly_flag"].mean()),
                    "business_meaning": "Combines explainable rules with Isolation Forest for review layer.",
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTLIER_COMPARISON_PATH, index=False)

    print(f"Saved outlier method comparison: {OUTLIER_COMPARISON_PATH}")

    return comparison


# --------------------------------------------------
# 9. MVD COVERAGE MATRIX
# --------------------------------------------------

def create_coverage_matrix() -> pd.DataFrame:
    """Create agenda-level MVD coverage matrix.

    This table is the proof file.
    It shows what is implemented directly, what is covered by project equivalent,
    and what is documented as not applicable.
    """

    print_step("Step 8: Create MVD coverage matrix")

    rows = [
        {
            "learning_day": "Day 1",
            "topic": "ML intro, supervised/unsupervised/RL, EDA",
            "status": "Covered",
            "project_artifact": "Full project workflow + README/deck later",
            "proof_note": "Supervised conversion model, unsupervised clustering/anomaly methods, and RL future-extension note.",
        },
        {
            "learning_day": "Day 2",
            "topic": "Preprocessing, leakage-safe features, train/test split",
            "status": "Covered",
            "project_artifact": "src/data/build_visitor_features.py, src/models/finalize_true_champion.py",
            "proof_note": "Visitor grain, target creation, numeric feature matrix, stratified train/test split.",
        },
        {
            "learning_day": "Day 3",
            "topic": "Imbalanced data",
            "status": "Covered",
            "project_artifact": "reports/tables/final_true_champion_comparison.csv",
            "proof_note": "Class weights, threshold optimisation, XGBoost scale_pos_weight, optional SMOTE model.",
        },
        {
            "learning_day": "Day 4",
            "topic": "Outlier detection",
            "status": "Covered",
            "project_artifact": "reports/tables/mvd_lof_outlier_summary.csv, reports/tables/mvd_outlier_method_comparison.csv",
            "proof_note": "LOF added here; rule + Isolation Forest anomaly layer already exists.",
        },
        {
            "learning_day": "Day 5",
            "topic": "PCA and dimensionality reduction",
            "status": "Covered",
            "project_artifact": "reports/tables/mvd_pca_explained_variance.csv, outputs/charts/mvd_pca_scree_plot.png",
            "proof_note": "PCA explained variance, scree plot, and 2D behaviour map.",
        },
        {
            "learning_day": "Day 6",
            "topic": "Autoencoders",
            "status": "Concept / optional extension",
            "project_artifact": "Coverage matrix note",
            "proof_note": "Autoencoder is not used as final tabular classifier. It is documented as an optional anomaly extension.",
        },
        {
            "learning_day": "Day 7",
            "topic": "Decision trees, feature importance, explainability",
            "status": "Covered",
            "project_artifact": "final champion Random Forest + final comparison tables",
            "proof_note": "Tree-based final champion; feature importance can be added in final reporting/polish.",
        },
        {
            "learning_day": "Day 8",
            "topic": "Clustering",
            "status": "Covered",
            "project_artifact": "reports/tables/mvd_kmeans_summary.csv, reports/tables/mvd_dbscan_summary.csv",
            "proof_note": "K-Means k=3/5/7 with elbow; DBSCAN grid with noise counts.",
        },
        {
            "learning_day": "Day 9",
            "topic": "Ensembles, stacking, blending",
            "status": "Covered",
            "project_artifact": "reports/tables/final_true_champion_comparison.csv",
            "proof_note": "Random Forest, boosting, and probability-average ensemble check.",
        },
        {
            "learning_day": "Day 10",
            "topic": "Boosting",
            "status": "Covered",
            "project_artifact": "reports/tables/final_true_champion_comparison.csv",
            "proof_note": "XGBoost baseline and tuned XGBoost included in final champion check.",
        },
        {
            "learning_day": "Day 11",
            "topic": "Hyperparameter tuning",
            "status": "Covered",
            "project_artifact": "src/models/finalize_true_champion.py",
            "proof_note": "RandomizedSearchCV tuning for Random Forest and XGBoost.",
        },
        {
            "learning_day": "Day 12",
            "topic": "Regression, logistic regression, GLM",
            "status": "Mostly covered",
            "project_artifact": "baseline Logistic Regression + forecasting pipeline",
            "proof_note": "Logistic regression baseline and KPI forecasting are covered. GLM count models can be documented as future extension if not added.",
        },
        {
            "learning_day": "Day 13",
            "topic": "Reinforcement learning",
            "status": "Not applicable / future extension",
            "project_artifact": "Coverage matrix note",
            "proof_note": "Not suitable for static supervised conversion prediction; future extension could be campaign policy or bandit optimisation.",
        },
        {
            "learning_day": "Day 14",
            "topic": "Transfer learning",
            "status": "Not applicable / documented",
            "project_artifact": "Coverage matrix note",
            "proof_note": "Image transfer learning is not suitable for tabular RetailRocket visitor behaviour data.",
        },
        {
            "learning_day": "Day 15",
            "topic": "AutoML",
            "status": "Covered",
            "project_artifact": "reports/tables/automl_benchmark_results.csv, app/pages/3_Model_Benchmark_Selection.py",
            "proof_note": "AutoML-style benchmark and final model comparison covered. PyCaret avoided for stability unless added later.",
        },
    ]

    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_MATRIX_PATH, index=False)

    print(f"Saved MVD coverage matrix: {COVERAGE_MATRIX_PATH}")

    return coverage


# --------------------------------------------------
# 10. MAIN SCRIPT
# --------------------------------------------------

def main() -> None:
    """Run all MVD method coverage steps."""

    warnings.filterwarnings("ignore")
    create_output_folders()

    print_step("Step 1: Load visitor features")

    data = load_visitor_features()

    print_step("Step 2: Prepare numeric feature matrix")

    X, y, visitor_ids = build_numeric_feature_matrix(data)

    pca_variance, pca_projection, _ = run_pca_coverage(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
    )

    kmeans_summary = run_kmeans_coverage(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
    )

    dbscan_summary = run_dbscan_coverage(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
    )

    lof_summary = run_lof_coverage(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
    )

    outlier_comparison = compare_outlier_methods(
        lof_summary=lof_summary,
    )

    coverage = create_coverage_matrix()

    print_step("Step 9: Save summary and finish")

    print("MVD method coverage completed.")
    print(f"PCA components saved: {len(pca_variance)}")
    print(f"K-Means rows saved: {len(kmeans_summary)}")
    print(f"DBSCAN rows saved: {len(dbscan_summary)}")
    print(f"Outlier comparison rows saved: {len(outlier_comparison)}")
    print(f"Coverage matrix rows saved: {len(coverage)}")


if __name__ == "__main__":
    main()
