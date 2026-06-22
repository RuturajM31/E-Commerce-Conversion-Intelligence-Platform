"""Build coherent, precomputed evidence for ML Validation & Evidence.

Purpose
-------
The Streamlit page must stay responsive and truthful. This script calculates
PCA, K-Means, DBSCAN, and Local Outlier Factor once, then saves compact tables
that every visual can read without refitting models inside Streamlit.

Input
-----
``data/processed/visitor_features.csv`` at one row per visitor.

Outputs
-------
The script writes projection, PCA, clustering-grid, business-profile, density,
and anomaly-investigation tables under ``reports/tables/`` plus one manifest
under ``reports/metadata/``.

Important evidence rule
-----------------------
Every Package 1 method, grid, selected parameter, chart conclusion, and table
uses the same deterministic visitor sample. Historical course artifacts remain
available separately, but they are not mixed into the same-sample visual story.
These outputs are offline analytical evidence, not live production metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FEATURES_PATH = Path("data/processed/visitor_features.csv")
LOF_SUMMARY_PATH = Path("reports/tables/mvd_lof_outlier_summary.csv")

TABLE_DIR = Path("reports/tables")
METADATA_DIR = Path("reports/metadata")

PROJECTION_PATH = TABLE_DIR / "ml_validation_projection_sample.csv"
PCA_VARIANCE_PATH = TABLE_DIR / "ml_validation_pca_variance.csv"
PCA_LOADINGS_PATH = TABLE_DIR / "ml_validation_pca_loadings.csv"
CLUSTER_PROFILE_PATH = TABLE_DIR / "ml_validation_cluster_profile.csv"
CLUSTER_BUSINESS_PATH = TABLE_DIR / "ml_validation_cluster_business_summary.csv"
KMEANS_GRID_PATH = TABLE_DIR / "ml_validation_kmeans_grid.csv"
DBSCAN_GRID_PATH = TABLE_DIR / "ml_validation_dbscan_grid.csv"
DBSCAN_K_DISTANCE_PATH = TABLE_DIR / "ml_validation_dbscan_k_distance.csv"
LOF_FEATURE_PROFILE_PATH = TABLE_DIR / "ml_validation_lof_feature_profile.csv"
LOF_INVESTIGATION_PATH = TABLE_DIR / "ml_validation_lof_investigation.csv"
METHOD_SUMMARY_PATH = TABLE_DIR / "ml_validation_method_summary.csv"
MANIFEST_PATH = METADATA_DIR / "ml_validation_manifest.json"

RANDOM_STATE = 42
DEFAULT_SAMPLE_ROWS = 8_000
SILHOUETTE_SAMPLE_ROWS = 3_000
KMEANS_GRID = tuple(range(1, 11))
KMEANS_DECISION_CANDIDATES = (3, 5, 7)
DBSCAN_EPS_GRID = (0.5, 1.0, 2.0)
DBSCAN_MIN_SAMPLES_GRID = (3, 5, 10)
K_DISTANCE_EXPORT_ROWS = 1_200
INVESTIGATION_ROWS = 150

RAW_FEATURES = [
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
]

OPTIONAL_CONTEXT_COLUMNS = [
    "visitorid",
    "purchase_intent_score",
    "purchased",
    "final_anomaly_flag",
    "visitor_segment",
]


# ---------------------------------------------------------------------------
# Small, reusable helpers
# ---------------------------------------------------------------------------


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Read one CSV when it exists; otherwise return an empty table."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def file_sha256(path: Path) -> str:
    """Return a stable SHA-256 hash for source-governance evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_slug(value: str) -> str:
    """Create a stable column suffix for exported standardized features."""

    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "feature"


def coerce_binary(series: pd.Series) -> pd.Series:
    """Convert common boolean, text, and numeric purchase flags to 0/1."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).gt(0).astype(float)

    mapping = {
        "true": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "purchased": 1.0,
        "1": 1.0,
        "false": 0.0,
        "no": 0.0,
        "n": 0.0,
        "not purchased": 0.0,
        "0": 0.0,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0.0)
        .astype(float)
    )


def safe_mode(series: pd.Series) -> str:
    """Return one stable dominant category for cluster context."""

    values = series.dropna().astype(str)
    if values.empty:
        return "Unavailable"
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[0])


def deterministic_metric_sample_size(rows: int) -> int:
    """Limit pairwise silhouette work while keeping results reproducible."""

    return min(max(rows, 0), SILHOUETTE_SAMPLE_ROWS)


def safe_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_noise: bool = True,
) -> float | None:
    """Calculate deterministic silhouette only when the label set is valid."""

    working_features = features
    working_labels = labels

    if ignore_noise and -1 in set(working_labels.tolist()):
        mask = working_labels != -1
        working_features = working_features[mask]
        working_labels = working_labels[mask]

    unique_labels = set(working_labels.tolist())
    if len(unique_labels) < 2 or len(working_labels) <= len(unique_labels):
        return None

    sample_size = deterministic_metric_sample_size(len(working_labels))
    if sample_size < 2:
        return None

    return float(
        silhouette_score(
            working_features,
            working_labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
    )


# ---------------------------------------------------------------------------
# Feature preparation and method grids
# ---------------------------------------------------------------------------


def prepare_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Create stable numerical behaviour features for unsupervised learning.

    Source variables:
        Raw visitor counts and activity duration from visitor_features.csv.

    Logic:
        Count-like variables are log transformed so extreme visitors do not
        dominate the geometry. View and cart shares preserve behavioural mix.

    Output:
        Seven finite numerical features. They are scaled next, then used by
        PCA, K-Means, DBSCAN, and LOF.
    """

    missing = [column for column in RAW_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(
            "visitor_features.csv is missing required columns: "
            f"{missing}"
        )

    clean = data[RAW_FEATURES].copy()
    for column in RAW_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        median = clean[column].median()
        clean[column] = clean[column].fillna(0 if pd.isna(median) else median)
        clean[column] = clean[column].clip(lower=0)

    total = clean["total_events"].clip(lower=1)
    derived = pd.DataFrame(index=clean.index)
    derived["Log total events"] = np.log1p(clean["total_events"])
    derived["Log views"] = np.log1p(clean["view_count"])
    derived["Log add-to-cart"] = np.log1p(clean["addtocart_count"])
    derived["Log unique items"] = np.log1p(clean["unique_items"])
    derived["Log activity hours"] = np.log1p(
        clean["activity_span_ms"] / 3_600_000
    )
    derived["View share"] = (clean["view_count"] / total).clip(0, 1)
    derived["Cart share"] = (
        clean["addtocart_count"] / total
    ).clip(0, 1)

    return derived.replace([np.inf, -np.inf], 0).fillna(0)


def evaluate_kmeans_grid(scaled: np.ndarray) -> pd.DataFrame:
    """Evaluate the complete k=1..10 elbow grid on one evidence sample."""

    rows: list[dict[str, Any]] = []

    for k in KMEANS_GRID:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        # Limit native BLAS threads so repeated grid fits remain stable on
        # laptops and CI runners with many logical CPU cores.
        with threadpool_limits(limits=1):
            labels = model.fit_predict(scaled)
        silhouette = (
            safe_silhouette(scaled, labels, ignore_noise=False)
            if k >= 2
            else None
        )
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": silhouette,
                "cluster_count": int(len(set(labels.tolist()))),
                "sample_rows": int(len(scaled)),
                "silhouette_evaluation_rows": deterministic_metric_sample_size(
                    len(scaled)
                ),
                "candidate_scope": (
                    "required_3_5_7"
                    if k in KMEANS_DECISION_CANDIDATES
                    else "elbow_support"
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_kmeans_k(grid: pd.DataFrame) -> int:
    """Choose the best same-sample candidate from the required 3/5/7 set."""

    candidates = grid.loc[
        grid["k"].isin(KMEANS_DECISION_CANDIDATES)
    ].copy()
    candidates["silhouette_score"] = pd.to_numeric(
        candidates["silhouette_score"],
        errors="coerce",
    )
    candidates = candidates.dropna(subset=["silhouette_score"])

    if candidates.empty:
        return 5

    best = candidates.sort_values(
        ["silhouette_score", "inertia", "k"],
        ascending=[False, True, True],
    ).iloc[0]
    return int(best["k"])


def evaluate_dbscan_grid(scaled: np.ndarray) -> pd.DataFrame:
    """Evaluate the required 3x3 DBSCAN parameter grid on one sample."""

    rows: list[dict[str, Any]] = []

    for eps in DBSCAN_EPS_GRID:
        for min_samples in DBSCAN_MIN_SAMPLES_GRID:
            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            ).fit_predict(scaled)
            non_noise = labels != -1
            cluster_count = len(set(labels.tolist()) - {-1})
            noise_count = int((labels == -1).sum())
            silhouette = safe_silhouette(
                scaled,
                labels,
                ignore_noise=True,
            )
            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "cluster_count": int(cluster_count),
                    "noise_count": noise_count,
                    "noise_rate": float(noise_count / len(labels)),
                    "non_noise_rows": int(non_noise.sum()),
                    "silhouette_score_no_noise": silhouette,
                    "sample_rows": int(len(labels)),
                    "silhouette_evaluation_rows": deterministic_metric_sample_size(
                        int(non_noise.sum())
                    ),
                }
            )

    return pd.DataFrame(rows)


def choose_dbscan_parameters(grid: pd.DataFrame) -> tuple[float, int]:
    """Choose one useful same-sample DBSCAN setting with guardrails.

    Selection rule:
        1. Require at least two clusters and a valid silhouette score.
        2. Prefer 2-12 clusters and no more than 20% noise.
        3. If that set is empty, allow up to 35% noise.
        4. Rank by silhouette, then lower noise, then fewer clusters.

    This avoids selecting an isolated mathematical maximum that fragments the
    sample into dozens of tiny groups.
    """

    data = grid.copy()
    data["silhouette_score_no_noise"] = pd.to_numeric(
        data["silhouette_score_no_noise"],
        errors="coerce",
    )
    valid = data.loc[
        data["silhouette_score_no_noise"].notna()
        & (data["cluster_count"] >= 2)
    ].copy()

    preferred = valid.loc[
        valid["cluster_count"].between(2, 12)
        & (valid["noise_rate"] <= 0.20)
    ]
    if preferred.empty:
        preferred = valid.loc[valid["noise_rate"] <= 0.35]
    if preferred.empty:
        preferred = valid

    if preferred.empty:
        return 1.0, 5

    best = preferred.sort_values(
        ["silhouette_score_no_noise", "noise_rate", "cluster_count"],
        ascending=[False, True, True],
    ).iloc[0]
    return float(best["eps"]), int(best["min_samples"])


def choose_lof_parameters(summary: pd.DataFrame, sample_rows: int) -> tuple[int, float]:
    """Choose LOF neighbourhood and contamination from existing evidence."""

    neighbours = 20
    contamination = 0.04

    if not summary.empty:
        for column in ["n_neighbors", "neighbors", "neighbour_count"]:
            if column in summary.columns:
                value = pd.to_numeric(
                    summary[column], errors="coerce"
                ).dropna()
                if not value.empty:
                    neighbours = int(value.iloc[0])
                    break

        for column in ["contamination", "outlier_rate"]:
            if column in summary.columns:
                value = pd.to_numeric(
                    summary[column], errors="coerce"
                ).dropna()
                if not value.empty and 0 < float(value.iloc[0]) < 0.5:
                    contamination = float(value.iloc[0])
                    break

    neighbours = max(5, min(neighbours, sample_rows - 1))
    return neighbours, contamination


def build_k_distance_table(
    scaled: np.ndarray,
    min_samples: int,
) -> pd.DataFrame:
    """Build the sorted k-distance curve used to inspect DBSCAN epsilon."""

    neighbours = max(2, min(min_samples, len(scaled)))
    model = NearestNeighbors(n_neighbors=neighbours)
    model.fit(scaled)
    distances, _ = model.kneighbors(scaled)
    kth_distance = np.sort(distances[:, -1])

    if len(kth_distance) > K_DISTANCE_EXPORT_ROWS:
        positions = np.linspace(
            0,
            len(kth_distance) - 1,
            K_DISTANCE_EXPORT_ROWS,
        ).round().astype(int)
        positions = np.unique(positions)
    else:
        positions = np.arange(len(kth_distance))

    denominator = max(len(kth_distance) - 1, 1)
    return pd.DataFrame(
        {
            "rank": positions + 1,
            "rank_percentile": positions / denominator,
            "k_distance": kth_distance[positions],
            "neighbour_rank": neighbours,
            "source_rows": len(kth_distance),
        }
    )


# ---------------------------------------------------------------------------
# Derived profiles and investigation evidence
# ---------------------------------------------------------------------------


def build_cluster_business_summary(
    sample: pd.DataFrame,
    labels: np.ndarray,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create cluster size, value, context, and conversion evidence."""

    working = pd.DataFrame(index=sample.index)
    working["cluster"] = labels.astype(int)
    working["total_events"] = pd.to_numeric(
        sample["total_events"], errors="coerce"
    )
    working["views"] = pd.to_numeric(
        sample["view_count"], errors="coerce"
    )
    working["add_to_cart"] = pd.to_numeric(
        sample["addtocart_count"], errors="coerce"
    )
    working["unique_items"] = pd.to_numeric(
        sample["unique_items"], errors="coerce"
    )
    working["activity_hours"] = pd.to_numeric(
        sample["activity_span_ms"], errors="coerce"
    ) / 3_600_000
    working["cart_share"] = feature_frame["Cart share"]
    working["view_share"] = feature_frame["View share"]

    if "purchased" in sample.columns:
        working["purchased"] = coerce_binary(sample["purchased"])
    else:
        working["purchased"] = np.nan

    if "purchase_intent_score" in sample.columns:
        working["purchase_intent_score"] = pd.to_numeric(
            sample["purchase_intent_score"],
            errors="coerce",
        )
    else:
        working["purchase_intent_score"] = np.nan

    if "visitor_segment" in sample.columns:
        working["visitor_segment"] = sample["visitor_segment"]

    grouped = working.groupby("cluster", observed=True)
    summary = grouped.agg(
        visitor_count=("cluster", "size"),
        purchase_rate=("purchased", "mean"),
        avg_purchase_intent_score=("purchase_intent_score", "mean"),
        avg_total_events=("total_events", "mean"),
        avg_views=("views", "mean"),
        avg_add_to_cart=("add_to_cart", "mean"),
        avg_unique_items=("unique_items", "mean"),
        avg_activity_hours=("activity_hours", "mean"),
        avg_cart_share=("cart_share", "mean"),
        avg_view_share=("view_share", "mean"),
    ).reset_index()

    summary["visitor_share"] = summary["visitor_count"] / len(working)
    if "visitor_segment" in working.columns:
        dominant = grouped["visitor_segment"].apply(safe_mode)
        summary["dominant_existing_segment"] = summary["cluster"].map(
            dominant
        )
    else:
        summary["dominant_existing_segment"] = "Unavailable"

    return summary.sort_values("cluster").reset_index(drop=True)


def build_lof_feature_profile(
    scaled_frame: pd.DataFrame,
    outlier_mask: np.ndarray,
) -> pd.DataFrame:
    """Compare standardized behaviour of flagged and normal visitors."""

    normal = scaled_frame.loc[~outlier_mask]
    outliers = scaled_frame.loc[outlier_mask]

    normal_mean = normal.mean() if not normal.empty else pd.Series(0, index=scaled_frame.columns)
    outlier_mean = outliers.mean() if not outliers.empty else pd.Series(0, index=scaled_frame.columns)

    profile = pd.DataFrame(
        {
            "feature": scaled_frame.columns,
            "normal_standardized_mean": normal_mean.values,
            "outlier_standardized_mean": outlier_mean.values,
        }
    )
    profile["standardized_gap"] = (
        profile["outlier_standardized_mean"]
        - profile["normal_standardized_mean"]
    )
    profile["absolute_gap"] = profile["standardized_gap"].abs()
    return profile.sort_values("absolute_gap", ascending=False).reset_index(drop=True)


def build_lof_investigation(
    sample: pd.DataFrame,
    scaled_frame: pd.DataFrame,
    lof_scores: np.ndarray,
    lof_outlier: np.ndarray,
    severity: np.ndarray,
) -> pd.DataFrame:
    """Create a governed top-outlier table with feature deviations."""

    investigation = pd.DataFrame(index=sample.index)
    investigation["sample_row_id"] = np.arange(len(sample))

    for column in OPTIONAL_CONTEXT_COLUMNS + RAW_FEATURES:
        if column in sample.columns:
            investigation[column] = sample[column].values

    investigation["lof_score"] = lof_scores
    investigation["lof_outlier"] = lof_outlier
    investigation["lof_severity"] = severity

    z_columns: list[str] = []
    feature_lookup: dict[str, str] = {}
    for feature in scaled_frame.columns:
        column = f"z_{feature_slug(feature)}"
        investigation[column] = scaled_frame[feature].values
        z_columns.append(column)
        feature_lookup[column] = feature

    absolute = investigation[z_columns].abs()
    dominant_column = absolute.idxmax(axis=1)
    investigation["dominant_deviation_feature"] = dominant_column.map(
        feature_lookup
    )
    investigation["dominant_deviation_z"] = [
        investigation.at[index, column]
        for index, column in dominant_column.items()
    ]

    investigation = investigation.sort_values(
        "lof_score",
        ascending=False,
    ).head(INVESTIGATION_ROWS)
    investigation.insert(0, "investigation_rank", range(1, len(investigation) + 1))
    return investigation.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------


def build_outputs(sample: pd.DataFrame) -> dict[str, Any]:
    """Calculate one internally consistent Package 1 evidence set."""

    feature_frame = prepare_feature_frame(sample)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)
    scaled_frame = pd.DataFrame(
        scaled,
        columns=feature_frame.columns,
        index=sample.index,
    )

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)

    # Evaluate complete same-sample grids before fitting the selected views.
    kmeans_grid = evaluate_kmeans_grid(scaled)
    selected_k = choose_kmeans_k(kmeans_grid)
    kmeans_grid["selected"] = kmeans_grid["k"].eq(selected_k)

    dbscan_grid = evaluate_dbscan_grid(scaled)
    selected_eps, selected_min_samples = choose_dbscan_parameters(dbscan_grid)
    dbscan_grid["selected"] = (
        dbscan_grid["eps"].eq(selected_eps)
        & dbscan_grid["min_samples"].eq(selected_min_samples)
    )

    kmeans = KMeans(
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    with threadpool_limits(limits=1):
        kmeans_labels = kmeans.fit_predict(scaled)

    dbscan = DBSCAN(
        eps=selected_eps,
        min_samples=selected_min_samples,
    )
    dbscan_labels = dbscan.fit_predict(scaled)

    point_type = np.full(len(sample), "Border", dtype=object)
    point_type[dbscan_labels == -1] = "Noise"
    point_type[dbscan.core_sample_indices_] = "Core"

    lof_summary = read_optional_csv(LOF_SUMMARY_PATH)
    lof_neighbours, lof_contamination = choose_lof_parameters(
        lof_summary,
        len(sample),
    )
    lof = LocalOutlierFactor(
        n_neighbors=lof_neighbours,
        contamination=lof_contamination,
    )
    lof_labels = lof.fit_predict(scaled)
    lof_scores = -lof.negative_outlier_factor_
    lof_outlier = lof_labels == -1

    watch_cut = float(np.quantile(lof_scores, 0.90))
    high_cut = float(np.quantile(lof_scores, 0.96))
    critical_cut = float(np.quantile(lof_scores, 0.99))
    severity = np.select(
        [
            lof_scores >= critical_cut,
            lof_scores >= high_cut,
            lof_scores >= watch_cut,
        ],
        ["Critical", "High", "Watch"],
        default="Normal",
    )

    lof_threshold = (
        float(lof_scores[lof_outlier].min())
        if lof_outlier.any()
        else float(np.quantile(lof_scores, 1 - lof_contamination))
    )

    projection = pd.DataFrame(
        {
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "PC3": coordinates[:, 2],
            "kmeans_cluster": kmeans_labels.astype(str),
            "dbscan_cluster": dbscan_labels.astype(str),
            "dbscan_point_type": point_type,
            "lof_score": lof_scores,
            "lof_outlier": lof_outlier,
            "lof_status": np.where(lof_outlier, "Outlier", "Normal"),
            "lof_severity": severity,
        },
        index=sample.index,
    )

    for column in OPTIONAL_CONTEXT_COLUMNS + RAW_FEATURES:
        if column in sample.columns:
            projection[column] = sample[column].values

    variance = pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2", "PC3"],
    )
    loadings.insert(0, "feature", feature_frame.columns)

    profile_frame = scaled_frame.copy()
    profile_frame["cluster"] = kmeans_labels
    cluster_profile = (
        profile_frame.groupby("cluster", as_index=False)
        .mean()
        .melt(
            id_vars="cluster",
            var_name="feature",
            value_name="standardized_mean",
        )
    )

    cluster_business = build_cluster_business_summary(
        sample,
        kmeans_labels,
        feature_frame,
    )
    k_distance = build_k_distance_table(
        scaled,
        selected_min_samples,
    )
    lof_feature_profile = build_lof_feature_profile(
        scaled_frame,
        lof_outlier,
    )
    lof_investigation = build_lof_investigation(
        sample,
        scaled_frame,
        lof_scores,
        lof_outlier,
        severity,
    )

    kmeans_selected = kmeans_grid.loc[kmeans_grid["selected"]].iloc[0]
    dbscan_selected = dbscan_grid.loc[dbscan_grid["selected"]].iloc[0]

    method_summary = pd.DataFrame(
        [
            {
                "method": "K-Means",
                "selected_parameters": f"k={selected_k}",
                "cluster_count": int(selected_k),
                "noise_or_outlier_count": 0,
                "noise_or_outlier_rate": 0.0,
                "silhouette_score": float(
                    kmeans_selected["silhouette_score"]
                ),
                "sample_rows": len(sample),
                "evidence_scope": "Package 1 deterministic sample",
            },
            {
                "method": "DBSCAN",
                "selected_parameters": (
                    f"eps={selected_eps}, "
                    f"min_samples={selected_min_samples}"
                ),
                "cluster_count": int(dbscan_selected["cluster_count"]),
                "noise_or_outlier_count": int(
                    dbscan_selected["noise_count"]
                ),
                "noise_or_outlier_rate": float(
                    dbscan_selected["noise_rate"]
                ),
                "silhouette_score": float(
                    dbscan_selected["silhouette_score_no_noise"]
                ),
                "sample_rows": len(sample),
                "evidence_scope": "Package 1 deterministic sample",
            },
            {
                "method": "LOF",
                "selected_parameters": (
                    f"n_neighbors={lof_neighbours}, "
                    f"contamination={lof_contamination:.3f}"
                ),
                "cluster_count": np.nan,
                "noise_or_outlier_count": int(lof_outlier.sum()),
                "noise_or_outlier_rate": float(lof_outlier.mean()),
                "silhouette_score": np.nan,
                "sample_rows": len(sample),
                "evidence_scope": "Package 1 deterministic sample",
            },
        ]
    )

    parameters = {
        "kmeans_k": selected_k,
        "kmeans_selection_candidates": list(KMEANS_DECISION_CANDIDATES),
        "dbscan_eps": selected_eps,
        "dbscan_min_samples": selected_min_samples,
        "dbscan_eps_grid": list(DBSCAN_EPS_GRID),
        "dbscan_min_samples_grid": list(DBSCAN_MIN_SAMPLES_GRID),
        "lof_neighbours": lof_neighbours,
        "lof_contamination": lof_contamination,
        "lof_score_threshold": lof_threshold,
    }

    return {
        "projection": projection,
        "variance": variance,
        "loadings": loadings,
        "cluster_profile": cluster_profile,
        "cluster_business": cluster_business,
        "kmeans_grid": kmeans_grid,
        "dbscan_grid": dbscan_grid,
        "dbscan_k_distance": k_distance,
        "lof_feature_profile": lof_feature_profile,
        "lof_investigation": lof_investigation,
        "method_summary": method_summary,
        "parameters": parameters,
        "features": list(feature_frame.columns),
    }


def save_outputs(
    outputs: dict[str, Any],
    *,
    source_rows: int,
    sample_rows: int,
) -> None:
    """Save compact tables and one transparent generation manifest."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Manifest labels and in-memory result keys are deliberately separated.
    # PCA uses short internal keys (``variance`` and ``loadings``), while the
    # exported contract keeps explicit names such as ``pca_variance``.
    output_specs = {
        "projection": ("projection", PROJECTION_PATH),
        "pca_variance": ("variance", PCA_VARIANCE_PATH),
        "pca_loadings": ("loadings", PCA_LOADINGS_PATH),
        "cluster_profile": ("cluster_profile", CLUSTER_PROFILE_PATH),
        "cluster_business": ("cluster_business", CLUSTER_BUSINESS_PATH),
        "kmeans_grid": ("kmeans_grid", KMEANS_GRID_PATH),
        "dbscan_grid": ("dbscan_grid", DBSCAN_GRID_PATH),
        "dbscan_k_distance": ("dbscan_k_distance", DBSCAN_K_DISTANCE_PATH),
        "lof_feature_profile": ("lof_feature_profile", LOF_FEATURE_PROFILE_PATH),
        "lof_investigation": ("lof_investigation", LOF_INVESTIGATION_PATH),
        "method_summary": ("method_summary", METHOD_SUMMARY_PATH),
    }

    for _, (output_key, path) in output_specs.items():
        outputs[output_key].to_csv(path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "offline analytical validation evidence",
        "source_path": str(FEATURES_PATH),
        "source_sha256": file_sha256(FEATURES_PATH),
        "source_grain": "one row per visitor",
        "source_key": "visitorid when available",
        "source_rows": source_rows,
        "sample_rows": sample_rows,
        "sample_rule": (
            "deterministic random sample with random_state=42"
        ),
        "random_state": RANDOM_STATE,
        "features": outputs["features"],
        "parameters": outputs["parameters"],
        "selection_rules": {
            "kmeans": (
                "highest same-sample silhouette among required k=3,5,7"
            ),
            "dbscan": (
                "valid same-sample silhouette with cluster/noise guardrails"
            ),
            "lof": (
                "existing validated contamination/neighbour settings when "
                "available; otherwise documented defaults"
            ),
        },
        "metric_rules": {
            "silhouette_sample_rows": SILHOUETTE_SAMPLE_ROWS,
            "dbscan_silhouette": "calculated after excluding noise rows",
            "lof_threshold": (
                "minimum LOF score among algorithm-flagged outliers"
            ),
        },
        "outputs": {
            label: str(path)
            for label, (_, path) in output_specs.items()
        },
        "limitations": [
            "The projection uses a deterministic sample for browser performance.",
            "PCA coordinates are a visual summary and not production predictions.",
            "K-Means, DBSCAN, and LOF are offline exploratory evidence.",
            "Silhouette calculations use at most 3,000 deterministic rows.",
            "Cluster recommendations require business testing before deployment.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Build and save the complete Package 1 evidence set."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
    )
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "data/processed/visitor_features.csv is missing. "
            "Run feature engineering first."
        )

    source = pd.read_csv(FEATURES_PATH)
    if source.empty:
        raise ValueError("visitor_features.csv contains no rows.")

    sample_rows = min(max(args.sample_rows, 500), len(source))
    sample = source.sample(
        sample_rows,
        random_state=RANDOM_STATE,
    ).copy()

    outputs = build_outputs(sample)
    save_outputs(
        outputs,
        source_rows=len(source),
        sample_rows=len(sample),
    )

    print("=== ML VALIDATION EVIDENCE RESULT ===")
    print(f"SOURCE_ROWS={len(source)}")
    print(f"SAMPLE_ROWS={len(sample)}")
    print(f"KMEANS_GRID_ROWS={len(outputs['kmeans_grid'])}")
    print(f"KMEANS_K={outputs['parameters']['kmeans_k']}")
    print(f"DBSCAN_GRID_ROWS={len(outputs['dbscan_grid'])}")
    print(f"DBSCAN_EPS={outputs['parameters']['dbscan_eps']}")
    print(
        "DBSCAN_MIN_SAMPLES="
        f"{outputs['parameters']['dbscan_min_samples']}"
    )
    print(
        "LOF_CONTAMINATION="
        f"{outputs['parameters']['lof_contamination']:.4f}"
    )
    print(f"PROJECTION={PROJECTION_PATH}")
    print(f"MANIFEST={MANIFEST_PATH}")
    print("GOOD: Coherent Package 1 analytical evidence created")
    print("=== END ML VALIDATION EVIDENCE RESULT ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
