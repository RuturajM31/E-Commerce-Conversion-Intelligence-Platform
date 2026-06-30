"""Nearest-neighbour and native model-explanation extension.

The visitor predictor continues to score manual examples. This module adds a
separate evidence explorer for real saved visitor rows. Heavy objects are built
only when the user opens the section and requests a search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.ui.closure_core import (
    add_metadata_columns,
    export_metadata,
    file_timestamp,
    finish_figure,
    read_csv_if_exists,
    safe_divide,
)
from app.ui.components import (
    inject_product_css,
    render_detail_cards,
    render_empty_state,
    render_evidence_chart,
    render_section_header,
    render_source_note,
)


FEATURE_PATH = Path("data/processed/visitor_features.csv")
SCORE_PATHS = (
    Path("data/processed/final_champion_visitor_scores.csv"),
    Path("data/processed/champion_visitor_scores.csv"),
)
ANOMALY_PATH = Path("reports/tables/top_anomalous_visitors.csv")
MODEL_PATH = Path("models/trained/final_champion_model.joblib")
METADATA_PATH = Path("models/metadata/final_champion_metadata.json")
GLOBAL_IMPACT_PATH = Path(
    "reports/visuals/ml_visual_intelligence/explainability/"
    "global_feature_impact.csv"
)

DEFAULT_FEATURES = (
    "total_events",
    "view_count",
    "addtocart_count",
    "unique_items",
    "activity_span_ms",
    "cart_to_view_ratio",
    "events_per_unique_item",
)

FEATURE_LABELS = {
    "total_events": "Total events",
    "view_count": "Views",
    "addtocart_count": "Add-to-cart events",
    "unique_items": "Unique items",
    "activity_span_ms": "Activity span",
    "cart_to_view_ratio": "Cart-to-view ratio",
    "events_per_unique_item": "Events per unique item",
}


def _score_path() -> Path | None:
    """Return the first available visitor score file."""

    for path in SCORE_PATHS:
        if path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_visitor_evidence() -> pd.DataFrame:
    """Load visitor features and left-join available score/anomaly evidence."""

    if not FEATURE_PATH.is_file():
        return pd.DataFrame()

    features = pd.read_csv(FEATURE_PATH)
    if "visitorid" not in features.columns:
        return pd.DataFrame()

    features["visitorid"] = features["visitorid"].astype(str)
    output = features.copy()

    score_path = _score_path()
    if score_path is not None:
        scores = pd.read_csv(score_path)
        if {"visitorid", "purchase_intent_score"}.issubset(scores.columns):
            scores = scores[[
                column
                for column in (
                    "visitorid",
                    "purchase_intent_score",
                    "predicted_conversion",
                    "production_threshold",
                )
                if column in scores.columns
            ]].copy()
            scores["visitorid"] = scores["visitorid"].astype(str)
            scores = scores.drop_duplicates("visitorid")
            output = output.merge(
                scores,
                on="visitorid",
                how="left",
                validate="one_to_one",
            )

    anomaly = read_csv_if_exists(str(ANOMALY_PATH))
    if not anomaly.empty and "visitorid" in anomaly.columns:
        anomaly_columns = [
            column
            for column in (
                "visitorid",
                "purchase_intent_segment",
                "anomaly_risk_score",
                "anomaly_risk_band",
                "final_anomaly_flag",
                "business_interpretation",
            )
            if column in anomaly.columns
        ]
        anomaly = anomaly[anomaly_columns].copy()
        anomaly["visitorid"] = anomaly["visitorid"].astype(str)
        anomaly = anomaly.drop_duplicates("visitorid")
        output = output.merge(
            anomaly,
            on="visitorid",
            how="left",
            validate="one_to_one",
        )

    return output


@st.cache_resource(show_spinner=False)
def build_neighbor_resource(
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build one cached scaled nearest-neighbour index."""

    data = load_visitor_evidence()
    if data.empty:
        raise FileNotFoundError("Visitor feature evidence is unavailable.")

    missing = [column for column in feature_names if column not in data.columns]
    if missing:
        raise ValueError("Missing neighbour features: " + ", ".join(missing))

    feature_frame = data[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))

    if feature_frame.isna().any().any():
        raise ValueError("Neighbour features remain missing after median handling.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    index = NearestNeighbors(
        metric="euclidean",
        algorithm="auto",
    )
    index.fit(scaled)

    return {
        "data": data.reset_index(drop=True),
        "feature_frame": feature_frame.reset_index(drop=True),
        "scaled": scaled,
        "scaler": scaler,
        "index": index,
        "feature_names": feature_names,
    }


def find_visitor_position(data: pd.DataFrame, visitor_id: str) -> int | None:
    """Return the exact visitor row position without fuzzy identity matching."""

    matches = data.index[data["visitorid"].astype(str).eq(str(visitor_id))]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_neighbours(
    resource: dict[str, Any],
    *,
    visitor_id: str,
    neighbor_count: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return selected visitor, neighbour rows, and standardized comparison."""

    data = resource["data"]
    position = find_visitor_position(data, visitor_id)
    if position is None:
        raise KeyError(f"Visitor ID not found: {visitor_id}")

    count = min(max(int(neighbor_count), 1) + 1, len(data))
    distances, indices = resource["index"].kneighbors(
        resource["scaled"][position].reshape(1, -1),
        n_neighbors=count,
    )

    rows = []
    for distance, index in zip(distances[0], indices[0]):
        if int(index) == position:
            continue
        record = data.iloc[int(index)].copy()
        record["neighbor_distance"] = float(distance)
        rows.append(record)
        if len(rows) >= neighbor_count:
            break

    neighbours = pd.DataFrame(rows).reset_index(drop=True)
    neighbours["neighbor_rank"] = neighbours.index + 1

    selected_scaled = pd.Series(
        resource["scaled"][position],
        index=resource["feature_names"],
        name="Selected visitor",
    )
    neighbour_positions = [int(value) for value in indices[0] if int(value) != position][:neighbor_count]
    mean_values = resource["scaled"][neighbour_positions].mean(axis=0)
    comparison = pd.DataFrame(
        {
            "Feature": list(resource["feature_names"]),
            "Selected visitor": selected_scaled.values,
            "Neighbour average": mean_values,
        }
    )

    return data.iloc[position], neighbours, comparison


def build_neighbor_score_figure(neighbours: pd.DataFrame) -> go.Figure:
    """Show distance and score for the selected visitor's neighbours."""

    frame = neighbours.copy()
    frame["visitorid"] = frame["visitorid"].astype(str)
    score_column = "purchase_intent_score"
    if score_column not in frame.columns:
        frame[score_column] = np.nan

    hover_columns = {
        column: True
        for column in (
            "neighbor_rank",
            "purchase_intent_segment",
            "final_anomaly_flag",
        )
        if column in frame.columns
    }

    figure = px.scatter(
        frame,
        x="neighbor_distance",
        y=score_column,
        color="anomaly_risk_band" if "anomaly_risk_band" in frame.columns else None,
        size="total_events" if "total_events" in frame.columns else None,
        hover_name="visitorid",
        hover_data=hover_columns,
    )
    figure.update_xaxes(title="Scaled feature distance")
    figure.update_yaxes(title="Purchase-intent score", tickformat=".0%")
    return finish_figure(
        figure,
        title="Nearest-visitor evidence",
        subtitle="Closer points have more similar standardized behaviour across the selected feature set.",
        height=500,
        legend_title="Anomaly band",
    )


def build_feature_comparison_figure(comparison: pd.DataFrame) -> go.Figure:
    """Compare selected visitor and neighbour average in standardized units."""

    melted = comparison.melt(
        id_vars="Feature",
        var_name="Population",
        value_name="Standardized value",
    )
    melted["Feature"] = melted["Feature"].map(FEATURE_LABELS).fillna(melted["Feature"])

    figure = px.bar(
        melted,
        x="Feature",
        y="Standardized value",
        color="Population",
        barmode="group",
    )
    return finish_figure(
        figure,
        title="Selected visitor versus local neighbourhood",
        subtitle="Positive values are above the full-population mean; negative values are below it.",
        height=500,
        legend_title="Comparison",
    )


def build_global_impact_figure(impact: pd.DataFrame) -> go.Figure:
    """Build an interactive global feature-impact ranking."""

    frame = impact.copy().sort_values(
        "mean_abs_contribution",
        ascending=True,
    )
    figure = px.bar(
        frame,
        x="mean_abs_contribution",
        y="display_name",
        orientation="h",
        color="mean_signed_contribution",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        hover_data={
            "impact_rank": True,
            "value_contribution_correlation": ":.3f",
        },
    )
    figure.update_xaxes(title="Mean absolute native contribution")
    figure.update_yaxes(title="")
    return finish_figure(
        figure,
        title="Global model feature impact",
        subtitle="Bar length measures average contribution magnitude; colour shows average signed direction.",
        height=500,
    )


def compute_local_contributions(row: pd.Series) -> pd.DataFrame:
    """Compute genuine native XGBoost contributions for one saved visitor."""

    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        return pd.DataFrame()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_columns", list(DEFAULT_FEATURES))
    if not all(column in row.index for column in feature_names):
        return pd.DataFrame()

    model = joblib.load(MODEL_PATH)
    values = pd.DataFrame(
        [[float(row[column]) for column in feature_names]],
        columns=feature_names,
    )

    try:
        import xgboost as xgb

        booster = model.get_booster()
        contribution_array = booster.predict(
            xgb.DMatrix(values, feature_names=feature_names),
            pred_contribs=True,
        )[0]
    except Exception:
        return pd.DataFrame()

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "display_name": [FEATURE_LABELS.get(name, name) for name in feature_names],
            "feature_value": [float(row[name]) for name in feature_names],
            "contribution": contribution_array[:-1],
        }
    )
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0,
        "Raises score",
        "Lowers score",
    )
    contributions["absolute_contribution"] = contributions["contribution"].abs()
    return contributions.sort_values("absolute_contribution", ascending=False)


def build_local_contribution_figure(contributions: pd.DataFrame) -> go.Figure:
    """Build a local positive/negative contribution bar chart."""

    frame = contributions.sort_values("contribution")
    figure = px.bar(
        frame,
        x="contribution",
        y="display_name",
        orientation="h",
        color="direction",
        hover_data={"feature_value": True, "absolute_contribution": False},
    )
    figure.update_xaxes(title="Native XGBoost contribution in model-margin units")
    figure.update_yaxes(title="")
    return finish_figure(
        figure,
        title="Selected visitor score drivers",
        subtitle="Positive contributions raise the model margin and negative contributions lower it.",
        height=520,
        legend_title="Direction",
    )


def neighborhood_action(selected: pd.Series, neighbours: pd.DataFrame) -> str:
    """Create a rules-based action from local score and anomaly composition."""

    selected_score = float(selected.get("purchase_intent_score", 0) or 0)
    neighbour_score = pd.to_numeric(
        neighbours.get("purchase_intent_score", pd.Series(dtype=float)),
        errors="coerce",
    ).mean()
    anomaly_rate = (
        neighbours.get("final_anomaly_flag", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .mean()
    )

    if anomaly_rate >= 0.25:
        return "Review unusual behaviour before campaign activation."
    if selected_score >= 0.8 and neighbour_score >= 0.6:
        return "Prioritise with a high-intent offer and measure incremental lift."
    if selected_score >= 0.5:
        return "Use a lower-cost retargeting message and watch local response."
    return "Keep in low-cost nurture rather than immediate paid outreach."


def render_visitor_similarity_explainability() -> None:
    """Render KNN and XAI evidence inside one lazy expander."""

    inject_product_css()

    render_section_header(
        "Similarity and explainability",
        "Explore real saved visitors, nearest neighbours, and native model drivers",
        (
            "This evidence explorer is separate from the manual score form. It uses saved visitor features, "
            "cached nearest-neighbour search, and native XGBoost contributions on demand."
        ),
    )

    with st.expander("Open visitor similarity and explanation explorer", expanded=False):
        global_impact = read_csv_if_exists(str(GLOBAL_IMPACT_PATH))
        if not global_impact.empty:
            global_figure = build_global_impact_figure(global_impact)
            strongest_global = global_impact.sort_values(
                "mean_abs_contribution",
                ascending=False,
            ).iloc[0]
            render_evidence_chart(
                figure=global_figure,
                key="closure_global_feature_impact",
                what_it_shows="Average native XGBoost contribution magnitude and signed direction across the governed explanation sample.",
                how_to_read="Longer bars matter more globally; colour indicates whether the average signed contribution raises or lowers the model margin.",
                actual_finding=(
                    f"The strongest global feature is {strongest_global['display_name']} "
                    f"with mean absolute contribution {float(strongest_global['mean_abs_contribution']):.3f}."
                ),
                conclusion="Global importance explains which behaviours the model relies on most across the evidence sample.",
                recommended_action="Use global impact to prioritise data-quality monitoring and explanation review, not causal intervention.",
                limitation="Global averages can hide visitor-level differences and do not prove causality.",
                source=str(GLOBAL_IMPACT_PATH),
                evidence_type="Native XGBoost global contribution summary",
                refreshed_at=file_timestamp(GLOBAL_IMPACT_PATH),
            )

        data = load_visitor_evidence()
        if data.empty:
            render_empty_state(
                title="Visitor evidence is unavailable",
                message="The production visitor feature table is missing or has no visitor ID.",
                next_action="Regenerate data/processed/visitor_features.csv and final visitor scores.",
            )
            return

        available_features = [column for column in DEFAULT_FEATURES if column in data.columns]
        selected_features = st.multiselect(
            "Features used for similarity",
            options=available_features,
            default=available_features,
            help="All selected features are standardized before distance calculation.",
            key="knn_feature_set",
        )

        if len(selected_features) < 2:
            st.warning("Select at least two features for a meaningful neighbour search.")
            return

        neighbor_count = st.slider(
            "Number of neighbours",
            min_value=3,
            max_value=20,
            value=8,
            key="knn_neighbor_count",
        )

        default_id = str(data.iloc[0]["visitorid"])
        visitor_id = st.text_input(
            "Exact visitor ID",
            value=default_id,
            help="Use an exact saved visitor ID. Identity is never fuzzy-matched.",
            key="knn_visitor_id",
        )

        run_search = st.button(
            "Find nearest visitors and explain score",
            type="primary",
            key="knn_run_search",
        )

        if not run_search:
            render_source_note(
                source=str(FEATURE_PATH),
                evidence_type="Saved production visitor features",
                refreshed_at=file_timestamp(FEATURE_PATH),
                extra=f"{len(data):,} visitor rows are available; heavy search runs only on request.",
            )
            return

        with st.spinner("Building or loading the cached neighbour index..."):
            resource = build_neighbor_resource(tuple(selected_features))
            try:
                selected, neighbours, comparison = get_neighbours(
                    resource,
                    visitor_id=visitor_id,
                    neighbor_count=neighbor_count,
                )
            except KeyError:
                st.error("Visitor ID was not found in the saved feature table.")
                return

        local_score = float(selected.get("purchase_intent_score", 0) or 0)
        neighbour_scores = pd.to_numeric(
            neighbours.get("purchase_intent_score", pd.Series(dtype=float)),
            errors="coerce",
        )
        local_anomaly_rate = (
            neighbours.get("final_anomaly_flag", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .mean()
        )

        render_detail_cards(
            [
                ("Selected visitor", str(selected["visitorid"]), f"Saved score {local_score:.1%}"),
                ("Neighbour score", f"{neighbour_scores.mean():.1%}", "Average across returned neighbours"),
                ("Local anomaly rate", f"{local_anomaly_rate:.1%}", "Share of neighbours with matching anomaly evidence"),
                ("Recommended action", "Rules-based", neighborhood_action(selected, neighbours)),
            ]
        )

        score_figure = build_neighbor_score_figure(neighbours)
        render_evidence_chart(
            figure=score_figure,
            key="closure_knn_score_map",
            what_it_shows="Nearest saved visitors by standardized feature distance, score, and anomaly band.",
            how_to_read="Smaller distance means more similar behaviour within the selected feature set.",
            actual_finding=f"The selected visitor score is {local_score:.1%}; the neighbour average is {neighbour_scores.mean():.1%}.",
            conclusion="Local neighbourhood evidence helps distinguish isolated scores from repeated behavioural patterns.",
            recommended_action=neighborhood_action(selected, neighbours),
            limitation="Nearest-neighbour similarity is descriptive and does not prove the same future outcome.",
            source=str(FEATURE_PATH),
            evidence_type="Cached scaled nearest-neighbour evidence",
            refreshed_at=file_timestamp(FEATURE_PATH),
        )

        feature_figure = build_feature_comparison_figure(comparison)
        strongest_gap = comparison.assign(
            gap=(comparison["Selected visitor"] - comparison["Neighbour average"]).abs()
        ).sort_values("gap", ascending=False).iloc[0]
        render_evidence_chart(
            figure=feature_figure,
            key="closure_knn_feature_comparison",
            what_it_shows="The selected visitor and local-neighbour average in standardized feature units.",
            how_to_read="Values above zero exceed the full-population mean; larger gaps explain local difference.",
            actual_finding=(
                f"The largest selected-versus-neighbour gap is {FEATURE_LABELS.get(strongest_gap['Feature'], strongest_gap['Feature'])}."
            ),
            conclusion="Feature-level comparison explains why apparently similar visitors are not identical.",
            recommended_action="Use the strongest gaps as investigation context, not as causal levers.",
            limitation="Standardized values depend on the current saved population and selected feature set.",
            source=str(FEATURE_PATH),
            evidence_type="Standardized local feature comparison",
            refreshed_at=file_timestamp(FEATURE_PATH),
        )

        contributions = compute_local_contributions(selected)
        if contributions.empty:
            render_empty_state(
                title="Local native contribution evidence is unavailable",
                message="The model does not expose compatible native XGBoost contributions for this row.",
                next_action="Use the precomputed global explainability evidence and verify the saved champion model type.",
            )
        else:
            contribution_figure = build_local_contribution_figure(contributions)
            strongest = contributions.iloc[0]
            render_evidence_chart(
                figure=contribution_figure,
                key="closure_local_contributions",
                what_it_shows="Native XGBoost positive and negative contributions for the selected visitor.",
                how_to_read="Bars to the right raise the model margin; bars to the left lower it.",
                actual_finding=(
                    f"The strongest local driver is {strongest['display_name']} with contribution {float(strongest['contribution']):+.3f}."
                ),
                conclusion="The score reflects multiple behavioural signals rather than one hidden rule.",
                recommended_action="Use drivers to explain and review the decision; do not present them as causal effects.",
                limitation="Contributions explain this model and this row only; they do not prove changing a feature changes real buyer behaviour.",
                source=str(MODEL_PATH),
                evidence_type="Native XGBoost local contribution evidence",
                refreshed_at=file_timestamp(MODEL_PATH),
            )

        st.caption("Neighbour evidence table")
        display_columns = [
            column
            for column in (
                "neighbor_rank",
                "visitorid",
                "neighbor_distance",
                "purchase_intent_score",
                "purchase_intent_segment",
                "anomaly_risk_band",
                "final_anomaly_flag",
                "total_events",
                "view_count",
                "addtocart_count",
                "unique_items",
            )
            if column in neighbours.columns
        ]
        st.dataframe(
            neighbours[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        neighbor_export = neighbours.copy()
        neighbor_export.insert(0, "selected_visitorid", str(selected["visitorid"]))
        neighbor_export = add_metadata_columns(
            neighbor_export,
            export_metadata(
                source=str(FEATURE_PATH),
                evidence_type="Nearest-neighbour comparison",
                filters={
                    "features": selected_features,
                    "neighbor_count": neighbor_count,
                },
            ),
        )
        st.download_button(
            "Download selected visitor and neighbours",
            data=neighbor_export.to_csv(index=False),
            file_name="visitor_neighbor_comparison.csv",
            mime="text/csv",
            use_container_width=True,
            key="closure_knn_export",
        )

        if not contributions.empty:
            contribution_export = add_metadata_columns(
                contributions,
                export_metadata(
                    source=str(MODEL_PATH),
                    evidence_type="Native local model contributions",
                ),
            )
            st.download_button(
                "Download selected visitor score drivers",
                data=contribution_export.to_csv(index=False),
                file_name="visitor_local_score_drivers.csv",
                mime="text/csv",
                use_container_width=True,
                key="closure_xai_export",
            )
