"""Tests for the E01 to E06 explainability package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.explainability_data import (
    FEATURE_COLUMNS,
    ExplainabilityBundle,
    prepare_feature_frame,
    select_representative_source_positions,
    sigmoid,
)
from src.visualization.explainability_visuals import (
    build_feature_impact_summary,
    generate_explainability_visual_package,
)


def build_synthetic_bundle() -> ExplainabilityBundle:
    """Create deterministic synthetic explanation data for renderer tests."""

    rng = np.random.default_rng(42)
    row_count = 180
    feature_count = len(FEATURE_COLUMNS)

    feature_frame = pd.DataFrame(
        {
            "total_events": rng.integers(1, 80, row_count),
            "view_count": rng.integers(1, 60, row_count),
            "addtocart_count": rng.integers(0, 12, row_count),
            "unique_items": rng.integers(1, 30, row_count),
            "activity_span_ms": rng.integers(
                1_000,
                5_000_000,
                row_count,
            ),
            "cart_to_view_ratio": rng.uniform(
                0,
                0.5,
                row_count,
            ),
            "events_per_unique_item": rng.uniform(
                1,
                8,
                row_count,
            ),
        }
    )

    shap_values = rng.normal(
        0,
        0.35,
        size=(row_count, feature_count),
    )
    shap_values[:, 0] += (
        feature_frame["total_events"].to_numpy()
        - feature_frame["total_events"].mean()
    ) / 75.0
    shap_values[:, 2] += (
        feature_frame["addtocart_count"].to_numpy()
        / 15.0
    )

    base_values = np.full(row_count, -1.2)
    probabilities = sigmoid(
        base_values + shap_values.sum(axis=1)
    )

    interaction_rows = 70
    interaction_values = rng.normal(
        0,
        0.04,
        size=(interaction_rows, feature_count, feature_count),
    )

    # Make the interaction tensor symmetric and add clear stable patterns.
    interaction_values = (
        interaction_values
        + np.transpose(
            interaction_values,
            (0, 2, 1),
        )
    ) / 2.0
    interaction_values[:, 0, 2] += 0.18
    interaction_values[:, 2, 0] += 0.18

    representatives = (
        select_representative_source_positions(
            np.asarray(probabilities)
        )
    )

    return ExplainabilityBundle(
        feature_frame=feature_frame,
        visitor_ids=pd.Series(
            [f"V{index:04d}" for index in range(row_count)]
        ),
        source_positions=np.arange(row_count),
        probabilities=np.asarray(probabilities),
        shap_values=shap_values,
        base_values=base_values,
        interaction_values=interaction_values,
        interaction_sample_positions=np.arange(
            interaction_rows
        ),
        representative_positions=representatives,
        priority_position=int(
            np.argmax(probabilities)
        ),
        source_rows=78_998,
        sample_rows=row_count,
        model_name="Tuned XGBoost",
        missing_counts={},
    )


def test_sigmoid_returns_expected_probability() -> None:
    """Zero model margin should equal 50% probability."""

    assert sigmoid(0.0) == pytest.approx(0.5)


def test_prepare_feature_frame_preserves_contract() -> None:
    """Feature preparation should keep canonical order and IDs."""

    raw = pd.DataFrame(
        {
            "visitorid": [10, 20],
            **{
                feature: [1.0, 2.0]
                for feature in FEATURE_COLUMNS
            },
        }
    )

    features, visitor_ids, missing = prepare_feature_frame(raw)

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert visitor_ids.tolist() == ["10", "20"]
    assert missing == {}


def test_representative_selection_returns_three_positions() -> None:
    """Low, medium, and high examples should be deterministic."""

    probabilities = np.linspace(0.01, 0.99, 101)

    selected = select_representative_source_positions(
        probabilities
    )

    assert set(selected) == {
        "Low intent",
        "Medium intent",
        "High intent",
    }
    assert selected["Low intent"] < selected["Medium intent"]
    assert selected["Medium intent"] < selected["High intent"]


def test_feature_impact_summary_ranks_all_features() -> None:
    """Global summary should rank every canonical feature exactly once."""

    summary = build_feature_impact_summary(
        build_synthetic_bundle()
    )

    assert len(summary) == len(FEATURE_COLUMNS)
    assert summary["feature"].is_unique
    assert summary["impact_rank"].tolist() == list(
        range(1, len(FEATURE_COLUMNS) + 1)
    )


def test_generate_explainability_visual_package(
    tmp_path: Path,
) -> None:
    """The complete synthetic package should create six clean visuals."""

    manifest = generate_explainability_visual_package(
        project_root=tmp_path,
        bundle=build_synthetic_bundle(),
    )

    assert set(manifest["qa"]) == {
        "MLV-E01",
        "MLV-E02",
        "MLV-E03",
        "MLV-E04",
        "MLV-E05",
        "MLV-E06",
    }

    for qa_result in manifest["qa"].values():
        assert qa_result["passed"] is True
        assert qa_result["width_px"] >= 1_600
        assert qa_result["height_px"] >= 850

    for relative_path in manifest["artifacts"].values():
        output_file = tmp_path / relative_path
        assert output_file.exists()
        assert output_file.stat().st_size > 10_000

    support = manifest["supporting_files"]

    for key in (
        "feature_impact",
        "representatives",
        "interaction_matrix",
        "insights",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    insight_text = (
        tmp_path / support["insights"]
    ).read_text(encoding="utf-8")

    assert "do not prove causality" in insight_text
    assert "does not contain matured production outcomes" in insight_text
