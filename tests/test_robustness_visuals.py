"""Tests for MLV-F01 to MLV-F05 robustness visuals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualization.robustness_data import (
    RobustnessBundle,
    build_sensitivity_pairs,
    build_stability_summary,
    prepare_sensitivity,
    prepare_stability,
)
from src.visualization.robustness_visuals import (
    generate_robustness_visual_package,
)


def build_stability() -> pd.DataFrame:
    """Create three-model, three-seed stability evidence."""

    rows = []

    values = {
        "Tuned XGBoost": [
            (11, 0.081, 0.794),
            (22, 0.122, 0.841),
            (33, 0.091, 0.802),
        ],
        "Tuned Random Forest": [
            (11, 0.080, 0.777),
            (22, 0.110, 0.810),
            (33, 0.088, 0.790),
        ],
        "XGBoost Baseline": [
            (11, 0.050, 0.765),
            (22, 0.067, 0.748),
            (33, 0.061, 0.723),
        ],
    }

    for model_name, model_values in values.items():
        for seed, pr_auc, roc_auc in model_values:
            rows.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "test_rows": 50_000,
                    "positive_rate": 0.00152,
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                }
            )

    return pd.DataFrame(rows)


def build_sensitivity() -> pd.DataFrame:
    """Create paired anomaly-sensitivity evidence."""

    rows = []

    values = {
        "Tuned XGBoost": (
            0.063,
            0.803,
            0.041,
            0.774,
        ),
        "Tuned Random Forest": (
            0.065,
            0.747,
            0.043,
            0.731,
        ),
        "XGBoost Baseline": (
            0.062,
            0.754,
            0.040,
            0.720,
        ),
    }

    for model_name, (
        all_pr,
        all_roc,
        non_pr,
        non_roc,
    ) in values.items():
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "evaluation_group": "all_validation_visitors",
                    "validation_rows": 283_412,
                    "pr_auc": all_pr,
                    "roc_auc": all_roc,
                    "note": "Full validation.",
                },
                {
                    "model_name": model_name,
                    "evaluation_group": "non_anomalous_validation_visitors",
                    "validation_rows": 282_471,
                    "pr_auc": non_pr,
                    "roc_auc": non_roc,
                    "note": "Anomalies removed.",
                },
            ]
        )

    return pd.DataFrame(rows)


def build_bundle() -> RobustnessBundle:
    """Create a complete synthetic robustness bundle."""

    stability = prepare_stability(build_stability())
    summary = build_stability_summary(stability)
    sensitivity = prepare_sensitivity(build_sensitivity())
    pairs = build_sensitivity_pairs(sensitivity)

    return RobustnessBundle(
        stability=stability,
        stability_summary=summary,
        sensitivity=sensitivity,
        sensitivity_pairs=pairs,
        champion_name="Tuned XGBoost",
        challenger_name="Tuned Random Forest",
        seed_count=3,
    )


def test_stability_summary_contains_all_models() -> None:
    """Summary should contain one row per model."""

    summary = build_stability_summary(
        prepare_stability(build_stability())
    )

    assert len(summary) == 3
    assert summary["model_name"].is_unique
    assert set(summary["seed_count"]) == {3}


def test_sensitivity_pairs_calculate_deltas() -> None:
    """Paired rows should calculate full-to-non-anomalous change."""

    pairs = build_sensitivity_pairs(
        prepare_sensitivity(build_sensitivity())
    )

    champion = pairs.loc[
        pairs["model_name"] == "Tuned XGBoost"
    ].iloc[0]

    assert champion["removed_rows"] == 941
    assert champion["pr_auc_delta"] == pytest.approx(-0.022)
    assert champion["roc_auc_delta"] == pytest.approx(-0.029)


def test_prepare_stability_rejects_duplicate_seed() -> None:
    """Duplicate model/seed rows must fail before charting."""

    frame = build_stability()
    broken = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        prepare_stability(broken)


def test_generate_robustness_visual_package(
    tmp_path: Path,
) -> None:
    """The complete package should create five clean visuals."""

    manifest = generate_robustness_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    assert set(manifest["qa"]) == {
        "MLV-F01",
        "MLV-F02",
        "MLV-F03",
        "MLV-F04",
        "MLV-F05",
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
        "stability_summary",
        "sensitivity_pairs",
        "insights",
        "manifest",
    ):
        assert (tmp_path / support[key]).exists()

    insight_text = (
        tmp_path / support["insights"]
    ).read_text(encoding="utf-8")

    assert "three runs per model" in insight_text
    assert "anomaly-flagged rows" in insight_text



def test_corrected_robustness_layouts_render_cleanly(tmp_path: Path) -> None:
    """F02-F04 should pass layout QA after collision fixes."""

    bundle = build_bundle()
    manifest = generate_robustness_visual_package(
        project_root=tmp_path,
        bundle=bundle,
    )
    for visual_id in ("MLV-F02", "MLV-F03", "MLV-F04"):
        assert manifest["qa"][visual_id]["passed"] is True


def test_corrected_f03_and_f04_render_without_footer_or_label_collisions(
    tmp_path: Path,
) -> None:
    """Corrected stability and sensitivity layouts must pass visual QA."""

    manifest = generate_robustness_visual_package(
        project_root=tmp_path,
        bundle=build_bundle(),
    )

    for visual_id in ("MLV-F03", "MLV-F04"):
        result = manifest["qa"][visual_id]
        assert result["passed"] is True
        assert result["width_px"] >= 1_600
        assert result["height_px"] >= 850
