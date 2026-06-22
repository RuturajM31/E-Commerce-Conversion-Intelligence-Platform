"""Tests preventing fabricated production outcome metrics."""

from pathlib import Path


APP_FILES = [
    Path("app/Executive_Overview.py"),
    Path("app/pages/1_Visitor_Intent_Predictor.py"),
    Path("app/pages/2_Batch_Scoring.py"),
    Path("app/pages/3_Model_Benchmark_Selection.py"),
]


def test_dashboard_does_not_use_hardcoded_conversion_fallback():
    """Dashboard lift must come from labeled evaluation evidence."""

    for path in APP_FILES:
        text = path.read_text(encoding="utf-8")

        assert "0.008269" not in text
        assert "final_true_champion_holdout.csv" in text


def test_production_features_are_not_treated_as_labeled_outcomes():
    """The unlabeled production feature table must not create buyer KPIs."""

    for path in APP_FILES:
        text = path.read_text(encoding="utf-8")

        assert 'visitor_features["converted"]' not in text


def test_monitoring_exposes_outcome_availability():
    """Monitoring must distinguish unavailable labels from real zeroes."""

    text = Path(
        "src/monitoring/build_monitoring_snapshot.py"
    ).read_text(encoding="utf-8")

    assert "ecommerce_outcome_labels_available" in text
    assert "if outcome_labels_available:" in text
