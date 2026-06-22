import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.model_config import (
    FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from src.models.model_evaluation import (
    train_and_evaluate_model,
)


def test_legacy_benchmark_uses_validation_not_holdout():
    rows = []

    for split_name, start in [
        ("train", 0),
        ("validation", 100),
        ("final_holdout", 200),
    ]:
        for index in range(12):
            value = start + index

            rows.append(
                {
                    "total_events": value + 2,
                    "view_count": value + 1,
                    "addtocart_count": index % 3,
                    "unique_items": index + 1,
                    "activity_span_ms": value + 100,
                    "cart_to_view_ratio": (
                        index % 3
                    ) / (value + 2),
                    "events_per_unique_item": (
                        value + 2
                    ) / (index + 2),
                    TARGET_COLUMN: index % 2,
                    SPLIT_COLUMN: split_name,
                }
            )

    data = pd.DataFrame(rows)

    config = {
        "model_name": "Test Logistic",
        "track": "test_track",
        "model_family": "linear",
        "reason": "split regression test",
        "uses_sample_weight": False,
        "max_rows": 1000,
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                    ),
                ),
            ]
        ),
    }

    _, result, thresholds = train_and_evaluate_model(
        model_config=config,
        full_training_data=data,
    )

    assert result["evaluation_split"] == "validation"
    assert result["training_rows"] == 12
    assert result["validation_rows"] == 12
    assert set(
        thresholds["evaluation_split"]
    ) == {"validation"}
