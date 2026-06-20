import pandas as pd

from src.models.finalize_true_champion import create_split


def test_saved_chronological_splits_are_used_directly():
    X = pd.DataFrame(
        {
            "feature": range(12),
        }
    )

    y = pd.Series(
        [
            0, 1, 0, 1,
            0, 1, 0, 1,
            0, 1, 0, 1,
        ]
    )

    visitor_ids = pd.Series(
        range(100, 112)
    )

    split_labels = pd.Series(
        [
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "validation",
            "validation",
            "final_holdout",
            "final_holdout",
            "final_holdout",
            "final_holdout",
        ]
    )

    splits = create_split(
        X=X,
        y=y,
        visitor_ids=visitor_ids,
        split_labels=split_labels,
    )

    assert splits["train"]["X"]["feature"].tolist() == [
        0, 1, 2, 3
    ]

    assert splits["validation"]["X"]["feature"].tolist() == [
        4, 5, 6, 7
    ]

    assert splits["final_holdout"]["X"]["feature"].tolist() == [
        8, 9, 10, 11
    ]


def test_holdout_rows_never_enter_train_or_validation():
    X = pd.DataFrame(
        {
            "feature": range(6),
        }
    )

    y = pd.Series(
        [0, 1, 0, 1, 0, 1]
    )

    visitor_ids = pd.Series(
        range(6)
    )

    labels = pd.Series(
        [
            "train",
            "train",
            "validation",
            "validation",
            "final_holdout",
            "final_holdout",
        ]
    )

    splits = create_split(
        X,
        y,
        visitor_ids,
        labels,
    )

    train_ids = set(
        splits["train"]["visitor_ids"]
    )

    validation_ids = set(
        splits["validation"]["visitor_ids"]
    )

    holdout_ids = set(
        splits["final_holdout"]["visitor_ids"]
    )

    assert train_ids.isdisjoint(holdout_ids)
    assert validation_ids.isdisjoint(holdout_ids)
