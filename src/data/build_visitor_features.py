# build_visitor_features.py
# Build leakage-safe rolling visitor snapshots from RetailRocket events.
#
# Raw grain:
#   One row = one visitor action on one item at one time
#
# Training grain:
#   One row = one visitor at one historical scoring snapshot
#
# Production scoring grain:
#   One row = one currently active visitor
#
# Business question:
#   Based only on behaviour from the previous 30 days,
#   will this active visitor purchase during the next 14 days?

from __future__ import annotations

import json
import math
from typing import Dict, List

import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.feature_engineering import add_derived_features


EVENTS_FILE = RAW_DATA_DIR / "events.csv"

TRAINING_OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "visitor_training_snapshots.csv"
)

SCORING_OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "visitor_features.csv"
)

SNAPSHOT_METADATA_FILE = (
    PROCESSED_DATA_DIR / "visitor_feature_snapshot_metadata.json"
)

SNAPSHOT_SUMMARY_FILE = (
    PROCESSED_DATA_DIR / "visitor_feature_snapshot_summary.csv"
)

OBSERVATION_DAYS = 30
TARGET_DAYS = 14
SNAPSHOT_STEP_DAYS = 7
MINIMUM_EVENTS = 2


def load_events() -> pd.DataFrame:
    """Load required event columns and create a UTC event timestamp."""

    columns_needed = [
        "timestamp",
        "visitorid",
        "event",
        "itemid",
        "transactionid",
    ]

    events = pd.read_csv(
        EVENTS_FILE,
        usecols=columns_needed,
    )

    events["event_time"] = pd.to_datetime(
        events["timestamp"],
        unit="ms",
        utc=True,
    )

    events = events.sort_values(
        "event_time"
    ).reset_index(drop=True)

    return events


def create_snapshot_schedule(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Create weekly snapshots with chronological purge gaps."""

    dataset_start = events["event_time"].min()
    dataset_end = events["event_time"].max()

    first_snapshot = (
        dataset_start
        + pd.Timedelta(days=OBSERVATION_DAYS)
    ).ceil("D")

    last_snapshot = (
        dataset_end
        - pd.Timedelta(days=TARGET_DAYS)
    ).floor("D")

    snapshot_dates = pd.date_range(
        start=first_snapshot,
        end=last_snapshot,
        freq=f"{SNAPSHOT_STEP_DAYS}D",
    )

    snapshot_count = len(snapshot_dates)

    if snapshot_count < 10:
        raise ValueError(
            "Not enough historical snapshots for "
            "chronological splitting."
        )

    # One purge snapshot prevents a target period from crossing
    # directly into the following evaluation split.
    purge_count = max(
        1,
        math.ceil(
            TARGET_DAYS / SNAPSHOT_STEP_DAYS
        ) - 1,
    )

    validation_count = max(
        3,
        round(snapshot_count * 0.20),
    )

    holdout_count = max(
        3,
        round(snapshot_count * 0.20),
    )

    train_count = (
        snapshot_count
        - validation_count
        - holdout_count
        - (2 * purge_count)
    )

    if train_count < 3:
        raise ValueError(
            "Not enough snapshots remain for training."
        )

    split_labels = (
        ["train"] * train_count
        + ["purge"] * purge_count
        + ["validation"] * validation_count
        + ["purge"] * purge_count
        + ["final_holdout"] * holdout_count
    )

    schedule = pd.DataFrame(
        {
            "snapshot_time": snapshot_dates,
            "data_split": split_labels,
        }
    )

    schedule["observation_start"] = (
        schedule["snapshot_time"]
        - pd.Timedelta(days=OBSERVATION_DAYS)
    )

    schedule["target_end"] = (
        schedule["snapshot_time"]
        + pd.Timedelta(days=TARGET_DAYS)
    )

    return schedule


def aggregate_behavior_features(
    observation_events: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate known behaviour into one row per visitor."""

    behavior = observation_events.copy()

    behavior["is_view"] = (
        behavior["event"]
        .eq("view")
        .astype(int)
    )

    behavior["is_addtocart"] = (
        behavior["event"]
        .eq("addtocart")
        .astype(int)
    )

    features = (
        behavior.groupby("visitorid")
        .agg(
            total_events=("event", "count"),
            view_count=("is_view", "sum"),
            addtocart_count=("is_addtocart", "sum"),
            unique_items=("itemid", "nunique"),
            first_event_time=("timestamp", "min"),
            last_event_time=("timestamp", "max"),
        )
        .reset_index()
    )

    features = features.loc[
        features["total_events"] >= MINIMUM_EVENTS
    ].copy()

    features["activity_span_ms"] = (
        features["last_event_time"]
        - features["first_event_time"]
    )

    features = add_derived_features(features)

    return features


def build_one_snapshot(
    events: pd.DataFrame,
    snapshot_row: pd.Series,
) -> pd.DataFrame:
    """Build one leakage-safe visitor training snapshot."""

    snapshot_time = snapshot_row["snapshot_time"]
    observation_start = snapshot_row["observation_start"]
    target_end = snapshot_row["target_end"]
    data_split = snapshot_row["data_split"]

    # Features contain only behaviour known at scoring time.
    observation_events = events.loc[
        (
            events["event_time"] > observation_start
        )
        & (
            events["event_time"] <= snapshot_time
        )
        & (
            events["event"] != "transaction"
        )
    ].copy()

    features = aggregate_behavior_features(
        observation_events
    )

    # Target contains only purchases after scoring time.
    future_buyers = set(
        events.loc[
            (
                events["event_time"] > snapshot_time
            )
            & (
                events["event_time"] <= target_end
            )
            & (
                events["event"] == "transaction"
            ),
            "visitorid",
        ].unique()
    )

    features["converted"] = (
        features["visitorid"]
        .isin(future_buyers)
        .astype(int)
    )

    features["snapshot_time"] = (
        snapshot_time.isoformat()
    )

    features["observation_start"] = (
        observation_start.isoformat()
    )

    features["target_end"] = (
        target_end.isoformat()
    )

    features["data_split"] = data_split

    return features


def build_latest_scoring_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Create the latest one-row-per-visitor scoring table."""

    scoring_time = events["event_time"].max()

    observation_start = (
        scoring_time
        - pd.Timedelta(days=OBSERVATION_DAYS)
    )

    observation_events = events.loc[
        (
            events["event_time"] > observation_start
        )
        & (
            events["event_time"] <= scoring_time
        )
        & (
            events["event"] != "transaction"
        )
    ].copy()

    scoring_features = aggregate_behavior_features(
        observation_events
    )

    scoring_features["scoring_time"] = (
        scoring_time.isoformat()
    )

    scoring_features["observation_start"] = (
        observation_start.isoformat()
    )

    duplicate_visitors = (
        scoring_features["visitorid"]
        .duplicated()
        .sum()
    )

    if duplicate_visitors:
        raise ValueError(
            "Production scoring features contain "
            "duplicate visitors."
        )

    return scoring_features


def validate_snapshot_dataset(
    visitor_data: pd.DataFrame,
) -> None:
    """Run leakage and data-quality checks."""

    required_splits = {
        "train",
        "validation",
        "final_holdout",
    }

    actual_splits = set(
        visitor_data["data_split"].unique()
    )

    if actual_splits != required_splits:
        raise ValueError(
            f"Unexpected data splits: "
            f"{sorted(actual_splits)}"
        )

    duplicate_rows = visitor_data.duplicated(
        subset=[
            "visitorid",
            "snapshot_time",
        ]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            "Duplicate visitor and snapshot "
            "combinations found."
        )

    if visitor_data["converted"].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if not visitor_data["converted"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Target must contain only zero and one."
        )

    for split_name in sorted(required_splits):
        split_data = visitor_data.loc[
            visitor_data["data_split"]
            == split_name
        ]

        if split_data["converted"].sum() == 0:
            raise ValueError(
                f"{split_name} contains no "
                "positive examples."
            )


def build_snapshot_summary(
    visitor_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create evidence for each retained snapshot."""

    summary = (
        visitor_data.groupby(
            [
                "data_split",
                "snapshot_time",
            ],
            as_index=False,
        )
        .agg(
            rows=("visitorid", "size"),
            unique_visitors=(
                "visitorid",
                "nunique",
            ),
            positive_rows=(
                "converted",
                "sum",
            ),
            positive_rate=(
                "converted",
                "mean",
            ),
        )
        .sort_values("snapshot_time")
    )

    return summary


def save_snapshot_metadata(
    events: pd.DataFrame,
    schedule: pd.DataFrame,
    visitor_data: pd.DataFrame,
    scoring_features: pd.DataFrame,
) -> Dict:
    """Save the exact observation and target design."""

    split_summary = (
        visitor_data.groupby("data_split")
        .agg(
            rows=("visitorid", "size"),
            unique_visitors=(
                "visitorid",
                "nunique",
            ),
            positive_rows=(
                "converted",
                "sum",
            ),
            positive_rate=(
                "converted",
                "mean",
            ),
            first_snapshot=(
                "snapshot_time",
                "min",
            ),
            last_snapshot=(
                "snapshot_time",
                "max",
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    metadata = {
        "dataset_start_utc": (
            events["event_time"]
            .min()
            .isoformat()
        ),
        "dataset_end_utc": (
            events["event_time"]
            .max()
            .isoformat()
        ),
        "observation_days": OBSERVATION_DAYS,
        "target_days": TARGET_DAYS,
        "snapshot_step_days": (
            SNAPSHOT_STEP_DAYS
        ),
        "minimum_behavior_events": (
            MINIMUM_EVENTS
        ),
        "target_definition": (
            "At least one transaction after "
            "snapshot_time and on or before "
            "target_end."
        ),
        "feature_definition": (
            "Non-transaction events after "
            "observation_start and on or before "
            "snapshot_time."
        ),
        "purged_snapshot_count": int(
            (
                schedule["data_split"]
                == "purge"
            ).sum()
        ),
        "retained_training_rows": int(
            len(visitor_data)
        ),
        "production_scoring_rows": int(
            len(scoring_features)
        ),
        "production_scoring_time_utc": (
            scoring_features[
                "scoring_time"
            ].iloc[0]
        ),
        "split_summary": split_summary,
    }

    with open(
        SNAPSHOT_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    return metadata


def build_visitor_dataset() -> pd.DataFrame:
    """Build training snapshots and production features."""

    events = load_events()
    schedule = create_snapshot_schedule(events)

    snapshot_tables: List[pd.DataFrame] = []

    for _, snapshot_row in schedule.iterrows():
        split_name = snapshot_row["data_split"]

        print(
            "Building snapshot: "
            f"{snapshot_row['snapshot_time']} "
            f"| {split_name}"
        )

        # Purged snapshots protect split boundaries.
        if split_name == "purge":
            continue

        snapshot_table = build_one_snapshot(
            events=events,
            snapshot_row=snapshot_row,
        )

        snapshot_tables.append(snapshot_table)

    visitor_data = pd.concat(
        snapshot_tables,
        ignore_index=True,
    )

    validate_snapshot_dataset(visitor_data)

    snapshot_summary = build_snapshot_summary(
        visitor_data
    )

    scoring_features = (
        build_latest_scoring_features(events)
    )

    TRAINING_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    visitor_data.to_csv(
        TRAINING_OUTPUT_FILE,
        index=False,
    )

    scoring_features.to_csv(
        SCORING_OUTPUT_FILE,
        index=False,
    )

    snapshot_summary.to_csv(
        SNAPSHOT_SUMMARY_FILE,
        index=False,
    )

    metadata = save_snapshot_metadata(
        events=events,
        schedule=schedule,
        visitor_data=visitor_data,
        scoring_features=scoring_features,
    )

    print(
        "\nLeakage-safe visitor datasets created."
    )

    print(
        f"Training rows: "
        f"{len(visitor_data):,}"
    )

    print(
        f"Production scoring rows: "
        f"{len(scoring_features):,}"
    )

    print("\nTraining split summary:")

    print(
        visitor_data.groupby("data_split")
        .agg(
            rows=("visitorid", "size"),
            positives=("converted", "sum"),
            positive_rate=(
                "converted",
                "mean",
            ),
        )
        .to_string()
    )

    print(
        "\nSaved training snapshots: "
        f"{TRAINING_OUTPUT_FILE}"
    )

    print(
        "Saved production scoring features: "
        f"{SCORING_OUTPUT_FILE}"
    )

    print(
        "Saved snapshot summary: "
        f"{SNAPSHOT_SUMMARY_FILE}"
    )

    print(
        "Saved metadata: "
        f"{SNAPSHOT_METADATA_FILE}"
    )

    print(
        "Purged snapshots: "
        f"{metadata['purged_snapshot_count']}"
    )

    return visitor_data


if __name__ == "__main__":
    build_visitor_dataset()
