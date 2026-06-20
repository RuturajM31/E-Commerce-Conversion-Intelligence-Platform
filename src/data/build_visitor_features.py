# build_visitor_features.py
# Build leakage-safe rolling visitor snapshots from RetailRocket events.
#
# Raw grain:
#   One row = one visitor action on one item at one time
#
# Training grain:
#   One row = one visitor at one historical scoring snapshot
#
# Business question:
#   Based only on behaviour from the previous 30 days,
#   will this active visitor purchase during the next 14 days?

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.feature_engineering import add_derived_features


EVENTS_FILE = RAW_DATA_DIR / "events.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"

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

    events = events.sort_values("event_time").reset_index(drop=True)

    return events


def create_snapshot_schedule(events: pd.DataFrame) -> pd.DataFrame:
    """Create weekly historical scoring dates with chronological splits."""

    dataset_start = events["event_time"].min()
    dataset_end = events["event_time"].max()

    first_snapshot = (
        dataset_start + pd.Timedelta(days=OBSERVATION_DAYS)
    ).ceil("D")

    last_snapshot = (
        dataset_end - pd.Timedelta(days=TARGET_DAYS)
    ).floor("D")

    snapshot_dates = pd.date_range(
        start=first_snapshot,
        end=last_snapshot,
        freq=f"{SNAPSHOT_STEP_DAYS}D",
    )

    snapshot_count = len(snapshot_dates)

    if snapshot_count < 10:
        raise ValueError(
            "Not enough historical snapshots for chronological splitting."
        )

    # A purge gap prevents a training target window from crossing
    # into the next evaluation period.
    purge_count = max(
        1,
        math.ceil(TARGET_DAYS / SNAPSHOT_STEP_DAYS) - 1,
    )

    validation_count = max(3, round(snapshot_count * 0.20))
    holdout_count = max(3, round(snapshot_count * 0.20))

    train_count = (
        snapshot_count
        - validation_count
        - holdout_count
        - (2 * purge_count)
    )

    if train_count < 3:
        raise ValueError(
            "Not enough snapshots remain for model training."
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


def build_one_snapshot(
    events: pd.DataFrame,
    snapshot_row: pd.Series,
) -> pd.DataFrame:
    """Build one leakage-safe visitor snapshot."""

    snapshot_time = snapshot_row["snapshot_time"]
    observation_start = snapshot_row["observation_start"]
    target_end = snapshot_row["target_end"]
    data_split = snapshot_row["data_split"]

    # Features use only behaviour known at snapshot time.
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

    observation_events["is_view"] = (
        observation_events["event"].eq("view").astype(int)
    )

    observation_events["is_addtocart"] = (
        observation_events["event"].eq("addtocart").astype(int)
    )

    features = (
        observation_events.groupby("visitorid")
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

    # Remove one-click visitors because their behaviour signal is too weak.
    features = features.loc[
        features["total_events"] >= MINIMUM_EVENTS
    ].copy()

    features["activity_span_ms"] = (
        features["last_event_time"]
        - features["first_event_time"]
    )

    features = add_derived_features(features)

    # Target uses only transactions after the scoring timestamp.
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
        features["visitorid"].isin(future_buyers).astype(int)
    )

    features["snapshot_time"] = snapshot_time.isoformat()
    features["observation_start"] = observation_start.isoformat()
    features["target_end"] = target_end.isoformat()
    features["data_split"] = data_split

    return features


def validate_snapshot_dataset(visitor_data: pd.DataFrame) -> None:
    """Run important leakage and data-quality checks."""

    required_splits = {
        "train",
        "validation",
        "final_holdout",
    }

    actual_splits = set(visitor_data["data_split"].unique())

    if actual_splits != required_splits:
        raise ValueError(
            f"Unexpected data splits: {sorted(actual_splits)}"
        )

    duplicate_rows = visitor_data.duplicated(
        subset=["visitorid", "snapshot_time"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            "Duplicate visitor and snapshot combinations found."
        )

    if visitor_data["converted"].isna().any():
        raise ValueError("Target contains missing values.")

    if not visitor_data["converted"].isin([0, 1]).all():
        raise ValueError("Target must contain only zero and one.")

    for split_name in sorted(required_splits):
        split_data = visitor_data.loc[
            visitor_data["data_split"] == split_name
        ]

        if split_data["converted"].sum() == 0:
            raise ValueError(
                f"{split_name} contains no positive examples."
            )


def build_snapshot_summary(
    visitor_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create evidence for every retained historical snapshot."""

    summary = (
        visitor_data.groupby(
            ["data_split", "snapshot_time"],
            as_index=False,
        )
        .agg(
            rows=("visitorid", "size"),
            unique_visitors=("visitorid", "nunique"),
            positive_rows=("converted", "sum"),
            positive_rate=("converted", "mean"),
        )
        .sort_values("snapshot_time")
    )

    return summary


def save_snapshot_metadata(
    events: pd.DataFrame,
    schedule: pd.DataFrame,
    visitor_data: pd.DataFrame,
) -> Dict:
    """Save the exact observation, target, and split design."""

    split_summary = (
        visitor_data.groupby("data_split")
        .agg(
            rows=("visitorid", "size"),
            unique_visitors=("visitorid", "nunique"),
            positive_rows=("converted", "sum"),
            positive_rate=("converted", "mean"),
            first_snapshot=("snapshot_time", "min"),
            last_snapshot=("snapshot_time", "max"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    metadata = {
        "dataset_start_utc": (
            events["event_time"].min().isoformat()
        ),
        "dataset_end_utc": (
            events["event_time"].max().isoformat()
        ),
        "observation_days": OBSERVATION_DAYS,
        "target_days": TARGET_DAYS,
        "snapshot_step_days": SNAPSHOT_STEP_DAYS,
        "minimum_behavior_events": MINIMUM_EVENTS,
        "target_definition": (
            "At least one transaction after snapshot_time and "
            "on or before target_end."
        ),
        "feature_definition": (
            "Non-transaction events after observation_start and "
            "on or before snapshot_time."
        ),
        "purged_snapshot_count": int(
            (schedule["data_split"] == "purge").sum()
        ),
        "retained_rows": int(len(visitor_data)),
        "split_summary": split_summary,
    }

    with open(
        SNAPSHOT_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    return metadata


def build_visitor_dataset() -> pd.DataFrame:
    """Build and save the leakage-safe rolling snapshot dataset."""

    events = load_events()
    schedule = create_snapshot_schedule(events)

    snapshot_tables: List[pd.DataFrame] = []

    for _, snapshot_row in schedule.iterrows():
        split_name = snapshot_row["data_split"]

        print(
            "Building snapshot: "
            f"{snapshot_row['snapshot_time']} | {split_name}"
        )

        # Purged dates exist only to protect chronological boundaries.
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

    summary = build_snapshot_summary(visitor_data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    visitor_data.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SNAPSHOT_SUMMARY_FILE, index=False)

    metadata = save_snapshot_metadata(
        events=events,
        schedule=schedule,
        visitor_data=visitor_data,
    )

    print("\nLeakage-safe visitor snapshot dataset created.")
    print(f"Rows: {len(visitor_data):,}")
    print(
        "Unique visitor-snapshot rows: "
        f"{visitor_data[['visitorid', 'snapshot_time']].drop_duplicates().shape[0]:,}"
    )

    print("\nSplit summary:")
    print(
        visitor_data.groupby("data_split")
        .agg(
            rows=("visitorid", "size"),
            positives=("converted", "sum"),
            positive_rate=("converted", "mean"),
        )
        .to_string()
    )

    print(f"\nSaved dataset: {OUTPUT_FILE}")
    print(f"Saved snapshot summary: {SNAPSHOT_SUMMARY_FILE}")
    print(f"Saved metadata: {SNAPSHOT_METADATA_FILE}")
    print(
        "Purged snapshots: "
        f"{metadata['purged_snapshot_count']}"
    )

    return visitor_data


if __name__ == "__main__":
    build_visitor_dataset()
