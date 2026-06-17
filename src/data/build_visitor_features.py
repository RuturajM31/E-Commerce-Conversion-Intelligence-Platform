# build_visitor_features.py
# This script converts raw RetailRocket event-level data into visitor-level ML data.
#
# Raw grain:
#   One row = one visitor action on one item at one time
#
# Output grain:
#   One row = one visitor
#
# Business goal:
#   Predict whether a visitor is likely to convert.

import pandas as pd

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


# Input file: raw RetailRocket events data

EVENTS_FILE = RAW_DATA_DIR / "events.csv"

# Output file: clean visitor-level feature table for ML

OUTPUT_FILE = PROCESSED_DATA_DIR / "visitor_features.csv"


def load_events():
    
    """Load only the columns we need from events.csv."""

    # We load only required columns to keep memory usage lower.
    
    columns_needed = ["timestamp", "visitorid", "event", "itemid", "transactionid"]

    events = pd.read_csv(EVENTS_FILE, usecols=columns_needed)

    return events


def create_target(events):
    
    """Create visitor-level target: converted or not converted."""

    # A visitor is converted if they have at least one transaction event.
    
    target = (
        events.assign(is_transaction=events["event"].eq("transaction").astype(int))
        .groupby("visitorid", as_index=False)["is_transaction"]
        .max()
        .rename(columns={"is_transaction": "converted"})
    )

    return target


def create_behavior_features(events):
    
    """Create visitor-level behaviour features from non-transaction events."""

    # We exclude transaction rows from input features to avoid target leakage.
    
    behavior_events = events[events["event"] != "transaction"].copy()

    # Create simple event indicator columns.
    
    behavior_events["is_view"] = behavior_events["event"].eq("view").astype(int)
    behavior_events["is_addtocart"] = behavior_events["event"].eq("addtocart").astype(int)

    # Aggregate many event rows into one row per visitor.
    
    features = (
        behavior_events.groupby("visitorid")
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

    # Activity span shows how long the visitor was active in the dataset.
    
    features["activity_span_ms"] = features["last_event_time"] - features["first_event_time"]

    # Cart-to-view ratio shows buying interest.
    # We add 1 to avoid division by zero.
    
    features["cart_to_view_ratio"] = features["addtocart_count"] / (features["view_count"] + 1)

    # Events per item shows repeat interest in the same products.
    features["events_per_unique_item"] = features["total_events"] / (features["unique_items"] + 1)

    return features


def build_visitor_dataset():
    
    """Build and save the final visitor-level ML dataset."""

    # Load raw events.
    
    events = load_events()

    # Create target table: one row per visitor.
    
    target = create_target(events)

    # Create feature table: one row per visitor.
    
    features = create_behavior_features(events)

    # Join features with target using visitorid.
    
    visitor_data = features.merge(target, on="visitorid", how="left")

    # If a visitor has no transaction, converted should be 0.
    visitor_data["converted"] = visitor_data["converted"].fillna(0).astype(int)

    # Save the final ML-ready dataset.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    visitor_data.to_csv(OUTPUT_FILE, index=False)

    # Print validation checks.
    print("Visitor feature dataset created.")
    print("Rows:", visitor_data.shape[0])
    print("Columns:", visitor_data.shape[1])
    print("Duplicate visitor IDs:", visitor_data["visitorid"].duplicated().sum())
    print("Conversion rate:", round(visitor_data["converted"].mean() * 100, 2), "%")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_visitor_dataset()