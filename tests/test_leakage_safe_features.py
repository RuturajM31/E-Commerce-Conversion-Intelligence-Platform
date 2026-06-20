import pandas as pd

from src.data.build_visitor_features import (
    build_one_snapshot,
    create_snapshot_schedule,
)


def make_event(timestamp, visitorid, event, itemid):
    return {
        "timestamp": int(pd.Timestamp(timestamp).timestamp() * 1000),
        "visitorid": visitorid,
        "event": event,
        "itemid": itemid,
        "transactionid": None,
        "event_time": pd.Timestamp(timestamp, tz="UTC"),
    }


def test_future_behavior_is_not_used_as_feature():
    events = pd.DataFrame(
        [
            make_event("2015-06-01", 1, "view", 101),
            make_event("2015-06-02", 1, "addtocart", 101),
            make_event("2015-06-11", 1, "view", 102),
            make_event("2015-06-12", 1, "transaction", 101),
        ]
    )

    snapshot_row = pd.Series(
        {
            "snapshot_time": pd.Timestamp(
                "2015-06-10",
                tz="UTC",
            ),
            "observation_start": pd.Timestamp(
                "2015-05-11",
                tz="UTC",
            ),
            "target_end": pd.Timestamp(
                "2015-06-24",
                tz="UTC",
            ),
            "data_split": "train",
        }
    )

    result = build_one_snapshot(events, snapshot_row)

    assert result.loc[0, "total_events"] == 2
    assert result.loc[0, "converted"] == 1


def test_purchase_after_target_window_is_not_positive():
    events = pd.DataFrame(
        [
            make_event("2015-06-01", 1, "view", 101),
            make_event("2015-06-02", 1, "view", 102),
            make_event("2015-07-01", 1, "transaction", 101),
        ]
    )

    snapshot_row = pd.Series(
        {
            "snapshot_time": pd.Timestamp(
                "2015-06-10",
                tz="UTC",
            ),
            "observation_start": pd.Timestamp(
                "2015-05-11",
                tz="UTC",
            ),
            "target_end": pd.Timestamp(
                "2015-06-24",
                tz="UTC",
            ),
            "data_split": "validation",
        }
    )

    result = build_one_snapshot(events, snapshot_row)

    assert result.loc[0, "converted"] == 0


def test_schedule_contains_purge_gaps():
    dates = pd.date_range(
        "2015-05-01",
        periods=140,
        freq="D",
        tz="UTC",
    )

    events = pd.DataFrame(
        {
            "event_time": dates,
        }
    )

    schedule = create_snapshot_schedule(events)

    assert "purge" in schedule["data_split"].values
    assert "train" in schedule["data_split"].values
    assert "validation" in schedule["data_split"].values
    assert "final_holdout" in schedule["data_split"].values

    split_order = schedule["data_split"].tolist()

    assert split_order.index("train") < split_order.index("validation")
    assert split_order.index("validation") < split_order.index(
        "final_holdout"
    )
