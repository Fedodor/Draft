"""Tests for chronological splitting and dataset fingerprinting."""

from __future__ import annotations

import pandas as pd
import pytest

from training.splits import (
    frame_fingerprint,
    resolve_time_column,
    schema_fingerprint,
    time_split,
)


def _frame(n: int = 100) -> pd.DataFrame:
    # Deliberately shuffle match_id so it cannot stand in for start_time.
    return pd.DataFrame(
        {
            "match_id": list(range(n))[::-1],
            "start_time": [1_700_000_000 + i * 60 for i in range(n)],
            "radiant_win": [i % 2 for i in range(n)],
            "feature": [float(i) for i in range(n)],
        }
    )


def test_prefers_start_time():
    column, used_start_time = resolve_time_column(_frame())
    assert column == "start_time"
    assert used_start_time is True


def test_falls_back_to_match_id():
    column, used_start_time = resolve_time_column(_frame().drop(columns=["start_time"]))
    assert column == "match_id"
    assert used_start_time is False


def test_partitions_are_chronological_and_disjoint():
    result = time_split(_frame(100), 0.6, 0.2, 0.2)

    assert len(result.train) == 60
    assert len(result.val) == 20
    assert len(result.test) == 20
    assert result.used_start_time is True

    # Every training timestamp precedes every validation timestamp, and so on.
    assert result.train["start_time"].max() <= result.val["start_time"].min()
    assert result.val["start_time"].max() <= result.test["start_time"].min()

    # No match appears in two partitions.
    ids = pd.concat([result.train, result.val, result.test])["match_id"]
    assert ids.is_unique


def test_split_is_deterministic_across_input_order():
    frame = _frame(60)
    first = time_split(frame, 0.5, 0.25, 0.25)
    second = time_split(frame.sample(frac=1.0, random_state=7), 0.5, 0.25, 0.25)
    assert list(first.train["match_id"]) == list(second.train["match_id"])
    assert list(first.test["match_id"]) == list(second.test["match_id"])


def test_fractions_must_sum_to_one():
    with pytest.raises(SystemExit):
        time_split(_frame(30), 0.6, 0.2, 0.3)


def test_schema_fingerprint_tracks_order_and_names():
    base = ["a", "b", "c"]
    assert schema_fingerprint(base) == schema_fingerprint(list(base))
    assert schema_fingerprint(base) != schema_fingerprint(["c", "b", "a"])
    assert schema_fingerprint(base) != schema_fingerprint(["a", "b"])


def test_frame_fingerprint_changes_with_data():
    frame = _frame(20)
    first = frame_fingerprint(frame, ["feature"])
    frame.loc[0, "feature"] = 999.0
    assert frame_fingerprint(frame, ["feature"]) != first
