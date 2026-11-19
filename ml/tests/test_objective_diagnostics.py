"""Tests for the objective-feasibility diagnostics.

Only two properties are load-bearing here, and both are about *time* rather
than about metrics:

* `bans_before_stage` must return exactly the bans that had already happened
  at a given pick stage, because the whole point of the timing study is that
  the harness convention (all bans at every stage) shows the model bans that
  had not been made yet;
* `build_timed_stage_frame` must honour that set instead of falling back to the
  full-draft ban flags, at zero picks as well as later.

They are pure-function tests: no dataset, no training, no ONNX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from training import ablation, objective_diagnostics as diag  # noqa: E402

pytest.importorskip("sklearn")


def _draft() -> dict:
    """Seven bans per side; picks at odd/even orders around them.

    Hero ids are `order + 100` so every assertion can be checked by eye.
    """
    rad_bans = [100, 102, 104, 110, 112, 114, 116]
    dire_bans = [101, 103, 105, 111, 113, 115, 117]
    rad_picks = [107, 109, 111, 113, 115]
    dire_picks = [108, 110, 112, 114, 116]
    return {
        "rad_picks": rad_picks,
        "dire_picks": dire_picks,
        "rad_bans": rad_bans,
        "dire_bans": dire_bans,
        "rad_pick_orders": [7, 9, 11, 13, 15],
        "dire_pick_orders": [8, 10, 12, 14, 16],
        "rad_ban_orders": [0, 2, 4, 10, 12, 14, 16],
        "dire_ban_orders": [1, 3, 5, 11, 13, 15, 17],
    }


def test_bans_before_first_pick_stops_at_the_first_pick():
    rad, dire = diag.bans_before_stage(_draft(), 0)
    # The first pick is at order 7, so bans at 0/2/4 (radiant) and 1/3/5 (dire)
    # are known; the bans in the second wave (10+) have not happened yet.
    assert rad == [100, 102, 104]
    assert dire == [101, 103, 105]


def test_bans_before_stage_uses_the_later_of_the_two_sides():
    draft = _draft()
    # k=2: radiant's second pick is order 9, dire's is 10, so the moment is 10.
    rad, dire = diag.bans_before_stage(draft, 2)
    assert rad == [100, 102, 104]
    assert dire == [101, 103, 105]
    # k=3: moment is max(11, 12) = 12, which admits the order-10 and -11 bans.
    rad, dire = diag.bans_before_stage(draft, 3)
    assert rad == [100, 102, 104, 110]
    assert dire == [101, 103, 105, 111]


def test_bans_before_stage_at_full_draft_keeps_every_earlier_ban():
    rad, dire = diag.bans_before_stage(_draft(), 5)
    # Full draft: moment is max(15, 16) = 16, so both order-16 entries are
    # excluded (strictly-before) and nothing earlier is lost.
    assert rad == [100, 102, 104, 110, 112, 114]
    assert dire == [101, 103, 105, 111, 113, 115]


def test_bans_before_stage_without_enough_picks_returns_nothing():
    draft = _draft()
    draft["rad_picks"] = draft["rad_picks"][:2]
    draft["rad_pick_orders"] = draft["rad_pick_orders"][:2]
    assert diag.bans_before_stage(draft, 4) == ([], [])


def _timing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [700],
            "start_time": [1_760_000_000],
            "radiant_win": [1],
            "duration": [2100],
            "avg_mmr": [0],
            "game_mode": [2],
            "lobby_type": [1],
            "patch": [0.0],
        }
    )


def test_timed_stage_frame_drops_bans_that_had_not_happened():
    """At zero picks, a hero banned in the second wave must not be flagged."""
    processor = ablation._load_processor(ML_ROOT / "datasets")
    schema = {
        "feature_names": ["duration", "rad_pick_0"],
        "categoricals": {},
        "defaults": {"duration": 0.0, "rad_pick_0": 0.0},
    }
    tokens = processor._safe_names
    # Hero 2 (axe) is banned first; hero 14 (pudge) is banned later.
    draft = {
        "rad_picks": [1, 11, 12, 13, 2],
        "dire_picks": [21, 22, 23, 25, 26],
        "rad_bans": [2, 14],
        "dire_bans": [3, 16],
        "rad_pick_orders": [7, 9, 11, 13, 15],
        "dire_pick_orders": [8, 10, 12, 14, 16],
        "rad_ban_orders": [0, 12],
        "dire_ban_orders": [1, 13],
    }
    out = diag.build_timed_stage_frame(
        _timing_frame(), {700: draft}, 0, schema=schema, processor=processor
    )
    assert len(out) == 1
    early_ban = f"rad_ban_{tokens[2]}"
    late_ban = f"rad_ban_{tokens[14]}"
    assert out[early_ban].iat[0] == 1.0
    assert out[late_ban].iat[0] == 0.0
    # Zero picks: no presence flag is set, and the schema defaults fill the rest.
    assert (out[[c for c in out.columns if c.startswith("rad_has_")]].sum(axis=1) == 0).all()
    assert out["duration"].iat[0] == 0.0


def test_timed_stage_frame_reports_only_rows_with_a_raw_draft():
    processor = ablation._load_processor(ML_ROOT / "datasets")
    schema = {"feature_names": ["duration"], "categoricals": {}, "defaults": {"duration": 0.0}}
    frame = pd.concat([_timing_frame(), _timing_frame().assign(match_id=701)], ignore_index=True)
    out = diag.build_timed_stage_frame(
        frame, {700: _draft()}, 0, schema=schema, processor=processor
    )
    assert out["match_id"].tolist() == [700]


def test_stage_timing_never_adds_bans_beyond_the_full_draft():
    """The timed set must be a subset of the full-draft bans, per side."""
    draft = _draft()
    for level in range(6):
        rad, dire = diag.bans_before_stage(draft, level)
        assert set(rad).issubset(set(draft["rad_bans"]))
        assert set(dire).issubset(set(draft["dire_bans"]))
        assert len(rad) <= 7 and len(dire) <= 7
        assert np.array_equal(sorted(rad), rad)


def test_hero_effect_precision_returns_noise_ratio():
    frame = pd.DataFrame(
        {
            "rad_has_a": [1.0, 0.0] * 40,
            "rad_has_b": [0.0, 1.0] * 40,
        }
    )
    labels = np.array([1, 0] * 40)
    result = diag.hero_effect_precision(frame, labels)
    # Only 40 picks per hero, so nothing clears the 50-pick floor.
    assert result == {"available": False}
