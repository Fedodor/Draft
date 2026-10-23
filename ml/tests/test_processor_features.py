"""Processor feature-contract tests.

The processor is the only place raw match JSON becomes model features, so the
chronological split key (`start_time`) must be populated here. These tests pin
the field mapping for both provider shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FEATURE_ENGINEERING = Path(__file__).resolve().parents[1] / "feature-engineering"
if str(FEATURE_ENGINEERING) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENGINEERING))

import processor  # noqa: E402

pytest.importorskip("pandas")


def _match(**overrides) -> dict:
    picks_bans = [
        {"hero_id": hero_id, "team": 0, "is_pick": True, "order": order}
        for order, hero_id in enumerate([1, 2, 3, 4, 5])
    ]
    picks_bans += [
        {"hero_id": hero_id, "team": 1, "is_pick": True, "order": 10 + order}
        for order, hero_id in enumerate([6, 7, 8, 9, 10])
    ]
    match = {
        "match_id": 123,
        "picks_bans": picks_bans,
        "radiant_win": True,
    }
    match.update(overrides)
    return match


@pytest.fixture()
def feature_processor(tmp_path):
    return processor.FeatureProcessor(
        raw_dir=tmp_path,
        output_dir=tmp_path,
    )


def test_start_time_from_opendota_field(feature_processor):
    features = feature_processor.process_match(_match(start_time=1_700_000_000))

    assert features is not None
    assert features["start_time"] == 1_700_000_000


def test_start_time_from_stratz_field(feature_processor):
    features = feature_processor.process_match(_match(startDateTime=1_494_167_425))

    assert features is not None
    assert features["start_time"] == 1_494_167_425


def test_start_time_defaults_to_zero_when_absent(feature_processor):
    features = feature_processor.process_match(_match())

    assert features is not None
    assert features["start_time"] == 0
