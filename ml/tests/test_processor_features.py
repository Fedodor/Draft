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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (145.0, 145),
        ("155", 155),
        (" 7 ", 7),
        (0, None),
        (-3, None),
        (None, None),
        (float("nan"), None),
        ("", None),
        ("_unknown_hero", None),
    ],
)
def test_resolve_matchup_hero_id_numeric_shapes(value, expected):
    assert processor.resolve_matchup_hero_id(value, {}) == expected


def test_resolve_matchup_hero_id_accepts_legacy_names():
    name_to_id = {"axe": 2, "crystal_maiden": 5}
    assert processor.resolve_matchup_hero_id("Axe", name_to_id) == 2
    assert processor.resolve_matchup_hero_id("_crystal_maiden", name_to_id) == 5
    assert processor.resolve_matchup_hero_id("not_a_hero", name_to_id) is None


def test_load_hero_matchups_reads_numeric_writer_shape(feature_processor, monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [
            {"hero_a": 1, "hero_b": 2, "hero_a_winrate": 0.6, "games": 50},
            {"hero_a": 3.0, "hero_b": 4.0, "hero_a_winrate": 0.4, "games": 60},
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)
    (feature_processor.output_dir / "hero_matchups.parquet").write_bytes(b"")

    feature_processor.hero_matchups = {}
    feature_processor._load_hero_matchups()

    assert feature_processor.hero_matchups[1][2]["winrate"] == pytest.approx(0.6)
    assert feature_processor.hero_matchups[3][4]["winrate"] == pytest.approx(0.4)


def test_load_hero_matchups_reads_legacy_name_shape(feature_processor, monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"hero_a": "axe", "hero_b": "pudge", "winrate": 0.55, "games": 20}]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)
    (feature_processor.output_dir / "hero_matchups.parquet").write_bytes(b"")

    feature_processor.hero_matchups = {}
    feature_processor._load_hero_matchups()

    assert feature_processor.hero_matchups[2][14]["winrate"] == pytest.approx(0.55)
