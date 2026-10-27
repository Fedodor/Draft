import json
from pathlib import Path

import pandas as pd

from training.export_scoring_seed import build_meta_rows

ROOT = Path(__file__).resolve().parents[2]


def test_build_meta_rows_covers_roster_and_normalizes_scores():
    winrates = pd.DataFrame(
        [
            {"hero_id": 1, "games": 1000, "wins": 560, "winrate": 0.56},
            {"hero_id": 2, "games": 1000, "wins": 440, "winrate": 0.44},
            {"hero_id": 3, "games": 10, "wins": 9, "winrate": 0.9},
        ]
    )

    rows = build_meta_rows(
        winrates,
        hero_ids=[1, 2, 3, 4],
        pick_counts={1: 100, 2: 200, 4: 50},
        total_matches=1000,
    )
    by_id = {row["hero_id"]: row for row in rows}

    assert set(by_id) == {1, 2, 3, 4}
    assert all(0.0 <= row["meta_score"] <= 1.0 for row in rows)
    assert all(0.0 <= row["winrate"] <= 1.0 for row in rows)
    assert all(0.0 <= row["pickrate"] <= 1.0 for row in rows)

    assert by_id[1]["meta_score"] > by_id[2]["meta_score"]
    assert by_id[4]["pickrate"] == 0.05
    assert by_id[3]["pickrate"] == 0.0
    # A hero missing from the winrate source stays neutral instead of vanishing.
    assert by_id[4]["winrate"] == 0.5
    # The 10-game 90% hero is shrunk toward 0.5, so it must not rank first.
    assert by_id[3]["meta_score"] < by_id[1]["meta_score"]


def test_build_meta_rows_handles_empty_and_single_hero_inputs():
    assert build_meta_rows(pd.DataFrame(), [], {}, 0) == []

    single = pd.DataFrame(
        [{"hero_id": 7, "games": 100, "wins": 60, "winrate": 0.6}]
    )
    rows = build_meta_rows(single, [7], {7: 10}, 100)
    assert len(rows) == 1
    assert rows[0]["meta_score"] == 0.5


def test_committed_seed_has_meta_for_every_hero():
    seed = json.loads(
        (ROOT / "infrastructure" / "seeders" / "scoring_seed.json").read_text(
            encoding="utf-8"
        )
    )
    heroes = json.loads(
        (ROOT / "infrastructure" / "seeders" / "heroes.json").read_text(
            encoding="utf-8"
        )
    )

    meta_ids = {row["hero_id"] for row in seed["meta"]}
    assert meta_ids == {int(hero["id"]) for hero in heroes}
    assert len(seed["meta"]) == len(heroes)
    assert seed["patch_version"] not in (None, "", "unknown")
