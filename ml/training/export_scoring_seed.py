#!/usr/bin/env python3
"""Export real matchup and same-team synergy rows for the desktop seeder.

Sources:
  - ml/datasets/processed/hero_matchups.parquet
  - ml/datasets/processed/draft_features.parquet
  - ml/datasets/processed/hero_winrates.parquet
  - infrastructure/seeders/heroes.json

Synergy is the observed same-team pair winrate minus the average individual
hero winrate, clamped to [-0.5, 0.5]. Only pairs with enough games are emitted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "datasets" / "processed"
OUTPUT = ROOT / "infrastructure" / "seeders" / "scoring_seed.json"
HEROES = ROOT / "infrastructure" / "seeders" / "heroes.json"
SCHEMA = ROOT / "ml" / "models" / "draft_model_xgboost_full_schema.json"
MIN_MATCHUP_GAMES = 20
MIN_SYNERGY_GAMES = 50


def active_patch() -> str:
    """Latest real patch known to the training schema, so the seed matches the model."""
    if SCHEMA.exists():
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        patches = schema.get("categoricals", {}).get("patch", [])
        versions = []
        for patch in patches:
            parts = str(patch).split(".")
            if parts and all(part.isdigit() for part in parts):
                versions.append((tuple(int(part) for part in parts), str(patch)))
        if versions:
            return max(versions)[1]
    return "unknown"


def safe_name(localized_name: str) -> str:
    return (
        localized_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace("-", "_")
    )


def main() -> None:
    heroes = json.loads(HEROES.read_text(encoding="utf-8"))
    id_by_name = {safe_name(hero["localized_name"]): int(hero["id"]) for hero in heroes}
    known_ids = set(id_by_name.values())

    matchups_df = pd.read_parquet(DATA / "hero_matchups.parquet")
    matchups = []
    skipped_unknown = 0
    for row in matchups_df.itertuples(index=False):
        hero_a = int(row.hero_a)
        hero_b = int(row.hero_b)
        games = int(row.games)
        winrate = float(row.hero_a_winrate)
        if hero_a not in known_ids or hero_b not in known_ids:
            skipped_unknown += 1
            continue
        if games < MIN_MATCHUP_GAMES or hero_a == hero_b:
            continue
        matchups.append(
            {
                "hero_id": hero_a,
                "vs_hero_id": hero_b,
                "advantage": winrate - 0.5,
                "winrate": winrate,
                "games": games,
            }
        )
        matchups.append(
            {
                "hero_id": hero_b,
                "vs_hero_id": hero_a,
                "advantage": (1.0 - winrate) - 0.5,
                "winrate": 1.0 - winrate,
                "games": games,
            }
        )
    matchups.sort(key=lambda row: (row["hero_id"], row["vs_hero_id"]))

    features = pd.read_parquet(DATA / "draft_features.parquet")
    rad_cols = [
        column
        for column in features.columns
        if column.startswith("rad_has_") and column[8:] in id_by_name
    ]
    dire_cols = [f"dire_has_{column[8:]}" for column in rad_cols]
    hero_ids = [id_by_name[column[8:]] for column in rad_cols]

    radiant = features[rad_cols].to_numpy(dtype=np.float32)
    dire = features[dire_cols].to_numpy(dtype=np.float32)
    radiant_win = features["radiant_win"].to_numpy(dtype=np.float32)

    games_matrix = radiant.T @ radiant + dire.T @ dire
    wins_matrix = radiant.T @ (radiant * radiant_win[:, None])
    wins_matrix += dire.T @ (dire * (1.0 - radiant_win)[:, None])

    winrates_df = pd.read_parquet(DATA / "hero_winrates.parquet")
    overall_wr = {
        int(row.hero_id): float(row.winrate)
        for row in winrates_df.itertuples(index=False)
    }

    synergies = []
    for i in range(len(hero_ids)):
        for j in range(i + 1, len(hero_ids)):
            games = float(games_matrix[i, j])
            if games < MIN_SYNERGY_GAMES:
                continue
            pair_winrate = float(wins_matrix[i, j] / games)
            hero_a = hero_ids[i]
            hero_b = hero_ids[j]
            expected = (overall_wr.get(hero_a, 0.5) + overall_wr.get(hero_b, 0.5)) / 2.0
            synergy = float(np.clip(pair_winrate - expected, -0.5, 0.5))
            synergies.append(
                {
                    "hero_id": hero_a,
                    "with_hero_id": hero_b,
                    "synergy": synergy,
                    "winrate": pair_winrate,
                    "games": int(games),
                }
            )
    synergies.sort(key=lambda row: (row["hero_id"], row["with_hero_id"]))

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "patch_version": active_patch(),
        "sources": [
            "ml/datasets/processed/hero_matchups.parquet",
            "ml/datasets/processed/draft_features.parquet",
            "ml/datasets/processed/hero_winrates.parquet",
        ],
        "minimum_games": {
            "matchup": MIN_MATCHUP_GAMES,
            "synergy": MIN_SYNERGY_GAMES,
        },
        "matchups": matchups,
        "synergies": synergies,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {OUTPUT}: {len(matchups)} ordered matchups, "
        f"{len(synergies)} synergy pairs "
        f"(skipped {skipped_unknown} rows with unknown hero ids)"
    )


if __name__ == "__main__":
    main()

