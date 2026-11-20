#!/usr/bin/env python3
"""Export Captains Mode drafts with their exact pick/ban order.

The feature dataset (`draft_features.parquet`) stores picks/bans as unordered
presence flags, so it cannot reconstruct "what did the draft look like before
pick k". The raw STRATZ payloads keep the `pickBans` array with an explicit
`order`, which is what this exporter uses.

Output: one JSON object per match, chronological:
    {"match_id":..., "start_time":..., "patch":..., "radiant_win":...,
     "actions":[{"order", "pick", "radiant", "hero_id"}, ...20]}

Read-only: it only reads `ml/datasets/raw` and the processed parquet.

Usage:
    ml/venv/Scripts/python.exe ml/backtest/export_drafts.py \
        --out ml/backtest/data/drafts_cm.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# Captains Mode has 10 picks and, depending on the era, 10 or 14 bans.
MIN_ACTIONS = 20
MAX_ACTIONS = 24


def _normalize_stratz(raw: dict) -> list[dict] | None:
    """STRATZ payload: string game mode, camelCase pickBans."""
    if raw.get("gameMode") != "CAPTAINS_MODE":
        return None
    raw_actions = raw.get("pickBans") or []
    actions = []
    for action in raw_actions:
        order, hero = action.get("order"), action.get("heroId")
        if order is None or hero is None:
            return None
        actions.append(
            {
                "order": int(order),
                "pick": bool(action.get("isPick")),
                "radiant": bool(action.get("isRadiant")),
                "hero_id": int(hero),
            }
        )
    return actions


def _normalize_opendota(raw: dict) -> list[dict] | None:
    """OpenDota payload: integer game_mode, snake_case picks_bans (team 0 = Radiant)."""
    if raw.get("game_mode") != 2:
        return None
    raw_actions = raw.get("picks_bans") or []
    actions = []
    for action in raw_actions:
        order, hero = action.get("order"), action.get("hero_id")
        if order is None or hero is None:
            return None
        actions.append(
            {
                "order": int(order),
                "pick": bool(action.get("is_pick")),
                "radiant": int(action.get("team", 0)) == 0,
                "hero_id": int(hero),
            }
        )
    return actions


def parse_match(path: str) -> dict | None:
    """Return a draft record for a Captains Mode match, else None."""
    try:
        return _parse_match(path)
    except Exception:
        # Raw payloads are heterogeneous; a malformed one is skipped, not fatal.
        return None


def _parse_match(path: str) -> dict | None:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        return None

    actions = _normalize_stratz(raw)
    if actions is None:
        actions = _normalize_opendota(raw)
    if actions is None:
        return None

    if not MIN_ACTIONS <= len(actions) <= MAX_ACTIONS:
        return None
    if sorted(action["order"] for action in actions) != list(range(len(actions))):
        return None

    heroes = [action["hero_id"] for action in actions]
    if len(set(heroes)) != len(heroes):
        return None

    picks = [action for action in actions if action["pick"]]
    bans = [action for action in actions if not action["pick"]]
    if len(picks) != 10 or len(bans) < 10:
        return None
    if sum(1 for action in picks if action["radiant"]) != 5:
        return None

    radiant_win = raw.get("didRadiantWin", raw.get("radiant_win"))
    match_id = raw.get("id", raw.get("match_id"))
    if radiant_win is None or match_id is None:
        return None

    start = raw.get("startDateTime")
    start_epoch = None
    if isinstance(start, str):
        try:
            start_epoch = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp())
        except ValueError:
            start_epoch = None
    elif isinstance(raw.get("start_time"), (int, float)):
        start_epoch = int(raw["start_time"])

    return {
        "match_id": int(match_id),
        "start_time": start_epoch,
        "radiant_win": bool(radiant_win),
        "actions": sorted(actions, key=lambda action: action["order"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="ml/datasets/raw")
    parser.add_argument("--parquet", default="ml/datasets/processed_v2/draft_features.parquet")
    parser.add_argument("--heroes", default="ml/datasets/heroes.json")
    parser.add_argument("--out", default="ml/backtest/data/drafts_cm.jsonl")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.json")))
    if not paths:
        print(f"no raw payloads under {args.raw_dir}", file=sys.stderr)
        return 2
    print(f"scanning {len(paths)} raw payloads with {args.workers} workers", flush=True)

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(parse_match, paths, chunksize=64), start=1):
            if record is not None:
                records.append(record)
            if index % 2000 == 0:
                print(f"  {index}/{len(paths)} scanned, {len(records)} CM drafts", flush=True)

    if not records:
        print("no Captains Mode drafts with a full 20-action order were found", file=sys.stderr)
        return 2

    # Patch and epoch time come from the processed dataset (the raw payloads
    # only carry a game version id).
    patches: dict[int, str] = {}
    epochs: dict[int, int] = {}
    try:
        import pandas as pd

        frame = pd.read_parquet(args.parquet, columns=["match_id", "start_time", "patch"])
        patches = dict(zip(frame["match_id"].astype("int64"), frame["patch"].astype(str)))
        epochs = dict(zip(frame["match_id"].astype("int64"), frame["start_time"].astype("int64")))
    except Exception as error:  # pragma: no cover - analysis convenience only
        print(f"warning: could not read {args.parquet}: {error}", file=sys.stderr)

    known_heroes: set[int] = set()
    try:
        with open(args.heroes, encoding="utf-8") as handle:
            heroes = json.load(handle)
        known_heroes = {int(hero["id"]) for hero in heroes}
    except Exception as error:  # pragma: no cover
        print(f"warning: could not read {args.heroes}: {error}", file=sys.stderr)

    unknown: Counter[int] = Counter()
    for record in records:
        match_id = record["match_id"]
        record["patch"] = patches.get(match_id)
        if record["start_time"] is None:
            record["start_time"] = epochs.get(match_id)
        for action in record["actions"]:
            if known_heroes and action["hero_id"] not in known_heroes:
                unknown[action["hero_id"]] += 1

    records.sort(key=lambda item: (item["start_time"] or 0, item["match_id"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    digest = hashlib.sha256()
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))

    earliest = next((r["start_time"] for r in records if r["start_time"]), None)
    latest = next((r["start_time"] for r in reversed(records) if r["start_time"]), None)
    manifest = {
        "generated_by": "ml/backtest/export_drafts.py",
        "source": os.path.join(args.raw_dir, "*.json"),
        "parquet": args.parquet,
        "matches": len(records),
        "scanned_files": len(paths),
        "actions": sum(len(r["actions"]) for r in records),
        "start_time_min": earliest,
        "start_time_max": latest,
        "patch_counts": dict(Counter(r.get("patch") or "unknown" for r in records).most_common()),
        "unknown_hero_ids": dict(unknown.most_common(10)),
        "sha256": digest.hexdigest(),
        "output": os.path.abspath(args.out),
    }
    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
