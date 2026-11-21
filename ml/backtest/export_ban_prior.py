"""Export per-patch hero ban prevalence for the desktop ban-suggestion list.

Reads the ordered Captains Mode draft corpus written by
``ml/backtest/export_drafts.py`` and writes
``infrastructure/seeders/ban_seed.json``, which ``seed_matchups`` loads into the
``hero_ban_stats`` table. Only counts are committed; no draft rows are.

The measure is descriptive: how often human teams banned a hero on a patch. It
is not evidence that banning that hero is good, and the UI says so.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", default="ml/backtest/data/drafts_cm.jsonl")
    parser.add_argument("--out", default="infrastructure/seeders/ban_seed.json")
    args = parser.parse_args()

    drafts_path = Path(args.drafts)
    games: Counter[str] = Counter()
    bans: dict[str, Counter[int]] = defaultdict(Counter)
    # Export window per patch, so the UI can say which drafts the counts cover
    # instead of implying they describe every All Pick ban on that patch.
    windows: dict[str, list[int]] = {}
    matches = 0

    with drafts_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            patch = str(record["patch"])
            games[patch] += 1
            matches += 1
            start_time = record.get("start_time")
            if start_time:
                window = windows.setdefault(patch, [start_time, start_time])
                window[0] = min(window[0], start_time)
                window[1] = max(window[1], start_time)
            for action in record["actions"]:
                if not action["pick"]:
                    bans[patch][int(action["hero_id"])] += 1

    rows = [
        {
            "patch_version": patch,
            "hero_id": hero_id,
            "bans": count,
            "games": games[patch],
        }
        for patch in sorted(bans)
        for hero_id, count in sorted(bans[patch].items())
    ]

    patch_info = {
        patch: {
            "drafts": games[patch],
            "window_start": windows.get(patch, [None, None])[0],
            "window_end": windows.get(patch, [None, None])[1],
        }
        for patch in sorted(games)
    }

    payload = {
        "source": "Captain's Mode drafts exported by ml/backtest/export_drafts.py",
        "generated_by": "ml/backtest/export_ban_prior.py",
        "matches": matches,
        "patches": patch_info,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"{matches} drafts, {len(games)} patches, {len(rows)} rows -> {out_path}")
    for patch in sorted(games):
        top = ", ".join(f"{hero}:{count}" for hero, count in bans[patch].most_common(3))
        print(f"  {patch}: {games[patch]} drafts; top bans {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
