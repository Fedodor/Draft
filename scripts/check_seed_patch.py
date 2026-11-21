#!/usr/bin/env python3
"""Fail when the shipped seeds disagree about which patch is active.

`seed_matchups` takes the active patch from `scoring_seed.json` and loads ban
counts from `ban_seed.json`. If the ban seed has no rows for that patch the app
still starts, reports healthy data and then shows an empty All Pick preview, so
the mismatch is checked here instead of being found by a user.

Usage:
    python scripts/check_seed_patch.py [scoring_seed.json] [ban_seed.json]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORING = ROOT / "infrastructure/seeders/scoring_seed.json"
DEFAULT_BANS = ROOT / "infrastructure/seeders/ban_seed.json"


def main(argv: list[str]) -> int:
    scoring_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_SCORING
    ban_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_BANS

    try:
        scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"{scoring_path.name}: cannot read scoring seed: {error}", file=sys.stderr)
        return 1

    patch = str(scoring.get("patch_version") or "").strip()
    if not patch:
        print(
            f"{scoring_path.name} has no patch_version, so the active patch is "
            "undefined. Regenerate it with `python ml/training/export_scoring_seed.py`.",
            file=sys.stderr,
        )
        return 1

    try:
        bans = json.loads(ban_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"{ban_path.name}: cannot read ban seed: {error}", file=sys.stderr)
        return 1

    rows = bans.get("rows") or []
    patches = sorted(
        {
            str(row.get("patch_version") or "").strip()
            for row in rows
            if isinstance(row, dict)
        }
        - {""}
    )
    if patch not in patches:
        print(
            f"{ban_path.name} has no ban rows for patch {patch}, the patch "
            f"{scoring_path.name} activates. Available patches: "
            f"{', '.join(patches) or '(none)'}.",
            file=sys.stderr,
        )
        print(
            "Re-export it with `python ml/backtest/export_ban_prior.py` once "
            f"drafts exist for {patch}, or point the scoring seed at a patch "
            "with ban counts.",
            file=sys.stderr,
        )
        return 1

    print(
        f"scoring seed and ban seed both cover patch {patch} "
        f"({len(rows)} ban rows across {len(patches)} patches)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
