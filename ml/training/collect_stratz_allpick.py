#!/usr/bin/env python3
"""
Stratz All Pick Hybrid Collector
===============================
Collects All Pick matches by:
  1. Discovering IDs via OpenDota publicMatches (filtered for ALL_PICK, supports pagination via less_than_match_id)
  2. Fetching full details via Stratz GraphQL bulk (matches(ids: [...])) - higher quality, includes averageRank, draft

This gives volume of OpenDota + quality of Stratz.
Saves to ml/datasets/raw/allpick/<id>.json in Stratz format (also works with OpenDota fallback).

Usage:
    python ml/training/collect_stratz_allpick.py --target 20000 --mmr 3000 --rate-limit 0.5
    python ml/training/collect_stratz_allpick.py --target 5000 --no-stratz  # OpenDota only (faster if Stratz limited)
"""

import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime

import requests

from settings import get_opendota_key, get_stratz_key

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ML_ROOT / "datasets" / "raw"
DEFAULT_ALLPICK = DEFAULT_RAW / "allpick"


STRATZ_URL = "https://api.stratz.com/graphql"
OPENDOTA_URL = "https://api.opendota.com/api"

STRATZ_BATCH_QUERY = """
query GetMatchesByIds($ids: [Long!]!) {
    matches(ids: $ids) {
        id
        didRadiantWin
        startDateTime
        durationSeconds
        gameMode
        lobbyType
        gameVersionId
        numHumanPlayers
        averageRank
        players {
            heroId
            isRadiant
            playerSlot
            lane
            steamAccountId
        }
        pickBans {
            heroId
            isPick
            isRadiant
            order
        }
    }
}
"""

STRATZ_SINGLE_QUERY = """
query GetMatch($id: Long!) {
    match(id: $id) {
        id
        didRadiantWin
        startDateTime
        durationSeconds
        gameMode
        lobbyType
        gameVersionId
        numHumanPlayers
        averageRank
        players {
            heroId
            isRadiant
            playerSlot
            lane
            steamAccountId
        }
        pickBans {
            heroId
            isPick
            isRadiant
            order
        }
    }
}
"""


def load_existing(raw_dir: Path, allpick_dir: Path):
    existing_raw = set()
    existing_allpick = set()
    for f in raw_dir.glob("*.json"):
        if f.name in ("metadata.json", "all_pick_matches.json", "allpick_ids.json"):
            continue
        try:
            existing_raw.add(int(f.stem))
        except:
            pass
    for f in allpick_dir.glob("*.json"):
        try:
            existing_allpick.add(int(f.stem))
        except:
            pass
    return existing_raw, existing_allpick


def discover_allpick_ids_opendota(existing: set, target: int, mmr: int, opendota_key: str, max_pages: int = 500):
    """Discover All Pick IDs via OpenDota publicMatches paginated by less_than_match_id (reliable)."""
    discovered = set()
    less_than = None
    pages = 0
    no_new_streak = 0

    print(f"Discovering All Pick IDs via OpenDota (mmr>={mmr}, target={target})...")
    session = requests.Session()
    while len(discovered) + len(existing) < target and pages < max_pages:
        params = {"limit": 100}
        # min_mmr is useful for captains but for All Pick we can scan more broadly
        # Use MMR filter only if >0
        if mmr and mmr > 0:
            params["min_mmr"] = mmr
        if opendota_key:
            params["api_key"] = opendota_key
        if less_than:
            params["less_than_match_id"] = less_than
        try:
            resp = session.get(f"{OPENDOTA_URL}/publicMatches", params=params, timeout=30)
            if resp.status_code == 429:
                print("  OpenDota 429, sleeping 60s...")
                time.sleep(60)
                continue
            if resp.status_code == 400 and "API key" in resp.text:
                print("  OpenDota API key invalid, retrying without key...")
                params.pop("api_key", None)
                opendota_key = None
                resp = session.get(f"{OPENDOTA_URL}/publicMatches", params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  OpenDota error {resp.status_code}: {resp.text[:300]}, stopping")
                break
            matches = resp.json()
            if not matches:
                print("  No matches returned, stopping")
                break
            new = 0
            for m in matches:
                mid = m.get("match_id")
                gm = m.get("game_mode")
                # 1 = All Pick
                if gm != 1:
                    continue
                if mid in existing or mid in discovered:
                    continue
                discovered.add(mid)
                new += 1
            pages += 1
            less_than = matches[-1].get("match_id")
            print(f"  Page {pages}: {new} new All Pick, total discovered={len(discovered)} (hit {len(existing)+len(discovered)}/{target})")
            if new == 0:
                no_new_streak += 1
                if no_new_streak >= 100:
                    print("  100 pages with no new All Pick, stopping discovery")
                    break
            else:
                no_new_streak = 0
            time.sleep(1.0)  # OpenDota rate limit
        except Exception as e:
            print(f"  Discovery error page {pages}: {e}")
            time.sleep(5)
            break

    print(f"Discovery complete: {len(discovered)} new IDs, {len(existing)} existing, {pages} pages")
    return sorted(discovered)


def fetch_stratz_batch(ids, stratz_key, rate_limit=0.6):
    """Fetch via Stratz matches(ids) in chunks of 20 with retries. Returns list of matches."""
    headers = {
        "Authorization": f"Bearer {stratz_key}",
        "User-Agent": "STRATZ_API",
        "Content-Type": "application/json"
    }
    all_matches = []
    # Stratz limits batch to ~20
    for i in range(0, len(ids), 10):
        chunk = ids[i:i+10]
        payload = {"query": STRATZ_BATCH_QUERY, "variables": {"ids": chunk}}
        for attempt in range(5):
            try:
                resp = requests.post(STRATZ_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code == 429:
                    wait = min(2**attempt * 10, 120)
                    print(f"    Stratz 429, waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = 2**attempt
                    print(f"    Stratz 5xx {resp.status_code}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    print(f"    GraphQL errors: {data['errors'][:2]}")
                    raise Exception(str(data["errors"]))
                matches = data.get("data", {}).get("matches", [])
                # Filter None entries
                matches = [m for m in matches if m and m.get("id")]
                all_matches.extend(matches)
                # progress
                if (i//20) % 10 == 0:
                    print(f"    Stratz batch {i//20+1}/{(len(ids)+19)//20}: got {len(matches)}/{len(chunk)}")
                break
            except Exception as e:
                if attempt == 4:
                    print(f"    Failed chunk {i}: {e}, falling back to singles")
                    # fallback to singles
                    for mid in chunk:
                        try:
                            r2 = requests.post(STRATZ_URL, json={"query": STRATZ_SINGLE_QUERY, "variables": {"id": mid}}, headers=headers, timeout=30)
                            if r2.status_code == 429:
                                time.sleep(30)
                                continue
                            d2 = r2.json()
                            m2 = d2.get("data", {}).get("match")
                            if m2:
                                all_matches.append(m2)
                            time.sleep(rate_limit)
                        except:
                            pass
                    break
                wait = 2**attempt
                print(f"    Chunk error {e}, retry in {wait}s")
                time.sleep(wait)
        time.sleep(rate_limit)
    return all_matches


def fetch_opendota_details(ids, opendota_key, allpick_dir: Path, rate_limit=0.4):
    """Fallback: fetch via OpenDota /matches/{id} for ids that Stratz missed."""
    saved = 0
    for mid in ids:
        target = allpick_dir / f"{mid}.json"
        if target.exists():
            continue
        try:
            params = {}
            if opendota_key:
                params["api_key"] = opendota_key
            resp = requests.get(f"{OPENDOTA_URL}/matches/{mid}", params=params, timeout=30)
            if resp.status_code == 429:
                print("    OpenDota 429, sleep 60")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            # keep only if it has players
            if not data.get("players"):
                continue
            with open(target, "w") as f:
                json.dump(data, f)
            saved += 1
            if saved % 100 == 0:
                print(f"    OpenDota fallback saved {saved}")
            time.sleep(rate_limit)
        except Exception as e:
            continue
    return saved


def main():
    parser = argparse.ArgumentParser(description="Stratz All Pick Hybrid Collector")
    parser.add_argument("--target", type=int, default=20000, help="Total All Pick target (existing + new)")
    parser.add_argument("--mmr", type=int, default=3000, help="min MMR for All Pick discovery")
    parser.add_argument("--max-pages", type=int, default=400, help="max publicMatches pages to scan")
    parser.add_argument("--rate-limit", type=float, default=0.6, help="Stratz sec between requests")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ALLPICK), help="allpick output dir")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW), help="raw dir")
    parser.add_argument("--stratz-key", type=str, default=None, help="Stratz key (env STRATZ_API_KEY or default)")
    parser.add_argument("--opendota-key", type=str, default=None, help="OpenDota key (env OPENDOTA_API_KEY or default)")
    parser.add_argument("--no-stratz", action="store_true", help="Skip Stratz, use OpenDota details only (faster when Stratz limited)")
    parser.add_argument("--batch-size", type=int, default=20, help="not used, kept for compat")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    allpick_dir = Path(args.output)
    allpick_dir.mkdir(parents=True, exist_ok=True)

    stratz_key = args.stratz_key or get_stratz_key()
    opendota_key = args.opendota_key or get_opendota_key()

    if not args.no_stratz and not stratz_key:
        parser.error("STRATZ_API_KEY is required unless --no-stratz is used")

    existing_raw, existing_allpick = load_existing(raw_dir, allpick_dir)
    print(f"Existing raw: {len(existing_raw)}, existing allpick: {len(existing_allpick)}")
    print(f"Target allpick: {args.target}, need: {max(0, args.target - len(existing_allpick))}")

    if len(existing_allpick) >= args.target:
        print("Already at target, nothing to do")
        return

    discovered = discover_allpick_ids_opendota(existing_allpick, args.target, args.mmr, opendota_key, args.max_pages)
    if not discovered:
        print("No new IDs discovered, try lowering --mmr or increasing --max-pages")
        return

    to_fetch = discovered[: args.target - len(existing_allpick)]
    print(f"Will fetch {len(to_fetch)} matches (discovered {len(discovered)})")

    saved_stratz = 0
    saved_opendota = 0
    missed = []

    if not args.no_stratz:
        print("\nFetching via Stratz bulk...")
        stratz_matches = fetch_stratz_batch(to_fetch, stratz_key, rate_limit=args.rate_limit)
        print(f"Stratz returned {len(stratz_matches)} matches")
        for m in stratz_matches:
            mid = m.get("id")
            if not mid:
                continue
            # Only keep All Pick (Stratz returns string mode)
            gm = m.get("gameMode")
            if gm not in ("ALL_PICK", "ALL_PICK_RANKED", "ALL_PICK_CUSTOM", None):
                # For Stratz, also accept any ALL_PICK variants but filter out captains just in case
                if gm and "ALL_PICK" not in str(gm).upper():
                    continue
            target = allpick_dir / f"{mid}.json"
            if not target.exists():
                with open(target, "w") as f:
                    json.dump(m, f, indent=2)
                saved_stratz += 1
        print(f"Saved {saved_stratz} via Stratz to {allpick_dir}")

        # Find which IDs Stratz missed (e.g., private matches)
        fetched_ids = {m.get("id") for m in stratz_matches if m.get("id")}
        missed = [mid for mid in to_fetch if mid not in fetched_ids and not (allpick_dir / f"{mid}.json").exists()]
        print(f"Stratz missed {len(missed)} IDs, falling back to OpenDota for those...")
        if missed:
            saved_opendota = fetch_opendota_details(missed[:2000], opendota_key, allpick_dir, rate_limit=0.5)
            print(f"Saved {saved_opendota} via OpenDota fallback")
    else:
        print("\nSkipping Stratz, fetching via OpenDota directly...")
        saved_opendota = fetch_opendota_details(to_fetch, opendota_key, allpick_dir, rate_limit=0.35)
        print(f"Saved {saved_opendota} via OpenDota")

    total_now = len(list(allpick_dir.glob("*.json")))
    print("\n=== FINAL ===")
    print(f"Existing before: {len(existing_allpick)}")
    print(f"Stratz saved: {saved_stratz}")
    print(f"OpenDota fallback saved: {saved_opendota}")
    print(f"Total allpick now: {total_now} / target {args.target}")
    print(f"Raw still: {len(list(raw_dir.glob('*.json')))} (filtered)")

if __name__ == "__main__":
    main()
