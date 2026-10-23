#!/usr/bin/env python3
"""Enhanced Data Collector for Smart Draft Assistant

Collects professional and ranked Dota 2 matches from:
- OpenDota API: standard collection with detail fetches
- Stratz GraphQL API: faster bulk collection with draft data included

Features:
- Automatic pagination for large datasets
- Resume capability (skips already downloaded matches)
- Patch-aware collection
- MMR filtering for ranked matches
- Progress tracking and metadata

Usage:
    # OpenDota collection
    python data_collector.py --matches 10000 --patches 7.41 7.40 --output ml/datasets/raw

    # Stratz collection (faster, includes draft data)
    python data_collector.py --source stratz --matches 5000 --patches 7.41 7.40 --output ml/datasets/raw
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

ML_ROOT = Path(__file__).resolve().parent.parent


class StratzCollector:
    """Collector using Stratz GraphQL API for enhanced match details.

    Stratz API doesn't support browsing/searching matches, but provides:
    - Higher quality match data with draft information
    - Additional fields (gameVersionId, bracket, etc.)
    - Batch fetching of multiple matches by ID

    Use OpenDota for discovery, then Stratz for enhanced details.
    """

    GRAPHQL_URL = "https://api.stratz.com/graphql"

    # GraphQL query for fetching matches with draft data by IDs
    MATCHES_BY_IDS_QUERY = """
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

    # Query for single match with full details
    MATCH_QUERY = """
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

    def __init__(
        self,
        api_key: str,
        rate_limit: float = 0.5,
        max_concurrent: int = 5,
        timeout: int = 60,
    ):
        if not api_key:
            raise ValueError("Stratz API key is required")
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _graphql_request(self, query: str, variables: Dict = None) -> Dict:
        """Make rate-limited GraphQL request with retries."""
        # Rate limiting
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "STRATZ_API",
            "Content-Type": "application/json",
        }

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(5):
            try:
                async with self.session.post(
                    self.GRAPHQL_URL, json=payload, headers=headers
                ) as resp:
                    if resp.status == 429:
                        wait_time = min(2**attempt * 10, 120)
                        print(
                            f"  Rate limited (429), waiting {wait_time}s... "
                            f"(attempt {attempt + 1}/5)"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    elif resp.status >= 500:
                        wait_time = 2**attempt
                        print(
                            f"  Server error ({resp.status}), waiting {wait_time}s... "
                            f"(attempt {attempt + 1}/5)"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    resp.raise_for_status()
                    result = await resp.json()

                    if "errors" in result:
                        raise Exception(f"GraphQL errors: {result['errors']}")

                    return result.get("data", {})

            except Exception as e:
                if attempt == 4:
                    raise
                wait_time = 2**attempt
                print(
                    f"  Request failed: {e}, retrying in {wait_time}s... "
                    f"(attempt {attempt + 1}/5)"
                )
                await asyncio.sleep(wait_time)

        raise Exception("Max retries exceeded")

    async def get_match_by_id(self, match_id: int) -> Optional[Dict]:
        """Fetch a single match by ID with full draft data."""
        result = await self._graphql_request(self.MATCH_QUERY, {"id": match_id})
        return result.get("match")

    async def get_matches_by_ids(self, match_ids: List[int]) -> List[Dict]:
        """Fetch multiple matches by IDs in a single request (up to 20)."""
        # Stratz limits batch size, split into chunks
        chunk_size = 20
        all_matches = []

        for i in range(0, len(match_ids), chunk_size):
            chunk = match_ids[i : i + chunk_size]
            try:
                result = await self._graphql_request(
                    self.MATCHES_BY_IDS_QUERY, {"ids": chunk}
                )
                matches = result.get("matches", [])
                all_matches.extend(matches)
            except Exception as e:
                print(f"  Error fetching batch {i // chunk_size + 1}: {e}")
                # Fall back to individual fetches for this chunk
                for mid in chunk:
                    try:
                        match = await self.get_match_by_id(mid)
                        if match:
                            all_matches.append(match)
                    except Exception:
                        pass

        return all_matches

    async def enhance_matches(
        self,
        match_ids: List[int],
        output_dir: Path,
        resume: bool = True,
    ) -> Dict:
        """Enhance existing OpenDota matches with Stratz data.

        Fetches matches from Stratz and saves enhanced versions.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        enhanced_dir = output_dir / "stratz_enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        # Check what's already enhanced
        enhanced_ids: Set[int] = set()
        if resume:
            for f in enhanced_dir.glob("*.json"):
                if f.stem.isdigit():
                    enhanced_ids.add(int(f.stem))

        to_enhance = [mid for mid in match_ids if mid not in enhanced_ids]
        print(
            f"Enhancing {len(to_enhance)} matches with Stratz data "
            f"({len(enhanced_ids)} already done)"
        )

        successful, failed = 0, 0

        # Process in batches
        for i in range(0, len(to_enhance), 20):
            batch = to_enhance[i : i + 20]
            try:
                matches = await self.get_matches_by_ids(batch)

                for match in matches:
                    match_id = match.get("id")
                    if match_id:
                        enhanced_file = enhanced_dir / f"{match_id}.json"
                        with open(enhanced_file, "w") as f:
                            json.dump(match, f, indent=2)
                        successful += 1

                print(
                    f"  Progress: {min(i + len(batch), len(to_enhance))}/"
                    f"{len(to_enhance)} (success: {successful}, failed: {failed})"
                )

            except Exception as e:
                print(f"  Error on batch {i // 20 + 1}: {e}")
                failed += len(batch)

        return {
            "total_enhanced": successful,
            "failed": failed,
            "enhanced_dir": str(enhanced_dir),
        }


class OpenDotaCollector:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,
        max_concurrent: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _rate_limited_get(self, url: str, params: Dict = None) -> Dict:
        """Make rate-limited GET request with exponential backoff retries."""
        if params is None:
            params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        # Rate limiting
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        for attempt in range(5):
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait_time = min(2**attempt * 5, 60)
                        print(
                            f"  Rate limited (429), waiting {wait_time}s... "
                            f"(attempt {attempt + 1}/5)"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    elif resp.status >= 500:
                        wait_time = 2**attempt
                        print(
                            f"  Server error ({resp.status}), waiting {wait_time}s... "
                            f"(attempt {attempt + 1}/5)"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except aiohttp.ClientError as e:
                if attempt == 4:
                    raise
                wait_time = 2**attempt
                print(
                    f"  Request failed: {e}, retrying in {wait_time}s... "
                    f"(attempt {attempt + 1}/5)"
                )
                await asyncio.sleep(wait_time)

        raise Exception("Max retries exceeded")

    async def get_pro_matches(
        self, limit: int = 100, patch: str = None, less_than_match_id: int = None
    ) -> List[Dict]:
        """Fetch professional matches with pagination."""
        url = f"{self.BASE_URL}/proMatches"
        params = {"limit": min(limit, 100)}
        if patch:
            params["patch"] = patch
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return await self._rate_limited_get(url, params)

    async def get_public_matches(
        self, limit: int = 100, mmr: int = None, less_than_match_id: int = None
    ) -> List[Dict]:
        """Fetch public matches with pagination (no server-side game_mode filter)."""
        url = f"{self.BASE_URL}/publicMatches"
        params = {"limit": min(limit, 100)}
        if mmr:
            params["mmr_ascending"] = "false"
            params["min_mmr"] = str(mmr)
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return await self._rate_limited_get(url, params)

    @staticmethod
    def _is_captains_mode(m: Dict) -> bool:
        """Captain's Mode (22) or Captain's Draft (23)."""
        return m.get("game_mode") in (22, 23)

    @staticmethod
    def _is_all_pick(m: Dict) -> bool:
        return m.get("game_mode") == 1

    async def get_match_details(self, match_id: int) -> Dict:
        """Fetch detailed match data."""
        url = f"{self.BASE_URL}/matches/{match_id}"
        return await self._rate_limited_get(url)

    async def collect_matches_paginated(
        self,
        target_count: int,
        output_dir: Path,
        patches: List[str] = None,
        include_pro: bool = True,
        include_ranked: bool = True,
        min_mmr: int = 4000,
        allpick_target: int = 0,
        allpick_mmr: int = 3000,
        max_scan_pages: int = 200,
        resume: bool = True,
    ) -> Dict:
        """Collect matches using pagination to get large datasets.

        - Captain's Mode/Draft (game_mode 22/23) details -> output_dir/<id>.json
          (draft training data)
        - All Pick (game_mode 1) details -> output_dir/allpick/<id>.json
          (hero matchup stats)

        Resume skips IDs that already have a downloaded file, not just known
        IDs. Discovery scans newest matches first (new patches prioritized).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        allpick_dir = output_dir / "allpick"
        allpick_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load prior metadata (dedupe across runs) ----
        metadata_file = output_dir / "metadata.json"
        meta = {}
        if resume and metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)

        known_train: Set[int] = set(meta.get("match_ids", []))  # pro + captains
        known_allpick: Set[int] = set(meta.get("allpick_ids", []))

        # ---- What is already on disk (source of truth for resume) ----
        downloaded_train: Set[int] = {
            int(f.stem)
            for f in output_dir.glob("*.json")
            if f.name not in ("metadata.json", "all_pick_matches.json", "allpick_ids.json")
            and f.stem.isdigit()
        }
        downloaded_allpick: Set[int] = {
            int(f.stem) for f in allpick_dir.glob("*.json") if f.stem.isdigit()
        }

        train_ids: Set[int] = set(known_train) | downloaded_train
        allpick_ids: Set[int] = set(known_allpick) | downloaded_allpick

        print(
            f"Training match target: {target_count} "
            f"(downloaded: {len(downloaded_train)}, known: {len(train_ids)})"
        )
        print(
            f"All Pick target:       {allpick_target} "
            f"(downloaded: {len(downloaded_allpick)}, known: {len(allpick_ids)})"
        )

        discovered_public: Set[int] = set(meta.get("public_ids", []))

        # ---- Discovery: professional matches (newest patches first) ----
        if include_pro:
            print("Discovering pro match IDs (newest patches first)...")
            for patch in patches or [
                "7.41",
                "7.40",
                "7.39",
                "7.38",
                "7.37",
                "7.36",
                "7.35",
            ]:
                if len(train_ids) >= target_count:
                    break
                print(f"  Patch {patch}...")
                last_match_id = None
                while len(train_ids) < target_count:
                    try:
                        matches = await self.get_pro_matches(
                            limit=100, patch=patch, less_than_match_id=last_match_id
                        )
                        if not matches:
                            break
                        new_ids = 0
                        for m in matches:
                            mid = m["match_id"]
                            if mid not in train_ids:
                                train_ids.add(mid)
                                new_ids += 1
                        last_match_id = matches[-1]["match_id"]
                        print(
                            f"    Batch: {len(matches)} matches ({new_ids} new), "
                            f"training total: {len(train_ids)}"
                        )
                        if new_ids == 0:
                            break  # exhausted this patch
                    except Exception as e:
                        print(f"    Error fetching pro matches for patch {patch}: {e}")
                        break

        # ---- Discovery: public captain's mode + All Pick (one combined sweep) ----
        if include_ranked:
            print(
                f"Discovering public matches "
                f"(captain's + All Pick, mmr>={min_mmr})..."
            )
            last_match_id = None
            captains_scan = 0
            allpick_scan = 0
            no_new_in_a_row = 0

            # Phase A: captain's mode (high MMR)
            while len(train_ids) < target_count and captains_scan < max_scan_pages:
                try:
                    matches = await self.get_public_matches(
                        limit=100, mmr=min_mmr, less_than_match_id=last_match_id
                    )
                    if not matches:
                        break
                    new_ids = 0
                    for m in matches:
                        mid = m["match_id"]
                        if mid in discovered_public:
                            continue
                        discovered_public.add(mid)
                        new_ids += 1
                        if self._is_captains_mode(m):
                            train_ids.add(mid)
                    captains_scan += 1
                    last_match_id = matches[-1]["match_id"]
                    print(
                        f"  Captains scan {captains_scan}: page={len(matches)} "
                        f"new_ids={new_ids}, "
                        f"captains total: {len(train_ids) - len(downloaded_train)}"
                    )
                    no_new_in_a_row = 0 if new_ids else no_new_in_a_row + 1
                    if no_new_in_a_row >= 3:
                        break
                except Exception as e:
                    print(f"  Error in captain's scan: {e}")
                    break

            # Phase B: All Pick (lower MMR threshold for volume)
            if (
                allpick_target > 0
                and len(allpick_ids) < allpick_target
                and len(downloaded_allpick) < allpick_target
            ):
                print(f"Discovering All Pick matches (mmr>={allpick_mmr})...")
                last_match_id = None
                no_new_in_a_row = 0
                while len(allpick_ids) < allpick_target and allpick_scan < max_scan_pages:
                    try:
                        matches = await self.get_public_matches(
                            limit=100, mmr=allpick_mmr, less_than_match_id=last_match_id
                        )
                        if not matches:
                            break
                        new_ids = 0
                        for m in matches:
                            if not self._is_all_pick(m):
                                continue
                            mid = m["match_id"]
                            if mid in allpick_ids:
                                continue
                            allpick_ids.add(mid)
                            discovered_public.add(mid)
                            new_ids += 1
                        allpick_scan += 1
                        last_match_id = matches[-1]["match_id"]
                        print(
                            f"  All Pick scan {allpick_scan}: {new_ids} new, "
                            f"allpick total: {len(allpick_ids)}"
                        )
                        no_new_in_a_row = 0 if new_ids else no_new_in_a_row + 1
                        if no_new_in_a_row >= 3:
                            break
                    except Exception as e:
                        print(f"  Error in All Pick scan: {e}")
                        break

        # ---- Build fetch queues (only IDs lacking a file) ----
        train_to_fetch: List[int] = [i for i in train_ids if i not in downloaded_train]
        allpick_to_fetch: List[int] = [
            i for i in allpick_ids if i not in downloaded_allpick
        ]

        print(f"\nTraining details to fetch: {len(train_to_fetch)}")
        print(f"All Pick details to fetch: {len(allpick_to_fetch)}")

        # ---- Persist metadata before fetching (resumable) ----
        metadata = {
            "collected_at": datetime.utcnow().isoformat(),
            "target_count": target_count,
            "allpick_target": allpick_target,
            "patches": patches,
            "include_pro": include_pro,
            "include_ranked": include_ranked,
            "min_mmr": min_mmr,
            "allpick_mmr": allpick_mmr,
            "match_ids": sorted(train_ids),
            "allpick_ids": sorted(allpick_ids),
            "public_ids": sorted(discovered_public),
            "train_to_fetch": train_to_fetch,
            "allpick_to_fetch": allpick_to_fetch,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # ---- Fetch details with concurrency ----
        async def fetch_one(
            match_id: int,
        ) -> Tuple[int, str, bool, Optional[str]]:
            """Return (match_id, dest_key, ok, error) with dest_key in {'train','allpick'}."""
            dest = (
                "allpick"
                if match_id in allpick_ids and match_id not in train_ids
                else "train"
            )
            match_file = (
                allpick_dir if dest == "allpick" else output_dir
            ) / f"{match_id}.json"
            if match_file.exists():
                return match_id, dest, True, None
            try:
                match = await self.get_match_details(match_id)
                if dest == "train" and not match.get("picks_bans"):
                    return match_id, dest, False, "No draft data"
                with open(match_file, "w") as f:
                    json.dump(match, f)
                return match_id, dest, True, None
            except Exception as e:
                return match_id, dest, False, str(e)

        # Train details first (more valuable), then all pick
        all_to_fetch = [("train", i) for i in train_to_fetch] + [
            ("allpick", i) for i in allpick_to_fetch
        ]

        if not all_to_fetch:
            print("Nothing new to fetch (already downloaded).")
        else:
            print(
                f"\nFetching details for {len(all_to_fetch)} matches "
                f"({len(train_to_fetch)} training, "
                f"{len(allpick_to_fetch)} all pick)..."
            )
            successful, failed = 0, 0
            errors: List[str] = []

            for i in range(0, len(all_to_fetch), self.max_concurrent * 10):
                batch = all_to_fetch[i : i + self.max_concurrent * 10]
                tasks = [fetch_one(mid) for _, mid in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed += 1
                        errors.append(f"{batch[j][1]}:{batch[j][0]}: {result}")
                    elif result[2]:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(f"{result[1]}:{result[0]}: {result[3]}")

                print(
                    f"  Progress: {min(i + len(batch), len(all_to_fetch))}/"
                    f"{len(all_to_fetch)} (success: {successful}, failed: {failed})"
                )

            metadata["successful"] = successful
            metadata["failed"] = failed
            metadata["errors"] = errors[:100]

        metadata["completed_at"] = datetime.utcnow().isoformat()
        metadata["requests_made"] = self.request_count

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


async def main():
    parser = argparse.ArgumentParser(
        description="Collect Dota 2 match data from OpenDota or Stratz API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source selection
    parser.add_argument(
        "--source",
        type=str,
        default="opendota",
        choices=["opendota", "stratz"],
        help="Data source: 'opendota' or 'stratz' (Stratz is faster for draft data)",
    )

    # Common options
    parser.add_argument(
        "--matches",
        type=int,
        default=10000,
        help="Target number of training matches (pro + captain's mode) to collect",
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        default=["7.41", "7.40", "7.39", "7.38", "7.37", "7.36", "7.35"],
        help="Patches to collect from (newest first)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ML_ROOT / "datasets" / "raw"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing data",
    )

    # Stratz-specific options
    parser.add_argument(
        "--stratz-key",
        type=str,
        default=None,
        help="Stratz API key (or set STRATZ_API_KEY env var)",
    )
    parser.add_argument(
        "--stratz-rate-limit",
        type=float,
        default=0.5,
        help="Stratz requests per second (default: 0.5)",
    )
    parser.add_argument(
        "--stratz-enhance",
        action="store_true",
        help="Enhance downloaded matches with Stratz data (adds draft details)",
    )
    parser.add_argument(
        "--stratz-max",
        type=int,
        default=1000,
        help="Max matches to enhance with Stratz data",
    )

    # OpenDota-specific options
    parser.add_argument(
        "--allpick-matches",
        type=int,
        default=None,
        help="Target number of All Pick matches for hero matchup stats "
        "(0 = skip; default = matches//2 when ranked enabled)",
    )
    parser.add_argument(
        "--include-pro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include professional matches (OpenDota only)",
    )
    parser.add_argument(
        "--include-ranked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include ranked public matches (OpenDota only)",
    )
    parser.add_argument(
        "--min-mmr",
        type=int,
        default=4000,
        help="Minimum average MMR for captain's mode matches (OpenDota only)",
    )
    parser.add_argument(
        "--allpick-mmr",
        type=int,
        default=3000,
        help="Minimum average MMR for All Pick matches (OpenDota only)",
    )
    parser.add_argument(
        "--max-scan-pages",
        type=int,
        default=200,
        help="Max public match pages to scan during discovery (OpenDota only)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Requests per second (OpenDota only)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent detail requests (OpenDota only)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenDota API key (or set OPENDOTA_API_KEY env var)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "stratz":
        # Stratz enhancement mode
        stratz_key = args.stratz_key or os.environ.get("STRATZ_API_KEY")
        if not stratz_key:
            print(
                "ERROR: Stratz API key required. Use --stratz-key or set "
                "STRATZ_API_KEY env var."
            )
            return

        print("=" * 60)
        print("STRATZ GRAPHQL DATA ENHANCER")
        print("=" * 60)
        print("Enhancing existing matches with Stratz data")
        print(f"Max matches: {args.stratz_max}")
        print(f"Output: {output_dir}")
        print()

        # Get list of downloaded matches to enhance
        downloaded_ids = [
            int(f.stem)
            for f in output_dir.glob("*.json")
            if f.name not in ("metadata.json", "stratz_metadata.json")
            and f.stem.isdigit()
        ]

        # Limit to specified max
        to_enhance = downloaded_ids[: args.stratz_max]

        async with StratzCollector(
            api_key=stratz_key,
            rate_limit=args.stratz_rate_limit,
            max_concurrent=5,
        ) as collector:
            result = await collector.enhance_matches(
                match_ids=to_enhance,
                output_dir=output_dir,
                resume=not args.no_resume,
            )

        print(f"\n{'=' * 60}")
        print("ENHANCEMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"Matches enhanced: {result.get('total_enhanced', 0)}")
        print(f"Failed: {result.get('failed', 0)}")
        print(f"Enhanced data saved to: {result.get('enhanced_dir')}")

    else:
        # OpenDota collection
        args.api_key = args.api_key or os.environ.get("OPENDOTA_API_KEY")

        allpick_target = args.allpick_matches
        if allpick_target is None and args.include_ranked:
            allpick_target = max(1000, args.matches // 2)
            print(f"NOTE: --allpick-matches not set, defaulting to {allpick_target}")
        if allpick_target is None:
            allpick_target = 0

        print("=" * 60)
        print("OPENDOTA DATA COLLECTOR")
        print("=" * 60)
        print(f"Target training matches: {args.matches}")
        print(f"Target All Pick matches: {allpick_target}")
        print(f"Patches: {args.patches}")
        print(f"Pro matches: {args.include_pro}")
        print(f"Ranked matches: {args.include_ranked}")
        print(f"Captains MMR: {args.min_mmr}")
        print(f"All Pick MMR: {args.allpick_mmr}")
        print(f"Output: {output_dir}")
        print(f"Resume: {not args.no_resume}")
        print()

        async with OpenDotaCollector(
            api_key=args.api_key,
            rate_limit=args.rate_limit,
            max_concurrent=args.max_concurrent,
        ) as collector:
            metadata = await collector.collect_matches_paginated(
                target_count=args.matches,
                output_dir=output_dir,
                patches=args.patches,
                include_pro=args.include_pro,
                include_ranked=args.include_ranked,
                min_mmr=args.min_mmr,
                allpick_target=allpick_target,
                allpick_mmr=args.allpick_mmr,
                max_scan_pages=args.max_scan_pages,
                resume=not args.no_resume,
            )

        print(f"\n{'=' * 60}")
        print("COLLECTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Training matches: {len(metadata.get('match_ids', []))}")
        print(f"All Pick matches: {len(metadata.get('allpick_ids', []))}")
        print(f"Successfully fetched: {metadata.get('successful', 0)}")
        print(f"Failed: {metadata.get('failed', 0)}")
        print(f"API requests: {metadata.get('requests_made', 0)}")
        print(f"Data saved to: {output_dir}")
        print(f"All Pick details: {output_dir / 'allpick'}")


if __name__ == "__main__":
    asyncio.run(main())
