"""Offline tests for the reconstructed OpenDota collector.

Network calls are replaced with fakes so the pagination, resume, and metadata
logic can be exercised without hitting the API.
"""

from __future__ import annotations

import asyncio
import json

from training.data_collector import OpenDotaCollector, StratzCollector


class FakeOpenDota(OpenDotaCollector):
    """OpenDotaCollector with deterministic, offline responses."""

    def __init__(self, pro_pages):
        super().__init__(rate_limit=0.0, max_concurrent=2)
        self._pro_pages = list(pro_pages)
        self.detail_calls = []

    async def get_pro_matches(self, limit=100, patch=None, less_than_match_id=None):
        if self._pro_pages:
            return self._pro_pages.pop(0)
        return []

    async def get_public_matches(self, limit=100, mmr=None, less_than_match_id=None):
        return []

    async def get_match_details(self, match_id):
        self.detail_calls.append(match_id)
        return {"match_id": match_id, "picks_bans": [{"hero_id": 1, "is_pick": True}]}


def test_imports_expose_expected_interfaces():
    for name in ("collect_matches_paginated", "get_match_details"):
        assert hasattr(OpenDotaCollector, name)
    assert OpenDotaCollector.BASE_URL.endswith("/api")
    assert StratzCollector.GRAPHQL_URL.endswith("/graphql")


def test_game_mode_helpers():
    assert OpenDotaCollector._is_captains_mode({"game_mode": 22})
    assert OpenDotaCollector._is_captains_mode({"game_mode": 23})
    assert not OpenDotaCollector._is_captains_mode({"game_mode": 1})
    assert OpenDotaCollector._is_all_pick({"game_mode": 1})
    assert not OpenDotaCollector._is_all_pick({"game_mode": 22})


def test_collect_discovers_and_downloads_matches(tmp_path):
    pages = [
        [{"match_id": 101}, {"match_id": 102}, {"match_id": 102}],
        [],
    ]
    collector = FakeOpenDota(pages)
    metadata = asyncio.run(
        collector.collect_matches_paginated(
            target_count=2,
            output_dir=tmp_path,
            patches=["7.41"],
            include_pro=True,
            include_ranked=False,
            resume=True,
        )
    )

    assert metadata["match_ids"] == [101, 102]
    assert metadata["successful"] == 2
    assert sorted(collector.detail_calls) == [101, 102]
    assert (tmp_path / "101.json").exists()
    assert (tmp_path / "102.json").exists()

    on_disk = json.loads((tmp_path / "metadata.json").read_text())
    assert on_disk["match_ids"] == [101, 102]
    assert on_disk["public_ids"] == []


def test_collect_without_discovery_still_writes_metadata(tmp_path):
    # include_pro=False and include_ranked=False previously raised NameError
    # because discovered_public was only defined inside the ranked branch.
    collector = FakeOpenDota([])
    metadata = asyncio.run(
        collector.collect_matches_paginated(
            target_count=5,
            output_dir=tmp_path,
            include_pro=False,
            include_ranked=False,
        )
    )

    assert metadata["match_ids"] == []
    assert metadata["public_ids"] == []
    assert (tmp_path / "metadata.json").exists()
