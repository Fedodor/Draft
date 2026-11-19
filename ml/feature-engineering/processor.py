#!/usr/bin/env python3
"""
Enhanced Feature Processor for Smart Draft Assistant
====================================================

Converts raw match JSON into ML-ready features (Parquet format) with:
- Hero presence (one-hot)
- Semantic team aggregates
- Draft balance features
- Pick order features
- Meta features
- Patch awareness

Usage:
    python processor.py --raw-dir ml/datasets/raw --output-dir ml/datasets/processed
"""

import hashlib
import json
import shutil
import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys

ML_ROOT = Path(__file__).resolve().parent.parent

# Column blocks of the emitted dataset. `META_COLUMNS` is emitted first by
# `process_match` and is pinned to the front of every generated dataset.
META_COLUMNS: Tuple[str, ...] = (
    'match_id', 'start_time', 'patch', 'radiant_win',
    'duration', 'avg_mmr', 'game_mode', 'lobby_type',
)
# Meta columns stored as integers (everything else is numeric float64).
INT_META_COLUMNS = frozenset({
    'match_id', 'start_time', 'radiant_win',
    'duration', 'avg_mmr', 'game_mode', 'lobby_type',
})
# Sparse win-rate columns whose position used to depend on the first match
# that happened to contain them (row-completion order in the old pipeline).
PER_HERO_WINRATE_PREFIXES: Tuple[str, ...] = ('rad_wr_', 'dire_wr_')
BAN_AVG_COLUMNS: Tuple[str, ...] = (
    'radiant_ban_avg_winrate', 'dire_ban_avg_winrate',
)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stable SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """Convert stats values (sets, numpy scalars) into JSON-serializable data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _sorted_patch_values(values) -> List[str]:
    """Deterministic patch ordering (numeric by component, 'unknown' last)."""
    cleaned = {str(value) for value in values if value is not None and str(value) != 'nan'}

    def _key(text: str):
        if text.lower() == 'unknown':
            return (1, ())
        try:
            return (0, tuple(int(part) for part in text.split('.')))
        except ValueError:
            return (0, ())

    return sorted(cleaned, key=_key)


def canonical_column_order(
    columns,
    hero_id_by_column: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Return a deterministic, documented column order for a dataset.

    Rule: meta block (code order) -> structural features in their code-defined
    emission order -> the two sparse ban-average columns -> per-hero win-rate
    columns sorted by (side, hero id). Structural features are emitted for
    every valid match, so their first appearance is their code order and does
    not depend on which match is processed first; the win-rate block is sorted
    explicitly because it is sparse.

    This order is a property of a *new* dataset. It is intentionally not the
    shipped 887-feature order (which is pinned by the bundled schema and the
    ONNX artifact) and must be exported as its own candidate schema.
    """
    hero_id_by_column = hero_id_by_column or {}
    present = list(dict.fromkeys(columns))
    known = set(present)

    meta = [name for name in META_COLUMNS if name in known]
    rest = [name for name in present if name not in META_COLUMNS]
    winrates = [name for name in rest if name.startswith(PER_HERO_WINRATE_PREFIXES)]
    ban_avg = [name for name in BAN_AVG_COLUMNS if name in known]
    structural = [name for name in rest if name not in winrates and name not in ban_avg]

    def _winrate_key(name: str):
        side = 0 if name.startswith('rad_wr_') else 1
        return (side, hero_id_by_column.get(name, 10 ** 9), name)

    return meta + structural + ban_avg + sorted(winrates, key=_winrate_key)

# OpenDota patch ID -> version string (from dotaconstants patch.json)
# Also includes Stratz gameVersionId mappings
PATCH_MAP = {
    # OpenDota IDs
    6: "6.76", 7: "6.77", 8: "6.78", 9: "6.79", 10: "6.80", 11: "6.81",
    12: "6.82", 13: "6.83", 14: "6.84", 15: "6.85", 16: "6.86", 17: "6.87",
    18: "6.88", 19: "7.00", 20: "7.01", 21: "7.02", 22: "7.03", 23: "7.04",
    24: "7.05", 25: "7.06", 26: "7.07", 27: "7.08", 28: "7.09", 29: "7.10",
    30: "7.11", 31: "7.12", 32: "7.13", 33: "7.14", 34: "7.15", 35: "7.16",
    36: "7.17", 37: "7.18", 38: "7.19", 39: "7.20", 40: "7.21", 41: "7.22",
    42: "7.23", 43: "7.24", 44: "7.25", 45: "7.26", 46: "7.27", 47: "7.28",
    48: "7.29", 49: "7.30", 50: "7.31", 51: "7.32", 52: "7.33", 53: "7.34",
    54: "7.35", 55: "7.36", 56: "7.37", 57: "7.38", 58: "7.39", 59: "7.40",
    60: "7.41",
    # Stratz gameVersionId mappings (from Stratz GraphQL API)
    182: "7.40", 181: "7.40", 180: "7.39", 179: "7.38",     178: "7.37", 177: "7.37", 176: "7.37", 175: "7.36", 173: "7.36", 172: "7.35", 171: "7.35",
    170: "7.35", 169: "7.35", 168: "7.34", 167: "7.34", 166: "7.34",
    165: "7.34", 164: "7.34", 163: "7.33", 162: "7.33", 161: "7.33",
    160: "7.33", 159: "7.33", 158: "7.32", 157: "7.32", 156: "7.32",
    155: "7.32", 154: "7.32", 153: "7.31", 152: "7.31", 151: "7.31",
    150: "7.31", 149: "7.30", 148: "7.30", 147: "7.30", 146: "7.30",
    145: "7.30", 144: "7.29", 143: "7.29", 142: "7.29", 141: "7.29",
    140: "7.28", 139: "7.28", 138: "7.28", 137: "7.28", 136: "7.27",
    135: "7.27", 134: "7.27", 133: "7.27", 132: "7.27", 131: "7.26",
    130: "7.26", 129: "7.26", 128: "7.25", 127: "7.25", 126: "7.24",
    125: "7.24", 124: "7.23", 123: "7.23", 122: "7.23", 121: "7.23",
    120: "7.23", 119: "7.23", 118: "7.23", 117: "7.22", 116: "7.22",
    115: "7.22", 114: "7.22", 113: "7.22", 112: "7.22", 111: "7.22",
    110: "7.22", 109: "7.21", 108: "7.21", 107: "7.21", 106: "7.21",
    105: "7.20", 104: "7.20", 103: "7.20", 102: "7.20", 101: "7.20",
    100: "7.19", 99: "7.19", 98: "7.19", 97: "7.19", 96: "7.19",
    95: "7.18", 94: "7.17", 93: "7.16", 92: "7.15", 91: "7.14",
    90: "7.13", 89: "7.12", 88: "7.11", 87: "7.10", 86: "7.09",
    85: "7.08", 84: "7.07", 83: "7.07", 82: "7.07", 81: "7.07",
    80: "7.06", 79: "7.06", 78: "7.06", 77: "7.06", 76: "7.06",
    75: "7.06", 74: "7.05", 73: "7.04", 72: "7.03", 71: "7.02",
    70: "7.01", 69: "7.00", 68: "6.88", 67: "6.88", 66: "6.88",
}

# Hand-tuned semantics keyed by hero NAME (applied to the correct IDs from the
# hero roster). Values describe initiation/stun/save/push/scaling etc. in [0,1].
NAMED_SEMANTICS = {
    "Axe": {"initiation": 0.9, "stun": 0.8, "save": 0.3, "push": 0.2, "scaling": 0.6, "mobility": 0.4, "durability": 0.9, "teamfight": 0.95, "splitpush": 0.2, "vision": 0.1, "magic_damage": 0.4, "physical_damage": 0.8, "pure_damage": 0.1, "disable_count": 2.0, "lane_early": 0.8, "lane_mid": 0.7, "lane_late": 0.6, "tempo": "early"},
    "Pudge": {"initiation": 0.85, "stun": 0.7, "save": 0.2, "push": 0.1, "scaling": 0.7, "mobility": 0.3, "durability": 0.85, "teamfight": 0.8, "splitpush": 0.1, "vision": 0.0, "magic_damage": 0.5, "physical_damage": 0.7, "pure_damage": 0.2, "disable_count": 1.5, "lane_early": 0.6, "lane_mid": 0.7, "lane_late": 0.8, "tempo": "mid"},
    "Riki": {"initiation": 0.3, "stun": 0.4, "save": 0.1, "push": 0.2, "scaling": 0.8, "mobility": 0.9, "durability": 0.3, "teamfight": 0.5, "splitpush": 0.6, "vision": 0.0, "magic_damage": 0.1, "physical_damage": 0.9, "pure_damage": 0.0, "disable_count": 0.5, "lane_early": 0.5, "lane_mid": 0.7, "lane_late": 0.9, "tempo": "late"},
    "Crystal Maiden": {"initiation": 0.2, "stun": 0.6, "save": 0.4, "push": 0.3, "scaling": 0.4, "mobility": 0.3, "durability": 0.2, "teamfight": 0.6, "splitpush": 0.2, "vision": 0.3, "magic_damage": 0.9, "physical_damage": 0.1, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.5, "lane_late": 0.3, "tempo": "early"},
    "Shadow Fiend": {"initiation": 0.3, "stun": 0.2, "save": 0.1, "push": 0.4, "scaling": 0.9, "mobility": 0.3, "durability": 0.3, "teamfight": 0.7, "splitpush": 0.5, "vision": 0.0, "magic_damage": 0.3, "physical_damage": 0.8, "pure_damage": 0.0, "disable_count": 0.0, "lane_early": 0.6, "lane_mid": 0.8, "lane_late": 0.9, "tempo": "late"},
    "Zeus": {"initiation": 0.2, "stun": 0.3, "save": 0.1, "push": 0.2, "scaling": 0.7, "mobility": 0.3, "durability": 0.2, "teamfight": 0.8, "splitpush": 0.1, "vision": 0.0, "magic_damage": 1.0, "physical_damage": 0.0, "pure_damage": 0.0, "disable_count": 0.0, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.6, "tempo": "mid"},
    "Lina": {"initiation": 0.3, "stun": 0.4, "save": 0.2, "push": 0.3, "scaling": 0.8, "mobility": 0.4, "durability": 0.3, "teamfight": 0.8, "splitpush": 0.4, "vision": 0.0, "magic_damage": 0.8, "physical_damage": 0.4, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.7, "tempo": "mid"},
    "Lich": {"initiation": 0.2, "stun": 0.5, "save": 0.3, "push": 0.2, "scaling": 0.5, "mobility": 0.3, "durability": 0.3, "teamfight": 0.7, "splitpush": 0.2, "vision": 0.0, "magic_damage": 0.9, "physical_damage": 0.1, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.8, "lane_mid": 0.6, "lane_late": 0.4, "tempo": "early"},
    "Lion": {"initiation": 0.4, "stun": 0.8, "save": 0.2, "push": 0.1, "scaling": 0.5, "mobility": 0.3, "durability": 0.2, "teamfight": 0.8, "splitpush": 0.1, "vision": 0.0, "magic_damage": 0.9, "physical_damage": 0.2, "pure_damage": 0.0, "disable_count": 2.0, "lane_early": 0.7, "lane_mid": 0.6, "lane_late": 0.4, "tempo": "early"},
    "Vengeful Spirit": {"initiation": 0.7, "stun": 0.5, "save": 0.4, "push": 0.2, "scaling": 0.6, "mobility": 0.5, "durability": 0.4, "teamfight": 0.7, "splitpush": 0.2, "vision": 0.0, "magic_damage": 0.5, "physical_damage": 0.6, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.6, "lane_mid": 0.7, "lane_late": 0.6, "tempo": "mid"},
    "Windranger": {"initiation": 0.3, "stun": 0.5, "save": 0.4, "push": 0.3, "scaling": 0.7, "mobility": 0.7, "durability": 0.4, "teamfight": 0.7, "splitpush": 0.3, "vision": 0.0, "magic_damage": 0.4, "physical_damage": 0.6, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.6, "lane_mid": 0.7, "lane_late": 0.8, "tempo": "mid"},
    "Shadow Shaman": {"initiation": 0.2, "stun": 0.7, "save": 0.2, "push": 0.6, "scaling": 0.5, "mobility": 0.3, "durability": 0.3, "teamfight": 0.7, "splitpush": 0.5, "vision": 0.0, "magic_damage": 0.8, "physical_damage": 0.2, "pure_damage": 0.0, "disable_count": 2.0, "lane_early": 0.7, "lane_mid": 0.6, "lane_late": 0.4, "tempo": "early"},
    "Tidehunter": {"initiation": 0.95, "stun": 0.7, "save": 0.2, "push": 0.2, "scaling": 0.7, "mobility": 0.3, "durability": 0.9, "teamfight": 0.95, "splitpush": 0.1, "vision": 0.0, "magic_damage": 0.5, "physical_damage": 0.6, "pure_damage": 0.0, "disable_count": 1.5, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.8, "tempo": "mid"},
    "Witch Doctor": {"initiation": 0.2, "stun": 0.4, "save": 0.3, "push": 0.2, "scaling": 0.5, "mobility": 0.3, "durability": 0.2, "teamfight": 0.8, "splitpush": 0.1, "vision": 0.0, "magic_damage": 0.9, "physical_damage": 0.1, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.6, "lane_late": 0.4, "tempo": "early"},
    "Luna": {"initiation": 0.3, "stun": 0.2, "save": 0.1, "push": 0.5, "scaling": 0.9, "mobility": 0.4, "durability": 0.3, "teamfight": 0.8, "splitpush": 0.6, "vision": 0.0, "magic_damage": 0.3, "physical_damage": 0.9, "pure_damage": 0.0, "disable_count": 0.0, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.9, "tempo": "late"},
    "Dragon Knight": {"initiation": 0.6, "stun": 0.5, "save": 0.2, "push": 0.5, "scaling": 0.8, "mobility": 0.3, "durability": 0.9, "teamfight": 0.8, "splitpush": 0.3, "vision": 0.0, "magic_damage": 0.2, "physical_damage": 0.9, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.8, "lane_mid": 0.8, "lane_late": 0.9, "tempo": "late"},
    "Drow Ranger": {"initiation": 0.2, "stun": 0.5, "save": 0.1, "push": 0.3, "scaling": 0.9, "mobility": 0.3, "durability": 0.2, "teamfight": 0.6, "splitpush": 0.4, "vision": 0.0, "magic_damage": 0.1, "physical_damage": 1.0, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.9, "tempo": "late"},
    "Juggernaut": {"initiation": 0.4, "stun": 0.2, "save": 0.3, "push": 0.4, "scaling": 0.8, "mobility": 0.7, "durability": 0.4, "teamfight": 0.8, "splitpush": 0.3, "vision": 0.0, "magic_damage": 0.2, "physical_damage": 0.9, "pure_damage": 0.0, "disable_count": 0.0, "lane_early": 0.8, "lane_mid": 0.8, "lane_late": 0.9, "tempo": "mid"},
    "Mirana": {"initiation": 0.5, "stun": 0.5, "save": 0.4, "push": 0.3, "scaling": 0.7, "mobility": 0.7, "durability": 0.3, "teamfight": 0.7, "splitpush": 0.3, "vision": 0.0, "magic_damage": 0.4, "physical_damage": 0.7, "pure_damage": 0.0, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.8, "lane_late": 0.7, "tempo": "mid"},
    "Omniknight": {"initiation": 0.3, "stun": 0.3, "save": 0.8, "push": 0.2, "scaling": 0.6, "mobility": 0.3, "durability": 0.7, "teamfight": 0.7, "splitpush": 0.1, "vision": 0.0, "magic_damage": 0.3, "physical_damage": 0.7, "pure_damage": 0.1, "disable_count": 1.0, "lane_early": 0.7, "lane_mid": 0.7, "lane_late": 0.6, "tempo": "mid"},
}


def resolve_matchup_hero_id(value: Any, name_to_id: Dict[str, int]) -> Optional[int]:
    """Normalize a matchup hero reference to a numeric hero id.

    The matchup writer changed shape over time: current parquet files store
    numeric ``hero_a``/``hero_b`` ids (also as floats once pandas widens the
    column), while older exports stored normalized hero names, sometimes
    prefixed with ``_``. Both shapes are accepted so the matchup block can be
    regenerated from either source.
    """
    if value is None:
        return None

    # pandas represents a missing scalar as NaN/NA.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        text = value.strip().lstrip('_')
        if not text:
            return None
        try:
            numeric = int(float(text))
        except ValueError:
            return name_to_id.get(text.lower())
    else:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None

    return numeric if numeric > 0 else None


class FeatureProcessor:
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        include_bans: bool = True,
        include_semantics: bool = True,
        include_meta: bool = True,
        include_winrates: bool = True,
        include_matchups: bool = True,
        include_allpick: bool = True,
        lookup_dir: Optional[Path] = None,
        workers: int = 8,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.include_bans = include_bans
        self.include_semantics = include_semantics
        self.include_meta = include_meta
        self.include_winrates = include_winrates
        # Matchup features append seven columns and change the 887-feature
        # contract, so they are opt-in rather than implied by file presence.
        self.include_matchups = include_matchups
        # Recomputing the All Pick win-rate/matchup tables rewrites
        # `<output_dir>/hero_winrates.parquet`. A regeneration that must reuse a
        # frozen table sets this to False and points `lookup_dir` at it.
        self.include_allpick = include_allpick
        # Derived lookup tables (win rates / matchups) are read from here. It
        # defaults to the output directory for backward compatibility, but a
        # regeneration into a fresh directory should point this at the frozen
        # tables so feature values do not silently fall back to 0.5.
        self.lookup_dir = Path(lookup_dir) if lookup_dir else self.output_dir
        self.workers = max(1, int(workers))
        self.stats: Dict[str, Any] = self._empty_stats()

        self.hero_semantics = {}
        self.hero_names = {}
        self.hero_matchups = {}  # matchup[hero_a][hero_b] = winrate
        self.all_hero_ids = set(range(1, 140))
        self._load_hero_data()
        self._load_hero_winrates()
        self._load_hero_matchups()

        self._safe_names = {
            hid: self.hero_names.get(hid, f"hero_{hid}")
                .lower().replace(' ', '_').replace("'", '').replace('-', '_')
            for hid in self.all_hero_ids
        }

        self._sem_keys = [
            'stun', 'initiation', 'save', 'push', 'scaling', 'mobility',
            'durability', 'teamfight', 'splitpush', 'vision',
            'magic_damage', 'physical_damage', 'pure_damage',
            'disable_count', 'lane_early', 'lane_mid', 'lane_late'
        ]
    
    def _load_hero_data(self) -> None:
        """Build hero name/semantic maps from roster, applying hand-tuned overrides by name."""
        hero_data_file = ML_ROOT / "datasets" / "heroes.json"
        if not hero_data_file.exists():
            print("Warning: hero roster not found; using default semantics")
            return
        try:
            with open(hero_data_file) as f:
                roster = json.load(f)
            heroes = roster.values() if isinstance(roster, dict) else roster
            for hero in heroes:
                hid = int(hero.get("id", 0))
                if hid <= 0:
                    continue
                name = hero.get("localized_name")
                if not name:
                    continue
                self.hero_names.setdefault(hid, name)
                override = NAMED_SEMANTICS.get(name)
                if override is not None:
                    self.hero_semantics[hid] = dict(override)
                else:
                    self.hero_semantics.setdefault(hid, self._semantics_from_roles(
                        hero.get("roles", []),
                        hero.get("attack_type", ""),
                        hero.get("primary_attr", "")
                    ))
        except Exception as e:
            print(f"Warning: failed to load hero data ({e}); using default semantics")
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Counters that describe a processing run (never used as features)."""
        return {
            "matches_seen": 0,
            "matches_processed": 0,
            "skipped_no_draft": 0,
            "skipped_malformed_draft": 0,
            "skipped_short_draft": 0,
            "skipped_error": 0,
            "entries_empty_slot": 0,
            "entries_missing_is_pick": 0,
            "entries_bad_is_pick": 0,
            "entries_bad_hero_id": 0,
            "duplicate_hero_in_draft": 0,
            "hero_picked_by_both_teams": 0,
            "pick_counts_radiant": {},
            "pick_counts_dire": {},
            "ban_counts_radiant": {},
            "ban_counts_dire": {},
            "patch_values": set(),
            "patch_version_values": set(),
        }

    def _bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def _bump_histogram(self, key: str, value: int) -> None:
        bucket = self.stats.setdefault(key, {})
        label = str(int(value))
        bucket[label] = bucket.get(label, 0) + 1

    def _load_hero_winrates(self) -> None:
        """Load hero win rates from processed All Pick data."""
        self.hero_winrates = {}
        winrate_file = self.lookup_dir / "hero_winrates.parquet"
        if not winrate_file.exists():
            return
        try:
            df = pd.read_parquet(winrate_file)
            ids = df['hero_id'].values.astype(int)
            wrs = df['winrate'].values.astype(float)
            games = df['games'].values.astype(int)
            mask = ids > 0
            for hid, wr, g in zip(ids[mask], wrs[mask], games[mask]):
                self.hero_winrates[int(hid)] = {'winrate': float(wr), 'games': int(g)}
            print(f"Loaded win rates for {len(self.hero_winrates)} heroes")
        except Exception as e:
            print(f"Warning: failed to load hero win rates ({e})")
    
    def _load_hero_matchups(self) -> None:
        """Load hero matchup winrates (hero_a vs hero_b).

        Supports both historical shapes:
        - the current writer: numeric ``hero_a``/``hero_b`` ids and
          ``hero_a_winrate``;
        - older exports: hero names in ``hero_a``/``hero_b`` and ``winrate``.
        """
        matchup_file = self.lookup_dir / "hero_matchups.parquet"
        if not matchup_file.exists():
            return
        try:
            df = pd.read_parquet(matchup_file)
            heroes_file = ML_ROOT / "datasets" / "heroes.json"
            name_to_id = {}
            if heroes_file.exists():
                with open(heroes_file) as f:
                    roster = json.load(f)
                for hid_str, hero in roster.items():
                    name = hero.get('localized_name', '').lower().replace(' ', '_').replace("'", '').replace('-', '_')
                    name_to_id[name] = int(hid_str)

            for _, row in df.iterrows():
                h_a = resolve_matchup_hero_id(row.get('hero_a'), name_to_id)
                h_b = resolve_matchup_hero_id(row.get('hero_b'), name_to_id)
                if h_a and h_b:
                    winrate = row.get('hero_a_winrate', row.get('winrate', 0.5))
                    if h_a not in self.hero_matchups:
                        self.hero_matchups[h_a] = {}
                    self.hero_matchups[h_a][h_b] = {
                        'winrate': float(winrate),
                        'games': int(row.get('games', 0)),
                    }
            print(f"Loaded matchup data for {len(self.hero_matchups)} heroes")
        except Exception as e:
            print(f"Warning: failed to load hero matchups ({e})")
    
    def _semantics_from_roles(self, roles: List[str], attack_type: str, primary_attr: str) -> Dict:
        """Estimate semantic attributes from hero roster roles/stats."""
        s = {
            "initiation": 0.3, "stun": 0.2, "save": 0.2, "push": 0.3,
            "scaling": 0.5, "mobility": 0.3, "durability": 0.4,
            "teamfight": 0.6, "splitpush": 0.3, "vision": 0.1,
            "magic_damage": 0.3, "physical_damage": 0.4, "pure_damage": 0.05,
            "disable_count": 0.5, "lane_early": 0.5, "lane_mid": 0.6,
            "lane_late": 0.55, "tempo": "mid",
        }
        has = set(str(r).title() for r in roles)
        
        if "Initiator" in has:
            s["initiation"] = 0.85
        if "Disabler" in has:
            s["stun"], s["disable_count"] = 0.75, s["disable_count"] + 1.0
        if "Durable" in has:
            s["durability"] = 0.85
        if "Carry" in has:
            s["scaling"] = 0.8
            s["physical_damage"] = max(s["physical_damage"], 0.65)
        if "Nuker" in has:
            s["magic_damage"] = 0.8
        if "Support" in has:
            s["save"] = 0.7
            s["vision"] = max(s["vision"], 0.5)
        if "Pusher" in has:
            s["push"] = 0.8
            s["splitpush"] = max(s["splitpush"], 0.5)
        if "Escape" in has:
            s["mobility"] = 0.85
            s["splitpush"] = max(s["splitpush"], 0.35)
        if "Jungler" in has:
            s["push"] = max(s["push"], 0.35)
        
        if has & {"Initiator", "Nuker", "Durable"}:
            s["teamfight"] = min(0.95, s["teamfight"] + 0.15)
        
        if "Carry" in has:
            s["lane_late"] += 0.30
            s["lane_early"] -= 0.15
            s["lane_mid"] += 0.10
        if "Support" in has:
            s["lane_early"] += 0.20
            s["lane_late"] -= 0.20
        if "Nuker" in has:
            s["lane_early"] += 0.10
            s["lane_late"] -= 0.10
        if "Pusher" in has:
            s["lane_mid"] += 0.05
        
        if "Support" in has and "Carry" not in has:
            s["tempo"] = "early"
        elif "Carry" in has and "Support" not in has:
            s["tempo"] = "late"
        elif "Initiator" in has or "Nuker" in has:
            s["tempo"] = "early"
        
        if primary_attr == "agi":
            s["physical_damage"] += 0.15
            s["scaling"] += 0.05
        elif primary_attr == "str":
            s["durability"] += 0.15
            s["initiation"] += 0.05
        elif primary_attr == "int":
            s["magic_damage"] += 0.10
        
        if attack_type == "Melee":
            s["durability"] += 0.05
            s["initiation"] += 0.05
        else:
            s["push"] += 0.05
        
        for k, v in s.items():
            if isinstance(v, float):
                s[k] = max(0.0, min(1.0, v))
        return s
    
    def get_semantics(self, hero_id: int) -> Dict:
        return self.hero_semantics.get(hero_id, {
            "initiation": 0.5, "stun": 0.5, "save": 0.5, "push": 0.5,
            "scaling": 0.5, "mobility": 0.5, "durability": 0.5,
            "teamfight": 0.5, "splitpush": 0.5, "vision": 0.5,
            "magic_damage": 0.5, "physical_damage": 0.5, "pure_damage": 0.0,
            "disable_count": 1.0, "lane_early": 0.5, "lane_mid": 0.5,
            "lane_late": 0.5, "tempo": "mid"
        })
    
    def safe_name(self, hero_id: int) -> str:
        return self._safe_names.get(hero_id, f"hero_{hero_id}")

    def _extract_player_lanes(self, match: Dict) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Return {hero_id: lane} per team from the match's players.

        Prefers STRATZ's explicit `lane` field; falls back to the OpenDota
        player_slot heuristic (0/128=safe, 1/129=mid, 2/130=off).
        """
        rad: Dict[int, str] = {}
        dire: Dict[int, str] = {}
        for p in (match.get('players') or []):
            hero_id = p.get('hero_id') or p.get('heroId')
            if not hero_id:
                continue
            lane = p.get('lane') or p.get('lane_role')
            if isinstance(lane, str) and lane:
                lane = lane.upper()
                label = ('safe' if 'SAFE' in lane
                         else 'mid' if 'MID' in lane
                         else 'off' if 'OFF' in lane
                         else 'other')
            else:
                slot = p.get('player_slot', p.get('playerSlot', -1))
                if slot is None or slot < 0:
                    continue
                pos = (slot if slot < 128 else slot - 128) % 5
                label = 'safe' if pos == 0 else 'mid' if pos == 1 else 'off' if pos == 2 else 'other'
            is_rad = p.get('isRadiant', p.get('is_radiant'))
            if is_rad is None:
                slot = p.get('player_slot', p.get('playerSlot', -1))
                is_rad = slot is not None and int(slot) < 128
            if is_rad:
                rad[hero_id] = label
            else:
                dire[hero_id] = label
        return rad, dire

    def _add_lane_features(self, features: Dict, rad_picks, dire_picks,
                           rad_lanes: Dict[int, str], dire_lanes: Dict[int, str]) -> None:
        for team, picks, lanes in [("rad", rad_picks, rad_lanes), ("dire", dire_picks, dire_lanes)]:
            counts = {"safe": 0, "mid": 0, "off": 0, "other": 0}
            for hero_id, _ in picks:
                label = lanes.get(hero_id, "other")
                counts[label] = counts.get(label, 0) + 1
            for lane in ("safe", "mid", "off", "other"):
                features[f"{team}_lane_{lane}"] = counts[lane]
    
    @property
    def hero_id_to_name(self):
        return self.hero_names
    
    def _convert_patch(self, patch) -> str:
        """Convert OpenDota patch ID to version string"""
        if patch is None:
            return "unknown"
        if isinstance(patch, int):
            return PATCH_MAP.get(patch, str(patch))
        return str(patch)
    
    def _resolve_is_pick(self, entry: Dict) -> Optional[bool]:
        """Resolve the pick/ban flag without guessing.

        OpenDota sends ``is_pick``; STRATZ sends ``isPick``. A missing flag is
        *not* assumed to mean "pick" (that is exactly how bans were silently
        counted as picks before 2026-10-28). Unknown entries are skipped and
        counted so a source-format change shows up in the run statistics
        instead of corrupting the presence vector.
        """
        raw = entry.get('is_pick', entry.get('isPick'))
        if raw is None:
            self._bump('entries_missing_is_pick')
            return None
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        if isinstance(raw, str) and raw.strip().lower() in ('true', 'false'):
            return raw.strip().lower() == 'true'
        self._bump('entries_bad_is_pick')
        return None

    @staticmethod
    def _resolve_team(entry: Dict) -> Optional[int]:
        """Return 0 for Radiant, 1 for Dire, or ``None`` when the side is unclear."""
        team = entry.get('team')
        if isinstance(team, bool):
            team = int(team)
        if isinstance(team, (int, float)) and team in (0, 1):
            return int(team)
        if isinstance(team, str):
            text = team.strip().lower()
            if text in ('radiant', '0'):
                return 0
            if text in ('dire', '1'):
                return 1
        is_radiant = entry.get('isRadiant')
        if isinstance(is_radiant, bool):
            return 0 if is_radiant else 1
        if isinstance(is_radiant, str) and is_radiant.strip().lower() in ('true', 'false'):
            return 0 if is_radiant.strip().lower() == 'true' else 1
        return None

    def process_match(
        self,
        match: Dict,
        *,
        allow_partial_draft: bool = False,
    ) -> Optional[Dict]:
        """Emit one feature row for `match`.

        `allow_partial_draft` is for simulation only. The dataset pipeline
        rejects drafts with fewer than five picks per side; the ablation harness
        needs the *same* feature code to describe a mid-draft snapshot, so it
        opts out of that quality gate explicitly. Dataset generation never sets
        it, so rejected matches stay rejected.
        """
        self._bump('matches_seen')
        try:
            if 'picks_bans' not in match and 'pickBans' not in match:
                self._bump('skipped_no_draft')
                return None

            radiant_picks = []
            dire_picks = []
            radiant_bans = []
            dire_bans = []

            pick_bans = match.get('picks_bans')
            if pick_bans is None:
                pick_bans = match.get('pickBans')
            if not isinstance(pick_bans, list) or not pick_bans:
                # STRATZ sends `"pickBans": null` for matches without a draft.
                self._bump('skipped_malformed_draft')
                return None

            for pb in pick_bans:
                if not isinstance(pb, dict):
                    self._bump('entries_bad_entry')
                    continue

                hero_id = pb.get('hero_id')
                if hero_id in (None, 0):
                    hero_id = pb.get('heroId')
                if hero_id in (None, 0):
                    # Empty ban slot; not an error.
                    self._bump('entries_empty_slot')
                    continue
                try:
                    hero_id = int(hero_id)
                except (TypeError, ValueError):
                    self._bump('entries_bad_hero_id')
                    continue
                if not 1 <= hero_id <= 255:
                    self._bump('entries_bad_hero_id')
                    continue

                team = self._resolve_team(pb)
                if team is None:
                    self._bump('entries_bad_team')
                    continue
                is_pick = self._resolve_is_pick(pb)
                if is_pick is None:
                    continue
                try:
                    order = int(pb.get('order') or 0)
                except (TypeError, ValueError):
                    order = 0

                if is_pick:
                    if team == 0:
                        radiant_picks.append((hero_id, order))
                    else:
                        dire_picks.append((hero_id, order))
                else:
                    if team == 0:
                        radiant_bans.append((hero_id, order))
                    else:
                        dire_bans.append((hero_id, order))

            radiant_picks.sort(key=lambda x: x[1])
            dire_picks.sort(key=lambda x: x[1])
            radiant_bans.sort(key=lambda x: x[1])
            dire_bans.sort(key=lambda x: x[1])

            rad_ids = [p[0] for p in radiant_picks]
            dire_ids = [p[0] for p in dire_picks]
            if len(rad_ids) > 5:
                self._bump('more_than_5_picks_radiant')
            if len(dire_ids) > 5:
                self._bump('more_than_5_picks_dire')
            if set(rad_ids) & set(dire_ids):
                self._bump('hero_picked_by_both_teams')
            all_draft_ids = rad_ids + dire_ids + [b[0] for b in radiant_bans] + [b[0] for b in dire_bans]
            if len(all_draft_ids) != len(set(all_draft_ids)):
                self._bump('duplicate_hero_in_draft')

            short_draft = len(radiant_picks) < 5 or len(dire_picks) < 5
            if short_draft and not allow_partial_draft:
                self._bump('skipped_short_draft')
                return None

            if not allow_partial_draft:
                # Counted after the gate so the histograms describe the rows that
                # actually reach the dataset (rejected short drafts would otherwise
                # inflate them and break reconciliation with `rows`).
                self._bump_histogram('pick_counts_radiant', len(radiant_picks))
                self._bump_histogram('pick_counts_dire', len(dire_picks))
                self._bump_histogram('ban_counts_radiant', len(radiant_bans))
                self._bump_histogram('ban_counts_dire', len(dire_bans))

            rad_pick_ids = {p[0] for p in radiant_picks}
            dire_pick_ids = {p[0] for p in dire_picks}
            rad_ban_ids = {b[0] for b in radiant_bans}
            dire_ban_ids = {b[0] for b in dire_bans}

            radiant_win = 1 if match.get('radiant_win', match.get('didRadiantWin', False)) else 0

            patch_value = match.get('patch') or match.get('gameVersionId')

            gm = match.get('game_mode') or match.get('gameMode', 0)
            if isinstance(gm, str):
                gm = 22 if 'CAPTAIN' in gm.upper() else 0

            features = {
                'match_id': match.get('match_id', match.get('id')),
                # Epoch seconds; used for chronological train/val/test splits.
                'start_time': match.get('start_time', match.get('startDateTime', 0)),
                'patch': self._convert_patch(patch_value),
                'radiant_win': radiant_win,
                'duration': match.get('duration', match.get('durationSeconds', 0)),
                'avg_mmr': match.get('avg_mmr', 0),
                'game_mode': gm,
                'lobby_type': match.get('lobby_type', 0),
            }

            # Lane-assignment features (experimental; disabled for main training parquet
            # to avoid distribution shift - enable when lane-enriched CM data available)
            # rad_lanes, dire_lanes = self._extract_player_lanes(match)
            # self._add_lane_features(features, radiant_picks, dire_picks, rad_lanes, dire_lanes)

            for hero_id in self.all_hero_ids:
                sn = self._safe_names[hero_id]
                features[f'rad_has_{sn}'] = 1 if hero_id in rad_pick_ids else 0
                features[f'dire_has_{sn}'] = 1 if hero_id in dire_pick_ids else 0
                if self.include_bans:
                    features[f'rad_ban_{sn}'] = 1 if hero_id in rad_ban_ids else 0
                    features[f'dire_ban_{sn}'] = 1 if hero_id in dire_ban_ids else 0

            for i, (hero_id, _) in enumerate(radiant_picks[:5]):
                features[f'rad_pick_{i}'] = hero_id
            for i, (hero_id, _) in enumerate(dire_picks[:5]):
                features[f'dire_pick_{i}'] = hero_id

            if self.include_semantics:
                for team_name, picks in [('radiant', radiant_picks), ('dire', dire_picks)]:
                    n = len(picks)
                    if n == 0:
                        continue

                    totals = {k: 0.0 for k in self._sem_keys}

                    for hero_id, _ in picks:
                        sem = self.get_semantics(hero_id)
                        for k in self._sem_keys:
                            totals[k] += sem.get(k, 0.5)

                    for k in self._sem_keys:
                        features[f'{team_name}_avg_{k}'] = totals[k] / n

                for k in ['stun', 'initiation', 'save', 'push', 'scaling', 'durability',
                         'teamfight', 'magic_damage', 'physical_damage', 'disable_count']:
                    rad_key = f'radiant_avg_{k}'
                    dire_key = f'dire_avg_{k}'
                    if rad_key in features and dire_key in features:
                        features[f'{k}_diff'] = features[rad_key] - features[dire_key]

            if self.include_winrates and self.hero_winrates:
                rad_winrates = []
                dire_winrates = []
                for hero_id, _ in radiant_picks:
                    wr = self.hero_winrates.get(hero_id, {}).get('winrate', 0.5)
                    features[f'rad_wr_{self.safe_name(hero_id)}'] = wr
                    rad_winrates.append(wr)
                for hero_id, _ in dire_picks:
                    wr = self.hero_winrates.get(hero_id, {}).get('winrate', 0.5)
                    features[f'dire_wr_{self.safe_name(hero_id)}'] = wr
                    dire_winrates.append(wr)

                if rad_winrates:
                    features['radiant_avg_winrate'] = sum(rad_winrates) / len(rad_winrates)
                    features['radiant_max_winrate'] = max(rad_winrates)
                    features['radiant_min_winrate'] = min(rad_winrates)
                if dire_winrates:
                    features['dire_avg_winrate'] = sum(dire_winrates) / len(dire_winrates)
                    features['dire_max_winrate'] = max(dire_winrates)
                    features['dire_min_winrate'] = min(dire_winrates)

                if rad_winrates and dire_winrates:
                    features['winrate_diff'] = (sum(rad_winrates) / len(rad_winrates)) - (sum(dire_winrates) / len(dire_winrates))

                rad_ban_wr = [self.hero_winrates.get(b[0], {}).get('winrate', 0.5) for b in radiant_bans]
                dire_ban_wr = [self.hero_winrates.get(b[0], {}).get('winrate', 0.5) for b in dire_bans]
                if rad_ban_wr:
                    features['radiant_ban_avg_winrate'] = sum(rad_ban_wr) / len(rad_ban_wr)
                if dire_ban_wr:
                    features['dire_ban_avg_winrate'] = sum(dire_ban_wr) / len(dire_ban_wr)

            if self.include_matchups and self.hero_matchups:
                rad_matchup_wrs = []
                dire_matchup_wrs = []
                for rad_h, _ in radiant_picks:
                    vs_dire = []
                    for dire_h, _ in dire_picks:
                        wr = self.hero_matchups.get(rad_h, {}).get(dire_h, {}).get('winrate', 0.5)
                        vs_dire.append(wr)
                    if vs_dire:
                        rad_matchup_wrs.append(sum(vs_dire) / len(vs_dire))

                for dire_h, _ in dire_picks:
                    vs_rad = []
                    for rad_h, _ in radiant_picks:
                        wr = self.hero_matchups.get(dire_h, {}).get(rad_h, {}).get('winrate', 0.5)
                        vs_rad.append(wr)
                    if vs_rad:
                        dire_matchup_wrs.append(sum(vs_rad) / len(vs_rad))

                if rad_matchup_wrs:
                    features['rad_matchup_avg'] = sum(rad_matchup_wrs) / len(rad_matchup_wrs)
                    features['rad_matchup_best'] = max(rad_matchup_wrs)
                    features['rad_matchup_worst'] = min(rad_matchup_wrs)
                if dire_matchup_wrs:
                    features['dire_matchup_avg'] = sum(dire_matchup_wrs) / len(dire_matchup_wrs)
                    features['dire_matchup_best'] = max(dire_matchup_wrs)
                    features['dire_matchup_worst'] = min(dire_matchup_wrs)
                if rad_matchup_wrs and dire_matchup_wrs:
                    features['matchup_advantage'] = (sum(rad_matchup_wrs) / len(rad_matchup_wrs)) - (sum(dire_matchup_wrs) / len(dire_matchup_wrs))

            if len(radiant_picks) >= 2:
                rad_hero_ids = [p[0] for p in radiant_picks]
                synergy_scores = []
                for i in range(len(rad_hero_ids)):
                    for j in range(i + 1, len(rad_hero_ids)):
                        h1, h2 = rad_hero_ids[i], rad_hero_ids[j]
                        wr1 = self.hero_winrates.get(h1, {}).get('winrate', 0.5)
                        wr2 = self.hero_winrates.get(h2, {}).get('winrate', 0.5)
                        synergy_scores.append(wr1 * wr2)
                if synergy_scores:
                    features['rad_synergy_avg'] = sum(synergy_scores) / len(synergy_scores)
                    features['rad_synergy_max'] = max(synergy_scores)
                    features['rad_synergy_min'] = min(synergy_scores)

            if len(dire_picks) >= 2:
                dire_hero_ids = [p[0] for p in dire_picks]
                synergy_scores = []
                for i in range(len(dire_hero_ids)):
                    for j in range(i + 1, len(dire_hero_ids)):
                        h1, h2 = dire_hero_ids[i], dire_hero_ids[j]
                        wr1 = self.hero_winrates.get(h1, {}).get('winrate', 0.5)
                        wr2 = self.hero_winrates.get(h2, {}).get('winrate', 0.5)
                        synergy_scores.append(wr1 * wr2)
                if synergy_scores:
                    features['dire_synergy_avg'] = sum(synergy_scores) / len(synergy_scores)
                    features['dire_synergy_max'] = max(synergy_scores)
                    features['dire_synergy_min'] = min(synergy_scores)

            if self.include_semantics:
                for team_name, picks in [('radiant', radiant_picks), ('dire', dire_picks)]:
                    if len(picks) < 2:
                        continue
                    attrs = []
                    for hero_id, _ in picks:
                        sem = self.get_semantics(hero_id)
                        attrs.append(sem.get('magic_damage', 0.5))
                        attrs.append(sem.get('physical_damage', 0.5))
                    if attrs:
                        features[f'{team_name}_damage_diversity'] = max(attrs) - min(attrs)

            if self.include_meta:
                # Known defect (not changed here): STRATZ omits `patch`, so this
                # is 'unknown' for ~95% of rows even though `patch` resolves via
                # `gameVersionId`. Making it mirror `patch` would add a duplicate
                # of an existing feature, so it is documented rather than patched.
                features['patch_version'] = self._convert_patch(match.get('patch'))

            self.stats.setdefault('patch_values', set()).add(str(features.get('patch')))
            self.stats.setdefault('patch_version_values', set()).add(
                str(features.get('patch_version'))
            )
            self._bump('matches_processed')
            return features

        except Exception as e:
            self._bump('skipped_error')
            print(f"Error processing match {match.get('match_id')}: {e}")
            return None
    
    def canonical_match_files(self) -> List[Path]:
        """Raw files in a canonical, platform-independent order.

        Directory listing order is not guaranteed, and the previous pipeline
        appended rows in worker-completion order, so two runs over the same
        corpus could produce different row order and different feature-column
        order. Sorting by numeric match id (the file stem) makes the traversal
        reproducible; non-numeric names sort after numeric ones by name.
        """
        files = [f for f in self.raw_dir.glob("*.json")
                 if f.name not in ("metadata.json", "all_pick_matches.json")]

        def _key(path: Path):
            stem = path.stem
            if stem.isdigit():
                return (0, int(stem), path.name)
            return (1, 0, path.name)

        return sorted(files, key=_key)

    def _process_file(self, path: Path) -> Optional[Dict]:
        try:
            with open(path, encoding='utf-8') as handle:
                match = json.load(handle)
        except Exception as error:
            self._bump('skipped_error')
            print(f"Error reading {path.name}: {error}")
            return None
        if not isinstance(match, dict):
            self._bump('skipped_malformed_draft')
            return None
        return self.process_match(match)

    def _process_chunk(self, files: List[Path]) -> List[Dict]:
        """Process `files` in parallel but return results in canonical order."""
        if not files:
            return []
        results: List[Optional[Dict]] = [None] * len(files)
        with ThreadPoolExecutor(max_workers=min(self.workers, len(files))) as executor:
            futures = {
                executor.submit(self._process_file, path): index
                for index, path in enumerate(files)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as error:  # pragma: no cover - defensive
                    self._bump('skipped_error')
                    print(f"Error processing {files[index].name}: {error}")
        return [record for record in results if record is not None]

    @property
    def _hero_id_by_column(self) -> Dict[str, int]:
        """Map a per-hero win-rate column name to its hero id."""
        return {
            f'rad_wr_{token}': hero_id
            for hero_id, token in self._safe_names.items()
        } | {
            f'dire_wr_{token}': hero_id
            for hero_id, token in self._safe_names.items()
        }

    @staticmethod
    def _union_columns(part_paths: List[Path]) -> List[str]:
        """Column union in first-appearance order across canonical parts."""
        ordered: List[str] = []
        seen = set()
        for path in part_paths:
            schema = pq.read_schema(path)
            for name in schema.names:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered

    def _copy_lookup_tables(self) -> None:
        """Copy the frozen lookup tables into the output directory.

        A regeneration that reuses externally frozen tables must carry them so
        the produced directory is self-contained; otherwise a later run would
        silently fall back to the 0.5 win-rate default.
        """
        copied = 0
        for name in ('hero_winrates.parquet', 'hero_matchups.parquet'):
            source = self.lookup_dir / name
            target = self.output_dir / name
            if source.is_file() and source.resolve() != target.resolve():
                shutil.copyfile(source, target)
                copied += 1
                print(f"Copied lookup table {name} from {self.lookup_dir}")
        if copied == 0 and self.include_winrates and not (
            self.output_dir / 'hero_winrates.parquet'
        ).is_file():
            print(
                "WARNING: no hero_winrates.parquet available; win-rate features "
                "will use the 0.5 default"
            )

    def _write_dataset(
        self,
        part_paths: List[Path],
        ordered_columns: List[str],
        output_file: Path,
        categories: Dict[str, List[str]],
    ) -> None:
        """Stream the canonical parts into one parquet with an explicit schema."""
        schema = self._dataset_arrow_schema(ordered_columns, categories)
        writer = pq.ParquetWriter(output_file, schema, compression='snappy')
        try:
            for path in part_paths:
                frame = pd.read_parquet(path)
                frame = frame.reindex(columns=ordered_columns)
                for name in ordered_columns:
                    if name in categories:
                        frame[name] = frame[name].astype('object').where(
                            frame[name].notna(), categories[name][0] if categories[name] else 'unknown'
                        ).astype(
                            pd.CategoricalDtype(categories=categories[name], ordered=False)
                        )
                    elif name in INT_META_COLUMNS:
                        frame[name] = pd.to_numeric(frame[name], errors='coerce').fillna(0).astype('int64')
                    else:
                        frame[name] = pd.to_numeric(frame[name], errors='coerce').fillna(0).astype('float64')
                table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                writer.write_table(table)
                del frame, table
        finally:
            writer.close()
        print(f"Wrote {output_file}")

    @staticmethod
    def _dataset_arrow_schema(
        ordered_columns: List[str],
        categories: Dict[str, List[str]],
    ) -> "pa.Schema":
        fields = []
        for name in ordered_columns:
            if name in categories:
                fields.append(pa.field(name, pa.dictionary(pa.int32(), pa.string())))
            elif name in INT_META_COLUMNS:
                fields.append(pa.field(name, pa.int64()))
            else:
                fields.append(pa.field(name, pa.float64()))
        return pa.schema(fields)

    def process_all(
        self,
        limit: int = None,
        *,
        chunk_size: int = 1000,
        load_into_memory: bool = True,
        dataset_label: str = "candidate",
    ) -> Optional[pd.DataFrame]:
        chunk_size = max(1, int(chunk_size))
        match_files = self.canonical_match_files()
        match_files = [f for f in match_files if f.name not in ("metadata.json", "all_pick_matches.json")]

        if limit:
            match_files = match_files[:limit]

        print(f"Processing {len(match_files)} matches from {self.raw_dir}...")
        print(f"Chunk size: {chunk_size}, workers: {self.workers}")

        part_dir = self.output_dir / "_parts"
        if part_dir.exists():
            for stale in part_dir.glob("part_*.parquet"):
                stale.unlink()
        part_dir.mkdir(parents=True, exist_ok=True)

        part_paths: List[Path] = []
        rows_written = 0
        for chunk_index, start in enumerate(range(0, len(match_files), chunk_size)):
            chunk = match_files[start:start + chunk_size]
            records = self._process_chunk(chunk)
            if not records:
                continue
            frame = pd.DataFrame(records)
            # A chunk-local stable dtype lets the parts be concatenated without
            # schema negotiation; final dtypes are applied in `_write_dataset`.
            frame.to_parquet(part_dir / f"part_{chunk_index:05d}.parquet", index=False)
            part_paths.append(part_dir / f"part_{chunk_index:05d}.parquet")
            rows_written += len(frame)
            del records, frame

            if (chunk_index + 1) % 5 == 0:
                print(
                    f"  {min(start + chunk_size, len(match_files))}/{len(match_files)} "
                    f"files, {rows_written} rows"
                )

        if rows_written == 0:
            raise ValueError("No valid matches processed!")

        union_columns = self._union_columns(part_paths)
        ordered_columns = canonical_column_order(union_columns, self._hero_id_by_column)
        output_file = self.output_dir / "draft_features.parquet"
        categories = {
            'patch': _sorted_patch_values(self.stats.get('patch_values', set())),
            'patch_version': _sorted_patch_values(
                self.stats.get('patch_version_values', set())
            ),
        }
        self._write_dataset(part_paths, ordered_columns, output_file, categories)

        for part in part_paths:
            try:
                part.unlink()
            except OSError:
                pass
        try:
            part_dir.rmdir()
        except OSError:
            pass

        self.last_output = {
            'path': str(output_file),
            'rows': rows_written,
            'columns': len(ordered_columns),
            'feature_columns': len([c for c in ordered_columns if c not in META_COLUMNS]),
            'column_order': ordered_columns,
            'categories': categories,
        }
        self.write_dataset_schema(
            output_file,
            config={
                'chunk_size': chunk_size,
                'workers': self.workers,
                'limit': limit,
                'include_bans': self.include_bans,
                'include_semantics': self.include_semantics,
                'include_meta': self.include_meta,
                'include_winrates': self.include_winrates,
                'include_matchups': self.include_matchups,
                'include_allpick': self.include_allpick,
                'files_considered': len(match_files),
                'files_requested': len(self.canonical_match_files()),
            },
            dataset_label=dataset_label,
        )
        print(f"\nSaved {rows_written} samples with {len(ordered_columns)} columns to {output_file}")
        print(f"Processed: {self.stats.get('matches_processed', 0)}, "
              f"skipped: {self.stats.get('skipped_no_draft', 0)} no-draft, "
              f"{self.stats.get('skipped_malformed_draft', 0)} malformed, "
              f"{self.stats.get('skipped_short_draft', 0)} short, "
              f"{self.stats.get('skipped_error', 0)} errors")

        if self.include_allpick:
            self.process_all_pick_matches()
        else:
            self._copy_lookup_tables()

        if load_into_memory:
            return pd.read_parquet(output_file)
        return None
    
    def write_dataset_schema(
        self,
        output_file: Path,
        *,
        config: Optional[Dict[str, Any]] = None,
        dataset_label: str = "candidate",
    ) -> Path:
        """Write `dataset_schema.json` next to the parquet.

        The file is the explicit contract for a regenerated dataset: column
        order, dtypes, categories, feature fingerprint, provenance, and run
        statistics. It never overwrites the shipped model schema.
        """
        info = getattr(self, 'last_output', None) or {}
        columns = list(info.get('column_order') or [])
        feature_columns = [name for name in columns if name not in META_COLUMNS]
        lookup_files = {}
        for name in ('hero_winrates.parquet', 'hero_matchups.parquet'):
            candidate = self.lookup_dir / name
            if candidate.is_file():
                lookup_files[name] = {
                    'path': str(candidate),
                    'sha256': _file_sha256(candidate),
                }
        payload = {
            'schema_version': 1,
            'dataset': dataset_label,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'output': {
                'path': str(output_file),
                'sha256': _file_sha256(output_file),
                'rows': info.get('rows'),
                'columns': len(columns),
            },
            'meta_columns': [name for name in columns if name in META_COLUMNS],
            'feature_columns': feature_columns,
            'feature_count': len(feature_columns),
            'feature_fingerprint': hashlib.sha256(
                json.dumps(feature_columns, separators=(',', ':')).encode('utf-8')
            ).hexdigest(),
            'column_order_rule': (
                'meta block (code order) -> structural features (code emission order) '
                '-> ban averages -> per-hero win rates sorted by (side, hero id)'
            ),
            'column_order': columns,
            'categories': info.get('categories', {}),
            'source': {
                'raw_dir': str(self.raw_dir),
                'lookup_dir': str(self.lookup_dir),
                'lookup_files': lookup_files,
            },
            'stats': _json_safe(self.stats),
            'config': config or {},
        }
        path = Path(output_file).with_name('dataset_schema.json')
        path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        print(f"Wrote dataset schema: {path}")
        return path

    def process_all_pick_matches(self):
        allpick_dir = self.raw_dir / "allpick"
        matches = []
        source = None
        if allpick_dir.exists():
            files = sorted(allpick_dir.glob("*.json"))
            if files:
                source = allpick_dir
                def _read(f):
                    with open(f) as fh:
                        return json.load(fh)
                with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
                    for result in pool.map(_read, files):
                        try:
                            matches.append(result)
                        except Exception:
                            pass
        
        if not matches:
            all_pick_file = self.raw_dir / "all_pick_matches.json"
            if all_pick_file.exists():
                with open(all_pick_file) as f:
                    matches = json.load(f)
                source = all_pick_file
        
        if not matches:
            print("No All Pick match details found, skipping matchup processing")
            return
        
        print(f"Processing {len(matches)} All Pick matches from {source} "
              f"({len([m for m in matches if m.get('players')])} with players)")
        
        # Track hero vs hero matchups
        matchup_stats = {}  # (hero_a, hero_b) -> {'wins_a': int, 'games': int}
        hero_wins = {}      # hero_id -> {'wins': int, 'games': int}
        
        for match in matches:
            radiant_win = bool(match.get('radiant_win', match.get('didRadiantWin', False)))
            
            # Get hero IDs for each team
            radiant_heroes = []
            dire_heroes = []
            
            for player in (match.get('players') or []):
                hero_id = player.get('hero_id') or player.get('heroId')
                if not hero_id:
                    continue
                is_radiant = player.get('isRadiant', player.get('is_radiant'))
                if is_radiant is None:
                    # Fall back to player_slot: 0-4 radiant, 128-132 dire
                    is_radiant = player.get('player_slot', -1) < 128
                if is_radiant:
                    radiant_heroes.append(hero_id)
                else:
                    dire_heroes.append(hero_id)
            
            # Update individual hero stats
            for hero_id in radiant_heroes + dire_heroes:
                if hero_id not in hero_wins:
                    hero_wins[hero_id] = {'wins': 0, 'games': 0}
                hero_wins[hero_id]['games'] += 1
                if (hero_id in radiant_heroes and radiant_win) or (hero_id in dire_heroes and not radiant_win):
                    hero_wins[hero_id]['wins'] += 1
            
            # Track matchups between radiant and dire heroes
            for rad_hero in radiant_heroes:
                for dire_hero in dire_heroes:
                    pair = tuple(sorted([rad_hero, dire_hero]))
                    if pair not in matchup_stats:
                        matchup_stats[pair] = {'wins_a': 0, 'games': 0, 'hero_a': pair[0], 'hero_b': pair[1]}
                    matchup_stats[pair]['games'] += 1
                    # First hero in sorted pair is 'a'
                    if pair[0] == rad_hero and radiant_win:
                        matchup_stats[pair]['wins_a'] += 1
                    elif pair[0] == dire_hero and not radiant_win:
                        matchup_stats[pair]['wins_a'] += 1
        
        # Save matchup stats
        matchup_data = []
        for pair, stats in matchup_stats.items():
            if stats['games'] >= 10:  # Only save if enough games
                hero_a, hero_b = stats['hero_a'], stats['hero_b']
                winrate_a = stats['wins_a'] / stats['games']
                matchup_data.append({
                    'hero_a': hero_a,
                    'hero_b': hero_b,
                    'games': stats['games'],
                    'hero_a_winrate': winrate_a,
                    'hero_b_winrate': 1 - winrate_a,
                })
        
        matchup_file = self.output_dir / "hero_matchups.parquet"
        pd.DataFrame(matchup_data).to_parquet(matchup_file, index=False)
        print(f"Saved {len(matchup_data)} hero matchups to {matchup_file}")
        
        # Save hero win rates
        hero_stats = []
        for hero_id, stats in hero_wins.items():
            if stats['games'] >= 10:
                hero_stats.append({
                    'hero_id': hero_id,
                    'games': stats['games'],
                    'wins': stats['wins'],
                    'winrate': stats['wins'] / stats['games']
                })
        
        hero_file = self.output_dir / "hero_winrates.parquet"
        pd.DataFrame(hero_stats).to_parquet(hero_file, index=False)
        print(f"Saved {len(hero_stats)} hero win rates to {hero_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process raw Dota 2 matches into ML features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--raw-dir", type=str, default=str(ML_ROOT / "datasets" / "raw"),
                        help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default=str(ML_ROOT / "datasets" / "processed"),
                        help="Output directory for processed features")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of matches to process")
    parser.add_argument("--no-bans", action="store_true",
                        help="Exclude ban features")
    parser.add_argument("--no-semantics", action="store_true",
                        help="Exclude semantic features")
    parser.add_argument("--no-meta", action="store_true",
                        help="Exclude meta features")
    parser.add_argument("--no-matchups", action="store_true",
                        help="Exclude the seven matchup features (keeps the 887 contract)")
    parser.add_argument("--no-allpick", action="store_true",
                        help="Reuse frozen lookup tables instead of recomputing them")
    parser.add_argument("--lookup-dir", type=str, default=None,
                        help="Directory holding hero_winrates/hero_matchups (default: output dir)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Reader threads (results are reassembled in canonical order)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Matches per streamed part; bounds peak memory")
    parser.add_argument("--load-into-memory", action="store_true",
                        help="Read the finished parquet back and return it (small datasets only)")
    parser.add_argument("--dataset-label", type=str, default="candidate",
                        help="Label written into dataset_schema.json")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FEATURE PROCESSOR")
    print("="*60)
    print(f"Raw dir: {args.raw_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Lookup dir: {args.lookup_dir or args.output_dir}")
    print(f"Include bans: {not args.no_bans}")
    print(f"Include semantics: {not args.no_semantics}")
    print(f"Include meta: {not args.no_meta}")
    print(f"Include matchups: {not args.no_matchups}")
    print(f"Recompute all-pick tables: {not args.no_allpick}")
    print()
    
    processor = FeatureProcessor(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        include_bans=not args.no_bans,
        include_semantics=not args.no_semantics,
        include_meta=not args.no_meta,
        include_matchups=not args.no_matchups,
        include_allpick=not args.no_allpick,
        lookup_dir=Path(args.lookup_dir) if args.lookup_dir else None,
        workers=args.workers,
    )
    
    try:
        processor.process_all(
            limit=args.limit,
            chunk_size=args.chunk_size,
            load_into_memory=args.load_into_memory,
            dataset_label=args.dataset_label,
        )
        info = processor.last_output
        print(f"\nComplete! Wrote {info['rows']} rows x {info['columns']} columns "
              f"({info['feature_columns']} features)")
        print(f"Output: {info['path']}")
        print(f"Schema: {Path(info['path']).with_name('dataset_schema.json')}")
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()