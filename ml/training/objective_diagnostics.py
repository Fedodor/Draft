#!/usr/bin/env python3
"""Objective feasibility study for the draft model (research only).

The 2026-10-28/29 experiments established that neither feature engineering nor
the model family lifts validation ROC-AUC above ~0.55. This script asks a
different question: **is full-match win probability the right target at all?**
It trains nothing. It audits the experiment, then runs cheap diagnostics on the
frozen validation partition to compare four candidate objectives:

    A. full-match win probability           (the current target)
    B. conditional draft-stage win          (predict at a decision point)
    C. next-pick recommendation ranking     (rank alternative heroes)
    D. pick-level marginal contribution     (per-pick effect)

What it measures (all on `processed_v2` + the frozen split config):

* true game mode per match, read from the raw STRATZ payloads, and the
  cross-tab against the shipped `game_mode` column (which mis-encodes it);
* per-mode validation metrics of the already-exported candidate, so the
  population the product actually operates in can be scored separately;
* how much of the partial-draft score comes from bans that had not happened
  yet: each stage is rebuilt twice, once with all bans (the harness
  convention) and once with only bans that preceded that stage in the real
  draft order;
* the spread of predicted probabilities within a stage, i.e. how much the
  score can possibly separate two candidate heroes;
* whether the corpus repeats draft contexts often enough to estimate
  next-pick effects, and how precisely a per-hero effect is identifiable.

The test partition is never read: `load_frozen_split` returns it and only its
size is recorded. No model is fitted, exported, or promoted.

Usage:
    python ml/training/objective_diagnostics.py
    python ml/training/objective_diagnostics.py --report ml/models/objective_study/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd

from training import ablation
from training.quality_gate import OnnxPredictor, baseline_metrics, compute_metrics
from training.splits import file_sha256, schema_fingerprint

DEFAULT_DATA = ML_ROOT / "datasets" / "processed_v2" / "draft_features.parquet"
DEFAULT_SPLIT_CONFIG = ML_ROOT / "configs" / "eval_split_v2.json"
DEFAULT_RAW_DIR = ML_ROOT / "datasets" / "raw"
DEFAULT_CANDIDATE = (
    ML_ROOT / "models" / "families" / "onnx" / "family_xgboost_combined"
)
DEFAULT_REPORT = ML_ROOT / "models" / "objective_study" / "diagnostics.json"
DEFAULT_MODE_CACHE = ML_ROOT / "models" / "objective_study" / "true_modes.tsv"

HEADER_BYTES = 4096
GAME_MODE_RE = re.compile(r'"gameMode"\s*:\s*"([A-Z_]+)"')
PICK_SLOTS = 5
TIMED_LEVELS = (0, 1, 2, 3)

# The stage table is taken from the recorded family comparison rather than
# recomputed: re-fitting would be a second sweep, and the saved report is the
# artefact of record for the 2026-10-29 session.
FAMILY_REPORT = ML_ROOT / "models" / "families" / "family_comparison_all_rows.json"


# --------------------------------------------------------------------------- #
# Raw-payload helpers
# --------------------------------------------------------------------------- #
def read_mode_cache(path: Path, wanted: Iterable[int]) -> Dict[int, str]:
    """Load `match_id<TAB>gameMode` rows produced by a previous scan."""
    wanted_set = {int(value) for value in wanted}
    modes: Dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stem, _, mode = line.rstrip("\n").partition("\t")
            if not mode:
                continue
            try:
                match_id = int(stem)
            except ValueError:
                continue
            if match_id in wanted_set:
                modes[match_id] = mode
    return modes


def scan_true_modes(raw_dir: Path, wanted: Iterable[int]) -> Dict[int, str]:
    """STRATZ `gameMode` enum per match id, read from the payload header.

    `gameMode` is one of the first keys in every STRATZ payload, so only the
    first few KB are read instead of parsing 100k JSON documents. OpenDota
    records have no `gameMode` key and are simply absent from the result; the
    caller treats them as "numeric OpenDota" and keeps their stored mode.

    The reads are threaded because a cold scan of 102k files in one directory
    is seek-bound, not CPU-bound. Results are cached by the caller: the first
    run of the study took over ten minutes, later runs read the cache.
    """
    wanted_set = {int(value) for value in wanted}
    names: List[str] = []
    for name in os.listdir(raw_dir):
        if not name.endswith(".json"):
            continue
        try:
            match_id = int(name[:-5])
        except ValueError:
            continue
        if match_id in wanted_set:
            names.append(name)
    print(f"  scanning {len(names)} payloads (cold read; cached for later runs)")

    def read_head(name: str) -> Tuple[int, str]:
        try:
            with open(raw_dir / name, "rb") as handle:
                head = handle.read(HEADER_BYTES).decode("utf-8", "ignore")
        except OSError:
            return -1, ""
        found = GAME_MODE_RE.search(head)
        return int(name[:-5]), (found.group(1) if found else "")

    modes: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        for match_id, mode in pool.map(read_head, names, chunksize=256):
            if match_id >= 0 and mode:
                modes[match_id] = mode
    return modes


def load_true_modes(
    raw_dir: Path,
    wanted: Iterable[int],
    *,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[int, str]:
    """True game mode per match, from the cache when available."""
    if cache_path is not None and cache_path.is_file() and not refresh:
        modes = read_mode_cache(cache_path, wanted)
        if modes:
            print(f"  mode cache: {cache_path} ({len(modes)} rows)")
            return modes
    modes = scan_true_modes(raw_dir, wanted)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8", newline="") as handle:
            for match_id in sorted(modes):
                handle.write(f"{match_id}\t{modes[match_id]}\n")
        print(f"  cached modes: {cache_path}")
    return modes


def load_ordered_drafts(raw_dir: Path, match_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """Ordered picks and bans per match, straight from the raw payload.

    The processed parquet stores picks in five fixed slots and bans as presence
    flags, so *when* a ban happened is lost. The partial-draft study needs the
    real order, so validation rows are re-read from the source. Entries with a
    null hero id (an unrevealed ban slot) are dropped, matching the processor.
    """
    drafts: Dict[int, Dict[str, Any]] = {}

    def read_draft(match_id: int):
        path = raw_dir / f"{int(match_id)}.json"
        if not path.is_file():
            return int(match_id), None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return int(match_id), None
        entries = payload.get("pickBans")
        if not isinstance(entries, list):
            return int(match_id), None
        rad_picks: List[Tuple[int, int]] = []
        dire_picks: List[Tuple[int, int]] = []
        rad_bans: List[Tuple[int, int]] = []
        dire_bans: List[Tuple[int, int]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            hero_id = entry.get("heroId")
            if not hero_id:
                continue
            try:
                hero_id = int(hero_id)
                order = int(entry["order"]) if entry.get("order") is not None else index
            except (TypeError, ValueError):
                continue
            radiant = entry.get("isRadiant")
            if not isinstance(radiant, bool):
                continue
            is_pick = entry.get("isPick")
            if not isinstance(is_pick, bool):
                continue
            if is_pick:
                (rad_picks if radiant else dire_picks).append((order, hero_id))
            else:
                (rad_bans if radiant else dire_bans).append((order, hero_id))
        rad_picks.sort()
        dire_picks.sort()
        rad_bans.sort()
        dire_bans.sort()
        return int(match_id), {
            "rad_picks": [hero for _, hero in rad_picks],
            "dire_picks": [hero for _, hero in dire_picks],
            "rad_bans": [hero for _, hero in rad_bans],
            "dire_bans": [hero for _, hero in dire_bans],
            "rad_pick_orders": [order for order, _ in rad_picks],
            "dire_pick_orders": [order for order, _ in dire_picks],
            "rad_ban_orders": [order for order, _ in rad_bans],
            "dire_ban_orders": [order for order, _ in dire_bans],
        }

    wanted = [int(value) for value in match_ids]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for match_id, draft in pool.map(read_draft, wanted, chunksize=64):
            if draft is not None:
                drafts[match_id] = draft
    return drafts


def bans_before_stage(draft: Mapping[str, Any], level: int) -> Tuple[List[int], List[int]]:
    """Bans that existed when `level` picks per side were known.

    `level == 0` means "just before the first pick". For k >= 1 the reference
    moment is the later of the two sides' k-th pick, so a ban counts only if it
    happened strictly before both teams had made that pick.
    """
    level = max(0, min(PICK_SLOTS, int(level)))
    rad_orders = draft["rad_pick_orders"]
    dire_orders = draft["dire_pick_orders"]
    if level == 0:
        if not rad_orders or not dire_orders:
            return [], []
        moment = min(min(rad_orders), min(dire_orders))
    else:
        if len(rad_orders) < level or len(dire_orders) < level:
            return [], []
        moment = max(rad_orders[level - 1], dire_orders[level - 1])

    def keep(heroes, orders):
        return [hero for hero, order in zip(heroes, orders) if order < moment]

    return (
        keep(draft["rad_bans"], draft["rad_ban_orders"]),
        keep(draft["dire_bans"], draft["dire_ban_orders"]),
    )


def build_timed_stage_frame(
    frame: pd.DataFrame,
    drafts: Mapping[int, Mapping[str, Any]],
    level: int,
    *,
    schema: Mapping[str, Any],
    processor,
) -> pd.DataFrame:
    """Rebuild rows with only the bans known at `level` picks per side.

    Mirrors `ablation.build_partial_draft_frame` (same production feature code,
    same default-filling) with one difference: the ban set is truncated to the
    bans that preceded the stage in the recorded draft order, instead of holding
    all bans at their full-draft values. Rows without a raw draft are dropped
    and counted by the caller.
    """
    processor_ = processor
    records: List[Dict[str, Any]] = []
    used = 0
    for index in range(len(frame)):
        match_id = int(frame["match_id"].iat[index])
        draft = drafts.get(match_id)
        if draft is None:
            continue
        rad_bans, dire_bans = bans_before_stage(draft, level)
        if level == 0:
            record = ablation._empty_pick_row(processor_, schema, rad_bans, dire_bans)
            record["match_id"] = match_id
            record["radiant_win"] = bool(frame["radiant_win"].iat[index])
            records.append(record)
            used += 1
            continue

        entries: List[Dict[str, Any]] = []
        for slot in range(level):
            entries.append(
                {
                    "hero_id": int(draft["rad_picks"][slot]),
                    "team": 0,
                    "is_pick": True,
                    "order": slot,
                }
            )
        for slot in range(level):
            entries.append(
                {
                    "hero_id": int(draft["dire_picks"][slot]),
                    "team": 1,
                    "is_pick": True,
                    "order": PICK_SLOTS + slot,
                }
            )
        for offset, hero in enumerate(rad_bans):
            entries.append({"hero_id": hero, "team": 0, "is_pick": False, "order": 100 + offset})
        for offset, hero in enumerate(dire_bans):
            entries.append({"hero_id": hero, "team": 1, "is_pick": False, "order": 200 + offset})
        payload: Dict[str, Any] = {
            "match_id": match_id,
            "start_time": int(frame["start_time"].iat[index]),
            "radiant_win": bool(frame["radiant_win"].iat[index]),
            "picks_bans": entries,
        }
        for column, key in (
            ("duration", "duration"),
            ("avg_mmr", "avg_mmr"),
            ("game_mode", "game_mode"),
            ("lobby_type", "lobby_type"),
            ("patch", "patch"),
        ):
            if column in frame.columns:
                payload[key] = frame[column].iat[index]
        record = processor_.process_match(payload, allow_partial_draft=True)
        if record is None:
            continue
        records.append(record)
        used += 1

    out = pd.DataFrame(records)
    categoricals = set(schema.get("categoricals", {}))
    missing: Dict[str, float] = {}
    for name in schema["feature_names"]:
        if name in categoricals:
            continue
        if name not in out.columns:
            missing[name] = ablation.default_for(schema, name)
        elif out[name].isna().any():
            out[name] = out[name].fillna(ablation.default_for(schema, name))
    if missing:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {name: np.full(len(out), value) for name, value in missing.items()},
                    index=out.index,
                ),
            ],
            axis=1,
        )
    return out


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def metrics_block(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    train_prior: float,
    bins: int,
) -> Dict[str, Any]:
    return {
        "metrics": compute_metrics(labels, probabilities, calibration_bins=bins),
        "baselines": baseline_metrics(labels, train_prior),
    }


def spread(values: np.ndarray) -> Dict[str, float]:
    """How far apart predictions are within one stage."""
    return {
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p95_minus_p05": float(np.percentile(values, 95) - np.percentile(values, 5)),
    }


def hero_effect_precision(frame: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    """Per-hero win rate: observed spread vs the spread binomial noise allows.

    A hero effect is identifiable if the observed variance of per-hero win
    rates is clearly larger than the variance expected from sampling alone.
    Selection into teams (players pick heroes they are good at) is *not*
    controlled here; that is exactly the confounding the report flags.
    """
    has_cols = [c for c in frame.columns if c.startswith("rad_has_")]
    if not has_cols:
        return {"available": False}
    matrix = frame[has_cols].to_numpy() > 0
    counts = matrix.sum(axis=0)
    wins = matrix.T.astype(np.float64) @ labels.astype(np.float64)
    keep = counts >= 50
    if keep.sum() < 10:
        return {"available": False}
    rates = wins[keep] / counts[keep]
    expected_var = float(np.mean((rates * (1 - rates)) / counts[keep]))
    observed_var = float(np.var(rates, ddof=1))
    half_widths = 1.96 * np.sqrt(rates * (1 - rates) / counts[keep])
    return {
        "available": True,
        "heroes_with_at_least_50_picks": int(keep.sum()),
        "picks_per_hero_median": float(np.median(counts[keep])),
        "picks_per_hero_min": int(counts[keep].min()),
        "observed_rate_std": float(np.sqrt(observed_var)),
        "expected_noise_std": float(np.sqrt(expected_var)),
        "signal_to_noise": float(np.sqrt(max(observed_var - expected_var, 0.0) / expected_var)),
        "median_ci95_half_width": float(np.median(half_widths)),
        "rate_min": float(rates.min()),
        "rate_max": float(rates.max()),
    }


def context_sufficiency(frame: pd.DataFrame) -> Dict[str, Any]:
    """Can the corpus support next-pick ranking?

    A ranking target needs the same draft context observed with *different*
    next picks, enough times to separate their outcomes. Two proxies are
    measured on the stored pick slots: repetition of the full 10-hero set, and
    repetition of a 2-picks-per-side context with its third-pick followers.
    Both are order-insensitive proxies for a real sequential draft.
    """
    has = [c for c in frame.columns if c.startswith("rad_has_")]
    dire = [c for c in frame.columns if c.startswith("dire_has_")]
    if not has or not dire:
        return {"available": False}
    tokens = np.array([c[len("rad_has_"):] for c in has])
    rad = frame[has].to_numpy() > 0
    dr = frame[dire].to_numpy() > 0
    hero_sets = [
        frozenset(tokens[rad[i]]) | frozenset(tokens[dr[i]])
        for i in range(len(frame))
    ]
    counts = pd.Series(hero_sets).value_counts()
    full_sets = {
        "distinct_full_drafts": int(len(counts)),
        "drafts_seen_more_than_once": int((counts > 1).sum()),
        "rows_in_unique_drafts": int((counts == 1).sum()),
        "max_occurrences": int(counts.max()) if len(counts) else 0,
    }

    slots = [f"rad_pick_{i}" for i in range(2)] + [f"dire_pick_{i}" for i in range(2)]
    if all(column in frame.columns for column in slots) and "rad_pick_2" in frame.columns:
        contexts = frame[slots].astype(int).astype(str).agg("-".join, axis=1).to_numpy()
        follower = frame["rad_pick_2"].astype(int).to_numpy()
        table = pd.DataFrame({"context": contexts, "follower": follower})
        per_context = table.groupby("context").size()
        follower_counts = table.groupby(["context", "follower"]).size()
        repeatable = per_context[per_context >= 50]
        cells = follower_counts[follower_counts.index.get_level_values(0).isin(repeatable.index)]
        return {
            "available": True,
            "full_hero_set": full_sets,
            "context_2v2": {
                "distinct_contexts": int(len(per_context)),
                "contexts_with_50_or_more_matches": int(len(repeatable)),
                "rows_in_those_contexts": int(repeatable.sum()) if len(repeatable) else 0,
                "median_distinct_followers_per_repeatable_context": (
                    float(follower_counts[follower_counts.index.get_level_values(0).isin(repeatable.index)]
                          .groupby(level=0).size().median())
                    if len(repeatable)
                    else 0.0
                ),
                "context_follower_cells_with_20_or_more_outcomes": int((cells >= 20).sum()),
            },
        }
    return {"available": True, "full_hero_set": full_sets}


def stage_table_from_report(path: Path) -> Dict[str, Any]:
    """Per-stage validation metrics already recorded by the family comparison."""
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    report = json.loads(path.read_text(encoding="utf-8"))
    family = next(
        (f for f in report["families"] if f["name"] == "xgboost_combined"), None
    )
    if family is None:
        return {"available": False, "reason": "no xgboost_combined family in report"}
    stages: Dict[str, Any] = {}
    for level, block in family["fits"][0]["partial_draft"].items():
        stages[level] = {
            "rows": block["metrics"]["rows"],
            "positives": block["metrics"]["positives"],
            "positive_rate": block["metrics"]["positive_rate"],
            "roc_auc": block["metrics"]["roc_auc"],
            "roc_auc_ci95": block["metrics"]["roc_auc_ci95"],
            "brier_score": block["metrics"]["brier_score"],
            "constant_prior_brier": block["baselines"]["constant_train_prior"]["brier_score"],
        }
    return {
        "available": True,
        "source": str(path),
        "seed": family["fits"][0]["seed"],
        "note": (
            "Recorded in the 2026-10-29 family comparison; these stages held all "
            "bans at full-draft values, so they are compared below against the "
            "ban-timing rebuild rather than recomputed."
        ),
        "stages": stages,
        "policy_floor": report.get("policy", {}).get("thresholds"),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    started = time.time()
    data_path = Path(args.data)
    raw_dir = Path(args.raw_dir)
    report_path = Path(args.report)
    candidate_dir = Path(args.candidate)

    schema = json.loads((candidate_dir / f"{candidate_dir.name}_schema.json").read_text(encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    onnx_path = candidate_dir / f"{candidate_dir.name}.onnx"
    model = OnnxPredictor(onnx_path, schema)

    dataset = pd.read_parquet(data_path)
    data_sha = file_sha256(data_path)
    train, val, test, split_info = ablation.load_frozen_split(
        dataset, Path(args.split_config), data_path=data_path
    )
    train_prior = float(train["radiant_win"].astype(int).mean())
    print(f"Dataset {len(dataset)}x{dataset.shape[1]} sha={data_sha[:12]}…")
    print(f"Split {split_info['sizes']} (validation scored; test never read)")
    print(f"Candidate {candidate_dir.name}: {len(feature_names)} features")

    # ---- A. true game mode ------------------------------------------------
    print("\n[1/6] True game mode from raw payloads…")
    true_modes = load_true_modes(
        raw_dir,
        dataset["match_id"].astype("int64"),
        cache_path=Path(args.mode_cache) if args.mode_cache else None,
        refresh=bool(args.refresh_modes),
    )
    ids = dataset["match_id"].astype("int64")
    mode_series = ids.map(true_modes)
    numeric = mode_series.isna()
    resolved = mode_series.where(
        ~numeric, "OPENDOTA_NUMERIC_" + dataset["game_mode"].astype(int).astype(str)
    )
    composition = resolved.value_counts()
    crosstab = pd.crosstab(dataset["game_mode"], resolved).to_dict()
    print("  " + ", ".join(f"{k}={v}" for k, v in composition.head(8).items()))
    misenc = {
        "rows_with_stratz_mode": int((~numeric).sum()),
        "rows_opendota_numeric": int(numeric.sum()),
        "captains_mode_stored_as_22": int(((mode_series == "CAPTAINS_MODE") & (dataset["game_mode"] == 22)).sum()),
        "captains_draft_stored_as_22": int(((mode_series == "CAPTAINS_DRAFT") & (dataset["game_mode"] == 22)).sum()),
        "real_all_pick_stored_as_0": int(((mode_series == "ALL_PICK") & (dataset["game_mode"] == 0)).sum()),
        "non_captains_modes_stored_as_0": int(
            mode_series.isin(
                ["RANDOM_DRAFT", "SINGLE_DRAFT", "TURBO", "ALL_RANDOM", "ALL_PICK_RANKED", "LEAST_PLAYED", "ALL_PICK"]
            ).sum()
        ),
    }

    # ---- B. per-mode validation metrics -----------------------------------
    print("\n[2/6] Per-mode validation metrics (inference contract)…")
    val_modes = ids.loc[val.index].map(true_modes)
    X_val = ablation.encode(val, schema, inference_defaults=True)
    y_val = val["radiant_win"].astype(int).to_numpy()
    val_pred = model.predict(X_val)
    overall = metrics_block(y_val, val_pred, train_prior=train_prior, bins=args.calibration_bins)

    segments: Dict[str, Any] = {}
    groups = {
        "captains_mode": val_modes == "CAPTAINS_MODE",
        "captains_draft": val_modes == "CAPTAINS_DRAFT",
        "opendota_captains_mode": val_modes.isna() & (val["game_mode"] == 2),
        "other_modes": val_modes.isin(["RANDOM_DRAFT", "SINGLE_DRAFT", "TURBO", "ALL_RANDOM", "ALL_PICK_RANKED", "LEAST_PLAYED", "ALL_PICK"]),
    }
    for name, mask in groups.items():
        mask = mask.to_numpy()
        rows = int(mask.sum())
        if rows < 100:
            segments[name] = {"rows": rows, "scored": False}
            continue
        segments[name] = {
            "rows": rows,
            **metrics_block(y_val[mask], val_pred[mask], train_prior=train_prior, bins=args.calibration_bins),
        }
        entry = segments[name]["metrics"]
        print(f"  {name:<24} n={rows:>5} AUC={entry['roc_auc']:.4f} Brier={entry['brier_score']:.4f}")
    ban_cols = [
        c
        for c in dataset.columns
        if c.startswith("rad_ban_") and c != "rad_ban_avg_winrate"
    ]
    has_no_ban = (dataset[ban_cols].to_numpy() > 0).sum(axis=1) == 0
    zero_ban_rows: Dict[str, Any] = {
        "all": int(has_no_ban.sum()),
        "all_fraction": float(has_no_ban.mean()),
        "by_true_mode": {},
    }
    for name in ("CAPTAINS_MODE", "CAPTAINS_DRAFT", "RANDOM_DRAFT", "SINGLE_DRAFT"):
        mask = (mode_series == name).to_numpy()
        if mask.sum():
            zero_ban_rows["by_true_mode"][name] = {
                "rows": int(mask.sum()),
                "rows_without_bans": int(has_no_ban[mask].sum()),
                "fraction": float(has_no_ban[mask].mean()),
            }

    # ---- C. recorded stage table -----------------------------------------
    print("\n[3/6] Recorded partial-draft stage table…")
    recorded_stages = stage_table_from_report(Path(args.family_report))
    if recorded_stages.get("available"):
        for level, entry in recorded_stages["stages"].items():
            print(
                f"  {level} picks/side: n={entry['rows']} AUC={entry['roc_auc']:.4f} "
                f"Brier={entry['brier_score']:.4f}"
            )

    # ---- D. ban-timing rebuild -------------------------------------------
    print("\n[4/6] Ban-timing rebuild (all bans vs bans known at the stage)…")
    processor = ablation._load_processor(data_path.parent)
    drafts = load_ordered_drafts(raw_dir, val["match_id"].astype("int64"))
    print(f"  raw drafts available for {len(drafts)}/{len(val)} validation rows")
    timing: Dict[str, Any] = {}
    for level in TIMED_LEVELS:
        all_bans = ablation.build_partial_draft_frame(
            val, level, schema=schema, processor=processor
        )
        timed = build_timed_stage_frame(
            val, drafts, level, schema=schema, processor=processor
        )
        # Paired comparison: both arms are scored on exactly the rows that have a
        # raw draft, so the delta is the ban set and nothing else.
        paired_ids = set(int(value) for value in drafts)
        all_bans = all_bans[all_bans["match_id"].astype("int64").isin(paired_ids)]
        timed = timed[timed["match_id"].astype("int64").isin(paired_ids)]

        def score(frame: pd.DataFrame) -> Dict[str, Any]:
            labels = frame["radiant_win"].astype(int).to_numpy()
            probabilities = model.predict(ablation.encode(frame, schema, inference_defaults=True))
            return {
                **metrics_block(labels, probabilities, train_prior=train_prior, bins=args.calibration_bins),
                "prediction_spread": spread(probabilities),
            }

        all_block = score(all_bans)
        timed_block = score(timed)
        bans_known = timed[[c for c in timed.columns if c.startswith("rad_ban_") and c != "rad_ban_avg_winrate"]]
        mean_bans = float((bans_known.to_numpy() > 0).sum(axis=1).mean())
        timing[str(level)] = {
            "rows": int(len(timed)),
            "mean_bans_per_side_known": mean_bans,
            "all_bans": all_block,
            "timed_bans": timed_block,
            "auc_delta_timed_minus_all": float(
                timed_block["metrics"]["roc_auc"] - all_block["metrics"]["roc_auc"]
            ),
        }
        print(
            f"  {level} picks/side: bans/side {mean_bans:.1f}  "
            f"AUC all={all_block['metrics']['roc_auc']:.4f} "
            f"timed={timed_block['metrics']['roc_auc']:.4f} "
            f"(delta {timing[str(level)]['auc_delta_timed_minus_all']:+.4f})  "
            f"spread p95-p05={timed_block['prediction_spread']['p95_minus_p05']:.4f}"
        )

    # ---- E. hero effects and context sufficiency -------------------------
    print("\n[5/6] Hero-effect identifiability and context repetition…")
    hero_effects = hero_effect_precision(train, train["radiant_win"].astype(int).to_numpy())
    contexts = context_sufficiency(dataset)
    if hero_effects.get("available"):
        print(
            f"  hero win rates: median picks {hero_effects['picks_per_hero_median']:.0f}, "
            f"observed sd {hero_effects['observed_rate_std']:.4f} vs noise sd "
            f"{hero_effects['expected_noise_std']:.4f} (SNR {hero_effects['signal_to_noise']:.1f})"
        )
    if contexts.get("available"):
        print(
            f"  full hero sets: {contexts['full_hero_set']['distinct_full_drafts']} distinct, "
            f"{contexts['full_hero_set']['drafts_seen_more_than_once']} repeated"
        )

    # ---- F. power ---------------------------------------------------------
    print("\n[6/6] Sampling precision…")
    auc_ci = overall["metrics"]["roc_auc_ci95"]
    power = {
        "validation_rows": int(len(y_val)),
        "observed_auc_ci95_half_width": float(
            (auc_ci["upper"] - auc_ci["lower"]) / 2
        ),
        "note": (
            "Half-width scales like 1/sqrt(n). The observed interval is what this "
            "validation set can resolve; differences smaller than it are not "
            "evidence of anything."
        ),
    }
    print(f"  validation AUC 95% CI half-width: {power['observed_auc_ci95_half_width']:.4f}")

    report = {
        "experiment_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Feasibility of alternative prediction objectives: research only, no "
            "training, no promotion, test partition untouched."
        ),
        "dataset": {
            "path": str(data_path),
            "sha256": data_sha,
            "rows": int(len(dataset)),
            "columns": int(dataset.shape[1]),
            "schema_fingerprint": schema_fingerprint(feature_names),
        },
        "split": {
            **split_info,
            "test_scored_here": False,
            "train_positive_rate": train_prior,
        },
        "candidate": {
            "model_dir": str(candidate_dir),
            "onnx": str(onnx_path),
            "feature_count": len(feature_names),
            "note": "the 2026-10-29 combined-ablation XGBoost candidate; not promoted",
        },
        "true_game_mode": {
            "distribution": {str(k): int(v) for k, v in composition.items()},
            "crosstab_processed_mode": {
                str(k): {str(kk): int(vv) for kk, vv in row.items()}
                for k, row in crosstab.items()
            },
            "misencoding": misenc,
            "rows_with_zero_bans": zero_ban_rows,
        },
        "validation": {
            "overall": overall,
            "by_true_mode": segments,
        },
        "recorded_partial_draft": recorded_stages,
        "ban_timing": timing,
        "hero_effects": hero_effects,
        "contexts": contexts,
        "power": power,
        "protocol": {
            "test_partition": "not read, not scored, not used for selection",
            "fitted_models": 0,
            "calibration_bins": int(args.calibration_bins),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Total {time.time() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Objective feasibility diagnostics (research only, no training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--family-report", type=Path, default=FAMILY_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--mode-cache",
        type=Path,
        default=DEFAULT_MODE_CACHE,
        help="cache of match_id<TAB>gameMode; rescanned only when absent or --refresh-modes",
    )
    parser.add_argument(
        "--refresh-modes",
        action="store_true",
        help="ignore the mode cache and rescan the raw payloads (slow on a cold disk)",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
