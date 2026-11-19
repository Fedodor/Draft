#!/usr/bin/env python3
"""Next-pick imitation experiment (research only).

Question: does the draft state predict the *observed* next pick better than a
hero-frequency baseline? This is an imitation target, not a win-probability
target and not a counterfactual ranking: it measures whether the state carries
decision-relevant information at all, and says nothing about whether the
imitated pick was good.

Protocol
--------
* Population: STRATZ `CAPTAINS_MODE` matches in the frozen train/validation
  partitions only. `game_mode` in the parquet is mis-encoded (AUDIT P1-7), so
  the mode is read from the raw payload's `gameMode` field for every match.
* Chronology: every action's `order` must be dense and unique (0..n-1); the
  sequence is rebuilt by sorting on it. A payload that fails this is excluded,
  never repaired.
* A *decision state* is the moment immediately before one pick: the picks and
  bans that had already happened, the acting side, and the hero that was picked.
* Features are built from the prefix only, so no future pick, future ban, final
  lineup or match result can enter the vector. The test partition is not parsed.
* Baselines and model are fitted on training states only; all metrics are
  validation only; uncertainty is a match-level cluster bootstrap.

The test partition is never read. Nothing is promoted or exported.

Usage:
    python ml/training/imitation.py
    python ml/training/imitation.py --max-train-matches 20000   # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from training import ablation
from training.splits import file_sha256, schema_fingerprint

DEFAULT_DATA = ML_ROOT / "datasets" / "processed_v2" / "draft_features.parquet"
DEFAULT_SPLIT_CONFIG = ML_ROOT / "configs" / "eval_split_v2.json"
DEFAULT_RAW_DIR = ML_ROOT / "datasets" / "raw"
DEFAULT_OUT = ML_ROOT / "models" / "imitation"

# The shipped contract has one column per hero id 1..139. Anything outside that
# range cannot be a class or a feature, so those states/heroes are excluded and
# counted rather than silently dropped.
HERO_MIN = 1
HERO_MAX = 139
ROSTER = np.arange(HERO_MIN, HERO_MAX + 1, dtype=np.int64)
N_HEROES = len(ROSTER)
HERO_INDEX = {int(hero): index for index, hero in enumerate(ROSTER)}

BLOCKS = ("rad_pick", "dire_pick", "rad_ban", "dire_ban")
N_BLOCKS = len(BLOCKS)
# Ten picks in Captain's Mode, grouped into the three decision phases. Stage is
# an interaction on the hero-removal blocks, so the model can hold different
# hero preferences per phase without a per-phase parameter explosion.
PICK_STAGE_GROUP = (0, 0, 1, 1, 1, 1, 1, 1, 2, 2)
STAGE_GROUPS = ("early", "mid", "late")
N_STAGE_GROUPS = len(STAGE_GROUPS)
# Removal (substitution) effects are pooled across stages: letting them vary by
# stage triples the parameter count and, at this sample size, the extra variance
# swamps the signal (measured: the first-pick top-1 halved). Stage-specific hero
# *preferences* are cheap and are carried by the six bias columns below.
N_REMOVAL_FEATURES = N_BLOCKS * N_HEROES  # 556
# Six always-on indicators, one per (stage group, acting side). Because the
# multinomial model gives every class its own coefficient per feature, one such
# column carries a full per-bucket hero intercept -- which is exactly the
# stage/side frequency baseline. Without them the model has only a global hero
# preference and cannot represent the baseline it is being compared against.
N_BIAS_FEATURES = N_STAGE_GROUPS * 2
N_FEATURES = N_REMOVAL_FEATURES + N_BIAS_FEATURES

EXPECTED_PICKS = 10
EXPECTED_PER_SIDE = 5
MAX_ACTIONS = 24

# Regularisation is chosen on an internal, chronologically-last slice of the
# *training* partition, never on the frozen validation partition, so the reported
# validation numbers are not selection-optimistic. The grid is tiny and reported
# in full.
# Set after a 4,000-match smoke run in which C=1.0 overfit badly (inner log loss
# 5.27 vs 4.39). The grid is reported in full and selected on the inner holdout.
MODEL_C_GRID = (0.03, 0.1, 0.3)
MODEL_MAX_ITER = 200
INNER_VALIDATION_SHARE = 0.15
BASELINE_SMOOTHING = 1.0
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0
LOG_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Raw payload -> ordered actions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    order: int
    is_pick: bool
    radiant: bool
    hero: Optional[int]


def parse_payload(payload: Mapping[str, Any]) -> Tuple[str, List[Action], Optional[str]]:
    """Ordered actions from one STRATZ payload, or an exclusion reason."""
    mode = payload.get("gameMode")
    if not isinstance(mode, str):
        return "UNKNOWN", [], "no_stratz_game_mode"
    entries = payload.get("pickBans")
    if not isinstance(entries, list) or not entries:
        return mode, [], "no_draft"
    if len(entries) > MAX_ACTIONS:
        return mode, [], "too_many_actions"

    actions: List[Action] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return mode, [], "malformed_entry"
        hero = entry.get("heroId")
        if hero is not None:
            try:
                hero = int(hero)
            except (TypeError, ValueError):
                return mode, [], "malformed_hero"
        radiant = entry.get("isRadiant")
        is_pick = entry.get("isPick")
        if not isinstance(radiant, bool) or not isinstance(is_pick, bool):
            return mode, [], "malformed_side_or_type"
        order = entry.get("order")
        if not isinstance(order, int):
            return mode, [], "missing_order"
        actions.append(Action(order=order, is_pick=is_pick, radiant=radiant, hero=hero))

    actions.sort(key=lambda item: item.order)
    orders = [item.order for item in actions]
    if len(set(orders)) != len(orders) or orders != list(range(len(orders))):
        return mode, [], "non_dense_order"
    return mode, actions, None


def draft_exclusion(actions: Sequence[Action]) -> Optional[str]:
    """Reject drafts whose pick/ban structure cannot support the target."""
    picks = [item for item in actions if item.is_pick]
    radiant = [item for item in picks if item.radiant]
    dire = [item for item in picks if not item.radiant]
    if len(picks) != EXPECTED_PICKS or len(radiant) != EXPECTED_PER_SIDE:
        return "pick_count"
    if len(dire) != EXPECTED_PER_SIDE:
        return "pick_count"
    if any(item.hero is None for item in picks):
        return "null_pick_hero"
    heroes = [item.hero for item in actions if item.hero is not None]
    if len(heroes) != len(set(heroes)):
        return "duplicate_hero"
    if any(item.hero is not None and not HERO_MIN <= item.hero <= HERO_MAX for item in picks):
        return "pick_hero_outside_roster"
    return None


# --------------------------------------------------------------------------- #
# Actions -> decision states
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class State:
    match_id: int
    partition: str
    pick_index: int
    acting_radiant: bool
    target: int
    rad_pick: Tuple[int, ...]
    dire_pick: Tuple[int, ...]
    rad_ban: Tuple[int, ...]
    dire_ban: Tuple[int, ...]

    def removed(self) -> set:
        return set(self.rad_pick) | set(self.dire_pick) | set(self.rad_ban) | set(self.dire_ban)


def states_for_match(
    match_id: int, partition: str, actions: Sequence[Action]
) -> List[State]:
    """One state per pick, holding only the actions that preceded it."""
    rad_pick: List[int] = []
    dire_pick: List[int] = []
    rad_ban: List[int] = []
    dire_ban: List[int] = []
    states: List[State] = []
    pick_index = 0
    for action in actions:
        if action.is_pick:
            hero = int(action.hero)  # guaranteed non-null by draft_exclusion
            if hero in set(rad_pick) | set(dire_pick) | set(rad_ban) | set(dire_ban):
                raise ValueError(f"match {match_id}: target {hero} already removed")
            states.append(
                State(
                    match_id=match_id,
                    partition=partition,
                    pick_index=pick_index,
                    acting_radiant=bool(action.radiant),
                    target=hero,
                    rad_pick=tuple(rad_pick),
                    dire_pick=tuple(dire_pick),
                    rad_ban=tuple(rad_ban),
                    dire_ban=tuple(dire_ban),
                )
            )
            pick_index += 1
            (rad_pick if action.radiant else dire_pick).append(hero)
        elif action.hero is not None:
            (rad_ban if action.radiant else dire_ban).append(int(action.hero))
    return states


# --------------------------------------------------------------------------- #
# Action cache (raw payload scan)
# --------------------------------------------------------------------------- #
def scan_actions(
    raw_dir: Path,
    match_ids: Sequence[int],
    *,
    workers: int = 16,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse the raw payload of every requested match, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    reasons: Dict[str, int] = {}

    def one(match_id: int):
        path = raw_dir / f"{int(match_id)}.json"
        if not path.is_file():
            return int(match_id), None, "missing_raw_file"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return int(match_id), None, "unreadable_raw_file"
        mode, actions, reason = parse_payload(payload)
        if reason is not None:
            return int(match_id), {"mode": mode}, reason
        return int(match_id), {"mode": mode, "actions": actions}, None

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for match_id, payload, reason in pool.map(one, list(match_ids), chunksize=64):
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            actions = payload["actions"]
            row = {"match_id": match_id, "game_mode": payload["mode"], "n_actions": len(actions)}
            for index in range(MAX_ACTIONS):
                if index < len(actions):
                    action = actions[index]
                    row[f"a{index}_type"] = 1 if action.is_pick else 0
                    row[f"a{index}_side"] = 1 if action.radiant else 0
                    row[f"a{index}_hero"] = int(action.hero) if action.hero is not None else 0
                else:
                    row[f"a{index}_type"] = -1
                    row[f"a{index}_side"] = -1
                    row[f"a{index}_hero"] = 0
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, reasons


def load_action_table(
    raw_dir: Path,
    out_dir: Path,
    partitions: Mapping[int, str],
    *,
    refresh: bool = False,
    workers: int = 16,
    max_train_matches: Optional[int] = None,
    sample_seed: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Cached ordered actions plus the scan-time exclusion counts.

    Payloads rejected during the scan never reach the cache, so the reason counts
    are stored beside it. Loading the cache without them silently loses the
    matches that were dropped before parsing, which makes the eligible
    population look larger than it is.
    """
    cache = out_dir / "actions.parquet"
    reasons_path = out_dir / "scan_reasons.json"
    if cache.is_file() and not refresh:
        frame = pd.read_parquet(cache)
        frame = frame[frame["match_id"].isin(partitions)].reset_index(drop=True)
        reasons: Dict[str, int] = {}
        if reasons_path.is_file():
            reasons = json.loads(reasons_path.read_text(encoding="utf-8"))
        print(f"Action cache: {cache} ({len(frame)} matches)")
        return frame, reasons

    wanted = sorted(partitions)
    if max_train_matches:
        rng = np.random.default_rng(sample_seed)
        train_ids = sorted(m for m, part in partitions.items() if part == "train")
        keep = set(
            rng.choice(
                train_ids, size=min(max_train_matches, len(train_ids)), replace=False
            ).tolist()
        )
        wanted = sorted(
            [m for m in wanted if partitions[m] == "validation" or int(m) in keep]
        )
    print(f"Parsing {len(wanted)} raw payloads…")
    frame, reasons = scan_actions(raw_dir, wanted, workers=workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    reasons_path.write_text(json.dumps(reasons, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {cache}")
    return frame, reasons


def actions_to_states(frame: pd.DataFrame, partitions: Mapping[int, str]) -> Tuple[List[State], Dict[str, int]]:
    """Validated states for every eligible match, plus exclusion counts."""
    reasons: Dict[str, int] = {}
    states: List[State] = []
    for row in frame.itertuples(index=False):
        match_id = int(row.match_id)
        partition = partitions[match_id]
        if row.game_mode != "CAPTAINS_MODE":
            reasons["unsupported_mode"] = reasons.get("unsupported_mode", 0) + 1
            continue
        actions = [
            Action(
                order=index,
                is_pick=bool(getattr(row, f"a{index}_type") == 1),
                radiant=bool(getattr(row, f"a{index}_side") == 1),
                hero=int(getattr(row, f"a{index}_hero")) or None,
            )
            for index in range(int(row.n_actions))
        ]
        reason = draft_exclusion(actions)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        try:
            states.extend(states_for_match(match_id, partition, actions))
        except ValueError:
            reasons["target_in_prefix"] = reasons.get("target_in_prefix", 0) + 1
    return states, reasons


# --------------------------------------------------------------------------- #
# Feature matrices
# --------------------------------------------------------------------------- #
def build_matrix(states: Sequence[State]) -> sparse.csr_matrix:
    """Sparse state x candidate-feature matrix (562 columns).

    A removal column identifies (block, hero): which side removed that hero and
    whether it was a pick or a ban. Only heroes in the prefix are marked, so the
    vector is a function of already-known actions. The final six columns are
    always-on (stage group, acting side) indicators, giving the model a
    per-bucket hero intercept -- i.e. the frequency baseline plus state.
    """
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for state in states:
        group = PICK_STAGE_GROUP[state.pick_index]
        columns: List[int] = [
            N_REMOVAL_FEATURES + group * 2 + (1 if state.acting_radiant else 0)
        ]
        for heroes, block in (
            (state.rad_pick, 0),
            (state.dire_pick, 1),
            (state.rad_ban, 2),
            (state.dire_ban, 3),
        ):
            offset = block * N_HEROES
            for hero in heroes:
                index = HERO_INDEX.get(int(hero))
                if index is not None:
                    columns.append(offset + index)
        indices.extend(columns)
        data.extend([1.0] * len(columns))
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(states), N_FEATURES),
    )


def legal_matrix(states: Sequence[State]) -> np.ndarray:
    """Boolean (n_states, n_heroes) mask of heroes not yet removed."""
    legal = np.ones((len(states), N_HEROES), dtype=bool)
    for row, state in enumerate(states):
        for removed in (*state.rad_pick, *state.dire_pick, *state.rad_ban, *state.dire_ban):
            index = HERO_INDEX.get(int(removed))
            if index is not None:
                legal[row, index] = False
    return legal


def targets(states: Sequence[State]) -> np.ndarray:
    return np.asarray([HERO_INDEX[int(state.target)] for state in states], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def frequency_scores(
    train_states: Sequence[State],
    *,
    by_stage_and_side: bool,
) -> Dict[Any, np.ndarray]:
    """Target-hero counts from training states only."""
    table: Dict[Any, np.ndarray] = {}

    def key(state: State):
        if not by_stage_and_side:
            return "global"
        return (PICK_STAGE_GROUP[state.pick_index], bool(state.acting_radiant))

    for state in train_states:
        bucket = table.setdefault(key(state), np.zeros(N_HEROES, dtype=np.float64))
        bucket[HERO_INDEX[int(state.target)]] += 1.0
    if not by_stage_and_side:
        return {"global": table["global"]}
    return table


def frequency_probabilities(
    scores: Mapping[Any, np.ndarray],
    states: Sequence[State],
    *,
    by_stage_and_side: bool,
    smoothing: float = BASELINE_SMOOTHING,
) -> Tuple[np.ndarray, List[Any]]:
    """Masked, smoothed, renormalised distribution plus unsupported keys.

    A (stage group, side) cell with no training states falls back to a uniform
    distribution over the legal candidates and is reported, never silently
    treated as a frequency estimate.
    """
    legal = legal_matrix(states)
    probabilities = np.zeros(legal.shape, dtype=np.float64)
    unsupported: List[Any] = []
    if by_stage_and_side:
        for row, state in enumerate(states):
            key = (PICK_STAGE_GROUP[state.pick_index], bool(state.acting_radiant))
            counts = scores.get(key)
            if counts is None:
                if key not in unsupported:
                    unsupported.append(key)
                counts = np.zeros(N_HEROES, dtype=np.float64)
            weights = np.where(legal[row], counts + smoothing, 0.0)
            total = weights.sum()
            probabilities[row] = weights / max(total, 1e-12)
    else:
        counts = scores["global"]
        weights = np.where(legal, counts[None, :] + smoothing, 0.0)
        totals = weights.sum(axis=1, keepdims=True)
        probabilities = np.where(legal, weights / np.maximum(totals, 1e-12), 0.0)
    return probabilities, unsupported


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rank_and_logp(probabilities: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Competition rank of the observed hero, ties broken by lower hero id."""
    rows = np.arange(len(truth))
    target_probability = probabilities[rows, truth]
    greater = (probabilities > target_probability[:, None]).sum(axis=1)
    # Equal scores: a lower hero id ranks first, so it counts as "ahead".
    equal_lower = (
        (probabilities == target_probability[:, None])
        & (np.arange(N_HEROES)[None, :] < truth[:, None])
    ).sum(axis=1)
    rank = greater + equal_lower
    logp = -np.log(np.clip(target_probability, LOG_FLOOR, 1.0))
    return rank.astype(np.int64), logp


def summarise(rank: np.ndarray, logp: np.ndarray) -> Dict[str, float]:
    return {
        "states": int(len(rank)),
        "top1": float((rank == 0).mean()),
        "top3": float((rank < 3).mean()),
        "mrr": float((1.0 / (rank + 1)).mean()),
        "log_loss": float(logp.mean()),
    }


def cluster_bootstrap(
    match_ids: np.ndarray,
    per_match: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
    comparisons: Sequence[Tuple[str, str]] = (),
) -> Dict[str, Any]:
    """Match-level bootstrap of metric values and paired differences.

    `per_match` maps a method name to a per-match mean (one value per unique
    match, aligned with `np.unique(match_ids)`). Resampling matches rather than
    states keeps the ten decisions of one draft together.
    """
    matches = np.unique(match_ids)
    methods = list(per_match)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates) for name in methods}
    for index in range(replicates):
        sample = rng.integers(0, len(matches), size=len(matches))
        for name in methods:
            draws[name][index] = per_match[name][sample].mean()
    result: Dict[str, Any] = {
        "replicates": int(replicates),
        "matches": int(len(matches)),
        "metrics": {},
    }
    for name in methods:
        result["metrics"][name] = {
            "ci95": [
                float(np.percentile(draws[name], 2.5)),
                float(np.percentile(draws[name], 97.5)),
            ]
        }
    for candidate, reference in comparisons:
        if candidate not in draws or reference not in draws:
            raise ValueError(
                f"bootstrap comparison {candidate} vs {reference} needs both methods"
            )
        delta = draws[candidate] - draws[reference]
        result.setdefault("differences", {})[f"{candidate}_minus_{reference}"] = {
            "mean": float(delta.mean()),
            "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "fraction_positive": float((delta > 0).mean()),
        }
    return result


def per_match_means(match_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of `values` within each unique match id, in sorted match order."""
    frame = pd.DataFrame({"match_id": match_ids, "value": values})
    return frame.groupby("match_id", sort=True)["value"].mean().to_numpy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    started = time.time()
    data_path = Path(args.data)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(data_path, columns=["match_id", "start_time", "radiant_win"])
    data_sha = file_sha256(data_path)
    train, val, test, split_info = ablation.load_frozen_split(
        dataset, Path(args.split_config), data_path=data_path
    )
    partitions: Dict[int, str] = {}
    for name, part in (("train", train), ("validation", val)):
        for match_id in part["match_id"].astype("int64"):
            partitions[int(match_id)] = name
    test_ids = set(test["match_id"].astype("int64"))
    if test_ids & set(partitions):
        raise SystemExit("frozen split overlaps train/validation and test")
    print(f"Dataset {len(dataset)} rows sha={data_sha[:12]}… split={split_info['sizes']}")
    print(f"Train+validation matches: {len(partitions)} (test never parsed)")

    # ---- actions ---------------------------------------------------------
    frame, reasons = load_action_table(
        raw_dir,
        out_dir,
        partitions,
        refresh=args.refresh_actions,
        workers=args.workers,
        max_train_matches=args.max_train_matches,
        sample_seed=args.sample_seed,
    )

    if set(frame["match_id"].astype("int64")) & test_ids:
        raise SystemExit("action table contains test-partition matches; refusing to continue")

    states, state_reasons = actions_to_states(frame, partitions)
    for key, value in state_reasons.items():
        reasons[key] = reasons.get(key, 0) + value
    if not states:
        raise SystemExit("no eligible decision states")
    train_states = [s for s in states if s.partition == "train"]
    val_states = [s for s in states if s.partition == "validation"]
    print(f"States: train {len(train_states)}, validation {len(val_states)}")
    if set(s.match_id for s in train_states) & set(s.match_id for s in val_states):
        raise SystemExit("a match contributes states to both partitions")

    X_train = build_matrix(train_states)
    X_val = build_matrix(val_states)
    y_train = targets(train_states)
    y_val = targets(val_states)
    legal_val = legal_matrix(val_states)
    print(f"Feature matrix: {X_train.shape}, nnz/row={X_train.nnz / X_train.shape[0]:.1f}")

    # ---- baselines -------------------------------------------------------
    global_scores = frequency_scores(train_states, by_stage_and_side=False)
    conditional_scores = frequency_scores(train_states, by_stage_and_side=True)
    global_val, _ = frequency_probabilities(
        global_scores, val_states, by_stage_and_side=False
    )
    stage_side_val, unsupported_cells = frequency_probabilities(
        conditional_scores, val_states, by_stage_and_side=True
    )
    baselines_val = {
        "global_frequency": global_val,
        "stage_side_frequency": stage_side_val,
    }

    # ---- model -----------------------------------------------------------
    # ---- regularisation, chosen inside the training partition ------------
    start_times = dict(
        zip(
            dataset["match_id"].astype("int64").tolist(),
            dataset["start_time"].astype("int64").tolist(),
        )
    )
    train_matches = sorted(
        {state.match_id for state in train_states},
        key=lambda match_id: start_times.get(match_id, 0),
    )
    cut = max(1, int(round(len(train_matches) * (1.0 - INNER_VALIDATION_SHARE))))
    inner_train_ids = set(train_matches[:cut])
    inner_fit = [state for state in train_states if state.match_id in inner_train_ids]
    inner_check = [state for state in train_states if state.match_id not in inner_train_ids]
    if not inner_fit or not inner_check:
        raise SystemExit("training partition is too small for an internal holdout")
    X_inner_fit = build_matrix(inner_fit)
    y_inner_fit = targets(inner_fit)
    X_inner_check = build_matrix(inner_check)
    y_inner_check = targets(inner_check)
    legal_inner = legal_matrix(inner_check)
    print(
        f"Inner holdout: {len(inner_fit)} train states, {len(inner_check)} check states "
        f"({len(inner_check)} from the last {len(train_matches) - cut} training matches)"
    )

    grid_values = tuple(
        float(part) for part in str(args.model_c_grid).split(",") if part.strip()
    )
    if not grid_values:
        raise SystemExit("--model-c-grid is empty")
    grid: List[Dict[str, Any]] = []
    for candidate_c in grid_values:
        probe = LogisticRegression(
            C=float(candidate_c), solver="lbfgs", max_iter=MODEL_MAX_ITER, tol=1e-3
        )
        probe.fit(X_inner_fit, y_inner_fit)
        probabilities = np.zeros((len(inner_check), N_HEROES), dtype=np.float64)
        probabilities[:, probe.classes_.astype(np.int64)] = probe.predict_proba(X_inner_check)
        probabilities = np.where(legal_inner, probabilities, 0.0)
        probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        rank, logp = rank_and_logp(probabilities, y_inner_check)
        entry = {"C": float(candidate_c), **summarise(rank, logp)}
        grid.append(entry)
        print(
            f"  C={candidate_c:<5g} inner top1={entry['top1']:.4f} "
            f"top3={entry['top3']:.4f} mrr={entry['mrr']:.4f} logloss={entry['log_loss']:.4f}"
        )
    best = sorted(grid, key=lambda row: (-row["top1"], row["log_loss"]))[0]
    selected_c = float(best["C"])
    print(f"  selected C={selected_c:g} (inner holdout only)")

    print(f"Fitting multinomial logistic regression on all training states (C={selected_c})…")
    model = LogisticRegression(
        C=selected_c,
        solver="lbfgs",
        max_iter=MODEL_MAX_ITER,
        tol=1e-3,
    )
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start
    print(f"  fit {fit_seconds:.1f}s, n_iter={int(np.max(model.n_iter_))}")

    # Align predict_proba columns with the roster: any hero that never appeared
    # as a target in training is a valid class with zero learned mass.
    predicted = np.zeros((len(val_states), N_HEROES), dtype=np.float64)
    predicted[:, model.classes_.astype(np.int64)] = model.predict_proba(X_val)
    masked = np.where(legal_val, predicted, 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise SystemExit("model assigned no mass to legal candidates for some state")
    model_probabilities = masked / totals

    # Ablation of the state itself: keep the six bucket-bias columns and drop
    # every removal column. This isolates "per-stage/side hero preference" (what
    # the frequency baseline also has) from "how the removed heroes shift the
    # prediction", which is the part that makes the question interesting.
    # Keep the full width (the estimator validates its feature count) and zero
    # only the removal block.
    keep_bias = sparse.diags(
        np.concatenate(
            [np.zeros(N_REMOVAL_FEATURES), np.ones(N_BIAS_FEATURES)]
        ).astype(np.float32)
    )
    X_bucket = X_val @ keep_bias
    bucket_only = np.zeros((len(val_states), N_HEROES), dtype=np.float64)
    bucket_only[:, model.classes_.astype(np.int64)] = model.predict_proba(X_bucket)
    bucket_only = np.where(legal_val, bucket_only, 0.0)
    bucket_only = bucket_only / np.maximum(bucket_only.sum(axis=1, keepdims=True), 1e-12)

    # ---- metrics ---------------------------------------------------------
    methods = {
        "global_frequency": baselines_val["global_frequency"],
        "stage_side_frequency": baselines_val["stage_side_frequency"],
        "model_bucket_only": bucket_only,
        "model": model_probabilities,
    }
    match_ids_val = np.asarray([s.match_id for s in val_states], dtype=np.int64)
    stage_of = np.asarray([PICK_STAGE_GROUP[s.pick_index] for s in val_states], dtype=np.int64)
    side_of = np.asarray([s.acting_radiant for s in val_states], dtype=bool)
    pick_of = np.asarray([s.pick_index for s in val_states], dtype=np.int64)
    candidates = legal_val.sum(axis=1)

    overall: Dict[str, Any] = {}
    ranks: Dict[str, np.ndarray] = {}
    for name, probabilities in methods.items():
        rank, logp = rank_and_logp(probabilities, y_val)
        ranks[name] = rank
        overall[name] = summarise(rank, logp)

    by_stage: Dict[str, Any] = {}
    for group_index, group_name in enumerate(STAGE_GROUPS):
        mask = stage_of == group_index
        by_stage[group_name] = {"states": int(mask.sum()), "matches": int(len(np.unique(match_ids_val[mask])))}
        for name, probabilities in methods.items():
            rank, logp = rank_and_logp(probabilities[mask], y_val[mask])
            by_stage[group_name][name] = summarise(rank, logp)

    by_pick: Dict[str, Any] = {}
    for pick_index in range(EXPECTED_PICKS):
        mask = pick_of == pick_index
        entry: Dict[str, Any] = {
            "states": int(mask.sum()),
            "matches": int(len(np.unique(match_ids_val[mask]))),
            "acting_radiant_states": int(mask.sum() - side_of[mask].sum()),
        }
        for name, probabilities in methods.items():
            rank, logp = rank_and_logp(probabilities[mask], y_val[mask])
            entry[name] = summarise(rank, logp)
        by_pick[str(pick_index)] = entry

    bootstrap: Dict[str, Any] = {}
    for metric, extractor in (
        ("top1", lambda r: (r == 0).astype(float)),
        ("top3", lambda r: (r < 3).astype(float)),
        ("mrr", lambda r: 1.0 / (r + 1)),
    ):
        per_match = {
            name: per_match_means(match_ids_val, extractor(ranks[name]))
            for name in ("stage_side_frequency", "model_bucket_only", "model")
        }
        bootstrap[metric] = cluster_bootstrap(
            match_ids_val,
            per_match,
            replicates=args.bootstrap,
            seed=BOOTSTRAP_SEED,
            comparisons=(
                ("model", "stage_side_frequency"),
                ("model_bucket_only", "stage_side_frequency"),
                ("model", "model_bucket_only"),
            ),
        )

    # ---- report ----------------------------------------------------------
    first_pick = by_pick["0"]
    report = {
        "experiment_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Does draft state predict the observed next pick beyond hero frequency? "
            "Imitation target; research only; no counterfactual claim, no promotion."
        ),
        "dataset": {
            "path": str(data_path),
            "sha256": data_sha,
            "rows": int(len(dataset)),
            "columns_fingerprint": schema_fingerprint(list(dataset.columns)),
        },
        "split": {
            "config": str(args.split_config),
            **split_info,
            "test_parsed_here": False,
        },
        "population": {
            "mode_source": "raw payload `gameMode` string; not the parquet column (AUDIT P1-7)",
            "modes_included": ["CAPTAINS_MODE"],
            "train_matches": int(len(set(s.match_id for s in train_states))),
            "validation_matches": int(len(set(s.match_id for s in val_states))),
            "train_states": len(train_states),
            "validation_states": len(val_states),
            "exclusions": reasons,
            "candidate_universe": [HERO_MIN, HERO_MAX],
            "distinct_targets_train": int(len(set(y_train.tolist()))),
            "distinct_targets_validation": int(len(set(y_val.tolist()))),
            "validation_candidate_size": {
                "mean": float(candidates.mean()),
                "min": int(candidates.min()),
                "max": int(candidates.max()),
            },
        },
        "features": {
            "blocks": list(BLOCKS),
            "stage_groups": list(STAGE_GROUPS),
            "stage_group_of_pick": list(PICK_STAGE_GROUP),
            "n_features": N_FEATURES,
            "nnz_per_row": float(X_train.nnz / X_train.shape[0]),
            "future_information": "none: every feature is a function of actions before the target pick",
        },
        "model": {
            "type": "multinomial logistic regression",
            "C": selected_c,
            "C_grid": list(MODEL_C_GRID),
            "C_selection": {
                "partition": "internal chronological holdout inside the training partition",
                "share": INNER_VALIDATION_SHARE,
                "frozen_validation_used": False,
                "grid": grid,
            },
            "max_iter": MODEL_MAX_ITER,
            "tol": 1e-3,
            "fit_seconds": fit_seconds,
            "iterations": int(np.max(model.n_iter_)),
            "classes": N_HEROES,
        },
        "baselines": {
            "smoothing": BASELINE_SMOOTHING,
            "unsupported_stage_side_cells": [list(map(str, cell)) for cell in unsupported_cells],
            "global_frequency": "target-hero counts over training states",
            "stage_side_frequency": "target-hero counts within (stage group, acting side) over training states",
            "tie_break": "lower hero id wins",
            "masking": "already picked or banned heroes excluded before ranking, then renormalised",
        },
        "validation": {
            "overall": overall,
            "by_stage_group": by_stage,
            "by_pick_index": by_pick,
            "bootstrap": bootstrap,
        },
        "headline": {
            "early_top1_model": first_pick["model"]["top1"],
            "early_top1_stage_side": first_pick["stage_side_frequency"]["top1"],
            "early_top1_global": first_pick["global_frequency"]["top1"],
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = out_dir / "imitation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 74)
    for name in ("global_frequency", "stage_side_frequency", "model_bucket_only", "model"):
        entry = overall[name]
        print(
            f"{name:<22} top1={entry['top1']:.4f} top3={entry['top3']:.4f} "
            f"mrr={entry['mrr']:.4f} logloss={entry['log_loss']:.4f}"
        )
    print("\nfirst pick (7 bans known, no picks):")
    for name in ("global_frequency", "stage_side_frequency", "model_bucket_only", "model"):
        entry = first_pick[name]
        print(f"  {name:<22} top1={entry['top1']:.4f} top3={entry['top3']:.4f} mrr={entry['mrr']:.4f}")
    print("\nmodel - stage_side_frequency (cluster bootstrap 95% CI):")
    for metric, block in bootstrap.items():
        delta = block["differences"]["model_minus_stage_side_frequency"]
        print(
            f"  {metric:<5} {delta['mean']:+.4f} "
            f"({delta['ci95'][0]:+.4f}..{delta['ci95'][1]:+.4f}) "
            f"positive in {delta['fraction_positive']:.1%} of resamples"
        )
    print("\nbucket-only - stage_side_frequency (is the intercept alone enough?):")
    for metric, block in bootstrap.items():
        delta = block["differences"]["model_bucket_only_minus_stage_side_frequency"]
        print(f"  {metric:<5} {delta['mean']:+.4f} ({delta['ci95'][0]:+.4f}..{delta['ci95'][1]:+.4f})")
    print("\nmodel - bucket-only (what the state terms add):")
    for metric, block in bootstrap.items():
        delta = block["differences"]["model_minus_model_bucket_only"]
        print(f"  {metric:<5} {delta['mean']:+.4f} ({delta['ci95'][0]:+.4f}..{delta['ci95'][1]:+.4f})")
    print(f"\nReport: {report_path}")
    print(f"Total {time.time() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Next-pick imitation experiment (research only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--max-train-matches",
        type=int,
        default=None,
        help="smoke runs only: sample this many training matches (validation unchanged)",
    )
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument(
        "--model-c-grid",
        type=str,
        default=",".join(str(value) for value in MODEL_C_GRID),
        help="regularisation grid, selected on an internal training holdout",
    )
    parser.add_argument(
        "--refresh-actions",
        action="store_true",
        help="re-parse raw payloads instead of using the action cache",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
