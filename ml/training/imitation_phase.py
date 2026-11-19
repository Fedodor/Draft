#!/usr/bin/env python3
"""Phase-aware imitation experiment (research only).

Session 10 showed the pooled imitation model beats a stage/side frequency table
overall (top-1 0.0551 vs 0.0405) but *loses* at the two opening picks
(0.0744 vs 0.0990 at pick 0). This script asks one narrow question:

    can a hybrid match or beat the frequency baseline at picks 0-1 without giving
    up the pooled model's gains from pick 2 onward?

Four strategies are compared on identical validation states and legal candidate
sets:

A. `stage_side_frequency` - the baseline, fitted on training states only;
B. `pooled_model`         - the session-10 model, refitted and checked against
                            its recorded metrics;
C. `hybrid_freq_then_pooled` - A for picks 0-1, B for picks 2-9;
D. `early_pick0`, `early_pick1`, `early_shared` - a separate, smaller model
   trained only on the decision states of that stage.

The decision rule is written to `decision_rule.json` **before** any validation
metric is computed and is refused on later runs if it changed. No production
behaviour, bundled artifact, schema, threshold or split is touched, and the test
partition is never parsed.

Usage:
    python ml/training/imitation_phase.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from training import ablation
from training import imitation as im

DEFAULT_DATA = im.DEFAULT_DATA
DEFAULT_SPLIT_CONFIG = im.DEFAULT_SPLIT_CONFIG
DEFAULT_RAW_DIR = im.DEFAULT_RAW_DIR
DEFAULT_OUT = ML_ROOT / "models" / "imitation" / "phase"
PREVIOUS_REPORT = ML_ROOT / "models" / "imitation" / "imitation_report.json"

INNER_VALIDATION_SHARE = im.INNER_VALIDATION_SHARE
POOLED_C_GRID = im.MODEL_C_GRID
EARLY_C_GRID = (0.01, 0.03, 0.1)
MODEL_MAX_ITER = im.MODEL_MAX_ITER
BOOTSTRAP_SEED = im.BOOTSTRAP_SEED

EARLY_INDICES = (0, 1)
STAGE_DEFINITIONS: Dict[str, Tuple[int, ...]] = {
    "pick0": (0,),
    "pick1": (1,),
    "picks01": (0, 1),
    "picks27": (2, 3, 4, 5, 6, 7),
    "picks89": (8, 9),
    "all": tuple(range(10)),
}

# The early model keeps only blocks that can be non-empty at its stage, plus a
# two-column acting-side bias. 280 columns for pick 0 (bans only), 558 for the
# shared 0-1 model (one pick is known at pick 1).
EARLY_BLOCKS: Dict[bool, Tuple[str, ...]] = {
    False: ("rad_ban", "dire_ban"),
    True: ("rad_pick", "dire_pick", "rad_ban", "dire_ban"),
}

DECISION_RULE = (
    "Pre-registered before any validation comparison. (1) The hybrid that uses "
    "the stage/side frequency table for picks 0-1 and the pooled model for picks "
    "2-9 is acceptable if it does not regress relative to that baseline at picks "
    "0-1 and preserves the pooled model's gains at picks 2-9. (2) A separate "
    "early-stage model counts as promising only if its validation top-1 at the "
    "target stage exceeds the stage/side frequency baseline AND the paired, "
    "match-clustered 95% interval for that difference excludes zero on the "
    "positive side. Top-1 is the primary early metric; top-3 and MRR are "
    "secondary and cannot on their own qualify a strategy. (3) Beating the "
    "pooled model's intercept-only component does not qualify a strategy. (4) If "
    "no early model qualifies, the recommendation is the frequency fallback, not "
    "a wider search. (5) A positive imitation result never establishes win-rate "
    "or counterfactual value. Thresholds: alpha = 0.05, positive delta required "
    "on top-1 at the target stage."
)

QUALIFYING_STAGES = ("pick0", "picks01")
# Every stage gets a paired, match-clustered interval: the early ones decide the
# rule, the later ones show whether the pooled model's gains survive.
BOOTSTRAP_STAGES = ("pick0", "pick1", "picks01", "picks27", "picks89", "all")


# --------------------------------------------------------------------------- #
# Pre-registration
# --------------------------------------------------------------------------- #
def pre_register(out_dir: Path) -> Dict[str, Any]:
    """Write the decision rule before any validation metric exists."""
    path = out_dir / "decision_rule.json"
    expected = {"rule": DECISION_RULE, "alpha": 0.05, "primary_metric": "top1"}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("rule") != DECISION_RULE or existing.get("alpha") != 0.05:
            raise SystemExit(
                "decision rule changed after pre-registration; refusing to run"
            )
        print(f"Decision rule: pre-registered at {existing.get('declared_at')}")
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected,
        "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "declared_before_validation_metrics": True,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Decision rule: pre-registered now -> {path}")
    return payload


# --------------------------------------------------------------------------- #
# Early-stage matrices and models
# --------------------------------------------------------------------------- #
def early_width(include_picks: bool) -> int:
    return len(EARLY_BLOCKS[include_picks]) * im.N_HEROES + 2


def build_early_matrix(
    states: Sequence[im.State], *, include_picks: bool
) -> sparse.csr_matrix:
    """Compact per-stage matrix: removals that can exist at the stage + side bias."""
    blocks = EARLY_BLOCKS[include_picks]
    width = early_width(include_picks)
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for state in states:
        columns: List[int] = [
            len(blocks) * im.N_HEROES + (1 if state.acting_radiant else 0)
        ]
        for block_index, block in enumerate(blocks):
            heroes = getattr(state, block)
            offset = block_index * im.N_HEROES
            for hero in heroes:
                index = im.HERO_INDEX.get(int(hero))
                if index is not None:
                    columns.append(offset + index)
        indices.extend(columns)
        data.extend([1.0] * len(columns))
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(states), width),
    )


def fit_logistic(X: sparse.csr_matrix, y: np.ndarray, *, C: float) -> LogisticRegression:
    model = LogisticRegression(
        C=float(C), solver="lbfgs", max_iter=MODEL_MAX_ITER, tol=1e-3
    )
    model.fit(X, y)
    return model


def aligned_probabilities(
    model: LogisticRegression, X: sparse.csr_matrix, legal: np.ndarray
) -> np.ndarray:
    """Class-aligned, legality-masked, renormalised probabilities."""
    probabilities = np.zeros((X.shape[0], im.N_HEROES), dtype=np.float64)
    probabilities[:, model.classes_.astype(np.int64)] = model.predict_proba(X)
    probabilities = np.where(legal, probabilities, 0.0)
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise SystemExit("a model assigned no mass to legal candidates")
    return probabilities / totals


def select_c(
    X_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    X_check: sparse.csr_matrix,
    y_check: np.ndarray,
    legal_check: np.ndarray,
    *,
    grid: Sequence[float],
    label: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Pick C on the internal chronological holdout (training rows only)."""
    rows: List[Dict[str, Any]] = []
    for C in grid:
        model = fit_logistic(X_fit, y_fit, C=C)
        probabilities = aligned_probabilities(model, X_check, legal_check)
        rank, logp = im.rank_and_logp(probabilities, y_check)
        entry = {"C": float(C), **im.summarise(rank, logp)}
        rows.append(entry)
        print(
            f"    {label} C={C:<6g} top1={entry['top1']:.4f} "
            f"top3={entry['top3']:.4f} logloss={entry['log_loss']:.4f}"
        )
    best = sorted(rows, key=lambda row: (-row["top1"], row["log_loss"]))[0]
    return float(best["C"]), rows


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def route(
    base: np.ndarray, replacement: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Replace `base` rows selected by `mask` with `replacement` rows."""
    out = base.copy()
    out[mask] = replacement[mask]
    return out


def stage_masks(pick_of: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        name: np.isin(pick_of, indices) for name, indices in STAGE_DEFINITIONS.items()
    }


def evaluate_stage(
    probabilities: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    match_ids: np.ndarray,
) -> Dict[str, Any]:
    rank, logp = im.rank_and_logp(probabilities[mask], truth[mask])
    return {
        **im.summarise(rank, logp),
        "matches": int(len(np.unique(match_ids[mask]))),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    started = time.time()
    out_dir = Path(args.out)
    rule = pre_register(out_dir)

    data_path = Path(args.data)
    dataset = pd.read_parquet(
        data_path, columns=["match_id", "start_time", "radiant_win"]
    )
    data_sha = im.file_sha256(data_path)
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

    frame, reasons = im.load_action_table(
        Path(args.raw_dir), Path(args.cache_dir), partitions, refresh=False
    )
    if set(frame["match_id"].astype("int64")) & test_ids:
        raise SystemExit("action table contains test-partition matches; refusing")
    states, state_reasons = im.actions_to_states(frame, partitions)
    for key, value in state_reasons.items():
        reasons[key] = reasons.get(key, 0) + value

    train_states = [s for s in states if s.partition == "train"]
    val_states = [s for s in states if s.partition == "validation"]
    if set(s.match_id for s in train_states) & set(s.match_id for s in val_states):
        raise SystemExit("a match contributes states to both partitions")
    print(
        f"States: train {len(train_states)} ({len({s.match_id for s in train_states})} matches), "
        f"validation {len(val_states)} ({len({s.match_id for s in val_states})} matches)"
    )

    X_train = im.build_matrix(train_states)
    X_val = im.build_matrix(val_states)
    y_train = im.targets(train_states)
    y_val = im.targets(val_states)
    legal_val = im.legal_matrix(val_states)
    pick_of = np.array([s.pick_index for s in val_states], dtype=np.int64)
    match_ids_val = np.array([s.match_id for s in val_states], dtype=np.int64)
    masks = stage_masks(pick_of)

    # ---- inner holdout (training matches only, chronological tail) --------
    start_times = dict(
        zip(
            dataset["match_id"].astype("int64").tolist(),
            dataset["start_time"].astype("int64").tolist(),
        )
    )
    train_matches = sorted(
        {s.match_id for s in train_states},
        key=lambda match_id: start_times.get(match_id, 0),
    )
    cut = max(1, int(round(len(train_matches) * (1.0 - INNER_VALIDATION_SHARE))))
    inner_ids = set(train_matches[:cut])
    inner_fit_idx = np.array(
        [i for i, s in enumerate(train_states) if s.match_id in inner_ids], dtype=np.int64
    )
    inner_check_idx = np.array(
        [i for i, s in enumerate(train_states) if s.match_id not in inner_ids],
        dtype=np.int64,
    )
    print(
        f"Inner holdout: {len(inner_fit_idx)} fit states, {len(inner_check_idx)} check states"
    )

    # ---- A. baseline ------------------------------------------------------
    global_scores = im.frequency_scores(train_states, by_stage_and_side=False)
    conditional_scores = im.frequency_scores(train_states, by_stage_and_side=True)
    global_val, _ = im.frequency_probabilities(
        global_scores, val_states, by_stage_and_side=False
    )
    baseline_val, unsupported = im.frequency_probabilities(
        conditional_scores, val_states, by_stage_and_side=True
    )

    # ---- B. pooled model (reproduction of session 10) ---------------------
    print("Pooled model (session-10 configuration):")
    pooled_c, pooled_grid = select_c(
        X_train[inner_fit_idx],
        y_train[inner_fit_idx],
        X_train[inner_check_idx],
        y_train[inner_check_idx],
        im.legal_matrix([train_states[i] for i in inner_check_idx]),
        grid=POOLED_C_GRID,
        label="pooled",
    )
    pooled = fit_logistic(X_train, y_train, C=pooled_c)
    pooled_val = aligned_probabilities(pooled, X_val, legal_val)
    print(f"  selected C={pooled_c:g}, refit on {X_train.shape[0]} states")

    keep_bias = sparse.diags(
        np.concatenate(
            [np.zeros(im.N_REMOVAL_FEATURES), np.ones(im.N_BIAS_FEATURES)]
        ).astype(np.float32)
    )
    intercept_val = aligned_probabilities(pooled, X_val @ keep_bias, legal_val)

    # ---- C. hybrid --------------------------------------------------------
    early_mask = masks["picks01"]
    hybrid_val = route(pooled_val, baseline_val, early_mask)

    # ---- D. early-stage models -------------------------------------------
    early_models: Dict[str, Any] = {}
    for name, indices, include_picks in (
        ("early_pick0", (0,), False),
        ("early_pick1", (1,), True),
        ("early_shared", EARLY_INDICES, True),
    ):
        selector = np.isin(np.array([s.pick_index for s in train_states]), indices)
        fit_rows = np.intersect1d(np.nonzero(selector)[0], inner_fit_idx)
        check_rows = np.intersect1d(np.nonzero(selector)[0], inner_check_idx)
        if len(fit_rows) == 0 or len(check_rows) == 0:
            raise SystemExit(f"{name}: no training states in the inner holdout")
        X_early_train = build_early_matrix(train_states, include_picks=include_picks)
        X_early_val = build_early_matrix(val_states, include_picks=include_picks)
        check_states = [train_states[i] for i in check_rows]
        print(f"  [{name}] {X_early_train.shape[1]} features:")
        chosen_c, grid = select_c(
            X_early_train[fit_rows],
            y_train[fit_rows],
            X_early_train[check_rows],
            y_train[check_rows],
            im.legal_matrix(check_states),
            grid=EARLY_C_GRID,
            label=name,
        )
        selector_all = np.isin(np.array([s.pick_index for s in train_states]), indices)
        model = fit_logistic(
            X_early_train[np.nonzero(selector_all)[0]],
            y_train[np.nonzero(selector_all)[0]],
            C=chosen_c,
        )
        probabilities = aligned_probabilities(model, X_early_val, legal_val)
        early_models[name] = {
            "C": chosen_c,
            "grid": grid,
            "features": int(X_early_train.shape[1]),
            "includes_picks": include_picks,
            "stages": list(indices),
            "train_states": int(selector_all.sum()),
            "probabilities": probabilities,
        }
        print(
            f"    selected C={chosen_c:g}, trained on {int(selector_all.sum())} states"
        )

    # ---- strategy table ---------------------------------------------------
    strategies: Dict[str, Dict[str, Any]] = {
        "stage_side_frequency": {"probabilities": baseline_val, "description": "A: baseline"},
        "pooled_model": {"probabilities": pooled_val, "description": "B: session-10 model"},
        "hybrid_freq_then_pooled": {
            "probabilities": hybrid_val,
            "description": "C: baseline at picks 0-1, pooled at 2-9",
        },
        "pooled_intercept_only": {
            "probabilities": intercept_val,
            "description": "diagnostic only: pooled model with removal columns zeroed",
        },
    }
    for name, payload in early_models.items():
        mask = np.isin(pick_of, payload["stages"])
        strategies[name] = {
            "probabilities": route(pooled_val, payload["probabilities"], mask),
            "description": f"D: {name} at its stages, pooled elsewhere",
        }

    results: Dict[str, Any] = {}
    for strategy, payload in strategies.items():
        results[strategy] = {
            stage: evaluate_stage(
                payload["probabilities"], y_val, mask, match_ids_val
            )
            for stage, mask in masks.items()
        }

    # ---- reproduction check ----------------------------------------------
    reproduction: Dict[str, Any] = {"previous_report": str(PREVIOUS_REPORT)}
    if PREVIOUS_REPORT.is_file():
        previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
        previous_overall = previous["validation"]["overall"]["model"]
        previous_pick0 = previous["validation"]["by_pick_index"]["0"]["model"]
        deltas = {
            "overall_top1": float(results["pooled_model"]["all"]["top1"] - previous_overall["top1"]),
            "pick0_top1": float(results["pooled_model"]["pick0"]["top1"] - previous_pick0["top1"]),
            "overall_brier": None,
            "previous_states": int(previous_overall["states"]),
            "current_states": int(results["pooled_model"]["all"]["states"]),
        }
        reproduction.update(
            {
                "checked": True,
                **deltas,
                "reproduced": bool(
                    abs(deltas["overall_top1"]) < 1e-9
                    and abs(deltas["pick0_top1"]) < 1e-9
                    and deltas["previous_states"] == deltas["current_states"]
                ),
            }
        )
        print(
            f"Reproduction: overall top1 delta {deltas['overall_top1']:+.2e}, "
            f"pick0 delta {deltas['pick0_top1']:+.2e}, "
            f"states {deltas['current_states']}"
        )
    else:
        reproduction["checked"] = False

    # ---- paired bootstrap ------------------------------------------------
    # Each block resamples *matches*, so the ten decisions of one draft stay
    # together; the ranks are computed once per (stage, method) and reused by all
    # three rank-based metrics.
    comparisons: Dict[str, Any] = {}
    for stage in BOOTSTRAP_STAGES:
        mask = masks[stage]
        stage_match_ids = match_ids_val[mask]
        stage_truth = y_val[mask]
        block: Dict[str, Any] = {"states": int(mask.sum())}
        ranks = {
            name: im.rank_and_logp(payload["probabilities"][mask], stage_truth)[0]
            for name, payload in strategies.items()
        }
        for metric, extractor in (
            ("top1", lambda r: (r == 0).astype(float)),
            ("top3", lambda r: (r < 3).astype(float)),
            ("mrr", lambda r: 1.0 / (r + 1)),
        ):
            per_match = {
                name: im.per_match_means(stage_match_ids, extractor(rank))
                for name, rank in ranks.items()
            }
            # Every non-reference strategy, including the intercept-only
            # diagnostic, gets a paired interval against the frequency table.
            refs = tuple(name for name in per_match if name != "stage_side_frequency")
            block[metric] = im.cluster_bootstrap(
                stage_match_ids,
                per_match,
                replicates=args.bootstrap,
                seed=BOOTSTRAP_SEED,
                comparisons=tuple((name, "stage_side_frequency") for name in refs),
            )["differences"]
        comparisons[stage] = block

    # ---- decision --------------------------------------------------------
    def qualifies(stage: str, strategy: str) -> bool:
        delta = comparisons[stage]["top1"][f"{strategy}_minus_stage_side_frequency"]
        return bool(delta["ci95"][0] > 0.0 and delta["mean"] > 0.0)

    qualified = {
        name: qualifies("picks01" if name == "early_shared" else ("pick0" if name == "early_pick0" else "pick1"), name)
        for name in early_models
    }
    best_early = max(
        early_models,
        key=lambda name: results[name][
            "picks01" if name == "early_shared" else ("pick0" if name == "early_pick0" else "pick1")
        ]["top1"],
    )
    hybrid_ok = all(
        abs(
            results["hybrid_freq_then_pooled"][stage]["top1"]
            - results["stage_side_frequency"][stage]["top1"]
        )
        < 1e-12
        for stage in ("pick0", "pick1", "picks01")
    ) and all(
        abs(
            results["hybrid_freq_then_pooled"][stage]["top1"]
            - results["pooled_model"][stage]["top1"]
        )
        < 1e-12
        # Only the stages the hybrid actually delegates matter; the aggregate is
        # *supposed* to differ, so comparing it here would always fail.
        for stage in ("picks27", "picks89")
    )
    any_early_qualified = any(qualified.values())

    if any_early_qualified:
        recommendation = "phase_aware_prototype"
    elif hybrid_ok:
        recommendation = "frequency_fallback_only"
    else:
        recommendation = "stop"

    chosen_strategy = {
        "phase_aware_prototype": best_early,
        "frequency_fallback_only": "hybrid_freq_then_pooled",
        "stop": None,
    }[recommendation]

    decision = {
        "rule": DECISION_RULE,
        "declared_at": rule.get("declared_at"),
        "early_qualifies": qualified,
        "any_early_model_qualifies": any_early_qualified,
        "best_early_model": best_early,
        "hybrid_preserves_baseline_and_gains": bool(hybrid_ok),
        "recommendation": recommendation,
        "chosen_strategy": chosen_strategy,
        "interpretation": {
            "phase_aware_prototype": "an early model beat the baseline with a positive paired interval",
            "frequency_fallback_only": (
                "no trained early-stage model qualified: route picks 0-1 to the "
                "training-only stage/side frequency table and keep the pooled "
                "model from pick 2 (strategy C); this is a routing decision, not "
                "a new model"
            ),
            "stop": "neither an early model nor the hybrid met the rule",
        }[recommendation],
    }

    report = {
        "experiment_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Phase-aware hybrid: can an early-stage model match or beat the "
            "stage/side frequency baseline at picks 0-1 without losing the pooled "
            "model's gains from pick 2? Imitation only; no win-rate claim."
        ),
        "decision_rule": rule,
        "dataset": {
            "path": str(data_path),
            "sha256": data_sha,
            "rows": int(len(dataset)),
        },
        "split": {**split_info, "test_parsed_here": False},
        "population": {
            "mode_source": "raw payload `gameMode`; not the parquet column (AUDIT P1-7)",
            "train_matches": int(len({s.match_id for s in train_states})),
            "validation_matches": int(len({s.match_id for s in val_states})),
            "train_states": len(train_states),
            "validation_states": len(val_states),
            "exclusions": reasons,
            "stage_state_counts": {
                stage: int(mask.sum()) for stage, mask in masks.items()
            },
            "stage_match_counts": {
                stage: int(len(np.unique(match_ids_val[mask])))
                for stage, mask in masks.items()
            },
        },
        "protocol": {
            "test_partition": "not parsed, not scored, not used for selection",
            "validation_use": "comparison only; exploratory, not a fresh final test",
            "inner_holdout": {
                "share": INNER_VALIDATION_SHARE,
                "fit_states": int(len(inner_fit_idx)),
                "check_states": int(len(inner_check_idx)),
                "contains_validation_matches": False,
            },
            "tie_break": "lower hero id wins",
            "masking": "legal candidates only, renormalised",
            "bootstrap": {"replicates": int(args.bootstrap), "cluster": "match_id"},
        },
        "baselines": {
            "smoothing": im.BASELINE_SMOOTHING,
            "unsupported_stage_side_cells": [list(map(str, cell)) for cell in unsupported],
        },
        "models": {
            "pooled": {
                "C": pooled_c,
                "grid": pooled_grid,
                "features": im.N_FEATURES,
                "selected_on": "internal chronological holdout (training matches only)",
            },
            "early": {
                name: {
                    key: value
                    for key, value in payload.items()
                    if key != "probabilities"
                }
                for name, payload in early_models.items()
            },
        },
        "reproduction": reproduction,
        "validation": {
            "strategies": {
                name: {
                    "description": payload["description"],
                    "by_stage": results[name],
                }
                for name, payload in strategies.items()
            },
            "bootstrap": comparisons,
        },
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---- console summary --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'strategy':<26} {'pick0':>7} {'pick1':>7} {'2-7':>7} {'8-9':>7} {'all':>7}")
    for name, payload in strategies.items():
        row = results[name]
        print(
            f"{name:<26} {row['pick0']['top1']:>7.4f} {row['pick1']['top1']:>7.4f} "
            f"{row['picks27']['top1']:>7.4f} {row['picks89']['top1']:>7.4f} "
            f"{row['all']['top1']:>7.4f}"
        )
    print("\ntop-1 deltas vs stage/side frequency (paired, match-clustered):")
    for stage in BOOTSTRAP_STAGES:
        block = comparisons[stage]["top1"]
        for name in list(early_models) + [
            "hybrid_freq_then_pooled",
            "pooled_model",
            "pooled_intercept_only",
        ]:
            delta = block[f"{name}_minus_stage_side_frequency"]
            flag = "*" if delta["ci95"][0] > 0 else " "
            print(
                f"  {stage:<8} {name:<26} {delta['mean']:+.4f} "
                f"({delta['ci95'][0]:+.4f}..{delta['ci95'][1]:+.4f}){flag}"
            )
    print(f"\nqualified early models: {qualified}")
    print(f"recommendation: {recommendation} — {decision['interpretation']}")
    print(f"Report: {report_path}")
    print(f"Total {time.time() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-aware imitation experiment (research only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ML_ROOT / "models" / "imitation",
        help="existing action cache; this script never rescans raw payloads",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
