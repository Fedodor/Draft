#!/usr/bin/env python3
"""Product-utility spike for phase-aware next-pick rankings (research only).

Session 11 established the routing: stage/side frequency for picks 0-1 and the
pooled imitation model from pick 2. This script answers a *product* question
instead of a modelling one: can that ranking be shown honestly, and how stable is
it between adjacent draft states?

It does four things and nothing else:

1. Refits the session-10 pooled model with its frozen configuration (C=0.1,
   selected on the internal training holdout) purely so per-state probabilities
   exist. No new model family, no hyperparameter search, no export.
2. Measures offline stability on the frozen validation partition: hit rates by
   stage, how often the top of the list is invalidated by the next ban or pick,
   how much the ranking of *surviving* heroes churns, and how much the contextual
   ranking differs from the frequency one.
3. Writes a self-contained HTML mockup from six uniformly sampled validation
   drafts, in a gitignored research directory. Nothing is wired into the app.
4. Records runtime and memory so the eventual UI cost is not a guess.

The test partition is never parsed. Nothing is promoted, exported or imported by
production code.

Usage:
    python ml/training/hybrid_presentation.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd

from training import ablation
from training import imitation as im
from training import imitation_phase as ph

DEFAULT_OUT = ML_ROOT / "models" / "hybrid_spike"
DEFAULT_HEROES = ML_ROOT / "datasets" / "heroes.json"

# Where the frequency table stops and the contextual model starts. Fixed by the
# session-11 pre-registered result, not re-tuned here.
ROUTE_CONTEXT_FROM = 2
# C selected on the internal training holdout in session 10; reused, not chosen.
FROZEN_C = 0.1
SAMPLE_MATCHES = 6
SAMPLE_SEED = 11
TAU_SAMPLE_MATCHES = 2000
DISPLAY_STAGES = (0, 2, 5)
SHORTLIST = 5

# The mockup must never make a claim the data cannot support. These exact phrases
# are forbidden anywhere in the generated document; the explainer uses wording
# that avoids them even while denying them.
FORBIDDEN_LABELS = (
    "best pick",
    "win probability",
    "win rate",
    "winrate",
    "counter strength",
    "optimal pick",
    "% chance of winning",
    "expected win",
)

HERO_PREFIX = "npc_dota_hero_"
# The mockup is shown to players, so internal slugs that do not match the name
# they know are mapped explicitly. Production reads `localized_name` from the
# hero database; this table covers only the research surface's 127-entry roster.
HERO_ALIASES = {
    "antimage": "Anti-Mage",
    "nevermore": "Shadow Fiend",
    "vengefulspirit": "Vengeful Spirit",
    "windrunner": "Windranger",
    "zuus": "Zeus",
    "necrolyte": "Necrophos",
    "queenofpain": "Queen of Pain",
    "skeleton_king": "Wraith King",
    "rattletrap": "Clockwerk",
    "furion": "Nature's Prophet",
    "life_stealer": "Lifestealer",
    "obsidian_destroyer": "Outworld Destroyer",
    "abyssal_underlord": "Underlord",
    "doom_bringer": "Doom",
    "treant": "Treant Protector",
    "wisp": "Io",
    "centaur": "Centaur Warrunner",
    "magnataur": "Magnus",
    "shredder": "Timbersaw",
    "keeper_of_the_light": "Keeper of the Light",
}


# --------------------------------------------------------------------------- #
# Routing and ranking primitives (pure, unit-tested)
# --------------------------------------------------------------------------- #
def routing_source(pick_index: int) -> str:
    """Which list the user would be looking at, in plain language."""
    return "observed-frequency" if pick_index < ROUTE_CONTEXT_FROM else "contextual-model"


def routing_label(pick_index: int) -> str:
    if pick_index < ROUTE_CONTEXT_FROM:
        return "Observed pick frequencies (first two picks)"
    return "Contextual model (from the third pick on)"


def hybrid_probabilities(
    baseline: np.ndarray, pooled: np.ndarray, pick_of: np.ndarray
) -> np.ndarray:
    """Frequency table for picks 0-1, pooled model from pick 2 (session 11)."""
    hybrid = baseline.copy()
    mask = pick_of >= ROUTE_CONTEXT_FROM
    hybrid[mask] = pooled[mask]
    return hybrid


def top_k(probabilities: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, ties broken by lower index (hero id)."""
    order = np.argsort(-probabilities, axis=1, kind="stable")
    return order[:, :k]


def survivor_ranks(
    scores: np.ndarray, survivors: np.ndarray, hero_ids: np.ndarray
) -> np.ndarray:
    """Ranks 0..n-1 of `scores` restricted to `survivors`, lower hero id first."""
    heroes = hero_ids[survivors]
    values = scores[survivors]
    order = np.lexsort((heroes, -values))
    rank = np.empty(len(heroes), dtype=np.int64)
    rank[order] = np.arange(len(heroes))
    return rank


def kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Kendall tau-b over a permutation pair (no ties by construction)."""
    n = len(rank_a)
    if n < 2:
        return 1.0
    if len(rank_b) != n:
        raise ValueError("rank vectors must have the same length")
    perm = rank_b[np.argsort(rank_a)]
    inversions = int(np.triu(perm[:, None] > perm[None, :], 1).sum())
    return 1.0 - 4.0 * inversions / (n * (n - 1))


def top1_survived(previous_scores: np.ndarray, current_scores: np.ndarray,
                  survivors: np.ndarray, hero_ids: np.ndarray) -> bool:
    """Was the previous #1 among survivors still #1 on the current scores?"""
    if survivors.sum() == 0:
        return True
    a = survivor_ranks(previous_scores, survivors, hero_ids)
    b = survivor_ranks(current_scores, survivors, hero_ids)
    return bool(np.argmin(a) == np.argmin(b))


# --------------------------------------------------------------------------- #
# Hero labels
# --------------------------------------------------------------------------- #
def sample_match_ids(match_ids: Sequence[int], count: int, seed: int) -> List[int]:
    """Deterministic uniform sample of match ids (sorted, not accuracy-selected)."""
    ids = list(match_ids)
    chosen = np.random.default_rng(seed).choice(
        ids, size=min(count, len(ids)), replace=False
    )
    return sorted(int(match_id) for match_id in chosen)


def load_heroes(path: Path) -> Dict[int, Dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    heroes: Dict[int, Dict[str, str]] = {}
    for key, meta in raw.items():
        slug = str(meta.get("name", ""))
        if slug.startswith(HERO_PREFIX):
            slug = slug[len(HERO_PREFIX):]
        heroes[int(key)] = {
            "name": HERO_ALIASES.get(slug, slug.replace("_", " ").title()),
            "attr": str(meta.get("primary_attr", "")),
        }
    return heroes


def hero_name(heroes: Mapping[int, Dict[str, str]], hero_id: int) -> str:
    meta = heroes.get(int(hero_id))
    return meta["name"] if meta else f"Hero {int(hero_id)}"


# --------------------------------------------------------------------------- #
# Offline stability
# --------------------------------------------------------------------------- #
def _as_tensor(array: np.ndarray, n_matches: int) -> np.ndarray:
    return array.reshape(n_matches, im.EXPECTED_PICKS, im.N_HEROES)


def transition_metrics(
    probabilities: np.ndarray,
    legal: np.ndarray,
    n_matches: int,
    *,
    tau_matches: np.ndarray,
    hero_ids: np.ndarray,
) -> List[Dict[str, Any]]:
    """Per-transition churn: what the next action removes and what it reorders."""
    probs = _as_tensor(probabilities, n_matches)
    masks = _as_tensor(legal.astype(bool), n_matches)
    rows = np.arange(n_matches)
    tau_rows = np.zeros(n_matches, dtype=bool)
    tau_rows[tau_matches] = True

    out: List[Dict[str, Any]] = []
    for step in range(im.EXPECTED_PICKS - 1):
        current, nxt = probs[:, step, :], probs[:, step + 1, :]
        legal_current, legal_next = masks[:, step, :], masks[:, step + 1, :]
        removed = legal_current & ~legal_next
        survivors = legal_current & legal_next

        head = top_k(current, 1)
        head_ids = head[:, 0]
        head_removed = removed[rows, head_ids]
        top3 = top_k(current, 3)
        top3_removed = np.take_along_axis(removed, top3, axis=1).any(axis=1)

        survivor_count = survivors.sum(axis=1)
        masked_current = np.where(survivors, current, -np.inf)
        masked_next = np.where(survivors, nxt, -np.inf)
        take = np.minimum(SHORTLIST, survivor_count)
        idx_current = top_k(masked_current, SHORTLIST)
        idx_next = top_k(masked_next, SHORTLIST)
        valid = np.arange(SHORTLIST)[None, :] < take[:, None]
        same = (idx_current[:, :, None] == idx_next[:, None, :]).any(axis=2)
        overlap = float((same & valid).sum() / max(valid.sum(), 1))
        retained = float(
            (idx_current[:, 0] == idx_next[:, 0]).mean()
        )

        taus = []
        for row in np.nonzero(tau_rows)[0]:
            if survivor_count[row] < 2:
                continue
            a = survivor_ranks(current[row], survivors[row], hero_ids)
            b = survivor_ranks(nxt[row], survivors[row], hero_ids)
            taus.append(kendall_tau(a, b))

        out.append(
            {
                "transition": f"pick {step} to pick {step + 1}",
                "from_pick": step,
                "states": int(n_matches),
                "removed_mean": float(removed.sum(axis=1).mean()),
                "top1_removed_share": float(head_removed.mean()),
                "top3_any_removed_share": float(top3_removed.mean()),
                "survivors_mean": float(survivor_count.mean()),
                "top1_retained_share": retained,
                "top5_overlap_share": overlap,
                "kendall_tau_mean": float(np.mean(taus)) if taus else None,
                "kendall_tau_matches": int(len(taus)),
            }
        )
    return out


def same_state_comparison(
    hybrid: np.ndarray,
    baseline: np.ndarray,
    legal: np.ndarray,
    pick_of: np.ndarray,
    *,
    tau_rows: np.ndarray,
    hero_ids: np.ndarray,
) -> Dict[str, Any]:
    """How different is the contextual ranking from the frequency ranking?"""
    head_h = top_k(hybrid, 1)[:, 0]
    head_b = top_k(baseline, 1)[:, 0]
    top5_h = top_k(hybrid, SHORTLIST)
    top5_b = top_k(baseline, SHORTLIST)
    same = (top5_h[:, :, None] == top5_b[:, None, :]).any(axis=2)

    taus = []
    for row in np.nonzero(tau_rows)[0]:
        legal_row = legal[row]
        if legal_row.sum() < 2:
            continue
        taus.append(
            kendall_tau(
                survivor_ranks(hybrid[row], legal_row, hero_ids),
                survivor_ranks(baseline[row], legal_row, hero_ids),
            )
        )

    by_stage: Dict[str, Dict[str, float]] = {}
    for stage, indices in ph.STAGE_DEFINITIONS.items():
        if stage == "all":
            continue
        mask = np.isin(pick_of, indices)
        by_stage[stage] = {
            "states": int(mask.sum()),
            "top1_agreement": float((head_h[mask] == head_b[mask]).mean()),
            "top5_overlap": float(same[mask].mean()),
        }
    return {
        "top1_agreement_overall": float((head_h == head_b).mean()),
        "top5_overlap_overall": float(same.mean()),
        "kendall_tau_mean": float(np.mean(taus)) if taus else None,
        "kendall_states": int(len(taus)),
        "by_stage": by_stage,
    }


# --------------------------------------------------------------------------- #
# Mockup
# --------------------------------------------------------------------------- #
def build_snapshot(
    *,
    draft_index: int,
    state: im.State,
    probabilities: np.ndarray,
    legal: np.ndarray,
    heroes: Mapping[int, Dict[str, str]],
    timeline: Sequence[Mapping[str, Any]],
    previous_head: int | None,
) -> Dict[str, Any]:
    # top_k returns column indices, which are hero_index = hero_id - HERO_MIN.
    order = top_k(probabilities[None, :], SHORTLIST)[0]
    candidates = []
    for position, hero_index in enumerate(order):
        hero_id = int(hero_index) + im.HERO_MIN
        candidates.append(
            {
                "rank": position + 1,
                "hero_id": hero_id,
                "hero": hero_name(heroes, hero_id),
                "attr": heroes.get(hero_id, {}).get("attr", ""),
                "tier": position + 1,
            }
        )
    head_id = int(order[0]) + im.HERO_MIN
    return {
        "draft_index": draft_index,
        "match_id": int(state.match_id),
        "pick_index": int(state.pick_index),
        "acting_side": "Radiant" if state.acting_radiant else "Dire",
        "source": routing_source(state.pick_index),
        "source_label": routing_label(state.pick_index),
        "legal_candidates": int(legal.sum()),
        "candidates": candidates,
        "previous_head": None if previous_head is None else hero_name(heroes, previous_head),
        "previous_head_still_legal": bool(
            previous_head is not None and legal[im.HERO_INDEX[previous_head]]
        ),
        "observed_target": hero_name(heroes, state.target),
        "timeline": list(timeline),
        "head_is_observed": bool(head_id == int(state.target)),
    }


def render_mockup(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    stability: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> str:
    """Self-contained HTML; no scripts, no network, no numeric probabilities."""
    blocks = []
    for snapshot in snapshots:
        timeline = "".join(
            f'<li class="act {entry["kind"]} {entry["side"].lower()}">'
            f'<span class="tag">{entry["label"]}</span>'
            f'<span class="who">{html.escape(entry["hero"])}</span></li>'
            for entry in snapshot["timeline"]
        )
        shortlist = "".join(
            f'<li class="cand"><span class="pos">{candidate["rank"]}</span>'
            f'<span class="name">{html.escape(candidate["hero"])}</span>'
            f'<span class="attr">{html.escape(candidate["attr"])}</span>'
            f'<span class="bar tier-{candidate["tier"]}"></span></li>'
            for candidate in snapshot["candidates"]
        )
        note = ""
        if snapshot["previous_head"] and not snapshot["previous_head_still_legal"]:
            note = (
                f'<p class="note warn">The top entry from the previous pick '
                f'({html.escape(snapshot["previous_head"])}) is no longer available.</p>'
            )
        blocks.append(
            f"""
        <section class="draft">
          <header>
            <h3>Example draft {snapshot["draft_index"]} &middot; pick {snapshot["pick_index"] + 1} of 10</h3>
            <span class="badge">{html.escape(snapshot["source_label"])}</span>
          </header>
          <p class="context">
            {snapshot["acting_side"]} to act &middot;
            {snapshot["legal_candidates"]} legal heroes left
          </p>
          <ol class="timeline">{timeline}</ol>
          {note}
          <ol class="shortlist">{shortlist}</ol>
          <p class="foot">
            Ordering only: {html.escape(snapshot["source"])}. Bar length is a
            qualitative tier, not a score.
          </p>
        </section>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Next-pick rankings — research mockup</title>
<style>
  :root {{
    --bg:#0f1419; --panel:#161d25; --line:#26313d; --text:#e6edf3;
    --muted:#93a4b4; --radiant:#4aa96c; --dire:#c0504d; --accent:#5b9bd5;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px; background:var(--bg); color:var(--text);
    font:14px/1.5 "Segoe UI", system-ui, sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 6px; }}
  h2 {{ font-size:15px; margin:28px 0 10px; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted); }}
  h3 {{ font-size:15px; margin:0; }}
  .wrap {{ max-width:860px; margin:0 auto; }}
  .banner {{ border:1px solid #6b4a12; background:#241a08; color:#f0c674;
    padding:12px 14px; border-radius:8px; margin:16px 0 8px; }}
  .explain {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px 16px; margin:12px 0 4px; }}
  .explain p {{ margin:0 0 8px; }}
  .explain p:last-child {{ margin:0; }}
  .draft {{ background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:16px; margin:12px 0; }}
  .draft header {{ display:flex; align-items:baseline; gap:12px; }}
  .badge {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em;
    border:1px solid var(--accent); color:var(--accent); border-radius:99px;
    padding:2px 9px; }}
  .context {{ color:var(--muted); margin:6px 0 10px; font-size:12px; }}
  .timeline {{ list-style:none; display:flex; flex-wrap:wrap; gap:4px;
    padding:0; margin:0 0 12px; }}
  .act {{ border:1px solid var(--line); border-radius:4px; padding:2px 6px;
    font-size:11px; display:flex; gap:5px; align-items:center; }}
  .act.ban {{ opacity:.55; }}
  .act.radiant .tag {{ color:var(--radiant); }}
  .act.dire .tag {{ color:var(--dire); }}
  .tag {{ text-transform:uppercase; letter-spacing:.05em; }}
  .shortlist {{ list-style:none; padding:0; margin:0; }}
  .cand {{ display:grid; grid-template-columns:22px 1fr 52px 120px;
    gap:10px; align-items:center; padding:7px 0;
    border-top:1px solid var(--line); }}
  .pos {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
  .name {{ font-weight:600; }}
  .attr {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
  .bar {{ height:6px; border-radius:99px; background:var(--accent);
    transform-origin:left; }}
  .tier-1 {{ transform:scaleX(1.00); }}
  .tier-2 {{ transform:scaleX(.82); }}
  .tier-3 {{ transform:scaleX(.66); }}
  .tier-4 {{ transform:scaleX(.52); }}
  .tier-5 {{ transform:scaleX(.40); }}
  .foot {{ color:var(--muted); font-size:11px; margin:10px 0 0; }}
  .note {{ font-size:12px; margin:0 0 10px; }}
  .note.warn {{ color:#e0a458; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  footer {{ color:var(--muted); font-size:11px; margin-top:28px;
    border-top:1px solid var(--line); padding-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Next-pick rankings</h1>
  <p class="context">Research mockup &middot; not shipped &middot; no production code path</p>

  <div class="banner">
    These lists show which heroes captains <em>commonly took</em> from a similar
    draft in roughly 75,000 historical matches. They are not a prediction of who
    wins, and the first entry is not "the right pick" for your team.
  </div>

  <div class="explain">
    <p><strong>First two picks:</strong> the list is a plain frequency table &mdash;
      what captains picked most often from the same draft phase and side, with
      already-picked and already-banned heroes removed.</p>
    <p><strong>From the third pick on:</strong> a regularised model also looks at
      <em>who has already been removed</em> and shifts the list accordingly.</p>
    <p>Every list excludes heroes that are already picked or banned. Bans appear
      only once they have happened in the draft.</p>
  </div>

  <h2>Sample drafts</h2>
  {"".join(blocks)}

  <h2>How stable is the list?</h2>
  <table>
    <tr><th>Check (validation drafts)</th><th>Value</th></tr>
    <tr><td>Top entry removed by the very next ban or pick</td>
        <td>{stability["top1_removed_share"]}</td></tr>
    <tr><td>Top entry still first among the remaining heroes after the next action</td>
        <td>{stability["top1_retained_share"]}</td></tr>
    <tr><td>Top five overlap with the previous pick, among remaining heroes</td>
        <td>{stability["top5_overlap_share"]}</td></tr>
    <tr><td>Contextual list vs frequency list, same draft state (top-1 agreement)</td>
        <td>{stability["model_vs_baseline_top1"]}</td></tr>
  </table>
  <p class="foot">
    Stability is not quality. A list that never moves can still be wrong, and a
    list that moves may simply be reacting to a ban.
  </p>

  <footer>
    Coverage: {coverage["matches"]} validation drafts, {coverage["states"]} decision
    states, frozen split, test partition never parsed. Metrics are exploratory:
    this validation window has been used by earlier experiments.
  </footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    n_matches = len({s.match_id for s in val_states})
    pick_of = np.array([s.pick_index for s in val_states], dtype=np.int64)
    # The tensor maths needs one contiguous block of ten states per match; verify
    # rather than assume, and abort if the state builder ever changes that.
    expected = np.tile(np.arange(im.EXPECTED_PICKS), n_matches)
    if not np.array_equal(pick_of, expected) or len(val_states) != n_matches * 10:
        raise SystemExit("validation states are not grouped as ten per match")

    print(
        f"Validation: {n_matches} matches, {len(val_states)} states; "
        f"dataset sha={data_sha[:12]}…"
    )

    legal_val = im.legal_matrix(val_states)
    y_val = im.targets(val_states)
    X_val = im.build_matrix(val_states)

    print("Fitting the frozen pooled configuration (C=0.1)…")
    tracemalloc.start()
    fit_started = time.time()
    model = ph.fit_logistic(im.build_matrix(train_states), im.targets(train_states), C=FROZEN_C)
    fit_seconds = time.time() - fit_started
    pooled_val = ph.aligned_probabilities(model, X_val, legal_val)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    scores = im.frequency_scores(train_states, by_stage_and_side=True)
    baseline_val, unsupported = im.frequency_probabilities(
        scores, val_states, by_stage_and_side=True
    )
    hybrid_val = hybrid_probabilities(baseline_val, pooled_val, pick_of)

    # ---- latency -----------------------------------------------------------
    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        ph.aligned_probabilities(model, X_val, legal_val)
        latencies.append(time.perf_counter() - start)
    seconds = float(np.median(latencies))

    # A draft update scores one state, so the per-call cost (including the
    # estimator's own overhead) is the number a UI would actually pay.
    single_row = X_val[:1]
    single_started = time.perf_counter()
    for _ in range(2000):
        model.predict_proba(single_row)
    single_row_ms = (time.perf_counter() - single_started) / 2000.0 * 1000.0

    # ---- hit rates (recomputed, must match the session-11 report) ----------
    hit_rates: Dict[str, Any] = {}
    for name, probabilities in (
        ("hybrid", hybrid_val),
        ("pooled_model", pooled_val),
        ("stage_side_frequency", baseline_val),
    ):
        stage_rows: Dict[str, Any] = {}
        for stage, indices in ph.STAGE_DEFINITIONS.items():
            mask = np.isin(pick_of, indices)
            rank, _ = im.rank_and_logp(probabilities[mask], y_val[mask])
            stage_rows[stage] = {
                "states": int(mask.sum()),
                "top1": float((rank == 0).mean()),
                "top3": float((rank < 3).mean()),
            }
        hit_rates[name] = stage_rows

    # ---- stability ---------------------------------------------------------
    hero_ids = im.ROSTER
    rng = np.random.default_rng(SAMPLE_SEED)
    tau_rows = np.sort(
        rng.choice(n_matches, size=min(TAU_SAMPLE_MATCHES, n_matches), replace=False)
    )
    transitions = transition_metrics(
        hybrid_val, legal_val, n_matches, tau_matches=tau_rows, hero_ids=hero_ids
    )
    comparison = same_state_comparison(
        hybrid_val, baseline_val, legal_val, pick_of, tau_rows=tau_rows, hero_ids=hero_ids
    )
    removed = [row["top1_removed_share"] for row in transitions]
    retained = [row["top1_retained_share"] for row in transitions]
    overlaps = [row["top5_overlap_share"] for row in transitions]
    taus = [row["kendall_tau_mean"] for row in transitions if row["kendall_tau_mean"] is not None]

    stability_summary = {
        "top1_removed_mean": float(np.mean(removed)),
        "top1_retained_mean": float(np.mean(retained)),
        "top5_overlap_mean": float(np.mean(overlaps)),
        "kendall_tau_mean": float(np.mean(taus)) if taus else None,
        "model_vs_baseline_top1_agreement": comparison["top1_agreement_overall"],
        "model_vs_baseline_top5_overlap": comparison["top5_overlap_overall"],
        "model_vs_baseline_kendall_tau": comparison["kendall_tau_mean"],
    }

    # ---- mockup snapshots --------------------------------------------------
    heroes = load_heroes(Path(args.heroes))
    validation_rows = val_states
    match_order = [s.match_id for s in validation_rows[::im.EXPECTED_PICKS]]
    sample_ids = sample_match_ids(match_order, SAMPLE_MATCHES, SAMPLE_SEED)
    sample_set = set(sample_ids)
    timeline_by_match: Dict[int, List[Dict[str, Any]]] = {}
    for row in frame.itertuples(index=False):
        match_id = int(row.match_id)
        if match_id not in sample_set:
            continue
        entries: List[Dict[str, Any]] = []
        for index in range(int(row.n_actions)):
            kind = getattr(row, f"a{index}_type")
            side = getattr(row, f"a{index}_side")
            hero = getattr(row, f"a{index}_hero")
            if hero is None or int(hero) == 0:
                continue
            entries.append(
                {
                    "kind": "pick" if kind == 1 else "ban",
                    "side": "Radiant" if side == 1 else "Dire",
                    "label": "pick" if kind == 1 else "ban",
                    "hero": hero_name(heroes, int(hero)),
                    "hero_id": int(hero),
                }
            )
        timeline_by_match[match_id] = entries

    position = {match_id: index for index, match_id in enumerate(match_order)}
    snapshots: List[Dict[str, Any]] = []
    excluded_stage_notes: List[str] = []
    for draft_index, match_id in enumerate(sample_ids, start=1):
        base = position[match_id] * im.EXPECTED_PICKS
        for pick_index in DISPLAY_STAGES:
            state = val_states[base + pick_index]
            prefix: List[Dict[str, Any]] = []
            # Rebuild the true interleaved prefix from the action table.
            seen_picks = 0
            for entry in timeline_by_match[match_id]:
                if entry["kind"] == "pick":
                    if seen_picks == pick_index:
                        break
                    seen_picks += 1
                prefix.append(entry)
            previous_head = None
            if pick_index > 0:
                previous = hybrid_val[base + pick_index - 1]
                previous_head = int(top_k(previous[None, :], 1)[0][0]) + im.HERO_MIN
            snapshots.append(
                build_snapshot(
                    draft_index=draft_index,
                    state=state,
                    probabilities=hybrid_val[base + pick_index],
                    legal=legal_val[base + pick_index],
                    heroes=heroes,
                    timeline=prefix,
                    previous_head=previous_head,
                )
            )

    stability_display = {
        "top1_removed_share": f"{stability_summary['top1_removed_mean'] * 100:.0f}%",
        "top1_retained_share": f"{stability_summary['top1_retained_mean'] * 100:.0f}%",
        "top5_overlap_share": f"{stability_summary['top5_overlap_mean'] * 100:.0f}%",
        "model_vs_baseline_top1": (
            f"{stability_summary['model_vs_baseline_top1_agreement'] * 100:.0f}%"
        ),
    }
    coverage = {
        "matches": n_matches,
        "states": len(val_states),
    }
    document = render_mockup(snapshots, stability=stability_display, coverage=coverage)

    lowered = document.lower()
    leaked = [phrase for phrase in FORBIDDEN_LABELS if phrase in lowered]
    if leaked:
        raise SystemExit(f"mockup contains a forbidden claim: {leaked}")

    (out_dir / "mockup.html").write_text(document, encoding="utf-8")
    (out_dir / "snapshots.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "experiment_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Product-utility spike: can phase-aware next-pick rankings be shown "
            "honestly, and how stable are they between adjacent draft states? "
            "Imitation only; no win-rate, quality or counterfactual claim."
        ),
        "dataset": {"path": str(data_path), "sha256": data_sha, "rows": int(len(dataset))},
        "split": {**split_info, "test_parsed_here": False},
        "population": {
            "train_states": len(train_states),
            "validation_states": len(val_states),
            "validation_matches": n_matches,
            "exclusions": reasons,
            "unsupported_stage_side_cells": [list(map(str, cell)) for cell in unsupported],
        },
        "routing": {
            "frequency_pick_indices": [0, 1],
            "contextual_from_pick_index": ROUTE_CONTEXT_FROM,
            "frozen": "session 11 pre-registered result; not re-selected here",
        },
        "model": {
            "family": "multinomial logistic regression",
            "C": FROZEN_C,
            "C_source": "session-10 internal chronological holdout; reused, not re-chosen",
            "features": im.N_FEATURES,
            "classes": int(len(model.classes_)),
            "train_states": int(len(train_states)),
            "fit_seconds": fit_seconds,
            "predict_seconds_full_validation": seconds,
            "predict_ms_per_1k_states": seconds * 1000.0 / (len(val_states) / 1000.0),
            "predict_us_per_state_batch": seconds * 1e6 / len(val_states),
            "predict_single_row_ms": single_row_ms,
            "predict_single_row_calls": 2000,
            "deployed_parameter_mb": (
                (im.N_FEATURES * len(model.classes_) + len(model.classes_)) * 8 / 1e6
            ),
            "probability_matrix_mb": hybrid_val.nbytes / 1e6,
            "analysis_tracemalloc_peak_mb": peak / 1e6,
        },
        "hit_rates": hit_rates,
        "stability": {
            "summary": stability_summary,
            "transitions": transitions,
            "tau_sample_matches": int(len(tau_rows)),
            "model_vs_baseline": comparison,
        },
        "mockup": {
            "path": str(out_dir / "mockup.html"),
            "sample_matches": sample_ids,
            "seed": SAMPLE_SEED,
            "stages_shown": list(DISPLAY_STAGES),
            "forbidden_labels_checked": list(FORBIDDEN_LABELS),
            "selection_rule": "uniform sample of validation match ids; not chosen for accuracy",
        },
        "limitations": [
            "Validation only, and this partition has been reused by earlier experiments.",
            "Imitation of observed choices; says nothing about win rate or strategic quality.",
            "Fit for analysis only; no artifact exported, promoted or imported by the app.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nHit rates (validation)")
    for name, stages in hit_rates.items():
        print(
            f"  {name:<22} pick0 {stages['pick0']['top1']:.4f} "
            f"pick1 {stages['pick1']['top1']:.4f} "
            f"2-7 {stages['picks27']['top1']:.4f} "
            f"8-9 {stages['picks89']['top1']:.4f} "
            f"all {stages['all']['top1']:.4f}"
        )
    print("\nStability (hybrid ranking, adjacent states)")
    print(f"  top1 removed by next action : {stability_summary['top1_removed_mean']:.4f}")
    print(f"  top1 retained among survivors: {stability_summary['top1_retained_mean']:.4f}")
    print(f"  top5 overlap                : {stability_summary['top5_overlap_mean']:.4f}")
    print(f"  Kendall tau                 : {stability_summary['kendall_tau_mean']}")
    print(
        f"  model vs baseline top1 agreement: "
        f"{stability_summary['model_vs_baseline_top1_agreement']:.4f}"
    )
    print(
        f"\nFit {fit_seconds:.1f}s, batch predict "
        f"{seconds * 1000 / (len(val_states) / 1000):.2f} ms per 1k states "
        f"({seconds * 1e6 / len(val_states):.1f} us/state), "
        f"single-state predict {single_row_ms:.3f} ms, "
        f"analysis peak {peak / 1e6:.0f} MB"
    )
    print(f"Mockup: {out_dir / 'mockup.html'}")
    print(f"Report: {out_dir / 'report.json'}")
    print(f"Total {time.time() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-aware ranking product-utility spike (research only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=im.DEFAULT_DATA)
    parser.add_argument("--split-config", type=Path, default=im.DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=im.DEFAULT_RAW_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ML_ROOT / "models" / "imitation",
        help="existing action cache; this script never rescans raw payloads",
    )
    parser.add_argument("--heroes", type=Path, default=DEFAULT_HEROES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
