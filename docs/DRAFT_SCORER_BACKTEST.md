# Deterministic scorer backtest against the historical corpus

Offline, read-only evaluation of the shipping heuristic scorer (`ml_enabled =
false`) on every Captains Mode draft with a usable pick/ban order in
`ml/datasets/raw/`. No model weights, thresholds, seed data or engine code were
changed to produce these numbers.

The short version: **the deterministic scorer does not beat chance at predicting
human picks.** A plain popularity baseline is already ~2.3x better at top-1 with
a single prior fitted on the warm-up window, and ~3.6x better with the prior
refreshed per patch. The scorer's ordering is dominated by the seeded per-hero
meta score; ranked on their own, composition and matchup+synergy carry only a
barely detectable ordering signal and matchup+synergy is worse than chance at
top-1.

The ban phase used to show no suggestions at all: the same pick cards stayed on
screen and named the banned hero 1.28% of the time. The app now ships a
per-patch ban-prevalence list for that phase (see [Ban phase](#ban-phase-what-the-app-shows-during-bans)),
measured at 4.41% top-1 with a rolling prior and 4.78% with the shipped
full-patch table. That is agreement with what teams ban, not evidence that the
suggested bans are good.
Gating the engine behind a fresh prior's shortlist ties that prior rather than
beating it (3.41% vs 3.38%, McNemar z = 0.4).

Refreshing the popularity prior is the lever that actually moves the numbers:
recomputing it per patch as the replay advances lifts popularity alone from
2.21% to 3.38% top-1, more than any blend of it with the scorer. Blending helps
against the scorer (0.94% → 2.53%) but its edge over popularity does not survive
a fresh prior. Nothing here is shipped.

The app ships with `ml_enabled = false` by default (`default_ml_enabled()` in
`models.rs`), so this is the path users actually get out of the box. The ML-on
path was measured too, on a matched sample, and is statistically
indistinguishable from it (see [ML enabled vs disabled](#ml-enabled-vs-disabled)).

## Dataset

| item | value |
| --- | --- |
| raw payloads scanned | 111,947 |
| Captains Mode drafts exported | 90,020 |
| patches | 7.30 – 7.41 |
| first / last start | 2022-01-15 / 2026-08-28 |
| unknown hero ids | 0 |
| sha256 of `drafts_cm.jsonl` | `571fd7274f49c9fac6a35f685c485949c5960d7283dd03450c528df88af4bb3d` |

`ml/backtest/export_drafts.py` reads both raw payload shapes present in the
corpus (STRATZ `gameMode: "CAPTAINS_MODE"` + `pickBans`, and OpenDota
`game_mode: 2` + `picks_bans`), keeps only drafts with a contiguous action
order, exactly 10 picks (5 per side), at least 10 bans and no repeated hero, and
joins `ml/datasets/processed_v2/draft_features.parquet` for patch and start
time. The parquet alone is not enough: it stores unordered presence flags, while
the replay needs the exact sequence.

Sampling: drafts are sorted by start time and 20,000 evenly spaced ones are
replayed (stride 90,020 / 20,000 ≈ 4.5; 199,435 pick events). The first 60% of
the sample (12,000 drafts) is warm-up for the popularity baseline; the remaining
8,000 drafts (79,764 pick events) are the evaluation slice. Even spacing is used
instead of a prefix so all patches are represented.

## Harness

`apps/desktop/src-tauri/examples/backtest_scorer.rs` drives the real engine:
`DraftEngine::initialize`, `reset`, `add_pick`/`add_ban` over the historical
order, and `score_breakdown(hero, team).total` for every hero still available.
The database is copied (with `-wal`/`-shm`) to a temp directory first, because
`Database::new` migrates and opens WAL, and the copy is discarded afterwards.
Close the desktop app before a run: a live writer can leave the copied WAL
inconsistent with the main file.

In the JSON, `db.matchups` and `db.synergies` are counts of *heroes* that have
rows (120 and 122), not row counts; the row counts above come from the database.

Correctness check: on 200 sampled pick events the harness's top-5 was compared
with `DraftEngine::get_recommendations(team, 5).top_picks`, i.e. the list the app
actually shows. **0 mismatches.**

Database used: 126 heroes, 9,052 matchup rows (120 heroes), 7,381 synergy rows
(122 heroes), active patch `7.41`, meta score sd 0.218.

Caveat on the seed: `infrastructure/seeders/scoring_seed.json` was generated
from `ml/datasets/processed` over the whole corpus, which includes the
evaluation era, and matchup advantage is `winrate - 0.5` from all-mode data.
Any agreement figure here is therefore an *optimistic* reference, not a clean
held-out result. The numbers below are low despite that, not because of it.

## Agreement with human picks (evaluation slice)

| metric | engine | popularity (global warm-up) | popularity (rolling per-patch) | random |
| --- | --- | --- | --- | --- |
| top-1 | 0.94% | 2.21% | 3.38% | 0.90% (measured) |
| top-3 | 3.2% | 6.0% | 9.0% | 2.70% |
| top-5 | 6.0% | 9.8% | 13.8% | 4.50% |
| mean rank of pick | 53.2 | 40.2 | 34.5 | 56.2 |
| mean percentile of pick | 0.527 | 0.644 | 0.695 | 0.500 |

The rolling per-patch prior is defined and measured in
[Blending the popularity prior](#blending-the-popularity-prior-into-the-ranking);
its column is the same evaluation slice and events as the other two. Every later
table that uses popularity says which prior it used — the ablation and ML
sections were run before the per-patch prior existed, so they quote the global
one.

Whole sample (all 20,000 drafts) is the same story: top-1 0.96%, top-3 3.0%,
top-5 5.5%, mean percentile 0.520. The engine's top-1 lift over random is
+0.06 percentage points; the global popularity prior's is +1.31.

## By draft phase

Engine, whole sample (opening = picks 1–2, early = 3–4, mid = 5–6, late = 7–8,
final = 9–10):

| phase | events | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- | --- |
| opening | 39,935 | 1.58% | 4.9% | 9.2% | 0.537 |
| early | 39,914 | 1.03% | 3.2% | 5.4% | 0.501 |
| mid | 39,833 | 0.81% | 2.7% | 4.9% | 0.513 |
| late | 39,887 | 0.66% | 2.2% | 4.1% | 0.534 |
| final | 39,866 | 0.72% | 2.2% | 3.8% | 0.518 |

Popularity baseline, evaluation slice. Top-1 is given for both priors; the
remaining columns use the rolling per-patch prior, which is the stronger one:

| phase | events | top-1 (global) | top-1 (per-patch) | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- | --- | --- |
| opening | 15,969 | 4.31% | 6.20% | 14.3% | 21.1% | 0.750 |
| early | 15,967 | 3.19% | 4.79% | 11.7% | 17.4% | 0.724 |
| mid | 15,929 | 1.80% | 3.01% | 8.8% | 13.6% | 0.696 |
| late | 15,949 | 0.88% | 1.76% | 6.3% | 9.9% | 0.672 |
| final | 15,950 | 0.85% | 1.12% | 4.2% | 6.9% | 0.633 |

The engine is strongest in the opening (where the popularity prior is also
strongest) and decays as the draft fills in. In the final phase the two converge
and both sit below the random floor (0.72% and 0.85% against 0.97%), which is
expected once only ~50 heroes remain and no ranking can beat listing every
remaining hero.

Per pick ordinal, engine top-1: 1.59, 1.57, 1.00, 1.06, 0.85, 0.77, 0.65, 0.66,
0.74, 0.70 percent.

## Ban-aware outcome proxy

Win rate of the side that made the pick, bucketed by the engine's rank for that
pick (Wilson 95% CI). These aggregates and the AUCs below cover all 20,000
replayed drafts, not only the evaluation slice — nothing is fitted here, so
including the warm-up era would not bias them:

| bucket | picks | win rate | 95% CI |
| --- | --- | --- | --- |
| rank 1 | 1,915 | 49.7% | 47.5 – 52.0 |
| rank 2–3 | 4,152 | 50.6% | 49.1 – 52.1 |
| rank 4–10 | 15,590 | 50.8% | 50.1 – 51.6 |
| rank > 10 | 177,778 | 49.9% | 49.7 – 50.1 |

There is no monotone relationship, and the rank-1 bucket's CI comfortably
contains 50%. Team-level AUC from summing per-pick engine scores: 0.528
(18,083 matches); from summing percentiles: 0.522. A scorer with usable outcome
signal would sit clearly above 0.55.

Ban threat (how often a ban removes the banning team's own top-ranked hero,
278,484 ban actions):

| slice | bans | rank-1 banned | rank-3 or better |
| --- | --- | --- | --- |
| all bans | 278,484 | 1.47% | 3.72% |
| opening bans (0 picks made) | 137,400 | 2.01% | 4.69% |
| bans after picks | 141,084 | 0.95% | 2.77% |

With ~120 heroes available, chance is ~0.8% for rank-1. Opening bans are ~2.5x
chance, later bans are at chance. So teams do occasionally ban what the scorer
likes, but no more than that: the scorer is not tracking the ban phase.

## Ban phase: what the app shows during bans

The proxy above asks whether bans remove heroes the engine likes. This section
asks what a user sees while banning, and what that list is worth.

**Shipped behaviour (schema v3).** `hero_ban_stats` stores `bans`/`games` per
patch, seeded from the same exported corpus with
`ml/backtest/export_ban_prior.py` into `infrastructure/seeders/ban_seed.json`
(1,487 rows, 12 patches). `commands.rs::get_ban_suggestions` ranks the available
heroes by ban rate on the active patch, falling back to the newest patch that
has counts, and `RecommendationPanel` renders that list in place of the pick
categories once a ban has been recorded; `DraftOverlay` badges the draft
`BANNING`. With no counts or an empty roster the list is empty rather than
invented. The overlay window shows the same list during bans, and both surfaces switch as
soon as GSI reports a ban turn (`pick: false`, which keeps `current_pick` set
while `is_picking_phase` is false), so the list is up before the first ban lands;
picks return on the next pick turn.

Before the list existed, the pick cards stayed on screen through bans and named
the banned hero 1.28% of the time. Evaluation slice, 111,282 ban actions:

| ranking | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- |
| pick list (previous behaviour) | 1.28% | 3.6% | 7.7% | 0.553 |
| ban prevalence, global warm-up | 3.28% | 8.5% | 10.3% | 0.622 |
| ban prevalence, rolling per-patch | 4.41% | 11.6% | 17.1% | 0.708 |
| shipped table (`hero_ban_stats`, full-patch) | 4.78% | 12.7% | 17.9% | 0.718 |

The 1.47% in the table above is the same engine measure over all 20,000 drafts;
restricted to the evaluation slice it is 1.28%, which is the row to compare here.
"Ban prevalence" is a plain frequency table with no engine involvement: how often
each hero was banned, from the warm-up window or, in the per-patch variant, from
the drafts already replayed in this patch (falling back to the global table until
a patch has 200 replayed bans).

Bans concentrate harder than picks: 4.41% top-1 and 17.1% top-5 for the per-patch
ban table, against 3.38% and 13.8% for the pick prior — a 3.4x gap in top-1 and
better than 2x in top-5, the largest difference measured anywhere in this report.

The last row is the harness reading `hero_ban_stats` from a seeded database —
the exact table `get_ban_suggestions` serves, ranked the way the engine ranks it
— so 4.78% / 12.7% / 17.9% is what the shipped ranking achieves on this slice.
It scores above the rolling row because the shipped counts cover the whole
patch, including the bans being predicted; the 0.37 pp gap between 4.78% and
4.41% is that in-sample advantage, not a better method. The rolling 4.41% stays
the honest estimate for a patch in progress. Either way the list only predicts
*which* heroes get banned; nothing here shows that following it improves a
draft.

How much of that depends on having the current patch's counts? Restricting to
patch 7.41 (4,175 drafts, 57,925 ban actions) and ranking with 7.41's table
gives 4.40% top-1 / 9.8% top-3 / 15.7% top-5. Ranking the same events with the
previous patch's table (7.40, which contains no 7.41 drafts) — the one-patch-lag
case — drops to **1.80% / 4.6% / 10.1%**, below the rolling prior on that slice
(4.02% / 10.2% / 15.5%). The shipped list therefore depends on counts for the
patch being played, which only exist once that patch has accumulated drafts; a
stale table is worse than a fresh rolling prior, not merely equal to it. The
engine used to show the newest available table in that situation; it now shows
no list at all (`ban suggestions are disabled` in the log), because a ranking
below the rolling prior misleads more than an empty panel does.</think>

## Component ablation

Same evaluation events, identical available pools, only the number used for the
ordering changes. `composition_only` ranks by the composition component,
`matchup_synergy_only` by the equal-weight mean of the matchup and synergy
components, `meta_only` by the seeded meta score.

| ranking | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- |
| engine total (shipped) | 0.94% | 3.2% | 6.0% | 0.527 |
| meta only | 1.16% | 4.7% | 7.9% | 0.513 |
| composition only | 0.65% | 2.6% | 4.4% | 0.525 |
| matchup+synergy only | 0.53% | 1.9% | 3.5% | 0.513 |
| popularity (global warm-up) | 2.21% | 6.0% | 9.8% | 0.644 |
| popularity (rolling per-patch) | 3.38% | 9.0% | 13.8% | 0.695 |
| random | 0.90% | 2.70% | 4.50% | 0.500 |

The two popularity rows are the same evaluation events; the ablation itself only
re-ranks the engine's components, so the prior does not affect its rows, but both
baselines are shown so the gap is not understated.

With 79,764 events the standard error is ~0.03 pp on top-1 and ~0.001 on mean
percentile, so the percentile differences of 0.013–0.027 are real but tiny.

What the ablation shows:

- **matchup+synergy is worse than chance at top-1.** Ranked by those two
  components alone, the human pick is the engine's first suggestion 0.53% of the
  time against 0.90% for a random pick — a significant deficit, not noise. Their
  mean percentile (0.513) shows a weak positive ordering signal, so they rank the
  human pick slightly better than random *on average* while putting the wrong
  hero first more often than chance. Treat them as noise-level for
  recommendation quality, not as evidence in either direction for the first slot.
- **composition carries a slightly stronger ordering signal** (mean percentile
  0.525) but also loses to chance at top-1 (0.65%).
- **meta is the only component that beats chance at top-1** (1.16%), which is
  consistent with the shipped total's behavior: the engine's ranking is mostly
  meta, diluted by the interaction components.
- The combination does beat every component on mean percentile (0.527), so the
  components are not redundant — the mixture is marginally better ordered than
  any single one, while still far behind a plain popularity prior.

## Blending the popularity prior into the ranking

Offline experiment, no product change: rank heroes by
`(1 - w) * engine_percentile + w * popularity_percentile`, both percentiles
taken inside the current available pool so the blend is scale-free. The prior is
the same warm-up-fitted popularity used above; `w = 0` reproduces the engine
exactly and `w = 1` reproduces popularity exactly (verified against the
standalone rows). Evaluation slice, 79,764 pick events:

| w (popularity weight) | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- |
| 0.0 (engine) | 0.94% | 3.2% | 6.0% | 0.527 |
| 0.1 | 1.82% | 4.8% | 7.8% | 0.541 |
| 0.2 | 2.02% | 5.3% | 9.0% | 0.558 |
| 0.3 | 2.23% | 6.1% | 9.4% | 0.578 |
| 0.4 | 2.39% | 6.6% | 9.6% | 0.601 |
| 0.5 | 2.48% | 6.9% | 10.0% | 0.623 |
| 0.6 | 2.49% | 7.1% | 10.3% | 0.637 |
| 0.7 | 2.53% | 7.3% | 10.8% | 0.644 |
| 0.8 | 2.47% | 7.3% | 10.7% | 0.647 |
| 0.9 | 2.42% | 7.1% | 10.4% | 0.647 |
| 1.0 (popularity) | 2.21% | 6.0% | 9.8% | 0.644 |

Paired McNemar on the same events at the a-priori equal weight `w = 0.5`
(chosen before looking at the sweep peak):

| comparison | blend top-1 only | baseline top-1 only | z |
| --- | --- | --- | --- |
| vs engine | 1,893 | 661 | 24.4 |
| vs popularity | 1,677 | 1,454 | 4.0 |

What this means:

- **Against the scorer, the blend wins decisively.** Top-1 goes 0.94% → 2.48%,
  top-3 3.2% → 6.9%, top-5 6.0% → 10.0% (z = 24.4 paired, p ≈ 1e-130).
- **Against popularity, the win is real but small at top-1** (+0.27 pp, paired
  z = 4.0, p ≈ 6e-5) and vanishes at mean percentile: 0.623 vs 0.644, because the
  prior dominates the ordering and the engine mostly reshuffles the head.
- **It is not a stable margin.** Splitting the evaluation slice in halves and
  keeping `w = 0.5` fixed: half A blend 2.40% vs popularity 1.95% (+0.45 pp),
  half B blend 2.57% vs 2.46% (+0.11 pp, inside noise at n ≈ 39,900). The tuned
  peak (`w = 0.7`) reaches 2.71% on half B, but that weight was picked on the same
  data that produced the peak, so it is not out-of-sample evidence.
- The blend is best understood as *popularity ordering with the engine breaking
  near-ties at the top*: it keeps the prior's hit rate while adding a little
  head precision, and it never improves overall ordering (mean percentile).

Verdict: blending is a clear improvement over shipping the scorer alone and a
marginal, era-dependent improvement over a plain popularity prior. Shipping it
would be a product and threshold change, which this backtest deliberately does
not make; the numbers are here to justify or reject that decision.

### With a rolling per-patch prior

The prior above is one global table fitted on the 12,000 warm-up drafts, so by
the last patch it is stale. Rerunning with `--prior per-patch` recomputes it from
the drafts already replayed *within the same patch* (expanding window, strictly
earlier drafts, falling back to the global warm-up table until a patch has 200
replayed picks). The evaluation events and engine ranking are identical, so only
the baseline and the blend change:

| prior | engine top-1 | popularity top-1 / top-3 / top-5 / pct | blend w=0.5 | blend w=0.7 |
| --- | --- | --- | --- | --- |
| global warm-up | 0.94% | 2.21% / 6.0% / 9.8% / 0.644 | 2.48% | 2.53% |
| rolling per-patch | 0.94% | 3.38% / 9.0% / 13.8% / 0.695 | 3.07% | 3.53% |

The blend's edge does **not** hold up:

- At the a-priori equal weight it is now *significantly worse* than popularity:
  3.07% vs 3.38% top-1, paired McNemar z = −4.0, and worse on both evaluation
  halves (3.15% vs 3.34% on A, 3.00% vs 3.42% on B).
- The tuned peak (`w = 0.7`) still edges popularity on top-1 (3.53% vs 3.38%, and
  3.60% vs 3.42% on the later half), but loses on every other metric: top-3 8.4%
  vs 9.0%, top-5 13.3% vs 13.8%, mean percentile 0.687 vs 0.695. A tuned weight
  that wins one metric by ~0.15 pp while losing the rest is not an edge.
- The size of the prior gain (+1.17 pp top-1 from freshness alone) dwarfs anything
  the blend contributed (+0.27 pp at best, on the stale prior). The ranking
  problem here is stale meta data, not missing engine signal at the top.

So the honest reading is: if the product wants better first suggestions, refresh
and version the popularity prior per patch; adding the heuristic scorer on top of
a fresh prior does not reliably improve it.

## Prior-gated re-ranking

Blending scores diluted the prior. Gating is the other way to combine them: take
the prior's top-K candidates, let the engine order *those*, and keep everything
else in engine order, so the first suggestion is the engine's favourite inside
the prior's shortlist. Evaluation slice, ranked by the rolling per-patch prior:

| ranking | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- |
| prior only (reference) | 3.38% | 9.0% | 13.8% | 0.695 |
| gate K=5 | 3.41% | 9.1% | 13.8% | 0.562 |
| gate K=10 | 2.83% | 8.6% | 14.0% | 0.588 |
| gate K=20 | 2.48% | 7.8% | 12.4% | 0.624 |
| gate K=40 | 2.15% | 6.1% | 10.4% | 0.648 |
| gate K=80 | 1.46% | 4.4% | 7.7% | 0.627 |
| engine only (reference) | 0.94% | 3.2% | 6.0% | 0.527 |

McNemar at K=5 against the plain prior on the same events: 1,625 gate-only
versus 1,601 prior-only, **z = 0.4** — a tie, not a win. Every larger K loses
top-1, and mean percentile drops everywhere (0.562 at K=5 against 0.695 for the
prior), because the shortlist discards the prior's ordering below the cut.

With the stale global prior the same gate does help: 2.39% at K=5 (McNemar
z = 2.7 against 2.21% popularity) and 2.45% at K=10. That is consistent with
everything above — the engine's ranking carries a little information that the
stale prior lacks, and adds nothing once the prior is fresh.

Verdict: gating does not beat the 3.38% per-patch baseline. Its best setting ties
it (3.41% top-1, z = 0.4) while ordering the rest of the list much worse.

## ML enabled vs disabled

Same sampler and the same 3,000 drafts (1,200 evaluation drafts, 11,969 pick
events), run twice: once deterministic, once with the bundled
`ml/models/draft_model_xgboost_full.onnx` (887 features) loaded and
`ml_enabled = true`, so the ML component carries its 25% weight.

| eval metric | ML off | ML on | popularity (global) | popularity (per-patch) |
| --- | --- | --- | --- | --- |
| top-1 | 0.77% | 0.81% | 2.36% | 3.37% |
| top-3 | 3.3% | 3.2% | 6.5% | 9.0% |
| top-5 | 6.0% | 5.6% | 10.4% | 13.7% |
| mean rank of pick | 53.3 | 53.3 | 39.9 | 35.3 |
| mean percentile of pick | 0.526 | 0.526 | 0.646 | 0.687 |

The per-patch column is a separate deterministic run on the same 3,000 drafts
(`backtest_perpatch_3000.json`); the ML rows do not use the prior, so they are
unchanged by it.

Outcome proxy on the same pair:

| bucket | ML off | ML on |
| --- | --- | --- |
| rank 1 | 48.9% [43.2–54.7], n=282 | 50.7% [45.0–56.4], n=292 |
| rank 2–3 | 51.3%, n=639 | 51.5%, n=647 |
| rank 4–10 | 49.7%, n=2,384 | 50.6%, n=2,379 |
| rank > 10 | 50.0%, n=26,620 | 49.9%, n=26,607 |
| AUC (score sum) | 0.524 | 0.529 |
| AUC (percentile sum) | 0.525 | 0.527 |

ML moves top-1 by +0.04 pp (SE ≈ 0.08 pp at this sample size) and the AUC by
+0.005 (SE ≈ 0.01 over 2,736 matches): both indistinguishable from zero. Per
phase, ML-on top-1 is 1.65% opening, 1.13% early, 0.89% mid, 0.65% late, 0.55%
final — the same decay as the deterministic path. Agreement stays at chance, and
the popularity prior remains ~3x ahead of ML-on with the global prior (2.36% vs
0.81%) and ~4x ahead with the per-patch one (3.37% vs 0.81%).

That is consistent with the model's own quality: `default_ml_enabled()`'s
comment records it at ~0.539 test ROC-AUC, near the 0.5 baseline, which is why
its weight is opt-in.

## Why: the ordering is mostly the meta score

The seeded meta score is not aligned with Captains Mode preference either:
across the 126 heroes, Pearson correlation between the seeded 7.41 winrate and
that hero's CM pick rate in 7.41 drafts is **-0.07**. Pick rate and win rate are
essentially independent here, so a meta-weighted ranking carries no draft
information.

Matchup advantage in the seed is `winrate - 0.5` from all-mode data (sd 0.139,
mean 0.000), which conflates a hero's base strength with its counter strength;
synergy sd is only 0.063. Neither shows up as agreement once the pick order is
replayed.

Per patch, engine top-1 / mean percentile on the evaluation slice: 7.39 1.18% /
0.541 (46,140 picks), 7.40 0.64% / 0.506 (24,534), 7.41 0.56% / 0.516 (9,090).
The engine is *worst* on the patch its seed data was built for, so stale patch
data is not the explanation.

## Limits

- The 7.41 seed covers the evaluation era (leakage), so these agreement numbers
  are optimistic. Removing the leakage could only lower them.
- One database snapshot was evaluated (patch 7.41, all other patches replayed
  against it), matching how the app ships.
- The shipped ban list is a prevalence table, not a quality model: it predicts
  *which* bans happen, not whether a suggested ban is good. Judging ban quality
  needs a counterfactual the corpus cannot provide. Its counts are full-patch,
  so its score is in-sample for the replayed patch; the rolling figure is the
  honest estimate, and a one-patch-stale table scores below it (1.80% vs 4.02%
  top-1 on 7.41), which is why the panel shows nothing instead of substituting
  an older patch's table.
- ML is disabled by default in the shipped config (`default_ml_enabled()`
  returns `false`, because the bundled model is near the 0.5 ROC-AUC baseline).
  The ML-on path was measured on a 3,000-draft matched sample rather than the
  full 20,000, because inference makes a run ~150x slower (≈28 min vs ≈11 s on
  the same 3,000 drafts); its differences from ML-off are inside sampling error.
- `player_profile` is `None` in this harness, so the comfort component is
  constant 0.5. A real user's comfort data would change ordering, but it is
  per-player and not part of the shipped default experience.
- 1,917 sampled matches contained at least one hero missing from the 126-hero
  database. The replay continues through them, but those picks and bans cannot
  be ranked, so 565 picks and 1,352 bans are omitted from the agreement
  denominators and the affected matches are excluded from the AUC and rank-bucket
  aggregates. Agreement rates are therefore computed only over rankable picks.
- The outcome proxy is a side win rate conditional on the engine's rank, not a
  causal estimate of picks the scorer would have won.
- The blend sweep picks its peak on the evaluation slice, so `w = 0.7` is an
  in-sample number. The equal weight `w = 0.5` and the eval-half split are the
  out-of-sample checks, and they are what the verdict rests on.
- The per-patch prior is an expanding window with a 200-pick warm-up per patch
  (200 replayed bans for the ban table); early patches with little history fall
  back to the global tables, so the boost is concentrated in the well-populated
  late patches. A different window or decay would change the magnitude, though
  not the direction of the result.
- The gate sweep, like the blend weights, picks its size on the evaluation
  slice. The K=5 comparison against the prior is paired, but five K values were
  tried, so 3.41% is an upper bound on what gating buys.

## Reproduce

```bash
python ml/backtest/export_drafts.py --out ml/backtest/data/drafts_cm.jsonl

cargo run --release --example backtest_scorer \
  --manifest-path apps/desktop/src-tauri/Cargo.toml -- \
  --drafts ml/backtest/data/drafts_cm.jsonl \
  --db "$APPDATA/com.smartdraftassistant.app/smart_draft.db" \
  --sample 20000 --verify 200 \
  --picks-csv ml/backtest/results/picks.csv \
  --out ml/backtest/results/backtest.json
```

`--patch 7.39` restricts to one patch, `--ban-counts-patch 7.40` ranks every
event with that patch's shipped counts instead of each draft's own (with
`--patch 7.41` this is the one-patch-lag, out-of-sample case), `--limit N`
truncates chronologically,
`--warmup-fraction` changes the popularity warm-up split, and `--prior per-patch`
switches the baseline and blend to the rolling per-patch prior. For the ML path
add `--ml 1` (default model `ml/models/draft_model_xgboost_full.onnx`) and point
`ORT_DYLIB_PATH` at `apps/desktop/src-tauri/target/release/resources/onnxruntime.dll`,
since that is where the bundled runtime sits; the harness fails instead of
silently scoring a neutral 0.5 if the model does not load.
The 7.41 comparison in the ban section comes from the same command with
`--patch 7.41 --warmup-fraction 0`, plus `--ban-counts-patch 7.40` for the lagged
run. `ml/backtest/results/backtest.json` holds the committed summary (including
`blend_sweep`, `blend_halves`, `blend_top1_discordance`, `gate_sweep`,
`gate_top1_discordance` and `ban_phase`). Within `ban_phase`, `shipped_eval`
ranks bans by the `hero_ban_stats` rows of the seeded database (the table the
app serves) and `shipped_patches` reports how many patches had counts; run
`seed_matchups` first or that ranking is skipped with a warning. The same
directory holds `ml/backtest/data/drafts_cm_manifest.json` (committed), which
pins the dataset hash above. The exported drafts themselves, the per-pick CSV and run logs are
gitignored; re-running the exporter reproduces the same sha256 for the same raw
corpus.
