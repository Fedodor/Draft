# Deterministic scorer backtest against the historical corpus

Offline, read-only evaluation of the shipping heuristic scorer (`ml_enabled =
false`) on every Captains Mode draft with a usable pick/ban order in
`ml/datasets/raw/`. No model weights, thresholds, seed data or engine code were
changed to produce these numbers.

The short version: **the deterministic scorer does not beat chance at predicting
human picks, and a plain popularity baseline is ~2.3x better at top-1.** Its
ordering is dominated by the seeded per-hero meta score; ranked on their own,
composition and matchup+synergy carry only a barely detectable ordering signal
and matchup+synergy is worse than chance at top-1.

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

| metric | engine | popularity baseline | random |
| --- | --- | --- | --- |
| top-1 | 0.94% | 2.21% | 0.90% (measured) |
| top-3 | 3.2% | 6.0% | 2.70% |
| top-5 | 6.0% | 9.8% | 4.50% |
| mean rank of pick | 53.2 | 40.2 | 56.2 |
| mean percentile of pick | 0.527 | 0.644 | 0.500 |

Whole sample (all 20,000 drafts) is the same story: top-1 0.96%, top-3 3.0%,
top-5 5.5%, mean percentile 0.520. The engine's top-1 lift over random is
+0.06 percentage points; popularity's is +1.31.

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

Popularity baseline, evaluation slice:

| phase | events | top-1 | top-3 | top-5 | mean pct |
| --- | --- | --- | --- | --- | --- |
| opening | 15,969 | 4.31% | 8.4% | 13.6% | 0.667 |
| early | 15,967 | 3.19% | 9.3% | 15.2% | 0.668 |
| mid | 15,929 | 1.80% | 5.8% | 9.6% | 0.660 |
| late | 15,949 | 0.88% | 3.5% | 5.8% | 0.630 |
| final | 15,950 | 0.85% | 2.9% | 4.6% | 0.595 |

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
| popularity baseline | 2.21% | 6.0% | 9.8% | 0.644 |
| random | 0.90% | 2.70% | 4.50% | 0.500 |

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

## ML enabled vs disabled

Same sampler and the same 3,000 drafts (1,200 evaluation drafts, 11,969 pick
events), run twice: once deterministic, once with the bundled
`ml/models/draft_model_xgboost_full.onnx` (887 features) loaded and
`ml_enabled = true`, so the ML component carries its 25% weight.

| eval metric | ML off | ML on | popularity |
| --- | --- | --- | --- |
| top-1 | 0.77% | 0.81% | 2.36% |
| top-3 | 3.3% | 3.2% | 6.5% |
| top-5 | 6.0% | 5.6% | 10.4% |
| mean rank of pick | 53.3 | 53.3 | 39.9 |
| mean percentile of pick | 0.526 | 0.526 | 0.646 |

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
final — the same decay as the deterministic path. Agreement stays at chance and
the popularity prior remains ~3x ahead with ML on.

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

`--patch 7.39` restricts to one patch, `--limit N` truncates chronologically,
and `--warmup-fraction` changes the popularity warm-up split. For the ML path
add `--ml 1` (default model `ml/models/draft_model_xgboost_full.onnx`) and point
`ORT_DYLIB_PATH` at `apps/desktop/src-tauri/target/release/resources/onnxruntime.dll`,
since that is where the bundled runtime sits; the harness fails instead of
silently scoring a neutral 0.5 if the model does not load.
`ml/backtest/results/backtest.json` holds the committed summary, and
`ml/backtest/data/drafts_cm_manifest.json` (committed) pins the dataset hash
above. The exported drafts themselves, the per-pick CSV and run logs are
gitignored; re-running the exporter reproduces the same sha256 for the same raw
corpus.
