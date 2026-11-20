# Deterministic scorer backtest against the historical corpus

Offline, read-only evaluation of the shipping heuristic scorer (`ml_enabled =
false`) on every Captains Mode draft with a usable pick/ban order in
`ml/datasets/raw/`. No model weights, thresholds, seed data or engine code were
changed to produce these numbers.

The short version: **the deterministic scorer does not beat chance at predicting
human picks, and a plain popularity baseline is ~2.3x better at top-1.** Its
ordering is reproduced almost exactly by the seeded per-hero meta score alone,
so matchup, synergy and composition contribute no measurable agreement signal.

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

Sampling: drafts are sorted by start time and every 5th one is replayed
(20,000 drafts, 199,435 pick events). The first 60% of the sample (12,000
drafts) is warm-up for the popularity baseline; the remaining 8,000 drafts
(79,764 pick events) are the evaluation slice. Even spacing is used instead of a
prefix so all patches are represented.

## Harness

`apps/desktop/src-tauri/examples/backtest_scorer.rs` drives the real engine:
`DraftEngine::initialize`, `reset`, `add_pick`/`add_ban` over the historical
order, and `score_breakdown(hero, team).total` for every hero still available.
The database is copied (with `-wal`/`-shm`) to a temp directory first, because
`Database::new` migrates and opens WAL, and the sample is discarded afterwards.

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
| top-3 | 3.2% | 6.0% | ~2.7% |
| top-5 | 6.0% | 9.8% | ~4.5% |
| mean rank of pick | 53.2 | 40.2 | ~60 |
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
strongest) and decays as the draft fills in. In the final phase the two
converge, but both are close to the random floor by then because only ~50 heroes
remain.

Per pick ordinal, engine top-1: 1.59, 1.57, 1.00, 1.06, 0.85, 0.77, 0.65, 0.66,
0.74, 0.70 percent.

## Ban-aware outcome proxy

Win rate of the side that made the pick, bucketed by the engine's rank for that
pick (Wilson 95% CI):

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

## Why: the ordering is the meta score

Ranking the same available pools by `meta_data.hero_meta_scores` alone, with no
engine, gives top-1 1.28% (whole sample) / 1.16% (evaluation), mean percentile
0.507 / 0.513 — statistically indistinguishable from the full engine, and
slightly better. Combination with the other components does not help.

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
- ML was disabled throughout; `ml_enabled = true` was not measured and is a
  different product surface.
- `player_profile` is `None` in this harness, so the comfort component is
  constant 0.5. A real user's comfort data would change ordering, but it is
  per-player and not part of the shipped default experience.
- 1,917 sampled matches contained at least one hero missing from the 126-hero
  database; those actions were scored as unavailable (565 picks, 1,352 bans
  skipped) rather than dropped, so the denominators stay honest.
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
and `--warmup-fraction` changes the popularity warm-up split.
`ml/backtest/results/backtest.json` holds the committed summary; the exported
drafts, the per-pick CSV and run logs are gitignored.
