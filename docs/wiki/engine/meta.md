# Patch Meta Data

The engine always had a meta layer — `meta_data`, `MetaData`, `W_META = 0.10`, and `DraftEngine::load_meta_data` — but nothing ever wrote to it. `get_meta` returned `None`, every hero scored a neutral `0.5`, no meta reason appeared, and `anti_meta_picks` was always empty.

- `ml/training/export_scoring_seed.py` builds one meta row per hero from `hero_winrates.parquet` (games, wins) and the `rad_has_*`/`dire_has_*` presence columns in `draft_features.parquet`.
- Winrate is shrunk toward `0.5` with a 100-game prior: `(wins + 50) / (games + 100)`.
- The exporter percentile-ranks shrunk winrate and pickrate across the roster, then blends `0.75 * winrate_rank + 0.25 * pickrate_rank`. `0.5` is the median hero, not a 50% winrate.
- `seed_matchups` validates and upserts the `meta` array from `scoring_seed.json` for the active patch; `--synthetic` writes deterministic meta. Re-running is idempotent.
- `DraftEngine::load_meta_data` loads the snapshot for the active patch and feeds the `W_META` component plus explanations (`meta > 0.65` → "Strong in current meta (54% winrate)", `meta < 0.35` → "Off-meta pick (47% winrate)").
- Recommendation categories are labels, not exclusive buckets: Top Picks contains every candidate, and Safe / High Impact / Comfort / Anti-Meta are additional labels. Anti-Meta means low meta with strong contextual fit (`meta < 0.35 && total score > 0.55`).
- `get_data_health` reports `meta_hero_count` and flags a missing snapshot.

Canonical source: [`../../STEP_09_META.md`](../../STEP_09_META.md). Related: [Scoring data flow](scoring.md), [Database, seeding, and patch](data.md).

## Verified state

Seeded database: `meta_data` has patch `7.41` with 126 hero scores.

## Open follow-ups

- The snapshot is global all-pick data, not segmented by MMR or pro scene.
- `auto_update_meta` still has no scheduler; meta refreshes only when the seed is regenerated and reseeded.
- Four heroes without a feature column (Outworld Destroyer, Ringmaster, Kez, Largo) use a neutral pickrate rank and exported pickrate `0.0`.
