# Step 9: Patch Meta Data

- Date: 2026-10-27
- Status: Implemented
- Depends on: Step 5 (shared database and active patch), Step 6 (processed datasets)
- Wiki summary: [`wiki/engine/meta.md`](wiki/engine/meta.md)

## Problem

The engine already had every part of a meta layer except data:

- `meta_data` table, `MetaData` model, `Database::upsert_meta` / `get_meta`.
- `W_META = 0.10` in the recommendation score.
- `DraftEngine::load_meta_data` and `get_meta_score`.

Nothing ever wrote `meta_data`, so `get_meta` always returned `None`, every hero
scored a neutral `0.5`, the UI never showed a meta reason, and the
`anti_meta_picks` category could not be populated. The health check also did not
notice the missing snapshot.

## Outcome

`seed_matchups` now seeds a patch-scoped meta snapshot for the full roster, the
engine uses it in scoring and explanations, and health reports its coverage.
Recommendation categories are now labels rather than mutually exclusive buckets,
so the best hero always appears in Top Picks and each card shows its real label.

## Data flow

```text
ml/datasets/processed/hero_winrates.parquet  (games, wins, winrate)
ml/datasets/processed/draft_features.parquet (rad_has_*/dire_has_* presence)
infrastructure/seeders/heroes.json           (roster)
        │
        ▼
ml/training/export_scoring_seed.py  →  scoring_seed.json["meta"]
        │
        ▼
infrastructure/seeders/seed_matchups.rs  →  meta_data (JSON columns)
        │
        ▼
DraftEngine::load_meta_data  →  W_META component + explanations + health
```

## Meta score

Raw winrates are noisy: the current all-pick sample ranges from 88 to 2,067
games per hero and includes a 90%-winrate hero with 10 games. The exporter
therefore:

1. Shrinks winrate toward `0.5` with a 100-game prior:
   `(wins + 50) / (games + 100)`.
2. Ranks the shrunk winrate and the observed pickrate across the roster with
   percentile ranks.
3. Blends them: `0.75 * winrate_rank + 0.25 * pickrate_rank`.
4. Clamps to `[0, 1]`.

`0.5` is the median hero, not a 50% winrate. The seed also carries the raw
`winrate` and `pickrate`, which the UI uses for explanations. Heroes without a
feature column (brand-new heroes and the renamed Outworld Destroyer) keep a
neutral pickrate rank instead of being treated as never picked.

## Explanations and categories

- `meta > 0.65` → `Strong in current meta (54% winrate)`.
- `meta < 0.35` → `Off-meta pick (47% winrate)`.
- Top Picks now receives every candidate; Safe, High Impact, Comfort, and
  Anti-Meta are additional labels.
- Anti-Meta means a low-meta hero with strong contextual fit
  (`meta < 0.35 && total score > 0.55`), i.e. a surprise pick, not a strong-meta
  hero the model dislikes.
- Each recommendation carries the category of the list it is shown in, so the
  card badge matches its section.

## Files

- `ml/training/export_scoring_seed.py` — `build_meta_rows`, percentile scoring.
- `infrastructure/seeders/scoring_seed.json` — `meta` rows for 126 heroes.
- `infrastructure/seeders/seed_matchups.rs` — validates and upserts meta; also
  writes deterministic synthetic meta under `--synthetic`.
- `apps/desktop/src-tauri/src/database.rs` — meta upsert writes integer
  `updated_at`; health reports `meta_hero_count`.
- `apps/desktop/src-tauri/src/engine.rs` — meta explanations and category
  labels.
- `apps/desktop/src/services/api.ts` — `DataHealth.meta_hero_count`.

## Acceptance

- `scoring_seed.json` has one meta row per hero, all values in `[0, 1]`, with a
  differentiated score range.
- Running `seed_matchups` writes `meta_data` for the active patch and is
  idempotent.
- `get_data_health` reports 126 meta heroes on the seeded database.
- Recommendations explain strong and off-meta heroes using the raw winrate.
- Top Picks is never empty just because its best hero is also aggressive/safe.

## Verification

```powershell
python ml\venv\Scripts\python.exe ml\training\export_scoring_seed.py
cargo run --manifest-path infrastructure\seeders\Cargo.toml --bin seed_matchups
cargo test --manifest-path apps\desktop\src-tauri\Cargo.toml --lib
cargo test --manifest-path infrastructure\seeders\Cargo.toml
python ml\venv\Scripts\python.exe -m pytest ml\tests -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
```

Seeded database result: `meta_data` has patch `7.41` and 126 hero scores.

## Residual risks

- The snapshot is global all-pick data, not segmented by MMR or pro scene; the
  `auto_update_meta` setting still has no scheduler.
- Pickrate is derived from the training features, so the four heroes without a
  feature column use a neutral pickrate rank.
- Meta is refreshed only when the seed is regenerated and reseeded.
