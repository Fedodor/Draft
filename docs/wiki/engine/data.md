# Database, Seeding, and Patch

- Type: Engine and data
- Status: Implemented (Step 5)
- Canonical source: [`../../STEP_05_DATA.md`](../../STEP_05_DATA.md)
- Related: [`scoring.md`](scoring.md), [`ipc.md`](ipc.md), [Remediation plan](../plans/remediation-plan.md)

## Why it matters

Before Step 5 the seeders wrote to `%APPDATA%\smart-draft-assistant` while the app read `%APPDATA%\com.smartdraftassistant.app`, the active patch was a `'7.41'` string literal in every SQL statement, and async commands ran blocking SQLite/ONNX work inline. Seeding "succeeded" but the app opened an empty or stale-patched database.

## One path

`apps/desktop/src-tauri/src/paths.rs` owns the location:

- `APP_IDENTIFIER = "com.smartdraftassistant.app"` must match `identifier` in `tauri.conf.json`.
- `DB_FILENAME = "smart_draft.db"`.
- `app_data_dir()` and `database_path()` resolve it without a Tauri handle.

The app and both seeders call the same module; the app warns at startup if Tauri's own `app_data_dir()` diverges.

## Active patch

Schema version 2 adds an `app_settings` key/value table. The active patch lives under `active_patch`:

- `Database::set_active_patch` / `get_active_patch` (with a fallback to the newest `hero_matchups.patch_version`).
- All matchup/synergy upserts and getters take a `patch` argument; no literal `7.41` remains.
- `DraftEngine::initialize` resolves the patch before loading matchups, synergies, and meta, and populates `DraftState.patch_version`.
- `export_scoring_seed.py` writes `patch_version` (highest numeric patch from the model schema) so `seed_matchups` seeds the patch the model knows.

## Seeding

- `seed_heroes` → `bulk_upsert_heroes` + `bulk_upsert_hero_semantics`, one transaction each.
- `seed_matchups` → real data by default (`scoring_seed.json`), `bulk_upsert_matchups` + `bulk_upsert_synergies` + `set_active_patch`; `--synthetic` writes patch `synthetic`.
- Re-running is idempotent. Rows from other patches are preserved, so delete a development database when switching from synthetic to real data.

## Health and runtime

- `Database::health()` / Tauri `get_data_health` report schema version, row counts, active patch, WAL flag, and issues; startup logs them.
- `Database::new` enables WAL; the health check verifies it.
- `get_recommendations`, `get_search_recommendations`, `predict_draft_value`, and `get_data_health` run on `spawn_blocking`.

## Verified state

`%APPDATA%\com.smartdraftassistant.app\smart_draft.db`: `schema 2, heroes 126, matchups 9052, synergies 7381, active_patch 7.41, journal wal`.

## Residual risks

- `dirs::data_dir()` ignores an `APPDATA` override on Windows, so tests reset the real database instead of redirecting.
- Databases at the old `%APPDATA%\smart-draft-assistant` path are orphaned.
- The exporter derives the patch from the model schema, not from the datasets' patch column.
