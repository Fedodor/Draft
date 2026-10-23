# Step 5 Design: Database Paths, Seeding, and Patch Source of Truth

- Date: 2026-10-23
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), C4 and H1

## Implementation status

- [x] Resolve one database path shared by the app and the seeders.
- [x] Store the active patch in the database and scope matchup/synergy/meta queries to it.
- [x] Make hero and matchup/synergy seeding idempotent and transactional.
- [x] Verify WAL mode and add a startup schema/seed health check.
- [x] Move blocking database and ML work off the async command threads.
- [x] Update README and docs with the real on-disk path.

## Problem

Three defects combined into one data-integrity risk:

1. The seeders used `dirs::data_dir()/smart-draft-assistant`, but the Tauri app opens `app_data_dir()` = `{data_dir}/com.smartdraftassistant.app`. Seeding never populated the database the app actually read.
2. The active patch was hardcoded to `'7.41'` in every SQL statement (`upsert_matchup`, `get_matchups_for_hero`, ...), so a new patch required code edits and old patch rows were mixed into lookups.
3. Seeding wrote row-by-row outside a transaction, and async commands performed blocking SQLite/ONNX work directly on the async runtime.

## Shared path

`apps/desktop/src-tauri/src/paths.rs` is now the single source of truth:

```rust
pub const APP_IDENTIFIER: &str = "com.smartdraftassistant.app";
pub const DB_FILENAME: &str = "smart_draft.db";
pub fn app_data_dir() -> Option<PathBuf>;
pub fn database_path() -> Option<PathBuf>;
```

`APP_IDENTIFIER` must stay in sync with `identifier` in `tauri.conf.json`. The app logs a warning at startup if Tauri's `app_data_dir()` ever differs from `paths::app_data_dir()`. Both seeders resolve the same directory, so a single seed run now targets the database the app opens.

Resulting path on Windows: `%APPDATA%\com.smartdraftassistant.app\smart_draft.db`.

## Active patch in the database

`schema_version` is bumped to 2. Migration v2 adds:

```sql
CREATE TABLE app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The active patch is stored under the `active_patch` key:

- `Database::set_active_patch` / `get_active_patch` read and write the setting. `get_active_patch` falls back to the most recently updated `hero_matchups.patch_version` when the key is absent.
- Every matchup/synergy upsert and getter takes a `patch: &str`; the hardcoded `'7.41'` literals are gone.
- `DraftEngine` resolves the patch once during `initialize` (`load_active_patch`) before loading matchups, synergies, and meta. When no patch is persisted it skips those lookups and logs a warning instead of silently using stale rows.
- `DraftState.patch_version` and `reset` are populated from the active patch.

`ml/training/export_scoring_seed.py` writes `patch_version` into `scoring_seed.json` by selecting the highest numeric patch from `draft_model_xgboost_full_schema.json` (`unknown` is ignored), so the seed always matches the model's categorical space. `seed_matchups` requires that field and fails with a regenerate hint when it is missing.

## Idempotent, transactional seeding

- `seed_heroes` calls `Database::bulk_upsert_heroes` and `bulk_upsert_hero_semantics`, each a single transaction of `INSERT ... ON CONFLICT ... DO UPDATE` statements.
- `seed_matchups` calls `bulk_upsert_matchups` / `bulk_upsert_synergies` (one transaction each) and then `set_active_patch`.
- Re-running any seeder is safe: values are overwritten, the schema is unchanged, and `active_patch` ends up pointing at the seeded patch.

Because upserts key on `(hero_id, vs_hero_id, patch_version)`, rows from a different patch are preserved rather than overwritten. When switching a development database from synthetic to real data, delete the database first so old `synthetic`-tagged rows do not linger.

## Health check and WAL

`Database::health()` returns a `DataHealth` record: schema version, hero/matchup/synergy counts, active patch, WAL flag, and a list of issues. It is exposed as the `get_data_health` Tauri command and logged at startup (issues as warnings). `Database::new` already enables `journal_mode=WAL`; the health check now asserts it.

## Off the async runtime

`get_recommendations`, `get_search_recommendations`, `predict_draft_value`, and `get_data_health` clone the shared `Arc` and run through `tauri::async_runtime::spawn_blocking`, so SQLite queries and ONNX inference no longer block the async command threads. A `JoinError` becomes `CommandError::Internal`.

## Tests and verification

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib        # 24 passed
cargo test --manifest-path infrastructure/seeders/Cargo.toml --bin seed_matchups  # 1 passed
npm test --prefix apps/desktop                                            # 8 passed
npx tsc --noEmit --project apps/desktop
python ml\venv\Scripts\python.exe -m pytest ml\tests -q                   # 15 passed
```

End-to-end seed into `%APPDATA%\com.smartdraftassistant.app\smart_draft.db`:

```
schema 2, heroes 126, matchups 9052, synergies 7381, active_patch 7.41, journal wal
```

The seeder unit test asserts `scoring_seed.json` still parses and carries a non-empty `patch_version`. The frontend smoke test covers `get_data_health`.

## Residual risks

- `dirs::data_dir()` ignores an `APPDATA` override on Windows (it uses the known-folder API), so tests cannot redirect seeding with an environment variable; reset the real database instead.
- Old databases under `%APPDATA%\smart-draft-assistant` are no longer read and can be deleted.
- The exporter derives the patch from the model schema rather than from the processed datasets' actual patch column.
