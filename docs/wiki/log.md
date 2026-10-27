# Wiki Log

Append-only chronology. Newest entries at the top. Format: `## [YYYY-MM-DD] ingest | Source Title`.

## [2026-10-27] update | Meta data pipeline (Step 9)

- Added [Patch meta data](engine/meta.md); canonical source: [`../STEP_09_META.md`](../STEP_09_META.md).
- Implemented the previously dead meta layer: `export_scoring_seed.py` now emits one `meta` row per hero, and `seed_matchups` validates and upserts it for the active patch (synthetic meta under `--synthetic`).
- Meta score is a percentile blend of shrunk winrate (0.75) and pickrate rank (0.25); winrate uses a 100-game prior `(wins + 50) / (games + 100)`.
- Fixed `Database::upsert_meta` to persist integer `updated_at`; `get_data_health` now reports `meta_hero_count` and flags a missing snapshot.
- Recommendations explain strong/off-meta heroes with the raw winrate; Top Picks now includes every candidate and Safe/Aggressive/Comfort/Anti-Meta are labels with correct categories.
- Verification: 40 Rust lib + 1 seeder + 11 frontend + 36 Python tests pass; `fmt`, `clippy -D warnings`, lint, and build clean; seeded DB has patch `7.41` meta for 126 heroes.

## [2026-10-23] update | ESLint (Step 8)

- Added ESLint 9 flat config `apps/desktop/eslint.config.js` (`@eslint/js`, `typescript-eslint` recommended, `eslint-plugin-react-hooks`, `eslint-plugin-react-refresh`); `react-refresh/only-export-components` is off because hooks are co-located with components.
- Installed the dev dependencies and changed `npm run lint` to `eslint . --max-warnings 0` (flat config drops `--ext`).
- Added a "Frontend lint" step to `scripts/verify.ps1`, so the gate and CI cover lint before build/tests.
- Canonical source: [`../STEP_08_CI.md`](../STEP_08_CI.md); see [CI and quality gates](ci/quality-gates.md).
- Verification: 20 files lint clean; full quality gate PASS.

## [2026-10-23] update | Features with start_time and model retrain (Step 6)

- Regenerated `ml/datasets/processed/draft_features.parquet` (102,118 rows, 890 columns) so it carries `start_time`; chronological splits no longer fall back to `match_id`.
- Tracked `ml/datasets/heroes.json` (defines feature names) so the 887-feature contract is reproducible on a clean checkout; added `ml/tests/test_processor_features.py`.
- Retrained and swapped in the bundled `draft_model_xgboost_full` (887 inputs; held-out test ROC-AUC 0.5388); repointed `generate_golden.py` at `draft_features.parquet` and refreshed the golden fixture.
- Canonical source: [`../STEP_06_TRAINING.md`](../STEP_06_TRAINING.md); see [Training pipeline](ml/training-pipeline.md).

## [2026-10-23] update | Partial key rotation (Step 1)

- Rotated the STRATZ key; the replacement is stored only in `ml/.env` and loads as `set`.
- Searched Git history: no secrets or generated datasets are tracked (`git ls-files` shows only `ml/.env.example`).
- OpenDota key rotation remains pending; recorded in the [Secrets management](security/secrets-management.md) rotation log without any key value.
- Updated [`../STEP_01_SECRETS_AND_ENV.md`](../STEP_01_SECRETS_AND_ENV.md), [`../AUDIT.md`](../AUDIT.md), [`../REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md), and the wiki index/plan status.

## [2026-10-23] update | Step 8 CI and quality gates

- Added [CI and quality gates](ci/quality-gates.md).
- Canonical source: [`../STEP_08_CI.md`](../STEP_08_CI.md).
- Added one entry point, `scripts/verify.ps1`, plus `scripts/secret_scan.py` (stdlib-only working-tree credential scanner).
- Added `.github/workflows/ci.yml`: a fast ubuntu secret-scan job and a windows-latest quality gate that runs the same script.
- Made `clippy -D warnings` enforceable by clearing ten pre-existing lints; added a `SemanticGetter` alias in `ml.rs`.
- Repository hygiene: ignore generated `ml/models/` artifacts (keep the bundled `draft_model_xgboost_full.*` release set) and `ml/experiments/`; removed a stray root file.
- Verification: `verify.ps1` runs end to end with all steps PASS (36 Rust lib + 1 seeder + 11 frontend + 28 Python tests; secret scan clean).
- Residual: ESLint is declared but not installed or configured, so it is not in the gate.


## [2026-10-23] update | Step 7 settings, GSI, and UI state

- Added [Settings, GSI, and UI state](engine/settings-gsi.md).
- Canonical source: [`../STEP_07_GSI_SETTINGS.md`](../STEP_07_GSI_SETTINGS.md).
- Persisted validated `AppConfig` as JSON in `app_settings`; implemented real `set_config`/`get_config` and `restart_gsi`; startup applies port, auth, auto-start, and model path.
- Remodeled `GSIDraft` to the real schema (`activeteam`/`pick`, `team2`/`team3` slots), added `player.team_name`, an `activeteam`-first turn state machine with a CM-order fallback, optional auth (401 on mismatch), and graceful bind errors.
- Made `DraftOverlay` side-aware, stopped draft reinit on team change, and fixed the settings toggle bindings, auth token field, validation, and reset.
- Verification: 36 Rust + 11 frontend tests pass; `cargo fmt --check` clean; clippy adds no new warnings; `npm run build` succeeds.

## [2026-10-23] update | Step 6 training pipeline

- Added [Training pipeline](ml/training-pipeline.md).
- Canonical source: [`../STEP_06_TRAINING.md`](../STEP_06_TRAINING.md).
- Added `training/splits.py` (`start_time`-first chronological train/val/test) and `training/manifest.py`.
- Reworked `train_full.py`: validation-only early stopping, single test evaluation, effective `--skip-train`/`--evaluate-only`, ONNX + schema + manifest export.
- `trainer.py` prefers `start_time` and uses a three-way split; added `onnx`/`onnxmltools` deps and `start_time` to the processor.
- Reconstructed `data_collector.py` as a multi-line module (`OpenDotaCollector`, `StratzCollector`); fixed `discovered_public` initialization, the summary counter, and boolean flags.
- Repaired `train_pipeline.py`: `sys.path` for both invocation styles, `BooleanOptionalAction` flags, and robust `--config` handling.
- Verification: 28 Python tests pass; smoke run exports model/ONNX/schema/manifest in ~17 s; collector and pipeline parse args under module and script invocation.

## [2026-10-23] update | Step 5 database paths, seeding, and patch

- Added [Database, seeding, and patch](engine/data.md).
- Canonical source: [`../STEP_05_DATA.md`](../STEP_05_DATA.md).
- Added shared `paths.rs`; the app and both seeders now resolve `%APPDATA%\com.smartdraftassistant.app\smart_draft.db`.
- Added `app_settings` (schema v2) with a persisted active patch; matchup/synergy/meta queries are patch-scoped.
- Made hero and matchup/synergy seeding idempotent and transactional; the exporter now writes `patch_version`.
- Added `Database::health` / `get_data_health`, verified WAL, and moved blocking commands to `spawn_blocking`.
- Verification: 24 Rust library tests, 1 seeder test, 8 frontend tests, 15 Python tests pass; seeded DB `schema 2 / 126 / 9052 / 7381 / 7.41`.
- Residual: ML feature builder still hardcodes `7.41`; the old DB path is orphaned.

## [2026-10-23] update | Step 4 Tauri IPC and manual drafting

- Added [Tauri IPC conventions](engine/ipc.md).
- Canonical source: [`../STEP_04_IPC.md`](../STEP_04_IPC.md).
- Applied `rename_all = "snake_case"` to every command and added a serializable `CommandError`.
- Added engine validation for heroes, availability, side capacity, and order; manual correction ignores turn state.
- Added a `DraftAction` undo stack, `undo_last_action`, and a pick/ban mode toggle in `HeroPicker`.
- Added a frontend smoke test covering the manual IPC commands.
- Verification: 24 Rust library tests, 8 frontend tests, and 15 Python tests pass.

## [2026-10-22] update | Step 3 scoring correctness

- Added [Scoring data flow](engine/scoring.md).
- Canonical source: [`../STEP_03_SCORING.md`](../STEP_03_SCORING.md).
- Wired real matchup/synergy data into the startup scorers and exposed `get_score_breakdown`.
- Canonicalized hero roles and preserved `PrimaryAttribute::All`.
- Exported 9,052 real matchup rows and 7,381 synergy pairs; synthetic seeding now requires `--synthetic`.
- Verification: 20 Rust library tests, 1 seeder test, and 15 Python tests pass.
- Residual: shared app/seeder database path and patch-aware queries remain in Step 5.

## [2026-10-22] update | Step 2 ML inference correctness

- Added [ML inference contract](ml/inference-contract.md).
- Canonical source: [`../STEP_02_ML_INFERENCE.md`](../STEP_02_ML_INFERENCE.md).
- Exported a 887-feature schema, fixed positive-class probability extraction, encoded patch categories, and replaced hardcoded defaults with schema medians.
- Added a Python ONNX golden fixture and Rust integration test.
- Verification: 11 Rust tests and 15 Python tests pass.
- Residual: regenerate the schema during training; run a desktop UI smoke test.

## [2026-10-22] update | Step 1 implementation

- Added root `.gitignore`, `ml/.env.example`, and `ml/training/settings.py`.
- Migrated 12 collector and diagnostic scripts off hardcoded credentials.
- Added 12 passing settings and source-scan tests.
- Source scan is clean; key rotation and the first Git commit remain pending.
- Canonical source: [`../STEP_01_SECRETS_AND_ENV.md`](../STEP_01_SECRETS_AND_ENV.md).

## [2026-10-22] ingest | Step 1 design: Secrets and Environment

- Added [Secrets management](security/secrets-management.md).
- Canonical source: [`../STEP_01_SECRETS_AND_ENV.md`](../STEP_01_SECRETS_AND_ENV.md).
- Captured the environment contract, optional versus required keys, repository guard, rotation runbook, and open decisions.
- Status: proposed, awaiting decisions before implementation.

## [2026-10-22] ingest | Remediation Plan

- Added [Remediation plan](plans/remediation-plan.md).
- Canonical source: [`../REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md).
- Captured the eight-step order, dependencies, execution waves, and acceptance gates.
- Status: proposed; no code changes made.

## [2026-10-22] ingest | Technical Audit

- Added [Project audit, 2026-10-22](audits/project-audit-2026-10-22.md).
- Canonical source: [`../AUDIT.md`](../AUDIT.md).
- Captured critical, high, medium, and low findings with evidence.
- Status: complete.

