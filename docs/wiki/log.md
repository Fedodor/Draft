# Wiki Log

Append-only chronology. Newest entries at the top. Format: `## [YYYY-MM-DD] ingest | Source Title`.

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

