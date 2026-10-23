# Remediation Plan: Smart Draft Assistant

- Date: 2026-10-22
- Status: Implemented (Steps 1-8)
- Source audit: [`AUDIT.md`](AUDIT.md)
- Wiki summary: [`wiki/plans/remediation-plan.md`](wiki/plans/remediation-plan.md)

## Goal

Turn the prototype into a safe, reproducible, and trustworthy draft assistant without a big-bang rewrite. Each step must leave the application buildable and testable.

## Guiding principles

1. Fix security and wrong-output defects before UX polish.
2. Establish one source of truth for every cross-language contract.
3. Add a regression test with every correctness fix.
4. Prefer explicit failure over silent fallback values.
5. Keep ML optional at runtime; the app must still start without a model.
6. Do not retrain the bundled model until the feature contract and split logic are fixed.

## Status board

| Step | Theme | Status | Depends on | Exit gate |
| --- | --- | --- | --- | --- |
| 1 | Secrets and environment | Implemented, rotation pending | None | No plaintext credentials; env contract enforced |
| 2 | ML inference correctness | Implemented | 1 | Rust matches Python ONNX output |
| 3 | Scoring correctness | Implemented | 5 | Scorers use real data and pass unit tests |
| 4 | Tauri IPC and manual drafting | Implemented | None | Manual pick/ban works end to end |
| 5 | Database paths, seeding, patch | Implemented | 1 | Seeders and app share one database |
| 6 | Training pipeline | Implemented | 1, 2, 5 | Reproducible training and export |
| 7 | Settings, GSI, and UI state | Implemented | 4, 5 | User configuration is persisted and applied |
| 8 | CI and quality gates | Implemented | 1-7 | Automated checks block regressions |

## Execution waves

- Wave 1, safety and unblocking: Step 1, Step 4, Step 5.
- Wave 2, correctness: Step 2, Step 3.
- Wave 3, reliability: Step 6, Step 7.
- Wave 4, enforcement: Step 8.

Steps inside a wave may run in parallel. Step 2 must not be closed by editing the feature vector only; the contract must be testable from Python to Rust.

## Step 1: Secrets and environment

**Outcome:** No credentials in source, one documented environment contract, and a Git-safe repository.

**Fixes:** C1, M4, part of L3.

**Tasks**

- [x] Create a root `.gitignore` before any `git init`.
- [ ] Rotate the OpenDota and STRATZ keys, then verify the old keys are revoked.
- [x] Remove every hardcoded key from `ml/training/`.
- [x] Add `ml/.env.example` with placeholders and comments.
- [x] Add a central loader, recommended `ml/training/settings.py`.
- [x] Load `ml/.env` with `python-dotenv`, `override=False`.
- [x] Treat `STRATZ_API_KEY` as required for STRATZ collectors.
- [x] Treat `OPENDOTA_API_KEY` as optional, with a warning about lower rate limits.
- [x] Add `python-dotenv` to `ml/requirements.txt`.
- [x] Add a source secret scan and tests; add `gitleaks` before the first commit.
- [x] Add unit tests for missing, present, and malformed keys.

**Acceptance**

- A text search for key names finds no literal values in source.
- Collectors fail with a clear `MissingApiKeyError` only when their required key is absent.
- `ml/.env` is ignored and never logged.
- A secret scan of the working tree is clean.
- The application still builds without Python or keys.

**Verification**

```powershell
rg -n "(OPENDOTA_API_KEY|STRATZ_API_KEY)\s*=\s*['\"]" ml
gitleaks detect --source . --no-git
& ml\venv\Scripts\python.exe -m pytest ml/tests -q
```

**Detailed design:** [`STEP_01_SECRETS_AND_ENV.md`](STEP_01_SECRETS_AND_ENV.md).

## Step 2: ML inference correctness

**Outcome:** Rust and Python produce the same prediction for the same input.

**Fixes:** C2, C3.

**Tasks**

- [x] Capture a golden input/output pair from Python ONNX Runtime.
- [x] Fix `ml.rs` to use `probabilities[0][1]`, the positive-class probability.
- [x] Define a machine-readable feature schema next to the model.
- [x] Make Rust read the schema instead of maintaining a second feature list.
- [x] Remove the hardcoded `patch_version = 7.41` from inference.
- [x] Encode `patch` and `patch_version` exactly as training did, or remove them from the model in Step 6.
- [x] Replace silent `0.0` defaults with explicit validation in development and a documented fallback in production.
- [x] Add a Rust integration test using the golden pair.
- [x] Add a contract test that compares feature names, count, and order.

**Acceptance**

- `label` and probability match Python within `1e-5`.
- A missing or extra feature fails the contract test.
- No inference path contains a patch literal.
- The UI shows the model's positive-class probability, not an inverted value.

**Verification**

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib ml::
& ml\venv\Scripts\python.exe -m pytest ml/tests/test_onnx_contract.py -q
```

**Detailed design:** [`STEP_02_ML_INFERENCE.md`](STEP_02_ML_INFERENCE.md).

## Step 3: Scoring correctness

**Outcome:** Recommendations are driven by real matchup and synergy data.

**Fixes:** C6, M1.

**Tasks**

- [x] Instantiate `MatchupScorer` and `SynergyScorer` with database data at startup.
- [x] Normalize hero roles and primary attributes at one boundary.
- [x] Support `PrimaryAttribute::All` end to end.
- [x] Replace synthetic matchup seeds with real data or label them as fallback-only.
- [x] Make the fallback hero explicit and testable.
- [x] Add unit tests for role weights, unknown heroes, empty data, and data updates.
- [x] Expose a debug view showing which scorer contributed to the score.

**Acceptance**

- Changing a matchup row changes the affected recommendation.
- No production path silently uses synthetic matchup data.
- Unknown heroes produce a defined, tested result.
- Role casing cannot change scoring behavior.

## Step 4: Tauri IPC and manual drafting

**Outcome:** Manual pick/ban works for both sides and both actions.

**Fixes:** C5, H3.

**Tasks**

- [x] Choose one IPC convention: `rename_all = "snake_case"` on every Rust command, matching the frontend's snake_case arguments.
- [x] Apply the convention to every command, not only the failing ones.
- [x] Add typed error responses for invalid hero IDs, invalid turns, and missing model files.
- [x] Add a pick/ban mode toggle in `HeroPicker`.
- [x] Allow manual correction regardless of the current turn.
- [x] Add undo/reset behavior that is explicit and testable.
- [x] Add a smoke test that invokes each manual command through the frontend API layer.

**Acceptance**

- Manual pick and manual ban both succeed for Radiant and Dire.
- The UI does not disable valid corrections solely because of turn state.
- Invalid input returns a user-visible error instead of a panic or silent failure.

**Verification**

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
npm test --prefix apps/desktop
npm run build --prefix apps/desktop
```

**Detailed design:** [`STEP_04_IPC.md`](STEP_04_IPC.md).

## Step 5: Database paths, seeding, and patch version

**Outcome:** One database location and one patch source of truth.

**Fixes:** C4, H7, M3, part of L2.

**Tasks**

- [x] Create one path-resolution module used by the app and seeders (`paths.rs`).
- [x] Point seeders at `app_data_dir()` or seed through the running app.
- [x] Update `README.md` with the real Windows path.
- [x] Store the active patch in configuration or the database, not in code (`app_settings`, schema v2).
- [x] Make seeding idempotent and transactional (`bulk_upsert_*`).
- [x] Move blocking database work off async command threads (`spawn_blocking`).
- [x] Enable and verify SQLite WAL mode (`DataHealth.wal_enabled`).
- [x] Add a startup health check for schema and seed completeness (`get_data_health`).

**Acceptance**

- Seeder output is visible to the app without manual copying.
- The patch shown in the UI comes from persisted configuration.
- Re-running the seeder does not duplicate rows.
- No async command blocks on a mutex held during I/O.

**Verification**

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
cargo test --manifest-path infrastructure/seeders/Cargo.toml --bin seed_matchups
npm test --prefix apps/desktop
python ml\venv\Scripts\python.exe -m pytest ml\tests -q
```

End-to-end: `schema 2, heroes 126, matchups 9052, synergies 7381, active_patch 7.41, journal wal`.

**Detailed design:** [`STEP_05_DATA.md`](STEP_05_DATA.md).

## Step 6: Training pipeline

**Outcome:** A clean checkout can collect, train, evaluate, and export a model reproducibly.

**Fixes:** H1, H2, M4, part of L2.

**Tasks**

- [x] Reconstruct `data_collector.py` into a normal multi-line module.
- [x] Restore the `OpenDotaCollector` interface expected by `train_pipeline.py`.
- [x] Fix `train_pipeline.py` import path, boolean flags, and `--config` handling so the one-command path runs.
- [x] Split data by `start_time`, not `match_id` (with documented fallback).
- [x] Create train, validation, and test partitions.
- [x] Use validation only for early stopping; evaluate once on test.
- [x] Fix boolean flags so "skip training" and "evaluate only" actually work (`train_full.py`).
- [x] Add `onnx` and `onnxmltools` to `ml/requirements.txt`.
- [x] Record dataset hash, feature schema, hyperparameters, and metrics in a model manifest.
- [x] Export the feature schema with the model.
- [x] Add an end-to-end fixture test that trains, exports, and validates the ONNX artifact.

**Acceptance**

- `python ml/training/train_full.py` runs on a clean environment; `python -m training.train_pipeline` and `python training/train_pipeline.py` both resolve the collector and parse arguments.
- Test metrics are reported only after model selection is complete.
- The exported model and schema are versioned together.
- The Rust contract test passes against the exported artifact.

**Verification**

```powershell
python ml\venv\Scripts\python.exe -m pytest ml\tests -q   # 28 passed
python ml\venv\Scripts\python.exe ml\training\train_full.py --model-stem draft_model_xgboost_smoke --n-estimators 5
python ml\venv\Scripts\python.exe -m training.train_pipeline --help
```

**Detailed design:** [`STEP_06_TRAINING.md`](STEP_06_TRAINING.md).

## Step 7: Settings, GSI, and UI state

**Outcome:** User preferences are persisted and reflected in the running app.

**Fixes:** H4, H5, M2.

**Tasks**

- [x] Persist team side, GSI port, overlay enabled, and model path.
- [x] Apply the configured GSI port at startup and show bind failures.
- [x] Add optional authentication for the local GSI listener.
- [x] Track turn and side from GSI events with an explicit state machine.
- [x] Fix the overlay to use the player's actual side.
- [x] Stop reinitializing draft state on team change; migrate or confirm reset.
- [x] Bind settings toggles to the correct values.
- [x] Add validation and reset-to-default behavior.

**Acceptance**

- Settings survive an app restart.
- A non-default GSI port is used by the listener.
- The overlay is correct for Radiant and Dire.
- Changing team does not silently destroy an active draft.

**Verification**

```powershell
cargo test --manifest-path apps\desktop\src-tauri\Cargo.toml --lib   # 36 passed
npm test                                                            # 11 passed (apps\desktop)
npm run build
```

**Detailed design:** [`STEP_07_GSI_SETTINGS.md`](STEP_07_GSI_SETTINGS.md).

## Step 8: CI and quality gates

**Outcome:** Regressions are caught automatically.

**Fixes:** L1, M5.

**Tasks**

- [x] Add deterministic scripts for format, lint, typecheck, and tests.
- [x] Run `cargo fmt --check`, `cargo clippy`, and `cargo test`.
- [x] Run TypeScript typecheck and the frontend test suite.
- [x] Run Python compile and pytest.
- [x] Run the ML feature-contract test.
- [x] Run secret scanning on every change.
- [x] Add a workflow now that the repository is initialized.
- [ ] Require the gate for merges (repository branch-protection setting).
- [x] Add documentation for local reproduction.

**Acceptance**

- A pull request with an inverted probability, a missing feature, or a secret fails CI.
- The full gate runs in a clean environment.
- Local commands match CI commands.

**Note:** ESLint is declared in `package.json` but is neither installed nor
configured, so the frontend gate is TypeScript typecheck + build + tests; see
[`STEP_08_CI.md`](STEP_08_CI.md).

**Verification**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1   # all steps PASS
```

**Detailed design:** [`STEP_08_CI.md`](STEP_08_CI.md).

## Cross-cutting definition of done

A step is complete only when:

1. The code builds and the relevant tests pass.
2. The behavior is documented in `docs/` or the wiki.
3. No new secret, hardcoded path, or silent fallback is introduced.
4. The step's acceptance criteria are demonstrated, not assumed.
5. The status board is updated.

## Risk register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Key rotation breaks collectors | Collection stops | Document both keys; fail with actionable errors |
| Feature schema change breaks the model | Predictions fail | Version schema with model; golden test |
| DB migration loses seed data | Empty app | Idempotent seeding and backup before migration |
| Big-bang refactor destabilizes the app | No usable build | Keep steps independently shippable |
| Model retraining before contract fix | New model repeats old drift | Gate Step 6 on Steps 2 and 5 |
| CI added before repository initialization | No enforcement | Prepare scripts in Step 8, enable after `git init` |

## Progress log

| Date | Step | Change | Evidence |
| --- | --- | --- | --- |
| 2026-10-22 | 0 | Audit and plan created | [`AUDIT.md`](AUDIT.md) |
| 2026-10-22 | 1 | Design drafted | [`STEP_01_SECRETS_AND_ENV.md`](STEP_01_SECRETS_AND_ENV.md) |
| 2026-10-22 | 1 | Loader, guard, collector migration, and tests implemented | 12 tests pass; source scan clean; key rotation pending |
| 2026-10-22 | 2 | Schema, positive-class inference, golden fixture, and packaging implemented | 15 Python and 11 Rust tests pass; example prediction `0.4867` |
| 2026-10-22 | 3 | Scorer data wiring, role normalization, real seed export, debug breakdown | 20 Rust library tests, 1 seeder test, 15 Python tests pass |
| 2026-10-23 | 4 | snake_case IPC convention, typed errors, validation, undo stack, HeroPicker pick/ban toggle, frontend smoke test | 24 Rust library tests, 8 frontend tests, 15 Python tests pass |
| 2026-10-23 | 5 | Shared `paths` module, `app_settings` active patch (schema v2), patch-scoped queries, transactional seeding, WAL + `get_data_health`, `spawn_blocking`, README path | 24 Rust + 1 seeder + 8 frontend + 15 Python tests pass; seeded DB `schema 2 / 126 / 9052 / 7381 / 7.41` |
| 2026-10-23 | 6 | Time-based 3-way splits, val-only early stopping, manifest + schema export, reconstructed `data_collector.py`, repaired one-command `train_pipeline.py` | 28 Python tests pass; `train_full.py` smoke run emits model/ONNX/features/schema/config/manifest; collector and pipeline parse args under module and script invocation |
| 2026-10-23 | 7 | Persistent validated config in `app_settings`, real GSI draft schema + `activeteam` state machine, optional auth, graceful bind errors, side-aware overlay, no reinit on team change, correct settings bindings | 36 Rust + 11 frontend tests pass; `cargo fmt --check` clean; clippy adds no new warnings; `npm run build` succeeds |
| 2026-10-23 | 6 | `splits.py` (start_time-first 3-way), `manifest.py`, reworked `train_full.py`, `trainer.py` val-only early stopping, `onnx`/`onnxmltools` deps, `start_time` in processor | 24 Python tests pass; smoke run trains, exports ONNX + schema + manifest in ~17 s |
| 2026-10-23 | 8 | `scripts/verify.ps1` + `scripts/secret_scan.py` unified gate, `.github/workflows/ci.yml`, generated-artifact ignore policy, cleared ten pre-existing clippy lints | `verify.ps1` all steps PASS: secret scan clean, 36 Rust + 1 seeder + 11 frontend + 28 Python tests, `fmt`/`clippy -D warnings` clean |


