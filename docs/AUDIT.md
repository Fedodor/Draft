# Technical Audit: Smart Draft Assistant

- Date: 2026-10-22
- Status: Complete
- Scope: Full repository: Tauri desktop app, React frontend, Rust backend, Python ML pipeline, SQLite seed data, GSI integration, and documentation.
- Canonical source: this document. The wiki version is a summary in [`wiki/audits/project-audit-2026-10-22.md`](wiki/audits/project-audit-2026-10-22.md).

## Executive summary

The application is a functional prototype with a clear architecture, but it is not yet safe or reliable enough for regular use. The main risks are:

1. API credentials are stored in plaintext and hardcoded in multiple scripts.
2. The ML model is called with the wrong probability and a drifted feature contract.
3. Several core product paths are broken or only partially wired: manual drafting, matchup/synergy scoring, and database seeding.
4. The training pipeline is not reproducible and cannot currently collect its own data.

The directory is not currently a Git repository, so there is no commit history here to leak. That is the best time to add secret hygiene before initializing version control.

> Remediation status: Steps 1-8 are implemented as of 2026-10-23, with key rotation and the "require the gate for merges" repository setting still pending. Step 1 (secrets and environment), Step 2 (ML inference correctness), Step 3 (scoring correctness), Step 4 (Tauri IPC and manual drafting), Step 5 (database paths, seeding, and active patch), Step 6 (training pipeline), Step 7 (settings, GSI, and UI state), and Step 8 (CI and quality gates) each have a design doc; see [`STEP_01_SECRETS_AND_ENV.md`](STEP_01_SECRETS_AND_ENV.md), [`STEP_02_ML_INFERENCE.md`](STEP_02_ML_INFERENCE.md), [`STEP_03_SCORING.md`](STEP_03_SCORING.md), [`STEP_04_IPC.md`](STEP_04_IPC.md), [`STEP_05_DATA.md`](STEP_05_DATA.md), [`STEP_06_TRAINING.md`](STEP_06_TRAINING.md), [`STEP_07_GSI_SETTINGS.md`](STEP_07_GSI_SETTINGS.md), and [`STEP_08_CI.md`](STEP_08_CI.md). Findings below remain the baseline audit.

## Severity summary

| Severity | Meaning | Main areas |
| --- | --- | --- |
| Critical | Produces wrong results, exposes credentials, or breaks a primary feature | Secrets, ML inference, feature contract, DB path, IPC, scoring |
| High | Major feature is unreliable or training cannot be trusted | Training pipeline, leakage, manual drafting UX, settings/GSI, CSP |
| Medium | Degrades correctness, maintainability, or performance | Patch hardcoding, role matching, async DB access, UI state, dependency gaps |
| Low | Documentation and housekeeping | Stale docs, empty docs, missing CI |

## Critical findings

### C1. Credentials are exposed in the working tree

- `ml/.env` contains `OPENDOTA_API_KEY` and `STRATZ_API_KEY`.
- 19 hardcoded credential matches exist across 12 files under `ml/training/`:
  `collect_allpick.py`, `collect_100k.py`, `collect_opendota_pro.py`, `collect_opendota.py`, `collect_stratz.py`, `collect_pro_matches.py`, `collect_stratz_tournaments.py`, `collect_stratz_allpick.py`, `fetch_versions.py`, `stratz_player_test.py`, `test_stratz_browse.py`, and `test_pro_players.py`.
- At audit time there was no root `.gitignore` and the project was not a Git repository. Initializing Git without a secret guard would have committed the keys. A root `.gitignore` was added during Step 1.

Impact: key misuse, rate-limit exhaustion, account suspension, and accidental publication.

### C2. ML inference uses the wrong class probability

- The ONNX model has two outputs: `label` (`int64`) and `probabilities` (`float32`).
- A live probe returned `label = [1]` and `probabilities = [[0.42500144, 0.57499856]]`.
- The Rust path reads `probabilities.first()` and treats `0.425` as the Radiant win probability.
- The correct value for label `1` is the second element, `0.57499856`.

Impact: predictions are materially wrong and biased toward 0.5 or the losing class.

### C3. The Rust feature vector does not match the bundled model

- The bundled model expects 887 inputs.
- The Rust feature builder omits `patch` and sets `patch_version` to the literal `7.41`.
- `train_full.py` encodes `patch` and `patch_version` as pandas categorical codes, so the model learned integer category codes rather than version strings.
- Missing values are silently filled with `0.0` in Rust, which is a valid-looking value for many features.
- Hero pick slots and several match-context features participate in the contract, but there is no shared schema or golden test between Python and Rust.

Impact: even with the probability fixed, predictions can be wrong because the model receives a different distribution from training.

### C4. Seeded data and the application use different database paths

- Seeders use `dirs::data_dir()/smart-draft-assistant`.
- The Tauri app uses `app_handle.path().app_data_dir()`, which resolves to `%APPDATA%\com.smartdraftassistant.app` on Windows.
- `README.md` documents `%APPDATA%\smart-draft-assistant`.

Impact: seeders appear to succeed, but the app opens an empty or incomplete database.

### C5. Manual draft IPC arguments do not match Tauri v2 conventions

- Tauri v2 converts Rust command arguments to camelCase by default.
- Frontend `api.ts` sends snake_case names such as `hero_id` and `model_path`.
- Commands such as `manual_pick`, `manual_ban`, `get_hero_details`, and `load_ml_model` lack `#[tauri::command(rename_all = "snake_case")]`.

Impact: manual pick/ban and model loading fail at runtime despite compiling.

### C6. Matchup and synergy scorers are not wired to data

- `MatchupScorer::with_data` and `SynergyScorer::with_data` exist but are not used by the engine.
- The engine falls back to static or synthetic behavior.
- Role names are compared with inconsistent casing.
- Seed matchup data is synthetic and not suitable for real recommendations.

Impact: recommendations are not driven by the matchup/synergy data the project collects.

## High findings

### H1. The data collection and training entry point is dead

- `ml/training/data_collector.py` is 31,144 bytes with zero newlines.
- Because the file starts with `#!`, the entire file is a single comment.
- `py_compile` succeeds, but importing the module yields no `OpenDotaCollector`.
- `train_pipeline.py` imports `OpenDotaCollector` and therefore fails before training.

Impact: the advertised one-command training path cannot run.

Resolved (Step 6): the module is reconstructed as multi-line code exposing `OpenDotaCollector` and `StratzCollector`, and `train_pipeline.py` runs under both `python -m training.train_pipeline` and `python training/train_pipeline.py`. See [`STEP_06_TRAINING.md`](STEP_06_TRAINING.md).

### H2. Training evaluation leaks data

- The test split is reused for early stopping and final evaluation.
- Splitting is keyed by `match_id`, not by time, so related matches can cross splits.
- CLI flags use `store_true, default=True`, making the "no train" path ineffective.

Impact: reported model quality is optimistic and not reproducible.

### H3. Manual drafting is blocked by UI logic

- `HeroPicker` disables actions when it is not the current turn.
- There is no pick/ban mode toggle.
- This removes the ability to correct a draft or simulate the opponent.

Impact: core manual workflow is unusable in normal testing.

### H4. Settings and GSI are not fully functional

- Settings toggles are bound incorrectly.
- Port and overlay preferences are not applied.
- GSI uses a fixed port `3000`.
- Turn tracking depends on incomplete state updates.

Impact: users cannot configure the app for their environment.

Resolved (Step 7): `AppConfig` is validated and persisted in `app_settings`; `set_config`/`get_config`/`restart_gsi` are real; startup applies the port, auth, auto-start, and model path, with bind failures surfaced. `GSIDraft` now parses the real payload (`activeteam`/`pick`, `team2`/`team3` slots) and the turn is derived from an explicit state machine. See [`STEP_07_GSI_SETTINGS.md`](STEP_07_GSI_SETTINGS.md).

### H5. Overlay assumes the player is always Radiant

- `DraftOverlay` treats Radiant as the allied team unconditionally.
- Team-side changes do not consistently update the overlay.

Impact: recommendations are reversed for Dire players.

Resolved (Step 7): `DraftOverlay` orients from `selectedTeam`, and the `gsi-team` event updates that side from `player.team_name`.

### H6. Tauri security posture is weak

- `tauri.conf.json` sets `"csp": null`.
- Development tooling and broad capabilities increase the attack surface.
- No explicit validation exists for model paths supplied by the frontend.

Impact: weaker defense in depth, especially if remote content is ever introduced.

### H7. Patch version is hardcoded

- `database.rs` uses a literal `7.41`.
- The ML feature builder also hardcodes `7.41`.
- Patch-specific data and model behavior can drift silently.

Impact: stale recommendations after a game patch.

## Medium findings

### M1. Role and hero fallback handling is inconsistent

- Role comparisons use inconsistent casing.
- Unknown heroes fall back to a placeholder rather than a validated error or a safe default.

### M2. Team changes reinitialize too much state

- `App.tsx` reinitializes draft state when the team changes.
- This can discard a valid in-progress draft.

Resolved (Step 7): `App.tsx` separates the one-time mount load from the
team-change effect, which now only refreshes recommendations.

### M3. Async commands use blocking database locks

- Database access uses blocking mutexes inside async command paths.
- Under contention this can stall the Tauri runtime.

### M4. Python dependency and flag hygiene

- `ml/requirements.txt` does not include `onnx` or `onnxmltools`, although export needs them.
- Several flags have ineffective defaults.
- `python-dotenv` is not part of the declared dependencies.

### M5. Documentation is stale or incomplete

- `README.md` claims 894 or 896 features; the actual model uses 887.
- `README.md` has no Documentation section.
- `docs/RefactorML.md` is empty.
- Setup and database paths are inconsistent.

## Low findings

### L1. No CI or quality gate

There is no automated check for Rust formatting/clippy/tests, TypeScript typecheck/lint, Python compile/tests, secret scanning, or the ML feature contract.

Resolved (Step 8): `scripts/verify.ps1` runs the full gate locally and in CI, `scripts/secret_scan.py` scans every change, and `.github/workflows/ci.yml` enforces it. ESLint remains declared but uninstalled/unconfigured, so lint is covered by typecheck + build + tests for now. See [`STEP_08_CI.md`](STEP_08_CI.md).

### L2. Duplicated or unclear ownership of configuration

Database paths, model paths, patch versions, ports, and feature names are defined in multiple layers without one source of truth.

### L3. Generated data is not separated from source

Large generated datasets and caches live next to source without a repository-level ignore policy.

## What works

- The Rust workspace compiles.
- The frontend typechecks.
- The ONNX model loads and runs.
- The SQLite schema exists and heroes can be seeded.
- The React UI and Tauri shell are structurally sound.
- The ML feature list is available in the model metadata and can be used as the basis for a contract test.

## Verification notes

- `cargo check` completed successfully.
- `tsc --noEmit` completed successfully.
- ONNX Runtime loaded `draft_model_xgboost_full.onnx` and produced the outputs quoted in C2.
- `data_collector.py` compiles but cannot be imported as a usable module.
- Credential scan found 19 matches across 12 files.
- `Test-Path .git` returned `False`.

## Recommended reading order

1. [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md): the 8-step fix plan.
2. [`STEP_01_SECRETS_AND_ENV.md`](STEP_01_SECRETS_AND_ENV.md): detailed first step.
3. [`wiki/index.md`](wiki/index.md): persistent wiki entry point.

