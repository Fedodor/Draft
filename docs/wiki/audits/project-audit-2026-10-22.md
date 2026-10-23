# Project Audit, 2026-10-22

- Type: Full technical audit
- Status: Complete
- Canonical source: [`../../AUDIT.md`](../../AUDIT.md)
- Scope: Tauri desktop app, React frontend, Rust backend, Python ML pipeline, data seeding, GSI, and documentation.

## Verdict

The project is a functional prototype, not yet a reliable product. The core architecture is sound, but credential handling, ML inference, and several primary product paths are broken or unverified.

The directory is not currently a Git repository, so no credentials are committed here. That makes this the right time to add a repository guard before `git init`.

> Remediation status: Step 1 is implemented with key rotation pending; Steps 2 (ML inference), 3 (scoring), 4 (Tauri IPC and manual drafting), 5 (database paths, seeding, and active patch), and 6 (training pipeline) are implemented. See [Secrets management](../security/secrets-management.md), [ML inference contract](../ml/inference-contract.md), [Scoring data flow](../engine/scoring.md), [Tauri IPC conventions](../engine/ipc.md), [Database, seeding, and patch](../engine/data.md), and [Training pipeline](../ml/training-pipeline.md).

## Critical findings

| ID | Finding | Impact |
| --- | --- | --- |
| C1 | Keys in `ml/.env` and hardcoded in 12 scripts | Credential exposure if shared or committed |
| C2 | Rust uses probability `0.425` instead of `0.575` | Inverted or biased ML predictions |
| C3 | Rust feature vector differs from the 887-input model | Silent wrong predictions |
| C4 | Seeders and app use different database paths | App sees an empty database |
| C5 | Tauri v2 arguments are camelCase, frontend sends snake_case | Manual draft and model loading fail |
| C6 | Matchup and synergy scorers are not wired | Recommendations ignore collected data |

Current status: C1 remediated with rotation pending; C2 and C3 remediated in Step 2; C6 and the role-casing part of M1 remediated in Step 3; C5 and H3 remediated in Step 4; C4, the hardcoded-patch part of H7, and the blocking-lock part of M3 remediated in Step 5; H1, H2, and the ONNX-dependency/flag parts of M4 remediated in Step 6; H4, H5, and M2 remediated in Step 7.

## High findings

- Training entry point imported a dead `data_collector.py` (single 31 KB line); fixed in Step 6 — the module is reconstructed and `train_pipeline.py` runs under both invocation styles.
- Test split reusing early stopping was fixed in Step 6; splitting now prefers `start_time` (with a documented `match_id` fallback and a manifest record).
- Manual drafting was blocked by turn logic and had no pick/ban toggle (fixed in Step 4).
- Settings, GSI port, and overlay side handling were incomplete; fixed in Step 7 — config persists and is applied, GSI parses the real schema with an explicit turn state machine, and the overlay is side-aware.
- `tauri.conf.json` has `"csp": null`.
- Patch version was hardcoded as `7.41`; now persisted in `app_settings` (Step 5). The ML feature builder still hardcodes it.

## Medium and low findings

- Inconsistent role casing and unsafe hero fallback.
- Draft state reset on team change; fixed in Step 7 (team change now only refreshes recommendations).
- Blocking database locks in async command paths (moved to `spawn_blocking` in Step 5).
- Missing `onnx`/`onnxmltools` dependencies and ineffective boolean flags.
- Stale README counts (894/896 versus 887), empty `docs/RefactorML.md`, no documentation index.
- No CI, no shared feature schema, no repository ignore policy.

## What works

- Rust compiles and the frontend typechecks.
- The ONNX model loads and runs.
- SQLite schema and hero seeding exist.
- The Tauri shell and React component structure are usable as a foundation.

## Evidence

- ONNX probe: `label = [1]`, `probabilities = [[0.42500144, 0.57499856]]`.
- `data_collector.py`: 31,144 bytes, zero newlines, single comment, no usable `OpenDotaCollector`.
- Credential scan: 19 matches across 12 files.
- `Test-Path .git`: `False`.

## Related

- [Remediation plan](../plans/remediation-plan.md)
- [Secrets management](../security/secrets-management.md)

