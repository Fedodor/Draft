# Smart Draft Assistant Wiki

Persistent, linked summaries of the project's audits, plans, and operational knowledge. Canonical detail lives in `docs/`; this wiki is the fast entry point and cross-reference layer.

## Start here

- [Technical audit, 2026-10-22](audits/project-audit-2026-10-22.md): what is broken and why it matters.
- [Remediation plan](plans/remediation-plan.md): the 8-step fix order and status.
- [Secrets management](security/secrets-management.md): environment contract and rotation runbook.
- [ML inference contract](ml/inference-contract.md): schema, positive class, and golden-vector checks.
- [Scoring data flow](engine/scoring.md): startup wiring, role normalization, and real seed data.
- [Tauri IPC conventions](engine/ipc.md): command naming, typed errors, manual drafting, and undo.
- [Database, seeding, and patch](engine/data.md): shared paths, active patch, transactional seeding, and health checks.
- [Training pipeline](ml/training-pipeline.md): chronological splits, validation-only early stopping, and the model manifest.
- [Settings, GSI, and UI state](engine/settings-gsi.md): persisted config, real GSI schema, turn state machine, and side-aware overlay.
- [CI and quality gates](ci/quality-gates.md): the single verify script, the CI workflow, and the merge gate.
- Canonical detail: [`../AUDIT.md`](../AUDIT.md), [`../REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md), [`../STEP_01_SECRETS_AND_ENV.md`](../STEP_01_SECRETS_AND_ENV.md), [`../STEP_02_ML_INFERENCE.md`](../STEP_02_ML_INFERENCE.md), [`../STEP_03_SCORING.md`](../STEP_03_SCORING.md), [`../STEP_04_IPC.md`](../STEP_04_IPC.md), [`../STEP_05_DATA.md`](../STEP_05_DATA.md), [`../STEP_06_TRAINING.md`](../STEP_06_TRAINING.md), [`../STEP_07_GSI_SETTINGS.md`](../STEP_07_GSI_SETTINGS.md), [`../STEP_08_CI.md`](../STEP_08_CI.md).

## Catalog

| Page | Type | Summary | Status | Updated | Source |
| --- | --- | --- | --- | --- | --- |
| [Project audit](audits/project-audit-2026-10-22.md) | Audit | Severity-ranked findings across ML, Rust, frontend, data, and security | Complete | 2026-10-22 | [`AUDIT.md`](../AUDIT.md) |
| [Remediation plan](plans/remediation-plan.md) | Plan | Eight ordered steps with dependencies, acceptance criteria, and waves | Implemented (Steps 1-8) | 2026-10-23 | [`REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md) |
| [Secrets management](security/secrets-management.md) | Security | Key inventory, env loader contract, rotation and scanning runbook | Implemented, rotation pending | 2026-10-22 | [`STEP_01_SECRETS_AND_ENV.md`](../STEP_01_SECRETS_AND_ENV.md) |
| [ML inference contract](ml/inference-contract.md) | ML | Feature schema, positive-class probability, patch encoding, golden-vector tests | Implemented | 2026-10-22 | [`STEP_02_ML_INFERENCE.md`](../STEP_02_ML_INFERENCE.md) |
| [Scoring data flow](engine/scoring.md) | Engine | Scorer wiring, role canonicalization, real seed export, debug breakdown | Implemented | 2026-10-22 | [`STEP_03_SCORING.md`](../STEP_03_SCORING.md) |
| [Tauri IPC conventions](engine/ipc.md) | Engine | snake_case command args, typed errors, manual drafting, undo, frontend smoke test | Implemented | 2026-10-23 | [`STEP_04_IPC.md`](../STEP_04_IPC.md) |
| [Database, seeding, and patch](engine/data.md) | Engine | Shared DB path, active patch in `app_settings`, transactional seeding, health check, WAL | Implemented | 2026-10-23 | [`STEP_05_DATA.md`](../STEP_05_DATA.md) |
| [Training pipeline](ml/training-pipeline.md) | ML | `start_time`-first 3-way split, validation-only early stopping, model manifest, reconstructed collector | Implemented | 2026-10-23 | [`STEP_06_TRAINING.md`](../STEP_06_TRAINING.md) |
| [Settings, GSI, and UI state](engine/settings-gsi.md) | Engine | Persisted config, real GSI schema, turn state machine, optional auth, side-aware overlay | Implemented | 2026-10-23 | [`STEP_07_GSI_SETTINGS.md`](../STEP_07_GSI_SETTINGS.md) |
| [CI and quality gates](ci/quality-gates.md) | Process | One `verify.ps1` gate, GitHub Actions workflow, secret scan, merge requirement | Implemented | 2026-10-23 | [`STEP_08_CI.md`](../STEP_08_CI.md) |

## Conventions

- Every wiki page links to its canonical source in `docs/`.
- `index.md` is the catalog; add new pages here when created.
- `log.md` is append-only and uses `## [YYYY-MM-DD] ingest | Source Title`.
- Wiki pages summarize and connect; they do not fork authoritative detail.
- Do not store credentials, tokens, or personal data in the wiki.

## Next pages

Create these as the corresponding steps begin, not in advance:

- Release and packaging gate (Tauri bundle smoke test).


