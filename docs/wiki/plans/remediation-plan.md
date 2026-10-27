# Remediation Plan

- Type: Execution plan
- Status: Implemented (Steps 1-8; Steps 9-10 post-remediation increments)
- Canonical source: [`../../REMEDIATION_PLAN.md`](../../REMEDIATION_PLAN.md)
- Source audit: [Project audit, 2026-10-22](../audits/project-audit-2026-10-22.md)

## Goal

Fix the prototype in eight ordered steps, keeping the application buildable between steps and using tests to lock in each correctness fix.

## Order

| Step | Theme | Exit gate | Status |
| --- | --- | --- | --- |
| 1 | Secrets and environment | No plaintext credentials; env contract enforced | Implemented, OpenDota rotation pending |
| 2 | ML inference correctness | Rust matches Python ONNX output | Implemented |
| 3 | Scoring correctness | Scorers use real data and pass tests | Implemented |
| 4 | Tauri IPC and manual drafting | Manual pick/ban works end to end | Implemented |
| 5 | Database paths, seeding, patch | Seeders and app share one database | Implemented |
| 6 | Training pipeline | Reproducible training and export | Implemented |
| 7 | Settings, GSI, UI state | Configuration persists and applies | Implemented |
| 8 | CI and quality gates | Automated checks block regressions | Implemented |

## Waves

1. Safety and unblocking: Steps 1, 4, 5.
2. Correctness: Steps 2, 3.
3. Reliability: Steps 6, 7.
4. Enforcement: Step 8.

Step 6 must not start until Steps 2 and 5 establish the feature contract and data path. Otherwise a retrained model will repeat the current drift.

## Non-negotiables

- No secrets in source, docs, logs, or the first Git commit.
- No silent `0.0` feature fallback in development.
- No model retraining while the evaluation split leaks data.
- No production recommendation path backed by synthetic matchup data.
- No step is complete without evidence and an updated status board.

## Implemented steps

Step 1 is documented in detail at [`../../STEP_01_SECRETS_AND_ENV.md`](../../STEP_01_SECRETS_AND_ENV.md), with its operational summary at [Secrets management](../security/secrets-management.md).

Step 2 is documented at [`../../STEP_02_ML_INFERENCE.md`](../../STEP_02_ML_INFERENCE.md), with its contract summary at [ML inference contract](../ml/inference-contract.md).

Step 3 is documented at [`../../STEP_03_SCORING.md`](../../STEP_03_SCORING.md), with its data-flow summary at [Scoring data flow](../engine/scoring.md).

Step 4 is documented at [`../../STEP_04_IPC.md`](../../STEP_04_IPC.md), with its contract summary at [Tauri IPC conventions](../engine/ipc.md).

Step 5 is documented at [`../../STEP_05_DATA.md`](../../STEP_05_DATA.md), with its summary at [Database, seeding, and patch](../engine/data.md).

Step 6 is documented at [`../../STEP_06_TRAINING.md`](../../STEP_06_TRAINING.md), with its summary at [Training pipeline](../ml/training-pipeline.md). The collector is reconstructed and the one-command `train_pipeline.py` path runs.

Step 7 is documented at [`../../STEP_07_GSI_SETTINGS.md`](../../STEP_07_GSI_SETTINGS.md), with its summary at [Settings, GSI, and UI state](../engine/settings-gsi.md).

Step 8 is documented at [`../../STEP_08_CI.md`](../../STEP_08_CI.md), with its summary at [CI and quality gates](../ci/quality-gates.md). One script (`scripts/verify.ps1`) runs the gate locally and in CI.

## Post-remediation increments

Step 9 (2026-10-27) implements the previously dead meta layer: a patch-scoped meta snapshot exported from winrates/pickrates, seeded by `seed_matchups`, used in scoring and explanations, reported by health, and paired with label-based recommendation categories. Detail: [`../../STEP_09_META.md`](../../STEP_09_META.md); summary: [Patch meta data](../engine/meta.md).

Step 10 (2026-10-27) adds a lazy score-breakdown disclosure to each recommendation card, surfacing the six weighted components from `get_score_breakdown`. Detail: [`../../STEP_10_SCORE_BREAKDOWN.md`](../../STEP_10_SCORE_BREAKDOWN.md); summary: [Score breakdown panel](../engine/score-breakdown.md).

## Related

- [Secrets management](../security/secrets-management.md)
- [ML inference contract](../ml/inference-contract.md)
- [Scoring data flow](../engine/scoring.md)
- [Tauri IPC conventions](../engine/ipc.md)
- [Database, seeding, and patch](../engine/data.md)
- [Training pipeline](../ml/training-pipeline.md)
- [Settings, GSI, and UI state](../engine/settings-gsi.md)
- [CI and quality gates](../ci/quality-gates.md)
- [Patch meta data](../engine/meta.md)
- [Score breakdown panel](../engine/score-breakdown.md)
- [Wiki index](../index.md)

