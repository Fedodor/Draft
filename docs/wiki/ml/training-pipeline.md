# Training Pipeline

- Type: ML
- Status: Implemented (Step 6)
- Canonical source: [`../../STEP_06_TRAINING.md`](../../STEP_06_TRAINING.md)
- Related: [`inference-contract.md`](inference-contract.md), [Remediation plan](../plans/remediation-plan.md)

## Why it matters

Reported model quality was optimistic and not reproducible: the test split was reused for early stopping, splitting used `match_id` rather than time, "no train" flags were inert, and nothing recorded the data/schema/hyperparameters behind an artifact.

## What is in place

- `training/splits.py`: `start_time`-first chronological `time_split` into train/validation/test, `match_id` fallback with an explicit flag, and stable `file_sha256`/`schema_fingerprint`/`frame_fingerprint` helpers.
- `training/manifest.py`: writes `{stem}_manifest.json` with dataset hash, feature-schema fingerprint and order, split provenance, hyperparameters, `best_iteration`, validation/test metrics, artifacts, and library versions.
- `training/train_full.py`: the model producer. Splits by `start_time`, early-stops on validation, evaluates the test set once, exports model/ONNX/features/schema/config/manifest, and supports effective `--skip-train` and `--evaluate-only` flags.
- `training/trainer.py`: `time_based_split` prefers `start_time`, `three_way_split` feeds `train_model`, which early-stops on validation and reports test separately.
- `feature-engineering/processor.py`: writes `start_time` so future parquets carry a real timestamp.
- `training/data_collector.py`: reconstructed as a multi-line module. `OpenDotaCollector` (pro/public discovery, captain's-mode and All Pick detail fetches, disk-based resume, resumable metadata) and `StratzCollector` (GraphQL batch fetch and draft-enhancement) restore the interface `train_pipeline.py` expects.
- `training/train_pipeline.py`: the one-command path runs under both module and script invocation; `--include-pro`/`--include-ranked` are `BooleanOptionalAction`, and `--config` handling is robust.

## Test coverage

`ml/tests/test_training_splits.py` (7), `ml/tests/test_training_fixture.py` (2), and `ml/tests/test_data_collector.py` (4) cover chronological disjointness, determinism, fingerprints, a full synthetic split → train → ONNX/schema/manifest run including `--skip-train` reuse, and offline collector pagination/dedup/resume plus the no-discovery `NameError` regression.

## Residual risks

- No `start_time` in the current processed parquet, so runs fall back to `match_id` until features are regenerated.
- The bundled `_full` model (887 inputs) does not match `train_full.py`'s 1139-feature selection; retraining remains gated on the feature-contract fix.
