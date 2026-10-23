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
- `training/train_full.py`: the model producer. Defaults to the runtime contract (`draft_features.parquet` → 887 features, stem `draft_model_xgboost_full`) and excludes feature families the Rust runtime cannot build (`rad_lane_*`, `dire_lane_*`, `emb_*`). Splits by `start_time`, early-stops on validation, evaluates the test set once, exports model/ONNX/features/schema/config/manifest, verifies the feature count on `--skip-train`/`--evaluate-only`, and supports effective `--skip-train` and `--evaluate-only` flags.
- `training/trainer.py`: `time_based_split` prefers `start_time`, `three_way_split` feeds `train_model`, which early-stops on validation and reports test separately.
- `feature-engineering/processor.py`: writes `start_time` so future parquets carry a real timestamp.
- `training/data_collector.py`: reconstructed as a multi-line module. `OpenDotaCollector` (pro/public discovery, captain's-mode and All Pick detail fetches, disk-based resume, resumable metadata) and `StratzCollector` (GraphQL batch fetch and draft-enhancement) restore the interface `train_pipeline.py` expects.
- `training/train_pipeline.py`: the one-command path runs under both module and script invocation; `--include-pro`/`--include-ranked` are `BooleanOptionalAction`, and `--config` handling is robust.

## Test coverage

`ml/tests/test_training_splits.py` (7), `ml/tests/test_training_fixture.py` (4), `ml/tests/test_data_collector.py` (4), and `ml/tests/test_processor_features.py` (3) cover chronological disjointness, determinism, fingerprints, a full synthetic split → train → ONNX/schema/manifest run including `--skip-train` reuse and feature-count rejection, runtime-unsupported feature exclusion, `start_time` field mapping for both provider shapes, and offline collector pagination/dedup/resume plus the no-discovery `NameError` regression.

## Residual risks

- The embedding dataset (`draft_features_emb.parquet`) is an abandoned experiment and still lacks `start_time`.

Resolved: the bundled `_full` model (887 inputs) now matches `train_full.py`'s default selection; the old 887-vs-1139 gap came from defaulting to the embedding dataset with an `_emb` stem while the runtime contract is the non-embedding 887 set.

Resolved: `draft_features.parquet` was regenerated (2026-10-23) with `start_time` (890 columns, no zero rows), so splits use real timestamps instead of the `match_id` fallback.

Retrained: the bundled `draft_model_xgboost_full` was regenerated from the start-time-aware parquet (887 inputs; held-out test ROC-AUC 0.5388 / LogLoss 0.6893 / Brier 0.2481), refreshing the schema defaults and the golden fixture (match `8741426861`, label 1, positive probability `0.526863`).
