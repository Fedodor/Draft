# Step 6 Design: Training Pipeline

- Date: 2026-10-23
- Status: Implemented (collector and one-command pipeline restored)
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), H1, H2, M4, part of L2

## Implementation status

- [x] Split training data by `start_time`, falling back to `match_id` only when absent.
- [x] Create train, validation, and test partitions.
- [x] Use validation only for early stopping; evaluate the test set once.
- [x] Fix boolean flags so `--skip-train` and `--evaluate-only` are effective.
- [x] Add `onnx` and `onnxmltools` to `ml/requirements.txt`.
- [x] Record dataset hash, feature schema, hyperparameters, and metrics in a model manifest.
- [x] Export the feature schema alongside the model.
- [x] Add a fixture end-to-end test (split → train → ONNX + schema + manifest).
- [x] Reconstruct `data_collector.py` into a normal multi-line module.
- [x] Restore the `OpenDotaCollector` interface expected by `train_pipeline.py`.
- [x] Fix `train_pipeline.py` import path, boolean flags, and `--config` handling so both `python -m training.train_pipeline` and `python training/train_pipeline.py` run.
- [x] Default `train_full.py` to the runtime contract and exclude runtime-unsupported families (887 features, not 1139).
- [x] Regenerate the processed parquet so it carries `start_time` (889 → 890 columns).

## Problem

The advertised training path was not reproducible:

1. `data_collector.py` is 31,144 bytes on a single line, so `train_pipeline.py` fails on import (H1).
2. The test split was reused for early stopping and final evaluation, and split by `match_id` rather than time (H2).
3. Boolean flags used `store_true, default=True`, so "no train" could not be requested, and `onnx`/`onnxmltools` were missing from the dependency list (M4).
4. Nothing recorded which data, schema, and hyperparameters produced an artifact (L2).

## Shared splitting and fingerprinting

`ml/training/splits.py` centralizes splitting so scripts cannot drift:

- `resolve_time_column` prefers `start_time`/`start_time_utc`; when absent it falls back to `match_id` (monotonic with time in Dota) and reports `used_start_time = False`.
- `time_split` sorts chronologically and returns contiguous train/validation/test partitions. Every training row precedes every validation row, which precedes every test row. Fractions must be positive and sum to 1.0.
- `file_sha256`, `schema_fingerprint`, and `frame_fingerprint` produce reproducible identifiers for the manifest.

`ml/feature-engineering/processor.py` now writes `start_time` into the processed features, so regenerated parquets split on real timestamps. `start_time`, `match_id`, and `radiant_win` are excluded from model inputs.

## Manifest

`ml/training/manifest.py` assembles a `{stem}_manifest.json` next to every export:

- dataset path, SHA-256, row count
- feature schema: count, fingerprint, ordered names
- split provenance: time column, whether `start_time` was used, fractions, sizes, boundaries
- hyperparameters and `best_iteration`
- validation and test metrics (test reported once, after selection)
- artifact filenames and library versions

## `train_full.py`

The script that produces the bundled feature set (default stem `draft_model_xgboost_full`, default data `ml/datasets/processed/draft_features.parquet`) was reworked:

- Exposes `select_feature_columns`, `encode_features`, `evaluate`, and `run` so it is testable.
- Excludes identity/label/time columns and the feature families the Rust runtime cannot reconstruct (`rad_lane_*`, `dire_lane_*`, `emb_*`), so the exported model always matches `ml.rs build_feature_map`.
- Defaults target the bundled contract: `draft_features.parquet` (889 columns → 887 features) as `draft_model_xgboost_full`. The embedding dataset is a separate experiment and is never the default.
- Splits by `start_time`, early-stops on validation, then evaluates the held-out test set exactly once.
- Writes `<stem>.joblib` (XGBoost native), `<stem>.onnx`, `<stem>_features.json`, `<stem>_schema.json`, `<stem>_config.json`, and `<stem>_manifest.json`.
- New CLI: `--train-frac/--val-frac/--test-frac`, `--n-estimators`, `--seed`, `--config`, `--current-patch`, `--skip-train`, `--evaluate-only`.
- `--skip-train` reuses an existing model (no fit, `best_iteration` recorded as null) and verifies the dataset's feature count matches the model's `n_features_in_`, failing with an actionable message instead of a raw shape error; `--evaluate-only` additionally suppresses artifact writes and only reports metrics plus the manifest.

`export_model_schema.py` defaults to the same `draft_features.parquet`, so the schema export matches the model contract.

## `trainer.py`

- `time_based_split` now prefers `start_time` and warns on the `match_id` fallback.
- Added `three_way_split`; `train_model` uses it, early-stops on validation, and reports validation and test separately. Test metrics are also attached under `metrics["validation"]`.
- Added `--train-frac` and `--val-frac`.

## Collector and one-command pipeline

`ml/training/data_collector.py` was reconstructed as a normal multi-line module:

- `OpenDotaCollector` (`_rate_limited_get`, `get_pro_matches`, `get_public_matches`, `_is_captains_mode`, `_is_all_pick`, `get_match_details`, `collect_matches_paginated`) restores the interface `train_pipeline.py` expects. `collect_matches_paginated` keeps disk state as the resume source of truth, writes `output_dir/<id>.json` for captain's mode and `output_dir/allpick/<id>.json` for All Pick, and buffers metadata before fetching so runs are resumable.
- `StratzCollector` (`MATCHES_BY_IDS_QUERY`, `MATCH_QUERY`, `_graphql_request`, `get_match_by_id`, `get_matches_by_ids`, `enhance_matches`) enhances already-downloaded matches with draft data.
- Repairs made during reconstruction: `discovered_public` is initialized before the ranked branch (the metadata write referenced it unconditionally, raising `NameError` when `include_ranked=False`); the summary counter checks the ok flag rather than the error field; and `--include-pro`/`--include-ranked` use `argparse.BooleanOptionalAction` (previously `store_true, default=True` made them impossible to disable).

`ml/training/train_pipeline.py` fixes:

- `ml/training` is added to `sys.path`, so `from settings import ...` and `from data_collector import ...` resolve under both `python -m training.train_pipeline` and `python training/train_pipeline.py`.
- `--include-pro`/`--include-ranked` use `argparse.BooleanOptionalAction`.
- Default model config is assigned before `--config` is read, and a missing `--config` path is a hard error (previously an unreadable config left `args.model_config` unset).

## Tests

`ml/tests/test_training_splits.py` (7 tests): start-time preference, `match_id` fallback, chronological and disjoint partitions, determinism across input order, fraction validation, and schema/frame fingerprint stability.

`ml/tests/test_training_fixture.py` (4 tests): runs the real pipeline on a synthetic parquet into a temp directory, asserting manifest metrics, split sizes, schema, and a valid ONNX session whose input width matches the manifest; then re-runs with `--skip-train` to confirm model reuse. It also asserts that runtime-unsupported families (`emb_*`, `rad_lane_*`, `dire_lane_*`) and identity/time columns are excluded from the feature set, and that `--skip-train` rejects a dataset whose feature count does not match the model.

`ml/tests/test_data_collector.py` (4 tests): collector interface presence, game-mode helpers, offline pagination/dedup/download via a fake collector, and the no-discovery path that previously raised `NameError`.

`ml/tests/test_processor_features.py` (3 tests): `start_time` is populated from the OpenDota `start_time` field, the STRATZ `startDateTime` field, and defaults to 0 when neither is present.

## Verification

```powershell
python ml\venv\Scripts\python.exe -m pytest ml\tests -q   # 33 passed
python ml\venv\Scripts\python.exe ml\training\train_full.py --model-stem draft_model_xgboost_smoke --n-estimators 5
python ml\venv\Scripts\python.exe -m training.data_collector --help
python ml\venv\Scripts\python.exe -m training.train_pipeline --help
```

The smoke run wrote model, ONNX, features, schema, config, and manifest in ~17 s, reporting validation then test metrics. The collector and pipeline both parse arguments under the module and script invocation styles, and `--no-include-pro/--no-include-ranked` are accepted.

## Residual risks

- The embedding dataset (`draft_features_emb.parquet`) is an abandoned experiment and was not regenerated, so it still lacks `start_time`; the default contract uses the non-embedding parquet.

Resolved (feature contract): `train_full.py` now defaults to the bundled contract (`draft_features.parquet` → 887 features, stem `draft_model_xgboost_full`) and excludes the families the runtime cannot build, so `--evaluate-only` reproduces the bundled model instead of failing with a feature-shape mismatch. The old 887-vs-1139 gap came from defaulting to the embedding dataset with a matching `_emb` stem while the runtime contract is the non-embedding 887 set.

Resolved (start time): `draft_features.parquet` was regenerated (2026-10-23) and now carries `start_time` (int64, no zero rows; range 2017-05 to 2026-08), so `time_split` uses real timestamps and no longer logs the `match_id` fallback. `ml/tests/test_processor_features.py` pins the `start_time` field mapping for both the OpenDota (`start_time`) and STRATZ (`startDateTime`) shapes. The bundled schema's winrate/synergy defaults still reflect the pre-regeneration data and will refresh on the next retrain.
