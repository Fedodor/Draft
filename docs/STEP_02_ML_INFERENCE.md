# Step 2 Design: ML Inference Correctness

- Date: 2026-10-22
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit finding: [`AUDIT.md`](AUDIT.md), C2 and C3

## Implementation status

- [x] Exported a runtime feature schema next to the bundled model.
- [x] Added a deterministic Python golden input/output fixture.
- [x] Fixed Rust probability extraction to use `probabilities[0][1]`.
- [x] Rust reads feature names from the schema, with legacy `features.json` fallback.
- [x] Removed the inference-time `patch_version = 7.41` literal.
- [x] Encoded `patch` and `patch_version` using the training category order.
- [x] Replaced hardcoded match defaults with schema medians.
- [x] Added schema, contract, probability-selection, and ONNX golden tests.
- [ ] UI smoke test after the next desktop build.

## Problem

The Rust runtime and Python exporter disagreed about the model contract in four ways:

1. `ml.rs` took the first `f32` output from ONNX. For this binary XGBoost export, `probabilities` is ordered `[P(Dire), P(Radiant)]`, so the first value was the *negative* class. It also risked returning the `label` output.
2. `patch_version` was hardcoded to `7.41`. `patch` was not inserted at all. Both are categorical features encoded as integer codes during training, not raw version numbers.
3. Match-only defaults differed from training (`duration` was `2180` instead of the median `2163`).
4. Rust maintained its own feature map and silently inserted `0.0` for every missing model feature, including features the live draft cannot know.

The bundled model has 887 features. The first two features are `patch` and `duration`; `patch_version` is index 640.

## Model contract

The model assets are:

- `ml/models/draft_model_xgboost_full.onnx`
- `ml/models/draft_model_xgboost_full_features.json` (legacy ordered feature list)
- `ml/models/draft_model_xgboost_full_schema.json` (new runtime contract)
- `apps/desktop/src-tauri/resources/onnxruntime.dll`

The schema records:

- `feature_names`: exact 887-name order used by the model.
- `categoricals`: category order for `patch` and `patch_version`.
- `defaults`: one fallback value per feature, derived from training-data medians.
- `positive_class_index`: `1`, meaning `P(Radiant wins)`.
- `input_name`: `float_input`.
- `probability_output`: `probabilities`.

`patch` categories run from `7.05` through `7.41`; `7.41` is code `24`. `patch_version` categories are `7.40`, `7.41`, `unknown`; `7.41` is code `1`. The schema overrides the dataset median with the active patch for those two defaults.

### Generating the schema

```powershell
& ml\venv\Scripts\python.exe ml\training\export_model_schema.py
```

The generator reads the processed parquet and the features JSON, verifies every feature exists, reproduces the training categorical encoding, computes medians, and writes the schema. It does not retrain the model. Regenerate it whenever the model or feature set changes.

### Generating the golden vector

```powershell
& ml\venv\Scripts\python.exe ml\tests\fixtures\generate_golden.py
```

The fixture selects the first complete `patch=7.41`, `patch_version=7.41` row, encodes it exactly as training did, runs the bundled ONNX model, and records the label and both class probabilities. The current fixture is match `8741426861`, label `1`, positive probability `0.529628`.

## Rust changes

### Schema loading

`OnnxModel::load_inner` now:

1. Loads `<stem>_schema.json` when present.
2. Uses `schema.feature_names` as the model feature order.
3. Falls back to `<stem>_features.json` only for legacy bundles without a schema.
4. Validates that the schema feature order matches the order used for inference.

A schema with a different feature count or order fails loading instead of producing silently wrong predictions.

### Probability extraction

`OnnxModel::predict` reads the named `probabilities` output and selects `positive_class_index`. If the named output is absent, it scans `f32` outputs with the same positive index. A single-value output still falls back to its first value. The `label` output is `i64` and is not treated as a probability.

`select_probability` is covered by unit tests with `[0.425, 0.575]` and with a one-element vector.

### Feature encoding

`build_feature_map` accepts the loaded schema and:

- Encodes `patch` from `state.patch_version`, falling back to `meta.patch_version`.
- Encodes `patch_version` through the same category list.
- Uses the `unknown` category when a categorical value is not recognized and that category exists.
- Uses schema medians for `duration`, `avg_mmr`, `game_mode`, and `lobby_type`.
- Keeps the schema medians for any other model feature missing from the live map at inference time.

The hardcoded patch and duration values remain only as a fallback for legacy model bundles without a schema. No inference path contains a patch literal.

### Callers and packaging

- `commands.rs::predict_draft_value` passes the loaded schema into `build_feature_map` and returns the Radiant positive-class probability.
- `engine/scorer.rs` passes the same schema when simulating a pick.
- `tauri.conf.json` bundles the schema next to the ONNX model and features file.

The overlay calls `predictDraftValue('radiant')` and renders `dire: 1 - radiantProb`, so the UI contract remains Radiant-positive.

## Tests

Python, `ml/tests/test_onnx_contract.py`:

- Schema count, feature order, and positive-class metadata.
- Patch category codes and match defaults.
- Golden vector reproduces the ONNX label and probabilities within `1e-6`.

Rust, `ml::tests`:

- `probabilities[0][1]` selection and single-value fallback.
- Category encoding, `unknown` fallback, and schema defaults.
- Feature-order mismatch, missing feature, and extra feature rejection.
- Exported schema contains 887 features and the expected categorical codes.
- Golden fixture reproduces the bundled ONNX positive probability within `1e-5`.

## Verification

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
& ml\venv\Scripts\python.exe -m pytest ml/tests -q
```

Current results: 20 Rust library tests pass; 15 Python tests pass.

## Residual risks

- The schema is generated from the current processed dataset. A future retrain must regenerate it; Step 6 should make this part of the training export and add a model manifest.
- The 887-feature model still contains match-time and leakage-prone columns. Step 6 must decide which features are available at draft time and retrain without post-draft information.
- The schema medians reduce zero-fill bias but do not make unavailable features meaningful. They are a documented fallback, not a substitute for a live-feature model.
- A desktop UI smoke test remains after the next build; the frontend source contract is Radiant-positive.

