# ML Inference Contract

The bundled XGBoost model has 887 inputs. `ml/models/draft_model_xgboost_full_schema.json` is the runtime source of truth for feature order, categorical encodings, fallback values, and the positive class. Rust validates the schema at load time and uses `probabilities[0][1]` as `P(Radiant wins)`.

- `patch` and `patch_version` use training category codes, not raw version numbers.
- Missing live features fall back to schema medians, not silent `0.0`.
- A golden ONNX fixture verifies Rust against Python within `1e-5`.
- Legacy model bundles without a schema fall back to `features.json` and hardcoded defaults.

Canonical source: [`../../STEP_02_ML_INFERENCE.md`](../../STEP_02_ML_INFERENCE.md).

## Open follow-ups

- Regenerate the schema as part of the Step 6 training export.
- Remove post-draft and leakage-prone features in Step 6.
- Run a desktop UI smoke test after the next build.
