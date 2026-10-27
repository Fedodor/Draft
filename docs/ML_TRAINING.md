# ML Training & Model Quality

This document describes the ML pipeline, the artifact that actually ships, and
the checks a model must pass before it can replace that artifact. It replaces
the earlier "0.6710 AUC / 896 features" summary, which described an experiment
that is **not** the bundled model and cannot be reproduced from the tracked
artifacts.

- Bundled model: `ml/models/draft_model_xgboost_full.onnx`
- Training data: `ml/datasets/processed/draft_features.parquet`
- Runtime: ONNX Runtime in `apps/desktop/src-tauri/src/ml.rs`
- ML is **opt-in** (`ml_enabled = false` by default); the app works without it.

## 1. Bundled model (measured, 2026-10-28)

| Metric | `as_trained` | `inference` (what the runtime can do) |
| --- | --- | --- |
| ROC-AUC | 0.5388 | **0.5122** |
| 95% CI | 0.5277-0.5500 | 0.5010-0.5234 |
| Brier score | 0.2481 | **0.2495** |
| Log loss | 0.6893 | 0.6922 |
| Accuracy @ 0.5 | 0.5294 | 0.5125 |
| ECE (10 equal-frequency bins) | 0.0124 | 0.0241 |

Reference on the same 10,213 held-out rows: a constant train-prior predictor
(p = 0.5245) scores Brier **0.2493**, log loss 0.6917. The model is slightly
**worse** than that baseline on Brier and barely above chance on AUC. Do not
describe it as accurate, calibrated-to-a-useful-degree, or production-ready.

Dataset and split:

| Item | Value |
| --- | --- |
| Rows | 102,118 |
| Columns | 890 (887 model inputs + `match_id`, `radiant_win`, `start_time`) |
| Class balance | 53,693 Radiant wins / 48,425 Dire wins |
| Dataset SHA-256 | `264a6ad5a0cfdecfb60797212d7878019db26a87cb32a4c836de2b27ce991621` |
| Patch | 7.41 (training export) |
| Split | chronological 80/10/10 on `start_time` |
| Train / validation / test | 81,694 / 10,211 / 10,213 matches |
| Match-id overlap | 0 between all partitions |

`as_trained` scores the logged match rows. `inference` replaces the fields the
live client cannot know with the schema default (training median):
`patch`, `patch_version`, `duration`, `avg_mmr`, `game_mode`, `lobby_type`.
The 0.0266 AUC gap between the two modes is the measured train/serve skew.

## 2. Quality gate

`ml/training/quality_gate.py` is the promotion gate. It never trains and never
rewrites the model or schema.

```bash
# Default: validate + evaluate in runtime-realistic mode
ml/venv/Scripts/python.exe ml/training/quality_gate.py

# Machine-readable report / custom path
ml/venv/Scripts/python.exe ml/training/quality_gate.py --json
ml/venv/Scripts/python.exe ml/training/quality_gate.py --report ml/models/gate.json

# Evaluate against logged values (comparison only; not the promotion default)
ml/venv/Scripts/python.exe ml/training/quality_gate.py --evaluation-mode as_trained

# Copy the bundle only when every mandatory check passes
ml/venv/Scripts/python.exe ml/training/quality_gate.py --promote-to ml/models/promoted
```

Default policy (applied to `inference` mode):

| Check | Threshold |
| --- | --- |
| ROC-AUC | ≥ 0.60 |
| Brier score | ≤ 0.235 |
| ECE | ≤ 0.10 |
| Test rows | ≥ 500 |

Before evaluating, the gate verifies:

- manifest, schema, feature list, and ONNX file exist and agree on feature
  count, order, and schema fingerprint;
- ONNX input is a 2-D `tensor(float)` with the declared feature width, and the
  declared positive-class/probability outputs exist;
- the dataset SHA-256 and row count match the manifest;
- the reconstructed chronological split matches the manifest, has strict time
  boundaries, no duplicated `match_id`, and no `match_id` overlap between
  partitions;
- labels are binary with both classes present;
- predictions are finite probabilities in `[0,1]`;
- every schema feature is encodable (diagnostics: missing columns are reported,
  NaN fills are counted, unknown categories are counted).

Exit codes:

| Code | Meaning | Promotion |
| --- | --- | --- |
| `0` | Passed; all mandatory checks green | Allowed |
| `1` | Quality failure (thresholds, fingerprint mismatch, split problem) | Refused |
| `2` | Evaluation error (missing/malformed artifact, schema drift, bad labels) | Refused |

The report (`<stem>_quality_report.json`) records both modes, the calibration
bins, constant baselines, split boundaries, encoding diagnostics, the leakage
audit, and the failed checks. `ml/models/*` is gitignored, so reports and
candidate artifacts stay local.

The bundled artifact currently exits **1**: `policy.roc_auc` and
`policy.brier_score` fail. That is intentional and visible, not a bug.

## 3. Feature contract

The model has **887 inputs**. The current processed parquet has no matchup
family; the processor would add seven matchup features if matchup data were
present, producing **894** names. The bundled parquet has no matchup columns,
so shipping them requires a regenerated dataset, a retrain, and a new export.

| Block | Count | Notes |
| --- | --- | --- |
| Hero presence (`rad_has_*`, `dire_has_*`) | 278 | 139 per side; ids `1..139` |
| Hero bans (`rad_ban_*`, `dire_ban_*`) | 279 | Includes legacy `hero_*` names for ids absent from the roster |
| Per-hero win rates (`rad_wr_*`, `dire_wr_*`) | 254 | 127 per side |
| Win-rate aggregates | 8 | Team avg/max/min, `winrate_diff`, ban averages |
| Synergy | 6 | Rad/Dire avg/max/min |
| Semantic team averages | 34 | 17 attributes × 2 sides |
| Semantic differentials | 10 | `stun_diff`, `initiation_diff`, ... |
| Damage diversity | 2 | `radiant_*`, `dire_*` |
| Match/meta | 6 | `patch`, `patch_version`, `duration`, `avg_mmr`, `game_mode`, `lobby_type` |
| Pick order | 10 | `rad_pick_0..4`, `dire_pick_0..4` |
| **Total** | **887** | |

Feature order is part of the contract. `draft_model_xgboost_full_schema.json`
is the runtime source of truth for order, categorical codes, and defaults.

Reproducibility notes (pinned by `ml/tests/test_feature_schema.py`):

- The real `FeatureProcessor` reproduces the exact 887-name set on a fixture
  with no matchup data.
- 631 structural features keep a code-defined order. The 254 per-hero win-rate
  columns and the two `*_ban_avg_winrate` columns are ordered by first
  appearance in the shipped dataset, not by code; the test pins their
  membership, not their position.
- With matchup data the processor emits exactly the 887 names plus the seven
  `*_matchup_*` / `matchup_advantage` features.
- `process_all()` collects thread results in completion order, so a regenerated
  dataset can have a different column order. Canonicalising that order is open
  work (P1-5).

### Historical experiment (not the bundled model)

Earlier notes in this repository reported **0.6710 ROC-AUC / 62.5% accuracy /
0.2273 Brier / 896 features / 103,162 samples**, with matchup features
described as the largest gain. Those numbers belong to a different dataset and
feature set (matchup-enriched, five-fold CV) and are **not reproducible from
the tracked 887-input artifacts**. They are retained here only as historical
context. Do not quote them as the current model.

## 4. Training pipeline

Prerequisites:

```bash
cd ml
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

### Reference training run

`train_full.py` is the script that produced the bundled artifact. It uses the
chronological split, early stopping on validation, and touches the test
partition once.

```bash
ml/venv/Scripts/python.exe ml/training/train_full.py
```

Re-evaluate without retraining:

```bash
ml/venv/Scripts/python.exe ml/training/train_full.py --evaluate-only
```

Export the runtime schema after a retrain (order, categorical codes, defaults):

```bash
ml/venv/Scripts/python.exe ml/training/export_model_schema.py
```

Then run the gate. A candidate may replace the bundled model only after:

1. `quality_gate.py` exits `0` in `inference` mode;
2. the golden ONNX fixture and schema tests still pass;
3. `scripts/verify.ps1` passes;
4. the change is reviewed as a data/model change, not a docs-only edit.

### One-command collection pipeline

`train_pipeline.py` collects and processes data before training:

```bash
python training/train_pipeline.py \
  --matches 100000 \
  --patches 7.41 7.40 7.39 7.38 7.37 \
  --include-stratz \
  --stratz-key YOUR_STRATZ_KEY \
  --api-key YOUR_OPENDOTA_KEY \
  --force
```

This is expensive (network + the full 1.9 GB dataset) and is not part of CI.

### GPU training

```bash
python -c "import xgboost as xgb; print(xgb.__version__)"
python training/trainer.py --device cuda --tree-method hist
```

The bundled `train_full.py` defaults to CPU unless configured otherwise; the
measured wall time in this environment was seconds for evaluation, not a
training benchmark.

## 5. Data sources

| Source | Scale | Format | Access |
| --- | --- | --- | --- |
| OpenDota proMatches | ~6K | JSON match detail | API key optional |
| OpenDota publicMatches | ~5K | JSON match detail | API key optional |
| STRATZ leagues | ~106K | GraphQL full match | Bearer token required |
| STRATZ pro players | ~1.4K | GraphQL full match | Bearer token required |
| All Pick | ~9.8K on disk | JSON | Win-rate/matchup baselines |

STRATZ setup:

1. Get a key at https://stratz.com/api.
2. Set `STRATZ_API_KEY` in `ml/.env` or pass `--stratz-key`.
3. Never commit `ml/.env`; the repository ignores it and runs a secret scan.

## 6. Improvement directions

Ranked by expected value, not novelty:

1. **Remove leakage and retrain (P1-6).** Drop `duration`, `avg_mmr`,
   `game_mode`, `lobby_type`, and raw pick-index columns, then evaluate the
   candidate in `inference` mode. The current model loses 0.0266 AUC purely
   from runtime substitution, and its post-draft fields are not available when
   a recommendation is made.
2. **Beat the constant baseline first.** A candidate that cannot beat Brier
   0.2493 and AUC 0.5 on the chronological holdout is not worth integrating,
   regardless of feature count.
3. **Matchup data at training time.** The processor can emit the seven matchup
   features, but the bundled dataset does not contain them. Regenerate with
   matchup data and retrain; do not bolt them onto the existing schema without
   training.
4. **Lane/role features.** The processor has lane extraction paths, but lane
   features are disabled in the bundled contract and need lane-enriched raw
   data.
5. **Calibration.** ECE is low (0.024 at inference), but that is partly because
   the predictions are close to the base rate. Re-check calibration after any
   model improves discrimination.
6. **Ensembles/embeddings.** Historical experiments did not beat the baseline
   they were compared against; treat them as experiments until a gated
   candidate shows a real gain.

## 7. Runtime integration

- `apps/desktop/src-tauri/src/ml.rs` loads the ONNX model and
  `_schema.json`, builds the feature map from the live draft, and reads
  `probabilities[class=1]` (`P(Radiant wins)`).
- Missing model/runtime or failed inference returns neutral `0.5`; the
  deterministic engine keeps working.
- Feature families the runtime cannot reconstruct (`rad_lane_*`,
  `dire_lane_*`, `emb_*`) are excluded by `train_full.py`.
- Regenerate the schema after any retrain; the Rust loader validates the
  feature list against the schema and refuses a mismatched bundle.

Useful commands:

```bash
cd apps/desktop/src-tauri
cargo run --example inference_check   # loads the bundled artifact
```

## 8. File locations

```
ml/
├── datasets/
│   ├── raw/                       # collected match JSON (not tracked)
│   ├── processed/
│   │   ├── draft_features.parquet # 102,118 × 890 (bundled training data)
│   │   ├── hero_matchups.parquet
│   │   └── hero_winrates.parquet
│   └── heroes.json                # tracked roster (127 entries)
├── feature-engineering/
│   └── processor.py               # raw match JSON -> features
├── training/
│   ├── train_full.py              # bundled-model training/evaluation/export
│   ├── export_model_schema.py     # runtime schema export
│   ├── quality_gate.py            # promotion gate (new)
│   ├── inference_contract.py      # schema-order encoding (new)
│   ├── splits.py                  # chronological split + fingerprints
│   ├── manifest.py                # manifest construction
│   ├── train_pipeline.py          # collect -> process -> train
│   └── data_collector.py          # OpenDota + STRATZ
├── tests/
│   ├── test_feature_schema.py     # processor -> 887-name contract (new)
│   ├── test_quality_gate.py       # gate/promotion behaviour (new)
│   ├── test_onnx_contract.py      # golden vector + schema
│   └── fixtures/                  # golden feature vector
└── models/                        # gitignored except the 4 bundled artifacts
```

## 9. Troubleshooting

| Issue | Solution |
| --- | --- |
| OpenDota 429 | Lower `--rate-limit`, add an API key |
| STRATZ 401 | Check the bearer token in `ml/.env` |
| Gate exits 2 | Read `evaluation.error` in the report: artifact, fingerprint, split, labels, or ONNX contract |
| Gate exits 1 | The model failed a threshold; improve/retrain it. Do not lower the policy to make it pass |
| `Duration`/`avg_mmr` warnings | Those fields are post-draft/constant; the runtime substitutes schema defaults |
| XGBoost UBJSON warning | Pre-existing artifact format warning; harmless |
| Low AUC | Check the split and baseline first, then features; adding columns rarely fixes a near-baseline model |

## 10. Next steps

1. Train the leakage-free candidate and run it through the gate.
2. Make the processor output order deterministic before regenerating data.
3. Regenerate with matchup data only as part of a gated retrain.
4. Keep ML opt-in until a candidate passes the gate and a live-session review.
