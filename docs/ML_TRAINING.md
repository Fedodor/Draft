# ML Training & AI Improvement Guide

This document explains how to improve the Smart Draft Assistant's AI by training it on real Dota 2 match data.

## Current Best Model (v2.0)

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | **0.6710** | All patches, wide XGBoost config |
| **Accuracy** | **62.5%** | |
| **Brier Score** | **0.2273** | Well-calibrated probabilities |
| **Training Samples** | **103,162** | Captain's Mode only |
| **Features** | **896** | Hero presence, semantics, winrates, synergies, matchups |
| **Data Sources** | OpenDota + STRATZ | 111K total matches collected |

### Final Summary

| Achievement | Value |
|-------------|-------|
| Best AUC | 0.6710 |
| Accuracy | 62.5% |
| Matches collected | 111,187 |
| Training samples | 103,162 |
| Features | 896 |
| Hero matchup pairs | 15,500 |
| GPU training time | ~5 min (5-fold CV) |

**Key insight:** Hero matchup features (`rad_matchup_avg`, `matchup_advantage`) provided the single biggest boost — from 0.56 to 0.67 AUC.

### Key Features (Top 10 by Gain)

1. `rad_matchup_avg` (85.6) — Radiant heroes' avg winrate vs Dire heroes
2. `matchup_advantage` (45.1) — Overall matchup differential
3. `duration` (40.8) — Game length
4. `rad_matchup_worst` (37.4) — Worst individual counter for Radiant
5. `winrate_diff` (37.1) — Raw hero winrate differential
6. `rad_matchup_best` (33.2) — Best individual matchup for Radiant
7. `rad_wr_dawnbreaker` (27.5) — Dawnbreaker winrate on Radiant
8. `rad_synergy_avg` (25.7) — Radiant team synergy score
9. `radiant_avg_winrate` (25.0) — Radiant avg hero winrate
10. `rad_has_bounty_hunter` (24.6) — Bounty Hunter presence

## Training Pipeline

### Prerequisites

```bash
cd ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Requirements: aiohttp pandas numpy xgboost scikit-learn pyarrow joblib onnx onnxmltools skl2onnx onnxruntime requests tqdm
```

### One-Command Pipeline (Recommended)

```bash
# Full pipeline: collect → process → train → evaluate → export ONNX
python training/train_pipeline.py \
  --matches 100000 \
  --patches 7.41 7.40 7.39 7.38 7.37 \
  --include-stratz \
  --stratz-key YOUR_STRATZ_KEY \
  --api-key YOUR_OPENDOTA_KEY \
  --force
```

This handles everything end-to-end:
1. **Collect** — OpenDota (pro + ranked) + STRATZ (tournament leagues + pro player matches)
2. **Process** — Feature engineering with hero matchup features
3. **Train** — XGBoost with GPU acceleration (device=cuda)
4. **Evaluate** — 5-fold CV with ROC-AUC, LogLoss, Brier, Calibration
5. **Export** — ONNX model for Tauri inference
6. **Report** — Comprehensive training report JSON

### Individual Steps

All scripts use `ml/datasets/` and `ml/models/` as defaults.

```bash
# 1. Collect data
python training/data_collector.py \
  --matches 100000 \
  --patches 7.41 7.40 7.39 7.38 7.37 \
  --include-pro \
  --include-ranked \
  --include-stratz \
  --min-mmr 4000 \
  --rate-limit 1.0 \
  --max-concurrent 3

# 2. Process features (adds matchup features)
python feature-engineering/processor.py

# 3. Train model (GPU XGBoost)
python training/trainer.py \
  --model-type xgboost \
  --train-frac 0.8 --val-frac 0.1 --test-size 0.1 \
  --time-split
```

> Splitting is chronological (`start_time` when present, otherwise `match_id`). Validation drives early stopping; the test set is evaluated once, after selection. `train_full.py` additionally writes a `*_manifest.json` recording the dataset hash, feature schema, hyperparameters, and validation/test metrics. See [`STEP_06_TRAINING.md`](STEP_06_TRAINING.md).

> The one-command `train_pipeline.py` path and `data_collector.py` were repaired in Step 6. `data_collector.py` is a normal multi-line module and `train_pipeline.py` runs under both `python -m training.train_pipeline` and `python training/train_pipeline.py`. `include-pro`/`include-ranked` accept `--no-` forms. See [`STEP_06_TRAINING.md`](STEP_06_TRAINING.md).

### GPU Training

```bash
# Verify CUDA
python -c "import xgboost as xgb; print(xgb.__version__); import torch; print(torch.cuda.is_available())"

# Train with GPU
python training/trainer.py --device cuda --tree-method hist
```

## Data Sources

| Source | Matches | Format | Access |
|--------|---------|--------|--------|
| OpenDota proMatches | ~6K | JSON (match detail) | API key optional |
| OpenDota publicMatches | ~5K | JSON (match detail) | API key optional |
| STRATZ Leagues | ~106K | GraphQL (full match) | **Bearer token required** |
| STRATZ Pro Players | ~1.4K | GraphQL (full match) | **Bearer token required** |

### STRATZ Setup

1. Get API key from https://stratz.com/api
2. Set as `--stratz-key` or in `ml/.env`:
   ```
   STRATZ_API_KEY=your_bearer_token
   ```

## Feature Engineering

The processor (`feature-engineering/processor.py`) creates:

| Category | Features | Description |
|--------|----------|-------------|
| Hero Presence | 278 | One-hot rad_has_*, dire_has_*, rad_ban_*, dire_ban_* |
| Semantics | 64 | Team avg init/stun/save/push/scaling/mobility/durability/teamfight/damage types |
| Winrates | 263 | Individual hero winrates + team avg/max/min + winrate_diff + ban winrates |
| Synergies | 6 | Team synergy (product of winrates) |
| **Matchups** | **7** | **Hero vs hero winrate differentials (NEW - biggest impact)** |
| Lane Assignment | 8 | rad_lane_safe/mid/off/other + dire_* (STRATZ `lane` field, player_slot fallback) |
| Meta | 8 | Patch, duration, game_mode, MMR, lobby |
| Pick Order | 10 | rad_pick_0-4, dire_pick_0-4 |

Total: **896 features** (dropped `patch`, `patch_version` categoricals)

> **Hero embeddings (experimental)**: `compute_embeddings.py` factors the 15.5K hero-matchup winrate matrix with TruncatedSVD into 126-dim per-hero vectors; `augment_embeddings.py` adds per-team average embeddings (`rad_emb_avg_*`/`dire_emb_avg_*`, +252 cols) → `draft_features_emb.parquet` (103,162 × 1148). A model trained on these scored time-split AUC 0.6351 vs 0.6424 for the 894-feature baseline, so they are retained as an experiment, not the bundled model.

> **Patch-aware**: the bundled full model includes `patch`/`patch_version` features; `trainer.py --per-patch` additionally trains a separate model per patch.

### Matchup Features (Critical)

Computed from 15,500 hero pairs with ≥20 games:
- `rad_matchup_avg` — Mean of `matchup[rad_hero][dire_hero]` for all 5 Radiant picks
- `dire_matchup_avg` — Same for Dire
- `matchup_advantage` — `rad_matchup_avg - dire_matchup_avg`
- `rad_matchup_best/worst` — Best/worst individual counter
- `dire_matchup_best/worst` — Same for Dire

## Model Training Details

### Best Hyperparameters (Wide Config)

```json
{
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "max_depth": 4,
  "learning_rate": 0.02,
  "subsample": 0.75,
  "colsample_bytree": 0.5,
  "min_child_weight": 30,
  "gamma": 0.5,
  "reg_alpha": 2.0,
  "reg_lambda": 5.0,
  "max_bin": 256,
  "device": "cuda",
  "tree_method": "hist",
  "verbosity": 0
}
```

### Training Performance (5-fold CV)

| Config | Samples | AUC | Acc | Brier | Time |
|--------|---------|-----|-----|-------|------|
| All patches / wide | 103,162 | **0.6710** | 62.5% | 0.2273 | 5 min |
| 7.37+ / wide | 89,233 | 0.6690 | 62.4% | 0.2276 | 4.5 min |
| 7.38+ / wide | 68,794 | 0.6645 | 62.3% | 0.2286 | 3.5 min |

### Patch Filtering

Best results with **all patches** (more data beats patch consistency). Filtering to recent patches (7.37+) gives similar AUC with less data.

## Improving Accuracy Further

### 1. More Data (Highest Impact)
- Collect **All Pick** matches for hero winrate baseline
- Add **Steam API** player data (MMR, behavior score)
- Expand STRATZ to **more leagues** (DPC qualifiers, regional leagues)

### 2. Better Features
- **Lane assignments** — Use STRATZ `lane` field from player data
- **Item timings** — Early game item progression from replay data
- **Hero embeddings** — 128-dim learned vectors replacing one-hot
- **Draft phase features** — Pick/ban order position, ban phase analysis

### 3. Model Architecture
- **Ensemble** — XGBoost + LightGBM + TabNet
- **Neural net** — TabTransformer for tabular data
- **Ranking loss** — Directly optimize pick ordering (LambdaMART)

### 4. Evaluation
- **Calibration** — Reliability diagrams, isotonic regression
- **Time-split validation** — Test on future patches only
- **Counterfactual** — "What if" draft simulations

## Deployment

### ONNX Export

```bash
python training/trainer.py --export-onnx --model-path ml/models/best_model.joblib
```

Output: `ml/models/draft_model_xgboost_full.onnx` plus its `_schema.json` and `_manifest.json` — ready for `ort` (ONNX Runtime) in Rust. The trainer emits this exact name; `ml/models/draft_model.onnx` is not a file this repository ships.

> ML is **opt-in at runtime**: `ml_enabled` defaults to `false` until a model clears the quality gate (ROC-AUC ≥ 0.60, Brier ≤ 0.235). The app works without a model.

### Rust Integration

Implemented in `apps/desktop/src-tauri/src/ml.rs`:

- `OnnxModel` wraps an `ort::session::Session` (runtime loaded from the bundled `onnxruntime.dll`, 1.28.0) plus the model's `_schema.json`.
- `build_feature_map(...)` reconstructs the live draft features (hero presence, bans, semantic aggregates, per-hero winrates, matchups, synergies) mirroring `processor.py`. Categorical `patch`/`patch_version` values use the training category codes; missing model features use schema medians.
- `predict_draft_value`, `load_ml_model`, and `get_ml_status` Tauri commands expose it to the UI. The returned value is `P(Radiant wins)` from `probabilities[0][1]`. If the model or runtime is unavailable, predictions degrade to neutral `0.5` without crashing.

The desktop engine's matchup and synergy seed is exported separately from the same processed datasets:

```bash
python training/export_scoring_seed.py
cargo run --manifest-path ../infrastructure/seeders/Cargo.toml --bin seed_matchups
```

Regenerate the runtime schema after any retrain:

```bash
python training/export_model_schema.py
```

Verify with the example (auto-uses the bundled runtime):

```bash
cd apps/desktop/src-tauri
cargo run --example inference_check   # → "Radiant win probability: 0.4867"
```

> **Model feature count**: the bundled `draft_model_xgboost_full.onnx` has **887 inputs**, with `patch` at index 0 and `patch_version` at index 640. `draft_model_xgboost_full_schema.json` is the runtime source of truth for feature order, category codes, and fallback values; see [`STEP_02_ML_INFERENCE.md`](STEP_02_ML_INFERENCE.md). Older exports used reduced subsets (600 / 863 inputs).

## File Locations

```
ml/
├── datasets/
│   ├── raw/                 # 111K JSON files
│   │   ├── *.json           # Individual match files
│   │   └── metadata.json    # Collection metadata
│   ├── processed/
│   │   ├── draft_features.parquet    # 103K × 896 features
│   │   ├── hero_matchups.parquet     # 15K hero pairs
│   │   └── hero_winrates.parquet     # 125 heroes
│   └── heroes.json                  # Hero roster (id → name)
├── training/
│   ├── train_pipeline.py     # Main pipeline
│   ├── data_collector.py     # OpenDota + STRATZ
│   ├── trainer.py            # XGBoost training
│   ├── collect_stratz.py     # STRATZ leagues
│   ├── collect_pro_matches.py # STRATZ pro players
│   └── compute_matchups.py   # Hero matchup computation
├── feature-engineering/
│   └── processor.py          # Feature engineering
├── models/
│   ├── draft_model_*.joblib
│   ├── draft_model_*.onnx
│   └── training_meta_*.json
└── experiments/              # Experiment tracking (optional)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenDota 429 rate limit | Add `--rate-limit 0.5`, use API key |
| STRATZ 401 unauthorized | Check bearer token in `.env` |
| GPU OOM | Reduce `max_bin`, use `device: cpu` |
| Low AUC | Add more data, check matchup features loaded |
| Slow processor | Use `--limit` for testing, check parallel I/O |

## Next Steps for ML Team

### Product Next Steps (shared with README)
1. [x] **Complete GSI integration** — Real-time draft detection in the Tauri overlay (idempotent server, stop/status commands, one-click config install)
2. [x] **ONNX inference in Rust** — Wire up `ml.rs` with ONNX Runtime; reconstructs the 887-feature vector from the schema, with neutral fallback (see `ml.rs` + `load_ml_model`/`get_ml_status` commands)
3. [x] **Tree search engine** — Beam-search minimax lookahead over opponent responses (`engine/simulation.rs` + `get_search_recommendations`)
4. [ ] **More data collection** — Expand STRATZ to more leagues, add Steam API player context

### ML Next Steps
5. [~] Collect All Pick matches for better hero baseline winrates — collectors capture `lane`; 8.5K All Pick on disk, expand with `collect_stratz_allpick.py --target <N>`
6. [~] Add lane/role features from STRATZ player data — `processor.py` extracts `rad_lane_*`/`dire_lane_*` (needs lane-enriched raw data for full effect)
7. [ ] Train LightGBM as ensemble partner
8. [~] Implement patch-aware model — `trainer.py --per-patch`; bundled model includes patch features
9. [ ] Build "what-if" counterfactual draft simulator