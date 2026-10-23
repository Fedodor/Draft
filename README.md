# Smart Draft Assistant

Production-quality desktop AI assistant for Dota 2 draft analysis.

## Features

- **Real-time Draft Analysis**: Auto-detects live drafts via Dota 2 Game State Integration (GSI)
- **Intelligent Recommendations**: Hybrid deterministic + ML scoring system
- **Explainable AI**: Every recommendation includes detailed reasoning
- **Desktop Overlay**: Low-latency, always-on-top companion window
- **Personalized**: Adapts to your hero pool and playstyle
- **Meta-Aware**: Tracks patch changes and pro meta trends

## Documentation

- [Technical audit](docs/AUDIT.md): severity-ranked findings across the app, ML pipeline, and data layer.
- [Remediation plan](docs/REMEDIATION_PLAN.md): the 8-step fix order, dependencies, and acceptance gates.
- [Step 1 design: Secrets and environment](docs/STEP_01_SECRETS_AND_ENV.md): credential rotation, env contract, and repository guard.
- [Step 2 design: ML inference correctness](docs/STEP_02_ML_INFERENCE.md): feature schema, positive-class probability, and golden-vector tests.
- [Step 3 design: Scoring correctness](docs/STEP_03_SCORING.md): startup scorer wiring, role normalization, real seed data, and score breakdowns.
- [Step 4 design: Tauri IPC and manual drafting](docs/STEP_04_IPC.md): snake_case command args, typed errors, validation, and undo.
- [Step 8 design: CI and quality gates](docs/STEP_08_CI.md): the single `verify.ps1` gate, the GitHub Actions workflow, and merge requirements.
- [Project wiki](docs/wiki/index.md): persistent summaries, security runbooks, and cross-references.

Reference:

- [Architecture](docs/ARCHITECTURE.md)
- [Draft engine](docs/ENGINE.md)
- [ML training](docs/ML_TRAINING.md)
- [GSI setup](docs/GSI.md)
- [Development phases](docs/DEVPHASES.md)

Some README metrics and paths are stale until the remediation steps land; the audit is the source of truth for known issues.

## Architecture

```
┌─────────────┐     GSI      ┌──────────────┐     ┌─────────────┐
│  Dota 2    │ ──────────▶ │  Rust Backend │ ───▶ │  React UI   │
│  Client    │  HTTP/JSON   │  (Tauri)      │  IPC │  Overlay    │
└─────────────┘              └──────────────┘     └─────────────┘
                                     │
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                    ┌────────┐ ┌──────────┐ ┌─────────┐
                    │ Draft  │ │   GSI    │ │   ML    │
                    │ Engine │ │ Server   │ │Inference│
                    └────────┘ └──────────┘ └─────────┘
```

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + Zustand
- **Backend**: Rust + Tauri v2 + Warp (HTTP) + Parking Lot (concurrency)
- **ML**: Python (training) → XGBoost/LightGBM → ONNX Runtime (inference)
- **Data**: SQLite + Parquet datasets
- **APIs**: OpenDota, STRATZ, Steam API

## Quick Start

### Prerequisites

- Node.js 20+
- Rust 1.75+
- Python 3.11+ (for ML training)
- Dota 2 installed

### Installation

```bash
# Clone and install frontend
cd apps/desktop
npm install

# Install Rust dependencies (auto on first build)
# Install Python ML dependencies
cd ../../ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### ML API Keys

Copy `ml/.env.example` to `ml/.env` and set the keys locally:

- `STRATZ_API_KEY` is required by STRATZ collectors.
- `OPENDOTA_API_KEY` is optional and enables higher OpenDota rate limits.

Never commit `ml/.env`. The repository ignores it and includes a working-tree secret scan in the test suite.

### Development

```bash
# Terminal 1: Frontend dev server
cd apps/desktop
npm run dev

# Terminal 2: Tauri dev (auto-reloads Rust)
cd apps/desktop
npm run tauri dev
```

### Database Seeding

The app stores hero data in a local SQLite database (`smart_draft.db`). Before first run, seed the database with the hero roster, matchups, and synergies:

```bash
# Build the seeders (one-time)
cargo build --release --manifest-path infrastructure/seeders/Cargo.toml

# Run from the project root (uses relative paths to JSON files)
RUST_LOG=info infrastructure/seeders/target/release/seed_heroes.exe
RUST_LOG=info infrastructure/seeders/target/release/seed_matchups.exe
```

This populates the database with:
- **126 heroes** — full Dota 2 roster (Str, Agi, Int, and Universal/All heroes) with base stats, attack types, roles, and image paths
- **126 hero semantics** — initiation, stun, save, push, scaling, mobility, durability, teamfight, splitpush, vision, damage types, lane strength, tempo profiles, and role probabilities
- **9,052 matchups + 7,381 synergies** — real data from `infrastructure/seeders/scoring_seed.json`, tagged with the active patch (`7.41`)

`seed_matchups` writes real data by default. Pass `--synthetic` only for UI work when the processed datasets are unavailable; synthetic rows are tagged with patch `synthetic` and never selected silently.

The database is created at (`paths::app_data_dir`, matching the Tauri `identifier`):
- **Windows**: `%APPDATA%\com.smartdraftassistant.app\smart_draft.db`
- **macOS**: `~/Library/Application Support/com.smartdraftassistant.app/smart_draft.db`
- **Linux**: `~/.local/share/com.smartdraftassistant.app/smart_draft.db`

Both the app and the seeders resolve this path through `apps/desktop/src-tauri/src/paths.rs`.

To regenerate hero data, edit `infrastructure/seeders/heroes.json` and `hero_semantics.json`, then re-run the seeders (they use `INSERT ... ON CONFLICT ... DO UPDATE`).

> The seeder depends on the main Tauri crate (`smart-draft-assistant`) and requires the `database` module to be `pub`. See `infrastructure/seeders/Cargo.toml` for dependencies.

### GSI Setup

1. Copy `gamestate_integration_smart_draft.cfg` to Dota 2 config:
   ```
   %APPDATA%\dota 2 beta\game\dota\cfg\gamestate_integration\
   ```
2. Restart Dota 2
3. Start a lobby with bots to test

## ML Training Pipeline Results (Current State)

The project includes a fully automated end-to-end ML training pipeline. **Current best model**:

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.6710** (historical) / 0.543 (current 889-feature retrain) |
| **Accuracy** | **62.5%** / 53.4% |
| **Brier Score** | **0.2273** / 0.2475 |
| **Training Samples** | **102,118** |
| **Features** | **889** |

### Final Summary

| Achievement | Value |
|-------------|-------|
| Best AUC | 0.6710 (historical, 103K/896, 5-fold CV) |
| Accuracy | 62.5% |
| Matches collected | 111,945 (+756 new, lane capture enabled) |
| Training samples | 102,118 (refreshed, 890 cols incl. `start_time`; 887 model inputs) |
| Features | 889 (lane infra + 126-dim SVD embeddings experiment) |
| Hero matchup pairs | 6,303 (current, threshold 10; 15,500 was lower threshold on larger allpick) |
| AllPick matches | 9,790 (+1,156 new) for winrates/matchups |
| GPU training time | ~5 min (5-fold CV) |

**Key insight:** Hero matchup features (`rad_matchup_avg`, `matchup_advantage`) provided the single biggest boost — from 0.56 to 0.67 AUC.

### Key Breakthrough: Hero Matchup Features

The most impactful features are hero-vs-hero counter data (15,500 computed pairs):
- `rad_matchup_avg` (gain: 85.6) — Average winrate of Radiant picks vs Dire picks
- `matchup_advantage` (gain: 45.1) — Overall matchup differential
- `rad_matchup_worst` (gain: 37.4) — Worst individual matchup for Radiant

### Dataset

- **111,945 CM matches** collected (106K Stratz leagues + 5K OpenDota pro + 1.4K pro player + 756 new) + **9,790 AllPick** for winrates/matchups
- **102,118 training samples** with full pick/ban data (refreshed)
- **Patches**: 6.88 → 7.41 (all major patches)
- **Sources**: Stratz GraphQL API (tournament leagues + pro players) + OpenDota proMatches

### One-Command Training

```bash
cd ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline (collect → process → train → export)
python training/train_pipeline.py --matches 100000 --patches 7.41 7.40 7.39 7.38 7.37
```

This single command will:
1. **Collect** matches from OpenDota (pro + ranked) + STRATZ (leagues + pro players)
2. **Process** raw matches into ML-ready features (Parquet)
3. **Train** XGBoost model with time-based validation
4. **Evaluate** with ROC-AUC, LogLoss, Brier Score, Calibration
5. **Export** to ONNX for fast inference
6. **Generate** comprehensive training report

### Pipeline Options

```bash
# Full options (path flags optional — defaults resolve to ml/datasets & ml/models)
python training/train_pipeline.py \
  --matches 100000 \                    # Target matches to collect
  --patches 7.41 7.40 7.39 7.38 7.37 \  # Patches to collect from
  --include-pro \                        # Include professional matches (default)
  --include-ranked \                     # Include ranked public matches (default)
  --include-stratz \                     # Include STRATZ tournament/pro data (default)
  --min-mmr 4000 \                       # Min MMR for ranked matches
  --rate-limit 1.0 \                     # API requests per second
  --api-key YOUR_KEY \                   # OpenDota API key (optional, higher limits)
  --stratz-key YOUR_KEY \                # STRATZ API key (required for STRATZ)
  --time-split \                         # Use time-based split (default)
  --test-size 0.2 \                      # Test split ratio
  --force                                # Continue even if min matches not met
```

### Individual Steps

All scripts default to `ml/datasets/` and `ml/models/` regardless of the working directory.

```bash
# 1. Collect data (OpenDota + STRATZ)
python training/data_collector.py \
  --matches 100000 \
  --patches 7.41 7.40 7.39 7.38 7.37 \
  --include-pro \
  --include-ranked \
  --include-stratz \
  --min-mmr 4000 \
  --rate-limit 1.0 \
  --max-concurrent 3

# 2. Process features (with matchup features)
python feature-engineering/processor.py

# 3. Train model (GPU-accelerated XGBoost)
python training/trainer.py \
  --model-type xgboost \
  --test-size 0.2 \
  --time-split \
  --device cuda
```

### Model Configuration

Custom configs via JSON:

```json
// ml/training/custom_config.json
{
  "n_estimators": 3000,
  "max_depth": 4,
  "learning_rate": 0.02,
  "subsample": 0.75,
  "colsample_bytree": 0.6,
  "min_child_weight": 30,
  "gamma": 0.5,
  "reg_alpha": 2.0,
  "reg_lambda": 5.0,
  "early_stopping_rounds": 200,
  "device": "cuda"
}
```

```bash
python training/trainer.py --config ml/training/custom_config.json
```

### Output Artifacts

After training, `ml/models/` contains:
- `draft_model_<timestamp>.joblib` — Primary model (joblib)
- `draft_model_<timestamp>.onnx` — ONNX export for inference
- `draft_model_<timestamp>_features.json` — Feature names
- `draft_model_<timestamp>_config.json` — Training config
- `draft_model_<timestamp>_history.json` — Training curves
- `draft_model_<timestamp>_report.json` — Full evaluation report

### Runtime ONNX Inference

The Tauri backend (`ml.rs`) loads the ONNX model via ONNX Runtime (`ort`), reconstructs the feature vector from the live draft state, and exposes `predict_draft_value`, `load_ml_model`, and `get_ml_status` commands. Verified end-to-end with the bundled `onnxruntime.dll` (1.28.0, `src-tauri/resources/`), producing real win-probability predictions. The app auto-loads the bundled model at startup; if the model or runtime is unavailable it degrades to neutral predictions without crashing.

> The bundled runtime is pinned to ONNX Runtime **1.28.0** while `ort` is **2.0.0-rc.13** (targets 1.27.x). This pairing is intentional: `ort` tolerates a *newer* runtime and only logs an informational notice. Do not bump `ort` alone — rc.14+ targets ONNX Runtime 1.31.x and requires replacing the DLL too (see `Cargo.toml`).

> The bundled model is `draft_model_xgboost_full.onnx` — trained on `ml/datasets/processed/draft_features.parquet` (889 columns, minus `match_id` and `radiant_win` → **887 inputs**). The feature builder (`build_feature_map` in `ml.rs`) fills everything computable from the live draft and defaults match-time fields (`duration`, `avg_mmr`, `game_mode`, `lobby_type`) to their training medians. Feature families the runtime cannot reconstruct (`rad_lane_*`, `dire_lane_*`, `emb_*`) are excluded by `train_full.py` so the exported model always matches this contract.

## Project Structure

```
smart-draft-assistant/
├── apps/desktop/           # Tauri + React app
│   ├── src/                # React frontend
│   │   ├── components/     # UI components
│   │   ├── stores/         # Zustand state
│   │   ├── services/       # API layer
│   │   └── types/          # TypeScript types
│   └── src-tauri/          # Rust backend
│       ├── src/
│       │   ├── engine/     # Draft engine & scorers
│       │   ├── gsi/        # Game State Integration
│       │   ├── commands/   # Tauri IPC commands
│       │   ├── models/     # Data models
│       │   ├── database/   # SQLite persistence
│       │   ├── simulation/ # Draft tree search
│       │   └── ml/         # ONNX inference
├── packages/shared-types/  # Shared TS/Rust types
├── ml/                     # ML pipeline
│   ├── datasets/
│   │   ├── raw/           # Original API responses (111K files)
│   │   ├── processed/     # Normalized features (parquet)
│   │   └── features/      # Precomputed ML features
│   ├── training/          # Training pipeline scripts
│   │   ├── train_pipeline.py    # Main automated pipeline
│   │   ├── data_collector.py    # OpenDota + STRATZ data collection
│   │   ├── trainer.py           # Model training
│   │   ├── collect_stratz.py    # STRATZ league collector
│   │   ├── collect_pro_matches.py # STRATZ pro player collector
│   │   └── compute_matchups.py  # Hero matchup computation
│   ├── feature-engineering/     # Feature engineering
│   │   └── processor.py
│   ├── models/            # Trained models (joblib + ONNX)
│   └── experiments/       # Experiment tracking
├── infrastructure/        # Scripts, migrations, seeders
│   └── seeders/           # Hero data seeders
│       ├── seed_heroes.rs        # Rust binary — loads JSON into SQLite
│       ├── generate_heroes.py    # Python — fetches OpenDota API, generates JSON
│       ├── heroes.json           # Full hero roster (126 heroes)
│       ├── hero_semantics.json   # Hero semantic attributes
│       └── scoring_seed.json     # Real data-derived matchups and synergies
├── docs/                  # Audit, remediation plan, and wiki
│   ├── AUDIT.md
│   ├── REMEDIATION_PLAN.md
│   ├── STEP_01_SECRETS_AND_ENV.md
│   ├── STEP_02_ML_INFERENCE.md
│   ├── STEP_03_SCORING.md
│   ├── STEP_04_IPC.md
│   └── wiki/
└── .opencode/rules/       # AI agent rules
```

## Configuration

Settings stored in Tauri store (`config.json`):
- GSI port (default: 3000)
- Overlay position/size
- ML model path
- Auto-update meta

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Foundation: Tauri, React, Draft State, Manual UI |
| 2 | ✅ | Core Intelligence: Matchup, Synergy, Composition Scorers |
| 3 | ✅ | Live Draft: GSI, Overlay, Lane capture |
| 4 | ✅ | ML Infrastructure: Dataset (111,790), Features (889 + lane/emb), Training, ONNX |
| 5 | ✅ | Advanced: Tree search, Embeddings, Lane features, Patch-aware |
| 6 | ⏳ | AI Coach: Replay Analysis, Decision Review |

## Next Steps & Improvements

### Completed
1. ✅ **GSI Integration** — Real-time draft detection via Dota 2 Game State Integration. GSI server (warp) with idempotent start, graceful stop, status endpoint, and one-click config install into Dota 2's `cfg/gamestate_integration/`.
2. ✅ **ONNX Inference in Rust** — `ml.rs` loads the exported XGBoost ONNX model via ONNX Runtime (`ort`), reconstructs the full feature vector (896 columns) from the live draft state, and exposes `predict_draft_value`, `load_ml_model`, and `get_ml_status` commands. Verified working end-to-end with a bundled `onnxruntime.dll` (1.28.0); graceful neutral fallback when the model/runtime is unavailable.
3. ✅ **Tree Search Engine** — Beam-search minimax lookahead over opponent responses (`engine/simulation.rs`), exposed via `get_search_recommendations`.
4. ✅ **Live Overlay UI** — Compact always-on-top, transparent overlay window showing live win probability, team hero icons, and top recommendations with reasoning. Toggle from the main window header (`Overlay` button).

### High Priority
4. ✅ **Live Overlay UI** — React overlay with hero icons, win probabilities, reasoning

### Medium Priority
5. 🚧 **Collect More Data** — STRATZ/OpenDota collectors now capture `lane` data (All Pick discovery verified; 8.5K All Pick matches on disk). Expand coverage by running `collect_stratz_allpick.py --target <N>` (rate-limit aware) over time.
6. 🚧 **Hero Embeddings** — 126-dim SVD embeddings computed from the matchup matrix (`feature-engineering/compute_embeddings.py`) and added as team-average features (`draft_features_emb.parquet`, +252 cols). A retrained model (`draft_model_xgboost_emb`, 1146 inputs) scored AUC 0.6351 on the time-split — below the 894-feature baseline (0.6424), so embeddings are kept as an experiment rather than the bundled model.
7. 🚧 **Lane Assignment Features** — `processor.py` now extracts lanes (STRATZ `lane` field preferred, `player_slot` heuristic fallback) into `rad_lane_*` / `dire_lane_*` counts. Collectors capture real lane data for future training sets.
8. 🚧 **Patch-Aware Model** — `trainer.py --per-patch` trains a separate model per patch; the bundled full model also includes `patch`/`patch_version` features.

### Low Priority
9. **Replay Analysis** — Post-game review with "what if" draft simulation
10. **Personalization** — Learn from user's own matches (hero pool, playstyle)
11. **Ensemble Models** — Combine XGBoost + LightGBM + Neural Net for robustness

> The exported full model (`ml/models/draft_model_xgboost_full.*`) trains on `draft_features.parquet` (889 columns) minus `match_id`, `radiant_win`, and the runtime-unsupported families → **887 model inputs**. It is the model bundled with the app. `python ml/training/train_full.py --evaluate-only` reproduces this contract.

## Contributing

1. Follow the [AI Agent Rules](.opencode/rules/smart-draft-assistant.md)
2. Run the quality gate before committing: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1`
3. Use the [Karpathy LLM Wiki method](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) for research
4. Keep ML as enhancement layer — core logic must work without it
5. Every recommendation must be explainable

## License

MIT