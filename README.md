# Smart Draft Assistant

A local-first desktop companion for Dota 2 drafting. It reads the live draft
through Game State Integration (GSI), scores heroes with a deterministic
engine, explains every recommendation, and can optionally load an experimental
ONNX model.

> **Status: working desktop tool, experimental ML.** The deterministic engine,
> GSI ingestion, UI, and search are usable. The bundled ML model is **not**
> trustworthy: it fails the quality gate and is disabled by default. See
> [Model quality](#model-quality) before enabling it.

## What works

| Area | Status | Notes |
| --- | --- | --- |
| Live draft detection | Working | GSI server, lifecycle reset, validation, dedupe, undo suppression |
| Deterministic scoring | Working | Matchup, synergy, composition, meta, and player-pool signals |
| Explanations | Working | Per-recommendation reasoning and score breakdown |
| Overlay | Working | Always-on-top companion window; polling pauses when GSI is quiet |
| Beam-search lookahead | Working | Bounded minimax over opponent responses |
| Patch meta | Working | Patch-scoped meta snapshot from seeded data |
| ML scoring | **Experimental, opt-in** | 887-input ONNX model; fails the promotion gate |
| Live GSI replay | Not verified | Payload-level tests exist; no captured real session was replayed in CI |
| Tauri runtime smoke test | Not verified | CSP/capabilities and overlay behaviour are type-checked, not launched in CI |

## Quick start

### Prerequisites

- Windows 10/11 (the verified platform)
- Node.js 20+
- Rust stable (with `rustfmt` and `clippy`)
- Python 3.10+ (ML tooling; 3.10.11 was used for verification)
- Dota 2, for live drafting

### Install

```bash
cd apps/desktop
npm install

cd ../../ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Rust dependencies build automatically on first `cargo`/Tauri invocation.

### Seed the database

The app reads heroes, semantics, matchups, synergies, and patch meta from
SQLite. Build the seeders once, then run them from the repository root:

```bash
cargo build --release --manifest-path infrastructure/seeders/Cargo.toml
RUST_LOG=info infrastructure/seeders/target/release/seed_heroes.exe
RUST_LOG=info infrastructure/seeders/target/release/seed_matchups.exe
```

This seeds 126 heroes and their semantics, real matchup/synergy rows, and a
patch-scoped meta snapshot. `seed_matchups --synthetic` exists for UI work but
writes rows tagged `synthetic`, which are never selected silently.

The database lives at `%APPDATA%\com.smartdraftassistant.app\smart_draft.db`
(Windows); paths are resolved by `apps/desktop/src-tauri/src/paths.rs`.

### Configure GSI

1. Copy `apps/desktop/gamestate_integration_smart_draft.cfg` to
   `%APPDATA%\dota 2 beta\game\dota\cfg\gamestate_integration\`.
2. Restart Dota 2.
3. Start a lobby or a real draft. The app listens on port 3000 by default.

### Run

```bash
# Terminal 1: frontend dev server
cd apps/desktop
npm run dev

# Terminal 2: Tauri app
cd apps/desktop
npm run tauri dev
```

ML API keys are only needed for data collection, not for running the app. Copy
`ml/.env.example` to `ml/.env` and set `STRATZ_API_KEY` (collectors) and
optionally `OPENDOTA_API_KEY`. `ml/.env` is ignored and scanned by CI.

## Verify

The single quality gate mirrors CI:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
```

It runs 13 steps: secret scan, Rust format/clippy/tests (desktop and seeders),
frontend lint/typecheck/build/tests, Python byte-compile, the ML
contract/quality-gate tests, and the remaining Python suite. Current baseline:

| Check | Result |
| --- | --- |
| Rust library tests | 69 passed, 1 ignored (benchmark) |
| Rust seeders | 1 passed |
| Frontend tests | 19 passed (3 files) |
| Python contract + gate tests | 14 passed |
| Remaining Python suite | 49 passed |
| Full gate | 13/13 steps |

`pwsh` is not installed in the verified environment; use Windows PowerShell as
above.

## Model quality

The app ships `ml/models/draft_model_xgboost_full.onnx` (887 inputs). It was
re-evaluated on the chronological held-out partition of
`draft_features.parquet` (102,118 matches; test = last 10,213):

| Mode | ROC-AUC | Brier | ECE |
| --- | --- | --- | --- |
| `as_trained` (logged values) | 0.5388 | 0.2481 | 0.0124 |
| `inference` (runtime defaults) | **0.5122** | **0.2495** | 0.0241 |
| Constant train-prior baseline | 0.5000 | 0.2493 | — |

The inference-mode model is barely above chance and slightly worse than a
constant predictor on Brier. The promotion gate
(`ml/training/quality_gate.py`) requires ROC-AUC ≥ 0.60, Brier ≤ 0.235,
ECE ≤ 0.10, and ≥ 500 test rows. The bundled artifact exits **1**.

```bash
ml/venv/Scripts/python.exe ml/training/quality_gate.py
```

Exit codes are deliberate: `0` pass/promotion eligible, `1` quality failure,
`2` evaluation error (missing artifact, schema drift, fingerprint mismatch,
bad labels). `--promote-to` copies the bundle only after exit `0`.

ML remains **off by default** (`ml_enabled = false`). The engine skips
inference entirely when disabled and renormalizes the deterministic weights.
Turning it on is a diagnostic opt-in, not a recommendation.

Training and gate details: [`docs/ML_TRAINING.md`](docs/ML_TRAINING.md).

## Architecture

```
Dota 2 ──GSI/HTTP──▶ Rust backend (Tauri v2)
                         ├── GSI lifecycle + validation
                         ├── Draft engine / scorers / beam search
                         ├── SQLite: heroes, matchups, synergies, meta
                         ├── ONNX Runtime (opt-in, schema-validated)
                         └── Tauri IPC ──▶ React UI + overlay
```

- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Zustand
- **Backend:** Rust, Tauri v2, warp, parking_lot, rusqlite
- **ML:** Python training → XGBoost → ONNX Runtime
- **Data:** SQLite for runtime state; Parquet for offline training
- **Sources:** OpenDota, STRATZ

## Project layout

```text
apps/desktop/
  src/                     React UI, stores, services
  src-tauri/src/           Rust: engine, gsi, commands, database, ml
infrastructure/seeders/    Hero/matchup/synergy seeders
ml/
  feature-engineering/     processor.py (raw JSON → features)
  training/                train_full.py, quality_gate.py, schema/manifest export
  tests/                   ONNX contract, schema reproduction, gate behaviour
  datasets/                Local raw/processed data (mostly untracked)
  models/                  Bundled ONNX + schema/features/config
scripts/verify.ps1         Single local/CI quality gate
docs/                      Roadmap, audits, design notes
```

## Documentation

Start with the live status, not the older design notes:

- [Roadmap and live status](docs/ROADMAP.md): what is verified, what is open,
  measurements, decisions, and next tasks.
- [Audit (2026-10-27)](docs/AUDIT_2026-10-27.md): severity-ranked findings with
  remediation status; original evidence preserved.
- [ML training and quality](docs/ML_TRAINING.md): bundled metrics, gate policy,
  feature contract, pipeline.
- [Draft engine](docs/ENGINE.md), [GSI](docs/GSI.md),
  [architecture](docs/ARCHITECTURE.md).
- [CI and quality gates](docs/STEP_08_CI.md),
  [ML inference contract](docs/STEP_02_ML_INFERENCE.md).
- [Wiki](docs/wiki/index.md).

Older step documents retain historical numbers where they are not the source
of truth. `ROADMAP.md` and the current code win.

## Known limitations

- The ML model is near a constant baseline and the gate fails it.
- The last confirmed full test run was on Windows with a local Python venv; no
  macOS/Linux runtime verification was performed.
- GSI is covered by fixture-driven lifecycle tests, but not by a replayed live
  session. The HTTP status mapping and Tauri event emission are type-checked,
  not exercised end-to-end in CI.
- The app has not been launched with the new CSP and split window capabilities
  in this environment.
- `confidence` currently equals the score, `lane_balance` is a placeholder, and
  risk levels are score thresholds; treat UI certainty as indicative.
- The processed dataset is ~1.9 GB across many loose files and is not
  versioned or retained beyond the local disk.

## Contributing

1. Run `scripts\verify.ps1` before proposing a change.
2. Keep ML an enhancement layer: the deterministic engine must work without it.
3. Every recommendation must remain explainable.
4. Treat model/schema changes as data changes: retrain, re-export the schema,
   run the gate, and update the golden fixture.
5. Preserve audit evidence; add status changes with the measurement or test
   that justifies them.

## License

MIT
