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
| Tauri runtime smoke test | Automated in CI | `scripts/ui_smoke.ps1` drives the dev build (`gui-smoke`) and the installed NSIS bundle (`gui-smoke-bundle`) over WebView2 CDP: CSP assets, four tabs, GSI→UI, card/header layout at 900 and 800 px, overlay toggle, embedded-asset origin, model/DLL resource resolution, and in-app search latency with ML on and off |
| First-run state | Working | An unseeded database shows a notice with the seeder commands instead of an empty-looking draft |

## Quick start

### Prerequisites

- Windows 10/11 (the verified platform)
- Node.js 20+
- Rust stable (with `rustfmt` and `clippy`)
- Python 3.12.13 via uv (source-only on Windows; python.org has no installer)
- uv (https://docs.astral.sh/uv/)
- Dota 2, for live drafting

### Install

```bash
cd apps/desktop
npm install

cd ../../ml
# Python 3.12.13 has no Windows installer; uv provides the managed build.
uv python install 3.12.13
uv venv --python 3.12.13 --seed venv
venv\Scripts\activate
pip install --require-hashes -r requirements.lock
```

`requirements.txt` lists the direct dependencies. `requirements.lock` pins the
full verified Python 3.12.13 set with SHA-256 hashes for the Windows wheels and
is what CI installs. From the repository root, check that it reproduces with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\update_ml_lock.ps1
```

The script creates a fresh 3.12.13 venv, installs the current hashed lock, runs
the ML suite, and compares the regenerated file. Add `-Update` to write it;
versions are intentionally not upgraded by this command.

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

It runs 14 steps: secret scan, seed-patch consistency, Rust format/clippy/tests
(desktop and seeders), frontend lint/typecheck/build/tests, Python byte-compile,
the ML contract/quality-gate tests, and the remaining Python suite. Current
baseline:

| Check | Result |
| --- | --- |
| Rust library tests | 87 passed, 2 ignored (benchmarks) |
| Rust seeders | 2 passed |
| Frontend tests | 32 passed (5 files) |
| Python contract + gate tests | 14 passed |
| Remaining Python suite | 152 passed |
| Full gate | 14/14 steps |

`pwsh` is not installed in the verified environment; use Windows PowerShell as
above.

The GUI smoke test is a separate step because it launches the real app and needs
ports 1420, 3000 and 9333 free:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ui_smoke.ps1
```

It builds the debug binary, starts Vite, drives the WebView2 renderer over CDP,
and fails on a blocked CSP asset, a missing tab, a broken overlay toggle, GSI
payloads that never reach the UI, card/header layout regressions at the default
and minimum window widths, or an ONNX model that does not load from the app's
resources, or search latency that drifts past the benchmarked envelope. The
latency probe toggles `ml_enabled` through the Settings command, times the
shortlist and lookahead searches over IPC after the GSI draft arrives, restores
the original setting, and writes `recommendation-latency.json` next to the
screenshots. Under GitHub Actions the same table also goes to the job summary
with a pass/drift marker per row, so the numbers are visible without the
artifact. Screenshots and logs land in `.ui-smoke/`. CI runs it twice:
`gui-smoke` against the dev build and `gui-smoke-bundle` against the packaged
release bundle, both on `windows-latest`.

`-Bundle` builds the NSIS installer, installs it silently into a temporary
directory, drives the installed executable with no dev server present, and
uninstalls it afterwards. It additionally asserts that the UI is served from the
app origin rather than Vite, that scripts and styles are embedded, and that the
model, its sidecars and `onnxruntime.dll` loaded from the installed
`resources/` directory:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ui_smoke.ps1 -Bundle
```

The bundle run needs a release build of the Tauri app (several minutes cold) and
ports 3000 and 9333 free; port 1420 must stay unused, which the script asserts.

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
- [Runtime-contract audit (2026-11-19)](docs/AUDIT_2026-11-19.md): CSP, model
  path loading, overlay dead-end, and other findings from a fresh pass.
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
- The runtime smoke test runs in CI against the dev build: no packaged bundle,
  no real Dota 2 stream, and no overlay drag/DPI verification yet.
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
