# Step 8 Design: CI and Quality Gates

- Date: 2026-10-23
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), L1, M5

## Implementation status

- [x] Add deterministic scripts for format, lint, typecheck, and tests.
- [x] Run `cargo fmt --check`, `cargo clippy -D warnings`, and `cargo test`.
- [x] Run TypeScript typecheck via the build and the frontend tests.
- [x] Run Python byte-compile and pytest (which includes the ML feature-contract test).
- [x] Run secret scanning on every change.
- [x] Add a GitHub Actions workflow now that the repository is initialized.
- [ ] Require the gate for merges (repository setting; runbook below).
- [x] Document local reproduction.
- [x] Clean the pre-existing Rust lints so `clippy -D warnings` is enforceable.

## Problem

There was no automated check for Rust formatting/clippy/tests, TypeScript
typecheck, Python compile/tests, secret scanning, or the ML feature contract
(L1). The plan and status board had drifted from the code (M5), and generated
ML artifacts sat next to source without an ignore policy (L3).

## Scripts

Everything runs through one entry point so local and CI commands cannot diverge.

- `scripts/verify.ps1` orchestrates the whole gate and exits non-zero if any
  step fails. It prints a per-step PASS/FAIL summary. Switches:
  `-SkipRust`, `-SkipFrontend`, `-SkipPython`, `-SkipSecrets`,
  `-ForceInstall`, and `-Python <path>`.
- `scripts/secret_scan.py` is a dependency-free scanner for the tracked tree
  (plus untracked, non-ignored files). It matches private-key blocks, cloud and
  provider token formats, credential-shaped assignments and query parameters,
  and skips allowlisted paths (`.env.example`, `docs/`, this scanner, the
  secret tests) and obvious placeholders.

`verify.ps1` resolves Python from `ml\venv\Scripts\python.exe` when present and
falls back to `python`. Frontend dependencies are only installed when
`node_modules` is missing unless `-ForceInstall` is passed.

### Gate contents

| Step | Command |
| --- | --- |
| Secret scan | `python scripts/secret_scan.py` |
| Rust format | `cargo fmt --manifest-path ... --all -- --check` |
| Rust lint | `cargo clippy --manifest-path ... --all-targets -- -D warnings` |
| Rust tests | `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib` |
| Rust tests (seeders) | `cargo test --manifest-path infrastructure/seeders/Cargo.toml` |
| Frontend build | `npm run build` (in `apps/desktop`; `tsc && vite build`) |
| Frontend tests | `npm test` (in `apps/desktop`) |
| Python compile | `python -m compileall -q ml/training ml/feature-engineering scripts` |
| Python tests | `python -m pytest ml/tests -q` |

The ML feature-contract test (`ml/tests/test_onnx_contract.py`) runs inside the
Python test step; it compares the exported schema against the bundled model and
the golden fixture. The Rust golden test (`ml::`) runs inside the Rust test
step and is skipped only when `onnxruntime.dll` is absent.

### Lint cleanup

Enabling `clippy -D warnings` required clearing ten pre-existing warnings: six
auto-deref sites (`commands.rs`), a derivable `impl` and a missing `Default`
(`engine.rs`), an `unnecessary_sort_by` (`engine.rs`), and a `type_complexity`
in `ml.rs` (addressed with a `SemanticGetter` alias). Clippy and both Rust test
suites are now clean.

## Continuous integration

`.github/workflows/ci.yml` runs on every push, pull request, and manual
dispatch, with a concurrency group that cancels superseded runs.

- `secret-scan` (ubuntu, Python 3.12): the fast, self-contained scan.
- `quality-gate` (windows-latest): installs the Rust toolchain
  (`rustfmt`, `clippy`) with `Swatinem/rust-cache`, Node 20 (npm cache), and
  Python 3.11 (pip cache); installs `ml/requirements.txt`; then runs
  `pwsh -File scripts/verify.ps1 -ForceInstall`.

Windows is the runner of record because the app targets Windows (bundled
`onnxruntime.dll`, Tauri shell) and the golden Rust test needs the bundled DLL.

## Repository hygiene

The root `.gitignore` now ignores `ml/experiments/` and all generated model
artifacts under `ml/models/` except the bundled release model
(`draft_model_xgboost_full.onnx` plus its `_config.json`, `_features.json`, and
`_schema.json`). `ml/datasets/` stays ignored (the processed Parquet is ~260 MB).
`.opencode/` only tracks `rules/smart-draft-assistant.md`; its `node_modules`,
`package.json`, and lockfile are ignored by `.opencode/.gitignore`.

## Requiring the gate for merges

Branch protection is a repository setting, not code:

1. GitHub -> Settings -> Branches -> add a rule for `master` (or `main`).
2. Enable "Require status checks to pass before merging".
3. Select `Secret scan` and `Quality gate (Windows)` as required checks.
4. Enable "Require branches to be up to date before merging" if desired.

## Verification

```powershell
# Full gate (Windows PowerShell or PowerShell 7)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1   # all steps PASS

# Individual steps
cargo fmt --manifest-path apps/desktop/src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path apps/desktop/src-tauri/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib   # 36 passed
npm run build --prefix apps/desktop
npm test --prefix apps/desktop                                       # 11 passed
python ml\venv\Scripts\python.exe -m pytest ml\tests -q              # 28 passed
python ml\venv\Scripts\python.exe scripts\secret_scan.py             # clean
```

The gate was run end to end on 2026-10-23: all steps PASS.

## Residual risks

- ESLint is declared in `package.json` but not installed and has no config, so
  `npm run lint` is not part of the gate; the frontend step is typecheck + build
  + tests. Add ESLint config and a dependency before enabling it in CI.
- `scripts/verify.ps1` targets Windows PowerShell 5.1+ semantics; `pwsh` is
  only available on the CI runner, so local reproduction uses `powershell.exe`.
- The gate installs `ml/requirements.txt` (XGBoost, LightGBM, ONNX) in CI,
  which is the slowest step; the Rust and frontend caches offset this.
- CI does not build the full Tauri bundle (only `npm run build` + Rust tests);
  a packaging smoke test remains future work.
