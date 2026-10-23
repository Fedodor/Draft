# CI and Quality Gates

- Type: Engine/Process
- Status: Implemented
- Updated: 2026-10-23
- Canonical source: [`../../STEP_08_CI.md`](../../STEP_08_CI.md)
- Plan: [`../plans/remediation-plan.md`](../plans/remediation-plan.md) (Step 8)
- Audit findings: L1, M5

## Summary

One script, `scripts/verify.ps1`, runs the whole quality gate locally and in CI
so the two cannot diverge. It exits non-zero on any failure and prints a
per-step summary.

## Gate contents

| Step | Command |
| --- | --- |
| Secret scan | `python scripts/secret_scan.py` |
| Rust format / lint | `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings` |
| Rust tests | desktop `--lib` (36) and seeders (1) |
| Frontend lint | `npm run lint` (ESLint flat config, `--max-warnings 0`) |
| Frontend | `npm run build` (tsc + vite) and `npm test` (11) |
| Python | `compileall` and `pytest ml/tests` (33, includes the ONNX contract test) |

Switches: `-SkipRust`, `-SkipFrontend`, `-SkipPython`, `-SkipSecrets`,
`-ForceInstall`, `-Python <path>`. Python resolves from
`ml\venv\Scripts\python.exe` when present, else `python`.

## Continuous integration

`.github/workflows/ci.yml` runs on push, pull request, and manual dispatch:

- `secret-scan` (ubuntu, Python 3.12): the fast self-contained scan.
- `quality-gate` (windows-latest): Rust (rustfmt + clippy) with `rust-cache`,
  Node 20, Python 3.11, then `pwsh -File scripts/verify.ps1 -ForceInstall`.

Windows is the runner of record (bundled `onnxruntime.dll`, Tauri shell, and the
Rust golden test needs the DLL).

## Notes

- `clippy -D warnings` is enforceable because ten pre-existing lints were fixed
  (auto-deref, derivable `impl`, missing `Default`, `unnecessary_sort_by`, and a
  `type_complexity` now aliased as `SemanticGetter`).
- Generated model artifacts under `ml/models/` are ignored except the bundled
  `draft_model_xgboost_full.*` release set.
- ESLint 9 flat config (`apps/desktop/eslint.config.js`) uses `@eslint/js` +
  `typescript-eslint` recommended plus the React hooks and refresh plugins; the
  `react-refresh/only-export-components` rule is off because hooks are
  co-located with components.
- Branch protection to require `Secret scan` and `Quality gate (Windows)` before
  merging is a repository setting; see the canonical doc for the runbook.
