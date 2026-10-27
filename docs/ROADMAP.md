# Roadmap & Live Status: Smart Draft Assistant

- Updated: 2026-10-28 (session 3)
- Source audit: [`AUDIT_2026-10-27.md`](AUDIT_2026-10-27.md) (historical findings, preserved)
- Predecessor audit: [`AUDIT.md`](AUDIT.md) (2026-10-22)
- Branch: `master`, HEAD `2ef74fd`, session-3 changes **uncommitted** (nothing was committed or pushed)
- Purpose: single source of truth for what is fixed, what is verified, and what is next.

Status vocabulary:

| Status | Meaning |
| --- | --- |
| `open` | Confirmed, not addressed. |
| `in_progress` | Partially addressed; the remaining work is listed. |
| `fixed_unverified` | Code changed and unit-tested where possible, but the end-to-end/runtime path was not exercised. |
| `verified_fixed` | Code inspected after the change **and** a test or measurement directly exercises it. |
| `blocked` | Cannot progress in this environment; reason recorded. |
| `not_reproducible` | Re-checked against the current tree and the finding does not hold. |

> Rule used here: a finding is never marked `verified_fixed` on the strength of the edit alone. The evidence column names the actual check.

---

## 1. Trustworthy baseline (Phase 0)

Recorded before any change on 2026-10-27, clean tree, branch `master`:

| Check | Command | Baseline result |
| --- | --- | --- |
| Rust unit tests | `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib` | 40 passed, 0 failed |
| Frontend tests | `npm test -- --run` (in `apps/desktop`) | 14 passed (2 files) |
| TypeScript | `npx tsc --noEmit` | clean |
| ESLint | `npm run lint` | clean |
| Production build | `npm run build` | success |
| Python tests | `ml/venv/Scripts/python.exe -m pytest ml/tests -q` | 49 passed (7 warnings) |

Environment: Windows (`win32`), PowerShell, Rust toolchain from the repository lockfile, Node via `apps/desktop/node_modules`, Python 3.10.11 from `ml/venv`.

Current tree after the remediation session:

| Check | Command | Result |
| --- | --- | --- |
| Rust tests | `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib` | **51 passed**, 0 failed, 1 ignored (benchmark) |
| Rust lint | `cargo clippy --manifest-path apps/desktop/src-tauri/Cargo.toml --all-targets -- -D warnings` | clean (exit 0) |
| Rust format | `cargo fmt --manifest-path apps/desktop/src-tauri/Cargo.toml --all` | applied |
| Frontend tests | `npm test` (in `apps/desktop`) | **19 passed** (3 files) |
| TypeScript | `npx tsc --noEmit` | clean |
| ESLint | `npm run lint` | clean |
| Build | `npm run build` | success (344 kB JS / 24 kB CSS) |
| Python tests | `ml/venv/Scripts/python.exe -m pytest ml/tests -q` | **49 passed**, 7 warnings |
| Full gate | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1` | **PASSED** — 12/12 steps (secret scan, Rust fmt/clippy/tests for desktop + seeders, frontend lint/build/tests, Python compile/tests) |

Second session (GSI lifecycle integration, 2026-10-27):

| Check | Command | Result |
| --- | --- | --- |
| Lifecycle tests | `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib lifecycle` | **18 passed**, 0 failed |
| Rust tests | `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib` | **69 passed**, 0 failed, 1 ignored |
| Rust lint | `cargo clippy --manifest-path apps/desktop/src-tauri/Cargo.toml --all-targets -- -D warnings` | clean (exit 0) |
| Rust format | `cargo fmt --manifest-path apps/desktop/src-tauri/Cargo.toml --all` | applied, no diff after |
| Full gate | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1` | **PASSED** — 12/12 (49 Python tests, 19 frontend tests) |

Third session (ML quality gate, schema reproducibility, leakage evidence, 2026-10-28):

| Check | Command | Result |
| --- | --- | --- |
| Dataset fingerprint | `Get-FileHash ml/datasets/processed/draft_features.parquet -Algorithm SHA256` | `264a6ad5…91621`, matches the manifest; 102,118 rows × 890 columns |
| Bundled model, as-trained | `python ml/training/train_full.py --evaluate-only` | test ROC-AUC **0.5388**, Brier **0.2481**, log loss **0.6893** |
| Quality gate (inference mode) | `python ml/training/quality_gate.py` | **FAIL** (exit 1): AUC **0.5122**, Brier **0.2495**, ECE **0.0241**; baseline Brier **0.2493** |
| Contract + gate tests | `python -m pytest ml/tests/test_feature_schema.py ml/tests/test_quality_gate.py -q` | **14 passed** |
| Python suite | `python -m pytest ml/tests -q` | **63 passed**, 24 warnings |
| Full gate | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1` | **PASSED** — 13/13 steps; Rust 69, frontend 19, Python 14 + 49 |


Notes:

- `cargo test` occasionally surfaces PowerShell exit code 1 because the test harness writes to stderr; the authoritative line is `test result: ok`. `scripts/verify.ps1` checks `$LASTEXITCODE` directly.
- No Tauri app launch, CSP validation, or live Dota 2 GSI session was performed in this environment.

---

## 2. Progress board

### P0 — correctness and trust

#### P0-1 — ML contributes 25% while near baseline
- **Status:** `in_progress` (mitigation `verified_fixed`; automated promotion gate `verified_fixed`; model quality `open`)
- **Files:** `src-tauri/src/models.rs`, `src-tauri/src/engine.rs`, `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs`, `src/App.tsx`, `src/stores/index.ts`, `src/components/SettingsPanel.tsx`, `ml/training/quality_gate.py` (new), `ml/training/inference_contract.py` (new), `ml/tests/test_quality_gate.py` (new)
- **Implemented:**
  - `AppConfig::ml_enabled` now defaults to `false` and is forwarded to `DraftEngine::set_ml_enabled` at startup and from `set_config`.
  - When disabled, `MLScorer::score` is not called at all (no inference), the ML term is reported as neutral `0.5`, and the five heuristic weights are renormalized so scores stay in `[0,1]` (`active_weights`, `weighted_total`).
  - The same gate is applied in `simulation.rs` so the search path cannot silently re-enable ML scoring.
  - Explanations now say "Model estimate: … (experimental)" and only when ML is enabled; the previous "ML model favors this pick" wording is gone.
  - Default model path corrected to the bundled `draft_model_xgboost_full.onnx`.
  - **Session 3:** `ml/training/quality_gate.py` validates the artifact bundle (manifest ↔ schema ↔ feature order), dataset fingerprint, chronological split independence, labels, and ONNX input/output contract, then re-evaluates the held-out test partition twice: `as_trained` (logged values) and `inference` (runtime defaults for `patch`, `patch_version`, `duration`, `avg_mmr`, `game_mode`, `lobby_type`). The promotion policy applies to `inference` by default.
  - Exit codes are explicit: **0** pass/promotion eligible, **1** quality failure, **2** evaluation error (untrustworthy or unrunnable). `--promote-to` copies the bundle only when the gate passed. Manual runtime opt-in is still allowed because ML is off by default; only automatic promotion is blocked.
- **Tests:** `test_weights_are_renormalized_when_ml_is_disabled`, `test_ml_gate_changes_only_the_ml_term`, `test_app_config_defaults_and_validation` (asserts `!ml_enabled` and the `.onnx` default); `ml/tests/test_quality_gate.py` covers pass, threshold failure, fingerprint mismatch, malformed/missing artifacts, schema order drift, invalid labels/predictions, baselines, leakage classification, and promotion refusal.
- **Fresh evidence (2026-10-28, full dataset):**
  - Policy: ROC-AUC ≥ 0.60, Brier ≤ 0.235, ECE ≤ 0.10, ≥ 500 test rows.
  - `inference`: ROC-AUC **0.5122** (95% CI 0.5010-0.5234), Brier **0.2495**, ECE **0.0241**, log loss 0.6922, accuracy 0.5125, 10,213 rows.
  - `as_trained`: ROC-AUC **0.5388** (CI 0.5277-0.5500), Brier **0.2481**, ECE **0.0124**.
  - Constant train-prior baseline on the same rows: Brier **0.2493** (probability 0.5245). The model is not better than the baseline on Brier and is barely above chance on AUC.
  - Decision: `promotion_eligible=false`, failed checks `policy.roc_auc`, `policy.brier_score`.
- **Remaining risk:** the shipped model is still untrustworthy. The gate now blocks *automatic* promotion, but a user can still opt in manually. Fixing quality needs a retrain and a feature ablation, not a threshold change.
- **Next action:** train a candidate with post-draft/ordinal fields removed (P1-6), evaluate it through the gate in `inference` mode, and only bundle it if all checks pass. Keep the current artifact in place until then.

#### P0-2 — Draft lifecycle corruption in GSI ingestion
- **Status:** `verified_fixed` (payload → engine; live GSI replay and warp/Tauri emission outstanding)
- **Files:** `src-tauri/src/gsi.rs`, `src-tauri/src/gsi/lifecycle_tests.rs` (new), `src-tauri/src/engine.rs`, `src-tauri/src/commands.rs`
- **Implemented:**
  - Match boundary: `GsiServerState::last_game_state` + `entering_new_draft()` reset the engine when `HERO_SELECTION` is entered from a non-draft state (init, in-game, post-game). Staying in hero selection or moving to strategy/pre-game does not reset. The payload that triggers the reset is **not** ingested (Dota can still include the previous game's slots in that first update); the cleared state is emitted and the next update populates the new draft.
  - Validation: every GSI pick/ban goes through `validate_pick`/`validate_ban` (known hero, hero still available, team not already at 5 picks, no duplicate).
  - Dedupe: keyed by hero id across **both** teams and across picks **and** bans, so slot renumbering or a picked hero appearing as a ban cannot double-count. Picks are reconciled before bans; if one snapshot contains the same hero in both, the pick wins.
  - Team attribution: `team_id` (2 = Radiant, 3 = Dire) wins over the `team2`/`team3` slot position; unrecognized ids fall back to the slot and warn.
  - Manual undo: `DraftEngine::suppressed_heroes` records a hero the user undid; GSI ingestion skips it until `reset()`. An explicit manual pick/ban lifts the suppression.
  - `reset()` clears the undo stack, suppression set, and recommendation cache, and restores `available_heroes`.
  - **Session 2:** `apply_gsi_payload` (auth → parse → lifecycle → reconcile) is extracted from the warp handler so tests drive the real ingestion core; the handler keeps status mapping and event emission only. `reconcile_draft` now enforces `MAX_BANS_PER_TEAM = 5`; before, bans were unbounded across renumbered/stale snapshots and could permanently remove extra heroes from the pool.
- **Tests:**
  - Inline (`gsi.rs`): `test_entering_new_draft_detection`, `test_draft_actions_prefers_team_id_over_slot`, `test_reconcile_dedupes_by_hero_across_picks_and_bans` (also asserts idempotence), `test_reconcile_skips_unknown_heroes_and_manual_undone_heroes`, `test_reset_restores_available_heroes_and_clears_suppression`, `test_undo_suppresses_hero_until_next_match`.
  - New (`gsi/lifecycle_tests.rs`, 18 tests): consecutive matches (`lifecycle_consecutive_matches_do_not_leak_state`, `lifecycle_strategy_time_after_draft_keeps_state`); duplicates/invalid (`lifecycle_duplicate_and_conflicting_actions_are_deduped`, `lifecycle_unknown_heroes_are_ignored_without_dropping_valid_actions`, `lifecycle_stale_snapshots_never_remove_or_duplicate_actions`, `lifecycle_malformed_payloads_are_rejected_without_mutating_state`, `lifecycle_auth_is_enforced_before_any_state_change`); manual undo (3 tests); attribution/order/ban cap (7 tests); 300-payload invariant property test (`lifecycle_invariants_hold_for_adversarial_payload_sequences`). Every lifecycle test also calls `assert_invariants` (no pick+ban, no duplicate, known heroes, pool consistency, per-side caps, consistent `pick_number`).
- **Verified by:** `cargo test ... --lib lifecycle` → 18 passed; full lib → 69 passed; clippy exit 0; `scripts/verify.ps1` → PASSED 12/12.
- **Remaining risk:** no captured/live GSI replay; hand-written fixtures only. The warp status mapping and `gsi-team`/`gsi-update` emissions are type-checked but not executed. Only the reset-triggering payload is skipped, so an out-of-order *second* stale `HERO_SELECTION` snapshot would be treated as new (documented; no heuristic invented). Undo suppression is hero-scoped for the match and cannot tell a stale re-add from a genuine later pick of the same hero; an explicit manual pick lifts it.
- **Next action:** replay a captured multi-match GSI session (or a real client session) through the running app to validate payload shape, HTTP status, and event emission. See P0-2 remediation in `AUDIT_2026-10-27.md`.

#### P0-3 — Incorrect Captain's Mode fallback sequence
- **Status:** `verified_fixed`
- **Files:** `src-tauri/src/gsi.rs`
- **Implemented:** `CM_ORDER` is now 4 bans → 4 picks → 4 bans → 4 picks → 2 bans → 2 picks (20 actions) with snake sides:
  - bans 0-3: R, D, R, D
  - picks 4-7: R, D, D, R
  - bans 8-11: D, R, D, R
  - picks 12-15: D, R, R, D
  - bans 16-17: R, D
  - picks 18-19: R, D
- **Tests:** `test_cm_order_matches_current_phase_layout` (phase boundaries + exact side order) and the updated `test_resolve_turn_falls_back_to_cm_order`.
- **Remaining risk:** the table encodes the audited sequence; official Valve documentation for the exact patch should be re-checked before the next CM change. GSI `activeteam`/`pick` still take precedence when present.
- **Next action:** none unless the published sequence changes.

#### P0-4 — Expensive recommendation execution
- **Status:** `in_progress`
- **Files:** `src-tauri/src/engine.rs`, `src-tauri/src/commands.rs`, `src/services/recommendationFeed.ts`, `src/hooks/useRecommendationFeed.ts`, `src/App.tsx`, `src/components/OverlayWindow.tsx`
- **Implemented:**
  - Shared, state-keyed recommendation cache (`get_recommendations_cached`) covering team, limit, ML flag, patch, picks, bans, and available heroes. The cache is invalidated on every draft mutation, meta/profile update, ML toggle, and reset.
  - Inference is skipped entirely when ML is off (the previous code still ran ONNX and only discarded the value).
  - `get_recommendations` and `get_score_breakdown` run under `spawn_blocking`; the async runtime is no longer blocked by scoring.
  - `predict_draft_value` takes an engine snapshot, drops the engine lock, then reads the ML engine — one consistent lock order (engine → ML).
  - Frontend: 200 ms trailing debounce, one in-flight orchestration layer per window, stale-response rejection via monotonic request ids, and cache sharing between main window and overlay.
  - Overlay polls only when GSI has been quiet (15 s poll, 10 s staleness threshold).
- **Measured (see §3):** release, 120 heroes, bundled model — ML on **23.2 ms/request**, ML off **0.2 ms**, cache hit **0.002 ms**.
- **Not done:** per-candidate feature-map reuse, batched ONNX inference, and scoring outside the state read guard. Current benchmark is below the 150 ms target but is a harness measurement, not in-app p95.
- **Next action:** benchmark inside the running app (p95 over ≥100 GSI-driven requests) before optimizing further; only then decide whether batch inference is justified.

### P1 — high impact

#### P1-1 — No debounce, coalescing, cancellation, or stale protection
- **Status:** `verified_fixed`
- **Files:** `src/services/recommendationFeed.ts`, `src/services/recommendationFeed.test.ts`, `src/hooks/useRecommendationFeed.ts`, `src/App.tsx`, `src/components/OverlayWindow.tsx`
- **Implemented:** `RecommendationFeed` applies draft payloads immediately, debounces only the recommendation request (200 ms), numbers requests, drops responses older than the last applied one, and keeps the previous recommendations on failure. The hook owns the single `gsi-update` subscription per window.
- **Tests:** 5 frontend tests — debounce coalescing, out-of-order rejection, failure keeps last data, `start()` ordering, and no stale initial snapshot overwriting a live payload.
- **Remaining risk:** React hook wiring is type-checked but not component-tested; cancellation is logical (stale rejection), not an abort of the IPC call.
- **Next action:** add a component-level test once a React test environment exists, or verify during a live draft.

#### P1-2 — Overlay blanks when ML prediction fails; ignores player side
- **Status:** `fixed_unverified`
- **Files:** `src/components/OverlayWindow.tsx`, `src/hooks/useRecommendationFeed.ts`
- **Implemented:** `predictDraftValue` runs in its own `try/catch`; failure sets the win-probability display to `—` and never clears recommendations. The overlay listens to `gsi-team` and feeds the current side into its recommendation requests.
- **Remaining risk:** no component/runtime test; visual verification during a real draft is pending.
- **Next action:** manual check with the model removed (`ml_enabled` on, model file renamed) to confirm picks still render.

#### P1-3 — Overlay toggle steals focus
- **Status:** `verified_fixed`
- **Files:** `src/components/Header.tsx`
- **Implemented:** `setFocus()` removed from the toggle path; the overlay is shown and set always-on-top only. Comment explains why.
- **Verification:** code inspection plus `rg` for `setFocus` (no remaining call sites).
- **Remaining risk:** no live Dota 2 focus test.
- **Next action:** confirm during the next real-draft session.

#### P1-4 — Settings with no runtime effect
- **Status:** `in_progress`
- **Files:** `src-tauri/src/models.rs`, `src-tauri/src/commands.rs`, `src-tauri/src/lib.rs`, `src/components/SettingsPanel.tsx`, `src/components/Header.tsx`, `src/stores/index.ts`
- **Implemented:** `ml_enabled` (engine gate), `overlay_enabled` (hidden when disabled), `overlay_position` and `overlay_size` (applied via `LogicalPosition`/`LogicalSize`). The inert `auto_update_meta` toggle was removed from the UI. Auth-disabled warning added to the GSI settings block. Stale default model path fixed.
- **Still inert:** `preferred_language` (stored, no UI or runtime use). `auto_update_meta` remains in `AppConfig` for backward compatibility but is no longer presented as a working control.
- **Remaining risk:** overlay geometry is applied from the main window at mount/config change; no test covers the Tauri window API calls.
- **Next action:** either implement localization or remove/hide `preferred_language`; decide on `auto_update_meta` (implement or delete from the model in a follow-up migration).

#### P1-5 — Feature pipeline cannot reproduce the shipped feature set
- **Status:** `fixed_unverified` (set reproduced and unit-tested; full real-data regeneration not run)
- **Files:** `ml/feature-engineering/processor.py`, `ml/tests/test_processor_features.py`, `ml/tests/test_feature_schema.py` (new), `ml/training/inference_contract.py` (new)
- **Implemented:** `_load_hero_matchups` now accepts numeric `hero_a`/`hero_b` ids (int, float, numeric string) and `hero_a_winrate`, and still accepts the legacy name + `winrate` shape. New `resolve_matchup_hero_id()` normalizes both. Tests cover numeric shapes, legacy names, and both parquet shapes via a stubbed reader.
- **Deliberately unchanged:** `all_hero_ids = range(1, 140)`. The shipped 887-feature contract (locked by the golden fixture and the manifest) includes `*_hero_139` has/ban features and separate win-rate features for 145/155. Changing the range without retraining would break the golden contract.
- **Session 3 evidence:** `ml/tests/test_feature_schema.py` drives the real `FeatureProcessor.process_match` on a deterministic synthetic fixture:
  - **887-name set reproduced exactly** without matchup data. All 633 non-win-rate inputs keep a stable code-defined order except the two `*_ban_avg_winrate` columns (631 exact-order matches).
  - 254 per-hero win-rate features match as a set; their order (and the position of the two ban averages) is first-seen row history in the shipped parquet, not a code contract. The shipped file lists the ban averages at the end (indices 885/886); the processor emits them beside the other win-rate aggregates.
  - With matchup data the processor emits **894** names: exactly the shipped set plus the 7 `*_matchup_*`/`matchup_advantage` features. The bundled parquet has no matchup columns, so shipping them requires a regenerated dataset, a retrain, and a new export.
  - A generated row encodes to a finite 887-element schema-ordered vector and the bundled ONNX model accepts it.
- **Not done:** lane features remain commented out; no full pipeline regeneration (dataset is 1.9 GB / 121,746 files). `process_all()` collects thread results in completion order, so the *column order* of a regenerated dataset can vary between runs; that does not change the name set but it is not reproducible. No retrain on regenerated data.
- **Next action:** make `process_all()` order deterministic (sort matches before processing or reindex to a canonical column list), then regenerate on a representative sample and diff names + order against the schema before any retrain.

#### P1-6 — Post-draft and ordinal features (leakage / train-serve skew)
- **Status:** `in_progress` (magnitude measured; ablation and retrain not done)
- **Files:** `ml/training/quality_gate.py`, `ml/training/inference_contract.py`, `ml/training/train_full.py`
- **Evidence (original):** `duration`, `avg_mmr`, `game_mode`, `lobby_type`, and raw `rad_pick_i`/`dire_pick_i` are numeric inputs; `duration` is post-draft and replaced by a median at inference.
- **Session 3 measurements (10,213 held-out rows):**
  - Replacing the six match-context fields with their schema defaults costs **0.0266 ROC-AUC** (0.5388 → 0.5122) and **0.0014 Brier** (0.2481 → 0.2495). That is the measured train/serve skew for this artifact, not an estimate.
  - `duration`: univariate AUC **0.4624**, gain rank **6**, 2,974 distinct values. It is a weak, slightly anti-correlated signal here, but the model does rely on it (top-10 gain).
  - `game_mode` univariate AUC **0.5103** (gain rank 327); `lobby_type` **0.5009** (no gain); `avg_mmr` is constant `0` (one value, no gain); `patch` is categorical (gain rank 131) and `patch_version` has no gain.
  - Ordinal pick slots (`rad_pick_0..4`, `dire_pick_0..4`) are near chance individually (AUC 0.4947-0.5027, gain ranks 298-411), so they are low-impact but still inputs the runtime only partially knows.
  - Split integrity is clean: train/validation/test sizes 81,694 / 10,211 / 10,213; zero `match_id` overlap; strict time boundaries (`train.max` 1,764,427,633 < `validation.min` 1,764,427,877; `validation.max` 1,775,425,453 < `test.min` 1,775,425,870). The gate checks all of this on every run.
- **Why untouched:** requires retraining plus held-out evaluation; changing the schema without that would invalidate the bundled model and the 887-feature contract.
- **Next action:** train an ablation that drops `duration`, `avg_mmr`, `game_mode`, `lobby_type`, and the raw pick-index columns (or replaces pick index with something available mid-draft), then compare AUC/Brier/ECE through the gate in `inference` mode. Only ship it if it beats the policy and the constant-prior baseline.

### P2 — medium impact

| ID | Finding | Status | Evidence / next action |
| --- | --- | --- | --- |
| P2-1 | `csp: null` | `fixed_unverified` | Explicit CSP in `tauri.conf.json`; needs a real app launch (assets, IPC, dev server) to prove nothing breaks. |
| P2-2 | Overlay has main-window privileges | `fixed_unverified` | New least-privilege `capabilities/overlay.json`; `default.json` restricted to `main`. Needs launch validation that drag/hide still work. |
| P2-3 | Arbitrary model path (DLL sideloading) | `fixed_unverified` | `canonical_model_path()` requires an existing file under app-data, resource dir, repo `ml/models`, or the dev fallback. No test yet. |
| P2-4 | GSI auth optional | `closed_by_design` | Auth remains optional but the Settings UI now warns that any local process can post a draft. |
| P2-5 | Misleading confidence / risk / lane semantics | `open` | `confidence == score`, `lane_balance` constant `0.5`, risk = score thresholds. Not changed (would be a UI semantics change requiring product input). |
| P2-6 | `partial_cmp().unwrap()` | `verified_fixed` | Replaced with `total_cmp` in `engine.rs` and both `simulation.rs` call sites. |
| P2-7 | Sync mutex in async commands, lock order | `in_progress` | `get_recommendations`, `get_score_breakdown`, `predict_draft_value` moved to `spawn_blocking`; predict lock order fixed (engine → ML). Other small commands still lock `std::sync::Mutex` briefly. |
| P2-8 | No log rotation, per-request info logs | `verified_fixed` | `rotate_log_if_needed` at 5 MB with one backup + test; per-recommendation and per-GSI-update logs dropped to `debug!`. |
| P2-9 | Dataset storage / versioning / retention | `open` | 1.9 GB, 121,746 loose files. No registry. Blocked by policy decision, not code. |
| P2-10 | Docs conflict (feature counts, AUC, lane status) | `in_progress` | `ROADMAP.md`, `AUDIT_2026-10-27.md`, `GSI.md`, `ENGINE.md`, `ML_TRAINING.md` and `README.md` now agree on 887 inputs, the measured 0.5388 as-trained / 0.5122 inference AUC, ML opt-in, and real ingestion rules. Remaining: older step/design docs and the empty `RefactorML.md`. |
| P2-11 | Data health not surfaced | `open` | `get_data_health` / `getMlStatus` exist; no UI panel. Candidate for Phase 4. |

### P3 — lower priority

| ID | Finding | Status | Evidence / next action |
| --- | --- | --- | --- |
| P3-1 | Linear hero lookups in the candidate loop | `open` | Low absolute cost (~126 heroes). Would need a `HashMap<HeroId, Hero>` on the engine. |
| P3-2 | `SearchConfig` unvalidated | `verified_fixed` | `SearchConfig::sanitized()` clamps candidates 1-50, depth ≤ 3, beam 1-20; wired into `get_search_recommendations`. |
| P3-3 | `overlay_size` has no behavior | `verified_fixed` | Header applies `LogicalSize` alongside position. |
| P3-4 | Brittle model resource paths | `in_progress` | Default model path fixed; `canonical_model_path` adds an allow-list. Bundling still uses relative `../../../ml/models`. |

---

## 3. Measurements

All numbers below are **measured in this environment**, not estimates from the earlier audit. They are not production p95 values.

### Model quality gate (2026-10-28)

- Command: `ml/venv/Scripts/python.exe ml/training/quality_gate.py` (or `--json`, `--report <path>`, `--evaluation-mode as_trained`).
- Artifact: `ml/models/draft_model_xgboost_full.onnx` + `_schema.json` + `_features.json` + `_manifest.json`; dataset SHA-256 `264a6ad5…91621`.
- Split: chronological 80/10/10 on `start_time`; test partition = last 10,213 matches (min 1,775,425,870, max 1,787,903,076).
- Policy (default): ROC-AUC ≥ 0.60, Brier ≤ 0.235, ECE ≤ 0.10, ≥ 500 test rows.

| Mode | ROC-AUC (95% CI) | Brier | Log loss | Accuracy | ECE | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `as_trained` (logged match fields) | 0.5388 (0.5277-0.5500) | 0.2481 | 0.6893 | 0.5294 | 0.0124 | reference only |
| `inference` (runtime defaults, policy default) | **0.5122** (0.5010-0.5234) | **0.2495** | 0.6922 | 0.5125 | 0.0241 | **fails AUC + Brier** |
| Constant train prior (0.5245) | 0.5 | 0.2493 | 0.6917 | 0.5274 | — | reference |

The gate exits **1** on these numbers, writes `draft_model_xgboost_full_quality_report.json`, and refuses `--promote-to`. A fingerprint mismatch also exits **1** (evaluation runs but is never promotion-eligible). It exits **2** for missing/malformed artifacts, schema drift, invalid labels, or malformed predictions.

Leakage/univariate audit from the same run (held-out rows):

| Feature | Availability | Univariate AUC | Gain rank |
| --- | --- | --- | --- |
| `duration` | post-draft (runtime uses median) | 0.4624 | 6 |
| `game_mode` | match setting (median) | 0.5103 | 327 |
| `lobby_type` | match setting (median) | 0.5009 | none |
| `avg_mmr` | constant 0 | — | none |
| `patch` | categorical | — | 131 |
| `rad_pick_0..4` | partial draft | 0.4964-0.5027 | 319-411 |
| `dire_pick_0..4` | partial draft | 0.4947-0.5012 | 298-381 |

Interpretation: post-draft substitution is worth ~0.027 AUC on this model, but the model is near a constant predictor in both modes. Removing leakage alone will not make it useful; see P1-6 and §8.

### Recommendation benchmark

- Test: `bench_recommendation_latency` in `src-tauri/src/engine.rs` (ignored by default).
- Command: `cargo test --release --manifest-path apps/desktop/src-tauri/Cargo.toml --lib bench_recommendation_latency -- --ignored --nocapture`
- Setup: 120 synthetic heroes, bundled `draft_model_xgboost_full.onnx` (887 features), empty matchup/synergy maps, 3 iterations after one warm-up, single Windows machine (no CPU pinning).
- Results:

| Path | Release | Debug |
| --- | --- | --- |
| ML enabled, full recommendation | 23.2 ms/request | 158.7 ms/request |
| ML disabled, full recommendation | 0.2 ms/request | 0.9 ms/request |
| Cache hit (`get_recommendations_cached`) | 0.002 ms/request | 0.032 ms/request |
| Single `MLScorer::score` (one candidate) | 0.144 ms | 1.447 ms |

Interpretation: inference dominates the ML-on path (~120 candidates × ~0.14 ms ≈ 17 ms) and is skipped entirely when ML is disabled. The cache makes the second window's identical request effectively free. Debug builds are 7-8× slower — do not quote debug numbers as user-facing latency.

### Not measured

- Model quality on a fresh held-out set: **now measured** for the bundled artifact (§ above); still no candidate model that passes the gate.
- In-app p95 latency, UI-thread responsiveness, or OGSI event rate.
- ONNX inferences per GSI event (expected ≤ 1 per coalesced request + cache hit for the second window, but not instrumented).
- Memory use, startup time with a cold ONNX session, or dataset I/O throughput.

---

## 4. Decisions and rationale

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-27 | ML off by default | The bundled model is statistically near a constant baseline; users must opt in. |
| 2026-10-27 | Renormalize heuristic weights when ML is off | Keeps scores on `[0,1]` and preserves relative ordering without inventing a replacement term. |
| 2026-10-27 | Skip inference rather than compute-and-discard | The previous gate still paid the ONNX cost; measurement showed 23.2 ms → 0.2 ms. |
| 2026-10-27 | Manual undo suppresses the hero for the match | Prevents a stale GSI snapshot from silently undoing a user correction. An explicit manual action lifts it. |
| 2026-10-27 | Dedupe GSI by hero id, not slot/order | GSI renumbers slots; hero id is the only stable identity in a draft. |
| 2026-10-27 | Cap GSI bans at 5 per side | The payload exposes 5 ban slots and the audited CM order has 5 bans per side; unbounded append across renumbered/stale snapshots corrupted the hero pool. |
| 2026-10-27 | Reset on `HERO_SELECTION` from a non-draft state | Distinguishes a new match from app startup mid-draft and from strategy time. |
| 2026-10-27 | Recommendation cache keyed by full state hash | Lets the overlay reuse the main window's result without a cross-window protocol. |
| 2026-10-27 | Overlay never calls `setFocus()` | An unfocused overlay is better than stealing input from Dota 2. |
| 2026-10-27 | Keep `all_hero_ids = 1..139` | The shipped 887-feature contract and golden fixture depend on it; changing it requires retraining. |
| 2026-10-27 | Do not change scoring weights | P0-1 is a model-quality problem, not a weight-tuning problem; blind reweighting would hide it. |
| 2026-10-28 | Gate `inference` mode, not `as_trained` | The runtime never sees post-draft fields; scoring logged values would overstate the model users can actually run. |
| 2026-10-28 | Three exit codes (0/1/2) | Separates "model is not good enough" (1) from "evaluation cannot be trusted" (2), which need different responses. |
| 2026-10-28 | Block automatic promotion, keep manual opt-in | ML is already off by default; the gate guards the bundled artifact without removing the existing diagnostic opt-in. |
| 2026-10-28 | Do not change the shipped schema to satisfy the gate | Thresholds are a floor for new candidates; mutating the bundled artifact would invalidate the app and the golden tests. |

---

## 5. Files changed in this session

Backend (`apps/desktop/src-tauri/`):

- `src/gsi.rs` — CM order, lifecycle reset, validation, hero-id dedupe, team-id attribution, suppression, draft reconciliation, tests. Session 2: `apply_gsi_payload` seam, per-team ban cap, outcome enum.
- `src/gsi/lifecycle_tests.rs` (new) — 18 payload-level lifecycle/integration tests plus the shared invariant checker.
- `src/engine.rs` — ML gate, weight renormalization, cache, suppression, `total_cmp`, explanations, benchmark, tests.
- `src/engine/simulation.rs` — `total_cmp`, `SearchConfig::sanitized`, ML gate in simulated scoring.
- `src/commands.rs` — cached recommendations, `spawn_blocking`, `set_config` ML application, model-path allow-list, lock order.
- `src/models.rs` — ML default off, corrected model path, config test.
- `src/lib.rs` — startup ML gate, log rotation + test.
- `tauri.conf.json` — CSP.
- `capabilities/default.json` — main window only; `capabilities/overlay.json` — new least-privilege overlay.
- `gen/schemas/capabilities.json` — regenerated by the Tauri build script.

Frontend (`apps/desktop/src/`):

- `services/recommendationFeed.ts` (new) + `services/recommendationFeed.test.ts` (new) — debounce, stale rejection, failure handling.
- `hooks/useRecommendationFeed.ts` (new) — one feed per window.
- `App.tsx` — uses the feed; team changes refresh instead of re-subscribing.
- `components/OverlayWindow.tsx` — isolated prediction, `gsi-team`, no blanking, stale-aware polling.
- `components/Header.tsx` — overlay geometry/enabled, no focus steal.
- `components/SettingsPanel.tsx` — ML default copy, auth warning, dead control removed.
- `stores/index.ts` — ML default off, corrected model path.

ML (`ml/`):

- `feature-engineering/processor.py` — matchup loader accepts numeric and legacy shapes; `resolve_matchup_hero_id`.
- `tests/test_processor_features.py` — resolver + both parquet shapes.
- `training/quality_gate.py` (new) — artifact/contract validation, chronological split checks, fresh evaluation in both modes, calibration/baseline reporting, leakage audit, promotion refusal.
- `training/inference_contract.py` (new) — schema-order encoder and runtime-default feature vector builder used by the gate and schema tests.
- `tests/test_feature_schema.py` (new) — deterministic processor → 887-name contract, matchup extension, and ONNX acceptance.
- `tests/test_quality_gate.py` (new) — gate thresholds, error paths, fingerprint mismatch, leakage audit, promotion blocking.

Scripts:

- `scripts/verify.ps1` — dedicated "Python ML contract + quality-gate tests" step; the remaining Python suite excludes those two modules so each test runs once (13 steps total).

Docs (`docs/`):

- `AUDIT_2026-10-27.md` — P0-1/P1-5/P1-6 remediation addenda; original findings preserved.
- `ROADMAP.md` (this file) — session-3 status, measurements, decisions, verification log.
- `GSI.md` — rewritten to describe the real lifecycle, validation, dedupe, suppression, and turn resolution.
- `ENGINE.md` — ML opt-in, weight renormalization, cache behavior.
- `ML_TRAINING.md` — replaced stale 0.6710 AUC / 896-feature claims with the bundled model's measured metrics, gate usage, and corrected feature inventory.
- `README.md` — rewritten: real setup/verification commands, model limitations, and accurate status.

---

## 6. Verification log

```
# Rust
cargo fmt        --manifest-path apps/desktop/src-tauri/Cargo.toml --all
cargo clippy     --manifest-path apps/desktop/src-tauri/Cargo.toml --all-targets -- -D warnings   # exit 0
cargo test       --manifest-path apps/desktop/src-tauri/Cargo.toml --lib                          # 69 passed, 1 ignored
cargo test       --manifest-path apps/desktop/src-tauri/Cargo.toml --lib lifecycle                # 18 passed
cargo test --release ... bench_recommendation_latency -- --ignored --nocapture                    # see §3

# Frontend (apps/desktop)
npx tsc --noEmit     # clean
npm run lint         # clean
npm test             # 19 passed, 3 files
npm run build        # success

# Python
ml/venv/Scripts/python.exe -m pytest ml/tests -q                                   # 63 passed, 24 warnings
ml/venv/Scripts/python.exe -m pytest ml/tests/test_feature_schema.py ml/tests/test_quality_gate.py -q  # 14 passed
ml/venv/Scripts/python.exe ml/training/train_full.py --evaluate-only               # test AUC 0.5388, Brier 0.2481
ml/venv/Scripts/python.exe ml/training/quality_gate.py                             # exit 1, promotion_eligible=false
```

Session 3 note: the Python suite grew from 49 to 63 tests (14 new contract/gate tests); the full gate gained one visible step and runs each test exactly once. Full gate result is recorded below after the run.

Full gate — **run and passed** at the end of this session:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1
```

Result: `Quality gate PASSED.` with **13/13** steps green (secret scan, Rust format desktop + seeders, Rust clippy desktop + seeders, Rust tests desktop + seeders, frontend lint, frontend typecheck + build, frontend tests, Python byte-compile, Python ML contract + quality-gate tests, Python tests). Counts from the run: Rust 69 passed / 1 ignored (desktop lib), 1 seeder test, frontend 19 tests / 3 files, Python 14 contract/gate tests + 49 remaining = 63 total. `pwsh` is not installed in this environment; use `powershell` as above.

Known warning noise (non-failing): XGBoost emits a CUDA/CPU device-mismatch warning plus UBJSON format warnings from the Python training fixture; these are pre-existing and unrelated to this session's changes.

Empty-result note: `docs/RefactorML.md` is an empty tracked file; it is listed here instead of being deleted (unresolved documentation debt, P2-10).

---

## 7. Open risks (ranked)

1. **Model trust (P0-1 / P1-6):** ML is off by default and automatic promotion is now gated, but a user can still opt in and get a near-baseline 25% term. Fresh inference-mode AUC is 0.5122 with Brier 0.2495, slightly worse than the constant-prior baseline.
2. **Lifecycle confidence (P0-2):** ingestion is now covered by 18 payload-level tests (consecutive matches, duplicates, undo, attribution, 300-payload invariants), but never driven through a real multi-match GSI stream. The warp status mapping and event emission are not executed by tests, and an out-of-order second stale hero-selection snapshot is still indistinguishable from a new draft.
3. **CSP / capability regression (P2-1, P2-2):** the app has not been launched with the new CSP and split capabilities; a broken permission would only appear at runtime.
4. **Semantics (P2-5):** confidence is the score, lane balance is constant, and risk is a score threshold. The UI can still imply more certainty than the engine has.
5. **Reproducibility (P1-5):** the 887-name set is reproduced by a fixture test, but lane features are disabled, the real 1.9 GB dataset was not regenerated, and `process_all()` column order depends on thread completion order. Matchup-data runs emit 894 names and require a retrain.
6. **Latency (P0-4):** 23.2 ms is a harness number; in-app p95 and inference count per event are unmeasured.
7. **Data platform (P2-9):** 1.9 GB of loose dataset files with no versioning or retention.

---

## 8. Next three tasks

1. **Candidate model: leakage ablation and retrain (P0-1, P1-6).** Train on the chronological split without `duration`, `avg_mmr`, `game_mode`, `lobby_type`, and raw pick-index columns; evaluate through `quality_gate.py` in `inference` mode; compare AUC/Brier/ECE to the constant-prior baseline. Promote only on exit 0.
2. **Live GSI replay + Tauri runtime validation (P0-2 residual, P2-1, P2-2, P1-2, P1-3).** Replay a captured multi-match GSI stream through the running app, then launch with the new CSP/split capabilities and confirm overlay show/hide/drag and the missing-model fallback. This is the only way to close the runtime assumptions the tests cannot cover.
3. **Deterministic regeneration + in-app latency (P1-5 residual, P0-4).** Make `process_all()` column order deterministic, regenerate a representative sample, diff names/order against the schema, then measure in-app p95 over ≥100 GSI-driven requests before optimizing further.

### Continuation note (2026-10-28, session 4)

Start with task 1. The gate is in place and the bundled artifact **fails** it: do not lower thresholds, do not edit the shipped schema, and do not mark P0-1 fixed because a gate exists. A passing candidate must beat the constant-prior baseline (Brier < 0.2493 on this test split) as well as the policy floor to be worth bundling. `ml/models/*` is gitignored: gate reports and candidate artifacts stay local; only source/tests/docs are tracked.

Before changing feature code, read `ml/tests/test_feature_schema.py`: 631 structural features have a pinned order, while the two `*_ban_avg_winrate` columns and the 254 per-hero win-rates are history-ordered. `process_all()` uses `as_completed`, so regenerate with a deterministic match order or add an explicit canonical column list. Any schema change requires retraining, re-exporting the schema, updating the golden fixture, and re-running `scripts/verify.ps1`.

---

## 9. Resume instructions

Start here, in this order:

1. Read this file, then §4-§5 of [`AUDIT_2026-10-27.md`](AUDIT_2026-10-27.md) for the evidence behind each ID.
2. Run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1` to confirm the baseline still holds (Rust 69 / frontend 19 / Python 63, 13/13 steps).
3. Run `ml/venv/Scripts/python.exe ml/training/quality_gate.py`; expect exit 1 until a candidate replaces the bundled model. Use `--evaluation-mode as_trained` only to compare against the manifest, not to judge runtime quality.
4. Pick the next task from §8 (currently the candidate-model ablation/retrain). The task ID maps 1:1 to the audit finding, so no re-audit is needed.
5. Update this file's status column and the verification log **after** running the check — never before.
6. Leave changes uncommitted unless the user explicitly authorizes a commit.

Known environment limitations to carry forward: no live Dota 2 client, no Tauri GUI launch in-session, Python dataset regeneration is impractical locally (1.9 GB / 121,746 files), and PowerShell may report a false non-zero exit for `cargo test` because of stderr piping.
