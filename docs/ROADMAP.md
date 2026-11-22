# Roadmap & Live Status: Smart Draft Assistant

- Updated: 2026-11-22 (session 16)
- Source audit: [`AUDIT_2026-11-19.md`](AUDIT_2026-11-19.md) (current pass), [`AUDIT_2026-10-27.md`](AUDIT_2026-10-27.md) (historical findings, preserved)
- Predecessor audit: [`AUDIT.md`](AUDIT.md) (2026-10-22)
- Branch: `master`, HEAD `a2b7bd9`; sessions 14-16 (All Pick drafts, ban suggestions, patch-scoped counts, popularity-first ordering) are committed on top of the session-13 audit
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

## 0. Current state (2026-11-22, session 16)

- HEAD `a2b7bd9`. The All Pick work (hand-recorded drafts, one-action bulk bans
  with undo, configurable ban count), the per-patch ban suggestion list, the
  patch-scoped seed consistency check, and the extended GUI smoke test all
  landed after the session-13 audit note.
- Measured this session by `scripts/verify.ps1`: Rust lib **84 passed** / 1
  ignored, seeders **2 passed**, frontend **32 passed** (5 files), Python **166**
  (14 contract/gate + 152), 14/14 steps, clippy and frontend lint/build clean.
- **The deterministic ordering is popularity-first.** [`DRAFT_SCORER_BACKTEST.md`](DRAFT_SCORER_BACKTEST.md)
  measured the previous context-only ordering at 0.94% top-1 agreement against
  2.21% for a global popularity prior. The shipped score now gives the seeded
  per-patch pick rate 60% and the contextual components 40%
  (`POPULARITY_ORDER_SHARE` in `engine/scorer.rs`). Re-run on the same 20,000
  draft sample: top-1 **2.46%**, top-3 **6.71%**, top-5 **10.49%**, mean
  percentile **0.649** (was 0.94% / 3.19% / 5.98% / 0.527), 0/200 ordering
  mismatches against the list the UI shows.
- The same run is a caution, not a win: the team-level outcome proxy fell from
  0.528 to **0.489** AUC. The priority score is a popularity-led ordering, not a
  win-probability estimate, and the UI must not present it as one.

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
- **Status:** `in_progress` (mitigation `verified_fixed`; automated promotion gate `verified_fixed`; model quality `open` — the ablation candidate was trained, evaluated and **rejected** in session 4)
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
- **Remaining risk:** the shipped model is still untrustworthy. The gate now blocks *automatic* promotion, but a user can still opt in manually. Fixing quality needs better training data, not a threshold change or another feature drop.
- **Session 4 evidence (ablation candidate, one-shot test):** `no_match_context` (881 features) scores test ROC-AUC **0.5136** (CI 0.5024-0.5248), Brier **0.2494**, log loss 0.6919, accuracy 0.5255, ECE 0.0139 — versus the bundled model's 0.5122 / 0.2495 / 0.6922 / 0.5125 / 0.0241 and the constant-prior baseline's 0.2493 / 0.6917 / 0.5274. The gate exits **1** (`policy.roc_auc`, `policy.brier_score`). The candidate is still a constant predictor in disguise and is *worse* than the constant prior on Brier, log loss and accuracy. It was **not** promoted and the bundled artifact is unchanged.
- **Next action (updated 2026-10-28, session 6):** the corrected dataset now exists (`ml/datasets/processed_v2/`, §3), so the blocking prerequisite is cleared. Nothing has been trained on it. The next model claim must use the frozen split in `ml/configs/eval_split_v2.json`, whose holdout is **already inspected** and cannot serve as a fresh final result.
- **Session 6 note:** no model work was performed. Pick/ban encoding is fixed at the data level (5 presence flags for 98.3% of rows, bans populated for 92.1%), which the previous near-baseline results could not be — they were measured on wrong features.

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
- **Status:** `in_progress` (magnitude measured; ablation trained and evaluated; candidate rejected; shipped schema unchanged)
- **Files:** `ml/training/quality_gate.py`, `ml/training/inference_contract.py`, `ml/training/train_full.py`
- **Evidence (original):** `duration`, `avg_mmr`, `game_mode`, `lobby_type`, and raw `rad_pick_i`/`dire_pick_i` are numeric inputs; `duration` is post-draft and replaced by a median at inference.
- **Session 3 measurements (10,213 held-out rows):**
  - Replacing the six match-context fields with their schema defaults costs **0.0266 ROC-AUC** (0.5388 → 0.5122) and **0.0014 Brier** (0.2481 → 0.2495). That is the measured train/serve skew for this artifact, not an estimate.
  - `duration`: univariate AUC **0.4624**, gain rank **6**, 2,974 distinct values. It is a weak, slightly anti-correlated signal here, but the model does rely on it (top-10 gain).
  - `game_mode` univariate AUC **0.5103** (gain rank 327); `lobby_type` **0.5009** (no gain); `avg_mmr` is constant `0` (one value, no gain); `patch` is categorical (gain rank 131) and `patch_version` has no gain.
  - Ordinal pick slots (`rad_pick_0..4`, `dire_pick_0..4`) are near chance individually (AUC 0.4947-0.5027, gain ranks 298-411), so they are low-impact but still inputs the runtime only partially knows.
  - Split integrity is clean: train/validation/test sizes 81,694 / 10,211 / 10,213; zero `match_id` overlap; strict time boundaries (`train.max` 1,764,427,633 < `validation.min` 1,764,427,877; `validation.max` 1,775,425,453 < `test.min` 1,775,425,870). The gate checks all of this on every run.
- **Why untouched:** requires retraining plus held-out evaluation; changing the schema without that would invalidate the bundled model and the 887-feature contract.
- **Session 4 ablation (validation, 10,211 rows, inference mode):** removing the six match-context fields collapses the train/serve gap to **0.0000 AUC** (`as_trained` 0.5358 == `inference` 0.5358) and halves ECE 0.0245 → **0.0151**, for AUC 0.5340 → 0.5358 and Brier 0.2486 → 0.2481. Removing the ten ordinal pick slots slightly *lowers* full-draft AUC (0.5340 → 0.5332) but improves the earliest draft stage (0 known picks: 0.5239 → 0.5293; combined ablation 0.5315). All AUC differences are inside overlapping 95% CIs, so none is significant on its own. The rationale is now structural, not metric-driven: the removed fields are ones the runtime substitutes.
- **Ordinal pick slots are worse than useless, not merely useless:** their schema defaults are *real hero ids* (66, 64 in the exported schema), so an open slot is indistinguishable from a genuine pick of that hero while the presence vector says the hero is absent. That contradiction is what the partial-draft table in §3 measures.
- **Next action:** the ablation is complete and the candidate failed the gate. Do not re-run it on the same data. Regenerate the dataset first (the presence vector is corrupted for the majority of rows), then repeat this experiment on clean features.

### P2 — medium impact

| ID | Finding | Status | Evidence / next action |
| --- | --- | --- | --- |
| P2-1 | `csp: null` | `fixed_unverified` | Explicit CSP in `tauri.conf.json`; needs a real app launch (assets, IPC, dev server) to prove nothing breaks. |
| P2-2 | Overlay has main-window privileges | `fixed_unverified` | New least-privilege `capabilities/overlay.json`; `default.json` restricted to `main`. Needs launch validation that drag/hide still work. |
| P2-3 | Arbitrary model path (DLL sideloading) | `fixed_unverified` | `canonical_model_path()` requires an existing file under app-data, resource dir, repo `ml/models`, or the dev fallback. No test yet. |
| P2-4 | GSI auth optional | `closed_by_design` | Auth remains optional but the Settings UI now warns that any local process can post a draft. |
| P2-5 | Misleading confidence / risk / lane semantics | `in_progress` | 2026-11-22: the duplicated "Confidence: N%" line, the score-threshold RISK chip and the constant-0.5 Lane Balance panel are gone; the card labels the number "Priority". Still open: the category labels are semantics thresholds, and `lane_balance` stays constant `0.5` in the payload (no longer rendered). The backtest re-run is why this is not cosmetic (ROADMAP §3a). |
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

### Feature ablation (2026-10-28, session 4)

- Harness: [`ml/training/ablation.py`](../ml/training/ablation.py); report `ml/models/ablation/ablation_report.json` (gitignored).
- Command: `ml/venv/Scripts/python.exe ml/training/ablation.py` — **exit 0**, 282.9 s, 4 variants.
- Fixed across variants: `train_full.WIDE_CONFIG` (depth 4, 2,500 cap, early stopping 200), seed 42, CPU `hist`, 8 threads. Early stopping on validation, then a refit at the selected round count so the exported ONNX contains exactly the trees that were evaluated (ONNX parity max |diff| 1.2e-07).
- Split: unchanged chronological 80/10/10 on `start_time`; 81,694 / 10,211 / 10,213; zero `match_id` overlap.
- Selection used **validation only** (inference ROC-AUC, 0.002 parsimony window). The test partition was scored once afterwards, by `quality_gate.py`.

Validation results (10,211 rows, `inference` mode; constant train prior 0.5245 scores Brier 0.2489 / log loss 0.6910):

| Variant | Features | ROC-AUC (95% CI) | Brier | Log loss | ECE | `as_trained` AUC | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (all shipped) | 887 | 0.5340 (0.5229-0.5452) | 0.2486 | 0.6904 | 0.0245 | 0.5508 | 641 |
| No match context | 881 | **0.5358** (0.5247-0.5470) | **0.2481** | **0.6893** | **0.0151** | 0.5358 | 376 |
| No pick order | 877 | 0.5332 (0.5221-0.5444) | 0.2487 | 0.6905 | 0.0248 | 0.5493 | 608 |
| Combined | 871 | 0.5344 (0.5232-0.5456) | 0.2481 | 0.6894 | 0.0203 | 0.5344 | 420 |

Partial-draft simulation on validation (AUC when only N picks per side are known; unknown pick slots take the schema default and the unknown heroes leave the presence vector):

| Variant | 0 | 1 | 2 | 3 | 5 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.5239 | 0.5267 | 0.5285 | 0.5300 | 0.5340 |
| No match context | 0.5209 | 0.5230 | 0.5299 | 0.5316 | 0.5358 |
| No pick order | 0.5293 | 0.5301 | 0.5303 | 0.5309 | 0.5332 |
| Combined | **0.5315** | **0.5326** | **0.5328** | **0.5332** | 0.5344 |

What the experiment establishes:

1. **The context features are pure train/serve skew.** With them the model loses 0.0167 AUC between training-time values and runtime defaults; without them the two modes are identical by construction (nothing to substitute). This is the only variant that halves ECE.
2. **Removing ordinal pick slots helps exactly where it should.** At the first-pick stage the baseline degrades to 0.5239 while the combined ablation holds 0.5315 (+0.0076). At full draft they are harmless. The cost is that their defaults are real hero ids, so an open slot impersonates a drafted hero.
3. **No variant produces useful discrimination.** Best validation AUC is 0.5358 with a 95% CI half-width of 0.011; the 0.60 policy floor is ~11 standard errors away. Candidate selection here is choosing among statistically indistinguishable near-baselines.

Single test result (10,213 rows, `quality_gate.py`, **exit 1**), evaluated once after selection was frozen:

| Metric | `no_match_context` candidate | Bundled model (`inference`) | Constant train prior |
| --- | --- | --- | --- |
| ROC-AUC | 0.5136 (0.5024-0.5248) | 0.5122 (0.5010-0.5234) | 0.5 |
| Brier | 0.2494 | 0.2495 | **0.2493** |
| Log loss | 0.6919 | 0.6922 | **0.6917** |
| Accuracy @ 0.5 | 0.5255 | 0.5125 | **0.5274** |
| ECE | **0.0139** | 0.0241 | — |

`as_trained` and `inference` are numerically identical for the candidate (0.5136 / 0.2494 in both), confirming the substitution gap is structurally gone. Validation → test drops 0.0222 AUC, so the validation window (mostly patch 7.40 plus a 7.39 tail) is not representative of the test window (7.40/7.41). **This test partition has now been consumed once** for model selection; further candidate iteration needs a new untouched evaluation window.

Decision: **not promoted**, bundled model unchanged, thresholds unchanged.

### Data-integrity finding behind these numbers (2026-10-28, session 4)

The processed dataset `draft_features.parquet` was built from STRATZ payloads (all 4,000 sampled raw files use `pickBans` with camelCase `isPick`), but `processor.py` read only `pb.get('is_pick', True)`. For every STRATZ match the key is absent, so the default `True` classified **bans as picks**:

- 82,739 of 102,118 rows carry **12** `rad_has_*` flags instead of 5; only 15,841 rows carry the correct 5.
- `rad_ban_*` is all-zero for 97,402 rows, because the banned heroes never reached the ban sets.
- `rad_pick_0..4` therefore holds the first five entries by draft order, which in Captain's Mode mixes bans into the pick slots.

The Rust runtime builds exactly 5 picks and up to 5 bans per side, so the shipped model was trained on a *different* presence distribution than it is served. This is larger than any group ablated above and is **not** fixed by feature removal: no ablation can recover information that was never encoded correctly. Parsing is fixed in `processor.py` (session 4) with a STRATZ-shape regression test, but the 1.9 GB / 111,947-file dataset was **not** regenerated, so every metric in this document still reflects the corrupted encoding.

Also confirmed: `patch_version` is constant `'unknown'` for all 81,694 training rows (learnable signal: none) while the schema default rewrites it to `'7.41'` at inference, and `avg_mmr` is constant `0`.

### Dataset regeneration (2026-10-28, session 6)

`processor.py` was made deterministic and streaming, then the raw corpus was
reprocessed into a **new** directory. The v1 parquet and the bundled model were
left untouched.

| Item | v1 | v2 |
| --- | --- | --- |
| Path | `ml/datasets/processed/` | `ml/datasets/processed_v2/` |
| SHA-256 | `264a6ad5…1621` | `472a5086…b6ce7` |
| Size | 20,941,113 B | 56,332,285 B |
| Shape | 102,118 × 890 | 102,118 × 890 (882 non-meta features) |
| Feature fingerprint | n/a (no schema record) | `9b04edda…b2599` |
| Rows with 12 presence flags per side | 82,739 | **0** |
| Rows with 5 presence flags (Radiant) | 15,841 | **100,369** (98.3%) |
| `rad_ban_*` all-zero rows | 97,402 | **8,031** (7.9%) |

Generation: `processor.py --output-dir datasets/processed_v2 --lookup-dir
datasets/processed --no-allpick --no-matchups` over 111,945 raw files → 102,118
rows in ~4 min; 8,929 `pickBans: null` and 866 short drafts rejected by the
documented rules.

**Determinism (measured, not assumed).** Two full runs with different workers
(8 vs 3) and chunk sizes (2,000 vs 777) produced identical columns, order, dtypes
and values; a content fingerprint over all 890 columns is
`e742e24b45572fcafba38bc4ecb06192c22836da8b44af777d1d389cd9171f9f` for both.
File bytes still differ because chunk size sets the parquet row-group layout
(56 vs 145 groups) — never compare raw file hashes across chunk settings.

**Source validation.** 450 matches re-parsed independently from raw JSON matched
the dataset's four pick/ban blocks exactly (0 mismatches).

**Newly visible source defects (not introduced by the fix):** 3,870 rows (3.8%)
contain a hero that is both picked and banned (session 7 confirmed there are *no*
rows with a repeated pick slot). Separately, 1,748 rows (1.7%) contain a picked
hero with id > 139, which has no one-hot column — a coverage gap, not a
duplicate; the session-6 note that those rows repeated a hero was wrong. Rows are
kept and counted; the ablation reports both populations.

### Frozen evaluation split for v2

`ml/configs/eval_split_v2.json`:

| Partition | Rows | Window (UTC) | Use |
| --- | --- | --- | --- |
| train | 81,695 | 2017-05-07 → 2025-11-29 | feature selection, fitting |
| validation | 10,212 | 2025-11-29 → 2026-04-05 | model selection, early stopping |
| test | 10,211 | 2026-04-05 → 2026-08-28 | reference only |

All 10,211 holdout matches are inside the session-5 test window that was already
scored, so this holdout is **not untouched**. A fresh final number requires
matches newer than `2026-08-28T07:44:36Z`.

### Ablation on the corrected dataset (2026-10-28, session 7)

Four variants × two populations, fixed config/seed (XGBoost depth 4, lr 0.02,
2,500 cap, early stop 200, CPU, seed 42), frozen split, **test partition never
scored**. Full tables: [`ML_TRAINING.md`](ML_TRAINING.md) §3b.

Population A — all 102,118 rows (train 81,695 / validation 10,212):

| Variant | Feat. | ROC-AUC (95% CI) | Brier | ECE | Acc |
| --- | --- | --- | --- | --- | --- |
| baseline | 887 | 0.5454 (0.5342-0.5565) | 0.2478 | 0.0267 | 0.5370 |
| no match context | 881 | 0.5444 (0.5333-0.5556) | 0.2475 | **0.0172** | **0.5463** |
| no pick order | 877 | 0.5471 (0.5360-0.5583) | 0.2478 | 0.0273 | 0.5388 |
| **combined (selected)** | 871 | **0.5481** (0.5369-0.5592) | **0.2473** | 0.0177 | 0.5402 |

Population B — 3,870 duplicate-hero rows excluded (train 79,614 / validation 9,145):
baseline 0.5466 / 0.2480, no-context 0.5485 / 0.2473, no-pick-order 0.5472 / 0.2477,
**combined 0.5498 / 0.2473** (CI 0.5381-0.5616). Constant train prior on the same
rows: Brier 0.2489 / 0.2490, log loss 0.6910 / 0.6911.

Partial draft (AUC, picks known per side 0/1/2/3/5): baseline
0.529/0.529/0.531/0.538/0.545; combined 0.527/0.527/0.532/0.536/0.548.

What it establishes:

1. Encoding was worth ~+0.011 to +0.014 AUC (v1 baseline 0.5340, v1 best 0.5358)
   and is the first run where every variant beats the constant prior on Brier and
   log loss.
2. It is still ~9 standard errors short of the 0.60 policy floor; all variant CIs
   overlap by >90%. The winner follows the pre-registered rule, not a measurable
   difference.
3. Removing pick order now *hurts* the first-pick stage by ~0.008 AUC — the
   reverse of the v1 finding, because the slots no longer encode masked bans.
4. Excluding duplicate-hero rows does not change the winner (+0.0001..+0.0040).

Two harness defects were found and fixed before the runs counted: the frozen
split config was not actually applied (the harness re-cut 80/10/10), and the
partial-draft simulation kept win rates, aggregates, semantics and synergy from
the *complete* draft, so it leaked future picks. Both are now tested.

### Model-family comparison on the corrected dataset (2026-10-29, session 8)

Three families, same 871-feature set, same frozen split, same partial-draft
frames, seeds 42/43/44, **validation only — the test partition was never read**.
Full tables: [`ML_TRAINING.md`](ML_TRAINING.md) §3c. Reports:
`ml/models/families/family_comparison_all_rows.json` and `…_no_duplicate_heroes.json`.

| Family | AUC (A, all rows) | Brier | ECE | Fit | AUC (B, no dups) |
| --- | --- | --- | --- | --- | --- |
| `xgboost_combined` (reference) | **0.5479** ±0.0004 | **0.2473** | 0.0181 | 136.9 s | **0.5481** ±0.0012 |
| `logistic_l2` (C = 0.01) | 0.5454 ±0.0000 | 0.2488 | 0.0346 | **4.3 s** | 0.5441 |
| `hist_gbdt` | 0.5380 ±0.0026 | 0.2479 | **0.0158** | 21.9 s | 0.5370 ±0.0009 |
| constant train prior | 0.5000 | 0.2489 | — | — | 0.2490 |

Partial draft (AUC, picks known per side 0/1/2/3/5): XGBoost
0.529/0.526/0.532/0.537/0.548; logistic **0.528/0.537/0.537/0.542**/0.545;
hist GBDT 0.526/0.516/0.524/0.532/0.538.

What it establishes:

1. **Family is not the bottleneck.** Three different learners within 0.01 AUC of
   each other, all near the constant baselines, none within reach of 0.60.
2. **The linear model beats the trees where recommendations are produced.** At
   one to three known picks per side it is +0.004 to +0.011 AUC ahead; XGBoost
   only leads at five picks (+0.003). Feature interactions pay off after the
   draft is mostly known, which is the least useful moment.
3. XGBoost is reliably better calibrated on full drafts (paired-bootstrap Brier
   difference +0.0015 for logistic, CI excludes zero); the AUC difference is not
   distinguishable (CI crosses zero). Nothing qualifies as a material
   improvement, so the reference is retained by the pre-registered rule.
4. All three fail the unchanged numeric floor on validation (AUC 0.538-0.548 vs
   0.60; Brier 0.2473-0.2491 vs 0.235). Best candidate is a failing candidate.
5. ONNX parity verified for XGBoost (2.98e-07) and logistic (2.47e-07) on full
   draft and 0/1/3-pick frames; `hist_gbdt` is not exportable through skl2onnx.
   Nothing touched the bundled artifact.

### Objective feasibility study (2026-10-29, session 9)

Research only: no model fitted for selection, nothing promoted, test partition
not read. Full tables in [`ML_TRAINING.md`](ML_TRAINING.md) §3d.

```
ml/venv/Scripts/python.exe ml/training/objective_diagnostics.py   # exit 0, ~40 s
  # -> ml/models/objective_study/diagnostics.json
```

| Finding | Number |
| --- | --- |
| Population that is actually Captains Mode | **92,221 rows (90.3%)**, not All Pick |
| Per-mode AUC (CM / Captains Draft / OpenDota CM) | 0.5480 / 0.5666 / 0.5336 |
| Early-stage AUC with all bans (harness convention) | 0.527 at 0 picks, 0.525 at 1 |
| Early-stage AUC with only bans known at that stage | **0.523 / 0.522** (-0.0044 / -0.0029) |
| Spread of predictions at the first pick | 5.9 pp (p05-p95) |
| 2-a-side draft contexts with >= 50 matches | **0** (101,400 distinct) |
| Full drafts seen more than once | 38 distinct (each twice) |
| Per-hero win-rate precision | median 3,249 picks, +/-1.7 pp, SNR 2.0 vs binomial noise |
| Validation AUC CI half-width | 0.0111 |

What it establishes:

1. **The data was never the wrong mode.** 90% of the corpus is Captains Mode, so
   the product-alignment worry is answered — but `game_mode` is mis-encoded in the
   dataset, which is why earlier sessions read it as All Pick (see AUDIT P1-7).
2. **The last optimistic early-stage number was partly future information.** The
   harness held all 7 bans per side at every stage; the real CM order reveals 3
   before the first pick and 5 by the third. Correcting that costs 0.0044 AUC at
   zero picks and 0.0029 at one — roughly a third of the signal those stages had.
3. **Full-match win probability is exhausted as a headline metric.** 0.548 with a
   CI half-width of 0.011 cannot reach 0.60 by tuning; the gap is ~5 SE.
4. **Next-pick ranking is not identifiable from this corpus.** No 2-a-side context
   repeats 50 times and only 38 full drafts repeat at all, so there is no
   counterfactual evidence for "this pick is worth +X%".
5. **A feasible decision-shaped experiment exists**: next-pick *imitation* (which
   hero a strong captain took, among the legal set), scored by top-k accuracy on
   held-out drafts. It needs no counterfactuals and tests whether draft state
   carries decision signal at all.

### Next-pick imitation experiment (2026-10-29, session 10)

Research only: no production inference change, nothing promoted, test partition
never parsed. Full tables in [`ML_TRAINING.md`](ML_TRAINING.md) §3e.

```
ml/venv/Scripts/python.exe ml/training/imitation.py      # exit 0, 486 s
  # -> ml/models/imitation/imitation_report.json
```

| Result | Number |
| --- | --- |
| Chronology | `order` dense+unique in every eligible CM payload; 94% one 24-action template; 0/60 slot mismatches vs the parquet |
| Population | 81,701 of 91,907 candidate matches; 745,930 train / 71,080 validation states (10 per match) |
| Excluded | 10,206: 3,022 null `order`, 2,005 malformed flags, 1,557 hero >139, 2,791 unsupported mode, 705 no STRATZ mode, 126 too many actions |
| Model vs stage/side frequency (top-1) | **+0.0146** (+0.0128..+0.0165), 100% of bootstrap resamples positive |
| Model vs its own intercept (top-1) | **+0.0249** (+0.0230..+0.0267) |
| First pick top-1 (base / model / intercept-only) | **0.0990 / 0.0744 / 0.1003** |
| From pick 2 on | model ahead at every pick; pick 7 0.0550 vs 0.0113 |

What it establishes:

1. **The state contains real next-pick information once picks exist.** The gain is
   large relative to the baselines (top-1 +36%, top-3 +37%) and survives a
   match-level cluster bootstrap.
2. **It is not there at the opening pick.** With seven bans and no picks, the
   model is *worse* than a frequency table (0.0744 vs 0.0990) while its own
   intercept alone matches it (0.1003) — the ban features actively mislead at
   that state. Early-draft usefulness is therefore still unproven.
3. **Capacity, not the question, was the first obstacle.** Two measured design
   corrections were needed: a per-phase hero intercept (six always-on columns),
   and pooling removal effects across phases (1,668 → 562 features). On a
   4,000-match smoke run the first version scored 0.0509 top-1 at the opening
   pick; the final 562-column configuration reports 0.0744 on full data.
4. **Imitation ≠ value.** The target is what a captain did, not what wins. No
   win-rate or counterfactual claim follows from these numbers.

### Phase-aware imitation experiment (2026-10-29, session 11)

Research only: no production change, nothing promoted, test partition never
parsed. Full tables in [`ML_TRAINING.md`](ML_TRAINING.md) §3f.

```
ml/venv/Scripts/python.exe ml/training/imitation_phase.py --bootstrap 1000
  # exit 0, 356 s
  # -> ml/models/imitation/phase/report.json (+ decision_rule.json, pre-registered)
```

| Strategy | pick 0 | pick 1 | picks 2-7 | picks 8-9 | all |
| --- | --- | --- | --- | --- | --- |
| stage/side frequency | 0.0990 | 0.0689 | 0.0241 | 0.0461 | 0.0405 |
| pooled model (session 10) | 0.0744 | 0.0457 | 0.0490 | 0.0685 | 0.0551 |
| **hybrid (freq 0-1, model 2-9)** | **0.0990** | **0.0689** | **0.0490** | **0.0685** | **0.0599** |
| early model, pick 0 only | 0.0575 | 0.0457 | 0.0490 | 0.0685 | 0.0534 |
| early model, pick 1 only | 0.0744 | 0.0629 | 0.0490 | 0.0685 | 0.0568 |
| early model, picks 0-1 shared | 0.0640 | 0.0563 | 0.0490 | 0.0685 | 0.0551 |
| model intercept only (diagnostic) | 0.1003 | 0.0716 | 0.0204 | 0.0039 | 0.0302 |

What it establishes:

1. **A trained early-stage model does not help.** Pick-0-only 0.0575 vs 0.0990
   table (Δ −0.0416, CI −0.0492..−0.0339); picks-0-1 shared 0.0601 vs 0.0840
   (Δ −0.0239, CI −0.0291..−0.0187); pick-1-only is closest and still inconclusive
   (Δ −0.0061, CI −0.0124..+0.0007). None qualifies under the pre-registered rule.
2. **The opening-pick information is the prior.** The model's intercept alone is
   level with the table at picks 0-1 (Δ +0.0012, CI −0.0018..+0.0041) but collapses
   to 0.0039 by picks 8-9, so the removal terms carry the later-stage signal and
   only noise at the opening.
3. **The hybrid is the best configuration so far**: 0.0599 overall versus 0.0551
   pooled and 0.0405 table, keeping the pooled gains at picks 2-7 (+0.0249,
   +0.0225..+0.0270) and 8-9 (+0.0223, +0.0175..+0.0274) intact. It is routing, not
   a new model.
4. **Holdout transfer failed for the early model** (0.0994 inner vs 0.0575
   validation), so internal-holdout early gains should not be trusted.
5. **Imitation ≠ value.** No win-rate, quality or counterfactual claim.

### Product-utility spike (2026-10-29, session 12)

Research only: no production change, no export, no promotion, test partition not
parsed. Full detail in [`HYBRID_PRODUCT_SPIKE.md`](HYBRID_PRODUCT_SPIKE.md).

```
ml/venv/Scripts/python.exe ml/training/hybrid_presentation.py   # exit 0, 98.6 s
  # -> ml/models/hybrid_spike/{mockup.html, snapshots.json, report.json}
```

| Result | Number |
| --- | --- |
| Hit rates reproduced (hybrid) | 0.0990 / 0.0689 / 0.0490 / 0.0685 / 0.0599 by stage; all identical to session 11 |
| #1 removed by the very next action | 8.0% overall; 20.2% at the 7 -> 8 ban block; 4.4-5.2% on one-removal transitions |
| #1 still first among survivors | 55.7% mean; 97.1% at 0 -> 1; 11.9% at the 1 -> 2 source switch |
| Ranking churn (Kendall tau, 2,000-draft sample) | 0.958 at 0 -> 1, 0.85-0.87 mid, 0.632 at 1 -> 2 |
| Contextual vs frequency list | top-1 agreement 31.1% overall (15.1% at picks 2-7, 10.1% at 8-9); tau 0.689 |
| Single-state inference | 0.135 ms; 0.56 MB of coefficients; 4.0 us/state batched |
| Mockup sample | 6 uniform drafts x picks 0/2/5; top entry matched the observed pick in 2 of 18 (no cherry-picking) |

What it establishes:

1. **The surface can be honest.** Ordinal five-hero lists, a source badge, removals
   named instead of hidden, no scores, no percentages, no confidence, no risk chip.
   The generator aborts if any forbidden claim phrase appears in the document.
2. **Churn tracks information.** One ban moves the order mildly (tau ~0.86); the big
   reshuffles coincide with the four-to-five-hero ban blocks and the 1 -> 2 source
   switch, which a UI can explain.
3. **The contextual list is not a re-skin** (85-90% top-1 disagreement from pick 2),
   but at 4.9-6.9% top-1 over ~124 heroes the argmax is weakly identified, so much
   of the disagreement is near-ties.
4. **Next step is a usability gate, not more modelling**: 5-8 players, three drafts,
   blinded frequency-vs-contextual pairs, with a scored comprehension check as the
   go/no-go. No win-rate claim is possible from it.

### Recommendation benchmark

- Test: `bench_recommendation_latency` in `src-tauri/src/engine.rs` (ignored by default).
- Command: `cargo test --release --manifest-path apps/desktop/src-tauri/Cargo.toml --lib bench_recommendation_latency -- --ignored --nocapture`
- Setup: 120 synthetic heroes with seeded per-patch pick rates (so popularity has a real spread), bundled `draft_model_xgboost_full.onnx` (887 features), empty matchup/synergy maps, 3 iterations after one warm-up, single Windows machine (no CPU pinning).
- Results (re-measured 2026-11-22 after batched inference):

| Path | Release | Debug |
| --- | --- | --- |
| ML enabled, full recommendation | 1.4 ms/request | 29.0 ms/request |
| ML disabled, full recommendation | 0.2 ms/request | 1.5 ms/request |
| Cache hit (`get_recommendations_cached`) | 0.002 ms/request | 0.011 ms/request |
| Single `MLScorer::score` (one candidate) | 0.064 ms | 0.944 ms |

Interpretation: the ML-on path builds one feature map per hero that can still reach the returned positions, lets the schema's 0.0 defaults stand in for absent presence one-hots instead of writing four zero keys per known hero, and scores that set in one ONNX call. That is why a single score is 0.064 ms instead of 0.144 ms and the full recommendation is 1.4 ms instead of 23.2 ms. Inference is skipped entirely when ML is disabled, and the cache still makes the second window's identical request effectively free. Debug builds are ~15× slower — do not quote debug numbers as user-facing latency.

### Shipped ordering: popularity-first (2026-11-22)

- Policy: `POPULARITY_ORDER_SHARE = 0.60` — the seeded per-patch pick-rate
  percentile carries 60% of the score, the five contextual components 40%.
- Harness: same 20,000-draft sample, same seeded database and flags as the
  committed run; the run writes a local report to
  `ml/backtest/results/backtest_popularity_first.json` and is not committed.
- The pick-rate prior is the full-corpus seeded table, so like the shipped ban
  table it is in-sample with respect to the replay; it is the ranking the app
  actually serves, not an out-of-sample estimate.

| Agreement (evaluation slice, 79,764 pick events) | before | after | global prior |
| --- | --- | --- | --- |
| top-1 | 0.94% | **2.46%** | 2.21% |
| top-3 | 3.19% | **6.71%** | 5.98% |
| top-5 | 5.98% | **10.49%** | 9.75% |
| mean rank of pick | 53.2 | **39.7** | 40.2 |
| mean percentile | 0.527 | **0.649** | 0.644 |

What moved the other way:

| Metric (whole sample) | before | after |
| --- | --- | --- |
| team-level AUC (sum of per-pick scores) | 0.528 | **0.489** |
| rank-1 bucket win rate | 49.7% (CI 47.5-52.0) | **51.7%** (CI 50.1-53.3) |

Interpretation: the popularity-first order tracks what captains do far better
and no longer has a rank-1 bucket that contains 50%, but summing those priority
scores no longer predicts the winner (it was already below the report's 0.55
usability floor). Agreement with human picks is the measured objective; win
probability is not. The rolling per-patch prior remains the stronger
out-of-sample reference (3.38% top-1) and is not what the app serves.

### Search benchmark (2026-11-22)

- Test: `bench_search_recommendation_latency` in `src-tauri/src/engine.rs` (ignored by default).
- Command: `cargo test --release --lib bench_search_recommendation_latency -- --ignored --nocapture`
- Setup: 120 synthetic heroes with seeded per-patch pick rates, bundled model, beam width 5, candidates 15, 3 iterations after one warm-up, same machine as the recommendation benchmark.

| Path | Release ML off | Release ML on | Debug ML off | Debug ML on |
| --- | --- | --- | --- | --- |
| shortlist only (`max_depth: 0`) | 0.3 ms | **2.8 ms** | 1.9 ms | 32.2 ms |
| lookahead depth 2 | 0.4 ms | **4.1 ms** | 2.8 ms | 54.8 ms |
| recommendation list (reference) | 0.1 ms | 1.4 ms | 0.9 ms | 18.3 ms |

The debug column is the same test without `--release`: ~12x release, so only
the release numbers describe user-facing latency. It is recorded because the
GUI smoke test re-checks these two paths in the running debug app.

Interpretation: ML-off search is unchanged. ML-on search now pays one batched inference per pool it needs instead of one inference per candidate: the root shortlist and the opponent shortlist are scored once per turn (the opponent beam is shared by all 15 branches, filtered to heroes still available), and only heroes that can still reach the requested positions are scored at all. ML adds at most `(1 - POPULARITY_ORDER_SHARE) * W_ML` = 0.10 to a total, so a hero more than 0.10 below the cut cannot enter it; on this seeded fixture that leaves about 35 of 120 heroes per pool. Lookahead is 4.1 ms, 77× faster than the 318 ms pre-batching measurement and close to the ML-off cost. Pruning is a provable bound rather than a heuristic, so the shortlist order is unchanged: `test_ml_pruning_preserves_the_top_of_the_pool` checks the pruned top against a full-pool sort, and the preselection-order test still passes. The fixture now seeds pick rates because without a spread every hero ties and nothing can be pruned (the unseeded worst case measured 7.9 ms shortlist / 15.2 ms lookahead before seeding).

**In-app check.** `scripts/ui_smoke.ps1` times these two configs over IPC after
the GSI draft arrives, toggling `ml_enabled` through `set_config` exactly as the
Settings panel does (one warm-up, then 5 shortlist / 3 lookahead calls, median
reported). Against the seeded 126-hero database (9052 matchups, 7381 synergies)
the debug app measured 6.5-10.3 ms ML-off / 41.4-48.0 ms ML-on for the shortlist
and 8.2-11.3 / 59.9-67.0 ms for the lookahead across four runs — 1.2-2.9x the
build-scaled fixture, the difference being the IPC round trip and the
matchup/synergy data the fixture leaves empty; divided by the ~12x debug factor
that is the release 2.8 / 4.1 ms again.
The check scales the release benchmark by 12 for the dev build, writes
`recommendation-latency.json` beside the screenshots, puts the same table in the
GitHub Actions job summary with a pass/drift/fail marker per row, prints `DRIFT` past 4x
that expectation and fails past 10x, so the pre-batching 318 ms lookahead would
have turned the smoke red.

### Not measured

- Model quality on a fresh held-out set: **measured twice** — the bundled artifact (§ above) and the session-4 ablation candidate (single test run, §3). Neither passes the gate; the candidate was rejected.
- In-app p95 latency (the GUI smoke test records point latency for the search
  path in the debug build), UI-thread responsiveness, or OGSI event rate.
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
| 2026-10-28 | Select ablation variants on validation only | Scoring four candidates on the test partition and keeping the best would make the reported test number an optimistically biased selection statistic. The test set was scored once, for the pre-committed winner. |
| 2026-10-28 | Export the candidate with validation metrics only | A manifest `test` block would have required touching test labels during the experiment. The gate reports test metrics instead, so parity is recorded as "no manifest metric" rather than hiding a selection leak. |
| 2026-10-28 | Refit at the selected round count before exporting | XGBoost predicts with `best_iteration` after early stopping but the ONNX converter exports the whole booster; exporting the probe would have shipped trees never evaluated. |
| 2026-10-28 | Reject the ablation candidate | Test AUC 0.5136 / Brier 0.2494 fails the policy and is worse than the constant prior on Brier, log loss and accuracy. Removing leakage fixed the serving contract, not the prediction. |
| 2026-10-28 | Fix `isPick` parsing but do not regenerate data in-session | The bug invalidates the shipped dataset, but regenerating 111,947 files is a separate, verifiable task; changing parsing without regenerating cannot silently alter any existing metric. |
| 2026-10-29 | Pre-register the phase-aware decision rule in a file | Early-stage deltas are small enough that a rule written after seeing them would be unfalsifiable. The runner refuses to continue if the stored rule changes. |
| 2026-10-29 | Recommend the frequency fallback at picks 0-1, not another early model | Three early-stage configurations all lose to the table at their own stage; continuing to search would be searching after a negative result. |
| 2026-10-29 | Keep the pooled model from pick 2 in the recommendation | Its +0.025/+0.022 top-1 gains at picks 2-7/8-9 are the only part of the imitation line that beats the baseline with an interval excluding zero. |
| 2026-10-29 | Show the ranking as an ordered shortlist with a source badge, no scores | The model is ~6% top-1; any percentage or confidence would imply precision the data does not have. Ordinal ordering plus a tier bar is all that is defensible. |
| 2026-10-29 | Enforce claim rules in the generator, not in review | A single careless label ("best pick", "win probability") would misrepresent the target; the render fails instead. |
| 2026-10-29 | Precompute the mockup, keep it out of the app | A static page answers the presentation question without touching production inference, so no runtime regression is possible. |

---

## 5. Files changed (remediation sessions)

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

- `feature-engineering/processor.py` — matchup loader accepts numeric and legacy shapes; `resolve_matchup_hero_id`. Session 4: `isPick`/`is_pick` ban-vs-pick parsing fix.
- `training/ablation.py` (new) — audited feature-group table, 4-variant ablation, validation-only selection rule, ONNX export + bundle validation, partial-draft simulation.
- `tests/test_ablation.py` (new) — 15 tests: group derivation/order, strict feature selection, derived-schema validity, round-count parity between probe and refit, partial-draft masking, selection tie-breaks, end-to-end export that proves the test partition is never scored.
- `tests/test_processor_features.py` — resolver + both parquet shapes; session 4 adds STRATZ `isPick` pick/ban separation and an OpenDota regression test.
- `training/quality_gate.py` (new) — artifact/contract validation, chronological split checks, fresh evaluation in both modes, calibration/baseline reporting, leakage audit, promotion refusal.
- `training/inference_contract.py` (new) — schema-order encoder and runtime-default feature vector builder used by the gate and schema tests.
- `tests/test_feature_schema.py` (new) — deterministic processor → 887-name contract, matchup extension, and ONNX acceptance.
- `tests/test_quality_gate.py` (new) — gate thresholds, error paths, fingerprint mismatch, leakage audit, promotion blocking.
- `training/ablation.py` (new, session 7) — frozen-split loader, audited feature groups, duplicate-hero mask, runtime-faithful partial-draft rebuild.
- `training/model_families.py` (new, session 8) — validation-only family comparison, paired bootstrap, candidate ONNX export with parity.
- `training/objective_diagnostics.py` (new, session 9) — true game-mode audit from raw payloads, per-mode metrics, ban-timing rebuild, context/hero-effect identifiability. Writes only to the gitignored `ml/models/objective_study/`.
- `training/imitation.py` (new, session 10) — next-pick imitation: raw action cache, decision states, legal-candidate masking, frequency baselines, multinomial logistic model, match-level bootstrap. Writes only to the gitignored `ml/models/imitation/`.
- `tests/test_ablation.py`, `tests/test_model_families.py`, `tests/test_dataset_regeneration.py` (new) — split/feature/partial-draft, family/preprocessing/export, and STRATZ pick-ban regression coverage.
- `tests/test_objective_diagnostics.py` (new, session 9) — ban-timing rule and timed stage rebuild.
- `training/imitation_phase.py` (new, session 11) — phase-aware comparison: pre-registered decision rule, early-stage matrices, hybrid routing, per-stage paired bootstrap. Writes only to the gitignored `ml/models/imitation/phase/`.
- `tests/test_imitation.py` (new, session 10) — chronology, future-information leakage, legal filtering, train-only baselines, tie-breaking, metric and bootstrap math.
- `tests/test_imitation_phase.py` (new, session 11) — pre-registration immutability, early-matrix blocks and widths, prefix-only invariance, routing, stage masks and aggregation.
- `training/hybrid_presentation.py` (new, session 12) — product-utility spike: refits the frozen configuration for per-state scores, measures churn, renders a self-contained HTML mockup with claim guards. Writes only to the gitignored `ml/models/hybrid_spike/`.
- `tests/test_hybrid_presentation.py` (new, session 12) — routing boundary, tie-breaking, survivor ranks, Kendall tau, synthetic churn, deterministic sampling, hero display names, forbidden-claim guards.

Scripts:

- `scripts/verify.ps1` — dedicated "Python ML contract + quality-gate tests" step; the remaining Python suite excludes those two modules so each test runs once (13 steps total).

Docs (`docs/`):

- `AUDIT_2026-10-27.md` — P0-1/P1-5/P1-6 remediation addenda; original findings preserved.
- `ROADMAP.md` (this file) — session-3 status, measurements, decisions, verification log.
- `GSI.md` — rewritten to describe the real lifecycle, validation, dedupe, suppression, and turn resolution.
- `ENGINE.md` — ML opt-in, weight renormalization, cache behavior.
- `ML_TRAINING.md` — replaced stale 0.6710 AUC / 896-feature claims with the bundled model's measured metrics, gate usage, and corrected feature inventory.
- `README.md` — rewritten: real setup/verification commands, model limitations, and accurate status.
- `HYBRID_PRODUCT_SPIKE.md` (new, session 12) — audit, prototype, stability, minimal user-feedback protocol, fresh-window requirements.

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

Session 4 verification (this session):

```
ml/venv/Scripts/python.exe ml/training/quality_gate.py --report ml/models/session4_baseline_gate.json
  # exit 1 — ROC-AUC 0.5122, Brier 0.2495, ECE 0.0241, failed: policy.roc_auc, policy.brier_score
ml/venv/Scripts/python.exe ml/training/ablation.py
  # exit 0 — 282.9 s, 4 variants, selected no_match_context (881 features), ONNX parity 1.19e-07
ml/venv/Scripts/python.exe ml/training/quality_gate.py \
  --model-dir ml/models/ablation/draft_model_ablation_no_match_context \
  --model-stem draft_model_ablation_no_match_context \
  --report ml/models/ablation/candidate_gate.json
  # exit 1 — inference ROC-AUC 0.5136, Brier 0.2494, ECE 0.0139, failed: policy.roc_auc, policy.brier_score
ml/venv/Scripts/python.exe -m pytest ml/tests -q
  # exit 0 — 80 passed, 24 warnings
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — Quality gate PASSED, 13/13 steps
```

Session 7 verification (this session, ablation on the corrected dataset):

```
ml/venv/Scripts/python.exe -m pytest ml/tests -q
  # exit 0 — 99 passed (test_ablation 21, test_dataset_regeneration 13)
ml/venv/Scripts/python.exe training/ablation.py \
  --data datasets/processed_v2/draft_features.parquet \
  --split-config configs/eval_split_v2.json \
  --report models/ablation_v2/all_rows_report.json \
  --model-dir models/ablation_v2/candidates_all_rows
  # exit 0 — 427.5 s, 4 variants, selected no_match_context_no_pick_order (871 features)
  #            validation inference AUC 0.5481, ONNX parity 2.4e-07; test not scored
ml/venv/Scripts/python.exe training/ablation.py ... --exclude-duplicates \
  --report models/ablation_v2/no_duplicates_report.json \
  --model-dir models/ablation_v2/candidates_no_duplicates
  # exit 0 — 392.1 s, selected the same variant, validation AUC 0.5498, parity 3.6e-07
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps
```

No `quality_gate.py` run was made for the candidates: that would consume the
already-inspected test window again, and the task explicitly forbids a new test
claim from it. The bundled model is unchanged.

Session 8 verification (this session, model-family comparison):

```
ml/venv/Scripts/python.exe -m pytest tests -q
  # exit 0 — 108 passed (test_model_families 9)
ml/venv/Scripts/python.exe training/model_families.py --population all
  # exit 0 — 588.2 s; XGBoost 0.5479, logistic 0.5454, hist 0.5380
  #            ONNX parity 2.98e-07 / 2.47e-07; hist not exportable; test not scored
ml/venv/Scripts/python.exe training/model_families.py --population no_duplicates
  # exit 0 — 597.8 s; XGBoost 0.5481, logistic 0.5441, hist 0.5370 (9,145 rows)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED." (the gate's tests, not the model)
```

Session 8 counts: Python 99 → **108** (9 new in `test_model_families.py`). The
general Python step runs 94 and the contract/gate step runs 14. No
`quality_gate.py` run was made: it would score the already-inspected test window.
The bundled ONNX artifact, schema, and both datasets are untouched.

`verify.ps1` must be invoked with a forward-slash path (`-File scripts/verify.ps1`);
a backslash-escaped path reaches PowerShell as `scriptsverify.ps1` and fails.

Session 9 verification (objective feasibility study — no model work):

```
ml/venv/Scripts/python.exe -m pytest tests/test_objective_diagnostics.py -q
  # exit 0 — 8 passed
ml/venv/Scripts/python.exe -m pytest tests -q
  # exit 0 — 116 passed, 25 warnings (26.1 s)
ml/venv/Scripts/python.exe training/objective_diagnostics.py --candidate models/objective_study/candidate
  # exit 0 — 39.7 s on the cached mode map (cold first run: ~12 min)
  #   overall validation AUC 0.5481, Brier 0.2473, 10,212 rows
  #   timed bans: 0 picks 0.5228 (-0.0044), 1 pick 0.5219 (-0.0029)
  #   test partition: not read
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED."
```

Session 9 also re-fitted the seed-42 reference once, to replace a stale ONNX
(the session-8 export on disk came from the duplicate-excluded run): AUC
0.5480650, Brier 0.2472861, 1,110 rounds — identical to the recorded values.
Python 108 → **116** (8 new in `test_objective_diagnostics.py`).

Session 10 verification (next-pick imitation — no production change):

```
ml/venv/Scripts/python.exe -m pytest tests/test_imitation.py -q
  # exit 0 — 24 passed
ml/venv/Scripts/python.exe -m pytest tests -q
  # exit 0 — 140 passed, 25 warnings (35.1 s)
ml/venv/Scripts/python.exe training/imitation.py --out models/imitation --refresh-actions
  # exit 0 — 486.2 s; 81,701 eligible matches, 71,080 validation states
  #   model top1 0.0551 vs stage/side 0.0405 (+0.0146, CI +0.0128..+0.0165)
  #   first pick: model 0.0744 vs base 0.0990 (bucket-only 0.1003)
  #   test partition: never parsed
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED."
```

Python 116 → **140** (24 new in `test_imitation.py`). No `quality_gate.py` run: it
would score the already-inspected test window.

Session 11 verification (phase-aware imitation — no production change):

```
ml/venv/Scripts/python.exe -m pytest tests/test_imitation_phase.py -q
  # exit 0 — 10 passed
ml/venv/Scripts/python.exe -m pytest tests -q
  # exit 0 — 150 passed, 25 warnings (28.8 s)
ml/venv/Scripts/python.exe training/imitation_phase.py --bootstrap 1000
  # exit 0 — 356.4 s; pooled model reproduced session 10 exactly (Δ top1 0.0e+00)
  #   hybrid 0.0599 overall vs pooled 0.0551, table 0.0405
  #   early models: pick0 −0.0416, picks01 −0.0239, pick1 −0.0061 (none qualifies)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED."
```

Python 140 → **150** (10 new in `test_imitation_phase.py`). No `quality_gate.py`
run: it would score the already-inspected test window.

Session 12 verification (product-utility spike — no production change):

```
ml/venv/Scripts/python.exe -m pytest tests/test_hybrid_presentation.py -q
  # exit 0 — 13 passed
ml/venv/Scripts/python.exe -m pytest tests -q
  # exit 0 — 163 passed, 25 warnings (26.5 s)
ml/venv/Scripts/python.exe training/hybrid_presentation.py
  # exit 0 — 98.6 s; hit rates reproduce session 11 exactly
  #   #1 removed by next action 0.0805; retained 0.5570; top5 overlap 0.6601; tau 0.8337
  #   single-state predict 0.135 ms; two runs produced byte-identical HTML
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED."
```

Python 150 → **163** (13 new in `test_hybrid_presentation.py`). Rust and frontend
are unchanged. No `quality_gate.py` run: it scores the win-probability target on
an already-inspected window, which is not what this spike measures.

Session 6 verification (session 6, dataset regeneration — no model work):

```
ml/venv/Scripts/python.exe feature-engineering/processor.py \
  --output-dir datasets/processed_v2 --lookup-dir datasets/processed \
  --no-allpick --no-matchups --workers 8 --chunk-size 2000 \
  --dataset-label candidate_v2_pickban_fix
  # exit 0 — 111,945 files, 102,118 rows x 890 cols, ~4 min
ml/venv/Scripts/python.exe feature-engineering/processor.py \
  --output-dir datasets/processed_v2_det --lookup-dir datasets/processed \
  --no-allpick --no-matchups --workers 3 --chunk-size 777
  # exit 0 — same 102,118 x 890; content fingerprint identical to the run above
ml/venv/Scripts/python.exe -m pytest ml/tests -q
  # exit 0 — 93 passed
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
  # exit 0 — 13/13 steps, "Quality gate PASSED." (the gate's tests, not the model)
ml/venv/Scripts/python.exe ml/training/quality_gate.py --report ml/models/session5_gate.json
  # exit 1 — ROC-AUC 0.5122, Brier 0.2495, ECE 0.0241; promotion_eligible=false
```

Session 6 counts: Python 80 → **93** (13 new: determinism/column-order/pick-ban
integrity/schema provenance, frozen-split config contract, frozen-split dataset
reproduction). The general Python step runs 79 and the contract/gate step runs
14, so no module runs twice. Rust and frontend are unchanged.

Session 3 result (for reference): `Quality gate PASSED.` with **13/13** steps green (secret scan, Rust format desktop + seeders, Rust clippy desktop + seeders, Rust tests desktop + seeders, frontend lint, frontend typecheck + build, frontend tests, Python byte-compile, Python ML contract + quality-gate tests, Python tests). Counts from that run: Rust 69 passed / 1 ignored (desktop lib), 1 seeder test, frontend 19 tests / 3 files, Python 14 contract/gate tests + 49 remaining = 63 total. `pwsh` is not installed in this environment; use `powershell` as above.

Session 4 counts: Python went 63 → **80** (15 ablation + 2 processor tests). The general Python step runs 66 tests and the dedicated contract/gate step runs 14, so every module still runs exactly once. Rust (69 lib + 1 seeder) and frontend (19 tests / 3 files) are unchanged because no Rust or frontend source was touched.

Known warning noise (non-failing): XGBoost emits a CUDA/CPU device-mismatch warning plus UBJSON format warnings from the Python training fixture; these are pre-existing and unrelated to this session's changes.

Empty-result note: `docs/RefactorML.md` is an empty tracked file; it is listed here instead of being deleted (unresolved documentation debt, P2-10).

---

## 7. Open risks (ranked)

1. **Recommendation quality (DRAFT_SCORER_BACKTEST, new 2026-11-22):** the popularity-first ordering lifts top-1 agreement from 0.94% to 2.46% and top-5 from 6.0% to 10.5% (random 0.90% / 4.5%), which is progress but still far from a solved ranking, and the rolling per-patch prior measured 3.38% top-1. The same run shows the team-level outcome proxy **fell** from 0.528 to 0.489 AUC: the score is a popularity-led priority order, not a win-probability estimate, and any UI or doc that implies otherwise is wrong.
2. **Evaluation validity (P0-1):** the corrected dataset exists, but its chronological holdout (2026-04-05 → 2026-08-28) is the same window the session-5 ablation already scored, so any number produced on it is a regression reference, not a fresh held-out result. A defensible final claim needs newly collected matches (nothing newer than 2026-08-28 exists in the corpus).
3. **Training-data integrity (P1-6 / P1-5):** the encoding defect is fixed at the data level (v2: 0 rows with 12 presence flags, bans populated for 92.1%). Remaining, and *not* fixable in the parser: 3,870 rows (3.8%) contain a hero twice in the same source draft, 8,031 rows (7.9%) have all ban slots unrevealed in the source, and 9,795 matches were rejected by the documented rules. Every metric in §3 — including the session-4 ablation — still refers to the corrupt v1 parquet, so the near-baseline result cannot yet be attributed to the features rather than the data.
4. **Model trust (P0-1 / P1-6):** ML is off by default and automatic promotion is gated, but a user can still opt in and get a weak 25% term. Bundled inference-mode AUC 0.5122 / Brier 0.2495. On the corrected dataset the best validation candidate reaches 0.5479 ± 0.0004 (A) / 0.5481 ± 0.0012 (B) with Brier 0.2473, i.e. it finally beats the constant prior — but it is still ~9 standard errors below the 0.60 floor and has no fresh test number. The session-8 family comparison rules out learner capacity as the cause: a regularised linear model on the same features is statistically indistinguishable on AUC and better at the early-draft stages where recommendations are actually produced. Nothing was promoted.
5. **Lifecycle confidence (P0-2):** ingestion is now covered by 18 payload-level tests (consecutive matches, duplicates, undo, attribution, 300-payload invariants), but never driven through a real multi-match GSI stream. The warp status mapping and event emission are not executed by tests, and an out-of-order second stale hero-selection snapshot is still indistinguishable from a new draft.
6. **CSP / capability regression (P2-1, P2-2):** the app has not been launched with the new CSP and split capabilities; a broken permission would only appear at runtime.
7. **Semantics (P2-5):** the card no longer repeats the score as "Confidence", no longer attaches a score-threshold RISK chip, and the Analysis tab no longer shows the constant-0.5 lane bars. Still open: the category labels are semantics thresholds, and the payload keeps the constant `lane_balance` for compatibility.
8. **Reproducibility (P1-5):** resolved for the data path — `process_all()` now processes files in canonical match-id order, reassembles by input index, writes an explicit column order, and records fingerprints + run statistics in `dataset_schema.json`; two full runs with different worker/chunk settings produced identical content. Still open: lane features are disabled, and matchup-data runs would emit seven extra names (the v2 run deliberately excludes them because `ml.rs` cannot rebuild them).
9. **Latency (P0-4):** the harness recommendation dropped from 23.2 ms to 1.4 ms and ML-on lookahead from 318 ms to 4.1 ms per search (§3) after batched inference and exact ML pruning; ML-off search stays at 0.3-0.4 ms. In-app p95, inferences per GSI event and cold-session startup remain unmeasured.
10. **Data platform (P2-9):** 1.9 GB of loose dataset files with no versioning or retention.
11. **Mode encoding (P1-7):** `game_mode` stores Captains Mode as 22 (All Pick) and every non-captain mode as 0, collapsing `CAPTAINS_MODE` with `CAPTAINS_DRAFT`. It does not affect the 871-feature candidate (the group is dropped), but the bundled 887-feature model trains on it and no mode-segmented analysis can be trusted until a v3 dataset fixes the mapping.
12. **Target definition (P0-1 residual):** the corpus cannot identify next-pick effects (0 of 101,400 2-a-side contexts occur 50 times), and early-stage win AUC falls to 0.523 once ban timing is honest. Any recommendation that presents a win probability as the value of a pick overstates what the data supports. Session 10 adds the one signal that *is* measurable — next-pick imitation, from pick 2 onward —   but it is a behavioural target, not a value one, and it is *worse than a frequency table at the opening pick* (0.0744 vs 0.0990 top-1). Session 11 tested the phase-aware fix: no early-stage model beats the table there (−0.0416 at pick 0), the intercept-only diagnostic is level with it (+0.0012, CI −0.0018..+0.0041), and the recommendation is therefore the frequency fallback at picks 0-1 with the pooled model from pick 2 (hybrid 0.0599 overall).

---

## 8. Next three tasks

1. **Finish the honest surface, then re-run the usability gate (P2-5, P0-1).** The backtest re-run settled the framing: the score is a popularity-led priority order, not a win probability. This session removed the duplicated "Confidence" line, the score-threshold RISK chip and the constant-0.5 lane bars. Next: label each list with its source (popularity vs context) and re-derive the category tags, then run the 5-8 player comprehension protocol in [`HYBRID_PRODUCT_SPIKE.md`](HYBRID_PRODUCT_SPIKE.md) §4 against the new ordering. A pass is still not evidence of win-rate value.
2. **Patch rollover and a fresh evaluation window (P2-11, P0-1).** Ban counts, meta and the popularity prior are all patch-scoped, so a new patch starts empty until the seeders run: surface the patch's data age in `DataHealthNotice` (the ban seed already records its date range) and collect matches after 2026-08-28, the only source of a fresh holdout for any model claim.
3. **Replay a captured GSI stream through the packaged app (P0-2, A-10).** CI now drives both the dev build and the installed NSIS bundle, but ingestion is still synthetic. A real multi-match replay is the remaining runtime gap; in-app p95 over ≥100 GSI-driven requests belongs in the same run (P0-4).

### Continuation note (2026-10-28, session 4)

Start with task 1. The gate is in place and the bundled artifact **fails** it: do not lower thresholds, do not edit the shipped schema, and do not mark P0-1 fixed because a gate exists. A passing candidate must beat the constant-prior baseline (Brier < 0.2493 on this test split) as well as the policy floor to be worth bundling. `ml/models/*` is gitignored: gate reports and candidate artifacts stay local; only source/tests/docs are tracked.

Before changing feature code, read `ml/tests/test_feature_schema.py`: 631 structural features have a pinned order, while the two `*_ban_avg_winrate` columns and the 254 per-hero win-rates are history-ordered. `process_all()` uses `as_completed`, so regenerate with a deterministic match order or add an explicit canonical column list. Any schema change requires retraining, re-exporting the schema, updating the golden fixture, and re-running `scripts/verify.ps1`.

### Continuation note (2026-10-28, session 5)

The session-4 ablation answered the question it was given: **removing the context group fixes the serving contract, removing the ordinal pick group helps the earliest draft stage, and neither produces useful prediction.** The best candidate (881 features) scored test AUC 0.5136 / Brier 0.2494 and was **not** promoted; the bundled artifact is byte-identical to session 3.

Do not repeat that experiment on this dataset. It measured features on a presence vector that is wrong for 81% of rows. The next real step is regeneration:

1. `ml/feature-engineering/processor.py` already separates `isPick` bans from picks (session 4) and has a STRATZ-shape test. Do **not** reintroduce `is_default_true` parsing anywhere.
2. Confirm the regenerated per-side presence count is 5 (plus capped bans) before training anything; `82,739` rows with 12 flags is the shape of the bug.
3. Expect the same 887 feature *names* but different *values*, so the golden fixture must be regenerated and the schema re-exported before `scripts/verify.ps1` can pass.
4. The 2026-10-28 test window has been consumed. Hold out a new one and record its boundaries in the manifest before the next candidate claim.
5. `ml/training/ablation.py` is the harness for the repeat experiment (`--only`, `--n-estimators`, `--seed`, `--time-column`); it never scores the test partition, and `quality_gate.py` stays the only place that does.

### Continuation note (2026-10-28, session 6)

Regeneration is complete; **no model was trained and nothing was promoted**. Read
`docs/ML_TRAINING.md` §3a before touching features again.

Facts to build on, not re-derive:

1. `ml/datasets/processed_v2/draft_features.parquet` (102,118 × 890, SHA-256
   `472a5086…b6ce7`) is the corrected dataset. v1 at `ml/datasets/processed/` is
   intact and is what the bundled model was trained on; do not delete or
   overwrite it, and do not point the gate at v2 until a candidate exists.
2. Presence encoding is fixed: 0 rows with 12 flags, 100,369 rows with exactly 5
   per side, bans populated for 92.1% of rows. If a future run shows 12 flags
   again, the parser regressed.
3. `--no-matchups` is required for anything that must serve through `ml.rs`.
   With matchups the dataset gains seven names the runtime cannot rebuild.
4. The v2 holdout is **the already-inspected window** (10,211/10,211 shared with
   the session-5 test set). Freeze any new split in a config file and label it
   honestly; a clean final number needs post-2026-08-28 data.
5. Determinism is verified by content, not file bytes: chunk size changes the
   parquet layout. Use the content fingerprint when comparing runs, and
   `test_processing_is_deterministic_across_workers_and_chunks` as the guard.
6. Two source defects remain and must be handled by the *consumer*, not hidden:
   duplicate heroes in 3,870 drafts, unrevealed bans in 8,031 rows.

### Continuation note (2026-10-28, session 7)

The corrected-data ablation is complete and **nothing was promoted**. Read
`ML_TRAINING.md` §3b for tables and `models/ablation_v2/*.json` for raw reports.

What is now known:

1. Correct pick/ban encoding is worth about +0.011 to +0.014 validation AUC
   (0.5340-0.5358 on v1 → 0.5454-0.5498 on v2). It was necessary but nowhere near
   sufficient: the policy floor is 0.60 and every CI upper bound is ≤0.562.
2. Every v2 variant finally beats the constant train prior on Brier and log loss
   (0.2473-0.2480 vs 0.2489-0.2490) — a ~0.001-0.0017 margin, not a product win.
3. The selected variant is `no_match_context_no_pick_order` (871 features) in both
   populations, by the pre-registered rule. Its edge over `no_pick_order` is
   inside the noise band, so do not describe it as the best feature set.
4. Removing pick order now *hurts* the first-pick stage (~−0.008 AUC at one known
   pick). The v1 recommendation to drop those slots for early-draft accuracy is
   overturned by correct data; only full-draft metrics favour the combined set.
5. The harness was wrong twice and both are now fixed and tested: `--split-config`
   actually applies the frozen boundaries, and partial-draft rows are rebuilt by
   the production feature code instead of keeping full-draft aggregates. Any
   future ablation must keep both.
6. No test number exists for these candidates and none may be claimed. A clean
   holdout requires matches after 2026-08-28; until then the 0.60 floor can only
   be assessed on validation.

### Continuation note (2026-10-29, session 8)

The model-family comparison is complete and **nothing was promoted**. Read
`ML_TRAINING.md` §3c and `ml/models/families/*.json`.

What is now known:

1. The learner is not the limitation. XGBoost 0.5479 ± 0.0004, regularised
   logistic 0.5454 (seed-invariant), `HistGradientBoosting` 0.5380 ± 0.0026 on
   the same 871 features, split, and partial-draft frames (population A); the
   duplicate-hero-excluded population reproduces the same ranking.
2. **At the stages the product uses, the linear model is better**: +0.011 AUC at
   one known pick per side, +0.005 at two and three; XGBoost only leads at five
   (+0.003). Do not describe XGBoost as the better draft-time model — it is the
   better *full-draft* model by ~0.003, with reliably better calibration
   (Brier +0.0015 for logistic, paired CI excludes zero).
3. Neither qualifies as a material improvement under the pre-registered rule
   (paired AUC CI excluding zero **and** Brier gain > 0.002), so the reference is
   retained. Reporting a 0.003 AUC gap as a winner would be noise-chasing.
4. All three fail the unchanged policy floor on validation. `quality_gate.py`
   was not re-run; the test partition was not read (`test_scored_here: false` in
   both reports).
5. The bundled ONNX model, the shipped schema, and both datasets are untouched.
   `hist_gbdt` cannot be exported through skl2onnx — a tooling limitation, not a
   contract change.
6. `scripts/verify.ps1` needs a forward-slash path: `-File scripts/verify.ps1`.
   A backslash-escaped path arrives as `scriptsverify.ps1` and fails.

Next: stop tuning estimators. Change the question (§8 task 1), or collect a
post-2026-08-28 window so any future claim can be tested on clean data.

### Continuation note (2026-10-29, session 9)

The objective study is complete. Read [`ML_TRAINING.md`](ML_TRAINING.md) §3d and
`ml/models/objective_study/diagnostics.json`. Nothing was trained for selection,
nothing was promoted, and the test partition was not read.

1. **The corpus is 90.3% Captains Mode** (87,746 STRATZ `CAPTAINS_MODE` + 4,475
   OpenDota CM). The earlier "this is mostly All Pick" reading was an artefact of
   the `game_mode` encoding defect, not a property of the data.
2. **`game_mode` is mis-encoded** (`22` for any `CAPTAIN*`, `0` for everything
   else). It is in the dropped candidate group, so no reported model metric
   changes; it does invalidate mode-segmented analysis and it corrupts the
   bundled model's feature. Recorded as AUDIT P1-7; fixing it needs a v3 dataset.
3. **The early-draft numbers were optimistic.** With only the bans that had
   actually happened, AUC at zero picks is 0.5228 (was 0.5271) and at one pick
   0.5219 (was 0.5248). The constant-prior Brier is 0.2498 at those stages, i.e.
the model is level with a constant predictor at the top of the draft.
4. **Win probability is finished as the headline metric.** 0.5481 with a CI
   half-width of 0.0111; the 0.60 floor is ~5 standard errors away. More matches
   narrow the interval, not the point estimate.
5. **Ranking is the product-right target but is not identifiable here**: 0 of
   101,400 two-a-side contexts occur 50 times, and 38 of 102,080 full drafts
   repeat at all. Session 8's caution still holds — do not ship a "this pick is
   worth +X%" claim.
6. **The next experiment is an imitation baseline**: predict the observed next
   pick among the legal set, scored by top-1/top-3 accuracy on held-out drafts.
   That is identifiable, cheap, and a genuine test of whether draft state carries
   decision signal.
7. **Environment note:** `objective_diagnostics.py` caches raw `gameMode` in
   `ml/models/objective_study/true_modes.tsv`. A cold scan of 102k payloads in one
   directory takes >10 minutes on this machine; the cache makes reruns ~40 s. Use
   `--refresh-modes` after adding data.

### Continuation note (2026-10-29, session 10)

The imitation experiment is complete. Read [`ML_TRAINING.md`](ML_TRAINING.md) §3e
and `ml/models/imitation/imitation_report.json`. No production behaviour changed,
nothing was promoted, and the test partition was never parsed.

1. **Chronology is trustworthy**: `order` is dense and unique (0..23) in every
   eligible CM payload, 94% share one template, and 60/60 sampled matches agree
   with the parquet's pick slots and label. Payloads without order (3,022) are
   excluded, not repaired.
2. **The state does predict the next pick, from pick 2 onward.** Model 0.0551
   top-1 / 0.1312 top-3 versus 0.0405 / 0.0959 for the stage/side frequency
   table; paired cluster bootstrap +0.0146 (+0.0128..+0.0165).
3. **It does not at the opening pick**, which is the product-relevant moment:
   0.0744 versus 0.0990. The model's intercept alone scores 0.1003 there, so the
   removal terms are what hurt. Do not claim early-draft usefulness.
4. **Two capacity fixes were required and measured**, not guessed: a per-phase
   hero intercept (the model could not represent its own baseline without it) and
   pooling removal coefficients across phases (1,668 → 562 features). Keep both
   when extending this line.
5. **Imitation is not value.** These numbers say nothing about win rate or pick
   quality, and no counterfactual claim is supported.
6. **Next:** the opening-pick question is closed by session 11 — the frequency
   fallback is the answer, and an early model does not beat it. The remaining step
   is a product decision plus a fresh holdout, not another model variant.
7. **Runtime note:** the run takes ~8 min, of which ~2 min is three grid fits and
   one final fit on 745,930 states. The action cache
   (`ml/models/imitation/actions.parquet` + `scan_reasons.json`) makes reruns ~7 min;
   keep `scan_reasons.json` with the cache or the exclusion accounting silently
   loses the payloads rejected before parsing.

### Continuation note (2026-10-29, session 11)

The phase-aware experiment is complete. Read [`ML_TRAINING.md`](ML_TRAINING.md) §3f
and `ml/models/imitation/phase/report.json`. No production behaviour changed,
nothing was promoted, and the test partition was never parsed.

1. **Do not build a trained early-stage model.** Three configurations (pick 0 only,
   pick 1 only, shared 0-1) all lose to the training-only stage/side frequency
   table at their own stage, two of them with intervals far from zero.
2. **At picks 0-1 the information is the prior.** The model's intercept alone is
   level with the table (+0.0012, CI −0.0018..+0.0041), and it collapses to 0.0039
   by picks 8-9, so the removal block carries the late signal and only noise early.
3. **Use the hybrid if the imitation line continues**: table at picks 0-1, pooled
   model from pick 2 (0.0599 overall; the pooled model's +0.0249 / +0.0223 stage
   gains are untouched).
4. **Distrust the internal holdout for early stages.** The pick-0 model scored
   0.0994 there and 0.0575 on validation — far outside sampling noise. Selection on
   the last 15% of training overstates early-stage capacity.
5. **Disclosure on record:** the first run's acceptance predicate for the hybrid
   also compared the aggregate, which the hybrid must change; it was corrected
   before the reported run, with every metric unchanged. Both reports are kept.
6. **Runtime:** 356 s end to end (pooled grid 3 fits + refit on 745,930 states, three
   early grids, six bootstrap blocks of 1,000 match-level resamples). The action
   cache in `ml/models/imitation/` must be present; keep `scan_reasons.json` with it.
7. **Do not expand this search further.** Imitation is behavioural, and the gate it
   could never pass (ROC-AUC ≥ 0.60, Brier ≤ 0.235) measures a different target.

### Continuation note (2026-10-29, session 12)

The product-utility spike is complete. Read
[`HYBRID_PRODUCT_SPIKE.md`](HYBRID_PRODUCT_SPIKE.md) and
`ml/models/hybrid_spike/report.json`. No production behaviour changed, nothing was
exported or promoted, and the test partition was never parsed.

1. **The surface is showable.** Generate it with `hybrid_presentation.py`; open
   `ml/models/hybrid_spike/mockup.html`. It shows six uniform validation drafts at
   picks 0/2/5 with a source badge and no scores, percentages, confidence or risk
   labels.
2. **Keep the claim guard.** `FORBIDDEN_LABELS` aborts the render if a label such as
   "best pick" or "win probability" appears. Extend the list rather than deleting
   entries if the copy changes.
3. **Stability is presentable**: τ ≈ 0.86 on one-ban transitions, with the big
   reshuffles at the 1→2 source switch and the 7→8 ban block. 8% of top entries are
   removed by the very next action (20% at 7→8), so a UI must name removals instead
   of silently re-sorting.
4. **Do not present the ordering as certainty.** The contextual list disagrees with
   the frequency list in ~85-90% of pick-2+ states, but at 4.9-6.9% top-1 over ~124
   heroes the argmax is near-tied; disagreement is not confidence.
5. **Next is the usability gate**, not more modelling: 5-8 players, blinded
   frequency-vs-contextual pairs, a scored comprehension check as go/no-go. If
   comprehension fails, stop the line; do not add copy to a misleading surface.
6. **Runtime:** the spike takes ~100 s, of which ~82 s is the single refit needed to
   obtain per-state probabilities. The model is 0.56 MB and scores one state in
   0.135 ms, so the product cost is negligible; the analysis peak (1.8 GB) is the
   training fit, not the UI.
7. **Artifacts are gitignored** (`ml/models/hybrid_spike/`). Two runs produced
   byte-identical HTML and identical metrics; regenerate rather than hand-edit.

### Continuation note (2026-11-19, session 13)

A fresh runtime-contract audit is in [`AUDIT_2026-11-19.md`](AUDIT_2026-11-19.md).
It found and fixed five defects that the green unit suite could not see, all on
the packaged-runtime path:

1. **The CSP blocked every hero portrait and both web fonts.** `tauri.conf.json`
   now allows the Steam CDN and the two Google Fonts origins;
   `test_csp_allows_assets_the_ui_loads` pins each origin in its specific
   directive. Actual rendering is still unverified (no GUI launch).
2. **A changed Model Path never loaded until restart, and startup bypassed the
   allow-list** used by `load_ml_model`. Both paths now share
   `canonical_model_path`, and `set_config` loads a changed path before
   persisting, so Save cannot succeed for a model that will not load.
3. **Disabling the overlay collapsed the main window** into a recovery button,
   making Settings unreachable. The main window now always renders the
   dashboard.
4. **Manual-draft ALLY/ENEMY counters ignored the selected side.**
5. **NaN overlay geometry could be persisted.** Both validators now reject it.

Counts: Rust 69 → **71**, frontend 19, Python **163** (14 contract/gate + 149),
gate 13/13. The ML training/test modules, frozen split, and hybrid spike doc are
now staged, so CI covers them after the next commit. Still open and ranked in
the report: unpinned Python dependencies, unused Tauri plugins/permissions,
numeric hero ids in explanations, and the unlaunched GUI/GSI runtime check.

### Continuation note (2026-11-22, session 16)

The roadmap was refreshed to HEAD `a2b7bd9` and the deterministic ordering was
changed to popularity-first. Read §0, §3a and risk 1 in §7.

1. `DRAFT_SCORER_BACKTEST.md` is now a source of product truth, not a side
   experiment: the context-only ordering agreed with 0.94% of human picks at
   top-1, against 2.21% for a global popularity prior.
2. The shipped score is `0.60 × popularity + 0.40 × context`
   (`POPULARITY_ORDER_SHARE`), where popularity is the seeded per-patch
   pick-rate percentile inside the current pool. Re-run on the same sample:
   top-1 2.46%, top-5 10.49%, mean percentile 0.649; 0/200 mismatches against
   the list the UI shows.
3. **The team-outcome proxy regressed** (0.528 → 0.489 AUC) while pick agreement
   roughly tripled. Do not quote the score as a win probability, and do not
   "fix" the AUC by lowering the popularity share — that metric was already
   below the report's 0.55 usability floor, and optimising it would undo the
   ordering change.
4. The prior behind those numbers is the full-corpus seed, i.e. in-sample for
   the replay, exactly like the shipped ban table. The rolling per-patch prior
   (3.38% top-1) remains the honest out-of-sample reference and is not what the
   app serves.
5. Make the harness a scheduled job before the next ordering change: a re-run
   whose top-1 falls below the global prior should fail rather than be
   discovered in a doc.

---

## 9. Resume instructions

Start here, in this order:

1. Read this file, then §4-§5 of [`AUDIT_2026-10-27.md`](AUDIT_2026-10-27.md) for the evidence behind each ID.
2. Run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1` to confirm the baseline still holds (Rust 87 / frontend 32 / Python 166, 14/14 steps).
3. Run `ml/venv/Scripts/python.exe ml/training/quality_gate.py`; expect exit 1 until a candidate replaces the bundled model. Use `--evaluation-mode as_trained` only to compare against the manifest, not to judge runtime quality.
4. Pick the next task from §8 (currently finishing the honest surface, then the patch-rollover and replay work; the modelling questions are closed in ML_TRAINING §3e-3f until a fresh holdout exists). The task ID maps 1:1 to the audit finding, so no re-audit is needed.
5. Update this file's status column and the verification log **after** running the check — never before.
6. Leave changes uncommitted unless the user explicitly authorizes a commit.

Known environment limitations to carry forward: no live Dota 2 client, no Tauri GUI launch in-session, and PowerShell may report a false non-zero exit for `cargo test` because of stderr piping. Full Python dataset regeneration **is** practical locally since session 6: ~4 min with `--workers 8 --chunk-size 2000` (35.7 MB output) or ~14 min with `--workers 3 --chunk-size 777` (56.3 MB), peak memory bounded by `--chunk-size`. The session-4 ablation takes ~283 s (CPU, 8 threads); the gate takes ~25 s with the joblib→UBJSON warning.
