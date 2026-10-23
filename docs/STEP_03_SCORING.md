# Step 3 Design: Scoring Correctness

- Date: 2026-10-22
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), C6 and M1

## Implementation status

- [x] Wire loaded matchup and synergy data into the startup scorers.
- [x] Normalize hero roles at the scorer boundary.
- [x] Preserve `PrimaryAttribute::All` through database reads and writes.
- [x] Replace synthetic matchup seeding with real data-derived rows.
- [x] Gate synthetic seeding behind an explicit `--synthetic` flag.
- [x] Make the fallback hero explicit and testable.
- [x] Add tests for role casing, loaded data, updates, unknown heroes, and empty data.
- [x] Expose per-component scores through the `get_score_breakdown` command.
- [x] Seed the shared application database (Step 5 unified the app/seeder path).

## Problem

The engine loaded `hero_matchups` and `hero_synergies` into `DraftEngine` fields, but `DraftEngine::initialize` never passed those maps to `MatchupScorer` or `SynergyScorer`. Both scorers stayed empty and returned `0.5` for every candidate, so 45% of the recommendation weight was neutral regardless of the draft.

Two related data issues made the visible scores unreliable:

1. `heroes.json` stores roles as `"Carry"`, `"Support"`, `"Pusher"`, but `CompositionScorer` keys were lowercase (`"carry"`, `"support"`, `"push"`). Every role lookup missed and fell back to weight `0.5`.
2. `seed_matchups` generated deterministic hash-based placeholder values by default, with no flag or marker separating them from real data.

## Scorer wiring

`DraftScorer` now owns the data boundary:

```rust
pub fn set_data(
    &mut self,
    matchup_data: HashMap<HeroId, Vec<HeroMatchup>>,
    synergy_data: HashMap<HeroId, Vec<HeroSynergy>>,
)
```

`DraftEngine::initialize` calls it after loading the database tables and logs the number of heroes with matchup/synergy rows. `simulation.rs` reads the same scorer instances, so search-based recommendations now use the loaded data as well.

## Role normalization

All roles pass through `canonical_role` before comparison:

- Trim and lowercase.
- `"pusher"` maps to the composition key `"push"`.
- `"disable"` maps to `"disabler"`.
- `"initiation"` maps to `"initiator"`.
- Duplicate aliases are removed per hero.

`CompositionScorer::with_hero_roles` stores canonical roles, and `score`/`explain` canonicalize the candidate's roles. Uppercase and lowercase data now produce identical scores.

## Primary attributes

`PrimaryAttribute` now has explicit `as_str` and `from_db` helpers. `all` is preserved on read and write instead of relying on a debug-format string in one direction and a match arm in the other. Unknown legacy values still fall back to `Str` for compatibility, and that behavior is covered by a unit test.

## Real matchup and synergy seed

`ml/training/export_scoring_seed.py` generates `infrastructure/seeders/scoring_seed.json` from the processed datasets:

- Matchups come from `hero_matchups.parquet`. Rows with fewer than 20 games are dropped, rows whose hero IDs are absent from `heroes.json` are skipped, and both directions are emitted.
- Synergies come from same-team pairs in `draft_features.parquet`. Synergy is the observed pair winrate minus the average individual hero winrate, clamped to `[-0.5, 0.5]`; pairs need at least 50 games.
- Current output: 9,052 ordered matchup rows and 7,381 synergy pairs. A clean seed produces exactly those counts plus 126 heroes.

`seed_matchups` now loads that file by default. It fails with an actionable message when the file is missing. `--synthetic` is the only way to write placeholder values, and it logs a warning that the data is development-only.

`ml/training/compute_matchups.py` was also corrected to strip the `rad_has_` prefix correctly, map hero names to IDs, and write the `hero_a_winrate`/`hero_b_winrate` schema used by the exporter.

## Unknown and fallback heroes

- Unknown hero IDs use the stable ML placeholder `hero_<id>` instead of dropping the feature silently. A test covers an unknown pick in `build_feature_map`.
- The single-hero development fallback is now covered by a test that asserts its ID, roles, and CM flag.
- Scorers return `0.5` for heroes with no matchup/synergy rows; empty data is tested explicitly.

## Debug view

`DraftEngine::score_breakdown(hero_id, team)` returns a `ScoreBreakdown` with the matchup, synergy, composition, meta, comfort, and ML components plus the weighted total. It is exposed as the Tauri command `get_score_breakdown(hero_id, team)` and through `api.getScoreBreakdown` in the frontend service.

```powershell
# from the frontend dev console, via the API service
api.getScoreBreakdown(1, 'radiant')
```

## Tests

Rust (`cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib`):

- Canonical role casing and aliases.
- Capitalized hero roles produce the expected composition weight and ally count.
- Loaded matchup data changes the score, and an updated row changes it again.
- Loaded synergy data changes the score.
- Empty scorer data and unknown heroes return `0.5`.
- `DraftScorer::set_data` wires both maps.
- `score_breakdown` exposes the components and weighted total.
- `PrimaryAttribute::All` round-trips.
- Unknown heroes use the stable ML placeholder.

Seeder (`cargo test --manifest-path infrastructure/seeders/Cargo.toml --bin seed_matchups`):

- `scoring_seed.json` parses into the expected model types and contains both datasets.

Current results: 20 Rust library tests pass; 1 seeder test passes; 15 Python tests pass.

## Verification

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
cargo test --manifest-path infrastructure/seeders/Cargo.toml --bin seed_matchups
& ml\venv\Scripts\python.exe -m pytest ml/tests -q
```

Regenerate the seed after refreshing the processed datasets:

```powershell
& ml\venv\Scripts\python.exe ml\training\export_scoring_seed.py
cargo run --manifest-path infrastructure/seeders/Cargo.toml --bin seed_matchups
```

## Residual risks

- ~~The app and the seeders used different data directories.~~ Resolved in Step 5; both resolve `paths::app_data_dir()`.
- ~~`get_matchups_for_hero` used a hardcoded patch version.~~ Resolved in Step 5; queries take the persisted active patch.
- Synergy is derived from observed same-team winrate uplift, not an independently collected synergy table. Pairs below 50 games are absent and score neutral.
- The debug breakdown is an IPC API, not a UI panel. A panel can be added in Step 7 without changing the backend contract.
- The generated seed file is data-derived and should be regenerated when the processed datasets or hero list changes.
