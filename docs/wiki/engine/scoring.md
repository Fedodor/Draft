# Scoring Data Flow

Recommendations combine six components: matchup, synergy, composition, meta, comfort, and ML. Until Step 3, the matchup and synergy scorers were never given the loaded database maps, so 45% of the weight was always neutral.

- `DraftEngine::initialize` wires `hero_matchups` and `hero_synergies` into `DraftScorer`.
- Hero roles are canonicalized to lowercase at the scorer boundary; `"Pusher"` maps to `"push"`.
- `PrimaryAttribute::All` round-trips through `as_str`/`from_db`.
- Real data comes from `infrastructure/seeders/scoring_seed.json`; `seed_matchups --synthetic` is the only placeholder path.
- `get_score_breakdown` exposes the per-component contribution.

Canonical source: [`../../STEP_03_SCORING.md`](../../STEP_03_SCORING.md).

## Open follow-ups

- ~~Unify the seeder and application database paths.~~ Done in Step 5; see [Database, seeding, and patch](data.md).
- ~~Make matchup queries patch-aware.~~ Done in Step 5; matchups/synergies/meta load for the persisted active patch.
- ~~Add a debug panel in Step 7; the IPC command already exists.~~ Done in Step 10; see [Score breakdown panel](score-breakdown.md).
