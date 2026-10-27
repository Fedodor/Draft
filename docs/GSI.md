# Game State Integration (GSI)

The Smart Draft Assistant uses Dota 2's built-in Game State Integration (GSI) to track the draft in real time without OCR or screen scraping.

## How it works

1. **Configuration**: a configuration file (`gamestate_integration_smart_draft.cfg`) is placed in the Dota 2 `cfg/gamestate_integration` directory. The app generates it from the configured port and optional auth token.
2. **Server**: the Tauri backend starts a lightweight HTTP server (`warp`) bound to `127.0.0.1` on the configured port (default 3000).
3. **Events**: Dota 2 POSTs JSON to `/gsi` whenever game state changes (hero picked, ban made, phase changed).

When a token is configured, the generated config includes an `auth` block and the listener rejects payloads whose token does not match. Auth is optional; the Settings panel warns when it is empty, because any local process can then post a draft.

## Payload mapping

`GSIGameState` models the fields the app consumes:

- `map.game_state`: phase detection (`DOTA_GAMERULES_STATE_HERO_SELECTION`, `STRATEGY_TIME`, `PRE_GAME`).
- `player.team_name`: the side the local client is on. Emitted to the frontend as `gsi-team` so the overlay orients itself without a manual toggle.
- `draft.team2` / `draft.team3`: per-team slot lists (`pickN_id`, `banN_id`, `team_id`). `team_id` 2 = Radiant, 3 = Dire and takes precedence over the `team2`/`team3` position; an unrecognized id falls back to the slot and logs a warning.
- `hero`: the local player's current hero.

## Draft ingestion rules

The listener is not a blind append. `reconcile_draft` in `src-tauri/src/gsi.rs` applies these rules:

- **Validate every action**: known hero, hero still available, team below 5 picks, team below 5 bans, and no duplicate across both teams.
- **Dedupe by hero id**, not by slot or order. GSI renumbers slots; hero id is the only stable identity, and the same hero cannot be both picked and banned. Picks are reconciled before bans, so if one snapshot contains the same hero in both, the pick wins.
- **Respect manual undo**: a hero the user undid is suppressed for the rest of the match, so a stale snapshot cannot re-add it. An explicit manual pick/ban lifts the suppression. Suppression is keyed by hero id, so it cannot distinguish a stale re-add from a genuine later pick of the same hero in the same match; that case is ignored until the next reset.
- **Reset on a new match**: entering `HERO_SELECTION` from a non-draft state (init, in-game, post-game) resets the engine, its undo stack, suppression set, and recommendation cache, and restores `available_heroes`. Starting the app mid-draft does not reset. The payload that triggers the reset is deliberately **not** ingested, because the first hero-selection update can still carry the finished game's slots.
- **Turn state**: `activeteam`/`pick` from GSI are authoritative when **both** are present. If either is missing or `activeteam` is not 2/3, the listener falls back to the Captain's Mode order (4 bans → 4 picks → 4 bans → 4 picks → 2 bans → 2 picks, with snake sides) exposed as `CM_ORDER` and pinned by a table-driven test.
- **Append-only**: ingestion adds actions it has not seen and never removes one because a snapshot omits it. Duplicate and stale full snapshots are therefore safe to replay. A stale snapshot cannot "un-pick" a hero.
- **Turn order is not a gate**: the snapshot is trusted for *what* was picked or banned; `activeteam`/`pick` only drive the next-turn display. A valid hero in an unexpected slot position is accepted, because GSI slot order is not a reliable chronology.

Invalid or duplicate actions are logged and skipped; they never abort the request. Counting is bounded per side (5 picks, 5 bans); anything beyond that comes from malformed or cross-match slots and is dropped, with the hero left in the pool.

### Known lifecycle ambiguities

These are documented rather than papered over with heuristics, because the GSI payload does not guarantee enough to distinguish the cases:

- Only the reset-triggering payload is skipped. If a second stale `HERO_SELECTION` snapshot from the previous match arrives out of order after the reset, it is structurally identical to a new draft and is accepted.
- There is no per-action identity for a pick (slots are renumbered), so a user undo is remembered per hero, not per slot.


## Events emitted to the frontend

| Event | Payload | Purpose |
| --- | --- | --- |
| `gsi-update` | `DraftState` | Full draft snapshot after each accepted update. |
| `gsi-team` | `"radiant"` \| `"dire"` | The local player's side. |

The frontend applies `gsi-update` immediately and debounces the expensive recommendation request (200 ms) inside `RecommendationFeed`; see `docs/ROADMAP.md` P0-4/P1-1.

## Testing

Unit tests inline in `gsi.rs` cover the CM order, new-draft detection, team-id attribution, hero-id dedupe/idempotence, unknown-hero rejection, manual-undo suppression, and available-hero restoration on reset.

`gsi/lifecycle_tests.rs` adds 18 payload-level lifecycle tests. They build realistic GSI JSON (`map`, `player`, `draft.team2`/`team3`, `pickN_id`/`banN_id`) and call `apply_gsi_payload` — the same function the warp handler calls. Scenarios cover consecutive matches, duplicate/conflicting/unknown/stale actions, malformed payloads, auth, manual undo and its suppression lifecycle, team attribution, the full 20-action CM fallback, per-team ban capacity, and a 300-payload deterministic property test that re-checks the state invariants after every step.

Run them with:

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib lifecycle
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
```

Not covered: the warp status mapping, Tauri event emission, and a replay of captured or live Dota 2 traffic. See the P0-2 remediation section in `docs/AUDIT_2026-10-27.md` for the remaining gaps.
