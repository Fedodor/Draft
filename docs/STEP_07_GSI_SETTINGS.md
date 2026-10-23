# Step 7 Design: Settings, GSI, and UI State

- Date: 2026-10-23
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), H4, H5, M2

## Implementation status

- [x] Persist team side (via GSI), GSI port, GSI auto-start, auth token, overlay settings, and model path.
- [x] Apply the configured GSI port and auth token at startup; report bind failures.
- [x] Add optional shared-secret authentication for the local GSI listener.
- [x] Track turn and side from GSI events with an explicit state machine (`activeteam`/`pick`, with a documented CM-order fallback).
- [x] Orient the overlay from the player's actual side.
- [x] Stop reinitializing draft state on team change.
- [x] Bind the settings toggles to the correct values.
- [x] Validate settings and provide reset-to-default that persists.

## Problem

The settings surface was cosmetic and the GSI integration parsed a schema Dota
never sends:

1. `set_config`/`get_config` were stubs that stored nothing, so preferences did
   not survive a restart (H4).
2. The GSI listener always used port `3000` and never surfaced a bind failure.
3. `GSIDraft` deserialized `picks`/`bans` arrays, but real GSI sends fixed
   `pickN_id`/`banN_id` slots plus `activeteam`/`pick`, so no pick or ban was
   ever recorded and the turn was derived from `picks.first()` (wrong) (H4).
4. `DraftOverlay` hardcoded Radiant as the allied team (H5).
5. `App.tsx` re-ran full initialization whenever the selected team changed,
   discarding the loaded draft (M2).

## Configuration persistence

`AppConfig` (in `models.rs`) gains `gsi_auto_start` and `gsi_auth_token` and now
derives `PartialEq`. Every field carries `#[serde(default)]` with explicit
defaults, so a blob written by an older build still loads and new fields take
their defaults. `AppConfig::validate` rejects a port below 1024, an empty model
path, and an empty (non-null) token.

The serialized config is stored as one JSON blob in the existing `app_settings`
key/value table under `CONFIG_KEY = "app_config"`, reusing the Step 5 database:

- `Database::set_app_config(&AppConfig)` / `get_app_config() -> Result<Option<AppConfig>>`.
- A corrupt blob is a hard error rather than a silent reset; startup logs it and
  falls back to defaults.

`commands.rs` now implements the real commands:

- `set_config(db, gsi, config)` validates, persists, and reconfigures the GSI
  server for the next start.
- `get_config(db)` returns the stored config or defaults when none exists.
- `restart_gsi(gsi, app)` stops the listener, waits for the port to free, and
  starts it again on the configured port.

`lib.rs` shares `Arc<Database>` as Tauri state, loads the config during setup,
prefers a configured model path when it exists (falling back to the bundled
resource), applies the GSI port/token, and starts the listener only when
`gsi_auto_start` is set.

## GSI: real schema, state machine, auth, and bind errors

`GSIDraft` is remodeled to the real payload: `activeteam`, `pick`,
`activeteam_time_remaining`, bonus timers, and `team2`/`team3` slot objects
(`GSITeamDraft` with `pick0_id`..`pick4_id` and `ban0_id`..`ban4_id`). Empty
slots (0) are skipped. `player.team_name` is parsed into `GSIPlayerSelf` so the
app learns which side the local player is on.

Turn tracking uses an explicit state machine:

- `side_from_team_id` maps `activeteam` (2 → Radiant, 3 → Dire).
- `resolve_turn` prefers the game's own `activeteam`/`pick`, and falls back to
  `CM_ORDER` (the canonical 20-action Captain's Mode sequence) using the number
  of completed picks and bans.
- `CM_ORDER` is asserted to be 5 picks and 5 bans per side.

The server itself:

- `GsiServer` holds a `GsiConfig { port, auth_token }` behind a `RwLock`;
  `configure(port, token)` updates it.
- `start` uses `try_bind_with_graceful_shutdown`, so a busy port is returned as
  `Err` and recorded in `last_error` instead of panicking via warp's `bind`.
- The shutdown channel is recreated per `start`, so the listener can be
  restarted on a new port.
- When a token is configured, the handler rejects any payload whose top-level
  `auth.token` does not match with `401 Unauthorized`.
- `generate_gsi_config_with_auth` writes an `auth { token }` block into the
  generated Dota config when a token is set.
- The server emits a `gsi-team` event with the local player's side and keeps
  emitting `gsi-update` with the draft state.

`GsiStatus` now reports `auth_required` and `last_error`.

## Frontend

- `packages/shared-types` mirrors the real `GSIDraft`/`GSITeamDraft`,
  `GSIGameState.player`, the new `AppConfig` fields, and the extended
  `GsiStatus`. `Hero.primary_attr` now includes `'all'` to match the backend.
- `DraftOverlay` derives the allied/enemy halves and ban filters from
  `selectedTeam`, so Dire players are oriented correctly.
- `App.tsx` splits initialization into three effects: a one-time mount load
  (heroes, backend config, draft, initial recommendations), a team-change effect
  that refreshes only recommendations, and long-lived listeners. The `gsi-team`
  event updates the selected side; a ref keeps the latest team available to the
  listener without re-subscribing.
- `SettingsPanel` hydrates from `getConfig` on mount, binds auto-start to
  `gsi_auto_start`, exposes the auth token, validates before saving, persists
  through `set_config`, and restarts the listener when the port/auth/auto-start
  changed. It shows the live GSI state and any bind error. Reset persists
  defaults.
- `Header` shows the active patch from the draft state instead of a literal.

## Tests

- `models.rs`: `AppConfig` defaults and validation; `GSIPlayerSelf::team_side`;
  `GSITeamDraft` skips empty slots.
- `gsi.rs`: config generation with and without auth; default/configured server
  status; `side_from_team_id`; `resolve_turn` preferring `activeteam` and falling
  back to `CM_ORDER`; `CM_ORDER` shape; `draft_actions` reading real slots.
- `database.rs`: `AppConfig` round-trip and forward-compatible load of an older
  blob missing the new fields.
- `api.smoke.test.ts`: `restart_gsi`, `get_config`, and `set_config` argument
  shapes.

## Verification

```powershell
cargo test --manifest-path apps\desktop\src-tauri\Cargo.toml --lib   # 36 passed
cargo clippy --manifest-path apps\desktop\src-tauri\Cargo.toml --all-targets
cargo fmt --manifest-path apps\desktop\src-tauri\Cargo.toml -- --check
npm test            # 11 passed (apps\desktop)
npm run build       # tsc + vite build
```

Clippy reports only pre-existing warnings (auto-deref in `commands.rs`,
`DraftScorer`/`DraftEngine` derives, and the `ml.rs` type); the new code adds
none.

## Residual risks

- The CM-order fallback approximates turn order and is only used when GSI omits
  `activeteam`; the game's own `activeteam`/`pick` is authoritative.
- Only `player.team_name` is consumed from the GSI `player` block; per-player
  details are unchanged.
- GSI binding and auth are exercised through unit tests; a live Dota client is
  still needed for an end-to-end check.
