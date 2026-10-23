# Settings, GSI, and UI State

- Type: Engine
- Status: Implemented (Step 7)
- Canonical source: [`../../STEP_07_GSI_SETTINGS.md`](../../STEP_07_GSI_SETTINGS.md)
- Related: [Database, seeding, and patch](data.md), [Tauri IPC conventions](ipc.md), [Remediation plan](../plans/remediation-plan.md)

## Why it matters

Preferences did not persist, the GSI listener ignored the configured port, the
overlay assumed the player was Radiant, and the draft parser expected a schema
Dota never sends. The result was an app that could not be configured and that
misread real games.

## What is in place

- Persistent config: `AppConfig` is validated and stored as a JSON blob in the
  `app_settings` table; `set_config`/`get_config` are real and `restart_gsi`
  rebinds the listener. Startup applies port, auth, auto-start, and model path.
- Real GSI schema: `GSIDraft` uses `activeteam`/`pick` and `team2`/`team3` slot
  objects; empty slots are ignored.
- Turn state machine: `resolve_turn` prefers the game's `activeteam`/`pick` and
  falls back to the canonical `CM_ORDER`.
- Auth: an optional shared secret is written into the generated Dota config and
  enforced by the listener (`auth.token`, 401 on mismatch).
- Bind failures are returned and surfaced through `GsiStatus.last_error` instead
  of panicking.
- Side-aware overlay: `DraftOverlay` orients from `selectedTeam`; `gsi-team`
  events update that side from the local player.
- No draft reinitialization on team change: `App.tsx` separates mount, team
  change, and event listeners.

## Test coverage

`models.rs`, `gsi.rs`, and `database.rs` cover config defaults/validation,
config round-trip and forward-compatible load, the turn state machine and CM
order, real slot parsing, and config generation with/without auth. The frontend
smoke test covers `restart_gsi`, `get_config`, and `set_config`.

## Residual risks

- The CM-order fallback is an approximation, used only when `activeteam` is
  absent.
- Live Dota end-to-end verification is still required.
