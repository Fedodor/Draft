# Step 4 Design: Tauri IPC and Manual Drafting

- Date: 2026-10-23
- Status: Implemented
- Parent plan: [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md)
- Audit findings: [`AUDIT.md`](AUDIT.md), C5 and H3

## Implementation status

- [x] Choose one IPC convention for every command.
- [x] Add typed error responses for invalid teams, heroes, turns, and model files.
- [x] Validate picks and bans in the engine before mutating state.
- [x] Add a pick/ban mode toggle in `HeroPicker`.
- [x] Allow manual correction regardless of the current turn.
- [x] Add explicit undo and reset behavior with Rust tests.
- [x] Add a frontend smoke test covering the manual commands.

## Problem

Tauri v2 converts command arguments to camelCase by default, but every frontend call in `api.ts` sends snake_case names. `manual_pick`, `manual_ban`, `get_hero_details`, and `load_ml_model` compiled but failed at runtime. There was also no pick/ban toggle, the picker disabled valid corrections outside the current turn, and commands returned bare strings instead of machine-readable errors.

## IPC convention

Every command in `commands.rs` now declares:

```rust
#[tauri::command(rename_all = "snake_case")]
```

This keeps the Rust parameter names and the frontend `api.ts` arguments in one convention (`hero_id`, `model_path`, `player_id`). The smoke test asserts the exact argument objects sent to `invoke`, so a future drift fails a test instead of the running app.

## Typed errors

`models.rs` defines a serializable `CommandError`:

| Kind | Raised when |
| --- | --- |
| `invalid_team` | The `team` argument is not `radiant` or `dire`. |
| `invalid_hero_id` | The hero is not in the loaded roster. |
| `hero_unavailable` | The hero is already picked or banned. |
| `team_full` | A side already has five picks. |
| `invalid_turn` | The supplied order is not a positive integer. |
| `missing_model_file` | `load_ml_model` receives a path that does not exist. |
| `model_error` | The ML engine fails to load or predict. |
| `internal` | Filesystem and other unexpected failures. |

`EngineError` is the engine-internal form; `From<EngineError> for CommandError` maps it at the command boundary. The frontend turns any rejection into a readable message with `describeCommandError` and surfaces it in `HeroPicker`.

## Validation and manual correction

`DraftEngine::validate_pick` and `validate_ban` run before `add_pick`/`add_ban`:

- The hero must exist in `hero_db`.
- The hero must still be in `available_heroes`.
- A side cannot exceed five picks.
- `order` must be at least 1.

Turn state is intentionally not part of validation. Manual drafting is a correction tool, so a pick or ban is accepted even when `current_pick` points at the other side. `HeroPicker` shows a non-blocking `Not your turn` badge instead of disabling the grid.

## Undo and reset

`DraftEngine` keeps an undo stack of `DraftAction::{Pick, Ban}`. `undo_last_action` pops the latest action, removes it from the draft, and returns the hero to `available_heroes`; it returns `None` for an empty stack. `reset` clears the stack and rebuilds `available_heroes` from the roster.

The frontend exposes both through `api.undoLastAction`, `api.resetDraft`, and the two header buttons in `HeroPicker`. After every mutation the picker refetches draft state and recommendations so the UI does not depend on the next GSI event.

## Tests

Rust (`cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib`):

- Unknown and already-used heroes are rejected.
- Invalid order and full teams are rejected.
- Undo restores picks, bans, `pick_number`, and availability.
- Reset clears the undo stack.

Frontend (`npm test --prefix apps/desktop`):

- `manual_pick`, `manual_ban`, `reset_draft`, `undo_last_action`, `get_score_breakdown`, `get_hero_details`, and `load_ml_model` send the expected command names and snake_case arguments.

Current results: 24 Rust library tests pass; 8 frontend tests pass; 15 Python tests pass.

## Verification

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
npm test --prefix apps/desktop
npm run build --prefix apps/desktop
```

## Residual risks

- GSI updates and manual actions share one engine state; undoing a GSI-populated pick is allowed and a later GSI event may re-add it.
- The picker caps a side at five picks but does not enforce the full captains-mode turn schedule.
- `set_config` and `get_config` remain stubs; Step 7 owns persistence.
