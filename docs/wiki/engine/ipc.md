# Tauri IPC Conventions

- Type: Engine contract
- Status: Implemented
- Updated: 2026-10-23
- Canonical source: [`../../STEP_04_IPC.md`](../../STEP_04_IPC.md)

## Convention

Every Tauri command declares `#[tauri::command(rename_all = "snake_case")]`, so frontend argument names match Rust parameter names directly (`hero_id`, `model_path`, `player_id`). The `api.ts` smoke test asserts the exact command name and argument object for each manual command.

## Errors

Commands return a serializable `CommandError` with a `kind` tag: `invalid_team`, `invalid_hero_id`, `hero_unavailable`, `team_full`, `invalid_turn`, `missing_model_file`, `model_error`, and `internal`. The frontend converts any rejection to a readable message with `describeCommandError`.

## Manual drafting

`DraftEngine::validate_pick` and `validate_ban` check roster membership, availability, side capacity, and positive order before mutating state. Turn state is not enforced, so a user can correct either side at any time; `HeroPicker` shows an informational `Not your turn` badge instead of disabling the grid.

## Undo and reset

The engine keeps a `DraftAction` undo stack. `undo_last_action` removes the latest pick or ban and returns the hero to the pool; `reset_draft` clears the stack and rebuilds availability. `HeroPicker` refetches draft state and recommendations after each mutation.

## Verification

```powershell
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --lib
npm test --prefix apps/desktop
```

Current results: 24 Rust library tests and 8 frontend tests pass.
