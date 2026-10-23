# Game State Integration (GSI)

The Smart Draft Assistant uses Dota 2's built-in Game State Integration (GSI) to track the draft in real-time without resorting to OCR or screen scraping.

## How it Works

1.  **Configuration**: A configuration file (`gamestate_integration_smart_draft.cfg`) is placed in the Dota 2 `cfg/gamestate_integration` directory.
2.  **Server**: The Tauri backend starts a lightweight HTTP server (using `warp`) listening on a local port (e.g., 3000).
3.  **Events**: Dota 2 sends JSON POST requests to this server whenever a change occurs in the game state (hero picked, ban made, game phase changed).

## GSI Payload Mapping

The GSI server maps the incoming JSON to our internal `DraftState` model.

### Key Mappings:
- `map.game_state`: Used to determine if the game is in the `DOTA_GAMERULES_STATE_HERO_SELECTION` phase.
- `hero`: Contains information about the player's current hero.
- `draft`: (If enabled in GSI) Contains the full list of picks and bans.

## Implementation Details

The GSI listener is implemented as an async task in Rust:

```rust
pub async fn start_gsi_server(engine: Arc<DraftEngine>, app_handle: AppHandle) {
    let route = warp::post()
        .and(warp::body::json())
        .map(move |payload: Value| {
            // Process payload
            // Update engine state
            // Emit 'gsi-update' event to frontend
            "OK"
        });

    warp::serve(route).run(([127, 0, 0, 1], 3000)).await;
}
```

## Advantages of GSI

- **Low Latency**: Updates are received almost instantly.
- **Accuracy**: No errors from OCR or visual noise.
- **Robustness**: Does not break with UI updates or different resolutions.
- **Low Overhead**: Minimal CPU/RAM usage compared to screen capture.
