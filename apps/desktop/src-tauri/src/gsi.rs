//! Game State Integration (GSI) Server
//!
//! Receives real-time game state from Dota 2 client via HTTP POST

use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use tauri::Emitter;
use tracing::{debug, info, warn};
use warp::{Filter, Rejection, Reply};

use crate::engine::DraftEngine;
use crate::models::{GSIDraft, GSIGameState, TeamSide};

/// Bind metadata and configuration for the GSI listener.
#[derive(Debug, Clone)]
pub struct GsiConfig {
    pub port: u16,
    pub auth_token: Option<String>,
}

impl Default for GsiConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            auth_token: None,
        }
    }
}

/// Shared server state, cloned across GsiServer clones
#[derive(Default)]
struct GsiServerState {
    running: AtomicBool,
    last_error: Mutex<Option<String>>,
}

/// GSI server binding metadata exposed to commands/frontend
#[derive(Debug, Clone, serde::Serialize)]
pub struct GsiStatus {
    pub running: bool,
    pub port: u16,
    pub endpoint: String,
    pub auth_required: bool,
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct GsiServer {
    engine: Arc<std::sync::Mutex<DraftEngine>>,
    config: Arc<RwLock<GsiConfig>>,
    state: Arc<GsiServerState>,
    /// Replaced on every `start` so the server can be restarted on a new port.
    shutdown: Arc<Mutex<Option<tokio::sync::watch::Sender<bool>>>>,
}

impl GsiServer {
    pub fn new(engine: Arc<std::sync::Mutex<DraftEngine>>) -> Self {
        Self {
            engine,
            config: Arc::new(RwLock::new(GsiConfig::default())),
            state: Arc::new(GsiServerState::default()),
            shutdown: Arc::new(Mutex::new(None)),
        }
    }

    /// Update the port and auth token used by the next `start`.
    ///
    /// Takes effect on the next start; calling it while running only affects a
    /// later restart, so the change is logged.
    pub fn configure(&self, port: u16, auth_token: Option<String>) {
        let mut config = self.config.write().unwrap();
        config.port = port;
        config.auth_token = auth_token;
        if self.is_running() {
            info!("GSI config updated; restart required to apply");
        }
    }

    pub fn is_running(&self) -> bool {
        self.state.running.load(Ordering::SeqCst)
    }

    pub fn last_error(&self) -> Option<String> {
        self.state.last_error.lock().unwrap().clone()
    }

    pub fn port(&self) -> u16 {
        self.config.read().unwrap().port
    }

    pub fn auth_token(&self) -> Option<String> {
        self.config.read().unwrap().auth_token.clone()
    }

    pub fn status(&self) -> GsiStatus {
        let config = self.config.read().unwrap();
        GsiStatus {
            running: self.is_running(),
            port: config.port,
            endpoint: format!("http://127.0.0.1:{}/gsi", config.port),
            auth_required: config.auth_token.is_some(),
            last_error: self.last_error(),
        }
    }

    /// Start the GSI server. Idempotent — calling while already running is a no-op.
    ///
    /// Binds to `127.0.0.1:<port>`; a bind failure (for example the port is
    /// already in use) is returned as `Err` and recorded in `last_error`
    /// instead of panicking.
    pub async fn start(
        &self,
        app_handle: tauri::AppHandle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.state.running.swap(true, Ordering::SeqCst) {
            info!("GSI server already running on port {}", self.port());
            return Ok(());
        }
        *self.state.last_error.lock().unwrap() = None;

        let (port, auth_token) = {
            let config = self.config.read().unwrap();
            (config.port, config.auth_token.clone())
        };

        info!("Starting GSI server on port {}", port);

        let engine = self.engine.clone();
        let app_handle_clone = app_handle.clone();

        let route = warp::post()
            .and(warp::path("gsi"))
            .and(warp::body::json())
            .and(
                warp::any()
                    .map(move || (engine.clone(), app_handle_clone.clone(), auth_token.clone())),
            )
            .and_then(handle_gsi_update);

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        // `try_bind_*` returns an error instead of panicking when the port is
        // unavailable, which is what we surface to the user.
        let (addr, server) = match warp::serve(route)
            .try_bind_with_graceful_shutdown(([127, 0, 0, 1], port), shutdown_signal(shutdown_rx))
        {
            Ok(bound) => bound,
            Err(e) => {
                let message = format!("Failed to bind GSI port {port}: {e}");
                warn!("{}", message);
                *self.state.last_error.lock().unwrap() = Some(message.clone());
                self.state.running.store(false, Ordering::SeqCst);
                return Err(message.into());
            }
        };
        info!("GSI server listening on http://{}/gsi", addr);

        *self.shutdown.lock().unwrap() = Some(shutdown_tx);

        // Wait for the server (which exits when the shutdown signal fires).
        server.await;
        self.state.running.store(false, Ordering::SeqCst);
        *self.shutdown.lock().unwrap() = None;
        info!("GSI server stopped on port {}", port);
        Ok(())
    }

    /// Signal the running server to shut down gracefully.
    pub fn stop(&self) {
        if let Some(tx) = self.shutdown.lock().unwrap().take() {
            let _ = tx.send(true);
        }
    }
}

async fn shutdown_signal(mut rx: tokio::sync::watch::Receiver<bool>) {
    loop {
        if *rx.borrow() {
            break;
        }
        if rx.changed().await.is_err() {
            break;
        }
    }
}

/// Resolve the local player's side from a raw GSI payload's `auth.token`.
fn request_token(payload: &Value) -> Option<&str> {
    payload
        .get("auth")
        .and_then(|auth| auth.get("token"))
        .and_then(|token| token.as_str())
}

async fn handle_gsi_update(
    payload: Value,
    (engine, app_handle, auth_token): (
        Arc<std::sync::Mutex<DraftEngine>>,
        tauri::AppHandle,
        Option<String>,
    ),
) -> Result<warp::reply::Response, Rejection> {
    // Optional shared-secret check. Dota sends the configured token back in the
    // top-level `auth.token` field; reject anything else before parsing.
    if let Some(expected) = auth_token.as_deref() {
        if request_token(&payload) != Some(expected) {
            warn!("Rejected GSI payload with missing or invalid auth token");
            return Ok(warp::reply::with_status(
                warp::reply::json(&"UNAUTHORIZED"),
                warp::http::StatusCode::UNAUTHORIZED,
            )
            .into_response());
        }
    }

    debug!("Received GSI payload");

    // Parse the GSI game state
    let gsi_state: GSIGameState = match serde_json::from_value(payload.clone()) {
        Ok(state) => state,
        Err(e) => {
            warn!("Failed to parse GSI payload: {}", e);
            return Ok(warp::reply::json(&"ERROR").into_response());
        }
    };

    // Report the local player's side whenever Dota tells us, so the overlay can
    // orient itself even outside the draft phase.
    if let Some(side) = gsi_state.player.as_ref().and_then(|p| p.team_side()) {
        let _ = app_handle.emit("gsi-team", side);
    }

    // Check if we're in hero selection phase
    if !is_draft_phase(&gsi_state) {
        return Ok(warp::reply::json(&"OK").into_response());
    }

    // Process draft updates
    if let Some(draft) = &gsi_state.draft {
        info!(
            "GSI draft update: active_team={:?}, pick={:?}, game_state={}",
            draft.activeteam, draft.pick, gsi_state.map.game_state
        );
        process_draft_update(&engine, draft).await;
    }

    // Emit update event to frontend
    emit_update_event(&engine, &app_handle).await;

    Ok(warp::reply::json(&"OK").into_response())
}

fn is_draft_phase(state: &GSIGameState) -> bool {
    const HERO_SELECTION: &str = "DOTA_GAMERULES_STATE_HERO_SELECTION";
    const STRATEGY_TIME: &str = "DOTA_GAMERULES_STATE_STRATEGY_TIME";
    const PRE_GAME: &str = "DOTA_GAMERULES_STATE_PRE_GAME";

    matches!(
        state.map.game_state.as_str(),
        HERO_SELECTION | STRATEGY_TIME | PRE_GAME
    )
}

/// Which team acts, and whether that action is a pick, at a given point.
pub type Turn = (TeamSide, bool);

/// Canonical Captain's Mode action order (20 actions): six bans, six picks,
/// four bans, four picks. Used as a fallback when GSI omits `activeteam`.
pub const CM_ORDER: [Turn; 20] = [
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, true),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, true),
];

/// Map a GSI `activeteam` value onto a side (2 = Radiant, 3 = Dire).
pub fn side_from_team_id(team_id: i32) -> Option<TeamSide> {
    match team_id {
        2 => Some(TeamSide::Radiant),
        3 => Some(TeamSide::Dire),
        _ => None,
    }
}

/// Determine the next action from GSI directly when possible, falling back to
/// the canonical CM order using the number of completed picks and bans.
pub fn resolve_turn(draft: &GSIDraft, picks: usize, bans: usize) -> Option<Turn> {
    if let Some(side) = draft.activeteam.and_then(side_from_team_id) {
        if let Some(is_pick) = draft.pick {
            return Some((side, is_pick));
        }
    }
    CM_ORDER.get(picks + bans).copied()
}

/// Expand a real GSI draft into ordered engine picks and bans.
pub fn draft_actions(
    draft: &GSIDraft,
) -> (Vec<crate::models::DraftPick>, Vec<crate::models::DraftBan>) {
    use crate::models::{DraftBan, DraftPick, HeroId, PlayerId};

    let mut picks = Vec::new();
    let mut bans = Vec::new();

    let teams = [
        (draft.team2.as_ref(), TeamSide::Radiant),
        (draft.team3.as_ref(), TeamSide::Dire),
    ];
    for (team_draft, side) in teams {
        let Some(team_draft) = team_draft else {
            continue;
        };
        for (index, hero_id) in team_draft.pick_ids().into_iter().enumerate() {
            picks.push(DraftPick {
                hero_id: HeroId(hero_id),
                player_id: PlayerId(0),
                order: index as i32 + 1,
                team: side,
            });
        }
        for (index, hero_id) in team_draft.ban_ids().into_iter().enumerate() {
            bans.push(DraftBan {
                hero_id: HeroId(hero_id),
                order: index as i32 + 1,
                team: side,
            });
        }
    }

    (picks, bans)
}

async fn process_draft_update(engine: &Arc<std::sync::Mutex<DraftEngine>>, draft: &GSIDraft) {
    let engine_guard = engine.lock().unwrap();
    let (picks, bans) = draft_actions(draft);

    // Track processed picks/bans to avoid duplicates
    {
        let state = engine_guard.get_state();

        let existing_picks: std::collections::HashSet<_> = state
            .radiant_picks
            .iter()
            .chain(state.dire_picks.iter())
            .map(|p| (p.hero_id.0, p.team, p.order))
            .collect();

        let existing_bans: std::collections::HashSet<_> = state
            .bans
            .iter()
            .map(|b| (b.hero_id.0, b.team, b.order))
            .collect();

        for pick in &picks {
            let key = (pick.hero_id.0, pick.team, pick.order);
            if !existing_picks.contains(&key) {
                engine_guard.add_pick(pick.clone());
            }
        }

        for ban in &bans {
            let key = (ban.hero_id.0, ban.team, ban.order);
            if !existing_bans.contains(&key) {
                engine_guard.add_ban(ban.clone());
            }
        }
    }

    // Turn state machine driven by the game itself (`activeteam`/`pick`), with a
    // documented CM-order fallback.
    let mut state = engine_guard.state.write();
    let picks_count = state.radiant_picks.len() + state.dire_picks.len();
    let bans_count = state.bans.len();

    match resolve_turn(draft, picks_count, bans_count) {
        Some((side, is_pick)) => {
            state.current_pick = Some(side);
            state.is_picking_phase = is_pick;
        }
        None => {
            state.current_pick = None;
            state.is_picking_phase = false;
        }
    }
    state.pick_number = picks_count as i32;
    info!(
        "Engine draft state: radiant_picks={}, dire_picks={}, bans={}, turn={:?}",
        state.radiant_picks.len(),
        state.dire_picks.len(),
        state.bans.len(),
        state.current_pick
    );
}

async fn emit_update_event(
    engine: &Arc<std::sync::Mutex<DraftEngine>>,
    app_handle: &tauri::AppHandle,
) {
    let engine_guard = engine.lock().unwrap();
    let state = engine_guard.get_state();

    // Emit Tauri event to frontend
    let _ = app_handle.emit("gsi-update", &state);

    debug!(
        "Draft state updated: {} picks, {} bans",
        state.radiant_picks.len() + state.dire_picks.len(),
        state.bans.len()
    );
}

/// Generate GSI configuration file content for Dota 2.
pub fn generate_gsi_config(port: u16) -> String {
    generate_gsi_config_with_auth(port, None)
}

/// Generate GSI config, including an `auth` block when a token is set. Dota
/// echoes the token back in every payload so the listener can reject spoofed
/// requests.
pub fn generate_gsi_config_with_auth(port: u16, auth_token: Option<&str>) -> String {
    let auth_block = match auth_token {
        Some(token) => format!("    \"auth\"\n    {{\n        \"token\"\t\t\"{token}\"\n    }}\n"),
        None => String::new(),
    };

    format!(
        r#""GameStateIntegration"
{{
    "uri"			"http://127.0.0.1:{port}/gsi"
    "timeout"		"5"
    "buffer"		"0.1"
    "throttle"		"0.1"
    "heartbeat"		"30"
{auth_block}    "data"
    {{
        "provider"		"1"
        "map"			"1"
        "player"		"1"
        "hero"			"1"
        "abilities"		"0"
        "items"			"0"
        "draft"			"1"
        "wearables"		"0"
    }}
}}
"#
    )
}

/// Default GSI config file name expected by Dota 2
pub const GSI_CONFIG_FILENAME: &str = "gamestate_integration_smart_draft.cfg";

/// Resolve the Dota 2 gamestate_integration config directory for the current platform.
pub fn dota_gsi_config_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return Some(
                std::path::PathBuf::from(appdata)
                    .join("dota 2 beta")
                    .join("game")
                    .join("dota")
                    .join("cfg")
                    .join("gamestate_integration"),
            );
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return Some(
                std::path::PathBuf::from(home)
                    .join(".local/share/Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration"),
            );
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return Some(
                std::path::PathBuf::from(home)
                    .join("Library/Application Support/Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration"),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GSITeamDraft;

    #[test]
    fn test_gsi_config_generation() {
        let config = generate_gsi_config(3000);
        assert!(config.contains("http://127.0.0.1:3000/gsi"));
        assert!(config.contains("\"draft\"			\"1\""));
        assert!(!config.contains("\"auth\""));
    }

    #[test]
    fn test_gsi_config_with_auth_includes_token() {
        let config = generate_gsi_config_with_auth(4000, Some("secret-token"));
        assert!(config.contains("http://127.0.0.1:4000/gsi"));
        assert!(config.contains("\"auth\""));
        assert!(config.contains("\"secret-token\""));
    }

    #[test]
    fn test_gsi_server_is_not_running_by_default() {
        let server = GsiServer::new(Arc::new(std::sync::Mutex::new(DraftEngine::new())));
        assert!(!server.is_running());
        let status = server.status();
        assert_eq!(status.port, 3000);
        assert!(status.endpoint.ends_with("/gsi"));
        assert!(!status.auth_required);
        assert!(status.last_error.is_none());
    }

    #[test]
    fn test_gsi_server_configure_updates_port_and_auth() {
        let server = GsiServer::new(Arc::new(std::sync::Mutex::new(DraftEngine::new())));
        server.configure(4567, Some("tok".to_string()));
        let status = server.status();
        assert_eq!(status.port, 4567);
        assert!(status.auth_required);
        assert!(status.endpoint.contains(":4567"));
    }

    #[test]
    fn test_side_from_team_id() {
        assert_eq!(side_from_team_id(2), Some(TeamSide::Radiant));
        assert_eq!(side_from_team_id(3), Some(TeamSide::Dire));
        assert_eq!(side_from_team_id(0), None);
        assert_eq!(side_from_team_id(1), None);
    }

    #[test]
    fn test_resolve_turn_prefers_activeteam() {
        let draft = GSIDraft {
            activeteam: Some(3),
            pick: Some(true),
            ..Default::default()
        };
        assert_eq!(resolve_turn(&draft, 0, 0), Some((TeamSide::Dire, true)));
    }

    #[test]
    fn test_resolve_turn_falls_back_to_cm_order() {
        let draft = GSIDraft::default();
        // First action of CM is a Radiant ban.
        assert_eq!(resolve_turn(&draft, 0, 0), Some((TeamSide::Radiant, false)));
        // The 7th action (index 6) is the Radiant first pick.
        assert_eq!(resolve_turn(&draft, 0, 6), Some((TeamSide::Radiant, true)));
        // Beyond the 20-action CM order there is no next turn.
        assert_eq!(resolve_turn(&draft, 10, 10), None);
    }

    #[test]
    fn test_cm_order_has_five_picks_and_five_bans_per_side() {
        let radiant_picks = CM_ORDER
            .iter()
            .filter(|(side, pick)| *side == TeamSide::Radiant && *pick)
            .count();
        let dire_picks = CM_ORDER
            .iter()
            .filter(|(side, pick)| *side == TeamSide::Dire && *pick)
            .count();
        let radiant_bans = CM_ORDER
            .iter()
            .filter(|(side, pick)| *side == TeamSide::Radiant && !*pick)
            .count();
        let dire_bans = CM_ORDER
            .iter()
            .filter(|(side, pick)| *side == TeamSide::Dire && !*pick)
            .count();
        assert_eq!((radiant_picks, dire_picks), (5, 5));
        assert_eq!((radiant_bans, dire_bans), (5, 5));
    }

    #[test]
    fn test_draft_actions_reads_real_gsi_slots() {
        let mut team2 = GSITeamDraft {
            team_id: Some(2),
            ..Default::default()
        };
        team2.pick0_id = 5;
        team2.pick1_id = 0; // empty slot is skipped
        team2.pick2_id = 8;
        team2.ban0_id = 11;

        let mut team3 = GSITeamDraft {
            team_id: Some(3),
            ..Default::default()
        };
        team3.pick0_id = 20;
        team3.ban4_id = 30;

        let draft = GSIDraft {
            team2: Some(team2),
            team3: Some(team3),
            ..Default::default()
        };

        let (picks, bans) = draft_actions(&draft);
        assert_eq!(picks.len(), 3);
        assert_eq!(picks[0].hero_id.0, 5);
        assert_eq!(picks[0].team, TeamSide::Radiant);
        assert_eq!(picks[0].order, 1);
        assert_eq!(picks[1].hero_id.0, 8);
        assert_eq!(picks[1].order, 2);
        assert_eq!(picks[2].hero_id.0, 20);
        assert_eq!(picks[2].team, TeamSide::Dire);
        assert_eq!(picks[2].order, 1);

        assert_eq!(bans.len(), 2);
        assert_eq!(bans[0].hero_id.0, 11);
        assert_eq!(bans[0].team, TeamSide::Radiant);
        assert_eq!(bans[1].hero_id.0, 30);
        assert_eq!(bans[1].team, TeamSide::Dire);
        assert_eq!(bans[1].order, 1);
    }
}
