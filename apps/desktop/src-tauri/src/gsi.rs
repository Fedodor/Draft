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
    /// Last `map.game_state` seen, used to detect a new draft starting after a
    /// finished game.
    last_game_state: Mutex<Option<String>>,
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
        let server_state = self.state.clone();

        let route = warp::post()
            .and(warp::path("gsi"))
            .and(warp::body::json())
            .and(warp::any().map(move || {
                (
                    engine.clone(),
                    app_handle_clone.clone(),
                    auth_token.clone(),
                    server_state.clone(),
                )
            }))
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

/// What ingestion decided to do with one payload. The HTTP handler maps this
/// onto a status code and the frontend events; lifecycle tests assert on it
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GsiOutcome {
    /// An auth token is configured and the payload's token is missing/wrong.
    Unauthorized,
    /// The JSON could not be deserialized into [`GSIGameState`].
    ParseError,
    /// A new draft was detected. The engine was reset and this payload's draft
    /// contents were deliberately skipped (see [`apply_gsi_payload`]).
    DraftReset,
    /// Not a draft phase, or no `draft` component: the engine is unchanged.
    Ignored,
    /// The draft snapshot was reconciled; the counts are actions added.
    DraftApplied { picks: usize, bans: usize },
}

/// Result of applying one payload, including the side the local player is on
/// (emitted as `gsi-team` by the HTTP handler).
struct GsiUpdateResult {
    outcome: GsiOutcome,
    player_side: Option<TeamSide>,
}

/// Ingestion core: auth check -> parse -> new-draft detection -> reconcile.
///
/// This is deliberately free of `tauri::AppHandle` and of event emission so the
/// lifecycle tests can drive exactly the path the warp handler uses. The
/// handler is responsible for turning the outcome into an HTTP status and for
/// emitting `gsi-team` / `gsi-update`.
fn apply_gsi_payload(
    engine: &Arc<std::sync::Mutex<DraftEngine>>,
    server_state: &GsiServerState,
    payload: &Value,
    expected_token: Option<&str>,
) -> GsiUpdateResult {
    // Optional shared-secret check. Dota sends the configured token back in the
    // top-level `auth.token` field; reject anything else before parsing.
    if let Some(expected) = expected_token {
        if request_token(payload) != Some(expected) {
            return GsiUpdateResult {
                outcome: GsiOutcome::Unauthorized,
                player_side: None,
            };
        }
    }

    let gsi_state: GSIGameState = match serde_json::from_value(payload.clone()) {
        Ok(state) => state,
        Err(error) => {
            warn!("Failed to parse GSI payload: {}", error);
            return GsiUpdateResult {
                outcome: GsiOutcome::ParseError,
                player_side: None,
            };
        }
    };

    let player_side = gsi_state.player.as_ref().and_then(|p| p.team_side());

    // Detect the start of a new draft. Entering hero selection from any
    // non-draft state (init, in-game, post-game) means the previous draft's
    // state is stale; reset before ingesting the new one.
    let game_state = gsi_state.map.game_state.clone();
    let reset_needed = {
        let mut last = server_state.last_game_state.lock().unwrap();
        let reset = entering_new_draft(last.as_deref(), &game_state);
        *last = Some(game_state.clone());
        reset
    };
    if reset_needed {
        engine.lock().unwrap().reset();
        // Do not ingest this payload's draft contents: Dota can still include
        // slots from the game that just ended in the first hero-selection
        // update. The caller emits the cleared state and the next update
        // (throttled to 0.1 s by the generated config) populates the new draft.
        return GsiUpdateResult {
            outcome: GsiOutcome::DraftReset,
            player_side,
        };
    }

    if !is_draft_phase(&gsi_state) {
        return GsiUpdateResult {
            outcome: GsiOutcome::Ignored,
            player_side,
        };
    }

    let mut added_picks = 0;
    let mut added_bans = 0;
    if let Some(draft) = &gsi_state.draft {
        debug!(
            "GSI draft update: active_team={:?}, pick={:?}, game_state={}",
            draft.activeteam, draft.pick, game_state
        );
        (added_picks, added_bans) = process_draft_update(engine, draft);
    }

    GsiUpdateResult {
        outcome: GsiOutcome::DraftApplied {
            picks: added_picks,
            bans: added_bans,
        },
        player_side,
    }
}

async fn handle_gsi_update(
    payload: Value,
    (engine, app_handle, auth_token, server_state): (
        Arc<std::sync::Mutex<DraftEngine>>,
        tauri::AppHandle,
        Option<String>,
        Arc<GsiServerState>,
    ),
) -> Result<warp::reply::Response, Rejection> {
    debug!("Received GSI payload");

    let result = apply_gsi_payload(&engine, &server_state, &payload, auth_token.as_deref());

    // Report the local player's side whenever Dota tells us, so the overlay can
    // orient itself even outside the draft phase.
    if let Some(side) = result.player_side {
        let _ = app_handle.emit("gsi-team", side);
    }

    match result.outcome {
        GsiOutcome::Unauthorized => {
            warn!("Rejected GSI payload with missing or invalid auth token");
            return Ok(warp::reply::with_status(
                warp::reply::json(&"UNAUTHORIZED"),
                warp::http::StatusCode::UNAUTHORIZED,
            )
            .into_response());
        }
        GsiOutcome::ParseError => {
            return Ok(warp::reply::json(&"ERROR").into_response());
        }
        GsiOutcome::Ignored => return Ok(warp::reply::json(&"OK").into_response()),
        GsiOutcome::DraftReset => {
            emit_update_event(&engine, &app_handle).await;
            info!("New draft detected (previous state was not a draft); engine reset");
            return Ok(warp::reply::json(&"OK").into_response());
        }
        GsiOutcome::DraftApplied { .. } => {}
    }

    // Emit update event to frontend
    emit_update_event(&engine, &app_handle).await;

    Ok(warp::reply::json(&"OK").into_response())
}

const HERO_SELECTION: &str = "DOTA_GAMERULES_STATE_HERO_SELECTION";
const STRATEGY_TIME: &str = "DOTA_GAMERULES_STATE_STRATEGY_TIME";
const PRE_GAME: &str = "DOTA_GAMERULES_STATE_PRE_GAME";

fn is_draft_phase_name(game_state: &str) -> bool {
    matches!(game_state, HERO_SELECTION | STRATEGY_TIME | PRE_GAME)
}

fn is_draft_phase(state: &GSIGameState) -> bool {
    is_draft_phase_name(&state.map.game_state)
}

/// True when moving from `previous` to `current` starts a fresh draft.
///
/// The draft starts when hero selection begins after a state that is not part
/// of the draft flow (init, wait-for-players, in-game, post-game). Staying in
/// hero selection, or moving to strategy/pre-game time, must not reset.
fn entering_new_draft(previous: Option<&str>, current: &str) -> bool {
    current == HERO_SELECTION && previous.is_some_and(|state| !is_draft_phase_name(state))
}

/// Which team acts, and whether that action is a pick, at a given point.
pub type Turn = (TeamSide, bool);

/// Canonical Captain's Mode action order (20 actions), used as a fallback when
/// GSI omits `activeteam`/`pick`.
///
/// Current sequence: 4 bans -> 4 picks -> 4 bans -> 4 picks -> 2 bans -> 2
/// picks. Within each phase the sides snake, so the team that picked second in
/// the previous pick phase picks first in the next one.
///
///   0-3   bans: Radiant, Dire, Radiant, Dire
///   4-7   picks: Radiant, Dire, Dire, Radiant
///   8-11  bans: Dire, Radiant, Dire, Radiant
///   12-15 picks: Dire, Radiant, Radiant, Dire
///   16-17 bans: Radiant, Dire
///   18-19 picks: Radiant, Dire
pub const CM_ORDER: [Turn; 20] = [
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, true),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
    (TeamSide::Radiant, false),
    (TeamSide::Dire, false),
    (TeamSide::Radiant, true),
    (TeamSide::Dire, true),
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
///
/// `team2`/`team3` are the Radiant/Dire slots, but the authoritative side is
/// the slot's `team_id` (2 = Radiant, 3 = Dire) when it is present. An
/// unrecognized `team_id` falls back to the slot side and is logged by the
/// caller.
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
    for (team_draft, slot_side) in teams {
        let Some(team_draft) = team_draft else {
            continue;
        };
        let side = match team_draft.team_id {
            Some(team_id) => match side_from_team_id(team_id) {
                Some(side) => side,
                None => {
                    warn!(
                        "GSI team slot {} has unrecognized team_id {}; using slot side {:?}",
                        if slot_side == TeamSide::Radiant { 2 } else { 3 },
                        team_id,
                        slot_side
                    );
                    slot_side
                }
            },
            None => slot_side,
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

/// Current Captain's Mode exposes five ban slots per team (`ban0_id` ..
/// `ban4_id`). Ban capacity is not part of the pick validation, so it is
/// enforced here: more than five accumulated bans on one side means stale
/// cross-match slots or a malformed snapshot, not a legal draft. Extra bans are
/// dropped instead of corrupting `available_heroes`.
const MAX_BANS_PER_TEAM: usize = 5;

/// Apply a GSI draft snapshot to the engine.
///
/// The snapshot is treated as append-only: the engine only adds actions it has
/// not seen. Every candidate is validated (known hero, not already picked or
/// banned on either team, team not full) and heroes the user manually undid are
/// suppressed for the rest of the match. Returns how many picks and bans were
/// actually added.
fn reconcile_draft(engine: &DraftEngine, draft: &GSIDraft) -> (usize, usize) {
    let (picks, bans) = draft_actions(draft);
    let state = engine.get_state();

    // A hero can appear at most once in a draft, across both teams and across
    // picks and bans. Keying by hero id (not slot order) prevents the same hero
    // from being counted twice when GSI renumbers slots.
    let mut occupied: std::collections::HashSet<i32> = state
        .radiant_picks
        .iter()
        .chain(state.dire_picks.iter())
        .map(|pick| pick.hero_id.0)
        .chain(state.bans.iter().map(|ban| ban.hero_id.0))
        .collect();

    let mut bans_by_side = [
        state
            .bans
            .iter()
            .filter(|ban| ban.team == TeamSide::Radiant)
            .count(),
        state
            .bans
            .iter()
            .filter(|ban| ban.team == TeamSide::Dire)
            .count(),
    ];

    let mut added_picks = 0;
    let mut added_bans = 0;

    for pick in picks {
        if occupied.contains(&pick.hero_id.0) {
            continue;
        }
        if engine.is_suppressed(&pick.hero_id) {
            debug!("Skipping manually undone hero {} from GSI", pick.hero_id.0);
            continue;
        }
        match engine.validate_pick(&pick) {
            Ok(()) => {
                occupied.insert(pick.hero_id.0);
                engine.add_pick(pick);
                added_picks += 1;
            }
            Err(error) => warn!(
                "Ignoring invalid GSI pick (hero {}, team {:?}): {}",
                pick.hero_id.0, pick.team, error
            ),
        }
    }

    for ban in bans {
        if occupied.contains(&ban.hero_id.0) {
            continue;
        }
        if engine.is_suppressed(&ban.hero_id) {
            debug!("Skipping manually undone hero {} from GSI", ban.hero_id.0);
            continue;
        }
        let side_index = match ban.team {
            TeamSide::Radiant => 0,
            TeamSide::Dire => 1,
        };
        if bans_by_side[side_index] >= MAX_BANS_PER_TEAM {
            warn!(
                "Ignoring GSI ban (hero {}, team {:?}): team already has {} bans",
                ban.hero_id.0, ban.team, bans_by_side[side_index]
            );
            continue;
        }
        match engine.validate_ban(&ban) {
            Ok(()) => {
                occupied.insert(ban.hero_id.0);
                bans_by_side[side_index] += 1;
                engine.add_ban(ban);
                added_bans += 1;
            }
            Err(error) => warn!(
                "Ignoring invalid GSI ban (hero {}, team {:?}): {}",
                ban.hero_id.0, ban.team, error
            ),
        }
    }

    (added_picks, added_bans)
}

fn process_draft_update(
    engine: &Arc<std::sync::Mutex<DraftEngine>>,
    draft: &GSIDraft,
) -> (usize, usize) {
    let engine_guard = engine.lock().unwrap();
    let (added_picks, added_bans) = reconcile_draft(&engine_guard, draft);
    if added_picks > 0 || added_bans > 0 {
        debug!(
            "Applied GSI draft update: +{} picks, +{} bans",
            added_picks, added_bans
        );
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
    debug!(
        "Engine draft state: radiant_picks={}, dire_picks={}, bans={}, turn={:?}",
        state.radiant_picks.len(),
        state.dire_picks.len(),
        state.bans.len(),
        state.current_pick
    );

    (added_picks, added_bans)
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
    use crate::models::{GSITeamDraft, HeroId};

    fn engine_with_heroes(count: i32) -> DraftEngine {
        let mut engine = DraftEngine::new();
        engine.hero_db = (1..=count)
            .map(|id| {
                let mut hero = DraftEngine::fallback_heroes().remove(0);
                hero.id = HeroId(id);
                hero
            })
            .collect();
        engine.state.write().available_heroes =
            engine.hero_db.iter().map(|h| h.id.clone()).collect();
        engine
    }

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
        // The fifth action (index 4) is the Radiant first pick.
        assert_eq!(resolve_turn(&draft, 0, 4), Some((TeamSide::Radiant, true)));
        // The last action (index 19) is the Dire final pick.
        assert_eq!(resolve_turn(&draft, 10, 9), Some((TeamSide::Dire, true)));
        // Beyond the 20-action CM order there is no next turn.
        assert_eq!(resolve_turn(&draft, 10, 10), None);
    }

    #[test]
    fn test_cm_order_matches_current_phase_layout() {
        // 4 bans, 4 picks, 4 bans, 4 picks, 2 bans, 2 picks.
        let phases = [
            (0..4, false),
            (4..8, true),
            (8..12, false),
            (12..16, true),
            (16..18, false),
            (18..20, true),
        ];
        for (range, expected_is_pick) in phases {
            for index in range {
                assert_eq!(
                    CM_ORDER[index].1,
                    expected_is_pick,
                    "action {index} should be {}",
                    if expected_is_pick { "a pick" } else { "a ban" }
                );
            }
        }

        // Exact side order, so a regression in snake order is caught.
        let expected = [
            (TeamSide::Radiant, false),
            (TeamSide::Dire, false),
            (TeamSide::Radiant, false),
            (TeamSide::Dire, false),
            (TeamSide::Radiant, true),
            (TeamSide::Dire, true),
            (TeamSide::Dire, true),
            (TeamSide::Radiant, true),
            (TeamSide::Dire, false),
            (TeamSide::Radiant, false),
            (TeamSide::Dire, false),
            (TeamSide::Radiant, false),
            (TeamSide::Dire, true),
            (TeamSide::Radiant, true),
            (TeamSide::Radiant, true),
            (TeamSide::Dire, true),
            (TeamSide::Radiant, false),
            (TeamSide::Dire, false),
            (TeamSide::Radiant, true),
            (TeamSide::Dire, true),
        ];
        assert_eq!(CM_ORDER, expected);
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

    #[test]
    fn test_entering_new_draft_detection() {
        use super::HERO_SELECTION;
        // Starting the app mid-draft must not wipe the current draft.
        assert!(!entering_new_draft(None, HERO_SELECTION));
        assert!(!entering_new_draft(Some(HERO_SELECTION), HERO_SELECTION));
        assert!(!entering_new_draft(Some(STRATEGY_TIME), HERO_SELECTION));
        assert!(!entering_new_draft(Some(PRE_GAME), HERO_SELECTION));
        // A completed game (or any non-draft state) starts a new draft.
        assert!(entering_new_draft(
            Some("DOTA_GAMERULES_STATE_POST_GAME"),
            HERO_SELECTION
        ));
        assert!(entering_new_draft(
            Some("DOTA_GAMERULES_STATE_GAME_IN_PROGRESS"),
            HERO_SELECTION
        ));
        assert!(entering_new_draft(
            Some("DOTA_GAMERULES_STATE_INIT"),
            HERO_SELECTION
        ));
    }

    #[test]
    fn test_draft_actions_prefers_team_id_over_slot() {
        let mut team2 = GSITeamDraft {
            team_id: Some(3), // Dire, despite living in team2
            ..Default::default()
        };
        team2.pick0_id = 5;
        let mut team3 = GSITeamDraft {
            team_id: Some(2), // Radiant, despite living in team3
            ..Default::default()
        };
        team3.pick0_id = 6;

        let draft = GSIDraft {
            team2: Some(team2),
            team3: Some(team3),
            ..Default::default()
        };
        let (picks, _) = draft_actions(&draft);
        assert_eq!(picks[0].team, TeamSide::Dire);
        assert_eq!(picks[1].team, TeamSide::Radiant);
    }

    #[test]
    fn test_reconcile_dedupes_by_hero_across_picks_and_bans() {
        let engine = engine_with_heroes(10);
        let team2 = GSITeamDraft {
            pick0_id: 1,
            pick1_id: 2,
            ..Default::default()
        };
        let team3 = GSITeamDraft {
            ban0_id: 1, // same hero as a Radiant pick
            ..Default::default()
        };

        let draft = GSIDraft {
            team2: Some(team2),
            team3: Some(team3),
            ..Default::default()
        };

        let (picks, bans) = reconcile_draft(&engine, &draft);
        assert_eq!((picks, bans), (2, 0), "duplicate hero must not be banned");
        assert_eq!(engine.get_state().radiant_picks.len(), 2);
        assert_eq!(engine.get_state().bans.len(), 0);

        // Idempotent: the same snapshot must not add anything again.
        let (picks, bans) = reconcile_draft(&engine, &draft);
        assert_eq!((picks, bans), (0, 0));
        assert_eq!(engine.get_state().radiant_picks.len(), 2);
    }

    #[test]
    fn test_reconcile_skips_unknown_heroes_and_manual_undone_heroes() {
        let engine = engine_with_heroes(10);

        // Manually add then undo hero 1; it becomes suppressed for this match.
        let mut team2 = GSITeamDraft {
            pick0_id: 1,
            ..Default::default()
        };
        let first = GSIDraft {
            team2: Some(team2.clone()),
            ..Default::default()
        };
        assert_eq!(reconcile_draft(&engine, &first), (1, 0));
        assert!(engine.undo_last_action().is_some());

        team2.pick1_id = 2;
        team2.pick2_id = 999; // unknown hero
        let second = GSIDraft {
            team2: Some(team2),
            ..Default::default()
        };
        assert_eq!(reconcile_draft(&engine, &second), (1, 0));
        let state = engine.get_state();
        assert_eq!(state.radiant_picks.len(), 1);
        assert_eq!(state.radiant_picks[0].hero_id.0, 2);
    }

    #[test]
    fn test_reset_restores_available_heroes_and_clears_suppression() {
        let engine = engine_with_heroes(5);
        let team2 = GSITeamDraft {
            pick0_id: 1,
            ..Default::default()
        };
        let draft = GSIDraft {
            team2: Some(team2),
            ..Default::default()
        };
        reconcile_draft(&engine, &draft);
        engine.undo_last_action();
        assert!(engine.is_suppressed(&HeroId(1)));

        engine.reset();
        assert_eq!(engine.get_state().available_heroes.len(), 5);
        assert!(!engine.is_suppressed(&HeroId(1)));
    }
}

#[cfg(test)]
mod lifecycle_tests;
