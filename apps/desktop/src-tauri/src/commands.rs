//! Tauri Commands - IPC interface between frontend and backend
//!
//! All commands use `rename_all = "snake_case"` so frontend argument names
//! (`hero_id`, `model_path`, ...) map directly onto Rust parameters.

use crate::database::Database;
use crate::engine::simulation::SearchConfig;
use crate::engine::DraftEngine;
use crate::gsi::GsiServer;
use crate::ml::get_ml_engine;
use crate::models::*;
use std::sync::Arc;
use tauri::{command, AppHandle, Manager, State};
use tracing::{error, info};

fn parse_team(team: &str) -> Result<TeamSide, CommandError> {
    match team {
        "radiant" => Ok(TeamSide::Radiant),
        "dire" => Ok(TeamSide::Dire),
        other => Err(CommandError::InvalidTeam {
            team: other.to_string(),
        }),
    }
}

#[command(rename_all = "snake_case")]
pub async fn get_recommendations(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    team: String,
    limit: Option<usize>,
) -> Result<RecommendationSet, CommandError> {
    let team_side = parse_team(&team)?;
    let limit = limit.unwrap_or(10);
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let engine_guard = engine.lock().unwrap();
        let recommendations = (*engine_guard).get_recommendations_cached(team_side, limit);
        tracing::debug!(
            "get_recommendations(team={}, limit={}): {} categories, top pick: {:?}",
            team,
            limit,
            count_categories(&recommendations),
            recommendations
                .top_picks
                .first()
                .map(|r| (r.hero_id.0, r.score))
        );
        Ok(recommendations)
    })
    .await
    .map_err(|e| CommandError::Internal {
        message: e.to_string(),
    })?
}

#[command(rename_all = "snake_case")]
pub async fn get_score_breakdown(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    hero_id: i32,
    team: String,
) -> Result<ScoreBreakdown, CommandError> {
    let team_side = parse_team(&team)?;
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let engine_guard = engine.lock().unwrap();
        (*engine_guard)
            .score_breakdown(&HeroId(hero_id), team_side)
            .ok_or(CommandError::InvalidHeroId { hero_id })
    })
    .await
    .map_err(|e| CommandError::Internal {
        message: e.to_string(),
    })?
}

fn count_categories(recs: &RecommendationSet) -> usize {
    recs.top_picks.len()
        + recs.safe_picks.len()
        + recs.aggressive_picks.len()
        + recs.comfort_picks.len()
        + recs.anti_meta_picks.len()
}

#[command(rename_all = "snake_case")]
pub async fn get_search_recommendations(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    team: String,
    limit: Option<usize>,
    config: Option<SearchConfig>,
) -> Result<Vec<crate::engine::simulation::SimulationResult>, CommandError> {
    let team_side = parse_team(&team)?;
    let limit = limit.unwrap_or(10);
    let config = config.unwrap_or_default().sanitized();
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let engine_guard = engine.lock().unwrap();
        Ok((*engine_guard).get_recommendations_with_search(team_side, limit, config))
    })
    .await
    .map_err(|e| CommandError::Internal {
        message: e.to_string(),
    })?
}

#[command(rename_all = "snake_case")]
pub async fn get_draft_state(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
) -> Result<DraftState, CommandError> {
    let engine_guard = engine.lock().unwrap();
    Ok((*engine_guard).get_state())
}

#[command(rename_all = "snake_case")]
pub async fn manual_pick(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    hero_id: i32,
    team: String,
    player_id: Option<i64>,
    order: Option<i32>,
) -> Result<(), CommandError> {
    let team_side = parse_team(&team)?;
    let engine_guard = engine.lock().unwrap();
    let state = (*engine_guard).get_state();
    let pick = DraftPick {
        hero_id: HeroId(hero_id),
        player_id: player_id.unwrap_or(0).into(),
        order: order.unwrap_or(state.pick_number + 1),
        team: team_side,
    };

    (*engine_guard)
        .validate_pick(&pick)
        .map_err(CommandError::from)?;
    (*engine_guard).lift_suppression(&pick.hero_id);
    (*engine_guard).add_pick(pick);
    info!("Manual pick added: hero_id={}, team={}", hero_id, team);
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn manual_ban(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    hero_id: i32,
    team: String,
    order: Option<i32>,
) -> Result<(), CommandError> {
    let team_side = parse_team(&team)?;
    let engine_guard = engine.lock().unwrap();
    let state = (*engine_guard).get_state();
    let ban = DraftBan {
        hero_id: HeroId(hero_id),
        order: order.unwrap_or(state.bans.len() as i32 + 1),
        team: team_side,
    };

    (*engine_guard)
        .validate_ban(&ban)
        .map_err(CommandError::from)?;
    (*engine_guard).lift_suppression(&ban.hero_id);
    (*engine_guard).add_ban(ban);
    info!("Manual ban added: hero_id={}, team={}", hero_id, team);
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn undo_last_action(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
) -> Result<Option<DraftAction>, CommandError> {
    let engine_guard = engine.lock().unwrap();
    Ok((*engine_guard).undo_last_action())
}

#[command(rename_all = "snake_case")]
pub async fn reset_draft(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
) -> Result<(), CommandError> {
    let engine_guard = engine.lock().unwrap();
    (*engine_guard).reset();
    info!("Draft reset");
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn get_hero_list(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
) -> Result<Vec<Hero>, CommandError> {
    let engine_guard = engine.lock().unwrap();
    Ok(engine_guard.hero_db.clone())
}

#[command(rename_all = "snake_case")]
pub async fn get_hero_details(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    hero_id: i32,
) -> Result<Option<Hero>, CommandError> {
    let engine_guard = engine.lock().unwrap();
    Ok(engine_guard
        .hero_db
        .iter()
        .find(|h| h.id.0 == hero_id)
        .cloned())
}

#[command(rename_all = "snake_case")]
pub async fn set_config(
    db: State<'_, Arc<Database>>,
    gsi: State<'_, GsiServer>,
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    config: AppConfig,
) -> Result<(), CommandError> {
    config
        .validate()
        .map_err(|message| CommandError::Internal { message })?;
    db.set_app_config(&config)?;
    // Apply the GSI port/token to the server for the next start. A running
    // listener keeps its current binding until `restart_gsi` is called.
    gsi.inner()
        .configure(config.gsi_port, config.gsi_auth_token.clone());
    // Apply ML gating immediately; the model itself is loaded at startup.
    {
        let mut engine_guard = engine.lock().unwrap();
        engine_guard.set_ml_enabled(config.ml_enabled);
    }
    info!(
        "Config saved (gsi_port={}, ml_enabled={})",
        config.gsi_port, config.ml_enabled
    );
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn get_config(db: State<'_, Arc<Database>>) -> Result<AppConfig, CommandError> {
    match db.get_app_config() {
        Ok(Some(config)) => Ok(config),
        Ok(None) => Ok(AppConfig::default()),
        Err(e) => Err(CommandError::Internal {
            message: format!("failed to read stored config: {e}"),
        }),
    }
}

#[command(rename_all = "snake_case")]
pub async fn restart_gsi(gsi: State<'_, GsiServer>, app: AppHandle) -> Result<(), CommandError> {
    let server = gsi.inner().clone();
    server.stop();
    // Give the graceful shutdown a moment to release the port before rebinding.
    for _ in 0..40 {
        if !server.is_running() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let start_server = server.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = start_server.start(app).await {
            error!("GSI restart failed: {}", e);
        }
    });
    info!("GSI restart requested on port {}", server.port());
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn start_gsi(gsi: State<'_, GsiServer>, app: AppHandle) -> Result<(), CommandError> {
    let gsi_server = gsi.inner().clone();
    if gsi_server.is_running() {
        return Ok(());
    }
    tauri::async_runtime::spawn(async move {
        if let Err(e) = gsi_server.start(app).await {
            error!("GSI server error: {}", e);
        }
    });
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn stop_gsi(gsi: State<'_, GsiServer>) -> Result<(), CommandError> {
    gsi.inner().stop();
    info!("GSI server stop requested");
    Ok(())
}

#[command(rename_all = "snake_case")]
pub async fn get_gsi_status(
    gsi: State<'_, GsiServer>,
) -> Result<crate::gsi::GsiStatus, CommandError> {
    Ok(gsi.inner().status())
}

#[command(rename_all = "snake_case")]
pub async fn get_gsi_config(gsi: State<'_, GsiServer>) -> Result<String, CommandError> {
    let port = gsi.inner().port();
    let token = gsi.inner().auth_token();
    Ok(crate::gsi::generate_gsi_config_with_auth(
        port,
        token.as_deref(),
    ))
}

#[command(rename_all = "snake_case")]
pub async fn install_gsi_config(gsi: State<'_, GsiServer>) -> Result<String, CommandError> {
    use std::io::Write;

    let config_dir = crate::gsi::dota_gsi_config_dir().ok_or_else(|| CommandError::Internal {
        message: "Dota 2 config directory not found on this platform".to_string(),
    })?;
    std::fs::create_dir_all(&config_dir).map_err(|e| CommandError::Internal {
        message: format!("Failed to create config dir: {}", e),
    })?;

    let config_path = config_dir.join(crate::gsi::GSI_CONFIG_FILENAME);
    let token = gsi.inner().auth_token();
    let content = crate::gsi::generate_gsi_config_with_auth(gsi.inner().port(), token.as_deref());

    let mut file = std::fs::File::create(&config_path).map_err(|e| CommandError::Internal {
        message: format!("Failed to create config file: {}", e),
    })?;
    file.write_all(content.as_bytes())
        .map_err(|e| CommandError::Internal {
            message: format!("Failed to write config file: {}", e),
        })?;

    info!("GSI config installed to {:?}", config_path);
    Ok(config_path.to_string_lossy().to_string())
}

#[command(rename_all = "snake_case")]
pub async fn predict_draft_value(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
    team: String,
) -> Result<f32, CommandError> {
    let is_radiant = parse_team(&team)? == TeamSide::Radiant;
    let engine = engine.inner().clone();
    let ml_engine = get_ml_engine();
    tauri::async_runtime::spawn_blocking(move || {
        // Snapshot engine data first and drop its lock before touching the ML
        // engine. This keeps a single lock order (engine -> ML) across all
        // callers and avoids holding two global locks at once.
        let (state, hero_db, hero_semantics, meta_data, matchup_data) = {
            let engine_lock = engine.lock().unwrap();
            (
                (*engine_lock).get_state(),
                engine_lock.hero_db.clone(),
                engine_lock.hero_semantics.clone(),
                engine_lock.meta_data.clone(),
                engine_lock.matchup_data.clone(),
            )
        };

        let ml_engine_guard = ml_engine.read();
        if !ml_engine_guard.status().loaded {
            return Err(CommandError::ModelError {
                message: "ML model is not loaded".to_string(),
            });
        }

        let feature_schema = ml_engine_guard.feature_schema();
        let features = crate::ml::build_feature_map(
            &state,
            &hero_db,
            &hero_semantics,
            &meta_data,
            &matchup_data,
            feature_schema.as_deref(),
        );

        let prediction =
            ml_engine_guard
                .predict(&features)
                .map_err(|e| CommandError::ModelError {
                    message: e.to_string(),
                })?;

        Ok(if is_radiant {
            prediction
        } else {
            1.0 - prediction
        })
    })
    .await
    .map_err(|e| CommandError::Internal {
        message: e.to_string(),
    })?
}

#[command(rename_all = "snake_case")]
pub async fn get_data_health(
    engine: State<'_, Arc<std::sync::Mutex<DraftEngine>>>,
) -> Result<DataHealth, CommandError> {
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let engine_guard = engine.lock().unwrap();
        (*engine_guard).data_health()
    })
    .await
    .map_err(|e| CommandError::Internal {
        message: e.to_string(),
    })?
}

#[command(rename_all = "snake_case")]
pub fn load_ml_model(app: AppHandle, model_path: String) -> Result<(), CommandError> {
    let canonical = canonical_model_path(&app, &model_path)?;

    let ml_engine = get_ml_engine();
    let result = ml_engine
        .write()
        .load_model(&canonical.to_string_lossy())
        .map_err(|e| CommandError::ModelError {
            message: e.to_string(),
        });
    result
}

/// Resolve a user-supplied model path and reject anything outside the roots the
/// app owns. This prevents loading an arbitrary ONNX file (and, with it, an
/// `onnxruntime.dll` sitting next to it) from an untrusted directory.
fn canonical_model_path(
    app: &AppHandle,
    requested: &str,
) -> Result<std::path::PathBuf, CommandError> {
    let path = std::path::Path::new(requested);
    if !path.exists() {
        return Err(CommandError::MissingModelFile {
            path: requested.to_string(),
        });
    }

    let canonical = path.canonicalize().map_err(|e| CommandError::Internal {
        message: format!("Failed to resolve model path: {e}"),
    })?;

    let mut roots = Vec::new();
    if let Ok(dir) = app.path().app_data_dir() {
        roots.push(dir);
    }
    if let Ok(dir) = app.path().resource_dir() {
        roots.push(dir);
    }
    if let Ok(dir) = std::env::current_dir() {
        roots.push(dir.join("ml").join("models"));
    }
    // Development fallback for `cargo tauri dev` from the repository root.
    roots.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ml/models"));

    let allowed = roots
        .iter()
        .filter_map(|root| root.canonicalize().ok())
        .any(|root| canonical.starts_with(&root));

    if !allowed {
        return Err(CommandError::Internal {
            message: format!(
                "Model path is outside the allowed model directories: {}",
                canonical.display()
            ),
        });
    }

    Ok(canonical)
}

#[command(rename_all = "snake_case")]
pub fn get_ml_status() -> Result<crate::ml::MLStatus, CommandError> {
    let ml_engine = get_ml_engine();
    let status = ml_engine.read().status();
    Ok(status)
}
