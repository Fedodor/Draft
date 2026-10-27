//! Smart Draft Assistant - Tauri Backend
//!
//! Production-quality desktop AI assistant for Dota 2 draft analysis

pub mod commands;
pub mod database;
pub mod engine;
pub mod gsi;
pub mod ml;
pub mod models;
pub mod paths;

use std::sync::Arc;

use engine::DraftEngine;
use gsi::GsiServer;
use tauri::Manager;
use tracing::info;

/// Rotate the diagnostic log once it exceeds this size. One backup is kept.
const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

fn rotate_log_if_needed(log_path: &std::path::Path, max_bytes: u64) {
    let Ok(metadata) = std::fs::metadata(log_path) else {
        return;
    };
    if metadata.len() <= max_bytes {
        return;
    }
    let rotated = log_path.with_extension("log.1");
    let _ = std::fs::remove_file(&rotated);
    let _ = std::fs::rename(log_path, rotated);
}

/// Initialize tracing subscriber writing to a file (and stdout as fallback).
/// This captures the app's `tracing::info!/warn!/error!` calls that
/// `tauri_plugin_log` (which bridges the `log` crate) does not see.
fn init_file_logging() {
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("com.smartdraftassistant.app")
        .join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("app-diagnostics.log");
    rotate_log_if_needed(&log_path, MAX_LOG_BYTES);

    // ort emits ~40 lines of session-init INFO under `ort::logging`; keep that
    // quiet while still surfacing the one-time load notice under
    // `ort::load_dynamic` (e.g. the benign 1.27-vs-1.28 version note).
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort::logging=warn"));

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(file) => {
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .with_writer(std::sync::Mutex::new(file))
                .with_target(true)
                .finish();
            // Set the tracing dispatcher directly instead of .init(), because
            // .init() also installs tracing-log's LogTracer which claims the
            // `log` crate global logger, conflicting with tauri_plugin_log.
            // log-crate records (ort, warp, ...) are handled by tauri_plugin_log.
            let _ = tracing::dispatcher::set_global_default(subscriber.into());
            tracing::info!("Diagnostic log: {}", log_path.display());
        }
        Err(e) => {
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .finish();
            let _ = tracing::dispatcher::set_global_default(subscriber.into());
            tracing::warn!("Could not open diagnostic log file ({}); stdout only", e);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_file_logging();
    info!("Starting Smart Draft Assistant");

    // Create engine with database (will be initialized in setup)
    let draft_engine = Arc::new(std::sync::Mutex::new(DraftEngine::new()));
    let gsi_server = GsiServer::new(draft_engine.clone());

    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .level_for("ort", log::LevelFilter::Warn)
                .level_for("ort::execution_providers", log::LevelFilter::Warn)
                .build(),
        )
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(draft_engine.clone())
        .manage(gsi_server.clone())
        .invoke_handler(tauri::generate_handler![
            commands::get_recommendations,
            commands::get_score_breakdown,
            commands::get_search_recommendations,
            commands::get_draft_state,
            commands::manual_pick,
            commands::manual_ban,
            commands::undo_last_action,
            commands::reset_draft,
            commands::get_hero_list,
            commands::get_hero_details,
            commands::set_config,
            commands::get_config,
            commands::start_gsi,
            commands::stop_gsi,
            commands::restart_gsi,
            commands::get_gsi_status,
            commands::get_gsi_config,
            commands::install_gsi_config,
            commands::predict_draft_value,
            commands::load_ml_model,
            commands::get_ml_status,
            commands::get_data_health,
        ])
        .setup(move |app| {
            let engine = draft_engine.clone();
            let app_handle = app.handle().clone();

            // Initialize database and engine
            let data_dir = app_handle.path().app_data_dir()
                .expect("Failed to get app data dir");
            info!("App data dir: {}", data_dir.display());

            // The seeders resolve this path independently; warn if the Tauri
            // identifier and `paths::APP_IDENTIFIER` ever drift apart.
            if let Some(expected) = crate::paths::app_data_dir() {
                if expected != data_dir {
                    tracing::warn!(
                        "App data dir mismatch: Tauri uses {} but seeders use {}",
                        data_dir.display(),
                        expected.display()
                    );
                }
            }

            // Initialize database
            let database = Arc::new(database::Database::new(data_dir).expect("Failed to initialize database"));
            info!("Database initialized");

            // Share the database with Tauri state so config commands can use it.
            app.manage(database.clone());

            // Set database on engine and initialize
            {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard.set_database(database.clone());
                if let Err(e) = engine_guard.initialize() {
                    tracing::error!("Engine initialization failed: {}", e);
                }
                info!(
                    "Engine loaded: {} heroes, {} semantics, {} matchup entries, {} synergy entries",
                    engine_guard.hero_db.len(),
                    engine_guard.hero_semantics.len(),
                    engine_guard.matchup_data.len(),
                    engine_guard.synergy_data.len(),
                );

                match engine_guard.data_health() {
                    Ok(health) if health.healthy => info!(
                        "Data health OK: schema v{}, {} heroes, {} matchups, {} synergies, patch {}",
                        health.schema_version,
                        health.hero_count,
                        health.matchup_count,
                        health.synergy_count,
                        health.active_patch.as_deref().unwrap_or("unset"),
                    ),
                    Ok(health) => {
                        for issue in &health.issues {
                            tracing::warn!("Data health issue: {}", issue);
                        }
                    }
                    Err(e) => tracing::warn!("Data health check failed: {}", e),
                }
            }

            // Load user settings (port, auth token, model path, auto-start).
            let config = match database.get_app_config() {
                Ok(Some(config)) => config,
                Ok(None) => {
                    info!("No stored config; using defaults");
                    models::AppConfig::default()
                }
                Err(e) => {
                    tracing::warn!("Stored config is unreadable ({}); using defaults", e);
                    models::AppConfig::default()
                }
            };

            // Gate the ML component from persisted settings. This happens before
            // the GSI listener can serve a recommendation request.
            {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard.set_ml_enabled(config.ml_enabled);
            }

            // Load the ONNX model + runtime (non-fatal). A configured model path
            // wins if it exists; otherwise fall back to the bundled resource.
            // ml::load_model locates onnxruntime.dll itself (next to the model,
            // in resources/, or via ORT_DYLIB_PATH) before touching ort.
            // If either resource is absent the engine stays in neutral mode.
            let configured_model = std::path::PathBuf::from(&config.ml_model_path);
            let model_path = if configured_model.is_file() {
                Some(configured_model)
            } else {
                app_handle
                    .path()
                    .resolve(
                        "draft_model_xgboost_full.onnx",
                        tauri::path::BaseDirectory::Resource,
                    )
                    .ok()
            };
            if let Some(model_path) = model_path {
                info!("Loading ONNX model from: {}", model_path.display());
                let _ = crate::ml::get_ml_engine()
                    .write()
                    .load_model(&model_path.to_string_lossy());
            } else {
                info!("ONNX model resource path not resolved");
            }

            // Log ML engine status
            {
                let ml = crate::ml::get_ml_engine();
                let status = ml.read().status();
                info!(
                    "ML status: loaded={}, feature_count={}, model_path={:?}",
                    status.loaded, status.feature_count, status.model_path
                );
            }

            // Apply the configured GSI port/auth and start the listener unless
            // the user disabled auto-start. Bind failures are recorded on the
            // server and surfaced through `get_gsi_status`.
            let gsi_clone = app.state::<GsiServer>().inner().clone();
            gsi_clone.configure(config.gsi_port, config.gsi_auth_token.clone());
            if config.gsi_auto_start {
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = gsi_clone.start(app_handle.clone()).await {
                        tracing::error!("GSI server error: {}", e);
                    }
                });
            } else {
                info!("GSI auto-start disabled in settings");
            }

            info!("Startup complete");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::rotate_log_if_needed;

    #[test]
    fn test_rotate_log_keeps_one_backup_after_size_limit() {
        let dir = std::env::temp_dir().join(format!(
            "sda-log-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("app-diagnostics.log");
        std::fs::write(&log, b"0123456789").unwrap();

        rotate_log_if_needed(&log, 5);
        assert!(!log.exists(), "oversized log should be moved");
        let rotated = log.with_extension("log.1");
        assert_eq!(std::fs::read(&rotated).unwrap(), b"0123456789");

        // A log under the limit is left alone.
        std::fs::write(&log, b"abc").unwrap();
        rotate_log_if_needed(&log, 5);
        assert!(log.exists());
        assert_eq!(std::fs::read(&log).unwrap(), b"abc");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
