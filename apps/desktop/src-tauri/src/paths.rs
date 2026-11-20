//! Single source of truth for the on-disk database location.
//!
//! The app, seeds, and diagnostics all resolve the same directory. Tauri's
//! `app_data_dir()` is `{data_dir}/{identifier}`, so `APP_IDENTIFIER` must stay
//! in sync with `identifier` in `tauri.conf.json`.

use std::path::PathBuf;

/// Must match `identifier` in `tauri.conf.json`.
pub const APP_IDENTIFIER: &str = "com.smartdraftassistant.app";

/// Database filename shared by the app and the seeders.
pub const DB_FILENAME: &str = "smart_draft.db";

/// Overrides the data directory so a scratch seeded database can be driven end
/// to end without touching the installed one. Read here only, so the app and
/// the seeders can never disagree about the path.
pub const DATA_DIR_ENV: &str = "SMART_DRAFT_DATA_DIR";

/// Resolve the application data directory without requiring a Tauri handle.
pub fn app_data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(DATA_DIR_ENV).filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    dirs::data_dir().map(|dir| dir.join(APP_IDENTIFIER))
}

/// Resolve the SQLite path used by the app and the seeders.
pub fn database_path() -> Option<PathBuf> {
    app_data_dir().map(|dir| dir.join(DB_FILENAME))
}
