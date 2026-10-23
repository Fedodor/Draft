//! Hero Data Seeder
//! Loads hero data and semantics from JSON into SQLite database

use anyhow::Result;
use smart_draft_assistant::database::Database;
use smart_draft_assistant::models::*;
use smart_draft_assistant::paths::app_data_dir;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Same directory the Tauri app opens (see apps/desktop/src-tauri/src/paths.rs).
    let data_dir = app_data_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not resolve the OS application data directory"))?;

    std::fs::create_dir_all(&data_dir)?;

    let db = Database::new(data_dir)?;
    info!("Database initialized");

    // Load heroes from JSON
    let heroes_json = std::fs::read_to_string("infrastructure/seeders/heroes.json")?;
    let heroes: Vec<Hero> = serde_json::from_str(&heroes_json)?;

    info!("Loaded {} heroes from JSON", heroes.len());

    // Idempotent, transactional insert.
    db.bulk_upsert_heroes(&heroes)?;
    info!("Upserted {} heroes", heroes.len());

    // Load hero semantics from JSON
    let semantics_json = std::fs::read_to_string("infrastructure/seeders/hero_semantics.json")?;
    let semantics: Vec<HeroSemantics> = serde_json::from_str(&semantics_json)?;

    info!("Loaded {} hero semantics from JSON", semantics.len());

    // Idempotent, transactional insert.
    db.bulk_upsert_hero_semantics(&semantics)?;
    info!("Upserted {} hero semantics", semantics.len());

    info!("Seeding complete!");
    Ok(())
}
