//! Matchup & Synergy Data Seeder
//!
//! By default this loads real, data-derived rows from
//! `infrastructure/seeders/scoring_seed.json` (generated with
//! `ml/training/export_scoring_seed.py`).
//!
//! Pass `--synthetic` to generate deterministic placeholder data for UI work
//! when the processed datasets are unavailable. Synthetic rows are for
//! development only and are never selected silently.

use anyhow::Result;
use serde::Deserialize;
use smart_draft_assistant::database::Database;
use smart_draft_assistant::models::{Hero, HeroMatchup, HeroSynergy};
use smart_draft_assistant::paths::app_data_dir;
use std::env;
use std::path::Path;
use tracing::{info, warn};

const SEED_PATH: &str = "infrastructure/seeders/scoring_seed.json";

#[derive(Deserialize)]
struct ScoringSeed {
    #[serde(default)]
    patch_version: Option<String>,
    matchups: Vec<HeroMatchup>,
    synergies: Vec<HeroSynergy>,
}

const SYNTHETIC_PATCH: &str = "synthetic";

/// FNV-1a style hash over a pair of i32s, returns a u32.
fn pair_hash(a: i32, b: i32) -> u32 {
    let mut h: u32 = 2166136261;
    for byte in a.to_ne_bytes().iter().chain(b.to_ne_bytes().iter()) {
        h ^= *byte as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Map a hash into a float in [lo, hi].
fn hash_to_range(h: u32, lo: f32, hi: f32) -> f32 {
    let t = (h % 10000) as f32 / 10000.0;
    lo + t * (hi - lo)
}

/// Rough role complementarity between two heroes.
/// Returns a value in [-1, 1]: positive when roles differ (complementary),
/// negative when they heavily overlap (redundant).
fn role_complementarity(a: &Hero, b: &Hero) -> f32 {
    let shared = a.roles.iter().filter(|r| b.roles.contains(r)).count() as f32;
    let total = (a.roles.len().max(1) + b.roles.len().max(1)) as f32;
    let overlap = shared / total; // 0 (no overlap) .. ~0.5 (full overlap)
                                  // No overlap -> +0.6, full overlap -> -0.6
    0.6 - overlap * 1.2
}

fn seed_synthetic(db: &Database, heroes: &[Hero]) -> Result<()> {
    warn!(
        "SYNTHETIC MODE: writing placeholder matchup/synergy data; do not use \
         these values for production recommendations"
    );

    let n = heroes.len();

    // === Matchups (ordered pairs) ===
    let mut matchups: Vec<HeroMatchup> = Vec::with_capacity(n * (n - 1));
    for (i, h) in heroes.iter().enumerate() {
        for (j, vs) in heroes.iter().enumerate() {
            if i == j {
                continue;
            }
            let hsh = pair_hash(h.id.0, vs.id.0);
            // winrate in [0.40, 0.60]
            let winrate = hash_to_range(hsh, 0.40, 0.60);
            let advantage = winrate - 0.5;
            // games in [200, 4000]
            let games = (200 + (hsh / 7) % 3800) as i32;
            matchups.push(HeroMatchup {
                hero_id: h.id.clone(),
                vs_hero_id: vs.id.clone(),
                advantage,
                winrate,
                games,
            });
        }
    }
    db.bulk_upsert_matchups(&matchups, SYNTHETIC_PATCH)?;
    info!("Seeded {} synthetic matchups", matchups.len());

    // === Synergies (unordered pairs) ===
    // Each pair is stored once (lower-indexed hero as hero_id). Readers must
    // match either direction — see Database::get_synergies_for_hero, which
    // unions hero_id/with_hero_id matches and normalizes the result.
    let mut synergies: Vec<HeroSynergy> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &heroes[i];
            let b = &heroes[j];
            let hsh = pair_hash(a.id.0, b.id.0);
            let complement = role_complementarity(a, b);
            // noise in [-0.2, 0.2]
            let noise = hash_to_range(hsh, -0.2, 0.2);
            let synergy = (complement + noise).clamp(-0.8, 0.8);
            // winrate in [0.42, 0.58], biased by synergy
            let winrate = (0.5 + synergy * 0.15).clamp(0.42, 0.58);
            let games = (100 + (hsh / 11) % 2900) as i32;
            synergies.push(HeroSynergy {
                hero_id: a.id.clone(),
                with_hero_id: b.id.clone(),
                synergy,
                winrate,
                games,
            });
        }
    }
    db.bulk_upsert_synergies(&synergies, SYNTHETIC_PATCH)?;
    db.set_active_patch(SYNTHETIC_PATCH)?;
    info!("Seeded {} synthetic synergies", synergies.len());

    Ok(())
}

fn seed_real(db: &Database) -> Result<()> {
    let path = Path::new(SEED_PATH);
    if !path.exists() {
        anyhow::bail!(
            "{SEED_PATH} not found. Run `python ml/training/export_scoring_seed.py`, \
             or pass --synthetic for development-only placeholder data"
        );
    }

    let payload: ScoringSeed = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if payload.matchups.is_empty() || payload.synergies.is_empty() {
        anyhow::bail!(
            "{SEED_PATH} contains no matchup/synergy rows; regenerate it from the \
             processed datasets"
        );
    }

    let patch = payload
        .patch_version
        .clone()
        .filter(|patch| !patch.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{SEED_PATH} is missing `patch_version`; regenerate it with \
                 `python ml/training/export_scoring_seed.py`"
            )
        })?;

    info!(
        "Seeding real data for patch {patch} from {} ({} matchups, {} synergies)",
        SEED_PATH,
        payload.matchups.len(),
        payload.synergies.len()
    );
    db.bulk_upsert_matchups(&payload.matchups, &patch)?;
    db.bulk_upsert_synergies(&payload.synergies, &patch)?;
    db.set_active_patch(&patch)?;
    info!(
        "Seeded {} matchups and {} synergies for patch {patch}",
        payload.matchups.len(),
        payload.synergies.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = env::args().skip(1).collect();
    let synthetic = args.iter().any(|arg| arg == "--synthetic");
    if let Some(unknown) = args.iter().find(|arg| arg.as_str() != "--synthetic") {
        anyhow::bail!("Unknown argument {unknown:?}; supported: --synthetic");
    }

    // Same directory the Tauri app opens (see apps/desktop/src-tauri/src/paths.rs).
    let data_dir = app_data_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not resolve the OS application data directory"))?;
    std::fs::create_dir_all(&data_dir)?;

    let db = Database::new(data_dir)?;
    info!("Database initialized");

    let heroes = db.get_all_heroes()?;
    info!("Loaded {} heroes", heroes.len());
    if heroes.len() < 2 {
        anyhow::bail!("Need at least 2 heroes to seed matchups/synergies");
    }

    if synthetic {
        seed_synthetic(&db, &heroes)?;
    } else {
        seed_real(&db)?;
    }

    info!("Matchup/synergy seeding complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_seed_file_parses() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../infrastructure/seeders/scoring_seed.json");
        let payload: ScoringSeed =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        assert!(payload.matchups.len() > 1000);
        assert!(payload.synergies.len() > 1000);
        assert!(payload
            .patch_version
            .as_deref()
            .is_some_and(|patch| !patch.trim().is_empty()));
        assert!(payload
            .matchups
            .iter()
            .all(|row| row.hero_id != row.vs_hero_id));
        assert!(payload
            .synergies
            .iter()
            .all(|row| row.hero_id != row.with_hero_id));
    }
}
