//! Database Layer - SQLite persistence for hero data, matchups, meta, user profiles

use crate::models::*;
use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::info;

const DB_VERSION: u32 = 2;

/// Key used to persist the patch the app is currently recommending for.
pub const ACTIVE_PATCH_KEY: &str = "active_patch";

/// Key used to persist the serialized [`AppConfig`] as a JSON blob.
pub const CONFIG_KEY: &str = "app_config";

pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("smart_draft.db");

        let conn = Connection::open(&db_path)?;

        // Enable WAL mode for better concurrency
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        db.migrate()?;
        info!("Database initialized at {:?}", db_path);
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        // Create migrations table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        let current_version: u32 = conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )?;

        if current_version < DB_VERSION {
            self.run_migrations(&conn, current_version)?;
        }

        Ok(())
    }

    fn run_migrations(&self, conn: &Connection, from_version: u32) -> Result<()> {
        if from_version < 1 {
            self.migration_v1(conn)?;
            conn.execute("INSERT INTO schema_version (version) VALUES (1)", [])?;
        }
        if from_version < 2 {
            self.migration_v2(conn)?;
            conn.execute("INSERT INTO schema_version (version) VALUES (2)", [])?;
        }
        Ok(())
    }

    /// Key/value settings, including the active patch.
    fn migration_v2(&self, conn: &Connection) -> Result<()> {
        info!("Running migration v2: app_settings");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        Ok(())
    }

    fn migration_v1(&self, conn: &Connection) -> Result<()> {
        info!("Running migration v1: Initial schema");

        // Heroes table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS heroes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                localized_name TEXT NOT NULL,
                primary_attr TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                roles TEXT NOT NULL, -- JSON array
                legs INTEGER,
                img TEXT,
                icon TEXT,
                base_health INTEGER,
                base_mana INTEGER,
                base_armor REAL,
                base_attack_min INTEGER,
                base_attack_max INTEGER,
                base_str REAL,
                base_agi REAL,
                base_int REAL,
                str_gain REAL,
                agi_gain REAL,
                int_gain REAL,
                attack_range INTEGER,
                projectile_speed INTEGER,
                attack_rate REAL,
                move_speed INTEGER,
                turn_rate REAL,
                cm_enabled BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Hero semantics
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hero_semantics (
                hero_id INTEGER PRIMARY KEY,
                initiation REAL DEFAULT 0,
                stun REAL DEFAULT 0,
                save REAL DEFAULT 0,
                push REAL DEFAULT 0,
                scaling REAL DEFAULT 0,
                mobility REAL DEFAULT 0,
                durability REAL DEFAULT 0,
                teamfight REAL DEFAULT 0,
                splitpush REAL DEFAULT 0,
                vision REAL DEFAULT 0,
                magic_damage REAL DEFAULT 0,
                physical_damage REAL DEFAULT 0,
                pure_damage REAL DEFAULT 0,
                disable_count REAL DEFAULT 0,
                lane_strength_early REAL DEFAULT 0,
                lane_strength_mid REAL DEFAULT 0,
                lane_strength_late REAL DEFAULT 0,
                tempo_profile TEXT DEFAULT 'mid',
                roles_probability TEXT DEFAULT '{}', -- JSON
                FOREIGN KEY (hero_id) REFERENCES heroes(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Matchups
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hero_matchups (
                hero_id INTEGER NOT NULL,
                vs_hero_id INTEGER NOT NULL,
                advantage REAL NOT NULL,
                winrate REAL NOT NULL,
                games INTEGER NOT NULL,
                patch_version TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hero_id, vs_hero_id, patch_version),
                FOREIGN KEY (hero_id) REFERENCES heroes(id) ON DELETE CASCADE,
                FOREIGN KEY (vs_hero_id) REFERENCES heroes(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Synergies
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hero_synergies (
                hero_id INTEGER NOT NULL,
                with_hero_id INTEGER NOT NULL,
                synergy REAL NOT NULL,
                winrate REAL NOT NULL,
                games INTEGER NOT NULL,
                patch_version TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hero_id, with_hero_id, patch_version),
                FOREIGN KEY (hero_id) REFERENCES heroes(id) ON DELETE CASCADE,
                FOREIGN KEY (with_hero_id) REFERENCES heroes(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Meta data
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta_data (
                patch_version TEXT PRIMARY KEY,
                hero_meta_scores TEXT NOT NULL, -- JSON: hero_id -> score
                hero_pickrates TEXT NOT NULL, -- JSON: hero_id -> rate
                hero_winrates TEXT NOT NULL, -- JSON: hero_id -> rate
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Player profiles
        conn.execute(
            "CREATE TABLE IF NOT EXISTS player_profiles (
                steam_id TEXT PRIMARY KEY,
                hero_performance TEXT NOT NULL, -- JSON: hero_id -> performance
                preferred_roles TEXT NOT NULL, -- JSON array
                preferred_tempo TEXT DEFAULT 'mid',
                aggressive_tendency REAL DEFAULT 0.5,
                total_matches INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Draft history
        conn.execute(
            "CREATE TABLE IF NOT EXISTS draft_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                steam_id TEXT,
                patch_version TEXT,
                radiant_picks TEXT NOT NULL, -- JSON array
                dire_picks TEXT NOT NULL,
                bans TEXT NOT NULL,
                radiant_win BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_heroes_cm_enabled ON heroes(cm_enabled)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_matchups_hero ON hero_matchups(hero_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synergies_hero ON hero_synergies(hero_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_draft_history_steam ON draft_history(steam_id)",
            [],
        )?;

        Ok(())
    }

    // === Settings and health ===

    /// Upsert a key/value setting.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            params![key, value],
        )?;
        Ok(())
    }

    /// Read a setting, or `None` when it has never been written.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let value = conn
            .query_row(
                "SELECT value FROM app_settings WHERE key = ?",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value)
    }

    /// Persist the active patch used for matchup, synergy, and meta lookups.
    pub fn set_active_patch(&self, patch: &str) -> Result<()> {
        self.set_setting(ACTIVE_PATCH_KEY, patch)
    }

    /// Resolve the active patch from settings, falling back to the most
    /// recently updated patch present in the matchup table.
    pub fn get_active_patch(&self) -> Result<Option<String>> {
        if let Some(patch) = self.get_setting(ACTIVE_PATCH_KEY)? {
            if !patch.trim().is_empty() {
                return Ok(Some(patch));
            }
        }

        let conn = self.conn.lock().unwrap();
        let patch = conn
            .query_row(
                "SELECT patch_version FROM hero_matchups
                 WHERE patch_version IS NOT NULL
                 ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(patch)
    }

    /// Persist the full user configuration as a JSON blob.
    pub fn set_app_config(&self, config: &AppConfig) -> Result<()> {
        let json = serde_json::to_string(config)?;
        self.set_setting(CONFIG_KEY, &json)
    }

    /// Read the stored user configuration.
    ///
    /// Returns `Ok(None)` when nothing has been saved yet. A corrupt blob is a
    /// hard error so the caller can surface it rather than silently resetting
    /// the user's settings.
    pub fn get_app_config(&self) -> Result<Option<AppConfig>> {
        match self.get_setting(CONFIG_KEY)? {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    /// Startup check for schema version, seed completeness, and WAL mode.
    pub fn health(&self) -> Result<DataHealth> {
        let (schema_version, hero_count, matchup_count, synergy_count, wal_enabled) = {
            let conn = self.conn.lock().unwrap();
            let schema_version: u32 = conn.query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )?;
            let count = |table: &str| -> Result<i64> {
                Ok(
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                        row.get(0)
                    })?,
                )
            };
            let journal_mode: String =
                conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
            (
                schema_version,
                count("heroes")?,
                count("hero_matchups")?,
                count("hero_synergies")?,
                journal_mode.eq_ignore_ascii_case("wal"),
            )
        };

        let active_patch = self.get_active_patch()?;
        let mut issues = Vec::new();
        if schema_version < DB_VERSION {
            issues.push(format!(
                "schema version {schema_version} is older than {DB_VERSION}"
            ));
        }
        if hero_count == 0 {
            issues.push("no heroes seeded".to_string());
        }
        if matchup_count == 0 {
            issues.push("no matchups seeded".to_string());
        }
        if synergy_count == 0 {
            issues.push("no synergies seeded".to_string());
        }
        if active_patch.is_none() {
            issues.push("active patch is not set".to_string());
        }
        if !wal_enabled {
            issues.push("SQLite is not in WAL mode".to_string());
        }

        Ok(DataHealth {
            schema_version,
            hero_count,
            matchup_count,
            synergy_count,
            active_patch,
            wal_enabled,
            healthy: issues.is_empty(),
            issues,
        })
    }

    // === Hero Operations ===

    pub fn upsert_hero(&self, hero: &Hero) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let roles_json = serde_json::to_string(&hero.roles)?;

        conn.execute(
            "INSERT INTO heroes (id, name, localized_name, primary_attr, attack_type, roles, legs,
                img, icon, base_health, base_mana, base_armor, base_attack_min, base_attack_max,
                base_str, base_agi, base_int, str_gain, agi_gain, int_gain, attack_range,
                projectile_speed, attack_rate, move_speed, turn_rate, cm_enabled, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, localized_name=excluded.localized_name,
                primary_attr=excluded.primary_attr, attack_type=excluded.attack_type,
                roles=excluded.roles, legs=excluded.legs, img=excluded.img, icon=excluded.icon,
                base_health=excluded.base_health, base_mana=excluded.base_mana,
                base_armor=excluded.base_armor, base_attack_min=excluded.base_attack_min,
                base_attack_max=excluded.base_attack_max, base_str=excluded.base_str,
                base_agi=excluded.base_agi, base_int=excluded.base_int, str_gain=excluded.str_gain,
                agi_gain=excluded.agi_gain, int_gain=excluded.int_gain,
                attack_range=excluded.attack_range, projectile_speed=excluded.projectile_speed,
                attack_rate=excluded.attack_rate, move_speed=excluded.move_speed,
                turn_rate=excluded.turn_rate, cm_enabled=excluded.cm_enabled,
                updated_at=CURRENT_TIMESTAMP",
            params![
                hero.id.0, hero.name, hero.localized_name,
                hero.primary_attr.as_str(),
                format!("{:?}", hero.attack_type).to_lowercase(),
                roles_json, hero.legs, hero.img, hero.icon,
                hero.base_health, hero.base_mana, hero.base_armor,
                hero.base_attack_min, hero.base_attack_max,
                hero.base_str, hero.base_agi, hero.base_int,
                hero.str_gain, hero.agi_gain, hero.int_gain,
                hero.attack_range, hero.projectile_speed, hero.attack_rate,
                hero.move_speed, hero.turn_rate, hero.cm_enabled
            ],
        )?;
        Ok(())
    }

    /// Idempotent, transactional hero seed.
    pub fn bulk_upsert_heroes(&self, heroes: &[Hero]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO heroes (id, name, localized_name, primary_attr, attack_type, roles, legs,
                    img, icon, base_health, base_mana, base_armor, base_attack_min, base_attack_max,
                    base_str, base_agi, base_int, str_gain, agi_gain, int_gain, attack_range,
                    projectile_speed, attack_rate, move_speed, turn_rate, cm_enabled, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, localized_name=excluded.localized_name,
                    primary_attr=excluded.primary_attr, attack_type=excluded.attack_type,
                    roles=excluded.roles, legs=excluded.legs, img=excluded.img, icon=excluded.icon,
                    base_health=excluded.base_health, base_mana=excluded.base_mana,
                    base_armor=excluded.base_armor, base_attack_min=excluded.base_attack_min,
                    base_attack_max=excluded.base_attack_max, base_str=excluded.base_str,
                    base_agi=excluded.base_agi, base_int=excluded.base_int, str_gain=excluded.str_gain,
                    agi_gain=excluded.agi_gain, int_gain=excluded.int_gain,
                    attack_range=excluded.attack_range, projectile_speed=excluded.projectile_speed,
                    attack_rate=excluded.attack_rate, move_speed=excluded.move_speed,
                    turn_rate=excluded.turn_rate, cm_enabled=excluded.cm_enabled,
                    updated_at=CURRENT_TIMESTAMP",
            )?;
            for hero in heroes {
                let roles_json = serde_json::to_string(&hero.roles)?;
                stmt.execute(params![
                    hero.id.0,
                    hero.name,
                    hero.localized_name,
                    hero.primary_attr.as_str(),
                    format!("{:?}", hero.attack_type).to_lowercase(),
                    roles_json,
                    hero.legs,
                    hero.img,
                    hero.icon,
                    hero.base_health,
                    hero.base_mana,
                    hero.base_armor,
                    hero.base_attack_min,
                    hero.base_attack_max,
                    hero.base_str,
                    hero.base_agi,
                    hero.base_int,
                    hero.str_gain,
                    hero.agi_gain,
                    hero.int_gain,
                    hero.attack_range,
                    hero.projectile_speed,
                    hero.attack_rate,
                    hero.move_speed,
                    hero.turn_rate,
                    hero.cm_enabled
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Idempotent, transactional hero-semantics seed.
    pub fn bulk_upsert_hero_semantics(&self, semantics: &[HeroSemantics]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO hero_semantics (hero_id, initiation, stun, save, push, scaling,
                    mobility, durability, teamfight, splitpush, vision, magic_damage,
                    physical_damage, pure_damage, disable_count, lane_strength_early,
                    lane_strength_mid, lane_strength_late, tempo_profile, roles_probability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hero_id) DO UPDATE SET
                    initiation=excluded.initiation, stun=excluded.stun, save=excluded.save,
                    push=excluded.push, scaling=excluded.scaling, mobility=excluded.mobility,
                    durability=excluded.durability, teamfight=excluded.teamfight,
                    splitpush=excluded.splitpush, vision=excluded.vision,
                    magic_damage=excluded.magic_damage, physical_damage=excluded.physical_damage,
                    pure_damage=excluded.pure_damage, disable_count=excluded.disable_count,
                    lane_strength_early=excluded.lane_strength_early,
                    lane_strength_mid=excluded.lane_strength_mid,
                    lane_strength_late=excluded.lane_strength_late,
                    tempo_profile=excluded.tempo_profile, roles_probability=excluded.roles_probability",
            )?;
            for sem in semantics {
                let roles_json = serde_json::to_string(&sem.roles_probability)?;
                stmt.execute(params![
                    sem.hero_id.0,
                    sem.initiation,
                    sem.stun,
                    sem.save,
                    sem.push,
                    sem.scaling,
                    sem.mobility,
                    sem.durability,
                    sem.teamfight,
                    sem.splitpush,
                    sem.vision,
                    sem.magic_damage,
                    sem.physical_damage,
                    sem.pure_damage,
                    sem.disable_count,
                    sem.lane_strength.early,
                    sem.lane_strength.mid,
                    sem.lane_strength.late,
                    format!("{:?}", sem.tempo_profile).to_lowercase(),
                    roles_json
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_all_heroes(&self) -> Result<Vec<Hero>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, localized_name, primary_attr, attack_type, roles, legs,
                img, icon, base_health, base_mana, base_armor, base_attack_min, base_attack_max,
                base_str, base_agi, base_int, str_gain, agi_gain, int_gain, attack_range,
                projectile_speed, attack_rate, move_speed, turn_rate, cm_enabled
            FROM heroes WHERE cm_enabled = 1 ORDER BY id",
        )?;

        let heroes = stmt
            .query_map([], |row| {
                Ok(Hero {
                    id: HeroId(row.get(0)?),
                    name: row.get(1)?,
                    localized_name: row.get(2)?,
                    primary_attr: PrimaryAttribute::from_db(&row.get::<_, String>(3)?),
                    attack_type: match row.get::<_, String>(4)?.as_str() {
                        "melee" => AttackType::Melee,
                        "ranged" => AttackType::Ranged,
                        _ => AttackType::Melee,
                    },
                    roles: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                    legs: row.get(6)?,
                    img: row.get(7)?,
                    icon: row.get(8)?,
                    base_health: row.get(9)?,
                    base_mana: row.get(10)?,
                    base_armor: row.get(11)?,
                    base_attack_min: row.get(12)?,
                    base_attack_max: row.get(13)?,
                    base_str: row.get(14)?,
                    base_agi: row.get(15)?,
                    base_int: row.get(16)?,
                    str_gain: row.get(17)?,
                    agi_gain: row.get(18)?,
                    int_gain: row.get(19)?,
                    attack_range: row.get(20)?,
                    projectile_speed: row.get(21)?,
                    attack_rate: row.get(22)?,
                    move_speed: row.get(23)?,
                    turn_rate: row.get(24)?,
                    cm_enabled: row.get(25)?,
                })
            })?
            .collect::<SqliteResult<Vec<_>>>()?;

        Ok(heroes)
    }

    pub fn get_hero(&self, hero_id: HeroId) -> Result<Option<Hero>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, localized_name, primary_attr, attack_type, roles, legs,
                img, icon, base_health, base_mana, base_armor, base_attack_min, base_attack_max,
                base_str, base_agi, base_int, str_gain, agi_gain, int_gain, attack_range,
                projectile_speed, attack_rate, move_speed, turn_rate, cm_enabled
            FROM heroes WHERE id = ?",
        )?;

        let hero = stmt
            .query_row(params![hero_id.0], |row| {
                Ok(Hero {
                    id: HeroId(row.get(0)?),
                    name: row.get(1)?,
                    localized_name: row.get(2)?,
                    primary_attr: PrimaryAttribute::from_db(&row.get::<_, String>(3)?),
                    attack_type: match row.get::<_, String>(4)?.as_str() {
                        "melee" => AttackType::Melee,
                        "ranged" => AttackType::Ranged,
                        _ => AttackType::Melee,
                    },
                    roles: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                    legs: row.get(6)?,
                    img: row.get(7)?,
                    icon: row.get(8)?,
                    base_health: row.get(9)?,
                    base_mana: row.get(10)?,
                    base_armor: row.get(11)?,
                    base_attack_min: row.get(12)?,
                    base_attack_max: row.get(13)?,
                    base_str: row.get(14)?,
                    base_agi: row.get(15)?,
                    base_int: row.get(16)?,
                    str_gain: row.get(17)?,
                    agi_gain: row.get(18)?,
                    int_gain: row.get(19)?,
                    attack_range: row.get(20)?,
                    projectile_speed: row.get(21)?,
                    attack_rate: row.get(22)?,
                    move_speed: row.get(23)?,
                    turn_rate: row.get(24)?,
                    cm_enabled: row.get(25)?,
                })
            })
            .optional()?;

        Ok(hero)
    }

    // === Hero Semantics ===

    pub fn upsert_hero_semantics(&self, sem: &HeroSemantics) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let roles_json = serde_json::to_string(&sem.roles_probability)?;

        conn.execute(
            "INSERT INTO hero_semantics (hero_id, initiation, stun, save, push, scaling,
                mobility, durability, teamfight, splitpush, vision, magic_damage,
                physical_damage, pure_damage, disable_count, lane_strength_early,
                lane_strength_mid, lane_strength_late, tempo_profile, roles_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hero_id) DO UPDATE SET
                initiation=excluded.initiation, stun=excluded.stun, save=excluded.save,
                push=excluded.push, scaling=excluded.scaling, mobility=excluded.mobility,
                durability=excluded.durability, teamfight=excluded.teamfight,
                splitpush=excluded.splitpush, vision=excluded.vision,
                magic_damage=excluded.magic_damage, physical_damage=excluded.physical_damage,
                pure_damage=excluded.pure_damage, disable_count=excluded.disable_count,
                lane_strength_early=excluded.lane_strength_early,
                lane_strength_mid=excluded.lane_strength_mid,
                lane_strength_late=excluded.lane_strength_late,
                tempo_profile=excluded.tempo_profile, roles_probability=excluded.roles_probability",
            params![
                sem.hero_id.0,
                sem.initiation,
                sem.stun,
                sem.save,
                sem.push,
                sem.scaling,
                sem.mobility,
                sem.durability,
                sem.teamfight,
                sem.splitpush,
                sem.vision,
                sem.magic_damage,
                sem.physical_damage,
                sem.pure_damage,
                sem.disable_count,
                sem.lane_strength.early,
                sem.lane_strength.mid,
                sem.lane_strength.late,
                format!("{:?}", sem.tempo_profile).to_lowercase(),
                roles_json
            ],
        )?;
        Ok(())
    }

    pub fn get_all_hero_semantics(&self) -> Result<HashMap<HeroId, HeroSemantics>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT hero_id, initiation, stun, save, push, scaling, mobility, durability,
                teamfight, splitpush, vision, magic_damage, physical_damage, pure_damage,
                disable_count, lane_strength_early, lane_strength_mid, lane_strength_late,
                tempo_profile, roles_probability
            FROM hero_semantics",
        )?;

        let mut map = HashMap::new();
        let rows = stmt.query_map([], |row| {
            let hero_id: i32 = row.get(0)?;
            let roles_json: String = row.get(19)?;
            Ok((
                HeroId(hero_id),
                HeroSemantics {
                    hero_id: HeroId(hero_id),
                    initiation: row.get(1)?,
                    stun: row.get(2)?,
                    save: row.get(3)?,
                    push: row.get(4)?,
                    scaling: row.get(5)?,
                    mobility: row.get(6)?,
                    durability: row.get(7)?,
                    teamfight: row.get(8)?,
                    splitpush: row.get(9)?,
                    vision: row.get(10)?,
                    magic_damage: row.get(11)?,
                    physical_damage: row.get(12)?,
                    pure_damage: row.get(13)?,
                    disable_count: row.get(14)?,
                    lane_strength: LaneStrength {
                        early: row.get(15)?,
                        mid: row.get(16)?,
                        late: row.get(17)?,
                    },
                    tempo_profile: match row.get::<_, String>(18)?.as_str() {
                        "early" => TempoProfile::Early,
                        "late" => TempoProfile::Late,
                        _ => TempoProfile::Mid,
                    },
                    roles_probability: serde_json::from_str(&roles_json).unwrap_or_default(),
                },
            ))
        })?;

        for row in rows {
            let (id, sem) = row?;
            map.insert(id, sem);
        }

        Ok(map)
    }

    // === Matchups ===

    pub fn upsert_matchup(&self, matchup: &HeroMatchup, patch: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO hero_matchups (hero_id, vs_hero_id, advantage, winrate, games, patch_version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(hero_id, vs_hero_id, patch_version) DO UPDATE SET
                advantage=excluded.advantage, winrate=excluded.winrate, games=excluded.games",
            params![matchup.hero_id.0, matchup.vs_hero_id.0, matchup.advantage, matchup.winrate, matchup.games, patch],
        )?;
        Ok(())
    }

    pub fn bulk_upsert_matchups(&self, matchups: &[HeroMatchup], patch: &str) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO hero_matchups (hero_id, vs_hero_id, advantage, winrate, games, patch_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(hero_id, vs_hero_id, patch_version) DO UPDATE SET
                    advantage=excluded.advantage, winrate=excluded.winrate, games=excluded.games",
            )?;
            for m in matchups {
                stmt.execute(params![
                    m.hero_id.0,
                    m.vs_hero_id.0,
                    m.advantage,
                    m.winrate,
                    m.games,
                    patch
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_matchups_for_hero(&self, hero_id: HeroId, patch: &str) -> Result<Vec<HeroMatchup>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT hero_id, vs_hero_id, advantage, winrate, games
            FROM hero_matchups WHERE hero_id = ? AND patch_version = ?",
        )?;

        let matchups = stmt
            .query_map(params![hero_id.0, patch], |row| {
                Ok(HeroMatchup {
                    hero_id: HeroId(row.get(0)?),
                    vs_hero_id: HeroId(row.get(1)?),
                    advantage: row.get(2)?,
                    winrate: row.get(3)?,
                    games: row.get(4)?,
                })
            })?
            .collect::<SqliteResult<Vec<_>>>()?;

        Ok(matchups)
    }

    // === Synergies ===

    pub fn upsert_synergy(&self, synergy: &HeroSynergy, patch: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO hero_synergies (hero_id, with_hero_id, synergy, winrate, games, patch_version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(hero_id, with_hero_id, patch_version) DO UPDATE SET
                synergy=excluded.synergy, winrate=excluded.winrate, games=excluded.games",
            params![synergy.hero_id.0, synergy.with_hero_id.0, synergy.synergy, synergy.winrate, synergy.games, patch],
        )?;
        Ok(())
    }

    pub fn bulk_upsert_synergies(&self, synergies: &[HeroSynergy], patch: &str) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO hero_synergies (hero_id, with_hero_id, synergy, winrate, games, patch_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(hero_id, with_hero_id, patch_version) DO UPDATE SET
                    synergy=excluded.synergy, winrate=excluded.winrate, games=excluded.games",
            )?;
            for s in synergies {
                stmt.execute(params![
                    s.hero_id.0,
                    s.with_hero_id.0,
                    s.synergy,
                    s.winrate,
                    s.games,
                    patch
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_synergies_for_hero(&self, hero_id: HeroId, patch: &str) -> Result<Vec<HeroSynergy>> {
        let conn = self.conn.lock().unwrap();
        // Synergies are stored once per unordered pair (lower-indexed hero first),
        // so match either direction and normalize hero_id to the requested hero.
        let mut stmt = conn.prepare(
            "SELECT ?1, with_hero_id, synergy, winrate, games
             FROM hero_synergies WHERE hero_id = ?1 AND patch_version = ?2
             UNION ALL
             SELECT ?1, hero_id, synergy, winrate, games
             FROM hero_synergies
             WHERE with_hero_id = ?1 AND patch_version = ?2 AND hero_id != ?1",
        )?;

        let synergies = stmt
            .query_map(params![hero_id.0, patch], |row| {
                Ok(HeroSynergy {
                    hero_id: HeroId(row.get(0)?),
                    with_hero_id: HeroId(row.get(1)?),
                    synergy: row.get(2)?,
                    winrate: row.get(3)?,
                    games: row.get(4)?,
                })
            })?
            .collect::<SqliteResult<Vec<_>>>()?;

        Ok(synergies)
    }

    // === Meta Data ===

    pub fn upsert_meta(&self, meta: &MetaData) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let scores_json = serde_json::to_string(&meta.hero_meta_scores)?;
        let pickrates_json = serde_json::to_string(&meta.hero_pickrates)?;
        let winrates_json = serde_json::to_string(&meta.hero_winrates)?;

        conn.execute(
            "INSERT INTO meta_data (patch_version, hero_meta_scores, hero_pickrates, hero_winrates)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(patch_version) DO UPDATE SET
                hero_meta_scores=excluded.hero_meta_scores,
                hero_pickrates=excluded.hero_pickrates,
                hero_winrates=excluded.hero_winrates,
                updated_at=CURRENT_TIMESTAMP",
            params![
                meta.patch_version,
                scores_json,
                pickrates_json,
                winrates_json
            ],
        )?;
        Ok(())
    }

    pub fn get_meta(&self, patch_version: &str) -> Result<Option<MetaData>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT patch_version, hero_meta_scores, hero_pickrates, hero_winrates, updated_at
            FROM meta_data WHERE patch_version = ?",
        )?;

        let meta = stmt
            .query_row(params![patch_version], |row| {
                Ok(MetaData {
                    patch_version: row.get(0)?,
                    hero_meta_scores: serde_json::from_str(&row.get::<_, String>(1)?)
                        .unwrap_or_default(),
                    hero_pickrates: serde_json::from_str(&row.get::<_, String>(2)?)
                        .unwrap_or_default(),
                    hero_winrates: serde_json::from_str(&row.get::<_, String>(3)?)
                        .unwrap_or_default(),
                    updated_at: row.get(4)?,
                })
            })
            .optional()?;

        Ok(meta)
    }

    // === Player Profiles ===

    pub fn upsert_player_profile(&self, profile: &PlayerProfile) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let perf_json = serde_json::to_string(&profile.hero_performance)?;
        let roles_json = serde_json::to_string(&profile.preferred_roles)?;

        conn.execute(
            "INSERT INTO player_profiles (steam_id, hero_performance, preferred_roles, preferred_tempo, aggressive_tendency, total_matches)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(steam_id) DO UPDATE SET
                hero_performance=excluded.hero_performance,
                preferred_roles=excluded.preferred_roles,
                preferred_tempo=excluded.preferred_tempo,
                aggressive_tendency=excluded.aggressive_tendency,
                total_matches=excluded.total_matches,
                updated_at=CURRENT_TIMESTAMP",
            params![
                profile.steam_id, perf_json, roles_json,
                format!("{:?}", profile.preferred_tempo).to_lowercase(),
                profile.aggressive_tendency, profile.total_matches
            ],
        )?;
        Ok(())
    }

    pub fn get_player_profile(&self, steam_id: &str) -> Result<Option<PlayerProfile>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT steam_id, hero_performance, preferred_roles, preferred_tempo, aggressive_tendency, total_matches
            FROM player_profiles WHERE steam_id = ?"
        )?;

        let profile = stmt
            .query_row(params![steam_id], |row| {
                Ok(PlayerProfile {
                    steam_id: row.get(0)?,
                    hero_performance: serde_json::from_str(&row.get::<_, String>(1)?)
                        .unwrap_or_default(),
                    preferred_roles: serde_json::from_str(&row.get::<_, String>(2)?)
                        .unwrap_or_default(),
                    preferred_tempo: match row.get::<_, String>(3)?.as_str() {
                        "early" => TempoProfile::Early,
                        "late" => TempoProfile::Late,
                        _ => TempoProfile::Mid,
                    },
                    aggressive_tendency: row.get(4)?,
                    total_matches: row.get(5)?,
                })
            })
            .optional()?;

        Ok(profile)
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_database() -> (Database, PathBuf) {
        let dir = std::env::temp_dir().join(format!("sd-db-test-{}", uuid::Uuid::new_v4()));
        let db = Database::new(dir.clone()).expect("temp database");
        (db, dir)
    }

    #[test]
    fn test_app_config_round_trips() {
        let (db, dir) = temp_database();
        assert!(db.get_app_config().unwrap().is_none());

        let config = AppConfig {
            gsi_port: 4123,
            gsi_auto_start: false,
            gsi_auth_token: Some("token-123".to_string()),
            ml_model_path: "models/custom.onnx".to_string(),
            ..AppConfig::default()
        };
        db.set_app_config(&config).unwrap();

        let loaded = db.get_app_config().unwrap().expect("stored config");
        assert_eq!(loaded, config);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_stored_config_without_new_fields_gets_defaults() {
        let (db, dir) = temp_database();
        // Simulate a blob written by an older build that lacked gsi_auto_start
        // and gsi_auth_token.
        let json = r#"{"gsi_port":3001,"overlay_enabled":true,
            "overlay_position":{"x":1.0,"y":2.0},
            "overlay_size":{"width":10.0,"height":20.0},
            "ml_enabled":true,"ml_model_path":"m.onnx",
            "auto_update_meta":true,"preferred_language":"en"}"#;
        db.set_setting(CONFIG_KEY, json).unwrap();

        let loaded = db.get_app_config().unwrap().expect("stored config");
        assert_eq!(loaded.gsi_port, 3001);
        assert!(loaded.gsi_auto_start);
        assert!(loaded.gsi_auth_token.is_none());

        let _ = std::fs::remove_dir_all(dir);
    }
}
