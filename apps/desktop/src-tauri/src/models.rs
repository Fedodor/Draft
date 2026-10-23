//! Core data models for the Smart Draft Assistant

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::From;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HeroId(pub i32);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PlayerId(pub i64);

impl From<i64> for PlayerId {
    fn from(id: i64) -> Self {
        PlayerId(id)
    }
}

impl From<i32> for PlayerId {
    fn from(id: i32) -> Self {
        PlayerId(id as i64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MatchId(pub i64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hero {
    pub id: HeroId,
    pub name: String,
    pub localized_name: String,
    pub primary_attr: PrimaryAttribute,
    pub attack_type: AttackType,
    pub roles: Vec<String>,
    pub legs: i32,
    pub img: String,
    pub icon: String,
    pub base_health: i32,
    pub base_mana: i32,
    pub base_armor: f32,
    pub base_attack_min: i32,
    pub base_attack_max: i32,
    pub base_str: f32,
    pub base_agi: f32,
    pub base_int: f32,
    pub str_gain: f32,
    pub agi_gain: f32,
    pub int_gain: f32,
    pub attack_range: i32,
    pub projectile_speed: i32,
    pub attack_rate: f32,
    pub move_speed: i32,
    pub turn_rate: f32,
    pub cm_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryAttribute {
    Str,
    Agi,
    Int,
    All,
}

impl PrimaryAttribute {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Str => "str",
            Self::Agi => "agi",
            Self::Int => "int",
            Self::All => "all",
        }
    }

    /// Parse the database representation. Unknown legacy values fall back to
    /// Strength, matching the previous behavior, while `all` is preserved.
    pub fn from_db(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "agi" => Self::Agi,
            "int" => Self::Int,
            "all" => Self::All,
            _ => Self::Str,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AttackType {
    Melee,
    Ranged,
}

/// Semantic hero attributes for ML and scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroSemantics {
    pub hero_id: HeroId,
    pub initiation: f32,
    pub stun: f32,
    pub save: f32,
    pub push: f32,
    pub scaling: f32,
    pub mobility: f32,
    pub durability: f32,
    pub teamfight: f32,
    pub splitpush: f32,
    pub vision: f32,
    pub magic_damage: f32,
    pub physical_damage: f32,
    pub pure_damage: f32,
    pub disable_count: f32,
    pub lane_strength: LaneStrength,
    pub tempo_profile: TempoProfile,
    pub roles_probability: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneStrength {
    pub early: f32,
    pub mid: f32,
    pub late: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TempoProfile {
    Early,
    Mid,
    Late,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroMatchup {
    pub hero_id: HeroId,
    pub vs_hero_id: HeroId,
    pub advantage: f32,
    pub winrate: f32,
    pub games: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroSynergy {
    pub hero_id: HeroId,
    pub with_hero_id: HeroId,
    pub synergy: f32,
    pub winrate: f32,
    pub games: i32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum TeamSide {
    Radiant,
    Dire,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DraftPick {
    pub hero_id: HeroId,
    pub player_id: PlayerId,
    pub order: i32,
    pub team: TeamSide,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DraftBan {
    pub hero_id: HeroId,
    pub order: i32,
    pub team: TeamSide,
}

/// A single manual draft mutation, kept on an undo stack so corrections are
/// explicit and reversible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DraftAction {
    Pick { pick: DraftPick },
    Ban { ban: DraftBan },
}

impl DraftAction {
    pub fn hero_id(&self) -> HeroId {
        match self {
            DraftAction::Pick { pick } => pick.hero_id.clone(),
            DraftAction::Ban { ban } => ban.hero_id.clone(),
        }
    }
}

/// Validation failures raised by the draft engine before a mutation is applied.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum EngineError {
    #[error("unknown hero id {hero_id}")]
    UnknownHero { hero_id: i32 },
    #[error("hero {hero_id} is not available")]
    HeroUnavailable { hero_id: i32 },
    #[error("{team} already has {count} picks")]
    TeamFull { team: String, count: usize },
    #[error("invalid order {order}")]
    InvalidOrder { order: i32 },
}

/// Serializable error returned to the frontend by every Tauri command.
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommandError {
    #[error("invalid team: {team}")]
    InvalidTeam { team: String },
    #[error("unknown hero id {hero_id}")]
    InvalidHeroId { hero_id: i32 },
    #[error("hero {hero_id} is not available")]
    HeroUnavailable { hero_id: i32 },
    #[error("{team} already has {count} picks")]
    TeamFull { team: String, count: usize },
    #[error("invalid turn/order: {message}")]
    InvalidTurn { message: String },
    #[error("model file not found: {path}")]
    MissingModelFile { path: String },
    #[error("model error: {message}")]
    ModelError { message: String },
    #[error("{message}")]
    Internal { message: String },
}

impl From<EngineError> for CommandError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::UnknownHero { hero_id } => CommandError::InvalidHeroId { hero_id },
            EngineError::HeroUnavailable { hero_id } => CommandError::HeroUnavailable { hero_id },
            EngineError::TeamFull { team, count } => CommandError::TeamFull { team, count },
            EngineError::InvalidOrder { order } => CommandError::InvalidTurn {
                message: format!("order must be at least 1 (got {order})"),
            },
        }
    }
}

impl From<anyhow::Error> for CommandError {
    fn from(error: anyhow::Error) -> Self {
        CommandError::Internal {
            message: error.to_string(),
        }
    }
}

impl From<String> for CommandError {
    fn from(message: String) -> Self {
        CommandError::Internal { message }
    }
}

impl From<&str> for CommandError {
    fn from(message: &str) -> Self {
        CommandError::Internal {
            message: message.to_string(),
        }
    }
}

/// Startup database/seed health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataHealth {
    pub schema_version: u32,
    pub hero_count: i64,
    pub matchup_count: i64,
    pub synergy_count: i64,
    pub active_patch: Option<String>,
    pub wal_enabled: bool,
    pub healthy: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftState {
    pub radiant_picks: Vec<DraftPick>,
    pub dire_picks: Vec<DraftPick>,
    pub bans: Vec<DraftBan>,
    pub current_pick: Option<TeamSide>,
    pub pick_number: i32,
    pub is_picking_phase: bool,
    pub available_heroes: Vec<HeroId>,
    pub patch_version: String,
    pub timestamp: i64,
}

impl Default for DraftState {
    fn default() -> Self {
        Self {
            radiant_picks: Vec::new(),
            dire_picks: Vec::new(),
            bans: Vec::new(),
            current_pick: None,
            pick_number: 0,
            is_picking_phase: false,
            available_heroes: Vec::new(),
            patch_version: "7.41".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub hero_id: HeroId,
    pub score: f32,
    pub confidence: f32,
    pub explanations: Vec<String>,
    pub risk_level: RiskLevel,
    pub role_fit: Vec<String>,
    pub category: RecommendationCategory,
}

/// Per-component scores behind a recommendation. All values are in [0, 1]
/// with 0.5 neutral; `total` is the weighted combination.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreBreakdown {
    pub matchup: f32,
    pub synergy: f32,
    pub composition: f32,
    pub meta: f32,
    pub comfort: f32,
    pub ml: f32,
    pub total: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RecommendationCategory {
    Top,
    Safe,
    Aggressive,
    Comfort,
    AntiMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationSet {
    pub top_picks: Vec<Recommendation>,
    pub safe_picks: Vec<Recommendation>,
    pub aggressive_picks: Vec<Recommendation>,
    pub comfort_picks: Vec<Recommendation>,
    pub anti_meta_picks: Vec<Recommendation>,
    pub team_analysis: TeamAnalysis,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamAnalysis {
    pub missing: Vec<String>,
    pub strong: Vec<String>,
    pub composition_score: f32,
    pub lane_balance: LaneBalance,
    pub tempo: TempoProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneBalance {
    pub safe_lane: f32,
    pub mid_lane: f32,
    pub offlane: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerProfile {
    pub steam_id: String,
    pub hero_performance: HashMap<HeroId, HeroPerformance>,
    pub preferred_roles: Vec<String>,
    pub preferred_tempo: TempoProfile,
    pub aggressive_tendency: f32,
    pub total_matches: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroPerformance {
    pub hero_id: HeroId,
    pub games: i32,
    pub wins: i32,
    pub winrate: f32,
    pub avg_kda: f32,
    pub last_played: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaData {
    pub patch_version: String,
    pub hero_meta_scores: HashMap<HeroId, f32>,
    pub hero_pickrates: HashMap<HeroId, f32>,
    pub hero_winrates: HashMap<HeroId, f32>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSIPlayer {
    pub account_id: PlayerId,
    pub hero_id: HeroId,
    pub team: TeamSide,
}

/// The local player's own section of a GSI payload. Dota reports the side the
/// client is on through `player.team_name` ("radiant"/"dire").
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GSIPlayerSelf {
    #[serde(default)]
    pub team_name: Option<String>,
}

impl GSIPlayerSelf {
    /// Map the GSI `team_name` string onto a [`TeamSide`].
    pub fn team_side(&self) -> Option<TeamSide> {
        match self.team_name.as_deref() {
            Some("radiant") => Some(TeamSide::Radiant),
            Some("dire") => Some(TeamSide::Dire),
            _ => None,
        }
    }
}

/// One team's slot list inside a real Dota 2 GSI `draft` payload.
///
/// GSI does not send `picks`/`bans` arrays; it sends fixed `pickN_id`/`banN_id`
/// fields (0 means the slot is empty). `team_id` is 2 for Radiant and 3 for Dire.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GSITeamDraft {
    #[serde(default)]
    pub team_id: Option<i32>,
    #[serde(default)]
    pub pick0_id: i32,
    #[serde(default)]
    pub pick1_id: i32,
    #[serde(default)]
    pub pick2_id: i32,
    #[serde(default)]
    pub pick3_id: i32,
    #[serde(default)]
    pub pick4_id: i32,
    #[serde(default)]
    pub ban0_id: i32,
    #[serde(default)]
    pub ban1_id: i32,
    #[serde(default)]
    pub ban2_id: i32,
    #[serde(default)]
    pub ban3_id: i32,
    #[serde(default)]
    pub ban4_id: i32,
}

impl GSITeamDraft {
    /// Non-zero pick hero ids in slot order.
    pub fn pick_ids(&self) -> Vec<i32> {
        [
            self.pick0_id,
            self.pick1_id,
            self.pick2_id,
            self.pick3_id,
            self.pick4_id,
        ]
        .into_iter()
        .filter(|id| *id > 0)
        .collect()
    }

    /// Non-zero ban hero ids in slot order.
    pub fn ban_ids(&self) -> Vec<i32> {
        [
            self.ban0_id,
            self.ban1_id,
            self.ban2_id,
            self.ban3_id,
            self.ban4_id,
        ]
        .into_iter()
        .filter(|id| *id > 0)
        .collect()
    }
}

/// The real Dota 2 GSI `draft` component.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GSIDraft {
    /// 2 = Radiant, 3 = Dire, 0 = none.
    #[serde(default)]
    pub activeteam: Option<i32>,
    /// Whether the active team's current action is a pick (`true`) or ban (`false`).
    #[serde(default)]
    pub pick: Option<bool>,
    #[serde(default)]
    pub activeteam_time_remaining: Option<f64>,
    #[serde(default)]
    pub radiant_bonus_time: Option<f64>,
    #[serde(default)]
    pub dire_bonus_time: Option<f64>,
    #[serde(default)]
    pub team2: Option<GSITeamDraft>,
    #[serde(default)]
    pub team3: Option<GSITeamDraft>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSIGameState {
    pub map: GSIMap,
    #[serde(default)]
    pub player: Option<GSIPlayerSelf>,
    #[serde(default)]
    pub hero: Option<GSIHero>,
    #[serde(default)]
    pub draft: Option<GSIDraft>,
    #[serde(default)]
    pub players: HashMap<String, GSIPlayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSIMap {
    pub game_state: String,
    pub game_time: f64,
    pub clock_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSIHero {
    pub id: HeroId,
    pub level: i32,
}

fn default_true() -> bool {
    true
}

/// User-visible settings. `#[serde(default)]` on every field keeps stored
/// configs forward-compatible: a blob written by an older build still loads,
/// with new fields taking their defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_gsi_port")]
    pub gsi_port: u16,
    #[serde(default = "default_true")]
    pub gsi_auto_start: bool,
    #[serde(default)]
    pub gsi_auth_token: Option<String>,
    #[serde(default = "default_true")]
    pub overlay_enabled: bool,
    #[serde(default = "default_position")]
    pub overlay_position: Position,
    #[serde(default = "default_size")]
    pub overlay_size: Size,
    #[serde(default = "default_true")]
    pub ml_enabled: bool,
    #[serde(default = "default_model_path")]
    pub ml_model_path: String,
    #[serde(default = "default_true")]
    pub auto_update_meta: bool,
    #[serde(default = "default_language")]
    pub preferred_language: String,
}

fn default_gsi_port() -> u16 {
    3000
}
fn default_position() -> Position {
    Position { x: 100.0, y: 100.0 }
}
fn default_size() -> Size {
    Size {
        width: 400.0,
        height: 600.0,
    }
}
fn default_model_path() -> String {
    "ml/models/draft_model.onnx".to_string()
}
fn default_language() -> String {
    "en".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            gsi_port: default_gsi_port(),
            gsi_auto_start: true,
            gsi_auth_token: None,
            overlay_enabled: true,
            overlay_position: default_position(),
            overlay_size: default_size(),
            ml_enabled: true,
            ml_model_path: default_model_path(),
            auto_update_meta: true,
            preferred_language: default_language(),
        }
    }
}

impl AppConfig {
    /// Reject settings the app cannot honor. Called by `set_config` before the
    /// new config is persisted.
    pub fn validate(&self) -> Result<(), String> {
        if self.gsi_port < 1024 {
            return Err(format!(
                "gsi_port must be at least 1024 (got {})",
                self.gsi_port
            ));
        }
        if self.ml_model_path.trim().is_empty() {
            return Err("ml_model_path must not be empty".to_string());
        }
        if let Some(token) = &self.gsi_auth_token {
            if token.trim().is_empty() {
                return Err("gsi_auth_token must be null or a non-empty string".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    pub draft: DraftState,
    pub recommendations: Option<RecommendationSet>,
    pub is_processing: bool,
    pub last_update: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_attribute_all_roundtrips() {
        assert_eq!(PrimaryAttribute::All.as_str(), "all");
        assert_eq!(PrimaryAttribute::from_db("all"), PrimaryAttribute::All);
        assert_eq!(PrimaryAttribute::from_db(" ALL "), PrimaryAttribute::All);
        assert_eq!(PrimaryAttribute::from_db("agi"), PrimaryAttribute::Agi);
        assert_eq!(PrimaryAttribute::from_db("int"), PrimaryAttribute::Int);
        assert_eq!(PrimaryAttribute::from_db("str"), PrimaryAttribute::Str);
        assert_eq!(PrimaryAttribute::from_db("legacy"), PrimaryAttribute::Str);
    }

    #[test]
    fn test_app_config_defaults_and_validation() {
        let config = AppConfig::default();
        assert!(config.gsi_auto_start);
        assert!(config.gsi_auth_token.is_none());
        assert!(config.validate().is_ok());

        let invalid = AppConfig {
            gsi_port: 80,
            ..AppConfig::default()
        };
        assert!(invalid.validate().is_err());

        let no_model = AppConfig {
            ml_model_path: "  ".to_string(),
            ..AppConfig::default()
        };
        assert!(no_model.validate().is_err());

        let empty_token = AppConfig {
            gsi_auth_token: Some(String::new()),
            ..AppConfig::default()
        };
        assert!(empty_token.validate().is_err());
    }

    #[test]
    fn test_gsi_player_self_team_side() {
        let radiant = GSIPlayerSelf {
            team_name: Some("radiant".to_string()),
        };
        let dire = GSIPlayerSelf {
            team_name: Some("dire".to_string()),
        };
        let unknown = GSIPlayerSelf::default();
        assert_eq!(radiant.team_side(), Some(TeamSide::Radiant));
        assert_eq!(dire.team_side(), Some(TeamSide::Dire));
        assert_eq!(unknown.team_side(), None);
    }

    #[test]
    fn test_gsiteamdraft_skips_empty_slots() {
        let team = GSITeamDraft {
            pick0_id: 3,
            ban0_id: 7,
            ..GSITeamDraft::default()
        };
        assert_eq!(team.pick_ids(), vec![3]);
        assert_eq!(team.ban_ids(), vec![7]);
    }
}
