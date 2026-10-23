//! Draft Engine - Core recommendation logic
//!
//! Implements a modular scoring system for hero recommendations

use std::collections::HashMap;
use std::sync::Arc;

use crate::database::Database;
use crate::models::*;
use anyhow::Result;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

pub mod scorer;
pub mod simulation;

use scorer::{
    CompositionScorer, MLScorer, MatchupScorer, Scorer, SynergyScorer, W_COMFORT, W_COMPOSITION,
    W_MATCHUP, W_META, W_ML, W_SYNERGY,
};

fn team_label(team: TeamSide) -> &'static str {
    match team {
        TeamSide::Radiant => "radiant",
        TeamSide::Dire => "dire",
    }
}

/// Main draft engine that maintains state and generates recommendations
#[derive(Clone)]
pub struct DraftEngine {
    pub state: Arc<RwLock<DraftState>>,
    pub hero_db: Vec<Hero>,
    pub hero_semantics: HashMap<HeroId, HeroSemantics>,
    pub matchup_data: HashMap<HeroId, Vec<HeroMatchup>>,
    pub synergy_data: HashMap<HeroId, Vec<HeroSynergy>>,
    pub meta_data: MetaData,
    pub scorer: DraftScorer,
    pub player_profile: Option<PlayerProfile>,
    database: Option<Arc<Database>>,
    undo_stack: Arc<RwLock<Vec<DraftAction>>>,
    active_patch: Option<String>,
}

#[derive(Default)]
pub struct DraftScorer {
    matchup_scorer: MatchupScorer,
    synergy_scorer: SynergyScorer,
    composition_scorer: CompositionScorer,
    ml_scorer: Option<MLScorer>,
}

impl Clone for DraftScorer {
    fn clone(&self) -> Self {
        Self {
            matchup_scorer: self.matchup_scorer.clone(),
            synergy_scorer: self.synergy_scorer.clone(),
            composition_scorer: self.composition_scorer.clone(),
            ml_scorer: self.ml_scorer.clone(),
        }
    }
}

impl DraftScorer {
    pub fn set_ml_scorer(&mut self, ml_scorer: MLScorer) {
        self.ml_scorer = Some(ml_scorer);
    }

    pub fn set_data(
        &mut self,
        matchup_data: HashMap<HeroId, Vec<HeroMatchup>>,
        synergy_data: HashMap<HeroId, Vec<HeroSynergy>>,
    ) {
        self.matchup_scorer = MatchupScorer::with_data(matchup_data);
        self.synergy_scorer = SynergyScorer::with_data(synergy_data);
    }

    pub fn matchup_hero_count(&self) -> usize {
        self.matchup_scorer.hero_count()
    }

    pub fn synergy_hero_count(&self) -> usize {
        self.synergy_scorer.hero_count()
    }
}

impl Default for DraftEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftEngine {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DraftState::default())),
            hero_db: Vec::new(),
            hero_semantics: HashMap::new(),
            matchup_data: HashMap::new(),
            synergy_data: HashMap::new(),
            meta_data: MetaData {
                patch_version: "7.41".to_string(),
                hero_meta_scores: HashMap::new(),
                hero_pickrates: HashMap::new(),
                hero_winrates: HashMap::new(),
                updated_at: chrono::Utc::now().timestamp(),
            },
            scorer: DraftScorer::default(),
            player_profile: None,
            database: None,
            undo_stack: Arc::new(RwLock::new(Vec::new())),
            active_patch: None,
        }
    }

    pub fn set_database(&mut self, database: Arc<Database>) {
        self.database = Some(database);
    }

    /// Schema/seed/WAL health for the startup diagnostics panel.
    pub fn data_health(&self) -> Result<DataHealth, CommandError> {
        match &self.database {
            Some(db) => db.health().map_err(CommandError::from),
            None => Err(CommandError::Internal {
                message: "database is not initialized".to_string(),
            }),
        }
    }

    /// Initialize the engine - loads all data from database
    pub fn initialize(&mut self) -> Result<()> {
        info!("Initializing Draft Engine");

        // Resolve the persisted active patch before any patch-scoped query.
        self.load_active_patch()?;

        // Load hero data from database
        self.load_hero_data()?;

        // Load semantic data
        self.load_semantic_data()?;

        // Load matchup/synergy matrices
        self.load_matchup_data()?;
        self.load_synergy_data()?;

        // Load meta data for the active patch
        self.load_meta_data()?;

        // Wire the loaded matchup/synergy matrices into the scorers. These
        // fields were previously loaded but never used, leaving both scorers
        // permanently neutral.
        self.scorer
            .set_data(self.matchup_data.clone(), self.synergy_data.clone());
        info!(
            "Wired scorers with matchup data for {} heroes and synergy data for {} heroes",
            self.scorer.matchup_hero_count(),
            self.scorer.synergy_hero_count()
        );

        // Initialize ML scorer with loaded semantics and meta data
        let ml_scorer = MLScorer::new(
            self.hero_semantics.clone(),
            self.meta_data.clone(),
            self.hero_db.clone(),
            self.matchup_data.clone(),
        );
        self.scorer.set_ml_scorer(ml_scorer);

        // Provide hero role data to the composition scorer
        let hero_roles: HashMap<HeroId, Vec<String>> = self
            .hero_db
            .iter()
            .map(|h| (h.id.clone(), h.roles.clone()))
            .collect();
        self.scorer.composition_scorer = CompositionScorer::new().with_hero_roles(hero_roles);

        // Initialize available heroes and surface the persisted patch.
        let mut state = self.state.write();
        state.available_heroes = self
            .hero_db
            .iter()
            .filter(|h| h.cm_enabled)
            .map(|h| h.id.clone())
            .collect();
        if let Some(patch) = &self.active_patch {
            state.patch_version = patch.clone();
        }

        info!(
            "Draft Engine initialized with {} heroes (patch {})",
            self.hero_db.len(),
            self.active_patch.as_deref().unwrap_or("unset")
        );
        Ok(())
    }

    fn load_active_patch(&mut self) -> Result<()> {
        if let Some(db) = &self.database {
            self.active_patch = db.get_active_patch()?;
        }
        match &self.active_patch {
            Some(patch) => info!("Active patch: {patch}"),
            None => warn!("No active patch persisted; matchup/synergy/meta lookups are disabled"),
        }
        Ok(())
    }

    fn load_hero_data(&mut self) -> Result<()> {
        if let Some(db) = &self.database {
            let heroes = db.get_all_heroes()?;
            if !heroes.is_empty() {
                self.hero_db = heroes;
                info!("Loaded {} heroes from database", self.hero_db.len());
                return Ok(());
            }
        }

        // Fallback: create minimal hero data for development
        warn!("No heroes in database, using fallback data");
        self.hero_db = Self::fallback_heroes();
        Ok(())
    }

    fn fallback_heroes() -> Vec<Hero> {
        vec![Hero {
            id: HeroId(1),
            name: "npc_dota_hero_axe".to_string(),
            localized_name: "Axe".to_string(),
            primary_attr: PrimaryAttribute::Str,
            attack_type: AttackType::Melee,
            roles: vec![
                "initiator".to_string(),
                "durable".to_string(),
                "disabler".to_string(),
                "jungler".to_string(),
            ],
            legs: 2,
            img: "/apps/dota2/images/heroes/axe_full.png".to_string(),
            icon: "/apps/dota2/images/heroes/axe_icon.png".to_string(),
            base_health: 200,
            base_mana: 75,
            base_armor: 2.0,
            base_attack_min: 27,
            base_attack_max: 33,
            base_str: 25.0,
            base_agi: 20.0,
            base_int: 18.0,
            str_gain: 3.2,
            agi_gain: 2.0,
            int_gain: 1.8,
            attack_range: 150,
            projectile_speed: 0,
            attack_rate: 1.7,
            move_speed: 310,
            turn_rate: 0.6,
            cm_enabled: true,
        }]
    }

    fn load_semantic_data(&mut self) -> Result<()> {
        if let Some(db) = &self.database {
            let semantics = db.get_all_hero_semantics()?;
            if !semantics.is_empty() {
                self.hero_semantics = semantics;
                info!(
                    "Loaded {} hero semantics from database",
                    self.hero_semantics.len()
                );
                return Ok(());
            }
        }

        // Fallback: create sample data
        warn!("No hero semantics in database, using fallback");
        self.hero_semantics.insert(
            HeroId(1),
            HeroSemantics {
                hero_id: HeroId(1),
                initiation: 0.9,
                stun: 0.8,
                save: 0.3,
                push: 0.2,
                scaling: 0.6,
                mobility: 0.4,
                durability: 0.9,
                teamfight: 0.95,
                splitpush: 0.2,
                vision: 0.1,
                magic_damage: 0.4,
                physical_damage: 0.8,
                pure_damage: 0.1,
                disable_count: 2.0,
                lane_strength: LaneStrength {
                    early: 0.8,
                    mid: 0.7,
                    late: 0.6,
                },
                tempo_profile: TempoProfile::Early,
                roles_probability: {
                    let mut m = HashMap::new();
                    m.insert("initiator".to_string(), 0.9);
                    m.insert("durable".to_string(), 0.8);
                    m.insert("disabler".to_string(), 0.7);
                    m.insert("offlane".to_string(), 0.8);
                    m
                },
            },
        );
        Ok(())
    }

    fn load_matchup_data(&mut self) -> Result<()> {
        let (Some(db), Some(patch)) = (&self.database, self.active_patch.clone()) else {
            return Ok(());
        };
        for hero_id in &self.hero_db {
            let matchups = db.get_matchups_for_hero(hero_id.id.clone(), &patch)?;
            if !matchups.is_empty() {
                self.matchup_data.insert(hero_id.id.clone(), matchups);
            }
        }
        info!(
            "Loaded matchup data for {} heroes (patch {patch})",
            self.matchup_data.len()
        );
        Ok(())
    }

    fn load_synergy_data(&mut self) -> Result<()> {
        let (Some(db), Some(patch)) = (&self.database, self.active_patch.clone()) else {
            return Ok(());
        };
        for hero_id in &self.hero_db {
            let synergies = db.get_synergies_for_hero(hero_id.id.clone(), &patch)?;
            if !synergies.is_empty() {
                self.synergy_data.insert(hero_id.id.clone(), synergies);
            }
        }
        info!(
            "Loaded synergy data for {} heroes (patch {patch})",
            self.synergy_data.len()
        );
        Ok(())
    }

    fn load_meta_data(&mut self) -> Result<()> {
        let (Some(db), Some(patch)) = (&self.database, self.active_patch.clone()) else {
            return Ok(());
        };
        if let Some(meta) = db.get_meta(&patch)? {
            self.meta_data = meta;
            info!("Loaded meta data for patch {patch}");
        } else {
            self.meta_data.patch_version = patch;
        }
        Ok(())
    }

    /// Validate a pick before it is applied. Manual corrections deliberately do
    /// not fail on turn order; only malformed input and unavailable heroes do.
    pub fn validate_pick(&self, pick: &DraftPick) -> std::result::Result<(), EngineError> {
        if pick.order < 1 {
            return Err(EngineError::InvalidOrder { order: pick.order });
        }
        self.validate_hero_available(pick.hero_id.0)?;

        let state = self.state.read();
        let count = match pick.team {
            TeamSide::Radiant => state.radiant_picks.len(),
            TeamSide::Dire => state.dire_picks.len(),
        };
        if count >= 5 {
            return Err(EngineError::TeamFull {
                team: team_label(pick.team).to_string(),
                count,
            });
        }
        Ok(())
    }

    /// Validate a ban before it is applied.
    pub fn validate_ban(&self, ban: &DraftBan) -> std::result::Result<(), EngineError> {
        if ban.order < 1 {
            return Err(EngineError::InvalidOrder { order: ban.order });
        }
        self.validate_hero_available(ban.hero_id.0)
    }

    fn validate_hero_available(&self, hero_id: i32) -> std::result::Result<(), EngineError> {
        if !self.hero_db.iter().any(|hero| hero.id.0 == hero_id) {
            return Err(EngineError::UnknownHero { hero_id });
        }
        let state = self.state.read();
        if !state.available_heroes.iter().any(|id| id.0 == hero_id) {
            return Err(EngineError::HeroUnavailable { hero_id });
        }
        Ok(())
    }

    /// Update draft state with a pick
    pub fn add_pick(&self, pick: DraftPick) {
        let pick_for_log = pick.clone();
        let mut state = self.state.write();

        // Remove from available heroes
        state.available_heroes.retain(|h| h != &pick.hero_id);

        // Add to appropriate team
        match pick.team {
            TeamSide::Radiant => state.radiant_picks.push(pick.clone()),
            TeamSide::Dire => state.dire_picks.push(pick.clone()),
        }

        state.pick_number += 1;
        state.timestamp = chrono::Utc::now().timestamp();
        drop(state);

        self.undo_stack.write().push(DraftAction::Pick {
            pick: pick_for_log.clone(),
        });

        debug!("Pick added: {:?}", pick_for_log);
    }

    /// Update draft state with a ban
    pub fn add_ban(&self, ban: DraftBan) {
        let ban_for_log = ban.clone();
        let mut state = self.state.write();

        // Remove from available heroes
        state.available_heroes.retain(|h| h != &ban.hero_id);

        state.bans.push(ban.clone());
        state.timestamp = chrono::Utc::now().timestamp();
        drop(state);

        self.undo_stack.write().push(DraftAction::Ban {
            ban: ban_for_log.clone(),
        });

        debug!("Ban added: {:?}", ban_for_log);
    }

    /// Reverse the most recent pick or ban. Returns the removed action, or
    /// `None` when there is nothing to undo.
    pub fn undo_last_action(&self) -> Option<DraftAction> {
        let action = self.undo_stack.write().pop()?;
        let mut state = self.state.write();

        match &action {
            DraftAction::Pick { pick } => {
                match pick.team {
                    TeamSide::Radiant => state.radiant_picks.retain(|p| p.hero_id != pick.hero_id),
                    TeamSide::Dire => state.dire_picks.retain(|p| p.hero_id != pick.hero_id),
                }
                state.pick_number = (state.pick_number - 1).max(0);
            }
            DraftAction::Ban { ban } => {
                state.bans.retain(|b| b.hero_id != ban.hero_id);
            }
        }

        if !state.available_heroes.contains(&action.hero_id()) {
            state.available_heroes.push(action.hero_id());
            state.available_heroes.sort_by_key(|a| a.0);
        }
        state.timestamp = chrono::Utc::now().timestamp();

        info!("Undid {:?}", action);
        Some(action)
    }

    /// Number of actions currently available to undo.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.read().len()
    }

    /// Reset draft to initial state
    pub fn reset(&self) {
        let mut state = self.state.write();
        *state = DraftState {
            available_heroes: self
                .hero_db
                .iter()
                .filter(|h| h.cm_enabled)
                .map(|h| h.id.clone())
                .collect(),
            patch_version: self.active_patch.clone().unwrap_or_default(),
            ..Default::default()
        };
        drop(state);
        self.undo_stack.write().clear();
        info!("Draft reset");
    }

    /// Get current draft state
    pub fn get_state(&self) -> DraftState {
        self.state.read().clone()
    }

    /// Debug view: per-component scores for a candidate hero in the current
    /// draft. Returns `None` when the hero is not in the loaded hero database.
    pub fn score_breakdown(&self, hero_id: &HeroId, team: TeamSide) -> Option<ScoreBreakdown> {
        let state = self.state.read();
        let hero = self.hero_db.iter().find(|h| h.id == *hero_id)?;
        Some(self.score_hero(hero, &state, team))
    }

    fn score_hero(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> ScoreBreakdown {
        let matchup = self.scorer.matchup_scorer.score(hero, state, team);
        let synergy = self.scorer.synergy_scorer.score(hero, state, team);
        let composition = self.scorer.composition_scorer.score(hero, state, team);
        let meta = self.get_meta_score(&hero.id);
        let comfort = self.get_comfort_score(&hero.id);
        let ml = match &self.scorer.ml_scorer {
            Some(ml_scorer) => ml_scorer.score(hero, state, team),
            None => 0.5,
        };
        let total = (matchup * W_MATCHUP
            + synergy * W_SYNERGY
            + composition * W_COMPOSITION
            + meta * W_META
            + comfort * W_COMFORT
            + ml * W_ML)
            .clamp(0.0, 1.0);

        ScoreBreakdown {
            matchup,
            synergy,
            composition,
            meta,
            comfort,
            ml,
            total,
        }
    }

    /// Generate recommendations for the current draft state
    pub fn get_recommendations(&self, team: TeamSide, limit: usize) -> RecommendationSet {
        let state = self.state.read();
        let is_radiant = matches!(team, TeamSide::Radiant);

        let mut scored_heroes: Vec<(HeroId, f32, Vec<String>)> = Vec::new();

        for hero_id in &state.available_heroes {
            let hero = match self.hero_db.iter().find(|h| h.id == *hero_id) {
                Some(h) => h,
                None => continue,
            };

            let mut explanations = Vec::new();

            // All scorer outputs are in [0, 1] with 0.5 neutral; weights sum
            // to 1.0 so the combined score stays in [0, 1] without saturating.
            let breakdown = self.score_hero(hero, &state, team);
            explanations.extend(self.scorer.matchup_scorer.explain(hero, &state, team));
            explanations.extend(self.scorer.synergy_scorer.explain(hero, &state, team));
            explanations.extend(self.scorer.composition_scorer.explain(hero, &state, team));

            if breakdown.meta > 0.5 {
                explanations.push(format!(
                    "Strong in current meta ({:.0}%)",
                    breakdown.meta * 100.0
                ));
            }

            if breakdown.comfort > 0.5 {
                explanations.push("High comfort hero".to_string());
            }

            if breakdown.ml > 0.55 {
                explanations.push(format!(
                    "ML model favors this pick ({:.0}%)",
                    breakdown.ml * 100.0
                ));
            } else if breakdown.ml < 0.45 {
                explanations.push(format!(
                    "ML model dislikes this pick ({:.0}%)",
                    breakdown.ml * 100.0
                ));
            }

            scored_heroes.push((hero_id.clone(), breakdown.total, explanations));
        }

        // Sort by score descending
        scored_heroes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let total = scored_heroes.len();
        let saturated = scored_heroes.iter().filter(|(_, s, _)| *s >= 1.0).count();
        debug!(
            "Score distribution across {} candidates: max={:.3} min={:.3} saturated_at_1.0={}",
            total,
            scored_heroes.first().map(|(_, s, _)| *s).unwrap_or(0.0),
            scored_heroes.last().map(|(_, s, _)| *s).unwrap_or(0.0),
            saturated
        );

        // Take top N and categorize
        let top_n = scored_heroes.into_iter().take(limit).collect::<Vec<_>>();

        self.categorize_recommendations(top_n, &state, is_radiant)
    }

    fn get_meta_score(&self, hero_id: &HeroId) -> f32 {
        self.meta_data
            .hero_meta_scores
            .get(hero_id)
            .copied()
            .unwrap_or(0.5)
    }

    fn get_comfort_score(&self, hero_id: &HeroId) -> f32 {
        if let Some(profile) = &self.player_profile {
            profile
                .hero_performance
                .get(hero_id)
                .map(|p| p.winrate)
                .unwrap_or(0.5)
        } else {
            0.5
        }
    }

    fn categorize_recommendations(
        &self,
        scored: Vec<(HeroId, f32, Vec<String>)>,
        state: &DraftState,
        is_radiant: bool,
    ) -> RecommendationSet {
        let mut top_picks = Vec::new();
        let mut safe_picks = Vec::new();
        let mut aggressive_picks = Vec::new();
        let mut comfort_picks = Vec::new();
        let mut anti_meta_picks = Vec::new();

        for (hero_id, score, explanations) in scored {
            let _hero = self.hero_db.iter().find(|h| h.id == hero_id).unwrap();
            let semantics = self.hero_semantics.get(&hero_id);

            let risk_level = if score > 0.7 {
                RiskLevel::Low
            } else if score > 0.4 {
                RiskLevel::Medium
            } else {
                RiskLevel::High
            };

            let role_fit = semantics
                .map(|s| s.roles_probability.keys().cloned().collect())
                .unwrap_or_default();

            let hero_id_clone = hero_id.clone();
            let recommendation = Recommendation {
                hero_id,
                score,
                confidence: score,
                explanations,
                risk_level,
                role_fit,
                category: RecommendationCategory::Top,
            };

            // Categorize based on attributes
            if let Some(sem) = semantics {
                if sem.scaling > 0.7 && sem.teamfight > 0.7 {
                    aggressive_picks.push(recommendation.clone());
                } else if sem.durability > 0.7 && sem.save > 0.5 {
                    safe_picks.push(recommendation.clone());
                } else {
                    top_picks.push(recommendation.clone());
                }

                if self.get_comfort_score(&hero_id_clone) > 0.65 {
                    comfort_picks.push(recommendation.clone());
                }

                if self.get_meta_score(&hero_id_clone) > 0.7 && score < 0.6 {
                    anti_meta_picks.push(recommendation.clone());
                }
            } else {
                top_picks.push(recommendation);
            }
        }

        // Generate team analysis
        let team_analysis = self.analyze_team(state, is_radiant);

        RecommendationSet {
            top_picks: top_picks.into_iter().take(5).collect(),
            safe_picks: safe_picks.into_iter().take(3).collect(),
            aggressive_picks: aggressive_picks.into_iter().take(3).collect(),
            comfort_picks: comfort_picks.into_iter().take(3).collect(),
            anti_meta_picks: anti_meta_picks.into_iter().take(3).collect(),
            team_analysis,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    fn analyze_team(&self, state: &DraftState, is_radiant: bool) -> TeamAnalysis {
        let ally_picks = if is_radiant {
            &state.radiant_picks
        } else {
            &state.dire_picks
        };

        let mut total_stun = 0.0;
        let mut total_initiation = 0.0;
        let mut total_durability = 0.0;
        let mut total_push = 0.0;
        let mut total_scaling = 0.0;
        let mut total_save = 0.0;

        for pick in ally_picks {
            if let Some(sem) = self.hero_semantics.get(&pick.hero_id) {
                total_stun += sem.stun;
                total_initiation += sem.initiation;
                total_durability += sem.durability;
                total_push += sem.push;
                total_scaling += sem.scaling;
                total_save += sem.save;
            }
        }

        let pick_count = ally_picks.len() as f32;
        if pick_count > 0.0 {
            total_stun /= pick_count;
            total_initiation /= pick_count;
            total_durability /= pick_count;
            total_push /= pick_count;
            total_scaling /= pick_count;
            total_save /= pick_count;
        }

        let mut missing = Vec::new();
        let mut strong = Vec::new();

        if total_stun < 0.5 {
            missing.push("Stun/Disable".to_string());
        } else {
            strong.push("Disable".to_string());
        }
        if total_initiation < 0.5 {
            missing.push("Initiation".to_string());
        } else {
            strong.push("Initiation".to_string());
        }
        if total_durability < 0.5 {
            missing.push("Frontline".to_string());
        } else {
            strong.push("Frontline".to_string());
        }
        if total_save < 0.3 {
            missing.push("Save/Support".to_string());
        } else {
            strong.push("Save".to_string());
        }
        if total_push < 0.3 {
            missing.push("Push".to_string());
        } else {
            strong.push("Push".to_string());
        }
        if total_scaling < 0.5 {
            missing.push("Late Game Scaling".to_string());
        } else {
            strong.push("Late Game".to_string());
        }

        let composition_score = (total_stun
            + total_initiation
            + total_durability
            + total_save
            + total_push
            + total_scaling)
            / 6.0;

        // Determine tempo
        let tempo = if total_scaling > 0.6 {
            TempoProfile::Late
        } else if total_initiation > 0.6 {
            TempoProfile::Early
        } else {
            TempoProfile::Mid
        };

        TeamAnalysis {
            missing,
            strong,
            composition_score,
            lane_balance: LaneBalance {
                safe_lane: 0.5,
                mid_lane: 0.5,
                offlane: 0.5,
            },
            tempo,
        }
    }

    pub fn set_player_profile(&mut self, profile: PlayerProfile) {
        self.player_profile = Some(profile);
    }

    pub fn update_meta(&mut self, meta: MetaData) {
        self.meta_data = meta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_heroes_are_explicit_and_usable() {
        let heroes = DraftEngine::fallback_heroes();
        assert_eq!(heroes.len(), 1);
        assert_eq!(heroes[0].id, HeroId(1));
        assert!(!heroes[0].roles.is_empty());
        assert!(heroes[0].cm_enabled);
    }

    #[test]
    fn test_draft_scorer_wires_matchup_and_synergy_data() {
        let matchup_data = [(
            HeroId(1),
            vec![HeroMatchup {
                hero_id: HeroId(1),
                vs_hero_id: HeroId(2),
                advantage: 0.1,
                winrate: 0.6,
                games: 100,
            }],
        )]
        .into_iter()
        .collect();
        let synergy_data = [(
            HeroId(1),
            vec![HeroSynergy {
                hero_id: HeroId(1),
                with_hero_id: HeroId(3),
                synergy: 0.2,
                winrate: 0.6,
                games: 100,
            }],
        )]
        .into_iter()
        .collect();

        let mut scorer = DraftScorer::default();
        scorer.set_data(matchup_data, synergy_data);

        assert_eq!(scorer.matchup_hero_count(), 1);
        assert_eq!(scorer.synergy_hero_count(), 1);
    }

    #[test]
    fn test_score_breakdown_exposes_components_and_weighted_total() {
        let mut engine = DraftEngine::new();
        let mut heroes = DraftEngine::fallback_heroes();
        let mut enemy = heroes[0].clone();
        enemy.id = HeroId(2);
        heroes.push(enemy);
        engine.hero_db = heroes;

        engine.state.write().dire_picks.push(DraftPick {
            hero_id: HeroId(2),
            player_id: PlayerId(0),
            order: 1,
            team: TeamSide::Dire,
        });
        engine.scorer.set_data(
            [(
                HeroId(1),
                vec![HeroMatchup {
                    hero_id: HeroId(1),
                    vs_hero_id: HeroId(2),
                    advantage: 0.1,
                    winrate: 0.6,
                    games: 100,
                }],
            )]
            .into_iter()
            .collect(),
            HashMap::new(),
        );

        let breakdown = engine
            .score_breakdown(&HeroId(1), TeamSide::Radiant)
            .unwrap();
        assert!((breakdown.matchup - 0.55).abs() < 1e-5);
        let expected_total = (breakdown.matchup * W_MATCHUP
            + breakdown.synergy * W_SYNERGY
            + breakdown.composition * W_COMPOSITION
            + breakdown.meta * W_META
            + breakdown.comfort * W_COMFORT
            + breakdown.ml * W_ML)
            .clamp(0.0, 1.0);
        assert!((breakdown.total - expected_total).abs() < 1e-6);
        assert!(engine
            .score_breakdown(&HeroId(999), TeamSide::Radiant)
            .is_none());
    }

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

    fn radiant_pick(hero_id: i32, order: i32) -> DraftPick {
        DraftPick {
            hero_id: HeroId(hero_id),
            player_id: PlayerId(0),
            order,
            team: TeamSide::Radiant,
        }
    }

    #[test]
    fn test_validate_pick_rejects_unknown_and_unavailable_heroes() {
        let engine = engine_with_heroes(2);

        let unknown = engine.validate_pick(&radiant_pick(99, 1));
        assert_eq!(unknown, Err(EngineError::UnknownHero { hero_id: 99 }));

        engine.add_pick(radiant_pick(1, 1));
        let duplicate = engine.validate_pick(&radiant_pick(1, 2));
        assert_eq!(duplicate, Err(EngineError::HeroUnavailable { hero_id: 1 }));
    }

    #[test]
    fn test_validate_pick_rejects_invalid_order_and_full_team() {
        let engine = engine_with_heroes(6);

        let bad_order = engine.validate_pick(&radiant_pick(1, 0));
        assert_eq!(bad_order, Err(EngineError::InvalidOrder { order: 0 }));

        for id in 1..=5 {
            engine.add_pick(radiant_pick(id, id));
        }
        let full = engine.validate_pick(&radiant_pick(6, 6));
        assert_eq!(
            full,
            Err(EngineError::TeamFull {
                team: "radiant".to_string(),
                count: 5,
            })
        );
    }

    #[test]
    fn test_undo_restores_pick_and_ban_state() {
        let engine = engine_with_heroes(3);
        engine.add_pick(radiant_pick(1, 1));
        engine.add_ban(DraftBan {
            hero_id: HeroId(2),
            order: 1,
            team: TeamSide::Dire,
        });

        assert_eq!(engine.undo_depth(), 2);

        let undone_ban = engine.undo_last_action();
        assert!(matches!(undone_ban, Some(DraftAction::Ban { .. })));
        let state = engine.get_state();
        assert!(state.bans.is_empty());
        assert!(state.available_heroes.contains(&HeroId(2)));

        let undone_pick = engine.undo_last_action();
        assert!(matches!(undone_pick, Some(DraftAction::Pick { .. })));
        let state = engine.get_state();
        assert!(state.radiant_picks.is_empty());
        assert_eq!(state.pick_number, 0);
        assert!(state.available_heroes.contains(&HeroId(1)));

        assert!(engine.undo_last_action().is_none());
        assert_eq!(engine.undo_depth(), 0);
    }

    #[test]
    fn test_reset_clears_undo_stack() {
        let engine = engine_with_heroes(2);
        engine.add_pick(radiant_pick(1, 1));
        assert_eq!(engine.undo_depth(), 1);

        engine.reset();
        assert_eq!(engine.undo_depth(), 0);
        assert!(engine.undo_last_action().is_none());
        assert_eq!(engine.get_state().available_heroes.len(), 2);
    }
}
