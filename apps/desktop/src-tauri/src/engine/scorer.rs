//! Scoring system for hero recommendations
//!
//! Modular trait-based scorers for matchup, synergy, and composition evaluation

use crate::ml::build_feature_map;
use crate::models::{
    DraftPick, DraftState, Hero, HeroId, HeroMatchup, HeroSemantics, PlayerId, TeamSide,
};
use std::collections::HashMap;

/// Canonicalize hero role names from any data source (heroes.json uses
/// capitalized names such as "Carry"; the scorer uses lowercase keys).
fn canonical_role(role: &str) -> Option<String> {
    let normalized = role.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" => None,
        "pusher" => Some("push".to_string()),
        "disable" => Some("disabler".to_string()),
        "initiation" => Some("initiator".to_string()),
        other => Some(other.to_string()),
    }
}

fn canonical_roles(roles: &[String]) -> Vec<String> {
    let mut result = Vec::with_capacity(roles.len());
    for role in roles {
        if let Some(role) = canonical_role(role) {
            if !result.contains(&role) {
                result.push(role);
            }
        }
    }
    result
}

/// Component weights for the recommendation score. All scorer outputs are in
/// [0, 1] with 0.5 as neutral, and the weights sum to 1.0 so the total score
/// stays in [0, 1] without saturating at the clamp.
pub const W_MATCHUP: f32 = 0.25;
pub const W_SYNERGY: f32 = 0.20;
pub const W_COMPOSITION: f32 = 0.15;
pub const W_META: f32 = 0.10;
pub const W_COMFORT: f32 = 0.05;
pub const W_ML: f32 = 0.25;

/// Trait for all scorers in the recommendation pipeline
pub trait Scorer: Send + Sync {
    fn score(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> f32;
    fn explain(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> Vec<String>;
}

/// Matchup Scorer - evaluates how well a hero performs against enemy picks
#[derive(Clone)]
pub struct MatchupScorer {
    matchup_data: HashMap<HeroId, Vec<crate::models::HeroMatchup>>,
}

impl MatchupScorer {
    pub fn new() -> Self {
        Self {
            matchup_data: HashMap::new(),
        }
    }

    pub fn with_data(matchup_data: HashMap<HeroId, Vec<crate::models::HeroMatchup>>) -> Self {
        Self { matchup_data }
    }

    pub fn hero_count(&self) -> usize {
        self.matchup_data.len()
    }
}

impl Default for MatchupScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for MatchupScorer {
    fn score(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> f32 {
        let enemy_heroes: Vec<HeroId> = state
            .radiant_picks
            .iter()
            .chain(state.dire_picks.iter())
            .filter(|p| p.team != team)
            .map(|p| p.hero_id.clone())
            .collect();

        if enemy_heroes.is_empty() {
            return 0.5; // Neutral when no enemy picks
        }

        let mut total_advantage = 0.0;
        let mut count = 0;

        for enemy_id in enemy_heroes {
            if let Some(matchups) = self.matchup_data.get(&hero.id) {
                if let Some(matchup) = matchups.iter().find(|m| m.vs_hero_id == enemy_id) {
                    total_advantage += matchup.advantage;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.5
        } else {
            // Normalize from [-1, 1] to [0, 1]
            ((total_advantage / count as f32) + 1.0) / 2.0
        }
    }

    fn explain(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> Vec<String> {
        let mut explanations = Vec::new();

        let enemy_heroes: Vec<HeroId> = state
            .radiant_picks
            .iter()
            .chain(state.dire_picks.iter())
            .filter(|p| p.team != team)
            .map(|p| p.hero_id.clone())
            .collect();

        for enemy_id in enemy_heroes {
            if let Some(matchups) = self.matchup_data.get(&hero.id) {
                if let Some(matchup) = matchups.iter().find(|m| m.vs_hero_id == enemy_id) {
                    if matchup.advantage > 0.15 {
                        explanations.push(format!(
                            "Strong vs {} ({:.0}% winrate)",
                            enemy_id.0,
                            matchup.winrate * 100.0
                        ));
                    } else if matchup.advantage < -0.15 {
                        explanations.push(format!(
                            "Weak vs {} ({:.0}% winrate)",
                            enemy_id.0,
                            matchup.winrate * 100.0
                        ));
                    }
                }
            }
        }

        explanations
    }
}

/// Synergy Scorer - evaluates how well a hero synergizes with ally picks
#[derive(Clone)]
pub struct SynergyScorer {
    synergy_data: HashMap<HeroId, Vec<crate::models::HeroSynergy>>,
}

impl SynergyScorer {
    pub fn new() -> Self {
        Self {
            synergy_data: HashMap::new(),
        }
    }

    pub fn with_data(synergy_data: HashMap<HeroId, Vec<crate::models::HeroSynergy>>) -> Self {
        Self { synergy_data }
    }

    pub fn hero_count(&self) -> usize {
        self.synergy_data.len()
    }
}

impl Default for SynergyScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for SynergyScorer {
    fn score(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> f32 {
        let ally_heroes: Vec<HeroId> = state
            .radiant_picks
            .iter()
            .chain(state.dire_picks.iter())
            .filter(|p| p.team == team)
            .map(|p| p.hero_id.clone())
            .collect();

        if ally_heroes.is_empty() {
            return 0.5;
        }

        let mut total_synergy = 0.0;
        let mut count = 0;

        for ally_id in ally_heroes {
            if let Some(synergies) = self.synergy_data.get(&hero.id) {
                if let Some(synergy) = synergies.iter().find(|s| s.with_hero_id == ally_id) {
                    total_synergy += synergy.synergy;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.5
        } else {
            // Raw synergy is signed (typically [-0.8, 0.8]); map to [0, 1]
            // with 0.5 as neutral so it combines cleanly with other components.
            ((total_synergy / count as f32) + 1.0) / 2.0
        }
    }

    fn explain(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> Vec<String> {
        let mut explanations = Vec::new();

        let ally_heroes: Vec<HeroId> = state
            .radiant_picks
            .iter()
            .chain(state.dire_picks.iter())
            .filter(|p| p.team == team)
            .map(|p| p.hero_id.clone())
            .collect();

        for ally_id in ally_heroes {
            if let Some(synergies) = self.synergy_data.get(&hero.id) {
                if let Some(synergy) = synergies.iter().find(|s| s.with_hero_id == ally_id) {
                    if synergy.synergy > 0.15 {
                        explanations.push(format!(
                            "Synergizes with {} ({:.0}% winrate)",
                            ally_id.0,
                            synergy.winrate * 100.0
                        ));
                    }
                }
            }
        }

        explanations
    }
}

/// Composition Scorer - evaluates if a hero fills missing roles in team composition
#[derive(Clone)]
pub struct CompositionScorer {
    role_weights: HashMap<String, f32>,
    hero_roles: HashMap<HeroId, Vec<String>>,
}

impl CompositionScorer {
    pub fn new() -> Self {
        let mut role_weights = HashMap::new();
        role_weights.insert("initiator".to_string(), 1.0);
        role_weights.insert("disabler".to_string(), 1.0);
        role_weights.insert("durable".to_string(), 0.8);
        role_weights.insert("support".to_string(), 0.7);
        role_weights.insert("nuker".to_string(), 0.6);
        role_weights.insert("push".to_string(), 0.5);
        role_weights.insert("escape".to_string(), 0.4);
        role_weights.insert("carry".to_string(), 0.9);

        Self {
            role_weights,
            hero_roles: HashMap::new(),
        }
    }

    pub fn with_hero_roles(mut self, hero_roles: HashMap<HeroId, Vec<String>>) -> Self {
        self.hero_roles = hero_roles
            .into_iter()
            .map(|(hero_id, roles)| (hero_id, canonical_roles(&roles)))
            .collect();
        self
    }

    fn team_role_counts(&self, state: &DraftState, team: TeamSide) -> HashMap<String, f32> {
        let mut team_roles: HashMap<String, f32> = HashMap::new();
        for pick in state.radiant_picks.iter().chain(state.dire_picks.iter()) {
            if pick.team != team {
                continue;
            }
            if let Some(roles) = self.hero_roles.get(&pick.hero_id) {
                for role in roles {
                    *team_roles.entry(role.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
        team_roles
    }
}

impl Default for CompositionScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl Scorer for CompositionScorer {
    fn score(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> f32 {
        let team_roles = self.team_role_counts(state, team);

        // Score this hero based on missing roles
        let mut score = 0.0;
        let mut total_weight = 0.0;

        for role in canonical_roles(&hero.roles) {
            let weight = self.role_weights.get(&role).copied().unwrap_or(0.5);
            let current_count = team_roles.get(&role).copied().unwrap_or(0.0);

            // Higher score if team lacks this role
            let missing_factor = (1.0 - (current_count / 3.0).min(1.0)).max(0.2);
            score += weight * missing_factor;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            0.5
        } else {
            (score / total_weight).clamp(0.0, 1.0)
        }
    }

    fn explain(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> Vec<String> {
        let mut explanations = Vec::new();

        let team_roles = self.team_role_counts(state, team);

        for role in canonical_roles(&hero.roles) {
            let current = team_roles.get(&role).copied().unwrap_or(0.0) as i32;
            if current == 0 {
                explanations.push(format!("Fills missing {}", role));
            } else if current == 1 {
                explanations.push(format!("Adds second {}", role));
            }
        }

        explanations
    }
}

/// ML Scorer - uses trained ONNX model for statistical enhancement
#[derive(Clone)]
pub struct MLScorer {
    hero_semantics: HashMap<HeroId, HeroSemantics>,
    meta_data: crate::models::MetaData,
    hero_db: Vec<Hero>,
    matchup_data: HashMap<HeroId, Vec<HeroMatchup>>,
}

impl MLScorer {
    pub fn new(
        hero_semantics: HashMap<HeroId, HeroSemantics>,
        meta_data: crate::models::MetaData,
        hero_db: Vec<Hero>,
        matchup_data: HashMap<HeroId, Vec<HeroMatchup>>,
    ) -> Self {
        Self {
            hero_semantics,
            meta_data,
            hero_db,
            matchup_data,
        }
    }
}

impl Scorer for MLScorer {
    fn score(&self, hero: &Hero, state: &DraftState, team: TeamSide) -> f32 {
        // Get global ML engine
        let ml_engine = crate::ml::get_ml_engine();
        let engine_guard = ml_engine.read();

        if !engine_guard.is_loaded() {
            return 0.5; // Neutral if model not loaded
        }

        let is_radiant = matches!(team, TeamSide::Radiant);

        // Simulate the candidate pick so the model sees a complete draft
        let mut simulated_state = state.clone();
        let pick = DraftPick {
            hero_id: hero.id.clone(),
            player_id: PlayerId(0),
            order: simulated_state.pick_number + 1,
            team,
        };
        simulated_state.available_heroes.retain(|h| h != &hero.id);
        if is_radiant {
            simulated_state.radiant_picks.push(pick);
        } else {
            simulated_state.dire_picks.push(pick);
        }

        // Build the full feature vector and run inference
        let feature_schema = engine_guard.feature_schema();
        let features = build_feature_map(
            &simulated_state,
            &self.hero_db,
            &self.hero_semantics,
            &self.meta_data,
            &self.matchup_data,
            feature_schema.as_deref(),
        );

        match engine_guard.predict(&features) {
            Ok(prediction) => {
                // Model outputs Radiant win probability; convert to the picking team's value
                if is_radiant {
                    prediction
                } else {
                    1.0 - prediction
                }
            }
            Err(_) => 0.5,
        }
    }

    fn explain(&self, _hero: &Hero, _state: &DraftState, _team: TeamSide) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AttackType, PrimaryAttribute};

    fn hero(id: i32, roles: &[&str]) -> Hero {
        Hero {
            id: HeroId(id),
            name: format!("hero_{id}"),
            localized_name: format!("Hero {id}"),
            primary_attr: PrimaryAttribute::Str,
            attack_type: AttackType::Melee,
            roles: roles.iter().map(|role| role.to_string()).collect(),
            legs: 2,
            img: String::new(),
            icon: String::new(),
            base_health: 0,
            base_mana: 0,
            base_armor: 0.0,
            base_attack_min: 0,
            base_attack_max: 0,
            base_str: 0.0,
            base_agi: 0.0,
            base_int: 0.0,
            str_gain: 0.0,
            agi_gain: 0.0,
            int_gain: 0.0,
            attack_range: 0,
            projectile_speed: 0,
            attack_rate: 0.0,
            move_speed: 0,
            turn_rate: 0.0,
            cm_enabled: true,
        }
    }

    fn pick(hero_id: i32, team: TeamSide) -> DraftPick {
        DraftPick {
            hero_id: HeroId(hero_id),
            player_id: PlayerId(0),
            order: 1,
            team,
        }
    }

    #[test]
    fn test_canonical_role_normalizes_case_and_aliases() {
        assert_eq!(canonical_role("Carry"), Some("carry".to_string()));
        assert_eq!(canonical_role("SUPPORT"), Some("support".to_string()));
        assert_eq!(canonical_role("Pusher"), Some("push".to_string()));
        assert_eq!(canonical_role("Disable"), Some("disabler".to_string()));
        assert_eq!(canonical_role("  "), None);
    }

    #[test]
    fn test_composition_scorer_uses_capitalized_hero_roles() {
        let hero_roles = [(HeroId(1), vec!["Carry".to_string()])]
            .into_iter()
            .collect();
        let scorer = CompositionScorer::new().with_hero_roles(hero_roles);

        let mut state = DraftState::default();
        state.radiant_picks.push(pick(1, TeamSide::Radiant));

        let candidate = hero(2, &["Carry"]);
        let score = scorer.score(&candidate, &state, TeamSide::Radiant);
        assert!(
            (score - 2.0 / 3.0).abs() < 1e-5,
            "expected carry weight to apply and ally count to be normalized, got {score}"
        );
        assert_eq!(scorer.role_weights.get("carry"), Some(&0.9));
        assert_eq!(scorer.role_weights.get("push"), Some(&0.5));
    }

    #[test]
    fn test_matchup_scorer_uses_loaded_data() {
        let data = [(
            HeroId(1),
            vec![crate::models::HeroMatchup {
                hero_id: HeroId(1),
                vs_hero_id: HeroId(2),
                advantage: 0.1,
                winrate: 0.6,
                games: 100,
            }],
        )]
        .into_iter()
        .collect();
        let scorer = MatchupScorer::with_data(data);

        let mut state = DraftState::default();
        state.dire_picks.push(pick(2, TeamSide::Dire));

        let score = scorer.score(&hero(1, &[]), &state, TeamSide::Radiant);
        assert!((score - 0.55).abs() < 1e-5, "got {score}");
        assert_eq!(scorer.score(&hero(99, &[]), &state, TeamSide::Radiant), 0.5);
        assert_eq!(
            MatchupScorer::new().score(&hero(1, &[]), &state, TeamSide::Radiant),
            0.5
        );

        let updated = [(
            HeroId(1),
            vec![crate::models::HeroMatchup {
                hero_id: HeroId(1),
                vs_hero_id: HeroId(2),
                advantage: -0.1,
                winrate: 0.4,
                games: 100,
            }],
        )]
        .into_iter()
        .collect();
        let updated_score =
            MatchupScorer::with_data(updated).score(&hero(1, &[]), &state, TeamSide::Radiant);
        assert!((updated_score - 0.45).abs() < 1e-5, "got {updated_score}");
        assert!((updated_score - score).abs() > 1e-5);
    }

    #[test]
    fn test_synergy_scorer_uses_loaded_data() {
        let data = [(
            HeroId(1),
            vec![crate::models::HeroSynergy {
                hero_id: HeroId(1),
                with_hero_id: HeroId(2),
                synergy: 0.2,
                winrate: 0.6,
                games: 100,
            }],
        )]
        .into_iter()
        .collect();
        let scorer = SynergyScorer::with_data(data);

        let mut state = DraftState::default();
        state.radiant_picks.push(pick(2, TeamSide::Radiant));

        let score = scorer.score(&hero(1, &[]), &state, TeamSide::Radiant);
        assert!((score - 0.6).abs() < 1e-5, "got {score}");
        assert_eq!(scorer.score(&hero(99, &[]), &state, TeamSide::Radiant), 0.5);
        assert_eq!(
            SynergyScorer::new().score(&hero(1, &[]), &state, TeamSide::Radiant),
            0.5
        );
    }
}
