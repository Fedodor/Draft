//! Draft Simulation / Tree Search Layer
//!
//! Implements the decision engine that evaluates candidate picks by simulating
//! opponent responses and computing expected value (as described in UPDATEML.md)

use crate::engine::scorer::{
    CompositionScorer, MLScorer, MatchupScorer, Scorer, SynergyScorer, W_COMFORT, W_COMPOSITION,
    W_MATCHUP, W_META, W_ML, W_SYNERGY,
};
use crate::engine::DraftEngine;
use crate::models::*;

/// Configuration for tree search
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchConfig {
    pub max_candidates: usize,        // Top-K candidates to evaluate (10-20)
    pub max_depth: usize,             // Lookahead depth (1-2 steps)
    pub beam_width: usize,            // Beam search width
    pub opponent_model_enabled: bool, // Use opponent response model
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_candidates: 15,
            max_depth: 2,
            beam_width: 5,
            opponent_model_enabled: true,
        }
    }
}

/// Result of a simulated pick evaluation
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimulationResult {
    pub hero_id: HeroId,
    pub expected_value: f32,
    pub immediate_value: f32,
    pub opponent_response_value: f32,
    pub confidence: f32,
    pub explanation: Vec<String>,
}

/// Draft Simulator - evaluates picks using tree search
pub struct DraftSimulator {
    engine: Arc<DraftEngine>,
    config: SearchConfig,
}

impl DraftSimulator {
    pub fn new(engine: Arc<DraftEngine>, config: SearchConfig) -> Self {
        Self { engine, config }
    }

    // Helper methods to access scorers from engine
    fn matchup_scorer(&self) -> &MatchupScorer {
        &self.engine.scorer.matchup_scorer
    }

    fn synergy_scorer(&self) -> &SynergyScorer {
        &self.engine.scorer.synergy_scorer
    }

    fn composition_scorer(&self) -> &CompositionScorer {
        &self.engine.scorer.composition_scorer
    }

    fn ml_scorer(&self) -> Option<&MLScorer> {
        self.engine.scorer.ml_scorer.as_ref()
    }

    /// Evaluate all candidate picks and return ranked results
    pub fn evaluate_candidates(&self, team: TeamSide, limit: usize) -> Vec<SimulationResult> {
        let state = self.engine.get_state();
        let is_radiant = matches!(team, TeamSide::Radiant);

        // Get top candidates using heuristic scoring
        let candidates = self.get_top_candidates(&state, is_radiant, self.config.max_candidates);

        let mut results = Vec::new();

        for candidate_id in candidates {
            let hero = self
                .engine
                .hero_db
                .iter()
                .find(|h| h.id == candidate_id)
                .expect("Candidate hero should exist");

            // Evaluate this candidate
            let result = self.evaluate_pick(hero, &state, is_radiant, 0);
            results.push(result);
        }

        // Sort by expected value descending
        results.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());

        results.into_iter().take(limit).collect()
    }

    /// Get top candidate heroes using fast heuristic scoring
    fn get_top_candidates(&self, state: &DraftState, is_radiant: bool, max: usize) -> Vec<HeroId> {
        let team = if is_radiant {
            TeamSide::Radiant
        } else {
            TeamSide::Dire
        };
        let mut scored: Vec<(HeroId, f32)> = Vec::new();

        for hero_id in &state.available_heroes {
            let hero = self
                .engine
                .hero_db
                .iter()
                .find(|h| h.id == *hero_id)
                .unwrap();

            // Fast heuristic score
            let mut score = 0.0;

            // Matchup
            score += self.matchup_scorer().score(hero, state, team) * 0.3;

            // Synergy
            score += self.synergy_scorer().score(hero, state, team) * 0.3;

            // Composition
            score += self.composition_scorer().score(hero, state, team) * 0.2;

            // Meta
            score += self.engine.get_meta_score(hero_id) * 0.1;

            // Comfort
            score += self.engine.get_comfort_score(hero_id) * 0.1;

            scored.push((hero_id.clone(), score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(max).map(|(id, _)| id).collect()
    }

    /// Evaluate a single pick using tree search
    fn evaluate_pick(
        &self,
        hero: &Hero,
        state: &DraftState,
        is_radiant: bool,
        depth: usize,
    ) -> SimulationResult {
        // Simulate the pick
        let mut simulated_state = state.clone();
        let pick = DraftPick {
            hero_id: hero.id.clone(),
            player_id: PlayerId(0),
            order: simulated_state.pick_number + 1,
            team: if is_radiant {
                TeamSide::Radiant
            } else {
                TeamSide::Dire
            },
        };

        // Apply pick to simulated state
        simulated_state.available_heroes.retain(|h| h != &hero.id);
        if is_radiant {
            simulated_state.radiant_picks.push(pick);
        } else {
            simulated_state.dire_picks.push(pick);
        }
        simulated_state.pick_number += 1;
        simulated_state.current_pick = Some(if is_radiant {
            TeamSide::Dire
        } else {
            TeamSide::Radiant
        });

        // Immediate value (heuristic + ML)
        let immediate_value = self.calculate_immediate_value(hero, &simulated_state, is_radiant);

        // Opponent response value (if not at max depth)
        let opponent_response_value =
            if depth < self.config.max_depth && self.config.opponent_model_enabled {
                self.evaluate_opponent_responses(&simulated_state, !is_radiant, depth + 1)
            } else {
                0.0
            };

        let expected_value = immediate_value + opponent_response_value * 0.5; // Weight opponent response

        SimulationResult {
            hero_id: hero.id.clone(),
            expected_value,
            immediate_value,
            opponent_response_value,
            confidence: immediate_value, // Use immediate as confidence proxy
            explanation: vec![
                format!("Immediate value: {:.2}", immediate_value),
                format!("Opponent response: {:.2}", opponent_response_value),
                format!("Expected value: {:.2}", expected_value),
            ],
        }
    }

    /// Calculate immediate value of a pick using all scorers
    fn calculate_immediate_value(&self, hero: &Hero, state: &DraftState, is_radiant: bool) -> f32 {
        let team = if is_radiant {
            TeamSide::Radiant
        } else {
            TeamSide::Dire
        };
        let mut total: f32 = 0.0;

        // Matchup
        total += self.matchup_scorer().score(hero, state, team) * W_MATCHUP;

        // Synergy
        total += self.synergy_scorer().score(hero, state, team) * W_SYNERGY;

        // Composition
        total += self.composition_scorer().score(hero, state, team) * W_COMPOSITION;

        // Meta
        total += self.engine.get_meta_score(&hero.id) * W_META;

        // Comfort
        total += self.engine.get_comfort_score(&hero.id) * W_COMFORT;

        // ML (if available)
        if let Some(ml_scorer) = self.ml_scorer() {
            total += ml_scorer.score(hero, state, team) * W_ML;
        }

        total.clamp(0.0, 1.0)
    }

    /// Evaluate opponent's likely responses (minimax-style)
    fn evaluate_opponent_responses(
        &self,
        state: &DraftState,
        opponent_is_radiant: bool,
        _depth: usize,
    ) -> f32 {
        // Get opponent's best counter-picks
        let opponent_candidates =
            self.get_top_candidates(state, opponent_is_radiant, self.config.beam_width);

        if opponent_candidates.is_empty() {
            return 0.0;
        }

        // Opponent will pick the move that minimizes our value (maximizes theirs)
        let mut worst_case_value: f32 = 1.0;

        for opp_hero_id in opponent_candidates {
            let opp_hero = self
                .engine
                .hero_db
                .iter()
                .find(|h| h.id == opp_hero_id)
                .unwrap();

            // Simulate opponent pick
            let mut opp_state = state.clone();
            let pick = DraftPick {
                hero_id: opp_hero.id.clone(),
                player_id: PlayerId(0),
                order: opp_state.pick_number + 1,
                team: if opponent_is_radiant {
                    TeamSide::Radiant
                } else {
                    TeamSide::Dire
                },
            };

            opp_state.available_heroes.retain(|h| h != &opp_hero.id);
            if opponent_is_radiant {
                opp_state.radiant_picks.push(pick);
            } else {
                opp_state.dire_picks.push(pick);
            }
            opp_state.pick_number += 1;
            opp_state.current_pick = Some(if opponent_is_radiant {
                TeamSide::Dire
            } else {
                TeamSide::Radiant
            });

            // Evaluate from our perspective after opponent pick
            let our_value =
                self.calculate_our_value_after_opponent(&opp_state, !opponent_is_radiant);
            worst_case_value = worst_case_value.min(our_value);
        }

        worst_case_value
    }

    /// Calculate our draft value after opponent has picked
    fn calculate_our_value_after_opponent(&self, state: &DraftState, we_are_radiant: bool) -> f32 {
        // Use the same heuristic evaluation
        let (ally_picks, enemy_picks) = if we_are_radiant {
            (&state.radiant_picks, &state.dire_picks)
        } else {
            (&state.dire_picks, &state.radiant_picks)
        };

        let mut total_stun = 0.0;
        let mut total_initiation = 0.0;
        let mut total_durability = 0.0;

        for pick in ally_picks {
            if let Some(sem) = self.engine.hero_semantics.get(&pick.hero_id) {
                total_stun += sem.stun;
                total_initiation += sem.initiation;
                total_durability += sem.durability;
            }
        }

        for pick in enemy_picks {
            if let Some(sem) = self.engine.hero_semantics.get(&pick.hero_id) {
                total_stun -= sem.stun * 0.5;
                total_initiation -= sem.initiation * 0.5;
                total_durability -= sem.durability * 0.5;
            }
        }

        let ally_count = ally_picks.len().max(1) as f32;
        total_stun /= ally_count;
        total_initiation /= ally_count;
        total_durability /= ally_count;

        // Normalize to [0, 1]
        ((total_stun + total_initiation + total_durability) / 3.0 + 1.0) / 2.0
    }
}

/// Integration with recommendation system
impl DraftEngine {
    /// Get recommendations with tree search enhancement
    pub fn get_recommendations_with_search(
        &self,
        team: TeamSide,
        limit: usize,
        config: SearchConfig,
    ) -> Vec<SimulationResult> {
        let simulator = DraftSimulator::new(Arc::new(self.clone()), config);
        simulator.evaluate_candidates(team, limit)
    }
}

use std::sync::Arc;
