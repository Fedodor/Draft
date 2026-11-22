//! Draft Simulation / Tree Search Layer
//!
//! Implements the decision engine that evaluates candidate picks by simulating
//! opponent responses and computing expected value (as described in UPDATEML.md)

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

impl SearchConfig {
    /// Clamp user-supplied bounds so a single IPC call cannot request an
    /// unbounded tree search. Returns the config unchanged when already valid.
    pub fn sanitized(mut self) -> Self {
        self.max_candidates = self.max_candidates.clamp(1, 50);
        self.max_depth = self.max_depth.min(3);
        self.beam_width = self.beam_width.clamp(1, 20);
        self
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

    /// Evaluate all candidate picks and return ranked results
    pub fn evaluate_candidates(&self, team: TeamSide, limit: usize) -> Vec<SimulationResult> {
        let state = self.engine.get_state();
        let is_radiant = matches!(team, TeamSide::Radiant);

        // One opponent shortlist per turn, shared by every candidate branch.
        // Recomputing it per branch meant a full pool scoring per candidate.
        let opponent_beam = if self.config.max_depth > 0 && self.config.opponent_model_enabled {
            self.get_top_candidates(&state, !is_radiant, self.config.beam_width)
        } else {
            Vec::new()
        };

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
            let result = self.evaluate_pick(hero, &state, is_radiant, 0, &opponent_beam);
            results.push(result);
        }

        // Sort by expected value descending
        results.sort_by(|a, b| b.expected_value.total_cmp(&a.expected_value));

        results.into_iter().take(limit).collect()
    }

    /// Get the top candidate heroes with the same popularity-first ordering the
    /// recommendation list uses, so the search starts from the shortlist the
    /// user is looking at instead of a second, older set of weights. The pool
    /// is scored in one pass, including one batched ML call per state.
    fn get_top_candidates(&self, state: &DraftState, is_radiant: bool, max: usize) -> Vec<HeroId> {
        let team = if is_radiant {
            TeamSide::Radiant
        } else {
            TeamSide::Dire
        };
        let mut scored = self.engine.score_pool_for_state(state, team, max);

        // Score-only stable sort, matching `DraftEngine::get_recommendations`.
        scored.sort_by(|a, b| b.1.total.total_cmp(&a.1.total));
        scored.into_iter().take(max).map(|(id, _)| id).collect()
    }

    /// Evaluate a single pick using tree search
    fn evaluate_pick(
        &self,
        hero: &Hero,
        state: &DraftState,
        is_radiant: bool,
        depth: usize,
        opponent_beam: &[HeroId],
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
                self.evaluate_opponent_responses(&simulated_state, opponent_beam, is_radiant)
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

    /// Immediate value of a pick: the shipped popularity-first score for the
    /// simulated state, so search and recommendation agree on what a pick is
    /// worth as well as on the order of the shortlist.
    fn calculate_immediate_value(&self, hero: &Hero, state: &DraftState, is_radiant: bool) -> f32 {
        let team = if is_radiant {
            TeamSide::Radiant
        } else {
            TeamSide::Dire
        };
        let popularity = self.engine.popularity_score(state, &hero.id);
        self.engine.score_hero(hero, state, team, popularity).total
    }

    /// Evaluate opponent's likely responses (minimax-style)
    fn evaluate_opponent_responses(
        &self,
        state: &DraftState,
        opponent_beam: &[HeroId],
        we_are_radiant: bool,
    ) -> f32 {
        let opponent_is_radiant = !we_are_radiant;
        // Opponent will pick the move that minimizes our value (maximizes theirs)
        let mut worst_case_value: f32 = 1.0;
        let mut evaluated = false;

        for opp_hero_id in opponent_beam {
            // The shared shortlist was computed before our pick; a hero it
            // named may be the one this branch picked.
            if !state.available_heroes.contains(opp_hero_id) {
                continue;
            }
            let Some(opp_hero) = self.engine.hero_db.iter().find(|h| h.id == *opp_hero_id) else {
                continue;
            };

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
            let our_value = self.calculate_our_value_after_opponent(&opp_state, we_are_radiant);
            evaluated = true;
            worst_case_value = worst_case_value.min(our_value);
        }

        if evaluated {
            worst_case_value
        } else {
            0.0
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_popularity() -> DraftEngine {
        let mut engine = DraftEngine::new();
        engine.hero_db = (1..=4)
            .map(|id| {
                let mut hero = DraftEngine::fallback_heroes().remove(0);
                hero.id = HeroId(id);
                hero
            })
            .collect();
        engine.state.write().available_heroes =
            engine.hero_db.iter().map(|hero| hero.id.clone()).collect();
        engine.meta_data.hero_pickrates = [
            (HeroId(1), 0.05),
            (HeroId(2), 0.40),
            (HeroId(3), 0.20),
            (HeroId(4), 0.10),
        ]
        .into_iter()
        .collect();
        engine
    }

    #[test]
    fn test_candidate_preselection_matches_the_recommendation_order() {
        let engine = engine_with_popularity();
        let simulator = DraftSimulator::new(Arc::new(engine.clone()), SearchConfig::default());
        let state = engine.get_state();

        let candidates = simulator.get_top_candidates(&state, true, 4);
        let listed: Vec<HeroId> = engine
            .get_recommendations(TeamSide::Radiant, 4)
            .top_picks
            .iter()
            .map(|recommendation| recommendation.hero_id.clone())
            .collect();

        assert_eq!(candidates, listed);
    }
}
