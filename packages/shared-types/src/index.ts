// Shared types for Smart Draft Assistant
// Ensures type safety across TypeScript frontend and Rust backend (via Serde)

export type HeroId = number;
export type PlayerId = number;
export type MatchId = number;

export interface Hero {
  id: HeroId;
  name: string;
  localized_name: string;
  primary_attr: 'str' | 'agi' | 'int' | 'all';
  attack_type: 'Melee' | 'Ranged';
  roles: string[];
  legs: number;
  img: string;
  icon: string;
  base_health: number;
  base_mana: number;
  base_armor: number;
  base_attack_min: number;
  base_attack_max: number;
  base_str: number;
  base_agi: number;
  base_int: number;
  str_gain: number;
  agi_gain: number;
  int_gain: number;
  attack_range: number;
  projectile_speed: number;
  attack_rate: number;
  move_speed: number;
  turn_rate: number;
  cm_enabled: boolean;
}

// Semantic hero attributes for ML and scoring
export interface HeroSemantics {
  hero_id: HeroId;
  initiation: number;
  stun: number;
  save: number;
  push: number;
  scaling: number;
  mobility: number;
  durability: number;
  teamfight: number;
  splitpush: number;
  vision: number;
  magic_damage: number;
  physical_damage: number;
  pure_damage: number;
  disable_count: number;
  lane_strength: {
    early: number;
    mid: number;
    late: number;
  };
  tempo_profile: 'early' | 'mid' | 'late';
  roles_probability: Record<string, number>;
}

export interface HeroMatchup {
  hero_id: HeroId;
  vs_hero_id: HeroId;
  advantage: number;
  winrate: number;
  games: number;
}

export interface HeroSynergy {
  hero_id: HeroId;
  with_hero_id: HeroId;
  synergy: number;
  winrate: number;
  games: number;
}

export type TeamSide = 'radiant' | 'dire';

export interface DraftPick {
  hero_id: HeroId;
  player_id: PlayerId;
  order: number;
  team: TeamSide;
}

export interface DraftBan {
  hero_id: HeroId;
  order: number;
  team: TeamSide;
}

export interface DraftState {
  radiant_picks: DraftPick[];
  dire_picks: DraftPick[];
  bans: DraftBan[];
  current_pick: TeamSide | null;
  pick_number: number;
  is_picking_phase: boolean;
  available_heroes: HeroId[];
  patch_version: string;
  timestamp: number;
}

export interface Recommendation {
  hero_id: HeroId;
  score: number;
  confidence: number;
  explanations: string[];
  risk_level: 'low' | 'medium' | 'high';
  role_fit: string[];
  category: 'top' | 'safe' | 'aggressive' | 'comfort' | 'anti_meta';
}

export interface RecommendationSet {
  top_picks: Recommendation[];
  safe_picks: Recommendation[];
  aggressive_picks: Recommendation[];
  comfort_picks: Recommendation[];
  anti_meta_picks: Recommendation[];
  team_analysis: TeamAnalysis;
  timestamp: number;
}

export interface TeamAnalysis {
  missing: string[];
  strong: string[];
  composition_score: number;
  lane_balance: LaneBalance;
  tempo: 'early' | 'mid' | 'late';
}

export interface LaneBalance {
  safe_lane: number;
  mid_lane: number;
  offlane: number;
}

export interface PlayerProfile {
  steam_id: string;
  hero_performance: Record<HeroId, HeroPerformance>;
  preferred_roles: string[];
  preferred_tempo: 'early' | 'mid' | 'late';
  aggressive_tendency: number;
  total_matches: number;
}

export interface HeroPerformance {
  hero_id: HeroId;
  games: number;
  wins: number;
  winrate: number;
  avg_kda: number;
  last_played: number;
}

export interface MetaData {
  patch_version: string;
  hero_meta_scores: Record<HeroId, number>;
  hero_pickrates: Record<HeroId, number>;
  hero_winrates: Record<HeroId, number>;
  updated_at: number;
}

export interface GSIPlayer {
  account_id: PlayerId;
  hero_id: HeroId;
  team: TeamSide;
}

// Real Dota 2 GSI draft component: fixed slot ids (0 = empty), not arrays.
export interface GSITeamDraft {
  team_id?: number | null;
  pick0_id: number;
  pick1_id: number;
  pick2_id: number;
  pick3_id: number;
  pick4_id: number;
  ban0_id: number;
  ban1_id: number;
  ban2_id: number;
  ban3_id: number;
  ban4_id: number;
}

export interface GSIDraft {
  activeteam?: number | null;
  pick?: boolean | null;
  activeteam_time_remaining?: number | null;
  radiant_bonus_time?: number | null;
  dire_bonus_time?: number | null;
  team2?: GSITeamDraft | null;
  team3?: GSITeamDraft | null;
}

export interface GSIGameState {
  map: {
    game_state: string;
    game_time: number;
    clock_time: number;
  };
  player?: { team_name?: string | null } | null;
  hero: {
    id: HeroId;
    level: number;
  };
  draft?: GSIDraft;
  players: Record<string, GSIPlayer>;
}

export interface AppConfig {
  gsi_port: number;
  gsi_auto_start: boolean;
  gsi_auth_token: string | null;
  overlay_enabled: boolean;
  overlay_position: { x: number; y: number };
  overlay_size: { width: number; height: number };
  ml_enabled: boolean;
  ml_model_path: string;
  auto_update_meta: boolean;
  preferred_language: string;
}

export interface GsiStatus {
  running: boolean;
  port: number;
  endpoint: string;
  auth_required: boolean;
  last_error: string | null;
}

export interface TauriCommand<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type DraftPhase = 'ban' | 'pick' | 'none';

export interface EngineState {
  draft: DraftState;
  recommendations: RecommendationSet | null;
  is_processing: boolean;
  last_update: number;
}