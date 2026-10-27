import { invoke, type InvokeArgs } from '@tauri-apps/api/core';
import type {
  Hero,
  DraftState,
  RecommendationSet,
  AppConfig,
} from '@shared';

export interface GetRecommendationsParams {
  team: 'radiant' | 'dire';
  limit?: number;
}

export interface ManualPickParams {
  hero_id: number;
  team: 'radiant' | 'dire';
  player_id?: number;
  order?: number;
}

export interface ManualBanParams {
  hero_id: number;
  team: 'radiant' | 'dire';
  order?: number;
}

class ApiService {
  async getRecommendations(params: GetRecommendationsParams): Promise<RecommendationSet> {
    return invoke('get_recommendations', params as unknown as InvokeArgs);
  }

  async getScoreBreakdown(
    heroId: number,
    team: 'radiant' | 'dire',
  ): Promise<ScoreBreakdown> {
    return invoke('get_score_breakdown', { hero_id: heroId, team });
  }

  async getSearchRecommendations(
    team: 'radiant' | 'dire',
    limit?: number,
  ): Promise<SimulationResult[]> {
    return invoke('get_search_recommendations', { team, limit });
  }

  async getDraftState(): Promise<DraftState> {
    return invoke('get_draft_state');
  }

  async manualPick(params: ManualPickParams): Promise<void> {
    return invoke('manual_pick', params as unknown as InvokeArgs);
  }

  async manualBan(params: ManualBanParams): Promise<void> {
    return invoke('manual_ban', params as unknown as InvokeArgs);
  }

  async undoLastAction(): Promise<DraftAction | null> {
    return invoke('undo_last_action');
  }

  async resetDraft(): Promise<void> {
    return invoke('reset_draft');
  }

  async getHeroList(): Promise<Hero[]> {
    return invoke('get_hero_list');
  }

  async getHeroDetails(heroId: number): Promise<Hero | null> {
    return invoke('get_hero_details', { hero_id: heroId });
  }

  async setConfig(config: AppConfig): Promise<void> {
    return invoke('set_config', { config });
  }

  async getConfig(): Promise<AppConfig> {
    return invoke('get_config');
  }

  async startGsi(): Promise<void> {
    return invoke('start_gsi');
  }

  async stopGsi(): Promise<void> {
    return invoke('stop_gsi');
  }

  async restartGsi(): Promise<void> {
    return invoke('restart_gsi');
  }

  async getGsiStatus(): Promise<GsiStatus> {
    return invoke('get_gsi_status');
  }

  async getGsiConfig(): Promise<string> {
    return invoke('get_gsi_config');
  }

  async installGsiConfig(): Promise<string> {
    return invoke('install_gsi_config');
  }

  async predictDraftValue(team: 'radiant' | 'dire'): Promise<number> {
    return invoke('predict_draft_value', { team });
  }

  async loadMlModel(modelPath: string): Promise<void> {
    return invoke('load_ml_model', { model_path: modelPath });
  }

  async getMlStatus(): Promise<MlStatus> {
    return invoke('get_ml_status');
  }

  async getDataHealth(): Promise<DataHealth> {
    return invoke('get_data_health');
  }
}

export interface DataHealth {
  schema_version: number;
  hero_count: number;
  matchup_count: number;
  synergy_count: number;
  meta_hero_count: number;
  active_patch: string | null;
  wal_enabled: boolean;
  healthy: boolean;
  issues: string[];
}

export interface GsiStatus {
  running: boolean;
  port: number;
  endpoint: string;
  auth_required: boolean;
  last_error: string | null;
}

export interface MlStatus {
  loaded: boolean;
  feature_count: number;
  model_path: string | null;
}

export interface ScoreBreakdown {
  matchup: number;
  synergy: number;
  composition: number;
  meta: number;
  comfort: number;
  ml: number;
  total: number;
}

export interface SimulationResult {
  hero_id: number;
  expected_value: number;
  immediate_value: number;
  opponent_response_value: number;
  confidence: number;
  explanation: string[];
}

export type DraftAction =
  | { kind: 'pick'; pick: { hero_id: number; player_id: number; order: number; team: 'radiant' | 'dire' } }
  | { kind: 'ban'; ban: { hero_id: number; order: number; team: 'radiant' | 'dire' } };

export type CommandErrorKind =
  | 'invalid_team'
  | 'invalid_hero_id'
  | 'hero_unavailable'
  | 'team_full'
  | 'invalid_turn'
  | 'missing_model_file'
  | 'model_error'
  | 'internal';

export interface CommandError {
  kind: CommandErrorKind;
  team?: string;
  hero_id?: number;
  count?: number;
  message?: string;
  path?: string;
}

/** Human-readable message for a rejected `invoke` call. */
export function describeCommandError(error: unknown): string {
  const commandError = error as Partial<CommandError> | undefined;
  if (commandError && typeof commandError === 'object' && 'kind' in commandError) {
    switch (commandError.kind) {
      case 'invalid_team':
        return `Unknown team "${commandError.team ?? ''}"`;
      case 'invalid_hero_id':
        return `Unknown hero id ${commandError.hero_id ?? ''}`;
      case 'hero_unavailable':
        return `Hero ${commandError.hero_id ?? ''} is already picked or banned`;
      case 'team_full':
        return `${commandError.team ?? 'Team'} already has ${commandError.count ?? 5} picks`;
      case 'invalid_turn':
        return commandError.message ?? 'Invalid turn or order';
      case 'missing_model_file':
        return `Model file not found: ${commandError.path ?? ''}`;
      case 'model_error':
        return commandError.message ?? 'Model error';
      case 'internal':
        return commandError.message ?? 'Internal error';
      default:
        break;
    }
  }
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  return 'Unexpected error';
}

export const api = new ApiService();

// React Query-like hooks for data fetching
export function useApi() {
  return api;
}
