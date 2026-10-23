import { beforeEach, describe, expect, it, vi } from 'vitest';
import { invoke } from '@tauri-apps/api/core';
import { api } from './api';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

const mockedInvoke = vi.mocked(invoke);

describe('manual draft IPC smoke test', () => {
  beforeEach(() => {
    mockedInvoke.mockReset();
    mockedInvoke.mockResolvedValue(undefined);
  });

  it('sends manual_pick with snake_case arguments', async () => {
    await api.manualPick({ hero_id: 1, team: 'radiant', player_id: 42, order: 2 });
    expect(mockedInvoke).toHaveBeenCalledWith('manual_pick', {
      hero_id: 1,
      team: 'radiant',
      player_id: 42,
      order: 2,
    });
  });

  it('sends manual_ban with snake_case arguments', async () => {
    await api.manualBan({ hero_id: 5, team: 'dire', order: 3 });
    expect(mockedInvoke).toHaveBeenCalledWith('manual_ban', {
      hero_id: 5,
      team: 'dire',
      order: 3,
    });
  });

  it('sends reset_draft', async () => {
    await api.resetDraft();
    expect(mockedInvoke).toHaveBeenCalledWith('reset_draft');
  });

  it('sends undo_last_action and returns the action', async () => {
    mockedInvoke.mockResolvedValueOnce({ kind: 'pick', pick: { hero_id: 1, player_id: 0, order: 1, team: 'radiant' } });
    const action = await api.undoLastAction();
    expect(mockedInvoke).toHaveBeenCalledWith('undo_last_action');
    expect(action).toMatchObject({ kind: 'pick' });
  });

  it('sends get_score_breakdown with snake_case hero_id', async () => {
    await api.getScoreBreakdown(7, 'dire');
    expect(mockedInvoke).toHaveBeenCalledWith('get_score_breakdown', {
      hero_id: 7,
      team: 'dire',
    });
  });

  it('sends get_hero_details with snake_case hero_id', async () => {
    await api.getHeroDetails(9);
    expect(mockedInvoke).toHaveBeenCalledWith('get_hero_details', { hero_id: 9 });
  });

  it('sends load_ml_model with snake_case model_path', async () => {
    await api.loadMlModel('models/draft.onnx');
    expect(mockedInvoke).toHaveBeenCalledWith('load_ml_model', {
      model_path: 'models/draft.onnx',
    });
  });

  it('sends restart_gsi', async () => {
    await api.restartGsi();
    expect(mockedInvoke).toHaveBeenCalledWith('restart_gsi');
  });

  it('sends get_config and returns the stored config', async () => {
    mockedInvoke.mockResolvedValueOnce({
      gsi_port: 4123,
      gsi_auto_start: false,
      gsi_auth_token: 'tok',
      overlay_enabled: true,
      overlay_position: { x: 1, y: 2 },
      overlay_size: { width: 10, height: 20 },
      ml_enabled: true,
      ml_model_path: 'm.onnx',
      auto_update_meta: true,
      preferred_language: 'en',
    });
    const config = await api.getConfig();
    expect(mockedInvoke).toHaveBeenCalledWith('get_config');
    expect(config).toMatchObject({ gsi_port: 4123, gsi_auto_start: false });
  });

  it('sends set_config with the config object', async () => {
    const config = {
      gsi_port: 5000,
      gsi_auto_start: true,
      gsi_auth_token: null,
      overlay_enabled: true,
      overlay_position: { x: 1, y: 2 },
      overlay_size: { width: 10, height: 20 },
      ml_enabled: true,
      ml_model_path: 'm.onnx',
      auto_update_meta: true,
      preferred_language: 'en',
    };
    await api.setConfig(config);
    expect(mockedInvoke).toHaveBeenCalledWith('set_config', { config });
  });

  it('sends get_data_health and returns the report', async () => {
    mockedInvoke.mockResolvedValueOnce({
      schema_version: 2,
      hero_count: 126,
      matchup_count: 9052,
      synergy_count: 7381,
      active_patch: '7.41',
      wal_enabled: true,
      healthy: true,
      issues: [],
    });
    const health = await api.getDataHealth();
    expect(mockedInvoke).toHaveBeenCalledWith('get_data_health');
    expect(health).toMatchObject({ healthy: true, active_patch: '7.41' });
  });
});
