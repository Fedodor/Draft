import { describe, expect, it } from 'vitest';
import type { DataHealth } from './api';
import { describeDataHealth, SEED_COMMANDS } from './dataHealth';

function health(overrides: Partial<DataHealth> = {}): DataHealth {
  return {
    schema_version: 1,
    hero_count: 126,
    matchup_count: 500,
    synergy_count: 400,
    meta_hero_count: 126,
    active_patch: '7.41',
    wal_enabled: true,
    healthy: true,
    issues: [],
    ...overrides,
  };
}

describe('describeDataHealth', () => {
  it('reports a healthy database with its patch', () => {
    const summary = describeDataHealth(health());

    expect(summary.level).toBe('ok');
    expect(summary.detail).toContain('126 heroes');
    expect(summary.detail).toContain('7.41');
    expect(summary.command).toBeUndefined();
  });

  it('explains an unseeded database and how to seed it', () => {
    const summary = describeDataHealth(
      health({ hero_count: 0, matchup_count: 0, synergy_count: 0, meta_hero_count: 0, active_patch: null, healthy: false, issues: ['no heroes seeded'] }),
    );

    expect(summary.level).toBe('empty');
    expect(summary.title).toBe('Database is not seeded');
    expect(summary.command).toBe(SEED_COMMANDS);
    expect(summary.command).toContain('seed_heroes.exe');
  });

  it('lists missing signals without claiming the app is broken', () => {
    const summary = describeDataHealth(
      health({ matchup_count: 0, healthy: false, issues: ['no matchups seeded'] }),
    );

    expect(summary.level).toBe('warning');
    expect(summary.detail).toContain('no matchups seeded');
    expect(summary.command).toBe(SEED_COMMANDS);
  });

  it('names seed_matchups when the active patch has no ban data', () => {
    const summary = describeDataHealth(
      health({
        healthy: false,
        issues: ['no ban data seeded for patch 7.41: run seed_matchups to populate hero_ban_stats'],
      }),
    );

    expect(summary.level).toBe('warning');
    expect(summary.detail).toContain('seed_matchups');
    expect(summary.command).toBe(SEED_COMMANDS);
    expect(summary.command).toContain('seed_matchups.exe');
  });

  it('omits the seed command when seeding would not fix the issue', () => {
    const summary = describeDataHealth(
      health({ wal_enabled: false, healthy: false, issues: ['SQLite is not in WAL mode'] }),
    );

    expect(summary.level).toBe('warning');
    expect(summary.command).toBeUndefined();
  });

  it('warns when the health call itself failed', () => {
    const summary = describeDataHealth(null);

    expect(summary.level).toBe('warning');
    expect(summary.title).toBe('Database health unavailable');
  });
});
