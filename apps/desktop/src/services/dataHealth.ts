import type { DataHealth } from './api';

export type DataHealthLevel = 'ok' | 'warning' | 'empty';

export interface DataHealthSummary {
  level: DataHealthLevel;
  title: string;
  detail: string;
  /** Shell commands that fix the problem, when the fix is known. */
  command?: string;
}

export const SEED_COMMANDS = [
  'cargo build --release --manifest-path infrastructure/seeders/Cargo.toml',
  'infrastructure/seeders/target/release/seed_heroes.exe',
  'infrastructure/seeders/target/release/seed_matchups.exe',
].join('\n');

/**
 * Turn the backend health snapshot into the first-run state the UI shows.
 * `null` means the health call itself failed (no database handle yet).
 */
export function describeDataHealth(health: DataHealth | null): DataHealthSummary {
  if (!health) {
    return {
      level: 'warning',
      title: 'Database health unavailable',
      detail: 'The backend did not report database health, so the draft and recommendations may stay empty.',
    };
  }

  if (health.hero_count === 0) {
    return {
      level: 'empty',
      title: 'Database is not seeded',
      detail:
        'No heroes are loaded yet. Until the seeders run, the draft stays empty and no pick can be scored. ' +
        'Run these from the repository root, then press Recheck:',
      command: SEED_COMMANDS,
    };
  }

  if (!health.healthy) {
    const needsSeeding = health.issues.some((issue) => /seeded|active patch/i.test(issue));
    return {
      level: 'warning',
      title: 'Seeded data is incomplete',
      detail: `Recommendations still work, but some signals are missing: ${health.issues.join('; ')}.`,
      command: needsSeeding ? SEED_COMMANDS : undefined,
    };
  }

  return {
    level: 'ok',
    title: 'Data ready',
    detail: `${health.hero_count} heroes seeded, patch ${health.active_patch ?? 'unset'}.`,
  };
}
