import type { ScoreBreakdown } from './api';

/**
 * The recommendation weights. These mirror the `W_*` constants in
 * `apps/desktop/src-tauri/src/engine/scorer.rs`; keep them in sync.
 */
export const SCORE_COMPONENTS = [
  { key: 'matchup', label: 'Matchup', weight: 0.25 },
  { key: 'synergy', label: 'Synergy', weight: 0.2 },
  { key: 'composition', label: 'Composition', weight: 0.15 },
  { key: 'meta', label: 'Meta', weight: 0.1 },
  { key: 'comfort', label: 'Comfort', weight: 0.05 },
  { key: 'ml', label: 'ML', weight: 0.25 },
] as const;

export type ScoreComponentKey = (typeof SCORE_COMPONENTS)[number]['key'];

export interface ScoreRow {
  key: ScoreComponentKey;
  label: string;
  weight: number;
  value: number;
  contribution: number;
}

/** Expand a breakdown into display rows, including each weighted contribution. */
export function breakdownRows(breakdown: ScoreBreakdown): ScoreRow[] {
  return SCORE_COMPONENTS.map((component) => ({
    ...component,
    value: breakdown[component.key],
    contribution: breakdown[component.key] * component.weight,
  }));
}

/** Sum the weighted contributions; this should equal `breakdown.total`. */
export function breakdownTotal(rows: ScoreRow[]): number {
  return rows.reduce((sum, row) => sum + row.contribution, 0);
}
