import { describe, expect, it } from 'vitest';
import type { ScoreBreakdown } from './api';
import {
  SCORE_COMPONENTS,
  breakdownRows,
  breakdownTotal,
} from './scoreBreakdown';

const breakdown: ScoreBreakdown = {
  matchup: 0.8,
  synergy: 0.5,
  composition: 0.6,
  meta: 0.4,
  comfort: 0.2,
  ml: 0.7,
  total: 0.615,
};

describe('score breakdown rows', () => {
  it('uses weights that sum to one', () => {
    const totalWeight = SCORE_COMPONENTS.reduce(
      (sum, component) => sum + component.weight,
      0,
    );
    expect(totalWeight).toBeCloseTo(1, 10);
  });

  it('keeps every component and computes weighted contributions', () => {
    const rows = breakdownRows(breakdown);

    expect(rows.map((row) => row.key)).toEqual([
      'matchup',
      'synergy',
      'composition',
      'meta',
      'comfort',
      'ml',
    ]);
    expect(rows.find((row) => row.key === 'matchup')).toMatchObject({
      label: 'Matchup',
      value: 0.8,
      weight: 0.25,
      contribution: 0.2,
    });
    expect(rows.find((row) => row.key === 'ml')?.contribution).toBeCloseTo(
      0.175,
      10,
    );
  });

  it('reconstructs the backend total from the contributions', () => {
    expect(breakdownTotal(breakdownRows(breakdown))).toBeCloseTo(
      breakdown.total,
      6,
    );
  });
});
