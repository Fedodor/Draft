import { describe, expect, it } from 'vitest';
import { formatBanRate, isBanPhase } from './banPhase';

const ban = { hero_id: 1, order: 0, team: 'radiant' as const };

describe('isBanPhase', () => {
  it('is false before the draft starts', () => {
    expect(
      isBanPhase({ is_picking_phase: false, current_pick: null, bans: [] }),
    ).toBe(false);
  });

  it('is true on a ban turn, before any ban lands', () => {
    expect(
      isBanPhase({ is_picking_phase: false, current_pick: 'radiant', bans: [] }),
    ).toBe(true);
  });

  it('is true for hand-recorded bans without a live turn', () => {
    expect(
      isBanPhase({ is_picking_phase: false, current_pick: null, bans: [ban] }),
    ).toBe(true);
  });

  it('is false once the draft moves to picks', () => {
    expect(
      isBanPhase({ is_picking_phase: true, current_pick: 'radiant', bans: [ban] }),
    ).toBe(false);
  });
});

describe('formatBanRate', () => {
  it('renders a rounded percentage of the counted drafts', () => {
    expect(formatBanRate({ ban_rate: 0.4124, games: 4175 })).toBe(
      '41% of 4175 drafts',
    );
  });

  it('handles zero-ban samples without dividing or inventing a range', () => {
    expect(formatBanRate({ ban_rate: 0, games: 10 })).toBe('0% of 10 drafts');
  });
});
