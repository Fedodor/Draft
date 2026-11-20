import type { BanSuggestion, DraftState } from '@shared';

/**
 * Bans are in progress when a draft is underway but picks are not.
 *
 * `DraftState` has no explicit banning flag, so a recorded ban is the only
 * evidence that a draft started. Before the first ban the panel keeps showing
 * picks rather than guessing at a phase.
 */
export function isBanPhase(
  draft: Pick<DraftState, 'is_picking_phase' | 'bans'>,
): boolean {
  return !draft.is_picking_phase && draft.bans.length > 0;
}

/** Compact, explicit copy: prevalence over a counted sample, not a verdict. */
export function formatBanRate(
  suggestion: Pick<BanSuggestion, 'ban_rate' | 'games'>,
): string {
  return `${Math.round(suggestion.ban_rate * 100)}% of ${suggestion.games} drafts`;
}
