import type { BanSuggestion, DraftState } from '@shared';

/**
 * Bans are in progress when a draft is underway but picks are not.
 *
 * GSI keeps `current_pick` set on a ban turn (`pick: false`), so this is true
 * before the first ban lands. Recorded bans also count, which covers bans
 * entered by hand without live GSI.
 */
export function isBanPhase(
  draft: Pick<DraftState, 'is_picking_phase' | 'current_pick' | 'bans'>,
): boolean {
  if (draft.is_picking_phase) return false;
  return draft.current_pick !== null || draft.bans.length > 0;
}

/** Compact, explicit copy: prevalence over a counted sample, not a verdict. */
export function formatBanRate(
  suggestion: Pick<BanSuggestion, 'ban_rate' | 'games'>,
): string {
  return `${Math.round(suggestion.ban_rate * 100)}% of ${suggestion.games} drafts`;
}
