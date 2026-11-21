import type { BanSuggestion, DraftState } from '@shared';

/**
 * Bans are in progress when GSI is driving a draft and picks are not.
 *
 * GSI keeps `current_pick` set on a ban turn (`pick: false`), so this is true
 * before the first ban lands. Recorded bans also count, which covers a ban
 * turn whose `activeteam`/`pick` fields were missing.
 *
 * All Pick sends no draft data at all: its pre-pick bans are recorded by hand,
 * and they are just unavailable heroes, not an ongoing ban phase. Without
 * `has_live_draft` the panel would hide the picks it exists to suggest.
 */
export function isBanPhase(
  draft: Pick<
    DraftState,
    'has_live_draft' | 'is_picking_phase' | 'current_pick' | 'bans'
  >,
): boolean {
  if (!draft.has_live_draft || draft.is_picking_phase) return false;
  return draft.current_pick !== null || draft.bans.length > 0;
}

/** Compact, explicit copy: prevalence over a counted sample, not a verdict. */
export function formatBanRate(
  suggestion: Pick<BanSuggestion, 'ban_rate' | 'games'>,
): string {
  return `${Math.round(suggestion.ban_rate * 100)}% of ${suggestion.games} drafts`;
}
