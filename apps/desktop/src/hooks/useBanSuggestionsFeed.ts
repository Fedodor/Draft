import { useEffect } from 'react';
import type { DraftState } from '@shared';
import { api } from '../services/api';
import { isBanPhase } from '../services/banPhase';
import { useEngineStore } from '../stores';

/**
 * Fetch ban suggestions whenever `draft` is in the ban phase, storing them in
 * the engine store. Each window runs its own store, so both windows call this.
 *
 * Returns whether the draft is currently banning, so callers can switch the
 * list they render with the same rule that triggers the fetch.
 */
export function useBanSuggestionsFeed(
  draft: DraftState | null,
  limit = 10,
): boolean {
  const setBanSuggestions = useEngineStore((state) => state.setBanSuggestions);
  const banning = draft !== null && isBanPhase(draft);
  const banCount = draft?.bans.length ?? 0;
  const patch = draft?.patch_version ?? '';

  useEffect(() => {
    if (!banning) return;
    let cancelled = false;
    api
      .getBanSuggestions(limit)
      .then((suggestions) => {
        if (!cancelled) setBanSuggestions(suggestions);
      })
      .catch(() => {
        if (!cancelled) setBanSuggestions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [banning, banCount, patch, limit, setBanSuggestions]);

  return banning;
}
