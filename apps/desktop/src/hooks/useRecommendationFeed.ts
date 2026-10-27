import { useCallback, useEffect, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { DraftState, RecommendationSet, TeamSide } from '@shared';
import { api } from '../services/api';
import {
  DEFAULT_DEBOUNCE_MS,
  POLL_STALE_MS,
  RecommendationFeed,
} from '../services/recommendationFeed';

interface UseRecommendationFeedOptions {
  /** Read the current team on every request (use a ref for live values). */
  getTeam: () => TeamSide;
  limit: number;
  onDraft?: (draft: DraftState) => void;
  onRecommendations: (recommendations: RecommendationSet) => void;
  onError?: (error: unknown) => void;
  /** Optional safety poll; skipped while GSI updates are flowing. */
  pollMs?: number | null;
}

/**
 * One recommendation orchestration layer per window.
 *
 * It owns the `gsi-update` subscription, applies the draft payload immediately,
 * debounces the expensive recommendation request, and rejects out-of-order
 * responses. Consumers only provide state setters.
 */
export function useRecommendationFeed(options: UseRecommendationFeedOptions) {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const feedRef = useRef<RecommendationFeed | null>(null);

  const { limit, pollMs } = options;

  useEffect(() => {
    const feed = new RecommendationFeed({
      getTeam: () => optionsRef.current.getTeam(),
      limit,
      debounceMs: DEFAULT_DEBOUNCE_MS,
      fetchDraft: () => api.getDraftState(),
      fetchRecommendations: (team, requestLimit) =>
        api.getRecommendations({ team, limit: requestLimit }),
      onDraft: (draft) => optionsRef.current.onDraft?.(draft),
      onRecommendations: (recommendations) =>
        optionsRef.current.onRecommendations(recommendations),
      onError: (error) => optionsRef.current.onError?.(error),
    });

    feedRef.current = feed;
    const unlisten = listen<DraftState>('gsi-update', (event) => {
      feed.handleDraftEvent(event.payload);
    });
    void feed.start();

    let interval: number | undefined;
    if (pollMs) {
      interval = window.setInterval(() => {
        // GSI payloads are authoritative; only poll when the feed has gone
        // quiet (for example the listener is down but the app is still open).
        if (feed.isStale(POLL_STALE_MS)) {
          void feed.refresh();
        }
      }, pollMs);
    }

    return () => {
      feed.dispose();
      feedRef.current = null;
      unlisten.then((fn) => fn());
      if (interval !== undefined) window.clearInterval(interval);
    };
  }, [limit, pollMs]);

  return {
    /** Recompute for the current team (team change, manual refresh). */
    refresh: useCallback(() => feedRef.current?.refresh(), []),
  };
}
