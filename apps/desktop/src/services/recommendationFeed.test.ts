import { describe, expect, it, vi } from 'vitest';
import type { DraftState, RecommendationSet } from '@shared';
import { RecommendationFeed } from './recommendationFeed';

function draft(order: number): DraftState {
  return {
    radiant_picks: [{ hero_id: order, player_id: 0, order, team: 'radiant' }],
    dire_picks: [],
    bans: [],
    current_pick: 'radiant',
    pick_number: order,
    is_picking_phase: true,
    available_heroes: [],
    patch_version: '7.41',
    timestamp: order,
  };
}

function recommendations(heroId: number): RecommendationSet {
  return {
    top_picks: [
      {
        hero_id: heroId,
        score: 0.6,
        confidence: 0.6,
        explanations: [],
        risk_level: 'medium',
        role_fit: [],
        category: 'top',
      },
    ],
    safe_picks: [],
    aggressive_picks: [],
    comfort_picks: [],
    anti_meta_picks: [],
    team_analysis: {
      missing: [],
      strong: [],
      composition_score: 0.5,
      lane_balance: { safe_lane: 0.5, mid_lane: 0.5, offlane: 0.5 },
      tempo: 'mid',
    },
    timestamp: 0,
  };
}

describe('RecommendationFeed', () => {
  it('applies draft payloads immediately but debounces recommendation work', async () => {
    vi.useFakeTimers();
    try {
      const appliedDrafts: DraftState[] = [];
      const appliedRecommendations: RecommendationSet[] = [];
      let fetchCount = 0;

      const feed = new RecommendationFeed({
        getTeam: () => 'radiant',
        limit: 10,
        debounceMs: 200,
        fetchDraft: async () => draft(1),
        fetchRecommendations: async () => {
          fetchCount += 1;
          return recommendations(7);
        },
        onDraft: (value) => appliedDrafts.push(value),
        onRecommendations: (value) => appliedRecommendations.push(value),
      });

      feed.handleDraftEvent(draft(1));
      feed.handleDraftEvent(draft(2));
      feed.handleDraftEvent(draft(3));

      expect(appliedDrafts.map((d) => d.pick_number)).toEqual([1, 2, 3]);
      expect(fetchCount).toBe(0);

      await vi.advanceTimersByTimeAsync(199);
      expect(fetchCount).toBe(0);

      await vi.advanceTimersByTimeAsync(1);
      expect(fetchCount).toBe(1);
      expect(appliedRecommendations).toHaveLength(1);

      feed.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores an out-of-order response that completes after a newer request', async () => {
    const pending: Array<(value: RecommendationSet) => void> = [];
    const applied: number[] = [];
    const feed = new RecommendationFeed({
      getTeam: () => 'radiant',
      limit: 10,
      fetchDraft: async () => draft(1),
      fetchRecommendations: () =>
        new Promise<RecommendationSet>((resolve) => {
          pending.push(resolve);
        }),
      onRecommendations: (value) => applied.push(value.top_picks[0].hero_id),
    });

    const first = feed.refresh();
    const second = feed.refresh();
    expect(pending).toHaveLength(2);

    // Newer request finishes first and becomes authoritative.
    pending[1](recommendations(2));
    await second;
    expect(applied).toEqual([2]);

    // Older request finishes afterwards and must be discarded.
    pending[0](recommendations(1));
    await first;
    expect(applied).toEqual([2]);

    feed.dispose();
  });

  it('keeps the last good recommendations when a request fails', async () => {
    let call = 0;
    const applied: number[] = [];
    const errors: unknown[] = [];
    const feed = new RecommendationFeed({
      getTeam: () => 'radiant',
      limit: 10,
      fetchDraft: async () => draft(1),
      fetchRecommendations: async () => {
        call += 1;
        if (call === 1) return recommendations(1);
        throw new Error('model unavailable');
      },
      onRecommendations: (value) => applied.push(value.top_picks[0].hero_id),
      onError: (error) => errors.push(error),
    });

    await feed.refresh();
    expect(applied).toEqual([1]);

    await feed.refresh();
    expect(applied).toEqual([1]);
    expect(errors).toHaveLength(1);

    feed.dispose();
  });

  it('start() applies the draft before recommendations resolve', async () => {
    const events: string[] = [];
    const feed = new RecommendationFeed({
      getTeam: () => 'radiant',
      limit: 3,
      fetchDraft: async () => draft(4),
      fetchRecommendations: async () => {
        events.push('recommendations');
        return recommendations(9);
      },
      onDraft: () => events.push('draft'),
      onRecommendations: () => events.push('applied'),
    });

    await feed.start();
    expect(events).toEqual(['draft', 'recommendations', 'applied']);

    feed.dispose();
  });

  it('start() does not let a fetched snapshot overwrite a newer GSI payload', async () => {
    let resolveDraft: ((value: DraftState) => void) | undefined;
    const appliedDrafts: number[] = [];
    const appliedRecommendations: number[] = [];
    const feed = new RecommendationFeed({
      getTeam: () => 'radiant',
      limit: 3,
      fetchDraft: () =>
        new Promise<DraftState>((resolve) => {
          resolveDraft = resolve;
        }),
      fetchRecommendations: async () => recommendations(3),
      onDraft: (value) => appliedDrafts.push(value.pick_number),
      onRecommendations: (value) => appliedRecommendations.push(value.top_picks[0].hero_id),
    });

    const starting = feed.start();
    // A live payload lands while the initial draft read is still pending.
    feed.handleDraftEvent(draft(7));
    expect(appliedDrafts).toEqual([7]);

    resolveDraft?.(draft(1));
    await starting;

    expect(appliedDrafts).toEqual([7]);
    expect(appliedRecommendations).toEqual([]);

    feed.dispose();
  });
});
