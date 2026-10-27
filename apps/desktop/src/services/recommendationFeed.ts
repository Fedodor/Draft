import type { DraftState, RecommendationSet, TeamSide } from '@shared';

export interface RecommendationFeedOptions {
  /** Team to recommend for; read on every request so it stays current. */
  getTeam: () => TeamSide;
  limit: number;
  /** Trailing debounce for bursty GSI updates. */
  debounceMs?: number;
  fetchDraft: () => Promise<DraftState>;
  fetchRecommendations: (team: TeamSide, limit: number) => Promise<RecommendationSet>;
  onDraft?: (draft: DraftState) => void;
  onRecommendations: (recommendations: RecommendationSet) => void;
  onError?: (error: unknown) => void;
}

/**
 * Coalesces recommendation work for one window.
 *
 * - GSI payloads already contain the draft, so the draft is applied
 *   immediately and only the recommendation request is debounced.
 * - Requests are numbered; a response is applied only if no newer request has
 *   already been applied, so out-of-order completion cannot overwrite state.
 * - A failed request leaves the last successful recommendations in place.
 */
export class RecommendationFeed {
  private readonly options: RecommendationFeedOptions;
  private readonly debounceMs: number;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight = 0;
  private applied = 0;
  private disposed = false;
  private lastEventAt = 0;

  constructor(options: RecommendationFeedOptions) {
    this.options = options;
    this.debounceMs = options.debounceMs ?? 200;
  }

  /** Apply a GSI draft payload and schedule a coalesced refresh. */
  handleDraftEvent(draft: DraftState): void {
    if (this.disposed) return;
    this.lastEventAt = Date.now();
    this.options.onDraft?.(draft);
    this.schedule();
  }

  /** Schedule a refresh after the debounce window. */
  schedule(): void {
    if (this.disposed) return;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.runRecommendations();
    }, this.debounceMs);
  }

  /** Request recommendations for the current team without waiting. */
  async refresh(): Promise<void> {
    if (this.disposed) return;
    this.cancelScheduled();
    await this.runRecommendations();
  }

  /** Full refresh: re-read the draft and then recommend. */
  async start(): Promise<void> {
    if (this.disposed) return;
    const requestId = ++this.inFlight;
    const observedEvents = this.lastEventAt;
    try {
      const draft = await this.options.fetchDraft();
      // If a GSI payload arrived while the initial read was in flight, that
      // payload is newer than the fetched snapshot. Applying the snapshot
      // would visibly regress the draft, so let the event's scheduled refresh
      // take over instead.
      if (this.isRequestStale(requestId) || this.lastEventAt !== observedEvents) {
        return;
      }
      this.options.onDraft?.(draft);
      await this.runRecommendations(requestId);
    } catch (error) {
      this.reportError(error);
    }
  }

  /** True when no GSI payload has arrived for `staleMs`. */
  isStale(staleMs: number): boolean {
    return this.lastEventAt === 0 || Date.now() - this.lastEventAt > staleMs;
  }

  /** Stop accepting work and clear pending timers (React unmount). */
  dispose(): void {
    this.disposed = true;
    this.cancelScheduled();
  }

  private async runRecommendations(existingRequestId?: number): Promise<void> {
    if (this.disposed) return;
    const requestId = existingRequestId ?? ++this.inFlight;
    try {
      const team = this.options.getTeam();
      const recommendations = await this.options.fetchRecommendations(
        team,
        this.options.limit,
      );
      if (this.isRequestStale(requestId)) return;
      this.applied = requestId;
      this.options.onRecommendations(recommendations);
    } catch (error) {
      if (!this.disposed && requestId >= this.applied) {
        this.reportError(error);
      }
    }
  }

  private isRequestStale(requestId: number): boolean {
    return this.disposed || requestId < this.applied;
  }

  private cancelScheduled(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private reportError(error: unknown): void {
    if (this.disposed) return;
    this.options.onError?.(error);
  }
}

export const DEFAULT_DEBOUNCE_MS = 200;
export const POLL_STALE_MS = 10_000;
