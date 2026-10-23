import { useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { X, Crosshair, Shield, Sword, Sparkles } from 'lucide-react';
import type { DraftState, RecommendationSet } from '@shared';
import { api } from '../services/api';
import { useEngineStore } from '../stores';
import { HeroPortrait } from './HeroPortrait';

interface WinProb {
  radiant: number;
  dire: number;
}

const appWindow = getCurrentWindow();

export function OverlayWindow() {
  const { heroes, setHeroes, getHero } = useEngineStore();
  const [draft, setDraft] = useState<DraftState | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationSet | null>(null);
  const [winProb, setWinProb] = useState<WinProb | null>(null);
  const [gsiLive, setGsiLive] = useState(false);
  const selectedRef = useRef<'radiant' | 'dire'>('radiant');

  // Load hero roster once
  useEffect(() => {
    if (heroes.length === 0) {
      api.getHeroList().then(setHeroes).catch(() => {});
    }
  }, [heroes.length, setHeroes]);

  const refresh = async (forceTeam?: 'radiant' | 'dire') => {
    try {
      const [state, status] = await Promise.all([
        api.getDraftState(),
        api.getGsiStatus(),
      ]);
      setDraft(state);
      setGsiLive(status.running);

      // Recommend for the team whose turn it is (or keep the last selection)
      const turn = state.current_pick ?? selectedRef.current;
      if (forceTeam) {
        selectedRef.current = forceTeam;
      } else {
        selectedRef.current = turn;
      }

      const [recs, radiantProb] = await Promise.all([
        api.getRecommendations({ team: selectedRef.current, limit: 3 }),
        api.predictDraftValue('radiant'),
      ]);
      setRecommendations(recs);
      setWinProb({ radiant: radiantProb, dire: 1 - radiantProb });
    } catch {
      // Backend not ready yet
    }
  };

  // Listen for live GSI updates and poll on mount
  useEffect(() => {
    const unlisten = listen('gsi-update', () => {
      void refresh();
    });

    void refresh();

    const interval = window.setInterval(() => {
      void refresh();
    }, 5000);

    return () => {
      unlisten.then((fn) => fn());
      window.clearInterval(interval);
    };
  }, []);

  const radiantProb = winProb ? winProb.radiant * 100 : 50;
  const radiantPicks = draft?.radiant_picks ?? [];
  const direPicks = draft?.dire_picks ?? [];
  const topPicks = recommendations?.top_picks ?? [];

  return (
    <div className="w-screen h-screen bg-transparent select-none">
      <div className="flex flex-col h-full m-1.5 rounded-2xl border border-dota-border/70 bg-dota-panel/90 backdrop-blur-md shadow-2xl overflow-hidden">
        {/* Drag header */}
        <div
          data-tauri-drag-region
          className="flex items-center justify-between px-3 py-2 border-b border-dota-border bg-dota-bg/60 cursor-grab active:cursor-grabbing"
        >
          <div className="flex items-center gap-2" data-tauri-drag-region>
            <Crosshair className="w-4 h-4 text-dota-accent" />
            <span className="text-xs font-semibold uppercase tracking-wider text-dota-text">
              Draft Overlay
            </span>
            <span className={`w-2 h-2 rounded-full ${gsiLive ? 'bg-green-500 animate-pulse' : 'bg-red-500/70'}`} />
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => appWindow.hide()}
              className="p-1 rounded hover:bg-dota-border text-dota-textMuted hover:text-dota-text"
              title="Hide overlay"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Win probability */}
        <div className="px-3 pt-3">
          <div className="flex items-center justify-between text-xs font-semibold mb-1">
            <span className="flex items-center gap-1 text-dota-text">
              <Shield className="w-3.5 h-3.5 text-green-400" /> Radiant
            </span>
            <span className="font-mono text-dota-text">
              {radiantProb.toFixed(1)}%
            </span>
            <span className="flex items-center gap-1 text-dota-text">
              Dire <Sword className="w-3.5 h-3.5 text-red-400" />
            </span>
          </div>
          <div className="h-2 bg-dota-border rounded-full overflow-hidden flex">
            <div
              className="h-full bg-green-500/80 transition-all duration-500"
              style={{ width: `${radiantProb}%` }}
            />
            <div className="h-full flex-1 bg-red-500/80 transition-all duration-500" />
          </div>
        </div>

        {/* Teams */}
        <div className="flex flex-1 gap-2 px-3 py-2 min-h-0">
          <TeamColumn title="Radiant" picks={radiantPicks} accent="text-green-400" />
          <TeamColumn title="Dire" picks={direPicks} accent="text-red-400" />
        </div>

        {/* Top recommendations */}
        <div className="px-3 pb-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Sparkles className="w-3.5 h-3.5 text-dota-accent" />
            <span className="text-xs font-semibold uppercase tracking-wider text-dota-textMuted">
              Top Picks
            </span>
          </div>
          {topPicks.length === 0 ? (
            <div className="text-xs text-dota-textMuted text-center py-2">
              Waiting for draft…
            </div>
          ) : (
            <div className="space-y-1">
              {topPicks.map((rec, i) => {
                const hero = getHero(rec.hero_id);
                const explanation = rec.explanations[0] ?? '';
                return (
                  <div
                    key={rec.hero_id}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-dota-bg/60 border border-dota-border/50"
                  >
                    <span className="text-xs font-bold text-dota-accent w-4">{i + 1}</span>
                    <HeroPortrait heroId={rec.hero_id} size={28} />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-dota-text truncate">
                        {hero?.localized_name ?? `Hero ${rec.hero_id}`}
                      </div>
                      {explanation && (
                        <div className="text-[10px] text-dota-textMuted truncate">{explanation}</div>
                      )}
                    </div>
                    <span className="text-xs font-mono text-dota-text">
                      {Math.round(rec.score * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TeamColumn({
  title,
  picks,
  accent,
}: {
  title: string;
  picks: Array<{ hero_id: number; order: number }>;
  accent: string;
}) {
  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-dota-textMuted mb-1">
        {title} <span className={accent}>{picks.length}/5</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {picks.map((pick) => (
          <div key={pick.hero_id} className="relative">
            <HeroPortrait heroId={pick.hero_id} size={40} />
          </div>
        ))}
        {Array.from({ length: Math.max(0, 5 - picks.length) }).map((_, i) => (
          <div
            key={`empty-${i}`}
            className="w-10 h-10 rounded-lg border border-dashed border-dota-border/40 bg-dota-bg/30"
          />
        ))}
      </div>
    </div>
  );
}
