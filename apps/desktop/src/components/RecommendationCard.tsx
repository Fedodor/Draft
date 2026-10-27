import { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { useEngineStore, useUIStore } from '../stores';
import { HeroPortrait } from './HeroPortrait';
import { motion, AnimatePresence } from 'framer-motion';
import type { Recommendation } from '@shared';
import { api, describeCommandError, type ScoreBreakdown } from '../services/api';
import { breakdownRows, breakdownTotal } from '../services/scoreBreakdown';

interface RecommendationCardProps {
  recommendation: Recommendation;
}

export function RecommendationCard({ recommendation }: RecommendationCardProps) {
  const { getHero } = useEngineStore();
  const selectedTeam = useUIStore((state) => state.selectedTeam);
  const hero = getHero(recommendation.hero_id);
  const heroName = hero?.localized_name ?? `Hero ${recommendation.hero_id}`;
  const heroIcon = hero?.icon ?? '';

  const [showBreakdown, setShowBreakdown] = useState(false);
  const [breakdown, setBreakdown] = useState<ScoreBreakdown | null>(null);
  const [breakdownLoading, setBreakdownLoading] = useState(false);
  const [breakdownError, setBreakdownError] = useState<string | null>(null);

  useEffect(() => {
    if (!showBreakdown) return;
    let cancelled = false;
    setBreakdownLoading(true);
    setBreakdownError(null);
    api
      .getScoreBreakdown(recommendation.hero_id, selectedTeam)
      .then((next) => {
        if (!cancelled) setBreakdown(next);
      })
      .catch((error) => {
        if (!cancelled) setBreakdownError(describeCommandError(error));
      })
      .finally(() => {
        if (!cancelled) setBreakdownLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [showBreakdown, recommendation.hero_id, selectedTeam]);

  const scorePercent = Math.round(recommendation.score * 100);
  const confidencePercent = Math.round(recommendation.confidence * 100);
  const breakdownRowData = breakdown ? breakdownRows(breakdown) : [];

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'bg-green-500';
    if (score >= 0.5) return 'bg-yellow-500';
    if (score >= 0.3) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getRiskColor = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low': return 'bg-green-500/20 text-green-400';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400';
      case 'high': return 'bg-red-500/20 text-red-400';
    }
  };

  return (
    <div className="recommendation-card group relative overflow-hidden">
      {/* Hero Image Background */}
      <div className="absolute inset-0 opacity-5 group-hover:opacity-10 transition-opacity duration-300">
        {heroIcon ? (
          <img
            src={`https://cdn.cloudflare.steamstatic.com${heroIcon}`}
            alt=""
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full bg-dota-border/20" />
        )}
      </div>

      <div className="relative p-3">
        {/* Header: Hero Name + Score */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <HeroPortrait heroId={recommendation.hero_id} size={40} />
            <div className="min-w-0">
              <h5 className="font-medium text-sm truncate text-dota-text">{heroName}</h5>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className={`tag ${getCategoryClass(recommendation.category)} text-xs`}>
                  {formatCategory(recommendation.category)}
                </span>
                <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${getRiskColor(recommendation.risk_level)}`}>
                  {recommendation.risk_level.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
          
          <div className="text-right flex-shrink-0">
            <div className="text-lg font-bold text-dota-text">{scorePercent}%</div>
            <div className="text-xs text-dota-textMuted">Confidence: {confidencePercent}%</div>
          </div>
        </div>

        {/* Score Bar */}
        <div className="score-bar mb-2">
          <motion.div
            className={`score-fill ${getScoreColor(recommendation.score)}`}
            initial={{ width: 0 }}
            animate={{ width: `${scorePercent}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>

        {/* Explanations */}
        <AnimatePresence mode="popLayout">
          {recommendation.explanations.length > 0 && (
            <motion.ul
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-1 max-h-24 overflow-y-auto"
            >
              {recommendation.explanations.slice(0, 3).map((explanation, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 + i * 0.05 }}
                  className="flex items-start gap-1.5 text-xs text-dota-textMuted"
                >
                  <svg className="w-3 h-3 mt-0.5 flex-shrink-0 text-dota-accent" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="truncate">{explanation}</span>
                </motion.li>
              ))}
            </motion.ul>
          )}
        </AnimatePresence>

        {/* Role Fit */}
        {recommendation.role_fit.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-dota-border/50">
            {recommendation.role_fit.slice(0, 3).map((role) => (
              <span
                key={role}
                className="px-1.5 py-0.5 text-xs bg-dota-border text-dota-textMuted rounded"
              >
                {role}
              </span>
            ))}
          </div>
        )}

        {/* Score breakdown (lazy, via get_score_breakdown) */}
        <button
          type="button"
          onClick={() => setShowBreakdown((open) => !open)}
          aria-expanded={showBreakdown}
          className="mt-2 flex w-full items-center justify-between rounded border border-dota-border/50 px-2 py-1 text-xs text-dota-textMuted transition-colors hover:bg-dota-border/40 hover:text-dota-text"
        >
          <span>Score breakdown</span>
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${showBreakdown ? 'rotate-180' : ''}`}
          />
        </button>

        {showBreakdown && (
          <div className="mt-2 space-y-1.5 rounded border border-dota-border/50 bg-dota-bg/40 p-2">
            {breakdownLoading && (
              <div className="text-xs text-dota-textMuted">Loading breakdown…</div>
            )}
            {breakdownError && (
              <div className="text-xs text-red-400">{breakdownError}</div>
            )}
            {breakdown && !breakdownLoading && (
              <>
                {breakdownRowData.map((row) => (
                  <div key={row.key}>
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-dota-textMuted">{row.label}</span>
                      <span className="font-mono text-dota-text">
                        {Math.round(row.value * 100)}% × {Math.round(row.weight * 100)}% ={' '}
                        {(row.contribution * 100).toFixed(1)} pts
                      </span>
                    </div>
                    <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-dota-border">
                      <div
                        className={`h-full rounded-full ${getScoreColor(row.value)}`}
                        style={{ width: `${row.value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
                <div className="flex items-center justify-between border-t border-dota-border/50 pt-1.5 text-[11px] font-medium">
                  <span className="text-dota-textMuted">Weighted total</span>
                  <span className="font-mono text-dota-text">
                    {Math.round(breakdownTotal(breakdownRowData) * 100)}%
                  </span>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function getCategoryClass(category: Recommendation['category']): string {
  switch (category) {
    case 'top': return 'tag-top';
    case 'safe': return 'tag-safe';
    case 'aggressive': return 'tag-aggressive';
    case 'comfort': return 'tag-comfort';
    case 'anti_meta': return 'tag-anti-meta';
    default: return 'tag-top';
  }
}

function formatCategory(category: Recommendation['category']): string {
  switch (category) {
    case 'top': return 'TOP';
    case 'safe': return 'SAFE';
    case 'aggressive': return 'AGGRO';
    case 'comfort': return 'COMFORT';
    case 'anti_meta': return 'ANTI-META';
    default: return 'TOP';
  }
}