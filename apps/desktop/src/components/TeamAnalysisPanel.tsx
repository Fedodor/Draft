import { useTeamAnalysis, useEngineStore } from '../stores';
import { 
  Shield, Sword, Zap, 
  ArrowUpRight, 
  Timer, Target
} from 'lucide-react';

export function TeamAnalysisPanel() {
  const teamAnalysis = useTeamAnalysis();
  const { draft } = useEngineStore();

  if (!teamAnalysis) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-dota-textMuted">
          <Target className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Start a draft to see team analysis</p>
        </div>
      </div>
    );
  }

  const { missing, strong, composition_score, tempo } = teamAnalysis;

  const tempoIcons = {
    early: <Zap className="w-5 h-5 text-yellow-400" />,
    mid: <Sword className="w-5 h-5 text-orange-400" />,
    late: <ArrowUpRight className="w-5 h-5 text-green-400" />,
  };

  const tempoLabels = {
    early: 'Early Game',
    mid: 'Mid Game',
    late: 'Late Game',
  };

  return (
    <div className="flex-1 overflow-y-auto p-3 space-y-4">
      {/* Composition Score */}
      <div className="dota-panel p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-dota-text">Team Composition</h3>
          <div className="text-2xl font-bold text-dota-accent">
            {Math.round(composition_score * 100)}%
          </div>
        </div>
        <div className="h-2 bg-dota-border rounded-full overflow-hidden">
          <div 
            className="h-full bg-dota-accent rounded-full transition-all duration-500"
            style={{ width: `${composition_score * 100}%` }}
          />
        </div>
      </div>

      {/* Tempo Profile */}
      <div className="dota-panel p-4">
        <h3 className="font-semibold text-dota-text mb-3">Tempo Profile</h3>
        <div className="flex items-center gap-3">
          {tempoIcons[tempo]}
          <div>
            <div className="font-medium text-dota-text">{tempoLabels[tempo]}</div>
            <div className="text-sm text-dota-textMuted">
              {tempo === 'early' && 'Strong early pressure, aim to close early'}
              {tempo === 'mid' && 'Balanced mid-game power spike'}
              {tempo === 'late' && 'Scales well, play for late game'}
            </div>
          </div>
        </div>
      </div>

      {/* No lane panel: the engine only ever reports a constant 0.5 per lane,
          so showing bars would invent a lane assessment it does not make. */}

      {/* Strong Points */}
      {strong.length > 0 && (
        <div className="dota-panel p-4">
          <h3 className="font-semibold text-dota-text mb-3 flex items-center gap-2">
            <Sword className="w-5 h-5 text-green-400" />
            Strengths
          </h3>
          <div className="flex flex-wrap gap-2">
            {strong.map((s, i) => (
              <span key={i} className="px-3 py-1 bg-green-500/15 text-green-400 rounded-full text-sm border border-green-500/30">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Missing Elements */}
      {missing.length > 0 && (
        <div className="dota-panel p-4">
          <h3 className="font-semibold text-dota-text mb-3 flex items-center gap-2">
            <Shield className="w-5 h-5 text-red-400" />
            Missing
          </h3>
          <div className="flex flex-wrap gap-2">
            {missing.map((m, i) => (
              <span key={i} className="px-3 py-1 bg-red-500/15 text-red-400 rounded-full text-sm border border-red-500/30">
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Draft Phase Info */}
      <div className="dota-panel p-4">
        <h3 className="font-semibold text-dota-text mb-3 flex items-center gap-2">
          <Timer className="w-5 h-5 text-dota-accent" />
          Draft Progress
        </h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-dota-textMuted">Picks Completed</span>
            <span className="font-medium text-dota-text">
              {draft.radiant_picks.length + draft.dire_picks.length}/10
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-dota-textMuted">Bans Completed</span>
            <span className="font-medium text-dota-text">
              {draft.bans.length}/10
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-dota-textMuted">Current Phase</span>
            <span className="font-medium text-dota-text">
              {draft.is_picking_phase ? 'Picking' : 'Waiting'}
            </span>
          </div>
          {draft.current_pick && (
            <div className="flex justify-between">
              <span className="text-dota-textMuted">Next Pick</span>
              <span className="font-medium text-dota-text">
                #{draft.pick_number + 1} - {draft.current_pick === 'radiant' ? 'Radiant' : 'Dire'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
