import { useEngineStore, useUIStore } from '../stores';
import { api, describeCommandError } from '../services/api';
import { HeroPortrait } from './HeroPortrait';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, RotateCcw, Trash2 } from 'lucide-react';
import { useState, useMemo } from 'react';
import type { Hero, HeroId } from '@shared';

type DraftMode = 'pick' | 'ban';

export function HeroPicker() {
  const { heroes, draft, getHero, setDraft, setRecommendations } = useEngineStore();
  const { selectedTeam } = useUIStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRole, setSelectedRole] = useState<string>('all');
  const [mode, setMode] = useState<DraftMode>('pick');
  const [isBusy, setIsBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const availableHeroes = draft.available_heroes;

  const roles = useMemo(() => {
    const roleSet = new Set<string>();
    heroes.forEach(h => h.roles.forEach(r => roleSet.add(r)));
    return Array.from(roleSet).sort();
  }, [heroes]);

  const filteredHeroes = useMemo(() => {
    return availableHeroes
      .map(id => getHero(id))
      .filter((h): h is Hero => !!h)
      .filter(h => {
        if (searchQuery && !h.localized_name.toLowerCase().includes(searchQuery.toLowerCase())) {
          return false;
        }
        if (selectedRole !== 'all' && !h.roles.includes(selectedRole)) {
          return false;
        }
        return true;
      });
  }, [availableHeroes, searchQuery, selectedRole, getHero]);

  const refreshDraft = async () => {
    const [nextDraft, recommendations] = await Promise.all([
      api.getDraftState(),
      api.getRecommendations({ team: selectedTeam, limit: 10 }),
    ]);
    setDraft(nextDraft);
    setRecommendations(recommendations);
  };

  const runAction = async (action: () => Promise<unknown>) => {
    if (isBusy) return;
    setIsBusy(true);
    setErrorMessage(null);
    try {
      await action();
      await refreshDraft();
    } catch (error) {
      setErrorMessage(describeCommandError(error));
    } finally {
      setIsBusy(false);
    }
  };

  const handleHeroAction = (heroId: HeroId) => {
    if (mode === 'pick') {
      return runAction(() => api.manualPick({ hero_id: heroId, team: selectedTeam }));
    }
    return runAction(() => api.manualBan({ hero_id: heroId, team: selectedTeam }));
  };

  const handleUndo = () =>
    runAction(async () => {
      const action = await api.undoLastAction();
      if (!action) {
        throw { kind: 'internal', message: 'Nothing to undo' };
      }
    });

  const handleReset = () => runAction(() => api.resetDraft());

  const isCurrentTurn = draft.current_pick === selectedTeam;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-3 border-b border-dota-border bg-dota-panel/50">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-dota-textMuted">
            Manual Draft Control
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={handleUndo}
              disabled={isBusy}
              title="Undo last pick or ban"
              className="p-1.5 rounded text-dota-textMuted hover:text-dota-text hover:bg-dota-border disabled:opacity-40"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              disabled={isBusy}
              title="Reset draft"
              className="p-1.5 rounded text-dota-textMuted hover:text-red-400 hover:bg-dota-border disabled:opacity-40"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Mode toggle + turn info */}
        <div className="flex items-center gap-2 mb-3">
          <div className="grid grid-cols-2 gap-1 p-0.5 bg-dota-bg border border-dota-border rounded-lg flex-1">
            <button
              onClick={() => setMode('pick')}
              className={`py-1.5 text-xs font-medium rounded ${
                mode === 'pick'
                  ? 'bg-dota-accent text-dota-bg'
                  : 'text-dota-textMuted hover:text-dota-text'
              }`}
            >
              Pick
            </button>
            <button
              onClick={() => setMode('ban')}
              className={`py-1.5 text-xs font-medium rounded ${
                mode === 'ban'
                  ? 'bg-red-500 text-white'
                  : 'text-dota-textMuted hover:text-dota-text'
              }`}
            >
              Ban
            </button>
          </div>
          {!isCurrentTurn && (
            <span className="px-2 py-1 text-xs bg-yellow-500/20 text-yellow-400 rounded whitespace-nowrap">
              Not your turn
            </span>
          )}
        </div>

        {errorMessage && (
          <div className="mb-3 px-2 py-1.5 text-xs bg-red-500/10 border border-red-500/30 text-red-400 rounded">
            {errorMessage}
          </div>
        )}

        {/* Search & Filter */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dota-textMuted" />
            <input
              type="text"
              placeholder="Search heroes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text placeholder-dota-textMuted focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
            />
          </div>
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            className="px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent min-w-[140px]"
          >
            <option value="all">All Roles</option>
            {roles.map(role => (
              <option key={role} value={role}>{role}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Hero Grid */}
      <div className="flex-1 overflow-y-auto p-3">
        <AnimatePresence mode="popLayout">
          {filteredHeroes.map((hero) => (
            <motion.button
              key={hero.id}
              onClick={() => handleHeroAction(hero.id)}
              disabled={isBusy}
              className="hero-card w-full flex items-center gap-3 p-2 text-left disabled:opacity-50 disabled:cursor-not-allowed"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.15 }}
            >
              <HeroPortrait heroId={hero.id} size={40} />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate text-dota-text">{hero.localized_name}</div>
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {hero.roles.slice(0, 3).map(role => (
                    <span key={role} className="px-1.5 py-0.5 text-xs bg-dota-border text-dota-textMuted rounded">
                      {role}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-1 text-dota-textMuted">
                {mode === 'pick' ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                )}
              </div>
            </motion.button>
          ))}
        </AnimatePresence>

        {filteredHeroes.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-dota-textMuted">
            <Filter className="w-12 h-12 mb-3 opacity-50" />
            <p>No heroes match your search</p>
          </div>
        )}
      </div>

      {/* Current Draft Summary */}
      <div className="p-3 border-t border-dota-border bg-dota-panel/50">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xs text-dota-textMuted">ALLY PICKS</div>
            <div className="font-bold text-dota-text">{draft.radiant_picks.length}/5</div>
          </div>
          <div>
            <div className="text-xs text-dota-textMuted">ENEMY PICKS</div>
            <div className="font-bold text-dota-text">{draft.dire_picks.length}/5</div>
          </div>
          <div>
            <div className="text-xs text-dota-textMuted">BANS</div>
            <div className="font-bold text-dota-text">{draft.bans.length}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
