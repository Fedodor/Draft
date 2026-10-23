import { useEngineStore, useUIStore } from '../stores';
import { HeroPortrait } from './HeroPortrait';
import { motion, AnimatePresence } from 'framer-motion';

interface DraftTeamProps {
  title: string;
  picks: Array<{ hero_id: number; order: number }>;
  bans: Array<{ hero_id: number; order: number }>;
}

function DraftTeam({ title, picks, bans }: DraftTeamProps) {
  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className="flex items-center justify-between px-3 py-2 border-b border-dota-border">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-dota-textMuted">
          {title}
        </h3>
        <span className="text-xs text-dota-textMuted">
          {picks.length}/5
        </span>
      </div>

      {/* Picks */}
      <div className="flex-1 flex items-center justify-center gap-2 p-3 overflow-x-auto">
        <AnimatePresence mode="popLayout">
          {picks.map((pick) => (
            <motion.div
              key={`${pick.hero_id}-${pick.order}`}
              initial={{ opacity: 0, scale: 0.8, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: -10 }}
              transition={{ duration: 0.2 }}
              className="relative flex-shrink-0"
            >
              <HeroPortrait 
                heroId={pick.hero_id} 
                size={56}
                showOrder={true}
                order={pick.order}
              />
            </motion.div>
          ))}
          {picks.length < 5 && (
            <>
              {Array.from({ length: 5 - picks.length }).map((_, i) => (
                <div
                  key={`empty-pick-${i}`}
                  className="w-14 h-14 rounded-lg border-2 border-dashed border-dota-border/50 bg-dota-bg/50 flex-shrink-0"
                />
              ))}
            </>
          )}
        </AnimatePresence>
      </div>

      {/* Bans */}
      <div className="px-3 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-dota-textMuted">Bans:</span>
          <div className="flex gap-1 flex-1 overflow-x-auto">
            {bans.map((ban) => (
              <HeroPortrait 
                key={ban.hero_id}
                heroId={ban.hero_id} 
                size={32}
                isBan={true}
              />
            ))}
            {bans.length < 5 && (
              Array.from({ length: 5 - bans.length }).map((_, i) => (
                <div
                  key={`empty-ban-${i}`}
                  className="w-8 h-8 rounded border-2 border-dashed border-dota-border/30 bg-dota-bg/30 flex-shrink-0"
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function DraftOverlay() {
  const { draft, isProcessing } = useEngineStore();
  const selectedTeam = useUIStore((state) => state.selectedTeam);

  // Orient the panel from the player's own side, not a fixed Radiant assumption.
  const allyIsRadiant = selectedTeam === 'radiant';
  const allyPicks = allyIsRadiant ? draft.radiant_picks : draft.dire_picks;
  const enemyPicks = allyIsRadiant ? draft.dire_picks : draft.radiant_picks;
  const allySide = allyIsRadiant ? 'radiant' : 'dire';
  const enemySide = allyIsRadiant ? 'dire' : 'radiant';

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Phase Indicator */}
      <div className="px-4 py-2 border-b border-dota-border bg-dota-panel/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
              draft.is_picking_phase
                ? 'bg-dota-accent/20 text-dota-accent'
                : 'bg-dota-border text-dota-textMuted'
            }`}>
              {draft.is_picking_phase ? 'PICKING PHASE' : 'WAITING'}
            </span>
            {draft.current_pick && (
              <span className="text-xs text-dota-textMuted">
                {'Pick #'}{draft.pick_number + 1} - {draft.current_pick === 'radiant' ? 'Radiant' : 'Dire'}'s turn
              </span>
            )}
          </div>
          {isProcessing && (
            <div className="flex items-center gap-1.5 text-xs text-dota-accent">
              <div className="w-3 h-3 border-2 border-dota-accent border-t-transparent rounded-full animate-spin" />
              Analyzing...
            </div>
          )}
        </div>
      </div>

      {/* Draft Grid */}
      <div className="flex-1 flex overflow-hidden">
        <DraftTeam
          title="ALLY"
          picks={allyPicks}
          bans={draft.bans.filter(b => b.team === allySide)}
        />
        <div className="w-px bg-dota-border" />
        <DraftTeam
          title="ENEMY"
          picks={enemyPicks}
          bans={draft.bans.filter(b => b.team === enemySide)}
        />
      </div>
    </div>
  );
}