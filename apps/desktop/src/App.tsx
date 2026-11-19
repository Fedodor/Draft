import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { TeamSide } from '@shared';
import { useConfigStore, useEngineStore, useUIStore } from './stores';
import { api, type DataHealth } from './services/api';
import { describeDataHealth } from './services/dataHealth';
import { useRecommendationFeed } from './hooks/useRecommendationFeed';
import { DraftOverlay } from './components/DraftOverlay';
import { SettingsPanel } from './components/SettingsPanel';
import { HeroPicker } from './components/HeroPicker';
import { TeamAnalysisPanel } from './components/TeamAnalysisPanel';
import { RecommendationPanel } from './components/RecommendationPanel';
import { Header } from './components/Header';
import { Toaster } from './components/Toaster';
import { DataHealthNotice } from './components/DataHealthNotice';

function App() {
  const { setHeroes, setDraft, setRecommendations, setProcessing, clearDraftData } = useEngineStore();
  const { setConfig } = useConfigStore();
  const { selectedTeam, activeTab, setActiveTab, setSelectedTeam } = useUIStore();
  const [dataHealth, setDataHealth] = useState<DataHealth | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(false);

  // A fresh install has an empty database, which otherwise looks like a broken
  // app: no heroes, no recommendations, no explanation. Surface it instead.
  const loadDataHealth = useCallback(async () => {
    setCheckingHealth(true);
    try {
      setDataHealth(await api.getDataHealth());
    } catch {
      setDataHealth(null);
    } finally {
      setCheckingHealth(false);
      setHealthChecked(true);
    }
  }, []);

  // Keep the latest selected team available to the long-lived feed listener
  // without re-subscribing (and restarting the feed) on every team change.
  const selectedTeamRef = useRef<TeamSide>(selectedTeam);
  useEffect(() => {
    selectedTeamRef.current = selectedTeam;
  }, [selectedTeam]);

  // Draft payloads + debounced recommendations live in one place. The feed
  // owns the `gsi-update` listener and rejects stale responses.
  const { refresh } = useRecommendationFeed({
    getTeam: () => selectedTeamRef.current,
    limit: 10,
    onDraft: setDraft,
    onRecommendations: setRecommendations,
    onError: (error) => console.error('Recommendation update error:', error),
  });

  // One-time initialization: hero data and backend config. The feed loads the
  // current draft and first recommendation set.
  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        setProcessing(true);

        const [heroes, config, health] = await Promise.all([
          api.getHeroList(),
          api.getConfig().catch(() => null),
          api.getDataHealth().catch(() => null),
        ]);
        if (cancelled) return;

        setHeroes(heroes);
        if (config) setConfig(config);
        setDataHealth(health);
        setHealthChecked(true);
        // An unseeded database has no heroes to show; drop draft data persisted
        // by an earlier session so the notice and the panels agree.
        if (health && health.hero_count === 0) clearDraftData();
      } catch (error) {
        console.error('Initialization error:', error);
      } finally {
        if (!cancelled) setProcessing(false);
      }
    };

    init();
    return () => {
      cancelled = true;
    };
  }, [setHeroes, setConfig, setProcessing, clearDraftData]);

  // Team change only refreshes recommendations; draft state is left intact.
  // The feed already fetches once on mount, so skip the first run.
  const teamChangeHandled = useRef(false);
  useEffect(() => {
    if (!teamChangeHandled.current) {
      teamChangeHandled.current = true;
      return;
    }
    void refresh();
  }, [selectedTeam, refresh]);

  // The local player's side, reported by Dota through GSI.
  useEffect(() => {
    const unlisten = listen<TeamSide>('gsi-team', (event) => {
      if (event.payload === 'radiant' || event.payload === 'dire') {
        setSelectedTeam(event.payload);
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [setSelectedTeam]);

  // The main window is not the overlay window. Hiding or disabling the
  // separate overlay must never replace this dashboard with a recovery
  // button, or the user cannot reach Settings to enable it again.
  return (
    <div className="fixed inset-0 flex flex-col h-screen w-screen bg-dota-bg overflow-hidden">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        selectedTeam={selectedTeam}
      />

      {healthChecked && (
        <DataHealthNotice
          summary={describeDataHealth(dataHealth)}
          onRecheck={() => void loadDataHealth()}
          checking={checkingHealth}
        />
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Main Draft Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <DraftOverlay />
        </div>

        {/* Side Panel */}
        <div className="w-96 flex flex-col border-l border-dota-border bg-dota-panel">
          {activeTab === 'recommendations' && <RecommendationPanel />}
          {activeTab === 'draft' && <HeroPicker />}
          {activeTab === 'analysis' && <TeamAnalysisPanel />}
          {activeTab === 'settings' && <SettingsPanel />}
        </div>
      </div>

      <Toaster />
    </div>
  );
}

export default App;
