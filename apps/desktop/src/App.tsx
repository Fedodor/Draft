import { useEffect, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { DraftState, TeamSide } from '@shared';
import { useConfigStore, useEngineStore, useUIStore } from './stores';
import { api } from './services/api';
import { DraftOverlay } from './components/DraftOverlay';
import { SettingsPanel } from './components/SettingsPanel';
import { HeroPicker } from './components/HeroPicker';
import { TeamAnalysisPanel } from './components/TeamAnalysisPanel';
import { RecommendationPanel } from './components/RecommendationPanel';
import { Header } from './components/Header';
import { Toaster } from './components/Toaster';

function App() {
  const { setHeroes, setDraft, setRecommendations, setProcessing } = useEngineStore();
  const { setConfig } = useConfigStore();
  const { selectedTeam, activeTab, setActiveTab, isOverlayVisible, setOverlayVisible, setSelectedTeam } = useUIStore();

  // Keep the latest selected team available to the long-lived GSI listener
  // without re-subscribing (and re-initializing) on every team change.
  const selectedTeamRef = useRef<TeamSide>(selectedTeam);
  useEffect(() => {
    selectedTeamRef.current = selectedTeam;
  }, [selectedTeam]);

  // One-time initialization: hero data, backend config, current draft.
  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        setProcessing(true);

        const [heroes, draft, config] = await Promise.all([
          api.getHeroList(),
          api.getDraftState(),
          api.getConfig().catch(() => null),
        ]);
        if (cancelled) return;

        setHeroes(heroes);
        setDraft(draft);
        if (config) setConfig(config);

        const recommendations = await api.getRecommendations({
          team: selectedTeamRef.current,
          limit: 10,
        });
        if (!cancelled) setRecommendations(recommendations);
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
  }, [setHeroes, setDraft, setConfig, setRecommendations, setProcessing]);

  // Team change only refreshes recommendations; draft state is left intact.
  useEffect(() => {
    let cancelled = false;
    api
      .getRecommendations({ team: selectedTeam, limit: 10 })
      .then((recs) => {
        if (!cancelled) setRecommendations(recs);
      })
      .catch((error) => console.error('Recommendation refresh error:', error));
    return () => {
      cancelled = true;
    };
  }, [selectedTeam, setRecommendations]);

  // GSI draft updates.
  useEffect(() => {
    const unlisten = listen<DraftState>('gsi-update', async () => {
      try {
        const [draft, recommendations] = await Promise.all([
          api.getDraftState(),
          api.getRecommendations({ team: selectedTeamRef.current, limit: 10 }),
        ]);
        setDraft(draft);
        setRecommendations(recommendations);
      } catch (error) {
        console.error('GSI update error:', error);
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [setDraft, setRecommendations]);

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

  if (!isOverlayVisible) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-dota-bg">
        <button
          onClick={() => setOverlayVisible(true)}
          className="dota-btn-primary"
        >
          Show Overlay
        </button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 flex flex-col h-screen w-screen bg-dota-bg overflow-hidden">
      <Header 
        activeTab={activeTab} 
        onTabChange={setActiveTab}
        selectedTeam={selectedTeam}
      />
      
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
