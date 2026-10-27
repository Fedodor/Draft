import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Hero,
  DraftState,
  RecommendationSet,
  TeamSide,
  HeroId,
  AppConfig,
} from '@shared';

// Engine state
interface EngineState {
  draft: DraftState;
  recommendations: RecommendationSet | null;
  isProcessing: boolean;
  lastUpdate: number;
  heroes: Hero[];
  heroMap: Map<HeroId, Hero>;
  setDraft: (draft: DraftState) => void;
  setRecommendations: (recs: RecommendationSet) => void;
  setProcessing: (processing: boolean) => void;
  setHeroes: (heroes: Hero[]) => void;
  getHero: (id: HeroId) => Hero | undefined;
}

// UI state
interface UIState {
  selectedTeam: TeamSide;
  overlayPosition: { x: number; y: number };
  overlaySize: { width: number; height: number };
  isOverlayVisible: boolean;
  isSettingsOpen: boolean;
  activeTab: 'recommendations' | 'draft' | 'analysis' | 'settings';
  setSelectedTeam: (team: TeamSide) => void;
  setOverlayPosition: (pos: { x: number; y: number }) => void;
  setOverlaySize: (size: { width: number; height: number }) => void;
  setOverlayVisible: (visible: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setActiveTab: (tab: UIState['activeTab']) => void;
}

// Config state
interface ConfigState {
  config: AppConfig;
  setConfig: (config: Partial<AppConfig>) => void;
  resetConfig: () => void;
}

const defaultConfig: AppConfig = {
  gsi_port: 3000,
  gsi_auto_start: true,
  gsi_auth_token: null,
  overlay_enabled: true,
  overlay_position: { x: 100, y: 100 },
  overlay_size: { width: 400, height: 600 },
  ml_enabled: false,
  ml_model_path: 'ml/models/draft_model_xgboost_full.onnx',
  auto_update_meta: true,
  preferred_language: 'en',
};

const initialDraftState: DraftState = {
  radiant_picks: [],
  dire_picks: [],
  bans: [],
  current_pick: null,
  pick_number: 0,
  is_picking_phase: false,
  available_heroes: [],
  patch_version: '7.41',
  timestamp: Date.now(),
};

// Engine Store
export const useEngineStore = create<EngineState>()(
  persist(
    (set, get) => ({
      draft: initialDraftState,
      recommendations: null,
      isProcessing: false,
      lastUpdate: 0,
      heroes: [],
      heroMap: new Map(),
      
      setDraft: (draft) => set({ draft, lastUpdate: Date.now() }),
      setRecommendations: (recommendations) => set({ recommendations, lastUpdate: Date.now() }),
      setProcessing: (isProcessing) => set({ isProcessing }),
      setHeroes: (heroes) => {
        const heroMap = new Map(heroes.map(h => [h.id, h]));
        set({ heroes, heroMap });
      },
      getHero: (id) => get().heroMap.get(id),
    }),
    {
      name: 'engine-store',
      partialize: (state) => ({
        draft: state.draft,
        recommendations: state.recommendations,
      }),
    }
  )
);

// UI Store
export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      selectedTeam: 'radiant',
      overlayPosition: { x: 100, y: 100 },
      overlaySize: { width: 400, height: 600 },
      isOverlayVisible: true,
      isSettingsOpen: false,
      activeTab: 'recommendations',
      
      setSelectedTeam: (selectedTeam) => set({ selectedTeam }),
      setOverlayPosition: (overlayPosition) => set({ overlayPosition }),
      setOverlaySize: (overlaySize) => set({ overlaySize }),
      setOverlayVisible: (isOverlayVisible) => set({ isOverlayVisible }),
      setSettingsOpen: (isSettingsOpen) => set({ isSettingsOpen }),
      setActiveTab: (activeTab) => set({ activeTab }),
    }),
    {
      name: 'ui-store',
    }
  )
);

// Config Store
export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      config: defaultConfig,
      setConfig: (config) => set((state) => ({ config: { ...state.config, ...config } })),
      resetConfig: () => set({ config: defaultConfig }),
    }),
    {
      name: 'config-store',
    }
  )
);

// Computed selectors
export const useAllyPicks = () => useEngineStore((state) => 
  state.draft.radiant_picks
);

export const useEnemyPicks = () => useEngineStore((state) => 
  state.draft.dire_picks
);

export const useBans = () => useEngineStore((state) => 
  state.draft.bans
);

export const useTopRecommendations = () => useEngineStore((state) => 
  state.recommendations?.top_picks ?? []
);

export const useSafeRecommendations = () => useEngineStore((state) => 
  state.recommendations?.safe_picks ?? []
);

export const useAggressiveRecommendations = () => useEngineStore((state) => 
  state.recommendations?.aggressive_picks ?? []
);

export const useComfortRecommendations = () => useEngineStore((state) => 
  state.recommendations?.comfort_picks ?? []
);

export const useAntiMetaRecommendations = () => useEngineStore((state) => 
  state.recommendations?.anti_meta_picks ?? []
);

export const useTeamAnalysis = () => useEngineStore((state) => 
  state.recommendations?.team_analysis
);

export const useAvailableHeroes = () => useEngineStore((state) => 
  state.draft.available_heroes
);

export const useIsProcessing = () => useEngineStore((state) => 
  state.isProcessing
);