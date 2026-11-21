import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Hero,
  DraftState,
  RecommendationSet,
  TeamSide,
  HeroId,
  AppConfig,
  BanSuggestion,
} from '@shared';

// Engine state
interface EngineState {
  draft: DraftState;
  recommendations: RecommendationSet | null;
  banSuggestions: BanSuggestion[];
  isProcessing: boolean;
  lastUpdate: number;
  heroes: Hero[];
  heroMap: Map<HeroId, Hero>;
  setDraft: (draft: DraftState) => void;
  setRecommendations: (recs: RecommendationSet) => void;
  setBanSuggestions: (suggestions: BanSuggestion[]) => void;
  clearDraftData: () => void;
  setProcessing: (processing: boolean) => void;
  setHeroes: (heroes: Hero[]) => void;
  getHero: (id: HeroId) => Hero | undefined;
}

// UI state
interface UIState {
  selectedTeam: TeamSide;
  activeTab: 'recommendations' | 'draft' | 'analysis' | 'settings';
  setSelectedTeam: (team: TeamSide) => void;
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
  has_live_draft: false,
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
      banSuggestions: [],
      isProcessing: false,
      lastUpdate: 0,
      heroes: [],
      heroMap: new Map(),
      
      setDraft: (draft) => set({ draft, lastUpdate: Date.now() }),
      setRecommendations: (recommendations) => set({ recommendations, lastUpdate: Date.now() }),
      setBanSuggestions: (banSuggestions) => set({ banSuggestions, lastUpdate: Date.now() }),
      // Used when the database holds no heroes: persisted draft/recommendation
      // state from an earlier session would otherwise contradict the UI.
      clearDraftData: () => set({ draft: initialDraftState, recommendations: null, banSuggestions: [], lastUpdate: Date.now() }),
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
      activeTab: 'recommendations',
      
      setSelectedTeam: (selectedTeam) => set({ selectedTeam }),
      setActiveTab: (activeTab) => set({ activeTab }),
    }),
    {
      name: 'ui-store',
      // Only these two are real UI state; overlay geometry lives in the
      // backend config, and the overlay window has its own local state.
      partialize: (state) => ({
        selectedTeam: state.selectedTeam,
        activeTab: state.activeTab,
      }),
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

export const useBanSuggestions = () => useEngineStore((state) =>
  state.banSuggestions
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