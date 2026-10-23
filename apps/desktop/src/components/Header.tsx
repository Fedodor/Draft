import { useEngineStore, useUIStore } from '../stores';
import { useGsiStatus } from '../hooks/useGsiStatus';
import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { 
  GitBranch, 
  BarChart2, 
  Settings,
  Shield,
  Sword,
  Eye,
  EyeOff
} from 'lucide-react';
import { useEffect, useState } from 'react';

interface HeaderProps {
  activeTab: 'recommendations' | 'draft' | 'analysis' | 'settings';
  onTabChange: (tab: 'recommendations' | 'draft' | 'analysis' | 'settings') => void;
  selectedTeam: 'radiant' | 'dire';
}

export function Header({ activeTab, onTabChange, selectedTeam }: HeaderProps) {
  const { setSelectedTeam } = useUIStore();
  const patchVersion = useEngineStore((state) => state.draft.patch_version);
  const gsi = useGsiStatus();
  const [overlayVisible, setOverlayVisible] = useState(false);

  useEffect(() => {
    WebviewWindow.getByLabel('overlay')
      .then((overlay) => overlay?.isVisible())
      .then((visible) => { if (visible !== undefined) setOverlayVisible(visible); })
      .catch(() => {});
  }, []);

  const toggleOverlay = async () => {
    const overlay = await WebviewWindow.getByLabel('overlay');
    if (!overlay) return;
    if (overlayVisible) {
      await overlay.hide();
      setOverlayVisible(false);
    } else {
      await overlay.show();
      await overlay.setAlwaysOnTop(true);
      await overlay.setFocus();
      setOverlayVisible(true);
    }
  };

  const tabs = [
    { id: 'recommendations', label: 'Picks', icon: Sword },
    { id: 'draft', label: 'Draft', icon: GitBranch },
    { id: 'analysis', label: 'Analysis', icon: BarChart2 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ] as const;

  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-dota-border bg-dota-panel/80 backdrop-blur-sm z-10">
      {/* Team Selector */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setSelectedTeam('radiant')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            selectedTeam === 'radiant'
              ? 'bg-dota-accent/20 text-dota-accent'
              : 'text-dota-textMuted hover:text-dota-text hover:bg-dota-border'
          }`}
        >
          <Shield className="w-4 h-4" />
          Radiant
        </button>
        <button
          onClick={() => setSelectedTeam('dire')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            selectedTeam === 'dire'
              ? 'bg-dota-accent/20 text-dota-accent'
              : 'text-dota-textMuted hover:text-dota-text hover:bg-dota-border'
          }`}
        >
          <Sword className="w-4 h-4" />
          Dire
        </button>
      </div>

      {/* Tab Navigation */}
      <nav className="flex items-center gap-1" role="tablist">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            role="tab"
            aria-selected={activeTab === id}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
              activeTab === id
                ? 'bg-dota-accent/15 text-dota-accent border border-dota-accent/30'
                : 'text-dota-textMuted hover:text-dota-text hover:bg-dota-border'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </nav>

      {/* Status Indicators */}
      <div className="flex items-center gap-3">
        <button
          onClick={toggleOverlay}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
            overlayVisible
              ? 'bg-dota-accent/20 text-dota-accent border-dota-accent/40'
              : 'text-dota-textMuted hover:text-dota-text hover:bg-dota-border border-transparent'
          }`}
          title="Toggle the always-on-top draft overlay"
        >
          {overlayVisible ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          Overlay
        </button>
        <div className="flex items-center gap-1.5 text-xs text-dota-textMuted" title={gsi.endpoint || 'GSI server'}>
          <span className={`w-2 h-2 rounded-full ${gsi.running ? 'bg-green-500 animate-pulse' : 'bg-red-500/70'}`} />
          <span>{gsi.running ? 'GSI Live' : 'GSI Off'}</span>
        </div>
        <div className="w-1px h-6 bg-dota-border" />
        <div className="text-xs text-dota-textMuted font-mono">
          Patch {patchVersion || '—'}
        </div>
      </div>
    </header>
  );
}