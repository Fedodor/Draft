import { useConfigStore } from '../stores';
import { api, describeCommandError, type MlStatus } from '../services/api';
import { useGsiStatus } from '../hooks/useGsiStatus';
import type { AppConfig } from '@shared';
import {
  Server, Monitor, Brain,
  Loader2, CheckCircle, AlertCircle
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

function validateConfig(config: AppConfig): string | null {
  if (!Number.isInteger(config.gsi_port) || config.gsi_port < 1024 || config.gsi_port > 65535) {
    return 'GSI port must be between 1024 and 65535';
  }
  if (!config.ml_model_path.trim()) {
    return 'Model path must not be empty';
  }
  if (config.gsi_auth_token !== null && !config.gsi_auth_token.trim()) {
    return 'Auth token must be empty (disabled) or a non-empty string';
  }
  return null;
}

function gsiSettingsChanged(a: AppConfig, b: AppConfig): boolean {
  return (
    a.gsi_port !== b.gsi_port ||
    a.gsi_auth_token !== b.gsi_auth_token ||
    a.gsi_auto_start !== b.gsi_auto_start
  );
}

export function SettingsPanel() {
  const { config, setConfig, resetConfig } = useConfigStore();
  const gsi = useGsiStatus();
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [mlStatus, setMlStatus] = useState<MlStatus | null>(null);
  const [installStatus, setInstallStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const savedConfigRef = useRef<AppConfig | null>(null);

  // Backend is the source of truth; hydrate the local store from it.
  useEffect(() => {
    let cancelled = false;
    api.getConfig()
      .then((stored) => {
        if (cancelled) return;
        savedConfigRef.current = stored;
        setConfig(stored);
      })
      .catch(() => {});
    api.getMlStatus()
      .then((s) => { if (!cancelled) setMlStatus(s); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [setConfig]);

  const handleInstallGsi = async () => {
    setInstallStatus('idle');
    try {
      const path = await api.installGsiConfig();
      console.info('GSI config installed at', path);
      setInstallStatus('success');
      setTimeout(() => setInstallStatus('idle'), 3000);
    } catch (error) {
      console.error('GSI config install failed:', error);
      setInstallStatus('error');
      setTimeout(() => setInstallStatus('idle'), 3000);
    }
  };

  const handleSave = async () => {
    const validationError = validateConfig(config);
    if (validationError) {
      setErrorMessage(validationError);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
      return;
    }

    setIsSaving(true);
    setSaveStatus('idle');
    setErrorMessage(null);
    const previous = savedConfigRef.current;
    const changed = previous === null || gsiSettingsChanged(previous, config);

    try {
      await api.setConfig(config);
      savedConfigRef.current = config;

      // Rebind the listener so a changed port/auth takes effect immediately.
      if (changed) {
        if (config.gsi_auto_start) {
          await api.restartGsi();
        } else {
          await api.stopGsi();
        }
      }

      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (error) {
      console.error('Failed to save config:', error);
      setErrorMessage(describeCommandError(error));
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Reset all settings to defaults?')) return;
    resetConfig();
    setErrorMessage(null);
    try {
      // Persist defaults; reload from the store after reset applies them.
      const defaults = useConfigStore.getState().config;
      await api.setConfig(defaults);
      savedConfigRef.current = defaults;
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (error) {
      console.error('Failed to reset config:', error);
      setErrorMessage(describeCommandError(error));
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {/* GSI Settings */}
      <SettingsSection title="Game State Integration" icon={<Server className="w-5 h-5" />}>
        <div className="space-y-4">
          <SettingItem
            label="GSI Port"
            description="Port for Dota 2 to send game state data"
          >
            <input
              type="number"
              value={Number.isNaN(config.gsi_port) ? '' : config.gsi_port}
              onChange={(e) => setConfig({ gsi_port: parseInt(e.target.value, 10) })}
              min={1024}
              max={65535}
              className="w-32 px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
            />
          </SettingItem>

          <SettingItem
            label="Auto-start GSI"
            description="Automatically start the GSI server on app launch"
          >
            <Toggle
              checked={config.gsi_auto_start}
              onChange={(checked) => setConfig({ gsi_auto_start: checked })}
            />
          </SettingItem>

          <SettingItem
            label="Auth Token"
            description="Optional shared secret; Dota must send the matching token"
          >
            <input
              type="text"
              value={config.gsi_auth_token ?? ''}
              onChange={(e) => setConfig({ gsi_auth_token: e.target.value || null })}
              className="w-full px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent font-mono"
              placeholder="(disabled)"
            />
          </SettingItem>

          <div className="pt-2 border-t border-dota-border/50 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${gsi.running ? 'bg-green-500' : 'bg-red-500/70'}`} />
              <span className="font-medium text-dota-text">
                {gsi.running ? `Listening on port ${gsi.port}` : 'GSI stopped'}
                {gsi.auth_required ? ' (auth)' : ''}
              </span>
            </div>
            {gsi.last_error && (
              <p className="text-xs text-red-400">{gsi.last_error}</p>
            )}
            {!gsi.auth_required && (
              <p className="text-xs text-amber-400">
                Auth token is empty: any local process can post a draft payload.
              </p>
            )}
            <code className="text-xs text-dota-textMuted bg-dota-bg px-2 py-1 rounded block whitespace-pre-wrap font-mono">
              {`gamestate_integration_smart_draft.cfg`}
            </code>
            <p className="text-xs text-dota-textMuted mt-1">
              Place this file in Dota 2 cfg/gamestate_integration/ directory
            </p>
            <button
              onClick={handleInstallGsi}
              className="mt-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-dota-accent/15 text-dota-accent border border-dota-accent/30 hover:bg-dota-accent/25 transition-colors"
            >
              {installStatus === 'success' ? 'Installed' : installStatus === 'error' ? 'Failed' : 'Install Config'}
            </button>
          </div>
        </div>
      </SettingsSection>

      {/* Overlay Settings */}
      <SettingsSection title="Overlay" icon={<Monitor className="w-5 h-5" />}>
        <div className="space-y-4">
          <SettingItem
            label="Enable Overlay"
            description="Show draft overlay during matches"
          >
            <Toggle 
              checked={config.overlay_enabled} 
              onChange={(checked) => setConfig({ overlay_enabled: checked })} 
            />
          </SettingItem>

          <SettingItem
            label="Overlay Position"
            description="X, Y coordinates on screen"
          >
            <div className="flex gap-2">
              <input
                type="number"
                value={config.overlay_position.x}
                onChange={(e) => setConfig({ overlay_position: { ...config.overlay_position, x: parseInt(e.target.value) } })}
                className="w-24 px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
                placeholder="X"
              />
              <input
                type="number"
                value={config.overlay_position.y}
                onChange={(e) => setConfig({ overlay_position: { ...config.overlay_position, y: parseInt(e.target.value) } })}
                className="w-24 px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
                placeholder="Y"
              />
            </div>
          </SettingItem>

          <SettingItem
            label="Overlay Size"
            description="Width and height in pixels"
          >
            <div className="flex gap-2">
              <input
                type="number"
                value={config.overlay_size.width}
                onChange={(e) => setConfig({ overlay_size: { ...config.overlay_size, width: parseInt(e.target.value) } })}
                className="w-24 px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
                placeholder="Width"
              />
              <input
                type="number"
                value={config.overlay_size.height}
                onChange={(e) => setConfig({ overlay_size: { ...config.overlay_size, height: parseInt(e.target.value) } })}
                className="w-24 px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent"
                placeholder="Height"
              />
            </div>
          </SettingItem>
        </div>
      </SettingsSection>

      {/* ML Settings */}
      <SettingsSection title="Machine Learning" icon={<Brain className="w-5 h-5" />}>
        <div className="space-y-4">
          <SettingItem
            label="Enable ML Recommendations"
            description="Experimental. The bundled model is near the 0.5 baseline; off by default."
          >
            <Toggle 
              checked={config.ml_enabled} 
              onChange={(checked) => setConfig({ ml_enabled: checked })} 
            />
          </SettingItem>

          <SettingItem
            label="Model Path"
            description="Path to ONNX model file"
          >
            <input
              type="text"
              value={config.ml_model_path}
              onChange={(e) => setConfig({ ml_model_path: e.target.value })}
              className="w-full px-3 py-2 bg-dota-bg border border-dota-border rounded-lg text-sm text-dota-text focus:outline-none focus:ring-2 focus:ring-dota-accent focus:border-transparent font-mono"
              placeholder="ml/models/draft_model_xgboost_full.onnx"
            />
          </SettingItem>

          <div className="pt-2 border-t border-dota-border/50">
            <div className="flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${mlStatus?.loaded ? 'bg-green-500' : 'bg-red-500/70'}`} />
              <span className="font-medium text-dota-text">
                {mlStatus?.loaded ? 'Model Loaded' : 'Model Not Loaded'}
              </span>
              {mlStatus && mlStatus.loaded && (
                <span className="text-xs text-dota-textMuted font-mono">
                  {mlStatus.feature_count} features
                </span>
              )}
            </div>
            {mlStatus?.model_path && (
              <code className="text-xs text-dota-textMuted bg-dota-bg px-2 py-1 rounded block mt-2 whitespace-pre-wrap font-mono">
                {mlStatus.model_path}
              </code>
            )}
          </div>
        </div>
      </SettingsSection>

      {/* Actions */}
      <div className="flex flex-col gap-2 pt-4 border-t border-dota-border">
        {errorMessage && (
          <p className="text-xs text-red-400">{errorMessage}</p>
        )}
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={handleReset}
            className="dota-btn-secondary"
          >
            Reset to Defaults
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="dota-btn-primary flex items-center gap-2"
          >
            {isSaving && <Loader2 className="w-4 h-4 animate-spin" />}
            {saveStatus === 'success' && <CheckCircle className="w-4 h-4" />}
            {saveStatus === 'error' && <AlertCircle className="w-4 h-4" />}
            <span>{isSaving ? 'Saving...' : 'Save Settings'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

interface SettingsSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function SettingsSection({ title, icon, children }: SettingsSectionProps) {
  return (
    <div className="dota-panel p-4">
      <h3 className="flex items-center gap-2 font-semibold text-dota-text mb-4">
        {icon}
        {title}
      </h3>
      {children}
    </div>
  );
}

interface SettingItemProps {
  label: string;
  description: string;
  children: React.ReactNode;
}

function SettingItem({ label, description, children }: SettingItemProps) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex-1 min-w-0">
        <label className="block font-medium text-sm text-dota-text mb-0.5">{label}</label>
        <p className="text-xs text-dota-textMuted">{description}</p>
      </div>
      <div className="flex-shrink-0 w-[200px]">{children}</div>
    </div>
  );
}

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative w-12 h-6 rounded-full transition-colors flex items-center px-1 ${
        checked ? 'bg-dota-accent' : 'bg-dota-border'
      }`}
      role="switch"
      aria-checked={checked}
    >
      <span className={`w-4 h-4 rounded-full bg-white shadow-md transition-transform ${
        checked ? 'translate-x-full' : 'translate-x-0'
      }`} />
    </button>
  );
}
