import type { DataHealthSummary } from '../services/dataHealth';

interface DataHealthNoticeProps {
  summary: DataHealthSummary;
  onRecheck: () => void;
  checking?: boolean;
}

export function DataHealthNotice({ summary, onRecheck, checking = false }: DataHealthNoticeProps) {
  if (summary.level === 'ok') return null;

  const tone =
    summary.level === 'empty'
      ? 'border-red-500/40 bg-red-500/10'
      : 'border-yellow-500/40 bg-yellow-500/10';

  return (
    <div role="status" className={`border-b px-4 py-2 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold text-dota-text">{summary.title}</div>
          <p className="mt-0.5 text-xs text-dota-textMuted">{summary.detail}</p>
          {summary.command && (
            <pre className="mt-1.5 overflow-x-auto rounded bg-dota-bg/70 p-2 font-mono text-[11px] leading-relaxed text-dota-text">
              {summary.command}
            </pre>
          )}
        </div>
        <button
          type="button"
          onClick={onRecheck}
          disabled={checking}
          className="shrink-0 rounded border border-dota-border px-2 py-1 text-xs text-dota-text transition-colors hover:bg-dota-border/40 disabled:opacity-50"
        >
          {checking ? 'Checking…' : 'Recheck'}
        </button>
      </div>
    </div>
  );
}
