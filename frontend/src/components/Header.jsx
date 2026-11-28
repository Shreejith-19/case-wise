import { Scale, History, Plus, CircleDot } from 'lucide-react';

export default function Header({
  historyCount,
  showHistory,
  onToggleHistory,
  onReset,
  hasResult,
  health,
}) {
  const isLive = health?.mode === 'live';

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/90 bg-white/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1440px] items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleHistory}
            title="Toggle chat history"
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
              showHistory
                ? 'border-[#1E3A5F] bg-[#1E3A5F]/5 text-[#1E3A5F]'
                : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
            }`}
          >
            <History className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Chats</span>
            {historyCount > 0 && (
              <span className="rounded bg-slate-100 px-1.5 py-0.2 text-[10px] text-slate-600">
                {historyCount}
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={onReset}
            className="group flex items-center gap-2.5 text-left focus:outline-none"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#1E3A5F] text-white shadow-xs transition group-hover:bg-[#172E4D]">
              <Scale className="h-4 w-4" />
            </div>
            <div>
              <span className="block text-[15px] font-semibold tracking-tight text-slate-900">
                CaseWise
              </span>
              <span className="hidden text-[11px] text-slate-500 sm:block">
                Indian Statutory Legal Research
              </span>
            </div>
          </button>
        </div>

        <div className="flex items-center gap-2">
          {health && (
            <span
              title={
                isLive
                  ? `Live C_full pipeline (${health.models?.llm || 'qwen3:8b'})`
                  : `KB ready: ${health.kb_ready ? 'yes' : 'no'}, Ollama: ${health.ollama_up ? 'up' : 'down'}`
              }
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
                isLive
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                  : 'border-amber-200 bg-amber-50 text-amber-800'
              }`}
            >
              <CircleDot className={`h-3 w-3 ${isLive ? 'text-emerald-600' : 'text-amber-600'}`} />
              <span>{isLive ? 'Live Engine' : 'Demo Mode'}</span>
            </span>
          )}

          {hasResult && (
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
            >
              <Plus className="h-3.5 w-3.5 text-slate-500" />
              <span>New Chat</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
