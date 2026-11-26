import { Search, ArrowUp, Loader2, MapPin, FileText, Gauge } from 'lucide-react';
import { JURISDICTIONS, DOC_TYPES, BUDGETS } from '../data/constants';

export default function QueryBar({
  question,
  setQuestion,
  jurisdiction,
  setJurisdiction,
  docType,
  setDocType,
  budget,
  setBudget,
  loading,
  onSubmit,
  compact = false,
  isFollowUp = false,
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (question.trim() && !loading) onSubmit();
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (question.trim() && !loading) onSubmit();
      }}
      className={`rounded-2xl border border-slate-200 bg-white shadow-sm transition focus-within:border-[#1E3A5F]/50 focus-within:ring-4 focus-within:ring-[#1E3A5F]/5 ${
        compact ? '' : 'mx-auto w-full max-w-2xl'
      }`}
    >
      <div className="flex items-start gap-3 px-4 pt-3.5 pb-2">
        <Search className="mt-1 h-4 w-4 shrink-0 text-slate-400" />
        <textarea
          rows={compact ? 2 : 3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isFollowUp
              ? 'Ask a follow-up question in this research thread, or ask about a new Act...'
              : 'Ask a legal question or look up an Act, section, or statutory rule...'
          }
          className="w-full resize-none bg-transparent text-[14px] leading-relaxed text-slate-900 placeholder:text-slate-400 focus:outline-none"
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 px-3.5 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <label className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50/70 px-2.5 py-1 text-xs text-slate-700 hover:bg-slate-100/80">
            <MapPin className="h-3 w-3 text-slate-500" />
            <select
              aria-label="Jurisdiction"
              value={jurisdiction}
              onChange={(e) => setJurisdiction(e.target.value)}
              className="bg-transparent font-medium text-slate-700 focus:outline-none"
            >
              {JURISDICTIONS.map((j) => (
                <option key={j} value={j}>
                  {j === 'All' ? 'All Jurisdictions' : j}
                </option>
              ))}
            </select>
          </label>

          <label className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50/70 px-2.5 py-1 text-xs text-slate-700 hover:bg-slate-100/80">
            <FileText className="h-3 w-3 text-slate-500" />
            <select
              aria-label="Instrument Type"
              value={docType}
              onChange={(e) => setDocType(e.target.value)}
              className="bg-transparent font-medium text-slate-700 focus:outline-none"
            >
              {DOC_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t === 'All' ? 'All Instruments' : t}
                </option>
              ))}
            </select>
          </label>

          <label className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50/70 px-2.5 py-1 text-xs text-slate-700 hover:bg-slate-100/80">
            <Gauge className="h-3 w-3 text-slate-500" />
            <select
              aria-label="Context Budget"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="bg-transparent font-medium text-slate-700 focus:outline-none"
            >
              {BUDGETS.map((b) => (
                <option key={b.value} value={b.value}>
                  {b.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          type="submit"
          disabled={!question.trim() || loading}
          className="inline-flex items-center gap-1.5 rounded-xl bg-[#1E3A5F] px-3.5 py-1.5 text-xs font-medium text-white shadow-xs transition hover:bg-[#172E4D] disabled:cursor-not-allowed disabled:opacity-45"
        >
          {loading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Searching...</span>
            </>
          ) : (
            <>
              <span>{isFollowUp ? 'Ask' : 'Search'}</span>
              <ArrowUp className="h-3.5 w-3.5" />
            </>
          )}
        </button>
      </div>
    </form>
  );
}
