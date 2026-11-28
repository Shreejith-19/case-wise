import { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  Download,
  ChevronDown,
  ChevronUp,
  Info,
  Clock,
} from 'lucide-react';
import { ABSTAIN_MESSAGES } from '../data/constants';
import { parseAnswerSegments, buildResearchMemo } from '../utils/formatters';

export default function AnswerPanel({
  result,
  isTurnActive = true,
  selectedIndex,
  onSelectPassage,
}) {
  const [copied, setCopied] = useState(false);
  const [showTrace, setShowTrace] = useState(false);

  if (!result) return null;

  const segments = parseAnswerSegments(result.answer);
  const abstainInfo = ABSTAIN_MESSAGES[result.reason] || {
    title: 'No Matching Provision Found',
    detail: 'No statutory provision matched your query with sufficient confidence.',
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(result.answer || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleExportMemo = () => {
    const md = buildResearchMemo(result);
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'legal-research-memo.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  const trace = result.trace || {};
  const hasTrace = Boolean(
    trace.parsed || trace.rerank_top !== undefined || trace.top_hits?.length
  );

  return (
    <div className="space-y-3">
      {/* User Question Header */}
      <div className="rounded-xl border border-slate-200/90 bg-slate-100/70 px-4 py-2.5">
        <p className="text-[13.5px] font-medium leading-relaxed text-slate-900">{result.question}</p>
        {result.effective_question && result.effective_question !== result.question && (
          <p className="mt-1 text-[11px] text-slate-500">
            Contextualized query: <span className="italic">{result.effective_question}</span>
          </p>
        )}
      </div>

      {result.abstained ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50/70 p-5">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-800">
              <AlertTriangle className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-amber-950">{abstainInfo.title}</h2>
                {result.seconds > 0 && (
                  <span className="inline-flex items-center gap-1 text-[11px] text-amber-800/75">
                    <Clock className="h-3 w-3" />
                    {result.seconds}s
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-amber-900/85">
                {abstainInfo.detail}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              <span className="text-xs font-semibold tracking-wide text-slate-700 uppercase">
                Answer
              </span>
              {result.seconds > 0 && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                  <Clock className="h-2.5 w-2.5" />
                  {result.seconds}s
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-600 transition hover:bg-slate-50"
              >
                {copied ? <Check className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                type="button"
                onClick={handleExportMemo}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-600 transition hover:bg-slate-50"
              >
                <Download className="h-3 w-3" />
                <span>Export Memo</span>
              </button>
            </div>
          </div>

          <p className="text-[14.5px] leading-[1.7] text-slate-800">
            {segments.map((seg, i) =>
              seg.type === 'quote' ? (
                <mark
                  key={i}
                  title="Verbatim statutory wording"
                  className="mx-0.5 rounded bg-emerald-50 px-1 py-0.5 text-emerald-950 ring-1 ring-emerald-200/80"
                >
                  &ldquo;{seg.value}&rdquo;
                </mark>
              ) : (
                <span key={i}>{seg.value}</span>
              )
            )}
          </p>

          {result.cited?.length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-medium tracking-wide text-slate-400 uppercase">
                  Cited Provisions
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] text-emerald-700">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  Direct statutory quotes highlighted
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {result.cited.map((c, idx) => {
                  const matchIdx = result.passages?.findIndex(
                    (p) => p.document_id === c.document_id && p.key === c.key
                  );
                  const targetIdx = matchIdx >= 0 ? matchIdx : 0;
                  const active = isTurnActive && selectedIndex === targetIdx;
                  return (
                    <button
                      key={`${c.document_id}-${c.key}-${idx}`}
                      type="button"
                      onClick={() => onSelectPassage(targetIdx)}
                      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-left text-xs transition ${
                        active
                          ? 'border-[#1E3A5F] bg-[#1E3A5F] text-white'
                          : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-[#1E3A5F]/40 hover:bg-white'
                      }`}
                    >
                      <span className="font-mono text-[10px] opacity-75">[{idx + 1}]</span>
                      <span className="max-w-[260px] truncate font-medium">{c.title}</span>
                      {c.label && (
                        <span className={`font-mono text-[11px] ${active ? 'text-blue-100' : 'text-slate-500'}`}>
                          {c.label}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {hasTrace && (
        <div className="rounded-xl border border-slate-200 bg-white">
          <button
            type="button"
            onClick={() => setShowTrace((v) => !v)}
            className="flex w-full items-center justify-between px-4 py-2.5 text-left text-xs font-medium text-slate-600 transition hover:bg-slate-50"
          >
            <span className="inline-flex items-center gap-1.5">
              <Info className="h-3.5 w-3.5 text-slate-400" />
              <span>Retrieval Details</span>
            </span>
            {showTrace ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>

          {showTrace && (
            <div className="space-y-2 border-t border-slate-100 px-4 py-3 text-[11px] text-slate-600">
              {trace.rerank_top !== null && trace.rerank_top !== undefined && (
                <div>
                  <span className="font-semibold text-slate-700">Best cross-encoder score:</span>{' '}
                  <span className="font-mono">{trace.rerank_top}</span>
                </div>
              )}
              {trace.parsed?.acts?.length > 0 && (
                <div>
                  <span className="font-semibold text-slate-700">Detected Acts:</span>{' '}
                  {trace.parsed.acts.join(', ')}
                </div>
              )}
              {trace.parsed?.refs?.length > 0 && (
                <div>
                  <span className="font-semibold text-slate-700">Detected Refs:</span>{' '}
                  {trace.parsed.refs.join(', ')}
                </div>
              )}
              {trace.parsed?.jurisdictions?.length > 0 && (
                <div>
                  <span className="font-semibold text-slate-700">Detected Jurisdictions:</span>{' '}
                  {trace.parsed.jurisdictions.join(', ')}
                </div>
              )}
              {trace.top_hits?.length > 0 && (
                <div>
                  <span className="font-semibold text-slate-700">Top Ranked Hits:</span>
                  <ul className="mt-1 space-y-0.5 pl-3">
                    {trace.top_hits.map((h, idx) => (
                      <li key={idx} className="truncate">
                        {h.title} — <span className="text-slate-500">{h.label}</span>
                        {h.ce_score !== null && h.ce_score !== undefined && (
                          <span className="ml-1 font-mono text-[10px] text-slate-400">
                            (CE {h.ce_score})
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {result.passages?.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs">
          <h3 className="mb-3 px-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">
            Relevant Provisions ({result.passages.length})
          </h3>
          <div className="space-y-2">
            {result.passages.map((p, idx) => {
              const isSelected = isTurnActive && selectedIndex === idx;
              return (
                <button
                  key={`${p.document_id}-${p.char_start}-${idx}`}
                  type="button"
                  onClick={() => onSelectPassage(idx)}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    isSelected
                      ? 'border-[#1E3A5F] bg-[#1E3A5F]/[0.03] ring-1 ring-[#1E3A5F]/20'
                      : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/70'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-slate-700">
                          {p.index || idx + 1}
                        </span>
                        <span className="truncate text-[13px] font-semibold text-slate-900">
                          {p.title}
                        </span>
                      </div>
                      {p.label && (
                        <p className="mt-1 text-xs font-medium text-[#1E3A5F]">{p.label}</p>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {p.pinned && (
                        <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-[#1E3A5F]">
                          Exact
                        </span>
                      )}
                      {p.ce_score !== null && p.ce_score !== undefined && (
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-500">
                          {p.ce_score}
                        </span>
                      )}
                      {p.jurisdiction && (
                        <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                          {p.jurisdiction}
                        </span>
                      )}
                    </div>
                  </div>
                  {p.text && (
                    <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-slate-600">
                      {p.text}
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
