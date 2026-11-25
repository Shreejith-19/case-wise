import { useState } from 'react';
import { CheckCircle2, AlertTriangle, Copy, Check, Download } from 'lucide-react';
import { ABSTAIN_MESSAGES } from '../data/constants';
import { parseAnswerSegments, buildResearchMemo } from '../utils/formatters';

export default function AnswerPanel({ result, selectedIndex, onSelectPassage }) {
  const [copied, setCopied] = useState(false);

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

  return (
    <div className="space-y-4">
      {result.abstained ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50/70 p-5">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-800">
              <AlertTriangle className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="text-sm font-semibold text-amber-950">{abstainInfo.title}</h2>
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
                  const active = selectedIndex === targetIdx;
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

      {result.passages?.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs">
          <h3 className="mb-3 px-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">
            Relevant Provisions ({result.passages.length})
          </h3>
          <div className="space-y-2">
            {result.passages.map((p, idx) => {
              const isSelected = selectedIndex === idx;
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
                    {p.jurisdiction && (
                      <span className="shrink-0 rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                        {p.jurisdiction}
                      </span>
                    )}
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
