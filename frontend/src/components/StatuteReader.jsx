import { useState, useEffect } from 'react';
import {
  BookOpen,
  Copy,
  Check,
  GitBranch,
  MapPin,
  Calendar,
  Loader2,
  Maximize2,
  ArrowLeft,
} from 'lucide-react';
import { formatDocType } from '../utils/formatters';

export default function StatuteReader({ passage }) {
  const [copiedCite, setCopiedCite] = useState(false);
  const [fullDoc, setFullDoc] = useState(null);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [docError, setDocError] = useState(null);
  const [xrefStack, setXrefStack] = useState([]);
  const [loadingXref, setLoadingXref] = useState(false);

  const active = xrefStack.length > 0 ? xrefStack[xrefStack.length - 1] : passage;

  useEffect(() => {
    setXrefStack([]);
  }, [passage?.document_id, passage?.char_start]);

  useEffect(() => {
    setFullDoc(null);
    setDocError(null);
  }, [active?.document_id, active?.char_start]);

  if (!active) {
    return (
      <div className="flex h-full min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/60 p-8 text-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-400">
          <BookOpen className="h-5 w-5" />
        </div>
        <h3 className="mt-3 text-sm font-medium text-slate-700">Statute Reader</h3>
        <p className="mt-1 max-w-xs text-xs leading-relaxed text-slate-500">
          Select a cited provision on the left to read the statutory text in full.
        </p>
      </div>
    );
  }

  const handleCopyCitation = async () => {
    const cite = `${active.title}${active.label ? `, ${active.label}` : ''}`;
    await navigator.clipboard.writeText(cite);
    setCopiedCite(true);
    setTimeout(() => setCopiedCite(false), 1800);
  };

  const handleLoadFullAct = async () => {
    setLoadingDoc(true);
    setDocError(null);
    try {
      const qs = new URLSearchParams({
        id: active.document_id,
        start: String(active.char_start ?? 0),
        end: String(active.char_end ?? 0),
        pad: '1200',
      });
      const res = await fetch(`/api/document?${qs.toString()}`);
      if (!res.ok) throw new Error('Unable to load the full statute text.');
      const data = await res.json();
      setFullDoc(data);
    } catch (e) {
      setDocError(e.message);
    } finally {
      setLoadingDoc(false);
    }
  };

  const handleOpenXref = async (xref) => {
    if (!xref?.document_id || !xref?.key) return;
    setLoadingXref(true);
    setDocError(null);
    try {
      const qs = new URLSearchParams({
        document_id: xref.document_id,
        key: xref.key,
      });
      const res = await fetch(`/api/provision?${qs.toString()}`);
      if (!res.ok) throw new Error('Unable to load referenced provision.');
      const data = await res.json();
      setXrefStack((prev) => [...prev, data]);
    } catch (e) {
      setDocError(e.message);
    } finally {
      setLoadingXref(false);
    }
  };

  const renderContextWindow = () => {
    if (!fullDoc?.text) return null;
    const ws = fullDoc.window_start ?? 0;
    const relStart = Math.max(0, (active.char_start ?? 0) - ws);
    const relEnd = Math.min(fullDoc.text.length, Math.max(relStart, (active.char_end ?? 0) - ws));
    const before = fullDoc.text.slice(0, relStart);
    const target = fullDoc.text.slice(relStart, relEnd);
    const after = fullDoc.text.slice(relEnd);

    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4">
        <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500">
          <span>Surrounding Statutory Context</span>
          <span className="font-medium text-[#1E3A5F]">Highlighted section</span>
        </div>
        <div className="thin-scroll max-h-[440px] overflow-y-auto whitespace-pre-wrap font-serif text-[14.5px] leading-[1.8] text-slate-700">
          {ws > 0 && <span className="text-slate-400">...</span>}
          <span>{before}</span>
          <mark className="rounded bg-blue-100/80 px-0.5 text-slate-950 ring-1 ring-blue-300">
            {target}
          </mark>
          <span>{after}</span>
          {(fullDoc.window_end ?? fullDoc.text.length) < (fullDoc.char_count ?? fullDoc.text.length) && (
            <span className="text-slate-400">...</span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-xs">
      {xrefStack.length > 0 && (
        <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/80 px-5 py-2.5">
          <button
            type="button"
            onClick={() => setXrefStack((prev) => prev.slice(0, -1))}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-[#1E3A5F] hover:underline"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>
              {xrefStack.length > 1 ? 'Back to previous provision' : 'Back to retrieved passage'}
            </span>
          </button>
          <span className="text-[11px] text-slate-500">Cross-reference preview</span>
        </div>
      )}

      <div className="border-b border-slate-100 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium text-slate-500">
              <span className="rounded bg-[#1E3A5F]/10 px-1.5 py-0.5 font-semibold text-[#1E3A5F]">
                {formatDocType(active.document_type)}
              </span>
              {active.jurisdiction && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                  <MapPin className="h-2.5 w-2.5" />
                  {active.jurisdiction}
                </span>
              )}
              {active.year && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                  <Calendar className="h-2.5 w-2.5" />
                  {active.year}
                </span>
              )}
            </div>
            <h2 className="mt-2 text-base font-semibold leading-snug text-slate-900">
              {active.title}
            </h2>
            {active.label && (
              <p className="mt-1 text-xs font-medium text-[#1E3A5F]">{active.label}</p>
            )}
          </div>

          <button
            type="button"
            onClick={handleCopyCitation}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
          >
            {copiedCite ? (
              <Check className="h-3.5 w-3.5 text-emerald-600" />
            ) : (
              <Copy className="h-3.5 w-3.5 text-slate-400" />
            )}
            <span>{copiedCite ? 'Copied' : 'Copy Citation'}</span>
          </button>
        </div>
      </div>

      <div className="p-5">
        {fullDoc ? (
          renderContextWindow()
        ) : (
          <div className="thin-scroll max-h-[440px] overflow-y-auto whitespace-pre-wrap rounded-xl border border-slate-100 bg-slate-50/50 p-4 font-serif text-[14.5px] leading-[1.8] text-slate-800">
            {active.text}
          </div>
        )}

        {!fullDoc && (
          <div className="mt-3 flex items-center gap-3">
            <button
              type="button"
              onClick={handleLoadFullAct}
              disabled={loadingDoc}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-[#1E3A5F] transition hover:bg-slate-50"
            >
              {loadingDoc ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Maximize2 className="h-3.5 w-3.5" />
              )}
              <span>Read Surrounding Sections</span>
            </button>
            {docError && <span className="text-xs text-red-600">{docError}</span>}
          </div>
        )}

        {active.xref?.length > 0 && (
          <div className="mt-5 border-t border-slate-100 pt-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
                <GitBranch className="h-3.5 w-3.5 text-[#1E3A5F]" />
                <span>Related Cross-References</span>
              </div>
              {loadingXref && <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />}
            </div>
            <div className="flex flex-wrap gap-2">
              {active.xref.map((x, i) => (
                <button
                  key={`${x.document_id}-${x.key || i}-${x.direction}`}
                  type="button"
                  onClick={() => handleOpenXref(x)}
                  disabled={!x.key || loadingXref}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-left text-xs text-slate-700 transition hover:border-[#1E3A5F]/40 hover:bg-white disabled:cursor-default disabled:opacity-75"
                >
                  <span className="text-[10px] font-medium text-slate-400 uppercase">
                    {x.direction === 'cited_by' ? 'Cited by' : 'Refers to'}
                  </span>
                  <span className="font-medium text-[#1E3A5F]">{x.label}</span>
                  <span className="max-w-[200px] truncate text-slate-500">· {x.title}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
