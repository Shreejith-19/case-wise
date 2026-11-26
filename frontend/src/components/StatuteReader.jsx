import { useState, useEffect } from 'react';
import { BookOpen, Copy, Check, GitBranch, MapPin, Calendar, Loader2, Maximize2 } from 'lucide-react';
import { formatDocType } from '../utils/formatters';

export default function StatuteReader({ passage }) {
  const [copiedCite, setCopiedCite] = useState(false);
  const [fullDoc, setFullDoc] = useState(null);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [docError, setDocError] = useState(null);

  useEffect(() => {
    setFullDoc(null);
    setDocError(null);
  }, [passage?.document_id, passage?.char_start]);

  if (!passage) {
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
    const cite = `${passage.title}${passage.label ? `, ${passage.label}` : ''}`;
    await navigator.clipboard.writeText(cite);
    setCopiedCite(true);
    setTimeout(() => setCopiedCite(false), 1800);
  };

  const handleLoadFullAct = async () => {
    setLoadingDoc(true);
    setDocError(null);
    try {
      const res = await fetch(`/api/document?id=${encodeURIComponent(passage.document_id)}`);
      if (!res.ok) throw new Error('Unable to load the full statute text.');
      const data = await res.json();
      setFullDoc(data);
    } catch (e) {
      setDocError(e.message);
    } finally {
      setLoadingDoc(false);
    }
  };

  const renderContextWindow = () => {
    if (!fullDoc?.text) return null;
    const pad = 1200;
    const start = Math.max(0, passage.char_start - pad);
    const end = Math.min(fullDoc.text.length, passage.char_end + pad);
    const before = fullDoc.text.slice(start, passage.char_start);
    const target = fullDoc.text.slice(passage.char_start, passage.char_end);
    const after = fullDoc.text.slice(passage.char_end, end);

    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4">
        <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500">
          <span>Surrounding Statutory Context</span>
          <span className="font-medium text-[#1E3A5F]">Highlighted section</span>
        </div>
        <div className="thin-scroll max-h-[440px] overflow-y-auto whitespace-pre-wrap font-serif text-[14.5px] leading-[1.8] text-slate-700">
          {start > 0 && <span className="text-slate-400">...</span>}
          <span>{before}</span>
          <mark className="rounded bg-blue-100/80 px-0.5 text-slate-950 ring-1 ring-blue-300">
            {target}
          </mark>
          <span>{after}</span>
          {end < fullDoc.text.length && <span className="text-slate-400">...</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-xs">
      <div className="border-b border-slate-100 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium text-slate-500">
              <span className="rounded bg-[#1E3A5F]/10 px-1.5 py-0.5 font-semibold text-[#1E3A5F]">
                {formatDocType(passage.document_type)}
              </span>
              {passage.jurisdiction && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                  <MapPin className="h-2.5 w-2.5" />
                  {passage.jurisdiction}
                </span>
              )}
              {passage.year && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                  <Calendar className="h-2.5 w-2.5" />
                  {passage.year}
                </span>
              )}
            </div>
            <h2 className="mt-2 text-base font-semibold leading-snug text-slate-900">
              {passage.title}
            </h2>
            {passage.label && (
              <p className="mt-1 text-xs font-medium text-[#1E3A5F]">{passage.label}</p>
            )}
          </div>

          <button
            type="button"
            onClick={handleCopyCitation}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
          >
            {copiedCite ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5 text-slate-400" />}
            <span>{copiedCite ? 'Copied' : 'Copy Citation'}</span>
          </button>
        </div>
      </div>

      <div className="p-5">
        {fullDoc ? (
          renderContextWindow()
        ) : (
          <div className="thin-scroll max-h-[440px] overflow-y-auto whitespace-pre-wrap rounded-xl border border-slate-100 bg-slate-50/50 p-4 font-serif text-[14.5px] leading-[1.8] text-slate-800">
            {passage.text}
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
              {loadingDoc ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Maximize2 className="h-3.5 w-3.5" />}
              <span>Read Surrounding Sections</span>
            </button>
            {docError && <span className="text-xs text-red-600">{docError}</span>}
          </div>
        )}

        {passage.xref?.length > 0 && (
          <div className="mt-5 border-t border-slate-100 pt-4">
            <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
              <GitBranch className="h-3.5 w-3.5 text-[#1E3A5F]" />
              <span>Related Cross-References</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {passage.xref.map((x, i) => (
                <div
                  key={i}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700"
                >
                  <span className="text-[10px] font-medium text-slate-400 uppercase">
                    {x.direction === 'cited_by' ? 'Cited by' : 'Refers to'}
                  </span>
                  <span className="font-medium text-[#1E3A5F]">{x.label}</span>
                  <span className="max-w-[200px] truncate text-slate-500">· {x.title}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
