import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import Header from './components/Header';
import QueryBar from './components/QueryBar';
import AnswerPanel from './components/AnswerPanel';
import StatuteReader from './components/StatuteReader';

export default function App() {
  const [question, setQuestion] = useState('');
  const [jurisdiction, setJurisdiction] = useState('All');
  const [docType, setDocType] = useState('All');
  const [loading, setLoading] = useState(false);
  const [errorBanner, setErrorBanner] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedPassageIdx, setSelectedPassageIdx] = useState(0);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const openResult = (res) => {
    setResult(res);
    setSelectedPassageIdx(0);
    setErrorBanner(null);
    setHistory((prev) => {
      const filtered = prev.filter((item) => item.question !== res.question);
      return [res, ...filtered].slice(0, 15);
    });
  };

  const handleSubmit = async () => {
    const q = question.trim();
    if (!q) return;

    setLoading(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, jurisdiction, doc_type: docType, budget: 8000 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          res.status === 503
            ? 'The legal search service is currently unavailable. Please try again shortly.'
            : data.error || 'Unable to complete your search right now.'
        );
      }
      openResult(data);
    } catch (err) {
      setErrorBanner(
        err.message === 'Failed to fetch'
          ? 'Unable to reach the legal search service.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const activePassage = result?.passages?.[selectedPassageIdx] || null;

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFC] text-slate-900">
      <Header
        historyCount={history.length}
        showHistory={showHistory}
        onToggleHistory={() => setShowHistory((v) => !v)}
        hasResult={Boolean(result)}
        onReset={() => {
          setResult(null);
          setQuestion('');
          setErrorBanner(null);
        }}
      />

      <main className="mx-auto w-full max-w-[1360px] flex-1 px-4 py-6 sm:px-6">
        {showHistory && history.length > 0 && (
          <div className="mb-5 rounded-2xl border border-slate-200 bg-white p-3 shadow-xs">
            <div className="mb-2 flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
                Recent Searches
              </span>
              <button
                type="button"
                onClick={() => setShowHistory(false)}
                className="text-[11px] text-slate-400 hover:text-slate-700"
              >
                Close
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {history.map((item, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => {
                    setQuestion(item.question);
                    setResult(item);
                    setSelectedPassageIdx(0);
                  }}
                  className="max-w-xs truncate rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-left text-xs text-slate-700 transition hover:border-[#1E3A5F]/40 hover:bg-white"
                >
                  {item.question}
                </button>
              ))}
            </div>
          </div>
        )}

        {!result ? (
          <div className="mx-auto flex max-w-3xl flex-col justify-center py-16 sm:py-24">
            <div className="mb-7 text-center">
              <h1 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
                Indian Statutory Legal Research
              </h1>
              <p className="mx-auto mt-2.5 max-w-xl text-sm leading-relaxed text-slate-500">
                Search Central and State Acts, Rules, and Regulations with direct citations and
                verified statutory text.
              </p>
            </div>

            <QueryBar
              question={question}
              setQuestion={setQuestion}
              jurisdiction={jurisdiction}
              setJurisdiction={setJurisdiction}
              docType={docType}
              setDocType={setDocType}
              loading={loading}
              onSubmit={handleSubmit}
            />

            {errorBanner && (
              <div className="mx-auto mt-4 flex w-full max-w-2xl items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900">
                <AlertCircle className="h-4 w-4 shrink-0 text-amber-700" />
                <span>{errorBanner}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-5">
            <QueryBar
              compact
              question={question}
              setQuestion={setQuestion}
              jurisdiction={jurisdiction}
              setJurisdiction={setJurisdiction}
              docType={docType}
              setDocType={setDocType}
              loading={loading}
              onSubmit={handleSubmit}
            />

            {errorBanner && (
              <div className="flex items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900">
                <AlertCircle className="h-4 w-4 shrink-0 text-amber-700" />
                <span>{errorBanner}</span>
              </div>
            )}

            <div className="grid items-start gap-5 lg:grid-cols-12">
              <div className="lg:col-span-6">
                <AnswerPanel
                  result={result}
                  selectedIndex={selectedPassageIdx}
                  onSelectPassage={setSelectedPassageIdx}
                />
              </div>
              <div className="lg:sticky lg:top-20 lg:col-span-6">
                <StatuteReader passage={activePassage} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
