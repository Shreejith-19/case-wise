import { useState, useEffect, useRef } from 'react';
import { AlertCircle, Sparkles } from 'lucide-react';
import Header from './components/Header';
import ChatSidebar from './components/ChatSidebar';
import QueryBar from './components/QueryBar';
import AnswerPanel from './components/AnswerPanel';
import StatuteReader from './components/StatuteReader';

const ACTIVE_CHAT_KEY = 'casewise_active_chat_id';

export default function App() {
  const [question, setQuestion] = useState('');
  const [jurisdiction, setJurisdiction] = useState('All');
  const [docType, setDocType] = useState('All');
  const [budget, setBudget] = useState(8000);
  const [loading, setLoading] = useState(false);
  const [errorBanner, setErrorBanner] = useState(null);

  const [health, setHealth] = useState(null);
  const [examples, setExamples] = useState([]);

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(
    () => localStorage.getItem(ACTIVE_CHAT_KEY) || null
  );
  const [turns, setTurns] = useState([]);
  const [activeTurnIdx, setActiveTurnIdx] = useState(0);
  const [selectedPassageIdx, setSelectedPassageIdx] = useState(0);
  const [showHistory, setShowHistory] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= 1024 : true)
  );

  const bottomRef = useRef(null);

  const refreshChats = async () => {
    try {
      const res = await fetch('/api/chats');
      if (!res.ok) return;
      const data = await res.json();
      setChats(data.chats || []);
    } catch {
      // Backend might be restarting; keep current list
    }
  };

  const loadChat = async (chatId) => {
    if (!chatId) {
      setActiveChatId(null);
      setTurns([]);
      localStorage.removeItem(ACTIVE_CHAT_KEY);
      return;
    }
    try {
      const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`);
      if (!res.ok) {
        localStorage.removeItem(ACTIVE_CHAT_KEY);
        setActiveChatId(null);
        setTurns([]);
        return;
      }
      const chat = await res.json();
      const msgs = chat.messages || [];
      setActiveChatId(chat.id);
      localStorage.setItem(ACTIVE_CHAT_KEY, chat.id);
      setTurns(msgs);
      setJurisdiction(chat.jurisdiction || 'All');
      setDocType(chat.doc_type || 'All');
      setBudget(chat.budget || 8000);
      setActiveTurnIdx(Math.max(0, msgs.length - 1));
      setSelectedPassageIdx(0);
      setErrorBanner(null);
    } catch {
      setErrorBanner('Unable to load saved conversation.');
    }
  };

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setHealth(d))
      .catch(() => {});

    fetch('/api/examples')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d?.examples && setExamples(d.examples))
      .catch(() => {});

    refreshChats();
    const savedId = localStorage.getItem(ACTIVE_CHAT_KEY);
    if (savedId) {
      loadChat(savedId);
    }
  }, []);

  const handleNewChat = () => {
    setActiveChatId(null);
    localStorage.removeItem(ACTIVE_CHAT_KEY);
    setTurns([]);
    setQuestion('');
    setErrorBanner(null);
    setActiveTurnIdx(0);
    setSelectedPassageIdx(0);
  };

  const handleRenameChat = async (chatId, title) => {
    const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (res.ok) {
      refreshChats();
    }
  };

  const handleDeleteChat = async (chatId) => {
    const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      if (chatId === activeChatId) {
        handleNewChat();
      }
      refreshChats();
    }
  };

  const handleClearChats = async () => {
    const res = await fetch('/api/chats', { method: 'DELETE' });
    if (res.ok) {
      handleNewChat();
      setChats([]);
    }
  };

  const submitQuestion = async (overrideQuestion = null) => {
    const q = (overrideQuestion ?? question).trim();
    if (!q) return;

    setLoading(true);
    setErrorBanner(null);
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          jurisdiction,
          doc_type: docType,
          budget,
          chat_id: activeChatId,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          res.status === 503
            ? data.message || 'Live KB or Ollama is not ready yet.'
            : data.error || 'Unable to complete your search right now.'
        );
      }
      if (data.chat_id) {
        setActiveChatId(data.chat_id);
        localStorage.setItem(ACTIVE_CHAT_KEY, data.chat_id);
      }
      setTurns((prev) => {
        const next = [...prev, data];
        setActiveTurnIdx(next.length - 1);
        return next;
      });
      setSelectedPassageIdx(0);
      setQuestion('');
      refreshChats();
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
    } catch (err) {
      setErrorBanner(
        err.message === 'Failed to fetch'
          ? 'Unable to reach the CaseWise API server.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const activeTurn = turns[activeTurnIdx] || null;
  const activePassage = activeTurn?.passages?.[selectedPassageIdx] || null;

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFC] text-slate-900">
      <Header
        historyCount={chats.length}
        showHistory={showHistory}
        onToggleHistory={() => setShowHistory((v) => !v)}
        hasResult={turns.length > 0}
        onReset={handleNewChat}
        health={health}
      />

      <div className="flex flex-1">
        <ChatSidebar
          open={showHistory}
          chats={chats}
          activeChatId={activeChatId}
          onSelectChat={loadChat}
          onNewChat={handleNewChat}
          onRenameChat={handleRenameChat}
          onDeleteChat={handleDeleteChat}
          onClearChats={handleClearChats}
          onClose={() => setShowHistory(false)}
        />

        <main className="mx-auto w-full max-w-[1360px] flex-1 px-4 py-6 sm:px-6">
          {turns.length === 0 ? (
            <div className="mx-auto flex max-w-3xl flex-col justify-center py-10 sm:py-16">
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
                budget={budget}
                setBudget={setBudget}
                loading={loading}
                onSubmit={() => submitQuestion()}
              />

              {errorBanner && (
                <div className="mx-auto mt-4 flex w-full max-w-2xl items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900">
                  <AlertCircle className="h-4 w-4 shrink-0 text-amber-700" />
                  <span>{errorBanner}</span>
                </div>
              )}

              {examples.length > 0 && (
                <div className="mx-auto mt-8 w-full max-w-2xl">
                  <div className="mb-2.5 flex items-center gap-1.5 px-1 text-[11px] font-semibold tracking-wide text-slate-400 uppercase">
                    <Sparkles className="h-3.5 w-3.5 text-[#1E3A5F]" />
                    <span>Sample Statutory Questions</span>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {examples.slice(0, 6).map((ex) => (
                      <button
                        key={ex.question_id}
                        type="button"
                        disabled={loading}
                        onClick={() => {
                          setQuestion(ex.question);
                          if (health?.mode === 'live') {
                            submitQuestion(ex.question);
                          }
                        }}
                        className="rounded-xl border border-slate-200 bg-white p-3 text-left transition hover:border-[#1E3A5F]/40 hover:shadow-xs disabled:opacity-50"
                      >
                        <span className="inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600 uppercase">
                          {ex.category.replace('_', ' ')}
                        </span>
                        <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-slate-700">
                          {ex.question}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="grid items-start gap-5 lg:grid-cols-12">
              <div className="space-y-6 lg:col-span-6">
                {turns.map((turn, tIdx) => (
                  <AnswerPanel
                    key={turn.id || `${turn.question}-${tIdx}`}
                    result={turn}
                    isTurnActive={tIdx === activeTurnIdx}
                    selectedIndex={selectedPassageIdx}
                    onSelectPassage={(pIdx) => {
                      setActiveTurnIdx(tIdx);
                      setSelectedPassageIdx(pIdx);
                    }}
                  />
                ))}

                {errorBanner && (
                  <div className="flex items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-3 text-xs text-amber-900">
                    <AlertCircle className="h-4 w-4 shrink-0 text-amber-700" />
                    <span>{errorBanner}</span>
                  </div>
                )}

                <div ref={bottomRef} className="pt-1">
                  <QueryBar
                    compact
                    isFollowUp
                    question={question}
                    setQuestion={setQuestion}
                    jurisdiction={jurisdiction}
                    setJurisdiction={setJurisdiction}
                    docType={docType}
                    setDocType={setDocType}
                    budget={budget}
                    setBudget={setBudget}
                    loading={loading}
                    onSubmit={() => submitQuestion()}
                  />
                </div>
              </div>

              <div className="lg:sticky lg:top-20 lg:col-span-6">
                <StatuteReader passage={activePassage} />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
