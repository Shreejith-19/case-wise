import { useState } from 'react';
import { MessageSquare, Plus, Trash2, Pencil, Check, X, PanelLeftClose } from 'lucide-react';

function formatRelativeTime(ts) {
  if (!ts) return '';
  const diff = Math.max(0, Date.now() / 1000 - ts);
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function ChatSidebar({
  open,
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onRenameChat,
  onDeleteChat,
  onClearChats,
  onClose,
}) {
  const [editingId, setEditingId] = useState(null);
  const [draftTitle, setDraftTitle] = useState('');

  if (!open) return null;

  const startEdit = (e, chat) => {
    e.stopPropagation();
    setEditingId(chat.id);
    setDraftTitle(chat.title);
  };

  const commitEdit = async (e, id) => {
    e.stopPropagation();
    const clean = draftTitle.trim();
    if (clean) {
      await onRenameChat(id, clean);
    }
    setEditingId(null);
  };

  return (
    <aside className="fixed inset-y-14 left-0 z-20 flex w-72 shrink-0 flex-col border-r border-slate-200 bg-white shadow-lg lg:static lg:h-[calc(100vh-3.5rem)] lg:shadow-none">
      <div className="flex items-center justify-between border-b border-slate-100 px-3.5 py-3">
        <span className="text-xs font-semibold tracking-wide text-slate-600 uppercase">
          Chat History
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onNewChat}
            className="inline-flex items-center gap-1 rounded-lg bg-[#1E3A5F] px-2.5 py-1 text-[11px] font-medium text-white transition hover:bg-[#172E4D]"
          >
            <Plus className="h-3 w-3" />
            <span>New</span>
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Hide chat history"
            className="rounded-lg p-1 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="thin-scroll flex-1 space-y-1 overflow-y-auto p-2.5">
        {chats.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <MessageSquare className="mx-auto h-6 w-6 text-slate-300" />
            <p className="mt-2 text-xs font-medium text-slate-500">No saved conversations</p>
            <p className="mt-1 text-[11px] text-slate-400">
              Your legal research sessions will be saved here automatically.
            </p>
          </div>
        ) : (
          chats.map((chat) => {
            const active = chat.id === activeChatId;
            const isEditing = editingId === chat.id;
            return (
              <div
                key={chat.id}
                onClick={() => !isEditing && onSelectChat(chat.id)}
                className={`group cursor-pointer rounded-xl border p-2.5 transition ${
                  active
                    ? 'border-[#1E3A5F]/40 bg-[#1E3A5F]/[0.05]'
                    : 'border-transparent hover:border-slate-200 hover:bg-slate-50'
                }`}
              >
                {isEditing ? (
                  <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                    <input
                      autoFocus
                      value={draftTitle}
                      onChange={(e) => setDraftTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitEdit(e, chat.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      className="min-w-0 flex-1 rounded border border-[#1E3A5F]/40 bg-white px-2 py-1 text-xs text-slate-900 focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={(e) => commitEdit(e, chat.id)}
                      className="rounded p-1 text-emerald-600 hover:bg-emerald-50"
                    >
                      <Check className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(null);
                      }}
                      className="rounded p-1 text-slate-400 hover:bg-slate-100"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex items-start justify-between gap-2">
                      <p
                        className={`line-clamp-2 text-xs font-medium leading-snug ${
                          active ? 'text-[#1E3A5F]' : 'text-slate-800'
                        }`}
                      >
                        {chat.title}
                      </p>
                      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition group-hover:opacity-100">
                        <button
                          type="button"
                          onClick={(e) => startEdit(e, chat)}
                          title="Rename chat"
                          className="rounded p-1 text-slate-400 hover:bg-slate-200/70 hover:text-slate-700"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteChat(chat.id);
                          }}
                          title="Delete chat"
                          className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    <div className="mt-1.5 flex items-center justify-between text-[10px] text-slate-400">
                      <span>
                        {chat.message_count || 0}{' '}
                        {(chat.message_count || 0) === 1 ? 'turn' : 'turns'}
                      </span>
                      <span>{formatRelativeTime(chat.updated_at)}</span>
                    </div>
                  </>
                )}
              </div>
            );
          })
        )}
      </div>

      {chats.length > 0 && (
        <div className="border-t border-slate-100 px-3 py-2.5">
          <button
            type="button"
            onClick={onClearChats}
            className="w-full rounded-lg border border-slate-200 px-2.5 py-1.5 text-[11px] font-medium text-slate-500 transition hover:border-red-200 hover:bg-red-50/60 hover:text-red-700"
          >
            Clear All History
          </button>
        </div>
      )}
    </aside>
  );
}
