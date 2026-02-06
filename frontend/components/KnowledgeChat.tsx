import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, DocumentChunk } from '../types';
import { ICONS } from '../constants';
import Uploader from './Uploader';
import { createChat, askChatStream, listDocuments, deleteDocument, DocListItem } from '../services/backend';

function mapCitations(citations: any[]): DocumentChunk[] {
  return (citations || []).map((c: any, i: number) => ({
    id: `cite_${i}_${c?.filename ?? 'doc'}`,
    docName: c?.filename ?? 'Unknown',
    content: c?.snippet ?? '',
    metadata: { section: `chunk ${c?.chunk_index ?? ''}`, timestamp: Date.now() }
  }));
}

function uniqueFileNames(citations: DocumentChunk[]): string[] {
  const seen = new Set<string>();
  return (citations || []).map(c => c.docName).filter(name => { if (seen.has(name)) return false; seen.add(name); return true; });
}

const KnowledgeChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatId, setChatId] = useState<string>('');
  const [documents, setDocuments] = useState<DocListItem[]>([]);
  const [docSearch, setDocSearch] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [resourcesOpenForId, setResourcesOpenForId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const loadDocs = async () => {
    try {
      const res = await listDocuments();
      setDocuments(res.documents || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const { chat_id } = await createChat('EchoMind Chat');
        setChatId(chat_id);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  useEffect(() => {
    if (!resourcesOpenForId) return;
    const close = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) setResourcesOpenForId(null);
    };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [resourcesOpenForId]);

  const send = async () => {
    const q = input.trim();
    if (!q || !chatId || busy) return;
    setBusy(true);
    const userMsg: ChatMessage = { id: `u_${Date.now()}`, role: 'user', content: q, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    const assistantId = `a_${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', citations: undefined, timestamp: Date.now() };
    setMessages(prev => [...prev, assistantMsg]);
    try {
      await askChatStream(chatId, q, {
        onChunk: (text) => {
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: m.content + text } : m));
        },
        onDone: (result) => {
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: result.answer, citations: mapCitations(result.citations) } : m));
        },
        onError: (err) => {
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: err?.message || 'Request failed' } : m));
        }
      });
    } catch (err: any) {
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: err?.message || 'Request failed' } : m));
    } finally {
      setBusy(false);
    }
  };

  const filteredDocs = docSearch.trim()
    ? documents.filter(d => d.filename.toLowerCase().includes(docSearch.trim().toLowerCase()))
    : documents;

  const handleDeleteDoc = async (doc: DocListItem) => {
    if (!window.confirm(`Remove "${doc.filename}" from the knowledge base? This cannot be undone.`)) return;
    setDeletingId(doc.id);
    try {
      await deleteDocument(doc.id);
      await loadDocs();
    } catch (e) {
      console.error(e);
      alert((e as Error)?.message || 'Failed to delete');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex h-full min-h-0 gap-4 md:gap-5">
      {/* Center: Chat */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <div className="px-4 py-3 sm:px-5 sm:py-4 border-b border-white/10 flex items-center gap-2 shrink-0">
          <div className="opacity-80"><ICONS.Chat className="w-5 h-5" /></div>
          <div className="font-semibold">Knowledge Chat</div>
          <div className="ml-auto text-xs opacity-60">{chatId ? 'Connected' : 'Connecting...'}</div>
        </div>

        <div className="flex-1 min-h-0 overflow-auto p-4 sm:p-5 space-y-4">
          {messages.length === 0 && (
            <div className="text-sm opacity-70 text-center py-8">Ask questions about your resources. I’ll use them when relevant.</div>
          )}
          {messages.map(m => (
            <div key={m.id} className={`rounded-2xl p-4 border ${m.role === 'user' ? 'bg-white/10 border-white/10 ml-8' : 'bg-black/20 border-white/10 mr-8'}`}>
              <div className="text-xs opacity-60 mb-2">{m.role === 'user' ? 'You' : 'EchoMind'}</div>
              <div className="text-sm whitespace-pre-wrap">{m.content}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 relative" ref={m.id === resourcesOpenForId ? popoverRef : undefined}>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setResourcesOpenForId(resourcesOpenForId === m.id ? null : m.id); }}
                    className="text-xs font-medium text-cyan-400 hover:text-cyan-300 border border-white/20 hover:border-white/30 rounded-lg px-3 py-1.5 transition-colors"
                  >
                    Resources
                  </button>
                  {resourcesOpenForId === m.id && (
                    <div className="absolute top-full left-0 mt-1 z-20 rounded-lg border border-white/20 bg-slate-900/98 shadow-xl py-2 min-w-[200px] max-w-[320px]">
                      {uniqueFileNames(m.citations).map((name, i) => (
                        <div key={i} className="px-3 py-1.5 text-xs truncate" title={name}>{name}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <div className="px-4 py-3 sm:px-5 sm:py-4 border-t border-white/10 flex gap-3 shrink-0">
          <input
            className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-sm outline-none focus:border-white/30"
            placeholder="Ask something..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') send(); }}
          />
          <button
            className="rounded-xl px-5 py-3 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50 transition-colors"
            onClick={send}
            disabled={busy || !chatId}
          >
            {busy ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>

      {/* Right: Resources sidebar */}
      <aside className="w-64 lg:w-72 shrink-0 flex flex-col min-h-0 rounded-2xl border border-white/10 bg-black/20 overflow-hidden">
        <div className="px-4 py-3 border-b border-white/10 shrink-0">
          <div className="font-semibold text-sm flex items-center gap-2">
            <ICONS.File className="w-4 h-4 opacity-80" />
            Resources
          </div>
        </div>
        <div className="p-3 border-b border-white/10 shrink-0">
          <Uploader onComplete={loadDocs} />
        </div>
        <div className="p-3 border-b border-white/10 shrink-0">
          <div className="relative">
            <ICONS.Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 opacity-50 pointer-events-none" />
            <input
              type="text"
              placeholder="Search resources..."
              value={docSearch}
              onChange={e => setDocSearch(e.target.value)}
              className="w-full rounded-lg bg-black/30 border border-white/10 pl-9 pr-3 py-2 text-sm outline-none focus:border-white/30"
            />
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-auto p-3">
          {filteredDocs.length === 0 && (
            <div className="text-xs opacity-60 py-4 text-center">
              {documents.length === 0 ? 'No resources yet. Upload a document above.' : 'No matches.'}
            </div>
          )}
          <ul className="space-y-1">
            {filteredDocs.map(doc => (
              <li
                key={doc.id}
                className="group flex items-center gap-2 py-2 px-2 rounded-lg hover:bg-white/5"
                title={doc.filename}
              >
                <span className="flex-1 text-xs truncate min-w-0">{doc.filename}</span>
                <button
                  type="button"
                  onClick={() => handleDeleteDoc(doc)}
                  disabled={deletingId === doc.id}
                  className="shrink-0 p-1 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors"
                  title="Remove from knowledge base"
                >
                  <ICONS.Trash className="w-4 h-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
};

export default KnowledgeChat;
