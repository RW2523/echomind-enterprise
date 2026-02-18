import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, DocumentChunk, AppSettings } from '../types';
import { ICONS } from '../constants';
import Uploader from './Uploader';
import { askChatStream, listDocuments, deleteDocument, listTranscripts, DocListItem, TranscriptListItem } from '../services/backend';
import type { UseKnowledgeChatReturn } from '../hooks/useKnowledgeChat';

interface KnowledgeChatProps {
  settings?: AppSettings | null;
  knowledgeChat: UseKnowledgeChatReturn;
}

/** Styled Markdown renderer for assistant messages: headings, lists, bold, code, blockquotes. Lists use list-outside + pl to avoid layout break with bullets. */
const markdownComponents: React.ComponentProps<typeof ReactMarkdown>['components'] = {
  h1: ({ children }) => <h1 className="text-lg font-bold mt-4 mb-2 first:mt-0 text-white">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-bold mt-3 mb-1.5 text-white/95">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-2.5 mb-1 text-white/90">{children}</h3>,
  h4: ({ children }) => <h4 className="text-sm font-semibold mt-2 mb-1 text-white/85">{children}</h4>,
  p: ({ children }) => <p className="text-sm mb-2 last:mb-0 leading-relaxed text-white/90">{children}</p>,
  ul: ({ children }) => <ul className="list-disc list-outside pl-5 mb-2 space-y-1 text-sm text-white/90 [&_ul]:pl-5 [&_ul]:mt-1 [&_ol]:pl-5 [&_ol]:mt-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-outside pl-5 mb-2 space-y-1 text-sm text-white/90 [&_ul]:pl-5 [&_ul]:mt-1 [&_ol]:pl-5 [&_ol]:mt-1">{children}</ol>,
  li: ({ children }) => <li className="pl-1 [&>p]:my-0 first:[&>p]:mt-0 last:[&>p]:mb-0">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
  em: ({ children }) => <em className="italic text-white/95">{children}</em>,
  code: ({ className, children, ...props }) => {
    const isBlock = className?.includes('language-');
    if (isBlock) {
      return <code className="block font-mono text-white/90" {...props}>{children}</code>;
    }
    return <code className="rounded bg-white/10 px-1.5 py-0.5 text-xs font-mono text-cyan-200" {...props}>{children}</code>;
  },
  pre: ({ children }) => <pre className="rounded-lg bg-black/40 border border-white/10 p-3 my-2 overflow-x-auto text-xs">{children}</pre>,
  blockquote: ({ children }) => <blockquote className="border-l-2 border-cyan-500/60 pl-3 my-2 text-white/80 italic">{children}</blockquote>,
  hr: () => <hr className="border-white/10 my-3" />,
};

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

const KnowledgeChat: React.FC<KnowledgeChatProps> = ({ settings, knowledgeChat }) => {
  const { messages, setMessages, chatId, clearChat } = knowledgeChat;
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [documents, setDocuments] = useState<DocListItem[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptListItem[]>([]);
  const [transcriptsLoading, setTranscriptsLoading] = useState(false);
  const [transcriptsError, setTranscriptsError] = useState<string | null>(null);
  const [resourceTab, setResourceTab] = useState<'resources' | 'transcripts'>('resources');
  const [docSearch, setDocSearch] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [resourcesOpenForId, setResourcesOpenForId] = useState<string | null>(null);
  const [expandedTranscriptId, setExpandedTranscriptId] = useState<string | null>(null);
  const [resourcesPanelOpen, setResourcesPanelOpen] = useState(false);
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

  const loadTranscripts = async () => {
    setTranscriptsError(null);
    setTranscriptsLoading(true);
    try {
      const res = await listTranscripts();
      setTranscripts(res.transcripts || []);
    } catch (e) {
      console.error(e);
      setTranscriptsError((e as Error)?.message || 'Failed to load transcripts');
      setTranscripts([]);
    } finally {
      setTranscriptsLoading(false);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  useEffect(() => {
    if (resourceTab === 'transcripts') loadTranscripts();
  }, [resourceTab]);

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
      }, {
        persona: settings?.persona ?? undefined,
        context_window: settings?.contextWindow ?? undefined,
        advanced_rag: settings?.advancedRag ?? undefined,
        use_knowledge_base: true,
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

  const resourcesAside = (
    <aside className="w-full md:w-64 lg:w-72 h-full flex flex-col min-h-0 rounded-2xl border border-white/10 bg-[#080b14] overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 shrink-0 flex items-center justify-between">
        <div className="font-semibold text-sm flex items-center gap-2">
          <ICONS.File className="w-4 h-4 opacity-80" />
          Resources
        </div>
        <button type="button" onClick={() => setResourcesPanelOpen(false)} className="md:hidden p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation" aria-label="Close resources">
          <ICONS.Close className="w-5 h-5" />
        </button>
      </div>
      <div className="px-4 py-3 border-b border-white/10 shrink-0">
        <div className="flex rounded-lg bg-white/5 p-0.5 border border-white/10">
          <button type="button" onClick={() => setResourceTab('resources')} className={`flex-1 py-2 rounded-md text-xs font-medium transition-colors touch-manipulation ${resourceTab === 'resources' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white'}`}>Resources</button>
          <button type="button" onClick={() => setResourceTab('transcripts')} className={`flex-1 py-2 rounded-md text-xs font-medium transition-colors touch-manipulation ${resourceTab === 'transcripts' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white'}`}>Transcripts</button>
        </div>
      </div>
      {resourceTab === 'resources' && (
        <>
          <div className="p-3 border-b border-white/10 shrink-0">
            <Uploader onComplete={loadDocs} />
          </div>
          <div className="p-3 border-b border-white/10 shrink-0">
            <div className="relative">
              <ICONS.Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 opacity-50 pointer-events-none" />
              <input type="text" placeholder="Search resources..." value={docSearch} onChange={e => setDocSearch(e.target.value)} className="w-full rounded-lg bg-black/30 border border-white/10 pl-9 pr-3 py-2 text-sm outline-none focus:border-white/30" />
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-auto p-3">
            {filteredDocs.length === 0 && (
              <div className="text-xs opacity-60 py-4 text-center">{documents.length === 0 ? 'No resources yet. Upload a document above.' : 'No matches.'}</div>
            )}
            <ul className="space-y-1">
              {filteredDocs.map(doc => (
                <li key={doc.id} className="group flex items-center gap-2 py-2 px-2 rounded-lg hover:bg-white/5">
                  <span className="flex-1 text-xs truncate min-w-0">{doc.filename}</span>
                  <button type="button" onClick={() => handleDeleteDoc(doc)} disabled={deletingId === doc.id} className="shrink-0 p-2 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors touch-manipulation" title="Remove from knowledge base">
                    <ICONS.Trash className="w-4 h-4" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
      {resourceTab === 'transcripts' && (
        <div className="flex-1 min-h-0 flex flex-col p-3">
          <div className="shrink-0 flex items-center justify-between gap-2 mb-2">
            <span className="text-xs text-slate-500">From Live Transcript · tags = main keywords</span>
            <button type="button" onClick={loadTranscripts} disabled={transcriptsLoading} className="text-xs text-cyan-400 hover:text-cyan-300 disabled:opacity-50 touch-manipulation py-1">{transcriptsLoading ? 'Loading…' : 'Refresh'}</button>
          </div>
          {transcriptsError && <div className="shrink-0 text-xs text-red-400 mb-2 py-1">{transcriptsError}</div>}
          <div className="flex-1 min-h-0 overflow-auto">
            {transcriptsLoading && transcripts.length === 0 && <div className="text-xs opacity-60 py-4 text-center">Loading transcripts…</div>}
            {!transcriptsLoading && transcripts.length === 0 && !transcriptsError && <div className="text-xs opacity-60 py-4 text-center">No transcripts yet. Save from Live Transcript.</div>}
            {transcripts.length > 0 && (
              <ul className="space-y-1">
                {transcripts.map(t => (
                  <li key={t.id} className="rounded-lg border border-white/10 overflow-hidden">
                    <button type="button" onClick={() => setExpandedTranscriptId(expandedTranscriptId === t.id ? null : t.id)} className="w-full text-left py-2 px-2 hover:bg-white/5 flex flex-col gap-1.5 touch-manipulation">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="flex-1 text-xs truncate min-w-0" title={t.title}>{t.title}</span>
                        <span className="text-slate-500 text-[10px] shrink-0">{t.created_at?.slice(0, 10)}</span>
                        <span className={`shrink-0 transition-transform ${expandedTranscriptId === t.id ? 'rotate-180' : ''}`}>▼</span>
                      </div>
                      {(t.tags || []).length > 0 && (
                        <div className="flex flex-wrap gap-1 min-w-0">
                          {(t.tags || []).slice(0, 6).map((tag, i) => (
                            <span key={i} className="inline-block px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-[10px] truncate max-w-[100px]" title={tag}>{tag}</span>
                          ))}
                          {(t.tags || []).length > 6 && <span className="text-[10px] text-slate-500">+{(t.tags || []).length - 6}</span>}
                        </div>
                      )}
                    </button>
                    {expandedTranscriptId === t.id && (
                      <div className="px-3 py-2 bg-black/30 border-t border-white/10">
                        <p className="text-[10px] text-slate-500 mb-1">Tags</p>
                        <div className="flex flex-wrap gap-1">
                          {(t.tags || []).length === 0 ? <span className="text-xs text-slate-500">No tags</span> : (t.tags || []).map((tag, i) => (
                            <span key={i} className="inline-block px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-xs">{tag}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </aside>
  );

  return (
    <div className="flex flex-col md:flex-row h-full min-h-0 gap-3 md:gap-5">
      {/* Chat - first on mobile and desktop */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <div className="px-3 sm:px-4 py-3 sm:py-4 border-b border-white/10 flex items-center gap-2 shrink-0">
          <div className="opacity-80 shrink-0"><ICONS.Chat className="w-5 h-5" /></div>
          <div className="font-semibold truncate min-w-0 flex-1">Knowledge Chat</div>
          <button type="button" onClick={() => setResourcesPanelOpen(true)} className="md:hidden shrink-0 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center" aria-label="Open resources">
            <ICONS.File className="w-5 h-5" />
          </button>
          <button type="button" onClick={() => clearChat()} className="shrink-0 p-2 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center" aria-label="Clear chat" title="Clear chat">
            <ICONS.Trash className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-auto p-4 sm:p-5 space-y-4">
          {messages.length === 0 && (
            <div className="text-sm opacity-70 text-center py-8">Ask questions about your resources. I’ll use them when relevant.</div>
          )}
          {messages.map(m => (
            <div key={m.id} className={`rounded-2xl p-4 border ${m.role === 'user' ? 'bg-white/10 border-white/10 ml-8' : 'bg-black/20 border-white/10 mr-8'}`}>
              <div className="text-xs opacity-60 mb-2">{m.role === 'user' ? 'You' : 'EchoMind'}</div>
              {m.role === 'assistant' && !m.content && busy && messages[messages.length - 1]?.id === m.id ? (
                <div className="text-sm text-white/60 flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.2s]" />
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.4s]" />
                  <span className="ml-1">Thinking...</span>
                </div>
              ) : m.role === 'assistant' && m.content ? (
                <div className="text-sm markdown-response break-words overflow-hidden min-w-0">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {m.content}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="text-sm whitespace-pre-wrap">{m.content}</div>
              )}
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

        <div className="px-3 sm:px-5 py-3 sm:py-4 border-t border-white/10 flex gap-2 sm:gap-3 shrink-0">
          <input
            type="text"
            className="flex-1 min-w-0 rounded-xl bg-black/30 border border-white/10 px-4 py-3 min-h-[44px] text-sm outline-none focus:border-white/30"
            placeholder="Ask something..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') send(); }}
          />
          <button
            type="button"
            className="rounded-xl px-5 py-3 min-h-[44px] text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50 transition-colors touch-manipulation shrink-0"
            onClick={send}
            disabled={busy || !chatId}
          >
            {busy ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>

      {/* Mobile: resources panel overlay */}
      {resourcesPanelOpen && (
        <div className="fixed inset-0 z-40 md:hidden bg-black/60 backdrop-blur-sm" onClick={() => setResourcesPanelOpen(false)} aria-hidden />
      )}
      {/* Resources: slide-in drawer on mobile, sidebar on desktop */}
      <div className={`fixed md:relative inset-y-0 right-0 z-50 md:z-auto w-full max-w-sm md:max-w-none flex flex-col md:flex-none h-full md:h-auto transform ${resourcesPanelOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0'} transition-transform duration-200 md:transition-none md:w-64 lg:w-72 md:shrink-0`}>
        {resourcesAside}
      </div>
    </div>
  );
};

export default KnowledgeChat;
