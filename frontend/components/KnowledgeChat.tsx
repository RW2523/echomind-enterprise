import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, DocumentChunk, AppSettings } from '../types';
import { ICONS } from '../constants';
import Uploader from './Uploader';
import { askChatStream, listDocuments, deleteDocument, listTranscripts, deleteTranscript, getTranscript, DocListItem, TranscriptListItem, TranscriptDetail, type SourceOptions } from '../services/backend';
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

const CITATION_DEBUG = (import.meta as { env?: Record<string, string> }).env?.VITE_CITATION_DEBUG === '1';

function mapCitations(citations: any[]): DocumentChunk[] {
  if (CITATION_DEBUG) {
    console.log('[Citations] mapCitations input:', { raw: citations, count: (citations || []).length });
  }
  return (citations || []).map((c: any, i: number) => ({
    id: `cite_${i}_${c?.filename ?? 'doc'}`,
    docName: c?.filename ?? 'Unknown document',
    content: c?.snippet ?? '',
    metadata: {
      section: c?.section ?? undefined,
      sectionPath: c?.section_path ?? undefined,
      pageNumber: c?.page_number ?? undefined,
      score: typeof c?.score === 'number' ? c.score : undefined,
      docType: c?.doc_type ?? undefined,
      chunkIndex: c?.chunk_index ?? undefined,
      docId: c?.doc_id ?? undefined,
      timestamp: Date.now(),
    },
  }));
}

function uniqueFileNames(citations: DocumentChunk[]): string[] {
  const seen = new Set<string>();
  return (citations || []).map(c => c.docName).filter(name => { if (seen.has(name)) return false; seen.add(name); return true; });
}

/** Relevance score 0–1 → percentage label with colour class */
function scoreLabel(score?: number): { text: string; cls: string } | null {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const cls = pct >= 80 ? 'text-emerald-400' : pct >= 60 ? 'text-cyan-400' : pct >= 40 ? 'text-amber-400' : 'text-slate-400';
  return { text: `${pct}%`, cls };
}

/** Doc-type badge colour */
function docTypeBadge(docType?: string): { label: string; cls: string } | null {
  if (!docType) return null;
  const map: Record<string, { label: string; cls: string }> = {
    book: { label: 'Book', cls: 'bg-violet-500/20 text-violet-300' },
    glossary: { label: 'Glossary', cls: 'bg-teal-500/20 text-teal-300' },
    faq: { label: 'FAQ', cls: 'bg-sky-500/20 text-sky-300' },
    transcript: { label: 'Transcript', cls: 'bg-amber-500/20 text-amber-300' },
  };
  return map[docType.toLowerCase()] ?? { label: docType, cls: 'bg-white/10 text-slate-300' };
}

// ── PDF prefetch helper (loads into browser cache for instant preview) ────────
const API_BASE = (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE || '';
const _prefetchCache = new Set<string>();
function prefetchPdf(docId: string, _pageNumber?: number): void {
  if (_prefetchCache.has(docId)) return;
  _prefetchCache.add(docId);
  const url = `${API_BASE}/api/docs/${docId}/file`;
  fetch(url, { method: 'GET', credentials: 'same-origin' }).catch(() => {}); // Fire-and-forget; browser caches response
}

// ── Chunk Citation Modal ──────────────────────────────────────────────────────

interface ChunkModalProps {
  citations: DocumentChunk[];
  onClose: () => void;
}

const ChunkCitationModal: React.FC<ChunkModalProps> = ({ citations, onClose }) => {
  const [selected, setSelected] = useState<DocumentChunk | null>(null);

  // Prefetch PDFs when modal opens so "View in Document" loads instantly in new tab
  useEffect(() => {
    const docIds = new Set<string>();
    for (const c of citations) {
      if (c.metadata?.docId && c.metadata?.docType?.toLowerCase() !== 'transcript') {
        docIds.add(c.metadata.docId);
      }
    }
    for (const docId of docIds) {
      prefetchPdf(docId);
    }
  }, [citations]);

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center p-3 sm:p-4 bg-black/75 backdrop-blur-sm"
      onClick={() => { if (selected) setSelected(null); else onClose(); }}
    >
      <div
        className="w-full max-w-2xl max-h-[88vh] flex flex-col rounded-2xl border border-white/20 bg-[#0c1220] shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-white/10 shrink-0 flex items-center gap-3">
          {selected ? (
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors shrink-0"
              aria-label="Back to list"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          ) : (
            <div className="w-5 h-5 shrink-0 opacity-70">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
          )}
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm text-white truncate">
              {selected ? 'Chunk Preview' : `Sources  ·  ${citations.length} chunk${citations.length !== 1 ? 's' : ''}`}
            </h3>
            {selected && (
              <p className="text-[11px] text-slate-500 truncate mt-0.5">{selected.docName}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors shrink-0"
            aria-label="Close"
          >
            <ICONS.Close className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 min-h-0 overflow-auto">
          {selected ? (
            /* ── Chunk Detail / Preview ── */
            <div className="p-4 space-y-4">
              {/* Metadata strip */}
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2.5">
                {/* Document */}
                <div className="flex items-start gap-2">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">File</span>
                  <span className="text-xs text-white/90 break-all">{selected.docName}</span>
                </div>
                {/* Section path */}
                {selected.metadata.sectionPath && (
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">Path</span>
                    <span className="text-xs text-cyan-300/90 font-mono break-all leading-relaxed">{selected.metadata.sectionPath}</span>
                  </div>
                )}
                {/* Section title (when different from path) */}
                {selected.metadata.section && !selected.metadata.sectionPath && (
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">Section</span>
                    <span className="text-xs text-white/80">{selected.metadata.section}</span>
                  </div>
                )}
                {/* Row: page + relevance + type */}
                <div className="flex items-center flex-wrap gap-3">
                  {selected.metadata.pageNumber != null && (
                    <div className="flex items-center gap-1.5">
                      <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6M5 6h14M5 10h14" />
                      </svg>
                      <span className="text-[11px] text-slate-300">Page <span className="font-semibold text-white">{selected.metadata.pageNumber}</span></span>
                    </div>
                  )}
                  {(() => { const s = scoreLabel(selected.metadata.score); return s ? (
                    <div className="flex items-center gap-1.5">
                      <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span className={`text-[11px] font-semibold ${s.cls}`}>{s.text} relevance</span>
                    </div>
                  ) : null; })()}
                  {(() => { const b = docTypeBadge(selected.metadata.docType); return b ? (
                    <span className={`inline-block px-2 py-0.5 rounded-md text-[10px] font-medium ${b.cls}`}>{b.label}</span>
                  ) : null; })()}
                  {selected.metadata.chunkIndex != null && (
                    <span className="text-[10px] text-slate-500">chunk #{selected.metadata.chunkIndex}</span>
                  )}
                </div>
              </div>

              {/* Chunk text */}
              <div className="rounded-xl border border-cyan-500/20 bg-cyan-950/20 p-4">
                <p className="text-[10px] font-medium uppercase tracking-wider text-cyan-600 mb-2.5">Retrieved Text</p>
                <p className="text-sm text-white/85 leading-relaxed whitespace-pre-wrap font-mono">
                  {selected.content || <span className="text-slate-500 italic">No text preview available.</span>}
                </p>
              </div>

              {/* View in Document — opens PDF in new tab at the cited page */}
              {selected.metadata.docId && selected.metadata.docType?.toLowerCase() !== 'transcript' && (
                <a
                  href={`${API_BASE}/api/docs/${selected.metadata.docId}/file#page=${selected.metadata.pageNumber ?? 1}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  onMouseEnter={() => prefetchPdf(selected.metadata.docId!, selected.metadata.pageNumber ?? 1)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 hover:text-white text-sm font-medium transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  View in Document{selected.metadata.pageNumber != null ? ` — Page ${selected.metadata.pageNumber}` : ''}
                </a>
              )}
            </div>
          ) : (
            /* ── Chunk List ── */
            <ul className="divide-y divide-white/[0.06]">
              {citations.map((c, i) => {
                const sc = scoreLabel(c.metadata.score);
                const badge = docTypeBadge(c.metadata.docType);
                return (
                  <li key={c.id}>
                    <button
                      type="button"
                      onClick={() => setSelected(c)}
                      onMouseEnter={() => c.metadata?.docId && c.metadata?.docType?.toLowerCase() !== 'transcript' && prefetchPdf(c.metadata.docId, c.metadata.pageNumber ?? 1)}
                      className="w-full text-left px-4 py-3.5 hover:bg-white/[0.04] transition-colors group"
                    >
                      <div className="flex items-start gap-3">
                        {/* Index bubble */}
                        <span className="shrink-0 w-6 h-6 rounded-full bg-white/10 flex items-center justify-center text-[10px] font-bold text-slate-400 mt-px">{i + 1}</span>

                        <div className="flex-1 min-w-0 space-y-1">
                          {/* Filename */}
                          <div className="text-xs font-medium text-white/90 truncate">{c.docName}</div>

                          {/* Section path or section */}
                          {(c.metadata.sectionPath || c.metadata.section) && (
                            <div className="text-[11px] text-cyan-400/80 font-mono truncate leading-snug">
                              {c.metadata.sectionPath || c.metadata.section}
                            </div>
                          )}

                          {/* Meta row */}
                          <div className="flex items-center flex-wrap gap-x-3 gap-y-1 mt-1">
                            {c.metadata.pageNumber != null && (
                              <span className="text-[10px] text-slate-400">p. {c.metadata.pageNumber}</span>
                            )}
                            {sc && (
                              <span className={`text-[10px] font-medium ${sc.cls}`}>{sc.text} match</span>
                            )}
                            {badge && (
                              <span className={`inline-block px-1.5 py-0 rounded text-[10px] font-medium ${badge.cls}`}>{badge.label}</span>
                            )}
                          </div>

                          {/* Snippet */}
                          {c.content && (
                            <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed mt-0.5">
                              {c.content}
                            </p>
                          )}
                        </div>

                        {/* Arrow */}
                        <svg className="shrink-0 w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors mt-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Footer hint */}
        {!selected && (
          <div className="px-4 py-2 border-t border-white/[0.06] shrink-0">
            <p className="text-[10px] text-slate-600 text-center">Click any chunk to preview its content · click "View in Document" to open the page in a new tab</p>
          </div>
        )}
      </div>
    </div>
  );
};

const DEFAULT_SOURCE_OPTIONS: SourceOptions = {
  transcript: true,
  document: true,
  general: true,
};

const KnowledgeChat: React.FC<KnowledgeChatProps> = ({ settings, knowledgeChat }) => {
  const { messages, setMessages, chatId, newChat, loadChats } = knowledgeChat;
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [sourceOptions, setSourceOptions] = useState<SourceOptions>(DEFAULT_SOURCE_OPTIONS);
  const [documents, setDocuments] = useState<DocListItem[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptListItem[]>([]);
  const [transcriptsLoading, setTranscriptsLoading] = useState(false);
  const [transcriptsError, setTranscriptsError] = useState<string | null>(null);
  const [resourceTab, setResourceTab] = useState<'resources' | 'transcripts'>('resources');
  const [docSearch, setDocSearch] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [resourcesOpenForId, setResourcesOpenForId] = useState<string | null>(null);
  const [expandedTranscriptId, setExpandedTranscriptId] = useState<string | null>(null);
  const [previewTranscript, setPreviewTranscript] = useState<TranscriptDetail | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [resourcesPanelOpen, setResourcesPanelOpen] = useState(false);
  // Citation chunk modal: which message's citations are being shown
  const [chunkModalForId, setChunkModalForId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  /** Stream accumulator + rAF: avoids React 18 batching many chunks into one paint when TCP delivers a burst. */
  const streamBufRef = useRef('');
  const streamRafRef = useRef<number | null>(null);

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
    const onAdded = () => { if (resourceTab === 'transcripts') loadTranscripts(); };
    window.addEventListener('echomind:transcripts-added', onAdded);
    return () => window.removeEventListener('echomind:transcripts-added', onAdded);
  }, [resourceTab]);

  // Close citation modal on Escape key
  useEffect(() => {
    if (!chunkModalForId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setChunkModalForId(null); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [chunkModalForId]);

  const toggleSource = (key: keyof SourceOptions) => {
    setSourceOptions(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const send = async () => {
    const q = input.trim();
    if (!q || !chatId || busy) return;
    streamBufRef.current = '';
    if (streamRafRef.current != null) {
      cancelAnimationFrame(streamRafRef.current);
      streamRafRef.current = null;
    }
    setBusy(true);
    const userMsg: ChatMessage = { id: `u_${Date.now()}`, role: 'user', content: q, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    const assistantId = `a_${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', citations: undefined, timestamp: Date.now() };
    setMessages(prev => [...prev, assistantMsg]);

    const flushStreamToMessages = () => {
      streamRafRef.current = null;
      const delta = streamBufRef.current;
      streamBufRef.current = '';
      if (!delta) return;
      setMessages(prev => prev.map(m => (m.id === assistantId ? { ...m, content: m.content + delta } : m)));
    };

    try {
      await askChatStream(chatId, q, {
        onChunk: (text) => {
          streamBufRef.current += text;
          if (streamRafRef.current == null) {
            streamRafRef.current = requestAnimationFrame(flushStreamToMessages);
          }
        },
        onDone: (result) => {
          if (streamRafRef.current != null) {
            cancelAnimationFrame(streamRafRef.current);
            streamRafRef.current = null;
          }
          streamBufRef.current = '';
          if (CITATION_DEBUG) {
            console.log('[Citations] onDone received:', { answerLen: result.answer?.length, citations: result.citations, count: (result.citations || []).length });
          }
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: result.answer, citations: mapCitations(result.citations) } : m));
          loadChats(); // refresh sidebar so chat title updates from first message
        },
        onError: (err) => {
          if (streamRafRef.current != null) {
            cancelAnimationFrame(streamRafRef.current);
            streamRafRef.current = null;
          }
          streamBufRef.current = '';
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: err?.message || 'Request failed' } : m));
        }
      }, {
        persona: settings?.persona ?? undefined,
        context_window: settings?.contextWindow ?? undefined,
        advanced_rag: settings?.advancedRag ?? undefined,
        use_knowledge_base: true,
        source_options: sourceOptions,
      });
    } catch (err: any) {
      if (streamRafRef.current != null) {
        cancelAnimationFrame(streamRafRef.current);
        streamRafRef.current = null;
      }
      streamBufRef.current = '';
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

  const handleDeleteTranscript = async (t: TranscriptListItem) => {
    if (!window.confirm(`Delete "${t.title}"? This will remove it from embeddings and all stored data. This cannot be undone.`)) return;
    setDeletingId(t.id);
    try {
      await deleteTranscript(t.id);
      await loadTranscripts();
      setPreviewTranscript(prev => prev?.id === t.id ? null : prev);
    } catch (e) {
      console.error(e);
      alert((e as Error)?.message || 'Failed to delete transcript');
    } finally {
      setDeletingId(null);
    }
  };

  const handlePreviewTranscript = async (t: TranscriptListItem) => {
    setPreviewLoading(true);
    setPreviewTranscript(null);
    try {
      const detail = await getTranscript(t.id);
      setPreviewTranscript(detail);
    } catch (e) {
      console.error(e);
      alert((e as Error)?.message || 'Failed to load transcript');
    } finally {
      setPreviewLoading(false);
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
              <input type="text" placeholder="Search resources..." value={docSearch} onChange={e => setDocSearch(e.target.value)} className="w-full rounded-lg bg-black/30 border border-white/10 pl-9 pr-3 py-2 text-base outline-none focus:border-white/30" />
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
                    <div className="flex items-center gap-1 py-2 px-2 hover:bg-white/5">
                      <button type="button" onClick={() => setExpandedTranscriptId(expandedTranscriptId === t.id ? null : t.id)} className="flex-1 min-w-0 text-left flex flex-col gap-1.5 touch-manipulation">
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
                      <div className="shrink-0 flex items-center gap-0.5">
                        <button type="button" onClick={() => handlePreviewTranscript(t)} disabled={previewLoading} className="p-2 rounded text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/10 disabled:opacity-50 transition-colors touch-manipulation" title="Preview transcript">
                          <ICONS.Eye className="w-4 h-4" />
                        </button>
                        <button type="button" onClick={() => handleDeleteTranscript(t)} disabled={deletingId === t.id} className="p-2 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors touch-manipulation" title="Delete from embeddings and storage">
                          <ICONS.Trash className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
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
          <button type="button" onClick={() => newChat()} className="shrink-0 p-2 rounded-lg text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center" aria-label="New chat" title="New chat">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>
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
                <div className="mt-3">
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setChunkModalForId(m.id); }}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-400 hover:text-cyan-300 border border-cyan-500/30 hover:border-cyan-400/50 hover:bg-cyan-500/[0.07] rounded-lg px-3 py-1.5 transition-colors"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6M5 6h14M5 10h14" />
                    </svg>
                    Sources
                    <span className="ml-0.5 px-1.5 py-0 rounded-full bg-cyan-500/20 text-[10px] font-bold">{m.citations.length}</span>
                  </button>
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <div className="px-3 sm:px-5 py-3 sm:py-4 border-t border-white/10 flex flex-col gap-3 shrink-0">
          {/* <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500 shrink-0 mr-1">Search in</span>
            {(['transcript', 'document', 'general'] as const).map((key) => {
              const checked = sourceOptions[key];
              const label = { transcript: 'Transcript', document: 'Document', general: 'General' }[key];
              const Icon = { transcript: ICONS.Transcript, document: ICONS.File, general: ICONS.Chat }[key];
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => toggleSource(key)}
                  className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-[13px] font-medium transition-all duration-200 touch-manipulation active:scale-[0.97] ${
                    checked
                      ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30 shadow-[0_0_12px_-2px_rgba(20,184,166,0.15)]'
                      : 'bg-white/[0.04] text-slate-400 border border-white/[0.06] hover:bg-white/[0.08] hover:text-slate-300 hover:border-white/[0.1]'
                  }`}
                  style={{ transitionTimingFunction: 'cubic-bezier(0.25, 0.1, 0.25, 1)' }}
                >
                  <span className={`w-4 h-4 rounded-md flex items-center justify-center shrink-0 ${checked ? 'bg-teal-500/40' : 'bg-white/[0.06]'}`}>
                    {checked && (
                      <svg className="w-2.5 h-2.5 text-teal-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </span>
                  <Icon className="w-4 h-4 opacity-90" />
                  {label}
                </button>
              );
            })}
          </div> */}
          <div className="flex gap-2 sm:gap-3">
          <input
            type="text"
            className="flex-1 min-w-0 rounded-xl bg-black/30 border border-white/10 px-4 py-3 min-h-[44px] text-base outline-none focus:border-white/30"
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
      </div>

      {/* Mobile: resources panel overlay */}
      {resourcesPanelOpen && (
        <div className="fixed inset-0 z-40 md:hidden bg-black/60 backdrop-blur-sm" onClick={() => setResourcesPanelOpen(false)} aria-hidden />
      )}
      {/* Resources: slide-in drawer on mobile, sidebar on desktop */}
      <div className={`fixed md:relative inset-y-0 right-0 z-50 md:z-auto w-full max-w-sm md:max-w-none flex flex-col md:flex-none h-full md:h-auto transform ${resourcesPanelOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0'} transition-transform duration-200 md:transition-none md:w-64 lg:w-72 md:shrink-0`}>
        {resourcesAside}
      </div>

      {/* Chunk citation modal */}
      {chunkModalForId && (() => {
        const msg = messages.find(m => m.id === chunkModalForId);
        if (!msg?.citations?.length) return null;
        return (
          <ChunkCitationModal
            citations={msg.citations}
            onClose={() => setChunkModalForId(null)}
          />
        );
      })()}

      {/* Transcript preview modal */}
      {(previewTranscript || previewLoading) && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" onClick={() => !previewLoading && setPreviewTranscript(null)}>
          <div className="w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl border border-white/20 bg-[#0f172a] shadow-2xl overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between shrink-0">
              <h3 className="font-semibold text-white truncate">{previewTranscript?.title ?? 'Preview'}</h3>
              <button type="button" onClick={() => !previewLoading && setPreviewTranscript(null)} disabled={previewLoading} className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 disabled:opacity-50 transition-colors" aria-label="Close">
                <ICONS.Close className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-auto p-4 space-y-4">
              {previewLoading && (
                <div className="flex items-center justify-center py-12 text-slate-400">
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse mr-2" />
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.2s] mr-2" />
                  <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.4s] mr-2" />
                  <span className="text-sm">Loading…</span>
                </div>
              )}
              {!previewLoading && previewTranscript && (
                <>
                  {previewTranscript.raw_text && (
                    <div>
                      <p className="text-xs font-medium text-slate-500 mb-2">Raw transcript</p>
                      <pre className="text-sm text-white/90 whitespace-pre-wrap break-words font-sans bg-black/30 rounded-lg p-3 border border-white/10">{previewTranscript.raw_text}</pre>
                    </div>
                  )}
                  {previewTranscript.polished_text && (
                    <div>
                      <p className="text-xs font-medium text-slate-500 mb-2">Refined notes</p>
                      <pre className="text-sm text-white/90 whitespace-pre-wrap break-words font-sans bg-black/30 rounded-lg p-3 border border-white/10">{previewTranscript.polished_text}</pre>
                    </div>
                  )}
                  {!previewTranscript.raw_text && !previewTranscript.polished_text && (
                    <p className="text-sm text-slate-500">No text content available.</p>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeChat;
