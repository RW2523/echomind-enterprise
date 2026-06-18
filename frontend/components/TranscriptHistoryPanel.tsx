/**
 * TranscriptHistoryPanel
 *
 * Shows a browsable list of all saved sessions (live transcripts + boardroom).
 * Clicking a session loads its full text and Silent Assistant analysis cards.
 * Boardroom sessions linked to a transcript are surfaced inline.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  listTranscripts,
  getTranscript,
  getTranscriptAnalysis,
  deleteTranscript,
  listBoardroomSessions,
  getBoardroomSession,
  deleteBoardroomSession,
  type TranscriptListItem,
  type BoardroomSessionListItem,
} from '../services/backend';
import type { AnalysisCard } from '../types';
import { LABEL_CONFIG } from './AnalysisCardModal';
import BoardroomView from './BoardroomView';
import type { BoardroomSession } from '../types';

// ── helpers ──────────────────────────────────────────────────────────────────

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function labelBadge(card: AnalysisCard) {
  const cfg = LABEL_CONFIG[card.label];
  if (!cfg) return null;
  return (
    <span
      key={card.id}
      className={`inline-flex items-center gap-1 text-[10px] font-semibold rounded-full px-2 py-0.5 border ${cfg.badge} ${cfg.border}`}
    >
      {cfg.icon} {card.label}
    </span>
  );
}

// ── sub-components ────────────────────────────────────────────────────────────

interface DetailViewProps {
  item: TranscriptListItem;
  boardroomId?: string | null;
  onBack: () => void;
}

const DetailView: React.FC<DetailViewProps> = ({ item, boardroomId, onBack }) => {
  const [rawText, setRawText] = useState('');
  const [cards, setCards] = useState<AnalysisCard[]>([]);
  const [boardroomSession, setBoardroomSession] = useState<BoardroomSession | null>(null);
  const [showBoardroom, setShowBoardroom] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedCard, setSelectedCard] = useState<AnalysisCard | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      getTranscript(item.id)
        .then(d => { if (!cancelled) setRawText(d.raw_text || ''); })
        .catch(() => { if (!cancelled) setRawText('Failed to load transcript.'); }),  // avoid unhandled rejection (L32)
      getTranscriptAnalysis(item.id).then(d => { if (!cancelled) setCards(d.cards || []); }).catch(() => {}),
      boardroomId
        ? getBoardroomSession(boardroomId).then(s => { if (!cancelled) setBoardroomSession(s); }).catch(() => {})
        : Promise.resolve(),
    ]).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [item.id, boardroomId]);

  if (showBoardroom && boardroomSession) {
    return (
      <div className="h-full flex flex-col">
        <BoardroomView
          session={boardroomSession}
          onSessionUpdate={setBoardroomSession}
          onClose={() => setShowBoardroom(false)}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Detail header */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <button
          type="button"
          onClick={onBack}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          aria-label="Back"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-white truncate">{item.name || item.title}</div>
          <div className="text-[10px] text-slate-500">{fmtDate(item.created_at)}{item.location ? ` · ${item.location}` : ''}</div>
        </div>
        {boardroomSession && (
          <button
            type="button"
            onClick={() => setShowBoardroom(true)}
            className="shrink-0 rounded-xl px-3 py-1.5 text-xs font-semibold bg-violet-500/20 text-violet-400 border border-violet-500/30 hover:bg-violet-500/30 transition-colors"
          >
            View Boardroom
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">Loading…</div>
      ) : (
        <div className="flex-1 min-h-0 flex overflow-hidden">
          {/* Left: transcript text */}
          <div className="flex-1 min-w-0 overflow-auto p-4">
            <div className="rounded-xl border border-white/10 bg-black/20 p-4 min-h-full">
              <div className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wide">Transcript</div>
              {rawText ? (
                <p className="text-sm text-white/85 leading-relaxed whitespace-pre-wrap break-words">{rawText}</p>
              ) : (
                <p className="text-sm text-slate-500">No transcript text saved.</p>
              )}
            </div>
          </div>

          {/* Right: analysis cards */}
          <div className="w-64 lg:w-72 shrink-0 border-l border-white/10 overflow-auto p-3">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3 px-1">
              Silent Assistant
              {cards.length > 0 && (
                <span className="ml-1.5 text-cyan-400 font-bold">{cards.length}</span>
              )}
            </div>
            {cards.length === 0 ? (
              <div className="text-xs text-slate-600 px-1">No analysis cards for this session.</div>
            ) : (
              <div className="space-y-2">
                {cards.map(card => {
                  const cfg = LABEL_CONFIG[card.label];
                  if (!cfg) return null;
                  const isSelected = selectedCard?.id === card.id;
                  return (
                    <button
                      key={card.id}
                      type="button"
                      onClick={() => setSelectedCard(isSelected ? null : card)}
                      className={`w-full text-left rounded-xl border p-3 transition-all ${cfg.bg} ${cfg.border} ${isSelected ? 'ring-1 ring-white/20' : 'hover:brightness-110'}`}
                    >
                      <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                        <span className={`text-[10px] font-bold uppercase tracking-wide ${cfg.text}`}>
                          {cfg.icon} {card.label}
                        </span>
                        <span className="text-[10px] text-slate-500 ml-auto">{card.confidence.toFixed(0)}%</span>
                      </div>
                      <p className="text-xs text-slate-200 leading-relaxed line-clamp-2">{card.segment_text}</p>
                      {isSelected && (
                        <div className="mt-2 pt-2 border-t border-white/10">
                          <p className="text-[11px] text-slate-300 leading-relaxed">{card.explanation}</p>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Main component ─────────────────────────────────────────────────────────────

interface TranscriptHistoryPanelProps {
  onClose: () => void;
}

const TranscriptHistoryPanel: React.FC<TranscriptHistoryPanelProps> = ({ onClose }) => {
  const [transcripts, setTranscripts] = useState<TranscriptListItem[]>([]);
  const [boardroomSessions, setBoardroomSessions] = useState<BoardroomSessionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<TranscriptListItem | null>(null);
  const [tab, setTab] = useState<'transcripts' | 'boardroom'>('transcripts');
  const [openBoardroomSession, setOpenBoardroomSession] = useState<BoardroomSession | null>(null);
  const [brLoading, setBrLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [tRes, bRes] = await Promise.all([
        listTranscripts(),
        listBoardroomSessions(30).catch(() => ({ sessions: [] })),
      ]);
      setTranscripts(tRes.transcripts || []);
      setBoardroomSessions(bRes.sessions || []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Find boardroom session linked to a transcript
  const boardroomForTranscript = (tid: string) =>
    boardroomSessions.find(b => b.transcript_id === tid) ?? null;

  const openBoardroom = async (session: BoardroomSessionListItem) => {
    setBrLoading(true);
    try {
      const full = await getBoardroomSession(session.id);
      setOpenBoardroomSession(full);
    } catch {
      // ignore
    } finally {
      setBrLoading(false);
    }
  };

  const handleDeleteTranscript = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this transcript? This cannot be undone.')) return;
    setDeletingId(id);
    try {
      await deleteTranscript(id);
      setTranscripts(prev => prev.filter(t => t.id !== id));
    } catch {
      // ignore — item may already be gone
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteBoardroom = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this boardroom session? This cannot be undone.')) return;
    setDeletingId(id);
    try {
      await deleteBoardroomSession(id);
      setBoardroomSessions(prev => prev.filter(b => b.id !== id));
    } catch {
      // ignore
    } finally {
      setDeletingId(null);
    }
  };

  // ── Boardroom detail view ─────────────────────────────────────────────────
  if (openBoardroomSession) {
    return (
      <div className="h-full flex flex-col min-h-0">
        <BoardroomView
          session={openBoardroomSession}
          onSessionUpdate={setOpenBoardroomSession}
          onClose={() => setOpenBoardroomSession(null)}
        />
      </div>
    );
  }

  // ── Transcript detail view ─────────────────────────────────────────────────
  if (selected) {
    const linkedBoardroom = boardroomForTranscript(selected.id);
    return (
      <div className="h-full flex flex-col min-h-0">
        <DetailView
          item={selected}
          boardroomId={linkedBoardroom?.id ?? null}
          onBack={() => setSelected(null)}
        />
      </div>
    );
  }

  // ── List view ──────────────────────────────────────────────────────────────
  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Header */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">Session History</div>
          <div className="text-[10px] text-slate-500">All saved live transcripts and boardroom sessions</div>
        </div>
        <button
          type="button"
          onClick={load}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          title="Refresh"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M4 9a9 9 0 0115 0M20 15a9 9 0 01-15 0" />
          </svg>
        </button>
        <button
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          aria-label="Close history"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Tabs */}
      <div className="shrink-0 flex gap-0 border-b border-white/10 bg-black/10">
        <button
          type="button"
          onClick={() => setTab('transcripts')}
          className={`flex-1 py-2.5 text-xs font-semibold transition-colors ${tab === 'transcripts' ? 'text-cyan-400 border-b-2 border-cyan-500' : 'text-slate-400 hover:text-white'}`}
        >
          Live Transcripts
          {transcripts.length > 0 && (
            <span className="ml-1.5 text-[10px] rounded-full bg-white/10 px-1.5 py-0.5">{transcripts.length}</span>
          )}
        </button>
        <button
          type="button"
          onClick={() => setTab('boardroom')}
          className={`flex-1 py-2.5 text-xs font-semibold transition-colors ${tab === 'boardroom' ? 'text-violet-400 border-b-2 border-violet-500' : 'text-slate-400 hover:text-white'}`}
        >
          Boardroom
          {boardroomSessions.length > 0 && (
            <span className="ml-1.5 text-[10px] rounded-full bg-white/10 px-1.5 py-0.5">{boardroomSessions.length}</span>
          )}
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32 text-slate-500 text-sm">Loading…</div>
        ) : error ? (
          <div className="p-4 text-sm text-red-400">{error}</div>
        ) : tab === 'transcripts' ? (
          transcripts.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-sm">
              No transcripts saved yet.
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {transcripts.map(t => {
                const linked = boardroomForTranscript(t.id);
                return (
                  <div key={t.id} className="flex items-stretch group">
                    <button
                      type="button"
                      onClick={() => setSelected(t)}
                      className="flex-1 min-w-0 text-left px-4 py-3 hover:bg-white/5 transition-colors"
                    >
                      <div className="flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-white/90 truncate group-hover:text-white transition-colors">
                            {t.name || t.title}
                          </div>
                          <div className="text-[11px] text-slate-500 mt-0.5">
                            {fmtDate(t.created_at)}
                            {t.location ? ` · ${t.location}` : ''}
                          </div>
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {(t.tags || []).map(tag => (
                              <span key={tag} className="text-[10px] rounded-md bg-white/10 border border-white/10 px-1.5 py-0.5 text-slate-300">
                                {tag}
                              </span>
                            ))}
                            {linked && (
                              <span className="text-[10px] rounded-md bg-violet-500/20 border border-violet-500/30 px-1.5 py-0.5 text-violet-300">
                                Boardroom
                              </span>
                            )}
                          </div>
                        </div>
                        <svg className="w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteTranscript(e, t.id)}
                      disabled={deletingId === t.id}
                      className="shrink-0 px-3 opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 transition-all disabled:opacity-50"
                      title="Delete transcript"
                      aria-label="Delete transcript"
                    >
                      {deletingId === t.id ? (
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )
        ) : (
          /* Boardroom tab */
          boardroomSessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-sm">
              No boardroom sessions yet.
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {boardroomSessions.map(b => {
                const statusColor =
                  b.status === 'analysed' ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' :
                  b.status === 'transcribed' ? 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' :
                  b.status === 'processing' ? 'text-amber-400 bg-amber-500/10 border-amber-500/30' :
                  b.status === 'error' ? 'text-red-400 bg-red-500/10 border-red-500/30' :
                  'text-slate-400 bg-white/5 border-white/10';
                return (
                  <div key={b.id} className="flex items-stretch group">
                    <button
                      type="button"
                      onClick={() => openBoardroom(b)}
                      disabled={brLoading}
                      className="flex-1 min-w-0 text-left px-4 py-3 hover:bg-white/5 transition-colors disabled:opacity-50"
                    >
                      <div className="flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-medium text-white/90 group-hover:text-white transition-colors">
                              Session {b.id.slice(0, 8)}…
                            </span>
                            <span className={`text-[10px] font-semibold rounded-full px-2 py-0.5 border ${statusColor}`}>
                              {b.status}
                            </span>
                          </div>
                          <div className="text-[11px] text-slate-500 mt-0.5">
                            {fmtDate(b.created_at)} · {b.chunk_count} chunk{b.chunk_count !== 1 ? 's' : ''}
                          </div>
                          {b.transcript_id && (
                            <div className="text-[10px] text-slate-600 mt-0.5 truncate">
                              Transcript: {b.transcript_id.slice(0, 12)}…
                            </div>
                          )}
                        </div>
                        <svg className="w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteBoardroom(e, b.id)}
                      disabled={deletingId === b.id}
                      className="shrink-0 px-3 opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 transition-all disabled:opacity-50"
                      title="Delete session"
                      aria-label="Delete boardroom session"
                    >
                      {deletingId === b.id ? (
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default TranscriptHistoryPanel;
