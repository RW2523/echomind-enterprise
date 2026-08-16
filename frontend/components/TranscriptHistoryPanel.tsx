/**
 * TranscriptHistoryPanel
 *
 * Shows a browsable list of all saved sessions (live transcripts + boardroom).
 * Clicking a session loads its full text and Silent Assistant analysis cards.
 * Boardroom sessions linked to a transcript are surfaced inline.
 */
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  listTranscripts,
  getTranscript,
  getTranscriptAnalysis,
  getTranscriptAssistant,
  deleteTranscript,
  listBoardroomSessions,
  getBoardroomSession,
  deleteBoardroomSession,
  type TranscriptListItem,
  type BoardroomSessionListItem,
} from '../services/backend';
import type { AnalysisCard, SentenceCheck, TranscriptAssistantData, TranscriptSegment } from '../types';
import AnalysisCardModal, { checkStyle } from './AnalysisCardModal';
import AssistantSidebar, { useAssistantUnread, type AssistantTab } from './AssistantSidebar';
import { TranscriptSegmentLine } from './TranscriptSentences';
import TagChip from './TagChip';
import ProofPopover from './ProofPopover';
import BoardroomView from './BoardroomView';
import type { BoardroomSession } from '../types';
import { asCheck, deriveActionItems, roleLabel } from '../utils/silentAssistant';

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

// ── sub-components ────────────────────────────────────────────────────────────

/** Ranges of checked sentences inside the raw transcript text (char offsets, else indexOf of sentence_text). */
function locateChecks(rawText: string, checks: SentenceCheck[], segments: TranscriptSegment[]): { start: number; end: number; check: SentenceCheck }[] {
  const out: { start: number; end: number; check: SentenceCheck }[] = [];
  if (!rawText) return out;
  const segOffset = new Map<string, number>();
  for (const seg of segments) {
    if (!seg.text) continue;
    const i = rawText.indexOf(seg.text);
    if (i >= 0) segOffset.set(seg.paragraph_id, i);
  }
  const lower = rawText.toLowerCase();
  let searchFrom = 0;
  for (const c of checks) {
    const so = segOffset.get(c.segment_id);
    if (so != null && typeof c.char_start === 'number' && typeof c.char_end === 'number' && c.char_end > c.char_start) {
      const start = so + c.char_start, end = so + c.char_end;
      if (end <= rawText.length) { out.push({ start, end, check: c }); continue; }
    }
    const needle = (c.sentence_text || c.segment_text || '').trim();
    if (!needle) continue;
    let i = lower.indexOf(needle.toLowerCase(), searchFrom);
    if (i < 0) i = lower.indexOf(needle.toLowerCase());
    if (i >= 0) { out.push({ start: i, end: i + needle.length, check: c }); searchFrom = i + needle.length; }
  }
  // sort + drop overlaps (keep the earlier one)
  out.sort((a, b) => a.start - b.start || b.end - a.end);
  const clean: typeof out = [];
  let lastEnd = -1;
  for (const r of out) { if (r.start >= lastEnd) { clean.push(r); lastEnd = r.end; } }
  return clean;
}

const HighlightedTranscriptText: React.FC<{
  rawText: string;
  checks: SentenceCheck[];
  segments: TranscriptSegment[];
  selectedId: string | null;
  onSelect: (c: SentenceCheck) => void;
}> = ({ rawText, checks, segments, selectedId, onSelect }) => {
  const ranges = useMemo(() => locateChecks(rawText, checks, segments), [rawText, checks, segments]);
  if (!ranges.length) return <p className="text-sm text-white/85 leading-relaxed whitespace-pre-wrap break-words">{rawText}</p>;
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  ranges.forEach((r, i) => {
    if (r.start > cursor) nodes.push(<span key={`t-${i}`}>{rawText.slice(cursor, r.start)}</span>);
    const st = checkStyle(r.check);
    const first = r.check.evidence?.[0];
    const span = (
      <span
        key={r.check.id}
        onClick={() => onSelect(r.check)}
        className={`rounded px-0.5 border-b cursor-pointer ${st.bg} ${st.border} hover:brightness-125 ${selectedId === r.check.sentence_id ? 'ring-1 ring-white/40 brightness-125' : ''}`}
        title={r.check.tags?.map((t) => t.label ?? t.tag).join(', ') || r.check.label}
      >
        {rawText.slice(r.start, r.end)}
        {r.check.tags?.length ? (
          <span className="inline-flex items-center gap-1 ml-1 align-middle">
            {r.check.tags.slice(0, 2).map((t) => <TagChip key={t.tag} tag={t} />)}
          </span>
        ) : null}
      </span>
    );
    nodes.push(first ? <ProofPopover key={`p-${r.check.id}`} evidence={first} note={r.check.explanation} trigger="hover">{span}</ProofPopover> : span);
    cursor = r.end;
  });
  if (cursor < rawText.length) nodes.push(<span key="tail">{rawText.slice(cursor)}</span>);
  return <p className="text-sm text-white/85 leading-loose whitespace-pre-wrap break-words">{nodes}</p>;
};

interface DetailViewProps {
  item: TranscriptListItem;
  boardroomId?: string | null;
  onBack: () => void;
}

const DetailView: React.FC<DetailViewProps> = ({ item, boardroomId, onBack }) => {
  const [rawText, setRawText] = useState('');
  const [legacyCards, setLegacyCards] = useState<AnalysisCard[]>([]);
  const [assistant, setAssistant] = useState<TranscriptAssistantData | null>(null);
  const [boardroomSession, setBoardroomSession] = useState<BoardroomSession | null>(null);
  const [showBoardroom, setShowBoardroom] = useState(false);
  const [loading, setLoading] = useState(true);
  const [modalCheck, setModalCheck] = useState<SentenceCheck | null>(null);
  const [selectedSentenceId, setSelectedSentenceId] = useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [tab, setTab] = useState<AssistantTab>('checks');
  const [sheetOpen, setSheetOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setAssistant(null);
    setLegacyCards([]);
    Promise.all([
      getTranscript(item.id)
        .then(d => { if (!cancelled) setRawText(d.raw_text || ''); })
        .catch(() => { if (!cancelled) setRawText('Failed to load transcript.'); }),  // avoid unhandled rejection (L32)
      getTranscriptAnalysis(item.id).then(d => { if (!cancelled) setLegacyCards(d.cards || []); }).catch(() => {}),
      getTranscriptAssistant(item.id).then(d => { if (!cancelled) setAssistant(d); }).catch(() => {}),
      boardroomId
        ? getBoardroomSession(boardroomId).then(s => { if (!cancelled) setBoardroomSession(s); }).catch(() => {})
        : Promise.resolve(),
    ]).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [item.id, boardroomId]);

  // Prefer the richer /assistant checks; fall back to legacy /analysis cards.
  const checks: SentenceCheck[] = useMemo(() => {
    const src = assistant?.checks?.length ? assistant.checks : legacyCards;
    return src.map(asCheck);
  }, [assistant, legacyCards]);
  const checksById = useMemo(() => Object.fromEntries(checks.map((c) => [c.sentence_id, c])) as Record<string, SentenceCheck>, [checks]);
  const records = assistant?.records ?? [];
  const subjects = assistant?.subjects ?? [];
  const segments = assistant?.segments ?? [];
  const actionItems = useMemo(() => deriveActionItems(checks), [checks]);
  const hasSentenceSegments = segments.some((s) => s.sentences && s.sentences.length > 0);
  const counts = useMemo(() => ({ records: records.length, checks: checks.length, actions: actionItems.length }), [records.length, checks.length, actionItems.length]);
  const { unread } = useAssistantUnread(counts, tab);

  const openCheck = useCallback((c: SentenceCheck) => {
    setSelectedSentenceId(c.sentence_id);
    setSelectedSegmentId(c.segment_id);
    setModalCheck(c);
  }, []);

  const sidebar = (
    <AssistantSidebar
      cards={checks}
      records={records}
      actionItems={actionItems}
      subjects={subjects}
      selectedSegmentId={selectedSegmentId}
      onSelectSegment={setSelectedSegmentId}
      selectedSentenceId={selectedSentenceId}
      onSelectSentence={setSelectedSentenceId}
      onOpenCard={openCheck}
      tab={tab}
      onTabChange={setTab}
      unread={unread}
      showRecords={records.length > 0 || subjects.length > 0}
      namespaceEmptyHint="No records were pulled during this session."
      onClose={sheetOpen ? () => setSheetOpen(false) : undefined}
    />
  );

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
        <div className="flex-1 min-h-0 flex overflow-hidden relative">
          {/* Left: transcript text (checked sentences highlighted) */}
          <div className="flex-1 min-w-0 overflow-auto p-4">
            <div className="rounded-xl border border-white/10 bg-black/20 p-4 min-h-full">
              <div className="text-xs font-semibold text-slate-500 mb-3 uppercase tracking-wide flex items-center gap-2 flex-wrap">
                Transcript
                {checks.length > 0 && <span className="text-[10px] normal-case font-normal text-slate-600">highlighted = checked · hover for proof · click for details</span>}
              </div>
              {hasSentenceSegments ? (
                <div className="space-y-1 text-sm">
                  {segments.map((seg) => (
                    <TranscriptSegmentLine
                      key={seg.paragraph_id}
                      segment={seg}
                      checks={checksById}
                      sentenceStatus={{}}
                      selectedSentenceId={selectedSentenceId}
                      onSelectSentence={(id, c) => { setSelectedSentenceId(id); if (c) openCheck(c); }}
                      isSelected={selectedSegmentId === seg.paragraph_id}
                      onSelectSegment={() => setSelectedSegmentId(seg.paragraph_id === selectedSegmentId ? null : seg.paragraph_id)}
                    />
                  ))}
                </div>
              ) : rawText ? (
                <HighlightedTranscriptText
                  rawText={rawText}
                  checks={checks}
                  segments={segments}
                  selectedId={selectedSentenceId}
                  onSelect={openCheck}
                />
              ) : (
                <p className="text-sm text-slate-500">No transcript text saved.</p>
              )}
              {subjects.length > 0 && (
                <div className="mt-4 pt-3 border-t border-white/10 text-[11px] text-slate-400 flex flex-wrap gap-x-3 gap-y-1">
                  {subjects.filter((s) => s.status !== 'rejected').map((s) => (
                    <span key={s.id}><span className="text-slate-500">{roleLabel(s.kind)}:</span> <span className="text-white/90">{s.display_name}</span>{s.status === 'confirmed' ? ' ✓' : ''}</span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right: Silent Assistant (records / checks / actions) */}
          <div className="hidden md:block w-72 lg:w-80 shrink-0 border-l border-white/10 min-h-0">
            {sidebar}
          </div>

          {/* Mobile bottom sheet */}
          {(checks.length > 0 || records.length > 0) && (
            <button
              type="button"
              onClick={() => setSheetOpen(true)}
              className="md:hidden absolute bottom-4 right-4 z-30 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-slate-900/95 backdrop-blur px-4 py-2.5 text-xs font-semibold text-cyan-300 shadow-xl touch-manipulation min-h-[44px]"
            >
              Assistant
              <span className="rounded-full bg-white/10 px-1.5 py-0.5 text-[10px] tabular-nums">{checks.length + records.length}</span>
            </button>
          )}
          {sheetOpen && (
            <>
              <div className="fixed inset-0 z-40 md:hidden bg-black/60 backdrop-blur-sm" onClick={() => setSheetOpen(false)} aria-hidden />
              <div className="fixed inset-x-0 bottom-0 z-50 md:hidden h-[78vh] rounded-t-2xl border-t border-white/15 bg-slate-900 shadow-2xl flex flex-col overflow-hidden">
                <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-white/20 shrink-0" />
                <div className="flex-1 min-h-0">{sidebar}</div>
              </div>
            </>
          )}
        </div>
      )}

      {modalCheck && <AnalysisCardModal card={modalCheck} onClose={() => setModalCheck(null)} />}
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
