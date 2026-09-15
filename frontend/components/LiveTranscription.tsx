/**
 * Live Transcript screen — reads as Transcript → Verification → Actions.
 *
 *  ┌ session bar: [⚖️ Legal consultation ▾] [Lawyer | Client | Auto] [Flagged only | All insights] … [Start/Stop] [•••]
 *  ├ title strip: "Legal consultation — Contract Review" · started 10:14 · 12:34   (click → Session details)
 *  └ body: transcript (65%) | insights column (35%): Who is this? · Records | Checks | Actions
 *
 * Everything secondary (history, word cloud, export, hand-raise read-out, clear, mute, boardroom)
 * lives in the ••• overflow menu; name / location / tags / sources live in the Session details dialog.
 */
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { ICONS } from '../constants';
import { defaultTranscriptName } from '../services/backend';
import type { UseLiveTranscriptionReturn } from '../hooks/useLiveTranscription';
import type { AnalysisMode, SentenceCheck } from '../types';
import { PACKS, resolvePack } from '../packs';
import AnalysisCardModal from './AnalysisCardModal';
import WordCloudModal from './WordCloudModal';
import BoardroomView from './BoardroomView';
import TranscriptHistoryPanel from './TranscriptHistoryPanel';
import ScenarioPicker from './ScenarioPicker';
import RoleToggle from './RoleToggle';
import AssistantSidebar, { useAssistantUnread, type AssistantTab } from './AssistantSidebar';
import { TranscriptBlocks, buildSpeakerBlocks, blocksToText } from './TranscriptSentences';
import { useHandraise } from '../hooks/useHandraise';
import {
  LOCATION_DEFAULT_VALUE, SCENARIO_ICONS, UNTITLED_SESSION, copyToClipboard, displayLocation, findScenario,
  formatClockShort, formatDateTime, formatElapsed, isFlagged, isGenericSessionName, normalizeLocationInput,
  roleLabel, sessionDisplayTitle, sortChecksProblemsFirst,
} from '../utils/silentAssistant';

/** Knowledge-source choices: whole KB + every vertical pack namespace. */
const KB_SOURCES: { value: string; label: string }[] = [
  { value: '', label: 'All documents' },
  ...Object.values(PACKS).map((p) => ({ value: p.namespace, label: `${p.icon} ${p.name}` })),
];

const MODES: [AnalysisMode, string, string][] = [
  ['flags_only', 'Flagged only', 'Only surface wrong, risky or violating statements'],
  ['flags_and_records', 'All insights', 'Also pull records, contract clauses, policies and references'],
];

/** 1 Hz clock while `active` (elapsed-time hint). */
function useTicker(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

// ── Overflow menu ─────────────────────────────────────────────────────────────

interface MenuItem {
  id: string;
  label: string;
  hint?: string;
  icon?: React.ReactNode;
  onSelect?: () => void;
  disabled?: boolean;
  danger?: boolean;
  divider?: boolean;
}

const OverflowMenu: React.FC<{ items: MenuItem[] }> = ({ items }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="More actions"
        title="More actions"
        className={`p-2 rounded-xl min-h-[38px] min-w-[38px] flex items-center justify-center transition-colors touch-manipulation ${open ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/10'}`}
      >
        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8" /><circle cx="12" cy="12" r="1.8" /><circle cx="19" cy="12" r="1.8" /></svg>
      </button>
      {open && (
        <div role="menu" className="absolute right-0 top-full mt-1.5 z-50 w-64 rounded-xl border border-white/15 bg-slate-900 shadow-2xl py-1 overflow-hidden">
          {items.map((it) => it.divider ? (
            <div key={it.id} className="my-1 border-t border-white/[0.08]" role="separator" />
          ) : (
            <button
              key={it.id}
              type="button"
              role="menuitem"
              disabled={it.disabled}
              onClick={() => { setOpen(false); it.onSelect?.(); }}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-left text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                it.danger ? 'text-rose-300 hover:bg-rose-500/10' : 'text-slate-200 hover:bg-white/[0.07]'
              }`}
            >
              <span className="w-4 h-4 shrink-0 flex items-center justify-center text-slate-400">{it.icon}</span>
              <span className="flex-1 truncate">{it.label}</span>
              {it.hint && <span className="text-[10px] text-slate-500 shrink-0">{it.hint}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const I = {
  info: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  mic: <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>,
  muted: <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/><path strokeLinecap="round" d="M4 4l16 16" /></svg>,
  speak: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M18.364 5.636a9 9 0 010 12.728M11 5L6 9H2v6h4l5 4V5z" /></svg>,
  copy: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>,
  download: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>,
  history: <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
  boardroom: <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"><path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/></svg>,
  close: <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>,
};

// ── Session details dialog ────────────────────────────────────────────────────

interface SessionDetailsProps {
  onClose: () => void;
  title: string;
  autoTitle: string | null;
  sessionName: string;
  onNameChange: (v: string) => void;
  location: string;
  onLocationCommit: (v: string) => void;
  onCommit: () => void;
  tags: string[];
  newTagInput: string;
  setNewTagInput: (v: string) => void;
  addTag: () => void;
  removeTag: (t: string) => void;
  startedAt: Date | null;
  elapsedMs: number;
  ended: boolean;
  scenarioLabel: string;
  rolesLabel: string;
  kbDocs: number | null;
  namespace: string;
  analysisModeLabel: string;
  transcriptId: string | null;
  boardroomStatus: string | null;
}

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="flex items-baseline gap-3 text-xs">
    <span className="w-24 shrink-0 text-slate-500">{label}</span>
    <span className="min-w-0 text-slate-200 break-words">{children}</span>
  </div>
);

const SessionDetailsDialog: React.FC<SessionDetailsProps> = (p) => {
  const { onClose } = p;
  const [locDraft, setLocDraft] = useState(() => displayLocation(p.location));
  const explicitName = isGenericSessionName(p.sessionName) ? '' : p.sessionName;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const commitLocation = () => {
    const v = normalizeLocationInput(locDraft);
    p.onLocationCommit(v);
    setLocDraft(displayLocation(v));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={p.onClose}>
      <div
        role="dialog"
        aria-label="Session details"
        className="w-full max-w-md max-h-[92vh] overflow-auto rounded-2xl border border-white/15 bg-slate-900 shadow-2xl p-5 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Session details</div>
            <div className="text-sm font-semibold text-white truncate mt-0.5">{p.title}</div>
          </div>
          <button type="button" onClick={p.onClose} className="p-2 -mr-2 -mt-1 rounded-lg text-slate-400 hover:text-white hover:bg-white/10" aria-label="Close">
            <ICONS.Close className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Name</label>
            <input
              type="text"
              value={explicitName}
              onChange={(e) => p.onNameChange(e.target.value)}
              onBlur={p.onCommit}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); (e.target as HTMLInputElement).blur(); } }}
              placeholder={p.autoTitle ?? UNTITLED_SESSION}
              className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none min-h-[40px]"
            />
            {!explicitName && (
              <p className="mt-1 text-[11px] text-slate-500">{p.autoTitle ? 'Auto-titled from the conversation — type to override.' : 'A title is generated from the conversation once it starts.'}</p>
            )}
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
            <input
              type="text"
              value={locDraft}
              onChange={(e) => setLocDraft(e.target.value)}
              onBlur={commitLocation}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); (e.target as HTMLInputElement).blur(); } }}
              placeholder="Remote call"
              className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none min-h-[40px]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Tags</label>
            <div className="flex flex-wrap items-center gap-1.5">
              {p.tags.map((tag) => (
                <span key={tag} className="inline-flex items-center gap-1 rounded-lg bg-white/10 border border-white/10 pl-2 pr-1 py-1 text-xs text-white/90">
                  {tag}
                  <button type="button" onClick={() => p.removeTag(tag)} className="p-0.5 rounded text-slate-400 hover:text-white" aria-label={`Remove ${tag}`}>{I.close}</button>
                </span>
              ))}
              <input
                type="text"
                value={p.newTagInput}
                onChange={(e) => p.setNewTagInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); p.addTag(); } }}
                placeholder="Add tag"
                className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1.5 text-xs text-white placeholder-slate-500 w-28 focus:border-cyan-500/40 focus:outline-none min-h-[32px]"
              />
              <button type="button" onClick={p.addTag} disabled={!p.newTagInput.trim()} className="rounded-lg px-2.5 py-1.5 text-xs font-medium bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25 disabled:opacity-40 min-h-[32px]">
                Add
              </button>
            </div>
            {p.tags.length === 0 && <p className="mt-1 text-[11px] text-slate-500">Topic tags are added automatically when the session is stored.</p>}
          </div>
        </div>

        <div className="space-y-1.5 border-t border-white/[0.08] pt-3">
          <Row label="Started">{p.startedAt ? formatDateTime(p.startedAt) : '—'}</Row>
          <Row label={p.ended ? 'Duration' : 'Elapsed'}>{p.startedAt ? formatElapsed(p.elapsedMs) : '—'}</Row>
          <Row label="Conversation">{p.scenarioLabel}</Row>
          <Row label="Speakers">{p.rolesLabel}</Row>
          <Row label="Sources">{p.kbDocs == null ? '—' : `${p.kbDocs} document${p.kbDocs === 1 ? '' : 's'}`}{p.namespace && p.namespace !== 'default' ? <span className="text-slate-500"> · namespace {p.namespace}</span> : null}</Row>
          <Row label="Insights">{p.analysisModeLabel}</Row>
          {p.boardroomStatus && <Row label="Boardroom">{p.boardroomStatus}</Row>}
          {p.transcriptId && <Row label="Transcript"><span className="font-mono text-[11px] text-slate-400" title={p.transcriptId}>{p.transcriptId.slice(0, 12)}…</span></Row>}
        </div>

        <div className="flex justify-end">
          <button type="button" onClick={p.onClose} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15">Done</button>
        </div>
      </div>
    </div>
  );
};

// ── Screen ────────────────────────────────────────────────────────────────────

interface LiveTranscriptionProps {
  liveTranscription: UseLiveTranscriptionReturn;
}

const LiveTranscription: React.FC<LiveTranscriptionProps> = ({ liveTranscription }) => {
  const [showWordCloud, setShowWordCloud] = useState(false);
  const [showBoardroom, setShowBoardroom] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [modalCheck, setModalCheck] = useState<SentenceCheck | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [tab, setTab] = useState<AssistantTab>('checks');
  const [endedAt, setEndedAt] = useState<number | null>(null);
  const wasListeningRef = useRef(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const { speaking, speakCard, speakSummary, stop: stopSpeaking } = useHandraise();

  const {
    fullTranscript, partial, transcriptSegments, analysisCards, analyzingSegmentIds,
    selectedSegmentId, setSelectedSegmentId, listening, wsStatus, wsError,
    sessionName, sessionLocation, sessionStartedAt, customTags, newTagInput,
    setSessionName, setSessionLocation, setNewTagInput,
    openStartModal, startSession, handleStopAndExtractTags, clearAndReset, addTag, removeTag,
    micMuted, setMicMuted, showStartModal, modalName, setModalName, setShowStartModal,
    boardroomMode, setBoardroomMode, boardroomSession, setBoardroomSession, boardroomUploading, endBoardroomSession,
    scenario, setScenario, scenarios, scenarioSuggestion, acceptScenarioSuggestion, dismissScenarioSuggestion,
    kbNamespace, setKbNamespace, analysisMode, setAnalysisMode, subjectHint, setSubjectHint,
    myRole, setSpeakerRole, tagVocab, roles, sessionAck, checks, sentenceStatus, subjects, records, actionItems,
    wsWarning, clearWarning, confirmSubject, rejectSubject, selectedSentenceId, setSelectedSentenceId,
    sessionTitle, transcriptId, saveSessionDetails,
  } = liveTranscription;

  // ── Derived session identity ──
  const displayTitle = sessionDisplayTitle(sessionName, sessionTitle);
  const now = useTicker(listening);
  useEffect(() => {
    if (listening) { wasListeningRef.current = true; setEndedAt(null); }
    else if (wasListeningRef.current) { wasListeningRef.current = false; setEndedAt(Date.now()); }
  }, [listening]);
  useEffect(() => { setEndedAt(null); wasListeningRef.current = false; }, [sessionStartedAt]);
  const elapsedMs = sessionStartedAt
    ? Math.max(0, (listening ? now : (endedAt ?? sessionStartedAt.getTime())) - sessionStartedAt.getTime())
    : 0;
  const ended = !listening && endedAt != null;
  const hasSession = listening || !!sessionStartedAt;
  const hasText = transcriptSegments.length > 0 || !!fullTranscript.trim() || !!partial.trim();

  const onStartFromModal = () => {
    // An empty name is a placeholder: the server auto-titles the session (never shown as "transcript_…").
    startSession((modalName || '').trim() || defaultTranscriptName(), LOCATION_DEFAULT_VALUE);
  };

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [fullTranscript, partial]);

  // When boardroom session becomes available, show boardroom view
  useEffect(() => {
    if (boardroomSession && !showBoardroom) setShowBoardroom(true);
  }, [boardroomSession]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleEndBoardroom = useCallback(async () => {
    await endBoardroomSession();
    setShowBoardroom(true);
  }, [endBoardroomSession]);

  // ── Insights column state ──
  const showRecordsTab = analysisMode === 'flags_and_records' || records.length > 0 || subjects.length > 0;
  const counts = useMemo(() => ({ records: records.length, checks: analysisCards.length, actions: actionItems.length }), [records.length, analysisCards.length, actionItems.length]);
  const { unread } = useAssistantUnread(counts, tab);
  const totalUnread = unread.records + unread.checks + unread.actions;
  const flaggedCount = useMemo(() => (analysisCards as SentenceCheck[]).filter((c) => isFlagged(c)).length, [analysisCards]);
  useEffect(() => { if (!showRecordsTab && tab === 'records') setTab('checks'); }, [showRecordsTab, tab]);

  const openCheck = useCallback((check: SentenceCheck) => {
    setSelectedSentenceId(check.sentence_id);
    setSelectedSegmentId(check.segment_id);
    setModalCheck(check);
  }, [setSelectedSentenceId, setSelectedSegmentId]);

  // Transcript click: select the sentence and show it in the right panel (scrolls the Checks list to it);
  // on small screens there is no side column, so open the card directly.
  const onSentenceSelect = useCallback((sentenceId: string, check?: SentenceCheck) => {
    if (selectedSentenceId === sentenceId && !check) { setSelectedSentenceId(null); return; }
    setSelectedSentenceId(sentenceId);
    if (check) {
      setSelectedSegmentId(check.segment_id);
      setTab('checks');
      if (window.matchMedia('(max-width: 767px)').matches) setModalCheck(check);
    }
  }, [selectedSentenceId, setSelectedSentenceId, setSelectedSegmentId]);

  const suggested = scenarioSuggestion ? findScenario(scenarios, scenarioSuggestion.scenario) : undefined;
  const resolvedScenario = sessionAck?.scenario ?? (scenario === 'auto' ? null : scenario);
  const activeProfile = findScenario(scenarios, resolvedScenario ?? undefined);
  const kbEmpty = wsWarning?.code === 'namespace_empty' || (sessionAck != null && sessionAck.kb_docs === 0);
  const activePack = resolvePack();

  // ── Export / read-out helpers ──
  const transcriptText = useCallback(() => {
    const text = blocksToText(buildSpeakerBlocks(transcriptSegments));
    return text || fullTranscript;
  }, [transcriptSegments, fullTranscript]);
  const copyTranscript = useCallback(() => { void copyToClipboard(`${displayTitle}\n\n${transcriptText()}`); }, [displayTitle, transcriptText]);
  const downloadTranscript = useCallback(() => {
    const blob = new Blob([`${displayTitle}\n${sessionStartedAt ? formatDateTime(sessionStartedAt) : ''}\n\n${transcriptText()}\n`], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${displayTitle.replace(/[^\w\-]+/g, '_').replace(/^_+|_+$/g, '') || 'transcript'}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [displayTitle, sessionStartedAt, transcriptText]);
  const readLatestCheck = useCallback(() => {
    const list = analysisCards as SentenceCheck[];
    if (!list.length) return;
    const latestFlagged = [...list].reverse().find((c) => isFlagged(c));
    void speakCard(latestFlagged ?? sortChecksProblemsFirst(list)[0]);
  }, [analysisCards, speakCard]);

  const menuItems: MenuItem[] = useMemo(() => {
    const items: (MenuItem | null)[] = [
      { id: 'details', label: 'Session details…', icon: I.info, onSelect: () => setShowDetails(true), disabled: !hasSession },
      listening ? { id: 'mute', label: micMuted ? 'Unmute microphone' : 'Mute microphone', icon: micMuted ? I.muted : I.mic, onSelect: () => setMicMuted(!micMuted) } : null,
      { id: 'd1', label: '', divider: true },
      { id: 'speak-summary', label: speaking ? 'Stop reading' : 'Read out summary', hint: 'Handraise', icon: I.speak, disabled: !speaking && analysisCards.length === 0, onSelect: () => (speaking ? stopSpeaking() : void speakSummary(analysisCards)) },
      { id: 'speak-latest', label: 'Read out latest check', icon: I.speak, disabled: speaking || analysisCards.length === 0, onSelect: readLatestCheck },
      { id: 'd2', label: '', divider: true },
      { id: 'copy', label: 'Copy transcript', icon: I.copy, disabled: !hasText, onSelect: copyTranscript },
      { id: 'download', label: 'Download transcript (.txt)', icon: I.download, disabled: !hasText, onSelect: downloadTranscript },
      { id: 'wordcloud', label: 'Word cloud', icon: <ICONS.WordCloud className="w-4 h-4" />, disabled: !hasText, onSelect: () => setShowWordCloud(true) },
      { id: 'history', label: 'Session history', icon: I.history, onSelect: () => setShowHistory(true) },
      boardroomMode && listening ? { id: 'end-br', label: boardroomUploading ? 'Uploading boardroom audio…' : 'End boardroom recording', icon: I.boardroom, disabled: boardroomUploading, onSelect: () => { void handleEndBoardroom(); } } : null,
      boardroomSession && !listening ? { id: 'view-br', label: 'View boardroom', icon: I.boardroom, onSelect: () => setShowBoardroom(true) } : null,
      { id: 'd3', label: '', divider: true },
      { id: 'clear', label: listening ? 'Stop & clear session' : 'Clear session', icon: <ICONS.Trash className="w-4 h-4" />, danger: true, disabled: !hasSession && !hasText, onSelect: clearAndReset },
    ];
    return items.filter((x): x is MenuItem => x != null);
  }, [hasSession, listening, micMuted, setMicMuted, speaking, analysisCards, stopSpeaking, speakSummary, readLatestCheck, hasText, copyTranscript, downloadTranscript, boardroomMode, boardroomUploading, handleEndBoardroom, boardroomSession, clearAndReset]);

  const sidebar = (
    <AssistantSidebar
      cards={analysisCards}
      records={records}
      actionItems={actionItems}
      subjects={subjects}
      vocab={tagVocab}
      analyzingSegmentIds={analyzingSegmentIds}
      selectedSegmentId={selectedSegmentId}
      onSelectSegment={setSelectedSegmentId}
      selectedSentenceId={selectedSentenceId}
      onSelectSentence={setSelectedSentenceId}
      onOpenCard={openCheck}
      onConfirmSubject={confirmSubject}
      onRejectSubject={rejectSubject}
      tab={tab}
      onTabChange={setTab}
      unread={unread}
      showRecords={showRecordsTab}
      namespaceEmptyHint={kbEmpty ? 'This knowledge source has no documents yet — upload customer files, contracts or policies to pull records during the call.' : undefined}
      onClose={sheetOpen ? () => setSheetOpen(false) : undefined}
    />
  );

  // Show history panel overlay
  if (showHistory) {
    return (
      <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <TranscriptHistoryPanel onClose={() => setShowHistory(false)} />
      </div>
    );
  }

  // Show boardroom view overlay
  if (showBoardroom && boardroomSession) {
    return (
      <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <BoardroomView session={boardroomSession} onSessionUpdate={setBoardroomSession} onClose={() => setShowBoardroom(false)} />
      </div>
    );
  }

  const statusText = wsStatus === 'connecting' ? 'Connecting…' : wsStatus === 'loading' ? 'Loading speech model…' : null;
  // Before the first paragraph is committed the whole text is "live"; once stopped, show it as a normal block.
  const noSegments = transcriptSegments.length === 0;
  const partialForBlocks = listening ? (partial || (noSegments ? fullTranscript : '')) : partial;
  const blockSegments = !listening && noSegments && fullTranscript.trim()
    ? [{ paragraph_id: 'final', text: fullTranscript, role: myRole, started_at: sessionStartedAt?.getTime() }]
    : transcriptSegments;

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden relative">
      {/* Start dialog */}
      {showStartModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setShowStartModal(false)}>
          <div className="rounded-2xl border border-white/15 bg-slate-900 shadow-2xl max-w-lg w-full p-5 space-y-4 max-h-[92vh] overflow-auto" onClick={(e) => e.stopPropagation()}>
            <div>
              <div className="font-semibold text-white">Start a session</div>
              <p className="text-xs text-slate-400 mt-0.5">Pick the kind of conversation — it decides who is on the call, which records are pulled and which rules apply.</p>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Conversation type</label>
              <ScenarioPicker value={scenario} onChange={setScenario} scenarios={scenarios} />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Knowledge source</label>
                <select
                  value={kbNamespace}
                  onChange={(e) => setKbNamespace(e.target.value)}
                  className="w-full rounded-lg border border-white/15 bg-slate-900 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none min-h-[40px]"
                >
                  {KB_SOURCES.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}{activePack && s.value === activePack.namespace ? ' (this site)' : ''}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Who is on the call <span className="text-slate-600">(optional)</span></label>
                <input
                  type="text"
                  value={subjectHint}
                  onChange={(e) => setSubjectHint(e.target.value)}
                  placeholder="e.g. Priya Sharma"
                  className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none min-h-[40px]"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Session name <span className="text-slate-600">(optional)</span></label>
              <input
                type="text"
                value={modalName}
                onChange={(e) => setModalName(e.target.value)}
                placeholder="Auto-titled from the conversation"
                className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none min-h-[40px]"
              />
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setBoardroomMode(!boardroomMode)}
                role="switch"
                aria-checked={boardroomMode}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${boardroomMode ? 'bg-violet-500' : 'bg-white/10'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${boardroomMode ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </button>
              <span className="text-sm text-slate-300">Boardroom mode</span>
              <span className="text-xs text-slate-500">records full audio for diarised minutes</span>
            </div>
            <div className="flex gap-2 pt-1">
              <button type="button" onClick={() => setShowStartModal(false)} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-400 hover:bg-white/15">Cancel</button>
              <button type="button" onClick={onStartFromModal} className="ml-auto rounded-xl px-5 py-2 text-sm font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30">Start</button>
            </div>
          </div>
        </div>
      )}

      {/* Session details dialog */}
      {showDetails && (
        <SessionDetailsDialog
          onClose={() => setShowDetails(false)}
          title={displayTitle}
          autoTitle={sessionTitle}
          sessionName={sessionName}
          onNameChange={setSessionName}
          location={sessionLocation}
          onLocationCommit={(v) => { setSessionLocation(v); void saveSessionDetails(); }}
          onCommit={() => { void saveSessionDetails(); }}
          tags={customTags}
          newTagInput={newTagInput}
          setNewTagInput={setNewTagInput}
          addTag={addTag}
          removeTag={removeTag}
          startedAt={sessionStartedAt}
          elapsedMs={elapsedMs}
          ended={ended}
          scenarioLabel={activeProfile ? `${SCENARIO_ICONS[activeProfile.id] ?? ''} ${activeProfile.label}`.trim() : 'Auto-detect'}
          rolesLabel={`${roleLabel(roles.me)} · ${roleLabel(roles.other)}`}
          kbDocs={sessionAck ? sessionAck.kb_docs : null}
          namespace={sessionAck?.namespace ?? kbNamespace}
          analysisModeLabel={MODES.find((m) => m[0] === analysisMode)?.[1] ?? analysisMode}
          transcriptId={transcriptId}
          boardroomStatus={boardroomSession ? (boardroomUploading ? 'uploading audio…' : boardroomSession.status) : boardroomMode && listening ? 'recording' : null}
        />
      )}

      {/* ── Session bar: mode pill · speaker · insights mode · Start/Stop · ••• ── */}
      <div className={`shrink-0 flex flex-wrap items-center gap-2 px-3 sm:px-4 py-2.5 border-b transition-colors ${listening ? 'border-cyan-500/20' : 'border-white/10'}`}>
        <ScenarioPicker variant="pill" value={scenario} onChange={setScenario} scenarios={scenarios} resolved={sessionAck?.scenario ?? null} />
        <RoleToggle roles={roles} value={myRole} onChange={setSpeakerRole} disabled={!listening} />
        <div className="inline-flex items-center rounded-xl border border-white/10 bg-white/[0.04] p-0.5" role="group" aria-label="Insights mode">
          {MODES.map(([m, label, hint]) => (
            <button
              key={m}
              type="button"
              onClick={() => setAnalysisMode(m)}
              aria-pressed={analysisMode === m}
              title={hint}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors touch-manipulation min-h-[32px] whitespace-nowrap ${
                analysisMode === m ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          {statusText && <span className="text-[11px] text-slate-400 hidden sm:inline">{statusText}</span>}
          {wsError && <span className="text-[11px] text-rose-300 max-w-[140px] sm:max-w-[220px] truncate" title={wsError}>{wsError}</span>}
          {!listening ? (
            <button
              type="button"
              onClick={openStartModal}
              disabled={wsStatus === 'connecting' || wsStatus === 'loading'}
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 min-h-[38px] text-sm font-semibold bg-cyan-500/20 text-cyan-200 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 transition-colors touch-manipulation"
            >
              <ICONS.Mic className="w-4 h-4" />
              Start
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStopAndExtractTags}
              className="inline-flex items-center gap-2 rounded-xl px-4 py-2 min-h-[38px] text-sm font-semibold bg-rose-500/15 text-rose-200 border border-rose-500/30 hover:bg-rose-500/25 transition-colors touch-manipulation"
            >
              <span className="w-2.5 h-2.5 rounded-sm bg-rose-300" />
              Stop
            </button>
          )}
          <OverflowMenu items={menuItems} />
        </div>
      </div>

      {/* ── Title strip: session title · started/elapsed · live state ── */}
      <div className="shrink-0 flex flex-wrap items-center gap-x-3 gap-y-1.5 px-3 sm:px-4 py-2 border-b border-white/[0.06] bg-black/10 min-w-0">
        <button
          type="button"
          onClick={() => hasSession && setShowDetails(true)}
          disabled={!hasSession}
          className="group min-w-0 flex items-center gap-1.5 text-left rounded-lg -ml-1 px-1 py-0.5 hover:bg-white/[0.05] disabled:cursor-default disabled:hover:bg-transparent"
          title={hasSession ? 'Session details' : undefined}
        >
          <span className={`text-sm font-semibold truncate ${hasSession ? 'text-white' : 'text-slate-500'}`}>{hasSession ? displayTitle : 'No session'}</span>
          {hasSession && <span className="text-slate-500 group-hover:text-slate-300 shrink-0">{I.info}</span>}
          {boardroomMode && listening && (
            <span className="ml-1 text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-violet-500/20 text-violet-300 shrink-0">Boardroom</span>
          )}
        </button>
        {sessionStartedAt && (
          <span className="text-[11px] text-slate-500 tabular-nums whitespace-nowrap">
            started {formatClockShort(sessionStartedAt.getTime())} · {formatElapsed(elapsedMs)}{ended ? ' · ended' : ''}
          </span>
        )}
        {listening && (
          <button
            type="button"
            onClick={() => setMicMuted(!micMuted)}
            className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              micMuted ? 'text-rose-300 bg-rose-500/10 hover:bg-rose-500/20' : 'text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20'
            }`}
            title={micMuted ? 'Microphone muted — click to unmute' : 'Live — click to mute the microphone'}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${micMuted ? 'bg-rose-400' : 'bg-emerald-400 animate-pulse'}`} />
            {micMuted ? 'Muted' : 'Live'}
          </button>
        )}
        {!hasSession && !statusText && <span className="text-[11px] text-slate-500">Press Start — the transcript, checks and actions appear here as you speak.</span>}
        {statusText && <span className="text-[11px] text-slate-400 sm:hidden">{statusText}</span>}

        {/* Transient: scenario suggestion + warnings */}
        {scenarioSuggestion && suggested && (
          <div className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-500/10 pl-2.5 pr-1 py-0.5 text-[11px] text-amber-200">
            <span>{SCENARIO_ICONS[suggested.id] ?? '💬'}</span>
            <span>Sounds like a <b>{suggested.label.toLowerCase()}</b></span>
            <button type="button" onClick={acceptScenarioSuggestion} className="rounded-full px-2 py-0.5 text-[10px] font-semibold bg-amber-400/20 hover:bg-amber-400/30 text-amber-100">Switch</button>
            <button type="button" onClick={dismissScenarioSuggestion} className="rounded-full p-1 text-amber-300/70 hover:text-white hover:bg-white/10" aria-label="Dismiss suggestion">{I.close}</button>
          </div>
        )}
        {wsWarning && (
          <div className={`sm:ml-auto inline-flex items-center gap-2 rounded-lg border px-2.5 py-1 text-[11px] ${
            wsWarning.code === 'namespace_empty' ? 'border-amber-400/30 bg-amber-500/10 text-amber-200'
            : wsWarning.code === 'overloaded' || wsWarning.code === 'stt_dropped' ? 'border-rose-400/30 bg-rose-500/10 text-rose-200'
            : 'border-white/15 bg-white/5 text-slate-300'
          }`}>
            <span className="truncate max-w-[42ch]" title={wsWarning.message}>{wsWarning.message}</span>
            <button type="button" onClick={clearWarning} className="p-0.5 rounded hover:bg-white/10" aria-label="Dismiss warning">{I.close}</button>
          </div>
        )}
      </div>

      {/* ── Body: transcript (65%) | insights (35%) ── */}
      <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[minmax(0,13fr)_minmax(0,7fr)] overflow-hidden">
        <div
          className="min-w-0 min-h-0 overflow-auto px-4 sm:px-6 py-4 md:border-r md:border-white/[0.06]"
          onClick={() => setSelectedSentenceId(null)}
        >
          {hasText ? (
            <>
              <TranscriptBlocks
                segments={blockSegments}
                checks={checks}
                sentenceStatus={sentenceStatus}
                roles={roles}
                selectedSentenceId={selectedSentenceId}
                onSelectSentence={onSentenceSelect}
                partial={partialForBlocks}
                partialRole={myRole}
                className="max-w-3xl"
              />
              <div ref={transcriptEndRef} className="h-4" />
            </>
          ) : (
            <div className="h-full min-h-[200px] flex flex-col items-center justify-center text-center gap-2 text-slate-500">
              <div className={`w-11 h-11 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-center ${listening ? 'animate-pulse' : ''}`}>
                <ICONS.Mic className="w-5 h-5" />
              </div>
              <p className="text-sm">
                {listening ? (micMuted ? 'Microphone muted' : 'Listening…') : statusText ?? 'No transcript yet'}
              </p>
              {!listening && !statusText && activeProfile && (
                <p className="text-xs text-slate-600">{roleLabel(roles.me)} · {roleLabel(roles.other)} — {activeProfile.label}</p>
              )}
            </div>
          )}
        </div>

        {/* Insights column (md+) */}
        <div className="hidden md:flex min-w-0 min-h-0 flex-col bg-black/10">{sidebar}</div>
      </div>

      {/* Mobile: bottom-sheet toggle + sheet */}
      <button
        type="button"
        onClick={() => setSheetOpen(true)}
        className="md:hidden absolute bottom-4 right-4 z-30 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-slate-900/95 backdrop-blur px-4 py-2.5 text-xs font-semibold text-cyan-300 shadow-xl touch-manipulation min-h-[44px]"
        aria-label="Open insights"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        Insights
        {(totalUnread > 0 || flaggedCount > 0) && (
          <span className={`rounded-full px-1.5 py-0.5 text-[10px] tabular-nums ${flaggedCount > 0 ? 'bg-rose-500/30 text-rose-100' : 'bg-cyan-500/30 text-cyan-100'}`}>
            {flaggedCount > 0 ? flaggedCount : totalUnread}
          </span>
        )}
      </button>
      {sheetOpen && (
        <>
          <div className="fixed inset-0 z-40 md:hidden bg-black/60 backdrop-blur-sm" onClick={() => setSheetOpen(false)} aria-hidden />
          <div className="fixed inset-x-0 bottom-0 z-50 md:hidden h-[78vh] rounded-t-2xl border-t border-white/15 bg-slate-900 shadow-2xl flex flex-col overflow-hidden">
            <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-white/20 shrink-0" />
            <div className="flex-1 min-h-0">{sidebar}</div>
          </div>
        </>
      )}

      {/* Check detail modal (shared by transcript sentences, checks tab and actions tab) */}
      {modalCheck && <AnalysisCardModal card={modalCheck} vocab={tagVocab} onClose={() => setModalCheck(null)} />}

      {/* Word cloud modal */}
      {showWordCloud && (
        <WordCloudModal onClose={() => setShowWordCloud(false)} liveText={[fullTranscript, partial].filter(Boolean).join(' ')} listening={listening} />
      )}
    </div>
  );
};

export default LiveTranscription;
