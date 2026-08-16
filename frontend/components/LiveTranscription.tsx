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
import { TranscriptSegmentLine } from './TranscriptSentences';
import { SCENARIO_ICONS, findScenario, isProblem } from '../utils/silentAssistant';

function formatSessionDateTime(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${y}-${m}-${day} ${h}:${min}`;
}

/** Knowledge-source choices: whole KB + every vertical pack namespace. */
const KB_SOURCES: { value: string; label: string }[] = [
  { value: '', label: 'All documents' },
  ...Object.values(PACKS).map((p) => ({ value: p.namespace, label: `${p.icon} ${p.name}` })),
];

interface LiveTranscriptionProps {
  liveTranscription: UseLiveTranscriptionReturn;
}

const LiveTranscription: React.FC<LiveTranscriptionProps> = ({ liveTranscription }) => {
  const [showWordCloud, setShowWordCloud] = useState(false);
  const [showBoardroom, setShowBoardroom] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [modalCheck, setModalCheck] = useState<SentenceCheck | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [tab, setTab] = useState<AssistantTab>('checks');
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  const {
    fullTranscript,
    partial,
    transcriptSegments,
    analysisCards,
    analyzingSegmentIds,
    selectedSegmentId,
    setSelectedSegmentId,
    listening,
    wsStatus,
    wsError,
    sessionName,
    sessionLocation,
    sessionStartedAt,
    customTags,
    newTagInput,
    setSessionName,
    setSessionLocation,
    setNewTagInput,
    openStartModal,
    startSession,
    handleStopAndExtractTags,
    clearAndReset,
    addTag,
    removeTag,
    micMuted,
    setMicMuted,
    showStartModal,
    modalName,
    modalLocation,
    setModalName,
    setModalLocation,
    setShowStartModal,
    applyDefault,
    boardroomMode,
    setBoardroomMode,
    boardroomSession,
    setBoardroomSession,
    boardroomUploading,
    endBoardroomSession,
    // v2
    scenario,
    setScenario,
    scenarios,
    scenarioSuggestion,
    acceptScenarioSuggestion,
    dismissScenarioSuggestion,
    kbNamespace,
    setKbNamespace,
    analysisMode,
    setAnalysisMode,
    subjectHint,
    setSubjectHint,
    myRole,
    setSpeakerRole,
    tagVocab,
    roles,
    sessionAck,
    checks,
    sentenceStatus,
    subjects,
    records,
    actionItems,
    wsWarning,
    clearWarning,
    confirmSubject,
    rejectSubject,
    selectedSentenceId,
    setSelectedSentenceId,
  } = liveTranscription;

  const onStartFromModal = () => {
    const name = (modalName || '').trim() || defaultTranscriptName();
    const location = (modalLocation || '').trim() || 'default';
    startSession(name, location);
  };

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [fullTranscript, partial]);

  // When boardroom session becomes available, show boardroom view
  useEffect(() => {
    if (boardroomSession && !showBoardroom) setShowBoardroom(true);
  }, [boardroomSession]);

  const handleEndBoardroom = useCallback(async () => {
    await endBoardroomSession();
    setShowBoardroom(true);
  }, [endBoardroomSession]);

  // ── Assistant column state ──
  const showRecordsTab = analysisMode === 'flags_and_records' || records.length > 0 || subjects.length > 0;
  const counts = useMemo(() => ({ records: records.length, checks: analysisCards.length, actions: actionItems.length }), [records.length, analysisCards.length, actionItems.length]);
  const { unread } = useAssistantUnread(counts, tab);
  const totalUnread = unread.records + unread.checks + unread.actions;
  const problemCount = useMemo(() => (analysisCards as SentenceCheck[]).filter((c) => isProblem(c)).length, [analysisCards]);
  useEffect(() => { if (!showRecordsTab && tab === 'records') setTab('checks'); }, [showRecordsTab, tab]);

  const openCheck = useCallback((check: SentenceCheck) => {
    setSelectedSentenceId(check.sentence_id);
    setSelectedSegmentId(check.segment_id);
    setModalCheck(check);
  }, [setSelectedSentenceId, setSelectedSegmentId]);

  const onSentenceSelect = useCallback((sentenceId: string, check?: SentenceCheck) => {
    if (selectedSentenceId === sentenceId && !check) { setSelectedSentenceId(null); return; }
    setSelectedSentenceId(sentenceId);
    if (check) { setTab('checks'); setModalCheck(check); }
  }, [selectedSentenceId, setSelectedSentenceId]);

  const suggested = scenarioSuggestion ? findScenario(scenarios, scenarioSuggestion.scenario) : undefined;
  const resolvedScenario = sessionAck?.scenario ?? (scenario === 'auto' ? null : scenario);
  const activeProfile = findScenario(scenarios, resolvedScenario ?? undefined);
  const kbEmpty = wsWarning?.code === 'namespace_empty' || (sessionAck != null && sessionAck.kb_docs === 0);
  const activePack = resolvePack();

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
        <BoardroomView
          session={boardroomSession}
          onSessionUpdate={setBoardroomSession}
          onClose={() => setShowBoardroom(false)}
        />
      </div>
    );
  }

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden relative">
      {/* Start modal */}
      {showStartModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setShowStartModal(false)}
        >
          <div
            className="rounded-2xl border border-white/20 bg-slate-900 shadow-xl max-w-lg w-full p-5 space-y-4 max-h-[92vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div>
              <div className="font-semibold text-white">Start transcription</div>
              <p className="text-xs text-slate-400 mt-0.5">
                Pick what kind of conversation this is — the Silent Assistant checks what is said and pulls the right records.
              </p>
            </div>

            {/* Scenario */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Scenario</label>
              <ScenarioPicker value={scenario} onChange={setScenario} scenarios={scenarios} />
            </div>

            {/* Knowledge source + who is on the call */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Knowledge source</label>
                <select
                  value={kbNamespace}
                  onChange={(e) => setKbNamespace(e.target.value)}
                  className="w-full rounded-lg border border-white/20 bg-slate-900 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none min-h-[40px]"
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
                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none min-h-[40px]"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Name</label>
                <input
                  type="text"
                  value={modalName}
                  onChange={(e) => setModalName(e.target.value)}
                  placeholder="e.g. transcript_2025-02-12_14-30"
                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
                <input
                  type="text"
                  value={modalLocation}
                  onChange={(e) => setModalLocation(e.target.value)}
                  placeholder="e.g. Office"
                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </div>
            </div>
            <p className="text-[11px] text-slate-500">Name and location are saved with the transcript every 1 min and used in RAG.</p>

            {/* Boardroom mode toggle in modal */}
            <div className="flex items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => setBoardroomMode(!boardroomMode)}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${boardroomMode ? 'bg-violet-500' : 'bg-white/10'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${boardroomMode ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </button>
              <span className="text-sm text-slate-300">Boardroom Mode</span>
              <span className="text-xs text-slate-500">(records full audio)</span>
            </div>
            <div className="flex flex-wrap gap-2 pt-2">
              <button
                type="button"
                onClick={applyDefault}
                className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15"
              >
                Default
              </button>
              <button
                type="button"
                onClick={() => setShowStartModal(false)}
                className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-400 hover:bg-white/15"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={onStartFromModal}
                className="ml-auto rounded-xl px-4 py-2 text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30"
              >
                Start
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Top header bar */}
      <div
        className={`shrink-0 flex items-center gap-3 px-4 py-3 sm:px-5 sm:py-4 border-b transition-all duration-300 ${
          listening ? 'border-cyan-500/30 bg-cyan-500/5' : 'border-white/10'
        }`}
      >
        {/* Mic button */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => listening && setMicMuted(!micMuted)}
            disabled={!listening}
            className={`relative flex items-center justify-center w-10 h-10 rounded-xl touch-manipulation transition-colors ${
              !listening ? 'bg-white/5 cursor-default' : micMuted ? 'bg-red-500/20 hover:bg-red-500/30' : 'bg-emerald-500/20 hover:bg-emerald-500/30'
            }`}
            aria-label={listening ? (micMuted ? 'Unmute mic' : 'Mute mic') : 'Mic'}
          >
            <ICONS.Mic className={`w-5 h-5 ${!listening ? 'text-slate-400' : micMuted ? 'text-red-400' : 'text-emerald-400'}`} />
            {listening && !micMuted && (
              <span className="absolute inset-0 rounded-xl bg-emerald-400/20 animate-ping" style={{ animationDuration: '1.5s' }} />
            )}
          </button>
          <div>
            <div className="font-semibold text-sm">
              Real-Time Transcription
              {boardroomMode && listening && (
                <span className="ml-2 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-violet-500/20 border border-violet-500/30 text-violet-400">
                  Boardroom
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              {listening ? (
                <>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/20 border border-red-500/40 px-2.5 py-0.5 text-[10px] font-medium text-red-400 uppercase tracking-wider">Live</span>
                  <span className="flex items-center gap-1">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <span key={i} className="w-1 rounded-full bg-emerald-400/80 animate-listening-bar" style={{ height: 8, animationDelay: `${i * 0.12}s` }} />
                    ))}
                  </span>
                  <span className="text-[10px] text-slate-400/90">{micMuted ? 'Muted' : 'Listening…'}</span>
                </>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-white/10 border border-white/10 px-2.5 py-0.5 text-[10px] font-medium text-slate-400 uppercase tracking-wider">Stopped</span>
              )}
            </div>
          </div>
        </div>

        {/* Right controls */}
        <div className="ml-auto flex items-center gap-1.5 flex-wrap">
          {/* Boardroom mode indicator + end button */}
          {boardroomMode && listening && (
            <button
              type="button"
              onClick={handleEndBoardroom}
              disabled={boardroomUploading}
              className="rounded-xl px-3 py-2 text-xs font-semibold bg-violet-500/20 text-violet-400 border border-violet-500/30 hover:bg-violet-500/30 disabled:opacity-50 transition-colors touch-manipulation min-h-[40px]"
            >
              {boardroomUploading ? 'Uploading…' : 'End Boardroom'}
            </button>
          )}

          {/* Boardroom view button if session exists */}
          {boardroomSession && !listening && (
            <button
              type="button"
              onClick={() => setShowBoardroom(true)}
              className="rounded-xl px-3 py-2 text-xs font-semibold bg-violet-500/20 text-violet-400 border border-violet-500/30 hover:bg-violet-500/30 touch-manipulation min-h-[40px]"
            >
              View Boardroom
            </button>
          )}

          {/* History button */}
          <button
            type="button"
            onClick={() => setShowHistory(true)}
            className="shrink-0 p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Session history"
            title="Session history"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>

          <button
            type="button"
            onClick={() => setShowWordCloud(true)}
            className="shrink-0 p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Word cloud"
            title="Word cloud"
          >
            <ICONS.WordCloud className="w-5 h-5" />
          </button>
          <button
            type="button"
            onClick={clearAndReset}
            className="shrink-0 p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Clear transcript"
            title="Clear"
          >
            <ICONS.Trash className="w-5 h-5" />
          </button>
          {wsStatus === 'connecting' && <span className="text-xs text-slate-400">Connecting…</span>}
          {wsStatus === 'loading' && <span className="text-xs text-slate-400">Loading STT…</span>}
          {wsError && (
            <span className="text-xs text-red-400 max-w-[120px] sm:max-w-[200px] truncate" title={wsError}>
              {wsError}
            </span>
          )}
          {!listening ? (
            <button
              type="button"
              onClick={openStartModal}
              disabled={wsStatus === 'connecting' || wsStatus === 'loading'}
              className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 transition-colors touch-manipulation"
            >
              Start
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStopAndExtractTags}
              className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors touch-manipulation"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Session info bar */}
      {(listening || sessionName || sessionLocation || sessionStartedAt) && (
        <div className="shrink-0 px-3 sm:px-5 py-3 border-b border-white/10 bg-black/10 flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Name</span>
            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder="Transcript name"
              className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-36 sm:w-48 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]"
            />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Location</span>
            <input
              type="text"
              value={sessionLocation}
              onChange={(e) => setSessionLocation(e.target.value)}
              placeholder="Location"
              className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-24 sm:w-32 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]"
            />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Date &amp; time</span>
            <span className="text-sm text-slate-300">{sessionStartedAt ? formatSessionDateTime(sessionStartedAt) : '—'}</span>
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Tags</span>
            {customTags.map((tag) => (
              <span key={tag} className="inline-flex items-center gap-1 rounded-lg bg-white/10 border border-white/10 px-2 py-1.5 text-xs text-white/90">
                {tag}
                <button
                  type="button"
                  onClick={() => removeTag(tag)}
                  className="text-slate-400 hover:text-white leading-none p-0.5 touch-manipulation min-w-[28px]"
                  aria-label={`Remove ${tag}`}
                >
                  ×
                </button>
              </span>
            ))}
            <input
              type="text"
              value={newTagInput}
              onChange={(e) => setNewTagInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
              placeholder="+ Add tag"
              className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-20 sm:w-24 min-w-0 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]"
            />
            <button
              type="button"
              onClick={addTag}
              className="rounded-lg px-3 py-2 text-xs font-medium bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 touch-manipulation min-h-[40px]"
            >
              Add
            </button>
          </div>
        </div>
      )}

      {/* Silent Assistant session bar (scenario / who's speaking / mode) */}
      {(listening || sessionAck) && (
        <div className="shrink-0 px-3 sm:px-5 py-2 border-b border-white/10 bg-black/20 flex flex-wrap items-center gap-2">
          <ScenarioPicker
            variant="pill"
            value={scenario}
            onChange={setScenario}
            scenarios={scenarios}
            resolved={sessionAck?.scenario ?? null}
          />
          <RoleToggle roles={roles} value={myRole} onChange={setSpeakerRole} disabled={!listening} />
          {/* Analysis mode segmented control */}
          <div className="inline-flex items-center rounded-xl border border-white/10 bg-white/5 p-0.5" role="group" aria-label="Analysis mode">
            {([['flags_only', 'Problems only'], ['flags_and_records', 'Problems + records']] as [AnalysisMode, string][]).map(([m, label]) => (
              <button
                key={m}
                type="button"
                onClick={() => setAnalysisMode(m)}
                aria-pressed={analysisMode === m}
                className={`px-2.5 py-1.5 text-xs font-semibold rounded-lg transition-colors touch-manipulation min-h-[32px] ${
                  analysisMode === m ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {sessionAck && (
            <span className="text-[10px] text-slate-500 hidden lg:inline" title={`Namespace: ${sessionAck.namespace || 'all'}`}>
              {sessionAck.kb_docs} doc{sessionAck.kb_docs === 1 ? '' : 's'} · {sessionAck.namespace ? sessionAck.namespace : 'all documents'}
            </span>
          )}

          {/* Scenario suggestion chip */}
          {scenarioSuggestion && suggested && (
            <div className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/40 bg-amber-500/10 pl-3 pr-1 py-1 text-xs text-amber-200">
              <span>{SCENARIO_ICONS[suggested.id] ?? '💬'}</span>
              <span>Looks like a <b>{suggested.label.toLowerCase()}</b> — switch?</span>
              <button
                type="button"
                onClick={acceptScenarioSuggestion}
                className="rounded-full px-2 py-0.5 text-[11px] font-semibold bg-amber-400/20 hover:bg-amber-400/30 text-amber-100"
              >
                Switch
              </button>
              <button
                type="button"
                onClick={dismissScenarioSuggestion}
                className="rounded-full p-1 text-amber-300/70 hover:text-white hover:bg-white/10"
                aria-label="Dismiss suggestion"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          )}

          {/* Warning strip */}
          {wsWarning && (
            <div className={`basis-full sm:basis-auto sm:ml-auto inline-flex items-center gap-2 rounded-lg border px-2.5 py-1 text-[11px] ${
              wsWarning.code === 'namespace_empty' ? 'border-amber-400/40 bg-amber-500/10 text-amber-200'
              : wsWarning.code === 'overloaded' || wsWarning.code === 'stt_dropped' ? 'border-rose-400/40 bg-rose-500/10 text-rose-200'
              : 'border-white/15 bg-white/5 text-slate-300'
            }`}>
              <span className="font-semibold uppercase tracking-wider text-[9px] opacity-80">{wsWarning.code.replace(/_/g, ' ')}</span>
              <span className="truncate max-w-[40ch]" title={wsWarning.message}>{wsWarning.message}</span>
              <button type="button" onClick={clearWarning} className="p-0.5 rounded hover:bg-white/10" aria-label="Dismiss warning">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Main content: dual panel */}
      <div className="flex-1 min-h-0 flex gap-0 overflow-hidden">
        {/* LEFT: Transcript panel */}
        <div className="flex-1 min-w-0 min-h-0 p-4 sm:p-5 overflow-auto md:border-r border-white/10">
          <div className="rounded-2xl border border-white/10 bg-black/20 p-4 min-h-[240px] flex flex-col h-full">
            <div className="text-xs font-semibold opacity-60 mb-3 shrink-0 flex items-center gap-2 flex-wrap">
              Live transcript
              {analysisCards.length > 0 && (
                <span className="text-xs text-slate-500 font-normal">(highlighted = checked · hover for proof · click for details)</span>
              )}
              {activeProfile && listening && (
                <span className="ml-auto text-[10px] text-slate-500 font-normal">{SCENARIO_ICONS[activeProfile.id]} {activeProfile.label}</span>
              )}
            </div>
            <div className="flex-1 min-h-0 text-sm leading-relaxed overflow-auto" onClick={() => setSelectedSentenceId(null)}>
              {transcriptSegments.length > 0 ? (
                <div className="space-y-1">
                  {transcriptSegments.map((seg) => (
                    <TranscriptSegmentLine
                      key={seg.paragraph_id}
                      segment={seg}
                      checks={checks}
                      sentenceStatus={sentenceStatus}
                      vocab={tagVocab}
                      roles={roles}
                      selectedSentenceId={selectedSentenceId}
                      onSelectSentence={onSentenceSelect}
                      isSelected={seg.paragraph_id === selectedSegmentId}
                      onSelectSegment={() =>
                        setSelectedSegmentId(seg.paragraph_id === selectedSegmentId ? null : seg.paragraph_id)
                      }
                    />
                  ))}
                  {/* Live partial: text being actively recognised, not yet a committed segment */}
                  {partial && (
                    <p className="text-slate-300 opacity-75 leading-relaxed">{partial}</p>
                  )}
                  <div ref={transcriptEndRef} />
                </div>
              ) : (
                /* No committed segments yet — show streaming text directly */
                <span className="text-white/90">
                  {fullTranscript || (
                    <span className="text-slate-500 opacity-70">—</span>
                  )}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Assistant column (md+) */}
        <div className="hidden md:block w-72 lg:w-80 xl:w-96 shrink-0 min-h-0 bg-black/10">
          {sidebar}
        </div>
      </div>

      {/* Mobile: bottom-sheet toggle + sheet */}
      <button
        type="button"
        onClick={() => setSheetOpen(true)}
        className="md:hidden absolute bottom-4 right-4 z-30 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-slate-900/95 backdrop-blur px-4 py-2.5 text-xs font-semibold text-cyan-300 shadow-xl touch-manipulation min-h-[44px]"
        aria-label="Open Silent Assistant"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
        Assistant
        {(totalUnread > 0 || problemCount > 0) && (
          <span className={`rounded-full px-1.5 py-0.5 text-[10px] tabular-nums ${problemCount > 0 ? 'bg-rose-500/30 text-rose-100' : 'bg-cyan-500/30 text-cyan-100'}`}>
            {totalUnread > 0 ? `+${totalUnread}` : problemCount}
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
      {modalCheck && (
        <AnalysisCardModal card={modalCheck} vocab={tagVocab} onClose={() => setModalCheck(null)} />
      )}

      {/* Word cloud modal */}
      {showWordCloud && (
        <WordCloudModal
          onClose={() => setShowWordCloud(false)}
          liveText={[fullTranscript, partial].filter(Boolean).join(' ')}
          listening={listening}
        />
      )}
    </div>
  );
};

export default LiveTranscription;
