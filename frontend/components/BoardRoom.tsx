/**
 * Board Room Mode component.
 *
 * Phases:
 *   1. IDLE       – start screen with session title/location input
 *   2. LISTENING  – active session: multi-speaker live captions
 *   3. PROCESSING – EOS sent; waiting for final transcript + report generation
 *   4. REPORT     – report ready; interactive report viewer with export
 *   5. HISTORY    – past sessions list with reload / delete
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ICONS } from '../constants';
import {
  ActionItem,
  BoardRoomReport,
  BoardRoomSession,
  KeyDiscussionPoint,
  RagEvidence,
  ReportRow,
  SpeakerTurn,
  SPEAKER_COLORS,
  WsIncoming,
  boardroomWsUrl,
  deleteBoardRoomSession,
  downloadExport,
  getBoardRoomSession,
  getSessionReport,
  listBoardRoomSessions,
  triggerReportGeneration,
} from '../services/boardroom';

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatDuration(sec: number | null | undefined): string {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return iso;
  }
}

function getSpeakerColor(name: string, speakers: string[]): string {
  const idx = speakers.indexOf(name);
  return SPEAKER_COLORS[(idx >= 0 ? idx : 0) % SPEAKER_COLORS.length];
}

type Phase = 'idle' | 'listening' | 'processing' | 'report' | 'history';

// ── Sub-components ────────────────────────────────────────────────────────────

function SpeakerBadge({ name, speakers }: { name: string; speakers: string[] }) {
  const color = getSpeakerColor(name, speakers);
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide mr-2 shrink-0"
      style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
    >
      {name}
    </span>
  );
}

function PulsingDot({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
      <span className="text-xs text-red-400 font-semibold uppercase tracking-wide">Recording</span>
    </span>
  );
}

function ReportSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-px flex-1 bg-cyan-500/20" />
        <h3 className="text-xs font-bold uppercase tracking-widest text-cyan-400/70 shrink-0 px-2">{title}</h3>
        <div className="h-px flex-1 bg-cyan-500/20" />
      </div>
      {children}
    </div>
  );
}

function EvidenceCard({ ev }: { ev: RagEvidence }) {
  return (
    <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-3 mb-2">
      <div className="flex items-start gap-2 mb-1">
        <ICONS.File className="w-3.5 h-3.5 text-violet-400 shrink-0 mt-0.5" />
        <span className="text-[10px] font-semibold text-violet-400 uppercase tracking-wide">
          {ev.source}{ev.section ? ` › ${ev.section}` : ''}{ev.page ? ` · p.${ev.page}` : ''} · score {ev.score.toFixed(2)}
        </span>
      </div>
      <p className="text-xs text-slate-300 leading-relaxed">{ev.text.slice(0, 300)}{ev.text.length > 300 ? '…' : ''}</p>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

const BoardRoom: React.FC = () => {
  const [phase, setPhase] = useState<Phase>('idle');
  const [sessionTitle, setSessionTitle] = useState('');
  const [sessionLocation, setSessionLocation] = useState('');

  // Live session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  // liveTurns: merged consecutive same-speaker blocks (what the backend sends after turn-accumulation)
  const [liveTurns, setLiveTurns] = useState<SpeakerTurn[]>([]);
  const [allSpeakers, setAllSpeakers] = useState<string[]>([]);
  const [speakerCount, setSpeakerCount] = useState(0);
  const [duration, setDuration] = useState(0);
  const [wsStatus, setWsStatus] = useState<'connecting' | 'ready' | 'error' | 'closed'>('connecting');
  const [wsError, setWsError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [chunksReceived, setChunksReceived] = useState(0);
  // Processing pipeline stage (shown in the timeline after Stop)
  const [processingStage, setProcessingStage] = useState<
    'idle' | 'listening' | 'finalizing_audio' | 'transcribing' | 'rag_ingesting' | 'analyzing' | 'exporting' | 'completed'
  >('idle');
  const [transcriptionSource, setTranscriptionSource] = useState<string | null>(null);

  // Report state
  const [report, setReport] = useState<BoardRoomReport | null>(null);
  const [reportStatus, setReportStatus] = useState<ReportRow['status']>('not_generated');
  const [reportSessionId, setReportSessionId] = useState<string | null>(null);
  const [activeReportTab, setActiveReportTab] = useState<'summary' | 'points' | 'decisions' | 'actions' | 'evidence' | 'recommendations'>('summary');
  const [exporting, setExporting] = useState<'pdf' | 'pptx' | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [expandedTopics, setExpandedTopics] = useState<Set<number>>(new Set([0]));

  // History
  const [sessions, setSessions] = useState<BoardRoomSession[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const liveScrollRef = useRef<HTMLDivElement | null>(null);
  const reportPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Audio helpers ──────────────────────────────────────────────────────────

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
  }, []);

  const startAudio = useCallback(async (ws: WebSocket) => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;

    const ctx = new AudioContext({ sampleRate: 16000 });
    audioContextRef.current = ctx;

    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));
      }
      ws.send(pcm16.buffer);
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  }, []);

  // ── WebSocket lifecycle ────────────────────────────────────────────────────

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connectAndStart = useCallback(async () => {
    if (!sessionTitle.trim()) return;
    setPhase('listening');
    setLiveTurns([]);
    setAllSpeakers([]);
    setSpeakerCount(0);
    setDuration(0);
    setWsError(null);
    setWsStatus('connecting');

    const ws = new WebSocket(boardroomWsUrl());
    wsRef.current = ws;
    let started = false;

    ws.onopen = () => {
      // Wait for "ready" from server before sending "start"
    };

    ws.onmessage = async (evt) => {
      if (typeof evt.data !== 'string') return;
      let msg: WsIncoming;
      try { msg = JSON.parse(evt.data); } catch { return; }

      if (msg.type === 'loading') {
        setWsStatus('connecting');
        setProcessingStage('idle');
        return;
      }
      if (msg.type === 'ready' && !started) {
        started = true;
        setWsStatus('ready');
        setProcessingStage('listening');
        // Send start command
        ws.send(JSON.stringify({
          type: 'start',
          title: sessionTitle.trim() || 'Board Room Session',
          location: sessionLocation.trim() || 'default',
          sample_rate: 16000,
          format: 'pcm16',
          mode: 'boardroom',
        }));
        // Begin capturing audio
        await startAudio(ws).catch((err) => {
          setWsError(`Microphone access denied: ${err.message}`);
          setWsStatus('error');
        });
        if ((msg as any).session_id) setSessionId((msg as any).session_id);
        startTimeRef.current = Date.now();
        durationTimerRef.current = setInterval(() => {
          setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
        }, 1000);
        return;
      }
      // Spec-aligned session_started (also serves as the "ready" confirmation after start cmd)
      if (msg.type === 'session_started') {
        setSessionId(msg.session_id);
        setProcessingStage('listening');
        setWsStatus('ready');
        return;
      }
      if (msg.type === 'ready' && started && (msg as any).session_id) {
        setSessionId((msg as any).session_id);
        return;
      }
      // Periodic listening status heartbeat from backend
      if (msg.type === 'listening_status') {
        setDuration(Math.floor(msg.duration_seconds));
        setAudioLevel(Math.round(msg.audio_level * 100));
        setChunksReceived(msg.chunks_received);
        return;
      }
      if (msg.type === 'partial') {
        const turns = msg.turns || [];
        setLiveTurns(turns);
        setSpeakerCount(msg.speaker_count || 0);
        if (msg.audio_level !== undefined) setAudioLevel(Math.round((msg.audio_level ?? 0) * 100));
        if (msg.chunks_received !== undefined) setChunksReceived(msg.chunks_received);
        setAllSpeakers((prev) => {
          const newNames = turns.map((t) => t.speaker).filter((n) => !prev.includes(n));
          return newNames.length > 0 ? [...prev, ...newNames] : prev;
        });
        requestAnimationFrame(() => {
          if (liveScrollRef.current) {
            liveScrollRef.current.scrollTop = liveScrollRef.current.scrollHeight;
          }
        });
        return;
      }
      // Processing pipeline stage updates from backend (finalizing → transcribing → rag_ingesting → analyzing)
      if (msg.type === 'finalizing') {
        const stageMap: Record<string, typeof processingStage> = {
          transcribing: 'transcribing',
          rag_ingesting: 'rag_ingesting',
          analyzing: 'analyzing',
          finalizing_audio: 'finalizing_audio',
          // Sortformer diarization stages
          diarization_start: 'transcribing',
          diarization_failed: 'transcribing',
          // Validation warnings
          speaker_count_mismatch: 'transcribing',
          warning: 'transcribing',
        };
        const stage = stageMap[msg.status] ?? 'transcribing';
        setProcessingStage(stage);
        return;
      }
      // Spec-aligned transcript_completed
      if (msg.type === 'transcript_completed') {
        setSessionId(msg.session_id);
        setSpeakerCount(msg.speaker_count);
        setTranscriptionSource(msg.transcription_source ?? null);
        if (msg.segments?.length) {
          const names = [...new Set(msg.segments.map((s) => s.speaker))];
          setAllSpeakers(names);
        }
        return;
      }
      if (msg.type === 'final') {
        setSessionId(msg.session_id);
        stopAudio();
        setPhase('processing');
        setProcessingStage('analyzing');
        return;
      }
      if (msg.type === 'report_generating') {
        setReportStatus('generating');
        setProcessingStage('analyzing');
        return;
      }
      // Spec-aligned report_completed
      if (msg.type === 'report_completed') {
        setReportStatus('ready');
        setReportSessionId(msg.session_id);
        setProcessingStage('completed');
        try {
          const row = await getSessionReport(msg.session_id);
          if (row.report) { setReport(row.report); setPhase('report'); }
        } catch { setPhase('processing'); }
        return;
      }
      // Legacy report_ready (kept for backward compat)
      if (msg.type === 'report_ready') {
        setReportStatus('ready');
        setReportSessionId(msg.session_id);
        setProcessingStage('completed');
        try {
          const row = await getSessionReport(msg.session_id);
          if (row.report) { setReport(row.report); setPhase('report'); }
        } catch { setPhase('processing'); }
        return;
      }
      if (msg.type === 'report_error') {
        setWsError(`Report generation failed: ${msg.message}`);
        setPhase('processing');
        return;
      }
      if (msg.type === 'error') {
        setWsError(msg.message);
        setWsStatus('error');
        stopAudio();
        return;
      }
    };

    ws.onclose = () => {
      setWsStatus('closed');
      stopAudio();
    };

    ws.onerror = () => {
      setWsError('WebSocket connection error');
      setWsStatus('error');
      stopAudio();
    };
  }, [sessionTitle, sessionLocation, startAudio, stopAudio]);

  const stopSession = useCallback(() => {
    stopAudio();
    setProcessingStage('finalizing_audio');
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
    }
  }, [stopAudio]);

  // ── History ────────────────────────────────────────────────────────────────

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const { sessions: s } = await listBoardRoomSessions();
      setSessions(s);
    } catch {
      setSessions([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleViewHistory = useCallback(() => {
    setPhase('history');
    loadHistory();
  }, [loadHistory]);

  const handleDeleteSession = useCallback(async (id: string) => {
    if (!window.confirm('Delete this Board Room session? This cannot be undone.')) return;
    await deleteBoardRoomSession(id);
    loadHistory();
  }, [loadHistory]);

  const handleViewReport = useCallback(async (sessionId: string) => {
    setReportStatus('generating');
    setReport(null);
    setPhase('processing');
    setReportSessionId(sessionId);
    try {
      const row = await getSessionReport(sessionId);
      if (row.status === 'ready' && row.report) {
        setReport(row.report);
        setReportStatus('ready');
        setPhase('report');
      } else if (row.status === 'not_generated' || row.status === 'failed') {
        // Trigger generation
        await triggerReportGeneration(sessionId);
        setReportStatus('generating');
        // Poll for completion
        startReportPolling(sessionId);
      } else {
        setReportStatus(row.status as any);
        startReportPolling(sessionId);
      }
    } catch (e) {
      setWsError((e as Error).message);
      setPhase('history');
    }
  }, []);

  const startReportPolling = useCallback((sessionId: string) => {
    if (reportPollRef.current) clearInterval(reportPollRef.current);
    reportPollRef.current = setInterval(async () => {
      try {
        const row = await getSessionReport(sessionId);
        setReportStatus(row.status as any);
        if (row.status === 'ready' && row.report) {
          clearInterval(reportPollRef.current!);
          reportPollRef.current = null;
          setReport(row.report);
          setPhase('report');
        } else if (row.status === 'failed') {
          clearInterval(reportPollRef.current!);
          reportPollRef.current = null;
          setWsError('Report generation failed');
        }
      } catch {
        // ignore transient poll errors
      }
    }, 3000);
  }, []);

  // ── Export ─────────────────────────────────────────────────────────────────

  const handleExport = useCallback(async (format: 'pdf' | 'pptx') => {
    const sid = reportSessionId || report?.session_id;
    if (!sid) return;
    setExporting(format);
    setExportError(null);
    try {
      await downloadExport(sid, format);
    } catch (e) {
      setExportError((e as Error).message);
    } finally {
      setExporting(null);
    }
  }, [reportSessionId, report]);

  // ── Cleanup ────────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      closeWs();
      stopAudio();
      if (reportPollRef.current) clearInterval(reportPollRef.current);
    };
  }, [closeWs, stopAudio]);

  // ── Render ─────────────────────────────────────────────────────────────────

  // IDLE phase
  if (phase === 'idle') {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
        <div className="w-full max-w-lg">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 mb-4">
              <ICONS.BoardRoom className="w-8 h-8 text-cyan-400" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Board Room Mode</h2>
            <p className="text-sm text-slate-400 max-w-sm mx-auto">
              Multi-speaker meeting capture powered by NVIDIA Parakeet Multitalker ASR.
              Generates comprehensive RAG-enhanced reports with PDF/PPTX export.
            </p>
          </div>

          {/* Session setup form */}
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6 mb-4">
            <div className="mb-4">
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Session Title</label>
              <input
                type="text"
                value={sessionTitle}
                onChange={(e) => setSessionTitle(e.target.value)}
                placeholder="e.g. Q3 Strategy Review"
                className="w-full rounded-xl px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-cyan-500/50 focus:bg-white/10 transition-colors"
              />
            </div>
            <div className="mb-6">
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Location <span className="text-slate-600 font-normal normal-case">(optional)</span></label>
              <input
                type="text"
                value={sessionLocation}
                onChange={(e) => setSessionLocation(e.target.value)}
                placeholder="e.g. Conference Room A"
                className="w-full rounded-xl px-4 py-3 bg-white/5 border border-white/10 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-cyan-500/50 focus:bg-white/10 transition-colors"
              />
            </div>

            <button
              type="button"
              onClick={connectAndStart}
              disabled={!sessionTitle.trim()}
              className="w-full flex items-center justify-center gap-3 py-3.5 rounded-xl font-semibold text-sm transition-all
                bg-cyan-500 hover:bg-cyan-400 text-[#05070a]
                disabled:opacity-40 disabled:cursor-not-allowed
                shadow-[0_0_30px_rgba(34,211,238,0.2)]"
            >
              <ICONS.Play className="w-4 h-4" />
              Start Board Room Session
            </button>
          </div>

          {/* History button */}
          <button
            type="button"
            onClick={handleViewHistory}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm text-slate-400 hover:text-white border border-white/10 hover:border-white/20 hover:bg-white/5 transition-all"
          >
            <ICONS.Transcript className="w-4 h-4" />
            View Past Sessions
          </button>

          {/* Model info */}
          <p className="text-center text-[10px] text-slate-600 mt-4 leading-relaxed">
            Powered by nvidia/multitalker-parakeet-streaming-0.6b-v1 · TensorRT-LLM · RAG
          </p>
        </div>
      </div>
    );
  }

  // LISTENING phase
  if (phase === 'listening') {
    return (
      <div className="flex flex-col h-full min-h-0">
        {/* Session header */}
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <ICONS.BoardRoom className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white text-sm">{sessionTitle || 'Board Room Session'}</h3>
              {sessionLocation && <p className="text-[10px] text-slate-500">{sessionLocation}</p>}
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <PulsingDot active={wsStatus === 'ready'} />
            <span className="text-sm text-slate-400 font-mono">{formatDuration(duration)}</span>
            {speakerCount > 0 && (
              <span className="text-xs text-slate-500">
                <ICONS.Users className="w-3.5 h-3.5 inline mr-1" />{speakerCount} speaker{speakerCount !== 1 ? 's' : ''}
              </span>
            )}
            {/* Audio level bar */}
            {wsStatus === 'ready' && (
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-slate-600">Level</span>
                <div className="w-16 h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-400 transition-all duration-200"
                    style={{ width: `${Math.min(100, audioLevel)}%` }}
                  />
                </div>
              </div>
            )}
            {chunksReceived > 0 && (
              <span className="text-[10px] text-slate-600 hidden sm:inline">{chunksReceived} chunks</span>
            )}
          </div>
        </div>

        {/* Status bar */}
        {wsStatus === 'connecting' && (
          <div className="flex items-center gap-2 mb-3 text-xs text-cyan-400">
            <ICONS.Loader className="w-3.5 h-3.5 animate-spin" />
            Connecting to Parakeet ASR model…
          </div>
        )}
        {wsError && (
          <div className="mb-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400 flex items-center gap-2">
            <ICONS.AlertCircle className="w-4 h-4 shrink-0" />{wsError}
          </div>
        )}

        {/* Speaker legend */}
        {allSpeakers.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {allSpeakers.map((sp) => (
              <SpeakerBadge key={sp} name={sp} speakers={allSpeakers} />
            ))}
          </div>
        )}

        {/* Live captions — turn-based (one block per speaker turn) */}
        <div
          ref={liveScrollRef}
          className="flex-1 min-h-0 overflow-y-auto rounded-2xl border border-white/10 bg-white/3 p-4 space-y-3"
        >
          {liveTurns.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-slate-500">
              <ICONS.Mic className="w-6 h-6 mb-2 animate-pulse" />
              <p className="text-sm">Listening for speech…</p>
            </div>
          ) : (
            liveTurns.map((turn, i) => {
              const isLast = i === liveTurns.length - 1;
              return (
                <div
                  key={i}
                  className={`flex items-start gap-2 ${isLast ? 'pb-1' : ''}`}
                >
                  <SpeakerBadge name={turn.speaker} speakers={allSpeakers} />
                  <p className={`text-sm leading-relaxed flex-1 ${isLast ? 'text-white' : 'text-slate-300'}`}>
                    {turn.text}
                    {/* Blinking cursor on the last active turn */}
                    {isLast && wsStatus === 'ready' && (
                      <span className="inline-block w-0.5 h-4 bg-cyan-400 ml-0.5 align-middle animate-pulse" />
                    )}
                  </p>
                </div>
              );
            })
          )}
        </div>

        {/* Stop button */}
        <div className="mt-4 flex gap-3">
          <button
            type="button"
            onClick={stopSession}
            className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-xl font-semibold text-sm
              bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 hover:border-red-500/50 transition-all"
          >
            <ICONS.Stop className="w-4 h-4" />
            Stop Session & Generate Report
          </button>
        </div>
      </div>
    );
  }

  // PROCESSING phase
  if (phase === 'processing') {
    // Processing timeline stages in order
    const timelineStages: { key: typeof processingStage; label: string; detail: string }[] = [
      { key: 'finalizing_audio', label: 'Finalizing Audio', detail: 'Closing WAV file and validating recording' },
      { key: 'transcribing', label: 'Transcribing Speakers', detail: 'Running NVIDIA Parakeet multitalker ASR' },
      { key: 'rag_ingesting', label: 'Ingesting into RAG', detail: 'Adding transcript to knowledge base' },
      { key: 'analyzing', label: 'Generating Report', detail: 'TensorRT-LLM + RAG analysis' },
      { key: 'completed', label: 'Completed', detail: 'Report ready' },
    ];
    const stageOrder = timelineStages.map((s) => s.key);
    const currentIdx = stageOrder.indexOf(processingStage);

    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] px-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/20 mb-4">
              <ICONS.Loader className="w-8 h-8 text-violet-400 animate-spin" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">
              {processingStage === 'analyzing' ? 'Generating Report…' : 'Processing Session…'}
            </h3>
            <p className="text-sm text-slate-400">
              {transcriptionSource === 'vibevoice_fallback'
                ? 'Primary transcription failed — using VibeVoice-ASR fallback.'
                : 'Board Room pipeline running. Do not close this tab.'}
            </p>
          </div>

          {/* Processing timeline */}
          <div className="space-y-2 mb-6">
            {timelineStages.map((stage, idx) => {
              const done = idx < currentIdx;
              const active = idx === currentIdx;
              return (
                <div
                  key={stage.key}
                  className={`flex items-center gap-3 rounded-xl px-4 py-3 border transition-all ${
                    done
                      ? 'border-emerald-500/30 bg-emerald-500/5'
                      : active
                      ? 'border-cyan-500/40 bg-cyan-500/10'
                      : 'border-white/5 bg-white/2 opacity-40'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                    done ? 'bg-emerald-500' : active ? 'bg-cyan-500 animate-pulse' : 'bg-slate-700'
                  }`}>
                    {done ? (
                      <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="text-[9px] font-bold text-white">{idx + 1}</span>
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className={`text-sm font-semibold ${done ? 'text-emerald-400' : active ? 'text-cyan-300' : 'text-slate-500'}`}>
                      {stage.label}
                    </div>
                    {(done || active) && (
                      <div className="text-xs text-slate-500 truncate">{stage.detail}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {wsError && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400 flex items-center gap-2 mb-4">
              <ICONS.AlertCircle className="w-4 h-4 shrink-0" />{wsError}
            </div>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setPhase('idle')}
              className="flex-1 py-2.5 rounded-xl text-sm text-slate-400 border border-white/10 hover:border-white/20 hover:text-white transition-all"
            >
              Back to Start
            </button>
            {(reportSessionId || sessionId) && (
              <button
                type="button"
                onClick={() => handleViewReport(reportSessionId || sessionId!)}
                className="flex-1 py-2.5 rounded-xl text-sm text-cyan-400 border border-cyan-500/30 hover:border-cyan-500/50 hover:bg-cyan-500/10 transition-all"
              >
                Check Report
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // REPORT phase
  if (phase === 'report' && report) {
    const tabs: { id: typeof activeReportTab; label: string; count?: number }[] = [
      { id: 'summary', label: 'Summary' },
      { id: 'points', label: 'Discussion', count: report.key_discussion_points?.length },
      { id: 'decisions', label: 'Decisions', count: report.decisions?.length },
      { id: 'actions', label: 'Actions', count: report.action_items?.length },
      { id: 'evidence', label: 'Evidence', count: report.rag_evidence?.length },
      { id: 'recommendations', label: 'Insights', count: report.recommendations?.length },
    ];

    return (
      <div className="flex flex-col h-full min-h-0">
        {/* Report header */}
        <div className="flex items-start justify-between gap-4 mb-4 flex-wrap">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <ICONS.CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wide">Report Ready</span>
            </div>
            <h2 className="text-lg font-bold text-white">{report.topics?.[0] ? `Board Room: ${report.topics.slice(0, 2).join(', ')}` : 'Board Room Report'}</h2>
            <p className="text-xs text-slate-500 mt-0.5">{formatDate(report.generated_at)}</p>
          </div>
          {/* Export buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={() => handleExport('pdf')}
              disabled={exporting !== null}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold text-white bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 hover:border-red-500/50 transition-all disabled:opacity-50"
            >
              {exporting === 'pdf' ? <ICONS.Loader className="w-3.5 h-3.5 animate-spin" /> : <ICONS.Download className="w-3.5 h-3.5" />}
              Export PDF
            </button>
            <button
              type="button"
              onClick={() => handleExport('pptx')}
              disabled={exporting !== null}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold text-white bg-orange-500/20 hover:bg-orange-500/30 border border-orange-500/30 hover:border-orange-500/50 transition-all disabled:opacity-50"
            >
              {exporting === 'pptx' ? <ICONS.Loader className="w-3.5 h-3.5 animate-spin" /> : <ICONS.Download className="w-3.5 h-3.5" />}
              Export PPTX
            </button>
            <button
              type="button"
              onClick={() => { setPhase('idle'); setReport(null); }}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-400 border border-white/10 hover:border-white/20 hover:text-white transition-all"
            >
              <ICONS.Play className="w-3.5 h-3.5" />
              New Session
            </button>
            <button
              type="button"
              onClick={handleViewHistory}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-400 border border-white/10 hover:border-white/20 hover:text-white transition-all"
            >
              <ICONS.Transcript className="w-3.5 h-3.5" />
              History
            </button>
          </div>
        </div>

        {exportError && (
          <div className="mb-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-xs text-red-400 flex items-center gap-2">
            <ICONS.AlertCircle className="w-3.5 h-3.5 shrink-0" />{exportError}
          </div>
        )}

        {/* Participants strip */}
        {report.participants?.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {report.participants.map((p, i) => (
              <span
                key={p.name}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border"
                style={{
                  borderColor: `${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}55`,
                  color: SPEAKER_COLORS[i % SPEAKER_COLORS.length],
                  background: `${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}11`,
                }}
              >
                <ICONS.Users className="w-3 h-3" />
                {p.name} <span className="opacity-60">·{p.speaking_turns}t</span>
              </span>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mb-3 overflow-x-auto pb-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveReportTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeReportTab === tab.id
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                  : 'text-slate-400 hover:text-white border border-transparent hover:bg-white/5'
              }`}
            >
              {tab.label}
              {tab.count !== undefined && tab.count > 0 && (
                <span className={`inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] ${activeReportTab === tab.id ? 'bg-cyan-500/30 text-cyan-300' : 'bg-white/10 text-slate-400'}`}>
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          {/* Summary */}
          {activeReportTab === 'summary' && (
            <div className="space-y-4">
              <ReportSection title="Executive Summary">
                <div className="rounded-xl bg-white/5 border border-white/10 p-4">
                  <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{report.executive_summary || '—'}</p>
                </div>
              </ReportSection>
              {report.topics?.length > 0 && (
                <ReportSection title="Topics Covered">
                  <div className="flex flex-wrap gap-2">
                    {report.topics.map((topic) => (
                      <span key={topic} className="px-3 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">{topic}</span>
                    ))}
                  </div>
                </ReportSection>
              )}
              {report.contradictions?.length > 0 && (
                <ReportSection title="Contradictions & Gaps">
                  <div className="space-y-2">
                    {report.contradictions.map((c, i) => (
                      <div key={i} className="flex items-start gap-2 rounded-lg bg-amber-500/5 border border-amber-500/20 p-3">
                        <ICONS.AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                        <p className="text-sm text-slate-300">{c}</p>
                      </div>
                    ))}
                  </div>
                </ReportSection>
              )}
            </div>
          )}

          {/* Key Discussion Points */}
          {activeReportTab === 'points' && (
            <div className="space-y-3">
              {(report.key_discussion_points || []).map((kp: KeyDiscussionPoint, i: number) => (
                <div key={i} className="rounded-xl border border-white/10 bg-white/3 overflow-hidden">
                  <button
                    type="button"
                    className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-white/5 transition-colors"
                    onClick={() => setExpandedTopics((prev) => {
                      const next = new Set(prev);
                      if (next.has(i)) next.delete(i); else next.add(i);
                      return next;
                    })}
                  >
                    <span className="font-semibold text-sm text-white">{kp.topic}</span>
                    <span className="text-slate-500 text-lg">{expandedTopics.has(i) ? '−' : '+'}</span>
                  </button>
                  {expandedTopics.has(i) && (
                    <div className="px-4 pb-4">
                      <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap mb-3">{kp.summary}</p>
                      {kp.rag_evidence?.length > 0 && (
                        <div>
                          <p className="text-[10px] uppercase tracking-wider text-violet-400/70 font-semibold mb-2">Supporting Evidence</p>
                          {kp.rag_evidence.slice(0, 2).map((ev: RagEvidence, j: number) => (
                            <EvidenceCard key={j} ev={ev} />
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {!report.key_discussion_points?.length && (
                <p className="text-sm text-slate-500 text-center py-8">No discussion points extracted.</p>
              )}
            </div>
          )}

          {/* Decisions */}
          {activeReportTab === 'decisions' && (
            <div className="space-y-2">
              {(report.decisions || []).map((d: string, i: number) => (
                <div key={i} className="flex items-start gap-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 px-4 py-3">
                  <ICONS.CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-slate-200">{d}</p>
                </div>
              ))}
              {!report.decisions?.length && (
                <p className="text-sm text-slate-500 text-center py-8">No explicit decisions identified.</p>
              )}
            </div>
          )}

          {/* Action Items */}
          {activeReportTab === 'actions' && (
            <div>
              <div className="rounded-xl border border-white/10 overflow-hidden">
                <div className="grid grid-cols-[1fr_auto_auto] gap-4 px-4 py-2.5 bg-white/5 border-b border-white/10">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Action Item</span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Owner</span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Priority</span>
                </div>
                {(report.action_items || []).map((item: ActionItem, i: number) => {
                  const priColor = item.priority === 'High' ? 'text-red-400' : item.priority === 'Low' ? 'text-slate-400' : 'text-amber-400';
                  return (
                    <div key={i} className={`grid grid-cols-[1fr_auto_auto] gap-4 px-4 py-3 border-b border-white/5 ${i % 2 === 0 ? 'bg-white/2' : ''}`}>
                      <p className="text-sm text-slate-200">{item.item}</p>
                      <p className="text-sm text-slate-400 whitespace-nowrap">{item.owner}</p>
                      <p className={`text-sm font-semibold whitespace-nowrap ${priColor}`}>{item.priority}</p>
                    </div>
                  );
                })}
              </div>
              {!report.action_items?.length && (
                <p className="text-sm text-slate-500 text-center py-8">No action items identified.</p>
              )}
            </div>
          )}

          {/* RAG Evidence */}
          {activeReportTab === 'evidence' && (
            <div className="space-y-2">
              {(report.rag_evidence || []).map((ev: RagEvidence, i: number) => (
                <EvidenceCard key={i} ev={ev} />
              ))}
              {!report.rag_evidence?.length && (
                <p className="text-sm text-slate-500 text-center py-8">No matching knowledge base evidence found.</p>
              )}
            </div>
          )}

          {/* Recommendations */}
          {activeReportTab === 'recommendations' && (
            <div className="space-y-2">
              {(report.recommendations || []).map((r: string, i: number) => (
                <div key={i} className="flex items-start gap-3 rounded-xl bg-cyan-500/5 border border-cyan-500/20 px-4 py-3">
                  <ICONS.Zap className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-slate-200">{r}</p>
                </div>
              ))}
              {!report.recommendations?.length && (
                <p className="text-sm text-slate-500 text-center py-8">No recommendations generated.</p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  // HISTORY phase
  if (phase === 'history') {
    return (
      <div className="flex flex-col h-full min-h-0">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-white">Past Board Room Sessions</h3>
          <button
            type="button"
            onClick={() => setPhase('idle')}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-slate-400 border border-white/10 hover:border-white/20 hover:text-white transition-all"
          >
            <ICONS.Play className="w-3.5 h-3.5" />
            New Session
          </button>
        </div>

        {historyLoading ? (
          <div className="flex items-center justify-center py-12 text-slate-500">
            <ICONS.Loader className="w-5 h-5 animate-spin mr-2" />
            Loading sessions…
          </div>
        ) : sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500">
            <ICONS.BoardRoom className="w-8 h-8 mb-3 opacity-30" />
            <p className="text-sm">No Board Room sessions yet.</p>
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-y-auto space-y-2">
            {sessions.map((s) => (
              <div key={s.id} className="rounded-xl border border-white/10 bg-white/3 hover:bg-white/5 transition-colors p-4 flex items-center justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium text-white text-sm truncate">{s.title}</h4>
                    <span className={`shrink-0 px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide ${
                      s.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
                      s.status === 'active' ? 'bg-red-500/20 text-red-400' :
                      'bg-slate-500/20 text-slate-400'
                    }`}>{s.status}</span>
                  </div>
                  <div className="flex items-center gap-3 text-[10px] text-slate-500">
                    {s.location && <span>{s.location}</span>}
                    {s.location && <span>·</span>}
                    <span>{formatDate(s.started_at)}</span>
                    {s.duration_sec && <><span>·</span><span>{formatDuration(s.duration_sec)}</span></>}
                    {s.speaker_count > 0 && <><span>·</span><span><ICONS.Users className="w-3 h-3 inline mr-0.5" />{s.speaker_count} speakers</span></>}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {s.status === 'completed' && (
                    <button
                      type="button"
                      onClick={() => handleViewReport(s.id)}
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-cyan-400 border border-cyan-500/30 hover:border-cyan-500/60 hover:bg-cyan-500/10 transition-all"
                    >
                      <ICONS.Report className="w-3.5 h-3.5" />
                      Report
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => handleDeleteSession(s.id)}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                    aria-label="Delete session"
                  >
                    <ICONS.Trash className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return null;
};

export default BoardRoom;
