import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ICONS } from "../constants";
import { useAssistantSpeak } from "../hooks/useAssistantSpeak";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import {
  analyzeAssistantTranscript,
  defaultTranscriptName,
  dismissAssistantSuggestion,
  ignoreAssistantSuggestion,
  listAssistantSuggestions,
  approveAssistantSuggestion,
  markAssistantSuggestionSpoken,
} from "../services/backend";
import type { AppSettings, DocumentChunk, Suggestion } from "../types";
import { mapCitations } from "../utils/mapCitations";
import { CITATION_CHIP_CLASS, evidenceLabel, sourceOriginLabel } from "../utils/modeChrome";
import { ChunkCitationModal } from "./KnowledgeChat";
import ProductModeHeader, { type ModeStatusTone } from "./ProductModeHeader";

interface AssistantModeProps {
  liveTranscription: UseLiveTranscriptionReturn;
  settings: AppSettings;
}

/** Match Silent Assistant: periodic KB analysis cadence and prominent UI threshold. */
const ASSISTANT_KB_INTERVAL_MS = 60_000;
const ASSISTANT_MIN_CHARS_FOR_ANALYSIS = 90;
const ASSISTANT_HAND_RAISE_MIN_CONFIDENCE = 0.7;

function findInTranscript(haystack: string, needle: string): { start: number; end: number } | null {
  const n = needle.trim();
  if (!n || !haystack) return null;
  let i = haystack.indexOf(n);
  if (i >= 0) return { start: i, end: i + n.length };
  const compact = n.replace(/\s+/g, " ");
  i = haystack.indexOf(compact);
  if (i >= 0) return { start: i, end: i + compact.length };
  const tail = haystack.slice(-Math.min(haystack.length, n.length + 400));
  i = tail.indexOf(n.slice(0, Math.min(80, n.length)));
  if (i >= 0) {
    const start = haystack.length - tail.length + i;
    return { start, end: Math.min(haystack.length, start + n.length) };
  }
  return null;
}

function anchorTextForSuggestion(s: Suggestion): string | null {
  const te = s.trigger_excerpt?.trim();
  if (te) return te;
  const st = s.short_text;
  const m = st.match(/Latest bit:\s*([\s\S]+?)(?:\n\n|$)/i);
  if (m) return m[1].trim().slice(0, 1200);
  const sp = s.speak_text.match(/this part of the conversation:\s*([\s\S]+)$/i);
  return sp ? sp[1].trim().slice(0, 1200) : null;
}

type AssistantHL = { start: number; end: number; suggestionId: string; confidence: number };

function mergeAssistantHighlightRanges(raw: AssistantHL[]): AssistantHL[] {
  if (raw.length === 0) return [];
  const sorted = [...raw].filter((r) => r.end > r.start).sort((a, b) => a.start - b.start || b.end - a.end);
  const out: AssistantHL[] = [];
  for (const r of sorted) {
    if (!out.length) {
      out.push({ ...r });
      continue;
    }
    const cur = out[out.length - 1];
    if (r.start >= cur.end) {
      out.push({ ...r });
      continue;
    }
    cur.end = Math.max(cur.end, r.end);
    if (r.confidence > cur.confidence) {
      cur.suggestionId = r.suggestionId;
      cur.confidence = r.confidence;
    }
  }
  return out;
}

function TranscriptWithAssistantHighlights({
  text,
  ranges,
  containerRef,
  onHighlightClick,
}: {
  text: string;
  ranges: AssistantHL[];
  containerRef: React.RefObject<HTMLDivElement | null>;
  onHighlightClick?: (suggestionId: string) => void;
}) {
  if (!text) {
    return (
      <div ref={containerRef} className="text-slate-500 text-sm italic">
        Start listening to capture transcript…
      </div>
    );
  }
  if (ranges.length === 0) {
    return (
      <div ref={containerRef} className="whitespace-pre-wrap break-words text-[15px] text-slate-200 leading-relaxed select-text">
        {text}
      </div>
    );
  }
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;
  const hlClass =
    "border-b border-violet-400/40 decoration-violet-200/40 underline-offset-2 bg-violet-500/[0.08]";
  for (const r of sorted) {
    if (r.start > cursor) {
      parts.push(<span key={`t${key++}`}>{text.slice(cursor, r.start)}</span>);
    }
    const slice = text.slice(r.start, r.end);
    if (onHighlightClick) {
      parts.push(
        <button
          key={`h${key++}`}
          type="button"
          title="Open suggestion details"
          onClick={() => onHighlightClick(r.suggestionId)}
          className={`rounded-sm px-0.5 align-baseline font-inherit text-inherit cursor-pointer ${hlClass}`}
        >
          {slice}
        </button>
      );
    } else {
      parts.push(
        <mark key={`h${key++}`} className={`rounded-sm px-0.5 ${hlClass} text-inherit`}>
          {slice}
        </mark>
      );
    }
    cursor = Math.max(cursor, r.end);
  }
  if (cursor < text.length) {
    parts.push(<span key={`t${key++}`}>{text.slice(cursor)}</span>);
  }
  return (
    <div ref={containerRef} className="whitespace-pre-wrap break-words text-[15px] text-slate-200 leading-relaxed select-text">
      {parts}
    </div>
  );
}

type AssistantUiStatus =
  | "idle"
  | "connecting"
  | "listening"
  | "transcribing"
  | "thinking"
  | "suggestion"
  | "speaking";

function statusLabel(s: AssistantUiStatus): string {
  switch (s) {
    case "connecting":
      return "Connecting…";
    case "listening":
      return "Listening";
    case "transcribing":
      return "Transcribing";
    case "thinking":
      return "Thinking";
    case "suggestion":
      return "Suggestion available";
    case "speaking":
      return "Speaking";
    default:
      return "Stopped";
  }
}

function statusToneFor(s: AssistantUiStatus): ModeStatusTone {
  switch (s) {
    case "connecting":
      return "thinking";
    case "listening":
      return "listening";
    case "transcribing":
      return "transcribing";
    case "thinking":
      return "thinking";
    case "suggestion":
      return "suggestion";
    case "speaking":
      return "speaking";
    default:
      return "neutral";
  }
}

const AssistantMode: React.FC<AssistantModeProps> = ({ liveTranscription, settings }) => {
  const {
    fullTranscript,
    partial,
    listening,
    wsStatus,
    wsError,
    sessionStartedAt,
    sessionName,
    micMuted,
    setMicMuted,
    startSession,
    applyDefault,
    handleStopAndExtractTags,
    transcriptSegments,
  } = liveTranscription;

  const { speakApprovedText, speaking } = useAssistantSpeak();

  const [sessionId] = useState(() => {
    const k = "echomind_assistant_session_id";
    let id = sessionStorage.getItem(k);
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem(k, id);
    }
    return id;
  });

  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [preview, setPreview] = useState<Suggestion | null>(null);
  const [citationModal, setCitationModal] = useState<DocumentChunk[] | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [lastSpoken, setLastSpoken] = useState<Suggestion | null>(null);
  const [thinkingSuggestions, setThinkingSuggestions] = useState(false);

  const assistantMountStartedRef = useRef(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  const refreshSuggestions = useCallback(async () => {
    try {
      setLoadErr(null);
      const rows = await listAssistantSuggestions(sessionId, "pending");
      setSuggestions(rows);
    } catch (e: unknown) {
      setLoadErr((e as Error)?.message ?? "Could not load suggestions");
    }
  }, [sessionId]);

  useEffect(() => {
    void refreshSuggestions();
    const t = window.setInterval(() => void refreshSuggestions(), 10000);
    return () => window.clearInterval(t);
  }, [refreshSuggestions]);

  useEffect(() => {
    if (assistantMountStartedRef.current) return;
    assistantMountStartedRef.current = true;
    if (listening || wsStatus === "connecting" || wsStatus === "loading") return;
    if (wsStatus === "error") return;
    if (wsStatus !== "idle") return;
    applyDefault();
    void startSession(defaultTranscriptName(), "assistant_mode").catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot when Assistant screen mounts
  }, []);

  const transcriptBody = useMemo(() => {
    const parts = [fullTranscript.trim(), partial.trim()].filter(Boolean);
    return parts.join("\n");
  }, [fullTranscript, partial]);

  /** Committed text only — used to align highlights with spoken content (partial shown separately). */
  const displayTranscript = useMemo(() => {
    if (transcriptSegments.length > 0) {
      return transcriptSegments.map((s) => s.text).join("\n\n");
    }
    return fullTranscript.trim();
  }, [transcriptSegments, fullTranscript]);

  const transcriptBodyRef = useRef(transcriptBody);
  transcriptBodyRef.current = transcriptBody;
  const displayTranscriptRef = useRef(displayTranscript);
  displayTranscriptRef.current = displayTranscript;
  const lastAnalyzedOffsetRef = useRef(0);
  const analyzeInFlightRef = useRef(false);

  useEffect(() => {
    displayTranscriptRef.current = displayTranscript;
    const full = displayTranscript.trim();
    if (full.length < lastAnalyzedOffsetRef.current) {
      lastAnalyzedOffsetRef.current = 0;
    }
  }, [displayTranscript]);

  useEffect(() => {
    if (!listening) return;
    const tick = async () => {
      if (analyzeInFlightRef.current) return;
      const full = displayTranscriptRef.current.trim();
      const offset = Math.min(lastAnalyzedOffsetRef.current, full.length);
      const slice = full.slice(offset).trim();
      if (slice.length < ASSISTANT_MIN_CHARS_FOR_ANALYSIS) return;
      analyzeInFlightRef.current = true;
      setThinkingSuggestions(true);
      try {
        await analyzeAssistantTranscript({
          session_id: sessionId,
          mode: "assistant",
          transcript_text: slice,
          full_transcript: full,
          transcript_offset: offset,
          since_last_analysis: true,
          knowledge_base_enabled: true,
          context_window: settings.contextWindow ?? "all",
          persist_results: true,
        });
        lastAnalyzedOffsetRef.current = full.length;
        await refreshSuggestions();
      } catch {
        /* non-fatal */
      } finally {
        analyzeInFlightRef.current = false;
        setThinkingSuggestions(false);
      }
    };
    const id = window.setInterval(() => void tick(), ASSISTANT_KB_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [listening, sessionId, settings.contextWindow, refreshSuggestions]);

  const pending = useMemo(() => suggestions.filter((s) => s.status === "pending"), [suggestions]);
  /** Backend persists ≥70% confidence; client filter matches hand-raise + highlight rules. */
  const pendingProminent = useMemo(
    () => pending.filter((s) => s.confidence >= ASSISTANT_HAND_RAISE_MIN_CONFIDENCE),
    [pending]
  );
  const primaryHandRaise = pendingProminent[0] ?? null;

  const mergedAssistantRanges = useMemo(() => {
    const raw: AssistantHL[] = [];
    for (const s of pendingProminent) {
      const anchor = anchorTextForSuggestion(s);
      if (!anchor) continue;
      const pos = findInTranscript(displayTranscript, anchor);
      if (!pos) continue;
      raw.push({
        start: pos.start,
        end: pos.end,
        suggestionId: s.id,
        confidence: s.confidence,
      });
    }
    return mergeAssistantHighlightRanges(raw);
  }, [pendingProminent, displayTranscript]);

  const onTranscriptHighlightClick = useCallback(
    (suggestionId: string) => {
      const s = suggestions.find((x) => x.id === suggestionId);
      if (s) setPreview(s);
    },
    [suggestions]
  );

  const uiStatus: AssistantUiStatus = useMemo(() => {
    if (speaking) return "speaking";
    if (thinkingSuggestions) return "thinking";
    if (pendingProminent.length) return "suggestion";
    if (listening && wsStatus === "ready" && partial.trim().length > 0) return "transcribing";
    if (listening && (wsStatus === "connecting" || wsStatus === "loading")) return "connecting";
    if (listening && wsStatus === "ready") return "listening";
    return "idle";
  }, [speaking, thinkingSuggestions, pendingProminent.length, listening, wsStatus, partial]);

  const headerExtras = useMemo(() => {
    const out: { label: string; tone?: ModeStatusTone }[] = [];
    if (listening && micMuted) out.push({ label: "Muted", tone: "muted" });
    return out;
  }, [listening, micMuted]);

  const assistantHeaderRight = (
    <>
      {listening ? (
        <button
          type="button"
          onClick={() => void handleStopAndExtractTags()}
          className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15"
        >
          Stop
        </button>
      ) : (
        <button
          type="button"
          onClick={() => {
            applyDefault();
            void startSession(defaultTranscriptName(), "assistant_mode");
          }}
          className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-violet-600/80 text-white hover:bg-violet-600"
        >
          Start listening
        </button>
      )}
      <button
        type="button"
        onClick={() => setMicMuted(!micMuted)}
        disabled={!listening}
        className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 disabled:opacity-40"
      >
        {micMuted ? "Unmute mic" : "Mute mic"}
      </button>
    </>
  );

  const openPreview = useCallback((s: Suggestion) => {
    setPreview(s);
  }, []);

  const onSpeak = useCallback(
    async (s: Suggestion) => {
      if (actionBusy) return;
      setActionBusy(s.id);
      try {
        await approveAssistantSuggestion(s.id);
        await refreshSuggestions();
        setPreview(null);
        await speakApprovedText(s.speak_text || s.short_text, settings);
        try {
          await markAssistantSuggestionSpoken(s.id);
        } catch {
          /* spoken endpoint may 404 if already transitioned */
        }
        setLastSpoken(s);
        await refreshSuggestions();
      } catch (e: unknown) {
        setFlash((e as Error)?.message ?? "Could not speak suggestion");
        window.setTimeout(() => setFlash(null), 4000);
      } finally {
        setActionBusy(null);
      }
    },
    [actionBusy, speakApprovedText, settings, refreshSuggestions]
  );

  const onIgnore = useCallback(
    async (s: Suggestion) => {
      if (actionBusy) return;
      setActionBusy(s.id);
      try {
        await ignoreAssistantSuggestion(s.id);
        await refreshSuggestions();
        if (preview?.id === s.id) setPreview(null);
      } catch {
        setFlash("Ignore failed");
        window.setTimeout(() => setFlash(null), 3000);
      } finally {
        setActionBusy(null);
      }
    },
    [actionBusy, preview, refreshSuggestions]
  );

  const onDismiss = useCallback(
    async (s: Suggestion) => {
      if (actionBusy) return;
      setActionBusy(s.id);
      try {
        await dismissAssistantSuggestion(s.id);
        await refreshSuggestions();
        if (preview?.id === s.id) setPreview(null);
      } catch {
        setFlash("Dismiss failed");
        window.setTimeout(() => setFlash(null), 3000);
      } finally {
        setActionBusy(null);
      }
    },
    [actionBusy, preview, refreshSuggestions]
  );

  const citeChunks = useMemo(() => (preview ? mapCitations(preview.citations as unknown[]) : []), [preview]);

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {citationModal && citationModal.length > 0 ? (
        <ChunkCitationModal citations={citationModal} onClose={() => setCitationModal(null)} />
      ) : null}

      {lastSpoken ? (
        <div className="shrink-0 mx-4 mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-violet-500/25 bg-violet-950/25 px-3 py-2">
          <span className="text-xs text-violet-100/90">Last spoken: {lastSpoken.title}</span>
          <button
            type="button"
            onClick={() => setLastSpoken(null)}
            className="text-[11px] text-slate-500 hover:text-slate-300"
          >
            Dismiss
          </button>
        </div>
      ) : null}

      <ProductModeHeader
        title="Assistant"
        tagline="Listens continuously and checks the knowledge base every 60 seconds."
        status={statusLabel(uiStatus)}
        statusTone={statusToneFor(uiStatus)}
        extraStatuses={headerExtras}
        sessionName={sessionName?.trim() || null}
        showKnowledge
        knowledgeEnabled
        outputHint="When confidence is 70% or higher, you get a hand-raise and transcript highlights — review, then choose Speak now, Ignore, Dismiss, or Speak later. Nothing is spoken until you approve."
        rightSlot={assistantHeaderRight}
      />

      {flash ? (
        <div className="shrink-0 mx-4 mt-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-100">
          {flash}
        </div>
      ) : null}

      {wsError ? (
        <div className="shrink-0 mx-4 mt-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
          {wsError}
        </div>
      ) : null}

      {loadErr ? (
        <div className="shrink-0 mx-4 mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
          {loadErr}
        </div>
      ) : null}

      {/* Hand-raise strip — only when KB-backed suggestion meets confidence threshold */}
      {primaryHandRaise && !preview ? (
        <div className="shrink-0 mx-4 mt-3 rounded-xl border border-violet-500/35 bg-violet-950/40 px-4 py-3 flex flex-wrap items-center gap-3 justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-violet-300 shrink-0" aria-hidden>
              <ICONS.HandRaise className="w-6 h-6" />
            </span>
            <div className="min-w-0">
              <p className="text-sm font-medium text-violet-100">EchoMind has a suggestion</p>
              <p className="text-xs text-violet-200/70 truncate">{primaryHandRaise.title}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => openPreview(primaryHandRaise)}
            className="shrink-0 rounded-lg px-4 py-2 text-sm font-semibold bg-violet-600 text-white hover:bg-violet-500"
          >
            Review
          </button>
        </div>
      ) : null}

      <div className="flex-1 min-h-0 flex flex-col md:flex-row gap-0 md:gap-3 p-3 sm:p-4">
        {/* Transcript */}
        <div className="flex-1 min-h-0 flex flex-col min-w-0 rounded-xl border border-white/10 bg-black/20">
          <div className="shrink-0 px-3 py-2 border-b border-white/10 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Live transcript</span>
            {sessionStartedAt ? (
              <span className="text-[10px] text-slate-500">{sessionStartedAt.toLocaleString()}</span>
            ) : null}
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto px-3 py-3 space-y-2 text-[15px] text-slate-200 leading-relaxed">
            <p className="text-[11px] text-slate-500 mb-2">
              Live speech appears below. Knowledge-base matches at 70% confidence or higher are highlighted; click a highlight to review sources and feedback.
            </p>
            <TranscriptWithAssistantHighlights
              text={displayTranscript}
              ranges={mergedAssistantRanges}
              containerRef={transcriptRef}
              onHighlightClick={onTranscriptHighlightClick}
            />
            {partial ? (
              <p className="whitespace-pre-wrap break-words text-slate-400 border border-dashed border-white/15 rounded-lg px-2 py-1.5">
                {partial}
                <span className="inline-block w-2 h-3.5 ml-1 bg-cyan-400/70 rounded-sm animate-pulse align-middle" aria-hidden />
              </p>
            ) : null}
          </div>
        </div>

        {/* Suggestion queue + detail */}
        <div className="w-full md:w-[340px] shrink-0 flex flex-col gap-2 min-h-0">
          <div className="rounded-xl border border-white/10 bg-black/25 px-3 py-2">
            <div className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">Suggestion queue</div>
            <p className="text-[10px] text-slate-500 mb-1 leading-snug">
              New cards appear after each 60s knowledge-base check when confidence is at least 70%.
            </p>
            {pending.length === 0 ? (
              <p className="text-xs text-slate-500 py-2">No suggestions right now.</p>
            ) : (
              <ul className="max-h-40 overflow-y-auto space-y-1">
                {pending.map((s) => (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => openPreview(s)}
                      className={`w-full text-left rounded-lg px-2 py-1.5 text-xs border transition-colors ${
                        preview?.id === s.id
                          ? "border-violet-500/50 bg-violet-500/15 text-violet-100"
                          : "border-transparent hover:bg-white/5 text-slate-300"
                      }`}
                    >
                      <span className="font-medium text-white/90">{s.title}</span>
                      <span className="block text-slate-500 truncate">{s.category}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {preview ? (
            <div className="flex-1 min-h-0 overflow-y-auto rounded-xl border border-violet-500/25 bg-violet-950/20 p-3 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <h2 className="text-sm font-semibold text-white leading-snug">{preview.title}</h2>
                <button type="button" onClick={() => setPreview(null)} className="text-slate-500 hover:text-white p-1" aria-label="Close preview">
                  <ICONS.Close className="w-4 h-4" />
                </button>
              </div>
              <div className="flex flex-wrap gap-1 text-[10px]">
                <span className="rounded px-1.5 py-0.5 bg-slate-600/40 text-slate-200">{sourceOriginLabel(preview.source_origin)}</span>
                <span className="rounded px-1.5 py-0.5 bg-slate-700/50 text-slate-300">{evidenceLabel(preview.evidence_status)}</span>
                <span className="rounded px-1.5 py-0.5 bg-violet-800/40 text-violet-100">
                  {Math.round((preview.confidence ?? 0) * 100)}% confidence
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{preview.reason || "—"}</p>
              <p className="text-xs text-slate-300 leading-relaxed">{preview.short_text}</p>
              {citeChunks.length > 0 ? (
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Sources</p>
                  <div className="flex flex-wrap gap-1">
                    {citeChunks.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => setCitationModal(citeChunks)}
                        className={CITATION_CHIP_CLASS}
                        title={c.docName}
                      >
                        {(c.docName.length > 22 ? `${c.docName.slice(0, 20)}…` : c.docName) +
                          (c.metadata.pageNumber != null ? ` · p.${c.metadata.pageNumber}` : "")}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="button"
                  disabled={!!actionBusy}
                  onClick={() => void onSpeak(preview)}
                  className="rounded-lg px-3 py-2 text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40"
                >
                  Speak now
                </button>
                <button
                  type="button"
                  disabled={!!actionBusy}
                  onClick={() => void onIgnore(preview)}
                  className="rounded-lg px-3 py-2 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 disabled:opacity-40"
                >
                  Ignore
                </button>
                <button
                  type="button"
                  disabled={!!actionBusy}
                  onClick={() => void onDismiss(preview)}
                  className="rounded-lg px-3 py-2 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 disabled:opacity-40"
                >
                  Dismiss
                </button>
                <button
                  type="button"
                  disabled={!!actionBusy}
                  onClick={() => setPreview(null)}
                  className="rounded-lg px-3 py-2 text-xs font-semibold bg-white/5 text-slate-400 hover:bg-white/10 disabled:opacity-40"
                >
                  Speak later
                </button>
              </div>
            </div>
          ) : (
            <div className="flex-1 rounded-xl border border-dashed border-white/10 bg-white/[0.02] flex items-center justify-center p-4">
              <p className="text-xs text-slate-500 text-center">Open a suggestion from the queue or the hand-raise banner.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AssistantMode;
