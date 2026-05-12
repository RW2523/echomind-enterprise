import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import {
  acceptSilentFinding,
  analyzeAssistantTranscript,
  defaultTranscriptName,
  dismissSilentFinding,
  listSilentFindings,
  markSilentFindingUnhelpful,
} from "../services/backend";
import type { AppSettings, CorrectionFinding, DocumentChunk, SilentFindingStatusLabel } from "../types";
import { mapCitations } from "../utils/mapCitations";
import {
  CITATION_CHIP_CLASS,
  evidenceLabel,
  silentAssistantDisplaySeverity,
  silentAssistantDisplayStatus,
  silentFindingSourceChip,
} from "../utils/modeChrome";
import { ChunkCitationModal } from "./KnowledgeChat";
import ProductModeHeader, { type ModeStatusTone } from "./ProductModeHeader";

interface SilentAssistantModeProps {
  liveTranscription: UseLiveTranscriptionReturn;
  settings: AppSettings;
}

/** Send recent transcript to the backend on this cadence while listening (KB check). */
const SILENT_KB_CHECK_INTERVAL_MS = 60_000;
/** Highlights when confidence is strictly above 70% (product spec for Silent Assistant). */
const SILENT_HIGHLIGHT_MIN_CONFIDENCE = 0.7;
const SILENT_MIN_CHARS_FOR_KB_CHECK = 90;

function categoryLabel(cat: string): string {
  return cat.replace(/_/g, " ");
}

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

type HighlightRange = { start: number; end: number; status_label: SilentFindingStatusLabel; findingId: string };

type MergedRange = { start: number; end: number; status_label: SilentFindingStatusLabel; findingId: string };

function mergeHighlightRanges(ranges: HighlightRange[]): MergedRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].filter((r) => r.end > r.start).sort((a, b) => a.start - b.start || b.end - a.end);
  const out: MergedRange[] = [];
  for (const r of sorted) {
    if (!out.length) {
      out.push({ start: r.start, end: r.end, status_label: r.status_label, findingId: r.findingId });
      continue;
    }
    const cur = out[out.length - 1];
    if (r.start >= cur.end) {
      out.push({ start: r.start, end: r.end, status_label: r.status_label, findingId: r.findingId });
      continue;
    }
    cur.end = Math.max(cur.end, r.end);
    const rs = silentAssistantDisplaySeverity(r.status_label);
    const cs = silentAssistantDisplaySeverity(cur.status_label);
    if (rs > cs) {
      cur.status_label = r.status_label;
      cur.findingId = r.findingId;
    }
  }
  return out;
}

function highlightClass(sl: SilentFindingStatusLabel): string {
  switch (silentAssistantDisplayStatus(sl)) {
    case "Contradicted":
      return "border-b border-rose-400/40 decoration-rose-300/50 underline-offset-2 bg-rose-500/[0.06]";
    case "Unverified":
      return "border-b border-slate-400/35 decoration-slate-300/35 underline-offset-2 bg-slate-500/[0.06]";
    case "Needs Review":
      return "border-b border-cyan-400/30 decoration-cyan-200/35 underline-offset-2 bg-cyan-500/[0.05]";
    case "Supported":
      return "border-b border-emerald-400/25 decoration-emerald-200/30 underline-offset-2 bg-emerald-500/[0.04]";
    default:
      return "border-b border-white/10";
  }
}

function TranscriptWithHighlights({
  text,
  ranges,
  containerRef,
  onHighlightClick,
}: {
  text: string;
  ranges: MergedRange[];
  containerRef: React.RefObject<HTMLDivElement | null>;
  onHighlightClick?: (findingId: string) => void;
}) {
  if (!text) {
    return (
      <div ref={containerRef} className="text-slate-500 text-sm italic">
        Listening and checking quietly...
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
  for (const r of sorted) {
    if (r.start > cursor) {
      parts.push(<span key={`t${key++}`}>{text.slice(cursor, r.start)}</span>);
    }
    const slice = text.slice(r.start, r.end);
    const label = silentAssistantDisplayStatus(r.status_label);
    const title = `${label} — open details`;
    if (onHighlightClick) {
      parts.push(
        <button
          key={`h${key++}`}
          type="button"
          title={title}
          onClick={() => onHighlightClick(r.findingId)}
          className={`rounded-sm px-0.5 align-baseline font-inherit text-inherit cursor-pointer ${highlightClass(r.status_label)}`}
        >
          {slice}
        </button>
      );
    } else {
      parts.push(
        <mark key={`h${key++}`} className={`rounded-sm px-0.5 ${highlightClass(r.status_label)} text-inherit`} title={title}>
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

const SilentAssistantMode: React.FC<SilentAssistantModeProps> = ({ liveTranscription, settings }) => {
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

  const [sessionId] = useState(() => {
    const k = "echomind_silent_session_id";
    let id = sessionStorage.getItem(k);
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem(k, id);
    }
    return id;
  });

  const [allFindings, setAllFindings] = useState<CorrectionFinding[]>([]);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [selectedFinding, setSelectedFinding] = useState<CorrectionFinding | null>(null);
  const [citationModal, setCitationModal] = useState<DocumentChunk[] | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [manualBusy, setManualBusy] = useState(false);
  const [silentAnalyzeBusy, setSilentAnalyzeBusy] = useState(false);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const silentMountStartedRef = useRef(false);

  const combinedTranscript = useMemo(() => {
    if (transcriptSegments.length > 0) {
      return transcriptSegments.map((s) => s.text).join("\n\n");
    }
    return fullTranscript.trim();
  }, [transcriptSegments, fullTranscript]);

  /** Committed text for highlights (partial shown separately). */
  const displayTranscript = combinedTranscript;

  const refreshFindings = useCallback(async () => {
    try {
      setLoadErr(null);
      const rows = await listSilentFindings(sessionId, "all");
      setAllFindings(rows);
    } catch (e: unknown) {
      setLoadErr((e as Error)?.message ?? "Could not load findings");
    }
  }, [sessionId]);

  useEffect(() => {
    void refreshFindings();
    const t = window.setInterval(() => void refreshFindings(), 8000);
    return () => clearInterval(t);
  }, [refreshFindings]);

  useEffect(() => {
    if (silentMountStartedRef.current) return;
    silentMountStartedRef.current = true;
    if (listening || wsStatus === "connecting" || wsStatus === "loading") return;
    if (wsStatus === "error") return;
    if (wsStatus !== "idle") return;
    applyDefault();
    void startSession(defaultTranscriptName(), "silent_assistant_mode").catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot on Silent Assistant mount
  }, []);

  /** In-transcript highlights: active findings with confidence strictly above 70%. */
  const transcriptHighlightFindings = useMemo(
    () =>
      allFindings.filter(
        (f) =>
          f.user_action !== "dismissed" &&
          f.user_action !== "marked_unhelpful" &&
          (f.confidence ?? 0) > SILENT_HIGHLIGHT_MIN_CONFIDENCE
      ),
    [allFindings]
  );

  const pendingCards = useMemo(() => allFindings.filter((f) => f.user_action === "pending"), [allFindings]);

  const displayTranscriptRef = useRef(displayTranscript);
  displayTranscriptRef.current = displayTranscript;
  const lastSilentAnalyzedOffsetRef = useRef(0);
  const analyzeInFlightRef = useRef(false);

  useEffect(() => {
    const full = displayTranscript.trim();
    if (full.length < lastSilentAnalyzedOffsetRef.current) {
      lastSilentAnalyzedOffsetRef.current = 0;
    }
  }, [displayTranscript]);

  const mergedRanges = useMemo(() => {
    const raw: HighlightRange[] = [];
    for (const f of transcriptHighlightFindings) {
      const pos = findInTranscript(displayTranscript, f.original_text);
      if (!pos) continue;
      raw.push({
        start: pos.start,
        end: pos.end,
        status_label: f.status_label as SilentFindingStatusLabel,
        findingId: f.id,
      });
    }
    return mergeHighlightRanges(raw);
  }, [transcriptHighlightFindings, displayTranscript]);

  useEffect(() => {
    if (!listening) return;
    const tick = async () => {
      if (analyzeInFlightRef.current) return;
      const full = displayTranscriptRef.current.trim();
      const offset = Math.min(lastSilentAnalyzedOffsetRef.current, full.length);
      const slice = full.slice(offset).trim();
      if (slice.length < SILENT_MIN_CHARS_FOR_KB_CHECK) return;
      analyzeInFlightRef.current = true;
      setSilentAnalyzeBusy(true);
      try {
        await analyzeAssistantTranscript({
          session_id: sessionId,
          mode: "silent_assistant",
          transcript_text: slice,
          full_transcript: full,
          transcript_offset: offset,
          since_last_analysis: true,
          knowledge_base_enabled: true,
          context_window: settings.contextWindow ?? "all",
          persist_results: true,
        });
        lastSilentAnalyzedOffsetRef.current = full.length;
        await refreshFindings();
      } catch {
        /* non-fatal */
      } finally {
        analyzeInFlightRef.current = false;
        setSilentAnalyzeBusy(false);
      }
    };
    const id = window.setInterval(() => void tick(), SILENT_KB_CHECK_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [listening, sessionId, settings.contextWindow, refreshFindings]);

  const onTranscriptHighlightClick = useCallback(
    (findingId: string) => {
      const f = allFindings.find((x) => x.id === findingId);
      if (f) setSelectedFinding(f);
    },
    [allFindings]
  );

  const onManualVerify = useCallback(async () => {
    const sel = window.getSelection();
    const text = sel?.toString().trim() ?? "";
    if (text.length < 12) {
      setFlash("Select a bit more text to verify (at least 12 characters).");
      window.setTimeout(() => setFlash(null), 3000);
      return;
    }
    setManualBusy(true);
    setSilentAnalyzeBusy(true);
    try {
      await analyzeAssistantTranscript({
        session_id: sessionId,
        mode: "silent_assistant",
        transcript_text: text.slice(0, 8000),
        full_transcript: displayTranscriptRef.current.trim(),
        transcript_offset: 0,
        since_last_analysis: true,
        knowledge_base_enabled: true,
        context_window: settings.contextWindow ?? "all",
        persist_results: true,
      });
      await refreshFindings();
      setFlash("Verification complete (display-only).");
      window.setTimeout(() => setFlash(null), 2500);
    } catch (e: unknown) {
      setFlash((e as Error)?.message ?? "Verify failed");
      window.setTimeout(() => setFlash(null), 4000);
    } finally {
      setSilentAnalyzeBusy(false);
      setManualBusy(false);
    }
  }, [sessionId, settings.contextWindow, refreshFindings]);

  const runAction = useCallback(
    async (id: string, fn: (x: string) => Promise<CorrectionFinding>) => {
      setActionBusy(id);
      try {
        await fn(id);
        await refreshFindings();
        setSelectedFinding((prev) => (prev?.id === id ? null : prev));
      } catch (e: unknown) {
        setFlash((e as Error)?.message ?? "Action failed");
        window.setTimeout(() => setFlash(null), 3500);
      } finally {
        setActionBusy(null);
      }
    },
    [refreshFindings]
  );

  const detailCitations = useMemo(
    () => (selectedFinding ? mapCitations(selectedFinding.citations as unknown[]) : []),
    [selectedFinding]
  );

  const silentUiStatus = useMemo(() => {
    if (!listening) return { label: "Stopped", tone: "neutral" as ModeStatusTone };
    if (wsStatus === "connecting" || wsStatus === "loading") return { label: "Connecting…", tone: "thinking" as ModeStatusTone };
    if (silentAnalyzeBusy || manualBusy) return { label: "Checking silently", tone: "silent" as ModeStatusTone };
    if (wsStatus === "ready" && partial.trim().length > 0) return { label: "Transcribing", tone: "transcribing" as ModeStatusTone };
    if (listening && wsStatus === "ready") return { label: "Listening", tone: "listening" as ModeStatusTone };
    return { label: "Idle", tone: "neutral" as ModeStatusTone };
  }, [listening, wsStatus, silentAnalyzeBusy, manualBusy, partial]);

  const silentHeaderExtras = useMemo(() => {
    const out: { label: string; tone?: ModeStatusTone }[] = [];
    if (listening && micMuted) out.push({ label: "Muted", tone: "muted" });
    return out;
  }, [listening, micMuted]);

  const silentHeaderRight = (
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
            void startSession(defaultTranscriptName(), "silent_assistant_mode");
          }}
          className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-cyan-700/80 text-white hover:bg-cyan-600"
        >
          Start listening
        </button>
      )}
      <button
        type="button"
        disabled={!listening}
        onClick={() => setMicMuted(!micMuted)}
        className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 disabled:opacity-40"
      >
        {micMuted ? "Unmute mic" : "Mute mic"}
      </button>
      <button
        type="button"
        disabled={manualBusy}
        onClick={() => void onManualVerify()}
        className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-cyan-200 hover:bg-white/15 disabled:opacity-40"
      >
        {manualBusy ? "Verifying…" : "Verify selection"}
      </button>
    </>
  );

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {citationModal && citationModal.length > 0 ? (
        <ChunkCitationModal citations={citationModal} onClose={() => setCitationModal(null)} />
      ) : null}

      <ProductModeHeader
        title="Silent Assistant"
        tagline="Checks and highlights quietly without speaking."
        status={silentUiStatus.label}
        statusTone={silentUiStatus.tone}
        extraStatuses={silentHeaderExtras}
        sessionName={sessionName?.trim() || null}
        showKnowledge
        knowledgeEnabled
        outputHint="Never speaks. While you talk, new committed transcript is checked against the knowledge base every 60s. Highlights appear when confidence is above 70%; click a highlight for details."
        rightSlot={silentHeaderRight}
      />

      {flash ? <div className="shrink-0 mx-4 mt-2 text-xs text-amber-100/90 bg-amber-500/10 border border-amber-500/25 rounded-lg px-3 py-2">{flash}</div> : null}
      {wsError ? <div className="shrink-0 mx-4 mt-2 text-xs text-red-200 bg-red-500/10 border border-red-500/25 rounded-lg px-3 py-2">{wsError}</div> : null}
      {loadErr ? <div className="shrink-0 mx-4 mt-2 text-xs text-amber-100 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">{loadErr}</div> : null}

      <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-3 p-3 sm:p-4">
        <div className="flex-1 min-h-0 min-w-0 flex flex-col rounded-xl border border-white/10 bg-black/20">
          <div className="shrink-0 px-3 py-2 border-b border-white/10 flex justify-between items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">Transcript</span>
            {sessionStartedAt ? <span className="text-[10px] text-slate-500">{sessionStartedAt.toLocaleString()}</span> : null}
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto px-3 py-3">
            <p className="text-[11px] text-slate-500 mb-2">
              Spoken text appears here. Knowledge-base matches above 70% confidence are highlighted — click a highlight for evidence. Use &quot;Verify selection&quot; to check a chosen phrase anytime.
            </p>
            <TranscriptWithHighlights
              text={displayTranscript}
              ranges={mergedRanges}
              containerRef={transcriptRef}
              onHighlightClick={onTranscriptHighlightClick}
            />
            {partial ? (
              <p className="mt-2 whitespace-pre-wrap break-words text-slate-400 text-sm border border-dashed border-white/10 rounded-lg px-2 py-1.5">
                {partial}
                <span className="inline-block w-2 h-3.5 ml-1 bg-cyan-400/60 rounded-sm animate-pulse align-middle" aria-hidden />
              </p>
            ) : null}
          </div>
        </div>

        <div className="w-full lg:w-[360px] shrink-0 flex flex-col gap-2 min-h-0">
          <div className="rounded-xl border border-white/10 bg-black/25 min-h-0 flex flex-col flex-1">
            <div className="shrink-0 px-3 py-2 border-b border-white/10 space-y-0.5">
              <div className="text-xs font-medium uppercase tracking-wider text-slate-500">Findings</div>
              <p className="text-[10px] text-slate-500 leading-snug">Transcript highlights when confidence is above 70%.</p>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2">
              {pendingCards.length === 0 ? (
                <p className="text-xs text-slate-500 px-1 py-4 text-center">No issues detected yet.</p>
              ) : (
                pendingCards.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setSelectedFinding(f)}
                    className={`w-full text-left rounded-lg border px-3 py-2.5 transition-colors ${
                      selectedFinding?.id === f.id
                        ? "border-cyan-500/40 bg-cyan-950/30"
                        : "border-white/10 bg-white/[0.03] hover:bg-white/[0.06]"
                    }`}
                  >
                    <div className="flex flex-wrap gap-1.5 items-center mb-1">
                      <span className="text-[10px] font-semibold rounded px-1.5 py-0.5 bg-slate-700/60 text-slate-100">
                        {silentAssistantDisplayStatus(f.status_label)}
                      </span>
                      <span className="text-[10px] uppercase tracking-wide text-slate-500">{categoryLabel(f.category)}</span>
                      <span className="text-[10px] rounded px-1.5 py-0.5 bg-slate-600/40 text-slate-200">{silentFindingSourceChip(f.source_origin)}</span>
                      <span className="text-[10px] rounded px-1.5 py-0.5 bg-slate-700/50 text-slate-300">{evidenceLabel(f.evidence_status)}</span>
                    </div>
                    <p className="text-xs text-slate-200 line-clamp-3">{f.original_text}</p>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Evidence / detail strip */}
      <div className="shrink-0 border-t border-white/10 bg-black/30 px-4 py-3 max-h-[38vh] overflow-y-auto">
        {!selectedFinding ? (
          <p className="text-xs text-slate-500 text-center py-2">
            Click a highlighted phrase in the transcript, or a card in Findings, for evidence and actions. You can also select text and use Verify.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2 items-start justify-between">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-500">Flagged statement</p>
                <p className="text-sm text-slate-100 mt-0.5 whitespace-pre-wrap">{selectedFinding.original_text}</p>
              </div>
              <div className="flex flex-wrap gap-1 text-[10px]">
                <span className="rounded px-2 py-0.5 bg-slate-700/70 text-slate-100 font-medium">
                  {silentAssistantDisplayStatus(selectedFinding.status_label)}
                </span>
                <span className="rounded px-2 py-0.5 bg-slate-600/40 text-slate-200">{silentFindingSourceChip(selectedFinding.source_origin)}</span>
                <span className="rounded px-2 py-0.5 bg-slate-700/50 text-slate-300">{evidenceLabel(selectedFinding.evidence_status)}</span>
                <span className="rounded px-2 py-0.5 bg-slate-700/60 text-slate-300">
                  {Math.round((selectedFinding.confidence ?? 0) * 100)}% confidence
                </span>
              </div>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Explanation</p>
              <p className="text-xs text-slate-300 mt-0.5 leading-relaxed">{selectedFinding.reason}</p>
            </div>
            {selectedFinding.suggested_correction ? (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-500">Assistant interpretation</p>
                <p className="text-xs text-cyan-100/90 mt-0.5">{selectedFinding.suggested_correction}</p>
              </div>
            ) : null}
            {detailCitations.length > 0 ? (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Sources</p>
                <div className="flex flex-wrap gap-1">
                  {detailCitations.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => setCitationModal(detailCitations)}
                      className={`${CITATION_CHIP_CLASS} max-w-[200px]`}
                    >
                      {(c.docName.length > 20 ? `${c.docName.slice(0, 18)}…` : c.docName) +
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
                onClick={() => void runAction(selectedFinding.id, acceptSilentFinding)}
                className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-emerald-700/70 text-white hover:bg-emerald-600/90 disabled:opacity-40"
              >
                Accept
              </button>
              <button
                type="button"
                disabled={!!actionBusy}
                onClick={() => void runAction(selectedFinding.id, dismissSilentFinding)}
                className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-200 hover:bg-white/15 disabled:opacity-40"
              >
                Dismiss
              </button>
              <button
                type="button"
                disabled={!!actionBusy}
                onClick={() => void runAction(selectedFinding.id, markSilentFindingUnhelpful)}
                className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 hover:bg-white/15 disabled:opacity-40"
              >
                Mark unhelpful
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SilentAssistantMode;
