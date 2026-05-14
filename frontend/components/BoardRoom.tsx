import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { BoardRoomReport, BoardRoomSttStatus } from "../types";
import {
  boardRoomReportExportUrl,
  defaultTranscriptName,
  generateBoardRoomReport,
  getBoardRoomSttStatus,
} from "../services/backend";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import AssistantModeControls from "./assistant/AssistantModeControls";
import { newSessionId } from "./assistant/assistantUtils";

interface BoardRoomProps {
  liveTranscription: UseLiveTranscriptionReturn;
}

type Phase = "listen" | "configure" | "generating" | "ready";
type PreviewTab = "summary" | "minutes" | "validation" | "markdown";

const BoardRoom: React.FC<BoardRoomProps> = ({ liveTranscription }) => {
  const {
    fullTranscript,
    listening,
    wsStatus,
    wsError,
    sessionName,
    sessionLocation,
    openStartModal,
    startSession,
    handleStopAndExtractTags,
    clearAndReset,
    micMuted,
    setMicMuted,
    showStartModal,
    modalName,
    modalLocation,
    setModalName,
    setModalLocation,
    setShowStartModal,
    applyDefault,
  } = liveTranscription;

  const [phase, setPhase] = useState<Phase>("listen");
  const [sessionId, setSessionId] = useState(() => newSessionId());
  const [reportTitle, setReportTitle] = useState("");
  const [includeRag, setIncludeRag] = useState(true);
  const [wantPdf, setWantPdf] = useState(true);
  const [wantPptx, setWantPptx] = useState(false);
  const [report, setReport] = useState<BoardRoomReport | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const [previewTab, setPreviewTab] = useState<PreviewTab>("summary");
  const [sttStatus, setSttStatus] = useState<BoardRoomSttStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getBoardRoomSttStatus()
      .then((s) => {
        if (!cancelled) setSttStatus(s);
      })
      .catch(() => {
        if (!cancelled) setSttStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (wsError) return wsError;
    if (phase === "generating") return "Polishing transcript and validating against your knowledge base…";
    if (phase === "ready") return "Report ready — preview below or download exports.";
    if (phase === "configure") return "Session ended. Configure the report, then generate.";
    if (listening) {
      if (wsStatus === "loading") return "Loading multitalker ASR model…";
      if (wsStatus === "ready") return "Listening only — analysis starts after you stop the session.";
      return "Connecting to Board Room transcription…";
    }
    return "Idle — start a session to capture the meeting transcript.";
  }, [listening, phase, wsError, wsStatus]);

  const onStartFromModal = () => {
    const name = (modalName || "").trim() || defaultTranscriptName();
    const location = (modalLocation || "").trim() || "default";
    setSessionId(newSessionId());
    setReport(null);
    setGenError(null);
    setPhase("listen");
    setReportTitle(name);
    void startSession(name, location, { autoStore: true });
  };

  const onStop = () => {
    void handleStopAndExtractTags().then(() => {
      setPhase("configure");
    });
  };

  const onClear = () => {
    clearAndReset();
    setSessionId(newSessionId());
    setReport(null);
    setGenError(null);
    setPhase("listen");
    setReportTitle("");
  };

  const onGenerate = useCallback(async () => {
    const text = (fullTranscript || "").trim();
    if (!text) {
      setGenError("Transcript is empty. Record a session before generating a report.");
      return;
    }
    if (!wantPdf && !wantPptx) {
      setGenError("Select at least one export format (PDF or PPTX).");
      return;
    }
    setGenError(null);
    setPhase("generating");
    try {
      const res = await generateBoardRoomReport({
        session_id: sessionId,
        title: (reportTitle || sessionName || "Board Room Report").trim(),
        transcript: text,
        session_name: sessionName,
        session_location: sessionLocation,
        include_rag_validation: includeRag,
        analysis_scope: { documents: true, transcripts: false, books: true, faqs: true },
      });
      setReport(res);
      setPhase("ready");
      setPreviewTab("summary");
    } catch (e) {
      setGenError(e instanceof Error ? e.message : "Report generation failed.");
      setPhase("configure");
    }
  }, [fullTranscript, includeRag, reportTitle, sessionId, sessionLocation, sessionName, wantPdf, wantPptx]);

  const stepIndex = phase === "listen" ? 0 : phase === "configure" || phase === "generating" ? 1 : 2;

  return (
    <div className="h-full min-h-0 flex flex-col lg:flex-row gap-3 rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <div className="flex-1 min-h-0 flex flex-col min-w-0 p-4 sm:p-5">
        {showStartModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setShowStartModal(false)}>
            <div className="w-full max-w-md rounded-2xl border border-white/15 bg-[#0c1220] p-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <h2 className="text-lg font-semibold text-white mb-1">Start Board Room session</h2>
              <p className="text-xs text-slate-400 mb-4">
                EchoMind listens with multitalker Parakeet streaming ASR. Analysis and report export run after you stop.
              </p>
              <label className="block text-xs text-slate-400 mb-1">Session name</label>
              <input
                className="w-full mb-3 rounded-lg bg-black/30 border border-white/10 px-3 py-2 text-sm text-white"
                value={modalName}
                onChange={(e) => setModalName(e.target.value)}
                placeholder={defaultTranscriptName()}
              />
              <label className="block text-xs text-slate-400 mb-1">Location</label>
              <input
                className="w-full mb-4 rounded-lg bg-black/30 border border-white/10 px-3 py-2 text-sm text-white"
                value={modalLocation}
                onChange={(e) => setModalLocation(e.target.value)}
                placeholder="default"
              />
              <div className="flex flex-wrap gap-2 justify-end">
                <button type="button" className="text-xs text-slate-400 underline" onClick={applyDefault}>
                  Use default name
                </button>
                <button type="button" className="rounded-lg px-3 py-2 text-sm bg-white/10 text-white" onClick={() => setShowStartModal(false)}>
                  Cancel
                </button>
                <button type="button" className="rounded-lg px-3 py-2 text-sm bg-cyan-500/25 text-cyan-200 border border-cyan-500/35" onClick={onStartFromModal}>
                  Start listening
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px]">
          {["Listen", "Configure report", "Preview & export"].map((label, i) => (
            <span
              key={label}
              className={`rounded-full px-2.5 py-1 border ${
                i === stepIndex
                  ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
                  : i < stepIndex
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200/90"
                    : "border-white/10 text-slate-500"
              }`}
            >
              {i + 1}. {label}
            </span>
          ))}
          {sttStatus && (
            <span className="ml-auto text-slate-500">
              ASR: {sttStatus.model_name.split("/").pop()}
              {sttStatus.using_fallback && sttStatus.fallback_model_name
                ? ` · fallback ${sttStatus.fallback_model_name.split("/").pop()}`
                : sttStatus.loaded
                  ? " · warmed"
                  : sttStatus.available
                    ? " · loads on connect"
                    : " · unavailable"}
            </span>
          )}
        </div>

        <AssistantModeControls
          listening={listening}
          micMuted={micMuted}
          onMicMutedChange={setMicMuted}
          onOpenStart={openStartModal}
          onStop={onStop}
          onClear={onClear}
          showSaveToggle={false}
        />

        <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-slate-400 mb-3">{statusLabel}</div>

        <div className="flex-1 min-h-0 rounded-xl border border-white/10 bg-black/25 p-3 overflow-y-auto">
          <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-2">Live transcript</div>
          <p className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
            {fullTranscript || <span className="text-slate-500">Transcript will appear here while you listen.</span>}
          </p>
        </div>
      </div>

      <aside className="w-full lg:w-[min(420px,100%)] shrink-0 border-t lg:border-t-0 lg:border-l border-white/10 bg-[#080b12] flex flex-col min-h-[240px] lg:min-h-0 max-h-[55vh] lg:max-h-none p-4">
        {phase === "configure" && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-white">Report options</h3>
            <label className="block text-xs text-slate-400">Report title</label>
            <input
              className="w-full rounded-lg bg-black/30 border border-white/10 px-3 py-2 text-sm text-white"
              value={reportTitle}
              onChange={(e) => setReportTitle(e.target.value)}
            />
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input type="checkbox" checked={includeRag} onChange={(e) => setIncludeRag(e.target.checked)} />
              Cross-check claims with uploaded knowledge (RAG)
            </label>
            <div className="text-xs text-slate-400">Export formats</div>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input type="checkbox" checked={wantPdf} onChange={(e) => setWantPdf(e.target.checked)} />
              PDF
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input type="checkbox" checked={wantPptx} onChange={(e) => setWantPptx(e.target.checked)} />
              PowerPoint (PPTX)
            </label>
            {genError && <p className="text-xs text-red-400/90">{genError}</p>}
            <button
              type="button"
              onClick={() => void onGenerate()}
              className="w-full rounded-xl px-4 py-2.5 text-sm font-semibold bg-violet-500/20 text-violet-100 border border-violet-500/35"
            >
              Generate report
            </button>
          </div>
        )}

        {phase === "generating" && (
          <div className="flex flex-col items-center justify-center flex-1 text-center gap-3">
            <div className="w-10 h-10 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin" aria-hidden />
            <p className="text-sm text-slate-300">Building your Board Room report…</p>
            <p className="text-xs text-slate-500 max-w-xs">Polishing minutes, executive summary, and optional knowledge validation via TensorRT LLM.</p>
          </div>
        )}

        {phase === "ready" && report && (
          <div className="flex flex-col min-h-0 flex-1 gap-3">
            <div className="flex flex-wrap gap-1">
              {(
                [
                  ["summary", "Summary"],
                  ["minutes", "Minutes"],
                  ["validation", "Validation"],
                  ["markdown", "Markdown"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setPreviewTab(id)}
                  className={`rounded-lg px-2.5 py-1 text-[11px] border ${
                    previewTab === id ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200" : "border-white/10 text-slate-400"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto text-sm text-slate-200 space-y-2">
              {previewTab === "summary" && <p className="whitespace-pre-wrap">{report.executive_summary || "—"}</p>}
              {previewTab === "minutes" && <p className="whitespace-pre-wrap">{report.polished_transcript}</p>}
              {previewTab === "validation" && (
                <ul className="space-y-2 text-xs">
                  {(report.knowledge_checks || []).map((c, i) => (
                    <li key={i} className="rounded-lg border border-white/10 bg-white/5 p-2">
                      <div className="text-slate-400 uppercase text-[10px]">{c.classification.replace("_", " ")}</div>
                      <p className="text-slate-200 mt-1">{c.claim}</p>
                      {c.interpretation && <p className="text-slate-400 mt-1">{c.interpretation}</p>}
                    </li>
                  ))}
                  {!report.knowledge_checks?.length && <li className="text-slate-500">No knowledge checks returned.</li>}
                </ul>
              )}
              {previewTab === "markdown" && <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">{report.markdown}</pre>}
            </div>
            <div className="flex flex-wrap gap-2 pt-2 border-t border-white/10">
              {wantPdf && (
                <a
                  href={boardRoomReportExportUrl(report.report_id, "pdf")}
                  className="rounded-lg px-3 py-2 text-xs font-medium bg-cyan-500/20 text-cyan-200 border border-cyan-500/30"
                  download
                >
                  Download PDF
                </a>
              )}
              {wantPptx && (
                <a
                  href={boardRoomReportExportUrl(report.report_id, "pptx")}
                  className="rounded-lg px-3 py-2 text-xs font-medium bg-violet-500/20 text-violet-200 border border-violet-500/35"
                  download
                >
                  Download PPTX
                </a>
              )}
              <button type="button" className="text-xs text-slate-500 underline" onClick={() => setPhase("configure")}>
                Regenerate
              </button>
            </div>
          </div>
        )}

        {phase === "listen" && !listening && (
          <p className="text-xs text-slate-500">
            After you stop the session, choose PDF or PPTX and whether to validate discussion points against your knowledge base.
          </p>
        )}
      </aside>
    </div>
  );
};

export default BoardRoom;
