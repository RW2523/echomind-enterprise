import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { AppSettings, AssistantEvidence, AssistantInsight, AssistantInsightActionStatus, HandRaiseAction } from "../types";
import { defaultTranscriptName, listAssistantSessionInsights, patchAssistantInsightAction } from "../services/backend";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import { useAssistantAnalysis } from "../hooks/useAssistantAnalysis";
import { useSpeakAssistantTts } from "../hooks/useSpeakAssistantTts";
import AssistantModeControls from "./assistant/AssistantModeControls";
import AssistantTranscriptPanel from "./assistant/AssistantTranscriptPanel";
import InsightPanel from "./assistant/InsightPanel";
import HandRaiseCard from "./assistant/HandRaiseCard";
import SessionInsightsReview from "./assistant/SessionInsightsReview";
import EvidenceSourcePreviewModal from "./assistant/EvidenceSourcePreviewModal";
import { assistantInsightsMetaStorageKey, handRaiseTier } from "./assistant/assistantUtils";

function buildFollowUpDraft(ins: AssistantInsight): string {
  const ev = ins.evidence?.[0];
  const evSnip = (ev?.matched_text || "").slice(0, 500);
  const lines = [
    "Follow-up from Personal Assistant:",
    "",
    `Transcript: "${ins.transcript_text}"`,
    "",
    `Interpretation: ${ins.assistant_interpretation || "—"}`,
    "",
    evSnip
      ? `Evidence (${ev.source_name || "source"}):\n${evSnip}${(ev?.matched_text?.length || 0) > 500 ? "…" : ""}`
      : "",
  ];
  return lines.filter((x) => x !== "").join("\n");
}

function handRaiseActionToStatus(action: HandRaiseAction): AssistantInsightActionStatus | null {
  switch (action) {
    case "view_details":
      return "viewed";
    case "ignore":
      return "ignored";
    case "save_for_later":
      return "saved_for_later";
    case "ask_follow_up":
      return "asked_follow_up";
    case "speak_now":
      return null;
    default:
      return null;
  }
}

interface PersonalAssistantProps {
  settings: AppSettings;
  liveTranscription: UseLiveTranscriptionReturn;
  onOpenKnowledgeChatWithDraft?: (draft: string) => void;
}

const PersonalAssistant: React.FC<PersonalAssistantProps> = ({
  settings,
  liveTranscription,
  onOpenKnowledgeChatWithDraft,
}) => {
  const {
    fullTranscript,
    listening,
    wsStatus,
    wsError,
    transcriptId,
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

  const [saveTranscript, setSaveTranscript] = useState(true);
  const [panelInsight, setPanelInsight] = useState<AssistantInsight | null>(null);
  const [previewEvidence, setPreviewEvidence] = useState<AssistantEvidence[] | null>(null);
  const [previewInitialIndex, setPreviewInitialIndex] = useState(0);
  const [dismissedIds, setDismissedIds] = useState<string[]>([]);
  const [savedForLater, setSavedForLater] = useState<AssistantInsight[]>([]);

  const dismissedSet = useMemo(() => new Set(dismissedIds), [dismissedIds]);

  const analysis = useAssistantAnalysis("personal_assistant", fullTranscript, listening, wsStatus, {
    persistInsights: saveTranscript,
    transcriptId,
  });
  const {
    sessionId,
    insights,
    setInsights,
    analysisStatus,
    analysisError,
    runAnalysis,
    stopTimers,
    resetSession,
    prepareNewListeningSession,
    hydrateFromServer,
  } = analysis;

  useEffect(() => {
    let cancelled = false;
    const key = assistantInsightsMetaStorageKey("personal_assistant");
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return;
      const parsed = JSON.parse(raw) as { sessionId?: string; persist?: boolean };
      if (!parsed?.persist || !parsed.sessionId) return;
      void listAssistantSessionInsights(parsed.sessionId).then((res) => {
        if (!cancelled) hydrateFromServer(res.session_id, res.insights as AssistantInsight[]);
      });
    } catch {
      /* ignore */
    }
    return () => {
      cancelled = true;
    };
  }, [hydrateFromServer]);

  useEffect(() => {
    const key = assistantInsightsMetaStorageKey("personal_assistant");
    if (listening) {
      sessionStorage.setItem(key, JSON.stringify({ sessionId, persist: saveTranscript }));
    }
  }, [listening, saveTranscript, sessionId]);

  const { speak, stop, displayStatus, errorMessage, busy } = useSpeakAssistantTts(settings.voiceName);

  const speakStatusLine = useMemo(() => {
    if (displayStatus === "error") return `Error — ${errorMessage || "Playback failed."}`;
    if (displayStatus === "stopped") return "Stopped.";
    if (busy) return "Speaking or preparing audio…";
    return "Ready — tap Speak Now on a suggestion.";
  }, [displayStatus, errorMessage, busy]);

  const persistStatus = useCallback(
    async (ins: AssistantInsight, status: AssistantInsightActionStatus) => {
      if (!ins.persisted || !ins.id) return;
      try {
        await patchAssistantInsightAction(ins.id, status);
        setInsights((prev) => prev.map((x) => (x.id === ins.id ? { ...x, action_status: status } : x)));
      } catch {
        /* non-fatal */
      }
    },
    [setInsights]
  );

  const visibleInsights = useMemo(
    () => insights.filter((i) => !dismissedSet.has(i.id)),
    [insights, dismissedSet]
  );

  const handRaiseQueue = useMemo(() => {
    return visibleInsights
      .filter((i) => i.show_hand_raise)
      .sort((a, b) => handRaiseTier(b) - handRaiseTier(a) || b.confidence - a.confidence);
  }, [visibleInsights]);

  const primaryHandRaise = handRaiseQueue[0];
  const queuedHandRaises = handRaiseQueue.slice(1, 8);

  const statusLabel = useMemo(() => {
    if (wsError) return wsError;
    switch (analysisStatus) {
      case "idle":
        return listening ? "Connecting…" : "Idle — start a session to listen.";
      case "listening":
        return "Listening — live transcript; analysis every 60s when there is new text.";
      case "checking":
        return "Checking knowledge base…";
      case "found":
        return "Insights updated — tap highlights or suggestion card for details.";
      case "none":
        return "No strong evidence for this window.";
      default:
        return "";
    }
  }, [analysisStatus, listening, wsError]);

  const handleHandRaiseAction = (insight: AssistantInsight, action: HandRaiseAction) => {
    if (action !== "speak_now") {
      const st = handRaiseActionToStatus(action);
      if (st) void persistStatus(insight, st);
    }

    switch (action) {
      case "view_details":
        setPanelInsight(insight);
        break;
      case "ignore":
        setDismissedIds((prev) => (prev.includes(insight.id) ? prev : [...prev, insight.id]));
        setPanelInsight((cur) => (cur?.id === insight.id ? null : cur));
        break;
      case "save_for_later":
        setSavedForLater((prev) => (prev.some((x) => x.id === insight.id) ? prev : [...prev, insight]));
        break;
      case "ask_follow_up":
        if (onOpenKnowledgeChatWithDraft) {
          onOpenKnowledgeChatWithDraft(buildFollowUpDraft(insight));
        }
        break;
      case "speak_now": {
        const text =
          (insight.suggested_response && insight.suggested_response.trim()) ||
          (insight.assistant_interpretation && insight.assistant_interpretation.trim()) ||
          "";
        void (async () => {
          await persistStatus(insight, "spoke_now");
          await speak(text);
        })();
        break;
      }
      default:
        break;
    }
  };

  const onStartFromModal = () => {
    const name = (modalName || "").trim() || defaultTranscriptName();
    const location = (modalLocation || "").trim() || "default";
    prepareNewListeningSession();
    setPanelInsight(null);
    setDismissedIds([]);
    setSavedForLater([]);
    void startSession(name, location, { autoStore: saveTranscript });
  };

  const onStop = () => {
    stop();
    stopTimers();
    void handleStopAndExtractTags();
    void runAnalysis();
  };

  const onClear = () => {
    stop();
    stopTimers();
    clearAndReset();
    resetSession();
    setPanelInsight(null);
    setDismissedIds([]);
    setSavedForLater([]);
    try {
      sessionStorage.removeItem(assistantInsightsMetaStorageKey("personal_assistant"));
    } catch {
      /* ignore */
    }
  };

  const openInsightPanel = (ins: AssistantInsight) => {
    setPanelInsight(ins);
    void persistStatus(ins, "viewed");
  };

  const openSourcePreview = (ins: AssistantInsight, index = 0) => {
    const evidence = ins.evidence || [];
    if (!evidence.length) return;
    setPreviewInitialIndex(index);
    setPreviewEvidence(evidence);
    void persistStatus(ins, "viewed");
  };

  return (
    <div className="h-full min-h-0 flex flex-col lg:flex-row gap-3 rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <div className="flex-1 min-h-0 flex flex-col min-w-0 p-4 sm:p-5">
        {showStartModal && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
            onClick={() => setShowStartModal(false)}
          >
            <div
              className="rounded-2xl border border-white/20 bg-slate-900 shadow-xl max-w-md w-full p-5 space-y-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="font-semibold text-white">Start Personal Assistant</div>
              <p className="text-sm text-slate-400">
                Same live analysis as Silent Assistant, plus Hand Raise suggestions when confidence and evidence are
                strong enough. Nothing is spoken automatically.
              </p>
              <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={saveTranscript}
                  disabled={listening}
                  onChange={(e) => setSaveTranscript(e.target.checked)}
                  className="rounded border-white/20"
                />
                Save transcript to database &amp; RAG (same as Live Transcript)
              </label>
              <p className="text-[11px] text-slate-500">
                When saving the transcript, visible insights are stored in SQLite for review (not indexed into RAG).
              </p>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Name</label>
                <input
                  type="text"
                  value={modalName}
                  onChange={(e) => setModalName(e.target.value)}
                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white"
                  placeholder="Session name"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
                <input
                  type="text"
                  value={modalLocation}
                  onChange={(e) => setModalLocation(e.target.value)}
                  className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white"
                  placeholder="e.g. default"
                />
              </div>
              <div className="flex flex-wrap gap-2 pt-2">
                <button type="button" onClick={applyDefault} className="rounded-xl px-4 py-2 text-sm bg-white/10 text-slate-300">
                  Default
                </button>
                <button type="button" onClick={() => setShowStartModal(false)} className="rounded-xl px-4 py-2 text-sm bg-white/10 text-slate-400">
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={onStartFromModal}
                  className="rounded-xl px-4 py-2 text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                >
                  Start
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 mb-2">
          <h1 className="text-lg font-semibold text-white">Personal Assistant</h1>
          {listening && wsStatus === "ready" && (
            <span className="text-[10px] font-medium uppercase tracking-wide text-red-300/90 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" aria-hidden />
              Live
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
          saveTranscript={saveTranscript}
          onSaveTranscriptChange={setSaveTranscript}
          showSaveToggle
          disableSaveToggleWhileListening
        />

        <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-slate-400 mb-2">
          {statusLabel}
          {analysisError && <span className="block text-red-400/90 mt-1">{analysisError}</span>}
        </div>

        {primaryHandRaise && (
          <div className="mb-3 space-y-2">
            <HandRaiseCard
              insight={primaryHandRaise}
              askFollowUpEnabled={Boolean(onOpenKnowledgeChatWithDraft)}
              speakNowBusy={busy}
              onPreviewSource={() => openSourcePreview(primaryHandRaise, 0)}
              onAction={(a) => handleHandRaiseAction(primaryHandRaise, a)}
            />
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
              <span>Speak Now: {speakStatusLine}</span>
              {busy && (
                <button type="button" className="text-amber-200/90 underline font-medium" onClick={() => stop()}>
                  Stop speaking
                </button>
              )}
            </div>
          </div>
        )}

        {queuedHandRaises.length > 0 && (
          <div className="mb-3 rounded-lg border border-white/10 bg-black/20 p-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500 mb-2">More suggestions</div>
            <ul className="space-y-1">
              {queuedHandRaises.map((i) => (
                <li key={i.id}>
                  <button
                    type="button"
                    onClick={() => openInsightPanel(i)}
                    className="w-full text-left text-xs rounded-md px-2 py-1.5 text-slate-300 hover:bg-white/10 border border-transparent hover:border-white/10"
                  >
                    <span className="text-slate-400">{i.classification.replace("_", " ")}</span>
                    <span className="text-slate-600 mx-1">·</span>
                    <span>{(i.confidence * 100).toFixed(0)}%</span>
                    <span className="block text-[10px] text-slate-500 truncate mt-0.5">{i.transcript_text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {savedForLater.length > 0 && (
          <div className="mb-3 rounded-lg border border-violet-500/20 bg-violet-500/5 p-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-violet-300/80 mb-1">Saved for later (this session)</div>
            <ul className="text-[11px] text-slate-400 space-y-1 max-h-24 overflow-y-auto">
              {savedForLater.map((i) => (
                <li key={i.id}>
                  <button
                    type="button"
                    className="text-left w-full hover:text-slate-200 underline-offset-2 hover:underline"
                    onClick={() => openInsightPanel(i)}
                  >
                    {i.transcript_text.slice(0, 80)}
                    {i.transcript_text.length > 80 ? "…" : ""}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <SessionInsightsReview insights={insights} onSelectInsight={openInsightPanel} />

        <AssistantTranscriptPanel fullTranscript={fullTranscript} insights={visibleInsights} onHighlightClick={openInsightPanel} />
      </div>

      <aside className="w-full lg:w-[min(400px,100%)] shrink-0 border-t lg:border-t-0 lg:border-l border-white/10 bg-[#080b12] flex flex-col min-h-[200px] lg:min-h-0 max-h-[50vh] lg:max-h-none">
        <div className="p-3 border-b border-white/5 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Suggestion &amp; evidence
        </div>
        <InsightPanel
          selectedInsight={panelInsight}
          showSuggestedResponse
          emptyHint="Tap a highlighted phrase, View Details on the suggestion card, or an item under More suggestions."
        />
      </aside>
      {previewEvidence && previewEvidence.length > 0 && (
        <EvidenceSourcePreviewModal
          evidence={previewEvidence}
          initialIndex={previewInitialIndex}
          onClose={() => setPreviewEvidence(null)}
        />
      )}
    </div>
  );
};

export default PersonalAssistant;
