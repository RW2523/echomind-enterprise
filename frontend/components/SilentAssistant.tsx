import React, { useEffect, useMemo, useState } from "react";
import type { AssistantInsight, AssistantInsightActionStatus } from "../types";
import { defaultTranscriptName, listAssistantSessionInsights, patchAssistantInsightAction } from "../services/backend";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import { useAssistantAnalysis } from "../hooks/useAssistantAnalysis";
import AssistantModeControls from "./assistant/AssistantModeControls";
import AssistantTranscriptPanel from "./assistant/AssistantTranscriptPanel";
import InsightPanel from "./assistant/InsightPanel";
import SessionInsightsReview from "./assistant/SessionInsightsReview";
import { assistantInsightsMetaStorageKey } from "./assistant/assistantUtils";

interface SilentAssistantProps {
  liveTranscription: UseLiveTranscriptionReturn;
}

const SilentAssistant: React.FC<SilentAssistantProps> = ({ liveTranscription }) => {
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
  const [selectedInsight, setSelectedInsight] = useState<AssistantInsight | null>(null);

  const analysis = useAssistantAnalysis("silent_assistant", fullTranscript, listening, wsStatus, {
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
    const key = assistantInsightsMetaStorageKey("silent_assistant");
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
    const key = assistantInsightsMetaStorageKey("silent_assistant");
    if (listening) {
      sessionStorage.setItem(key, JSON.stringify({ sessionId, persist: saveTranscript }));
    }
  }, [listening, saveTranscript, sessionId]);

  const persistInsightStatus = async (ins: AssistantInsight | null, status: AssistantInsightActionStatus) => {
    if (!ins?.persisted || !ins.id) return;
    try {
      await patchAssistantInsightAction(ins.id, status);
      setInsights((prev) => prev.map((x) => (x.id === ins.id ? { ...x, action_status: status } : x)));
    } catch {
      /* non-fatal */
    }
  };

  const statusLabel = useMemo(() => {
    if (wsError) return wsError;
    switch (analysisStatus) {
      case "idle":
        return listening ? "Connecting…" : "Idle — start a session to listen.";
      case "listening":
        return "Listening — analysis every 60s when there is new text.";
      case "checking":
        return "Checking knowledge base…";
      case "found":
        return "Insight found — tap highlights for evidence.";
      case "none":
        return "No strong evidence for this window.";
      default:
        return "";
    }
  }, [analysisStatus, listening, wsError]);

  const onStartFromModal = () => {
    const name = (modalName || "").trim() || defaultTranscriptName();
    const location = (modalLocation || "").trim() || "default";
    prepareNewListeningSession();
    setSelectedInsight(null);
    void startSession(name, location, { autoStore: saveTranscript });
  };

  const onStop = () => {
    stopTimers();
    void handleStopAndExtractTags();
    void runAnalysis();
  };

  const onClear = () => {
    stopTimers();
    clearAndReset();
    resetSession();
    setSelectedInsight(null);
    try {
      sessionStorage.removeItem(assistantInsightsMetaStorageKey("silent_assistant"));
    } catch {
      /* ignore */
    }
  };

  const onHighlightClick = (ins: AssistantInsight) => {
    setSelectedInsight(ins);
    void persistInsightStatus(ins, "viewed");
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
              <div className="font-semibold text-white">Start Silent Assistant</div>
              <p className="text-sm text-slate-400">
                Live transcription runs like Live Transcript. Analysis checks your knowledge base every 60 seconds. Nothing is spoken.
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
                When saving the transcript, visible assistant insights are also stored locally (SQLite) for review — they are not added to the RAG index.
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

        <AssistantModeControls
          listening={listening}
          micMuted={micMuted}
          onMicMutedChange={setMicMuted}
          onOpenStart={openStartModal}
          onStop={onStop}
          onClear={onClear}
          showSaveToggle={false}
        />

        <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-slate-400 mb-2">
          {statusLabel}
          {analysisError && <span className="block text-red-400/90 mt-1">{analysisError}</span>}
        </div>

        <SessionInsightsReview
          insights={insights}
          onSelectInsight={(ins) => {
            setSelectedInsight(ins);
            void persistInsightStatus(ins, "viewed");
          }}
        />

        <AssistantTranscriptPanel fullTranscript={fullTranscript} insights={insights} onHighlightClick={onHighlightClick} />
      </div>

      <aside className="w-full lg:w-[min(400px,100%)] shrink-0 border-t lg:border-t-0 lg:border-l border-white/10 bg-[#080b12] flex flex-col min-h-[200px] lg:min-h-0 max-h-[50vh] lg:max-h-none">
        <div className="p-3 border-b border-white/5 text-xs font-semibold uppercase tracking-wide text-slate-500">Insight</div>
        <InsightPanel selectedInsight={selectedInsight} showSuggestedResponse={false} />
      </aside>
    </div>
  );
};

export default SilentAssistant;
