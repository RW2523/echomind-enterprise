import { useCallback, useEffect, useRef, useState } from "react";
import type { AssistantInsight } from "../types";
import { analyzeAssistantWindow, bulkSaveAssistantInsights } from "../services/backend";
import type { AnalysisUiStatus } from "../components/assistant/assistantUtils";
import { newSessionId, ROLLING_CHARS } from "../components/assistant/assistantUtils";

export type AssistantApiMode = "silent_assistant" | "personal_assistant";

export interface UseAssistantAnalysisOptions {
  /** When true, persist eligible insights after each analyze-window (same intent as saving transcript). */
  persistInsights?: boolean;
  /** Live transcript row id when the server sends `stored` (links insights to transcript). */
  transcriptId?: string | null;
}

export function useAssistantAnalysis(
  mode: AssistantApiMode,
  fullTranscript: string,
  listening: boolean,
  wsStatus: "idle" | "connecting" | "loading" | "ready" | "error",
  options?: UseAssistantAnalysisOptions
) {
  const [insights, setInsights] = useState<AssistantInsight[]>([]);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisUiStatus>("idle");
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(() => newSessionId());

  const sessionIdForApiRef = useRef(sessionId);
  sessionIdForApiRef.current = sessionId;

  const lastAnalyzedLengthRef = useRef(0);
  const analyzeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fullTranscriptRef = useRef(fullTranscript);
  fullTranscriptRef.current = fullTranscript;

  const persistInsightsRef = useRef(Boolean(options?.persistInsights));
  const transcriptIdRef = useRef<string | null>(options?.transcriptId ?? null);
  useEffect(() => {
    persistInsightsRef.current = Boolean(options?.persistInsights);
    transcriptIdRef.current = options?.transcriptId ?? null;
  }, [options?.persistInsights, options?.transcriptId]);

  const stopTimers = useCallback(() => {
    if (analyzeTimerRef.current) {
      clearInterval(analyzeTimerRef.current);
      analyzeTimerRef.current = null;
    }
  }, []);

  const runAnalysis = useCallback(async () => {
    const full = (fullTranscriptRef.current || "").trim();
    if (!full) {
      setAnalysisStatus("listening");
      return;
    }
    const start = lastAnalyzedLengthRef.current;
    const transcript_window = full.slice(start).trim();
    if (!transcript_window) {
      setAnalysisStatus("listening");
      return;
    }
    const rollStart = Math.max(0, start - ROLLING_CHARS);
    const rolling_context = full.slice(rollStart, start).trim();

    setAnalysisStatus("checking");
    setAnalysisError(null);
    try {
      const res = await analyzeAssistantWindow({
        session_id: sessionIdForApiRef.current,
        mode,
        transcript_window,
        rolling_context,
        analysis_scope: { documents: true, transcripts: false, books: true, faqs: true },
      });
      const incoming = res.insights || [];
      setInsights((prev) => {
        const byId = new Map(prev.map((x) => [x.id, x]));
        for (const i of incoming) {
          byId.set(i.id, i);
        }
        return Array.from(byId.values());
      });
      lastAnalyzedLengthRef.current = full.length;
      setAnalysisStatus(incoming.length > 0 ? "found" : "none");

      if (persistInsightsRef.current && incoming.length > 0) {
        try {
          const saveRes = await bulkSaveAssistantInsights({
            session_id: sessionIdForApiRef.current,
            mode,
            transcript_id: transcriptIdRef.current || undefined,
            insights: incoming,
          });
          const idMap = saveRes.id_map || {};
          if (Object.keys(idMap).length > 0) {
            setInsights((prev) => {
              const m = new Map(prev.map((x) => [x.id, x]));
              for (const [oldId, newId] of Object.entries(idMap)) {
                const row = m.get(oldId);
                if (!row) continue;
                if (newId !== oldId) m.delete(oldId);
                m.set(newId, { ...row, id: newId, persisted: true });
              }
              return [...m.values()];
            });
          }
        } catch {
          /* persistence must not break analysis */
        }
      }
    } catch (e) {
      setAnalysisError((e as Error)?.message || "Analysis failed");
      setAnalysisStatus("listening");
    }
  }, [mode]);

  useEffect(() => {
    if (!listening || wsStatus !== "ready") {
      stopTimers();
      if (!listening) {
        setAnalysisStatus("idle");
      }
      return;
    }
    setAnalysisStatus("listening");
    analyzeTimerRef.current = setInterval(() => {
      void runAnalysis();
    }, 60_000);
    return () => {
      stopTimers();
    };
  }, [listening, wsStatus, runAnalysis, stopTimers]);

  const resetSession = useCallback(() => {
    stopTimers();
    const id = newSessionId();
    setSessionId(id);
    lastAnalyzedLengthRef.current = 0;
    setInsights([]);
    setAnalysisStatus("idle");
    setAnalysisError(null);
  }, [stopTimers]);

  const prepareNewListeningSession = useCallback(() => {
    stopTimers();
    const id = newSessionId();
    setSessionId(id);
    lastAnalyzedLengthRef.current = 0;
    setInsights([]);
    setAnalysisError(null);
  }, [stopTimers]);

  const hydrateFromServer = useCallback(
    (sid: string, rows: AssistantInsight[]) => {
      stopTimers();
      setSessionId(sid);
      lastAnalyzedLengthRef.current = 0;
      setInsights(rows.map((r) => ({ ...r, persisted: true })));
      setAnalysisError(null);
      setAnalysisStatus("idle");
    },
    [stopTimers]
  );

  return {
    sessionId,
    insights,
    setInsights,
    analysisStatus,
    setAnalysisStatus,
    analysisError,
    setAnalysisError,
    runAnalysis,
    stopTimers,
    resetSession,
    prepareNewListeningSession,
    hydrateFromServer,
  };
}
