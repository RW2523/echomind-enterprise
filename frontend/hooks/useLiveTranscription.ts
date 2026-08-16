import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { transcribeWsUrl, getTranscriptTags, updateTranscript, createBoardroomSession, uploadBoardroomChunk, finalizeBoardroomSession, getBoardroomSession, linkBoardroomTranscript, getScenarios } from "../services/backend";
import type {
  AnalysisCard, TranscriptSegment, BoardroomSession,
  ScenarioId, Scenario, AnalysisMode, Role, TagSpec, SentenceCheck, CheckStatus,
  DetectedEntity, Subject, RecordHit, ActionItem, SessionAck, WsWarning, ScenarioSuggestion,
} from "../types";
import { getActiveNamespace } from "../packs";
import { FALLBACK_SCENARIOS, findScenario, deriveActionItems, deriveLegacyLabel, asCheck } from "../utils/silentAssistant";

const SCENARIO_STORAGE_KEY = "echomind.scenario";
const WARNING_TTL_MS = 25000;

function loadScenarioPref(): ScenarioId {
  try {
    const v = localStorage.getItem(SCENARIO_STORAGE_KEY);
    return (v && v.trim()) ? (v as ScenarioId) : "auto";
  } catch { return "auto"; }
}
function saveScenarioPref(v: ScenarioId) {
  try { localStorage.setItem(SCENARIO_STORAGE_KEY, v); } catch {}
}

/** Kyutai STT sample rate (24kHz). Backend sends this in ready message. */
const KYUTAI_SAMPLE_RATE = 24000;
const OPEN_TIMEOUT_MS = 15000;
const READY_TIMEOUT_MS = 300000;
const HEARTBEAT_INTERVAL_MS = 20000;  // Keep connection alive when tab backgrounded (audio stops)
const RECONNECT_DELAY_MS = 1200;
const MAX_RECONNECT_ATTEMPTS = 5;

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

export interface UseLiveTranscriptionReturn {
  fullTranscript: string;
  partial: string;
  /** Completed paragraph segments (for highlighting) */
  transcriptSegments: TranscriptSegment[];
  /** Analysis cards from Silent Assistant */
  analysisCards: AnalysisCard[];
  /** Segment IDs currently being analyzed (spinner state) */
  analyzingSegmentIds: Set<string>;
  /** ID of the currently selected segment/card (for bidirectional highlight) */
  selectedSegmentId: string | null;
  setSelectedSegmentId: (id: string | null) => void;
  listening: boolean;
  wsStatus: "idle" | "connecting" | "loading" | "ready" | "error";
  wsError: string | null;
  sessionName: string;
  sessionLocation: string;
  sessionStartedAt: Date | null;
  customTags: string[];
  newTagInput: string;
  setSessionName: (v: string) => void;
  setSessionLocation: (v: string) => void;
  setNewTagInput: (v: string) => void;
  openStartModal: () => void;
  startSession: (name: string, location: string) => Promise<void>;
  handleStopAndExtractTags: () => Promise<void>;
  clearAndReset: () => void;
  addTag: () => void;
  removeTag: (tag: string) => void;
  micMuted: boolean;
  setMicMuted: (muted: boolean) => void;
  showStartModal: boolean;
  modalName: string;
  modalLocation: string;
  setModalName: (v: string) => void;
  setModalLocation: (v: string) => void;
  setShowStartModal: (v: boolean) => void;
  applyDefault: () => void;
  /** Boardroom mode */
  boardroomMode: boolean;
  setBoardroomMode: (v: boolean) => void;
  boardroomSession: BoardroomSession | null;
  setBoardroomSession: (s: BoardroomSession | null) => void;
  boardroomUploading: boolean;
  endBoardroomSession: () => Promise<void>;

  // ── Silent Assistant v2 ────────────────────────────────────────────────────
  /** User's scenario preference ('auto' = let the server detect). Persisted in localStorage. */
  scenario: ScenarioId;
  /** Set scenario; when live also sends {type:'scenario'} so the server switches profile. */
  setScenario: (id: ScenarioId) => void;
  /** Scenario profiles from GET /api/transcribe/scenarios (static fallback until loaded). */
  scenarios: Scenario[];
  /** Server-detected scenario suggestion (accept -> setScenario, dismiss -> hide). */
  scenarioSuggestion: ScenarioSuggestion | null;
  acceptScenarioSuggestion: () => void;
  dismissScenarioSuggestion: () => void;
  /** KB namespace to search ('' = all documents). Default = active vertical pack. */
  kbNamespace: string;
  setKbNamespace: (ns: string) => void;
  analysisMode: AnalysisMode;
  /** Set analysis mode; when live also sends {type:'analysis_mode'} (see PROTOCOL notes). */
  setAnalysisMode: (m: AnalysisMode) => void;
  /** Optional 'who is on the call' name -> start.subject_hint.name */
  subjectHint: string;
  setSubjectHint: (v: string) => void;
  /** Role currently speaking (role id from session.roles) or null = unknown. */
  myRole: Role | null;
  /** Sends {type:'speaker', role}. */
  setSpeakerRole: (role: Role | null) => void;
  /** Tag vocabulary + role ids from the `session` ack (fallback: profile defaults). */
  tagVocab: TagSpec[];
  roles: { me: Role; other: Role };
  sessionAck: SessionAck | null;
  /** Sentence checks keyed by sentence_id (legacy analysis payloads fall back to segment_id). */
  checks: Record<string, SentenceCheck>;
  /** Per-sentence lifecycle (pending / checked / skipped / timeout / no_tags). */
  sentenceStatus: Record<string, CheckStatus>;
  entities: DetectedEntity[];
  subjects: Subject[];
  records: RecordHit[];
  /** Derived from checks tagged action-item / commitment / decision. */
  actionItems: ActionItem[];
  wsWarning: WsWarning | null;
  clearWarning: () => void;
  confirmSubject: (subjectId: string) => void;
  rejectSubject: (subjectId: string) => void;
  /** Sentence id selected in the transcript / checks (bidirectional highlight). */
  selectedSentenceId: string | null;
  setSelectedSentenceId: (id: string | null) => void;
}

export function useLiveTranscription(defaultName: () => string): UseLiveTranscriptionReturn {
  const [fullTranscript, setFullTranscript] = useState("");
  const [partial, setPartial] = useState("");
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[]>([]);
  const [analyzingSegmentIds, setAnalyzingSegmentIds] = useState<Set<string>>(new Set());
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const [wsStatus, setWsStatus] = useState<"idle" | "connecting" | "loading" | "ready" | "error">("idle");
  const [wsError, setWsError] = useState<string | null>(null);

  const [showStartModal, setShowStartModal] = useState(false);
  const [modalName, setModalName] = useState("");
  const [modalLocation, setModalLocation] = useState("");

  const [sessionName, setSessionName] = useState("");
  const [sessionLocation, setSessionLocation] = useState("");
  const [sessionStartedAt, setSessionStartedAt] = useState<Date | null>(null);
  const [customTags, setCustomTags] = useState<string[]>([]);
  const [newTagInput, setNewTagInput] = useState("");
  const [micMuted, setMicMuted] = useState(false);

  // ── Silent Assistant v2 state ─────────────────────────────────────────────
  const [scenario, setScenarioState] = useState<ScenarioId>(() => loadScenarioPref());
  const [scenarios, setScenarios] = useState<Scenario[]>(FALLBACK_SCENARIOS);
  const [scenarioSuggestion, setScenarioSuggestion] = useState<ScenarioSuggestion | null>(null);
  const [kbNamespace, setKbNamespaceState] = useState<string>(() => getActiveNamespace());
  const [analysisMode, setAnalysisModeState] = useState<AnalysisMode>(() =>
    loadScenarioPref() === "general" ? "flags_only" : "flags_and_records"
  );
  const [subjectHint, setSubjectHint] = useState("");
  const [myRole, setMyRole] = useState<Role | null>(null);
  const [sessionAck, setSessionAck] = useState<SessionAck | null>(null);
  const [checks, setChecks] = useState<Record<string, SentenceCheck>>({});
  const [sentenceStatus, setSentenceStatus] = useState<Record<string, CheckStatus>>({});
  const [entities, setEntities] = useState<DetectedEntity[]>([]);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [records, setRecords] = useState<RecordHit[]>([]);
  const [wsWarning, setWsWarning] = useState<WsWarning | null>(null);
  const [selectedSentenceId, setSelectedSentenceId] = useState<string | null>(null);
  const scenarioRef = useRef<ScenarioId>(scenario);
  scenarioRef.current = scenario;
  const scenariosRef = useRef<Scenario[]>(scenarios);
  scenariosRef.current = scenarios;
  const kbNamespaceRef = useRef(kbNamespace);
  kbNamespaceRef.current = kbNamespace;
  const analysisModeRef = useRef(analysisMode);
  analysisModeRef.current = analysisMode;
  const analysisModeTouchedRef = useRef(false);
  const subjectHintRef = useRef("");
  subjectHintRef.current = subjectHint;
  const myRoleRef = useRef<Role | null>(null);
  myRoleRef.current = myRole;
  const dismissedSuggestionsRef = useRef<Set<string>>(new Set());
  const resolvedScenarioRef = useRef<string | null>(null); // server-side scenario from the last `session` ack
  const warningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const checkOrderRef = useRef<string[]>([]); // insertion order of check keys (for analysisCards)

  // Boardroom mode state
  const [boardroomMode, setBoardroomMode] = useState(false);
  const [boardroomSession, setBoardroomSession] = useState<BoardroomSession | null>(null);
  const [boardroomUploading, setBoardroomUploading] = useState(false);
  const boardroomSessionIdRef = useRef<string | null>(null);
  const boardroomRecorderRef = useRef<MediaRecorder | null>(null);
  const boardroomChunksRef = useRef<Blob[]>([]);
  // Actual container the recorder used ('webm' | 'ogg' | ...) so the upload format matches reality. (audit L)
  const boardroomFormatRef = useRef<string>("webm");
  const boardroomChunkIndexRef = useRef(0);
  const boardroomModeRef = useRef(false);
  boardroomModeRef.current = boardroomMode;

  const wsRef = useRef<WebSocket | null>(null);
  const micMutedRef = useRef(false);
  micMutedRef.current = micMuted;
  const recRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const boardroomPollCancelRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const transcriptForTagsRef = useRef("");
  const lastStoredTranscriptIdRef = useRef<string | null>(null);
  const pendingTagsRef = useRef<string[] | null>(null);
  const listeningRef = useRef(false);
  listeningRef.current = listening;
  const userInitiatedCloseRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const sessionNameRef = useRef("");
  const sessionLocationRef = useRef("");
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  sessionNameRef.current = sessionName;
  sessionLocationRef.current = sessionLocation;

  const stopMic = useCallback((sendStop: boolean) => {
    heartbeatIntervalRef.current && clearInterval(heartbeatIntervalRef.current);
    heartbeatIntervalRef.current = null;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (sendStop) wsRef.current.send(JSON.stringify({ type: "stop" }));
      else wsRef.current.close(); // App unmount: close without sending stop
    }
    processorRef.current?.disconnect();
    processorRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    recRef.current?.getTracks().forEach((t) => t.stop());
    recRef.current = null;
    wsRef.current = null;
    setListening(false);
  }, []);

  /** Send a JSON control message if the socket is open. Returns true when sent. */
  const wsSend = useCallback((payload: Record<string, unknown>): boolean => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }, []);

  /** Clear all Silent Assistant v2 state (new session / clear). */
  const resetAssistantState = useCallback(() => {
    setChecks({});
    checkOrderRef.current = [];
    setSentenceStatus({});
    setEntities([]);
    setSubjects([]);
    setRecords([]);
    setSessionAck(null);
    resolvedScenarioRef.current = null;
    setScenarioSuggestion(null);
    dismissedSuggestionsRef.current = new Set();
    setMyRole(null);
    setSelectedSentenceId(null);
    setWsWarning(null);
    if (warningTimerRef.current) { clearTimeout(warningTimerRef.current); warningTimerRef.current = null; }
  }, []);

  const showWarning = useCallback((code: string, message: string) => {
    setWsWarning({ code, message, at: Date.now() });
    if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
    warningTimerRef.current = setTimeout(() => setWsWarning(null), WARNING_TTL_MS);
  }, []);

  /** Upsert a check (keyed by sentence_id, falling back to segment_id for legacy payloads). */
  const upsertCheck = useCallback((raw: any): SentenceCheck => {
    const key: string = raw.sentence_id || raw.segment_id;
    // Defensive normalisation: tags may arrive as plain ids, evidence may omit `kind`.
    const tags = Array.isArray(raw.tags)
      ? raw.tags.map((t: any) => (typeof t === "string" ? { tag: t } : t)).filter((t: any) => t && t.tag)
      : [];
    const evidence = Array.isArray(raw.evidence)
      ? raw.evidence.filter((e: any) => e && typeof e.quote === "string").map((e: any) => ({ ...e, kind: e.kind ?? (e.rule_id ? "rule" : "document") }))
      : [];
    const check: SentenceCheck = asCheck({
      ...raw,
      tags,
      evidence,
      id: raw.id ?? key,
      segment_id: raw.segment_id ?? key,
      segment_text: raw.segment_text ?? raw.sentence_text ?? "",
      sentence_id: key,
      sentence_text: raw.sentence_text ?? raw.segment_text ?? "",
      label: raw.label ?? deriveLegacyLabel({ ...raw, tags }),
      status: "checked",
    });
    if (!checkOrderRef.current.includes(key)) checkOrderRef.current = [...checkOrderRef.current, key];
    setChecks((prev) => ({ ...prev, [key]: check }));
    setSentenceStatus((prev) => ({ ...prev, [key]: "checked" }));
    return check;
  }, []);

  const doStart = useCallback(
    async (name: string, location: string, isReconnect = false) => {
      if (listeningRef.current && !isReconnect) return;
      userInitiatedCloseRef.current = false;
      sessionNameRef.current = name || "";
      sessionLocationRef.current = location || "default";
      // On reconnect: keep existing transcript visible so user doesn't see a blank screen
      if (!isReconnect) {
        setFullTranscript("");
        setPartial("");
        setTranscriptSegments([]);
        setSelectedSegmentId(null);
        resetAssistantState();
      } else {
        // The new server session has no memory of pending sentences from before the drop.
        setSentenceStatus((prev) => {
          const next: Record<string, CheckStatus> = {};
          for (const [k, v] of Object.entries(prev)) if (v !== "pending") next[k] = v;
          return next;
        });
      }
      // Always clear in-flight analysis spinners: the new connection is a fresh server session
      // and will never emit analysis_done for paragraph ids from before the drop. (audit M)
      setAnalyzingSegmentIds(new Set());
      setWsError(null);
      setWsStatus("connecting");

      const ws = new WebSocket(transcribeWsUrl());
      wsRef.current = ws;

      const handleError = (err: string) => {
        setWsError(err);
        setWsStatus("error");
        stopMic(false);
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "loading") setWsStatus("loading");
          if (msg.type === "ready") setWsStatus("ready");
          if (msg.type === "partial") {
            // Session is stable (producing transcripts) -> safe to reset the reconnect budget. (M21)
            reconnectAttemptsRef.current = 0;
            const t = msg.text ?? "";
            setFullTranscript(t);
            transcriptForTagsRef.current = t;
            // partial_text = text of the current live paragraph (not yet committed as a segment).
            // Falls back to full text when no segments exist yet (first ~2 s of speech).
            setPartial(msg.partial_text ?? "");
            // Update segment texts from partial payload (preserve v1 label + v2 role/sentences)
            if (Array.isArray(msg.segments)) {
              setTranscriptSegments((prev) => {
                const prevMap = new Map(prev.map((s) => [s.paragraph_id, s]));
                return msg.segments.map((s: { paragraph_id: string; text: string; role?: string | null; sentences?: any[] }) => {
                  const p = prevMap.get(s.paragraph_id);
                  return {
                    paragraph_id: s.paragraph_id,
                    text: s.text,
                    label: p?.label,
                    confidence: p?.confidence,
                    role: s.role ?? p?.role,
                    sentences: Array.isArray(s.sentences) ? s.sentences : p?.sentences,
                  };
                });
              });
            }
          }
          if (msg.type === "segment") {
            // A paragraph has been completed — add/update in segments list (+ v2 role/sentences)
            const patch: Partial<TranscriptSegment> = { text: msg.text };
            if (msg.role !== undefined) patch.role = msg.role;
            if (Array.isArray(msg.sentences)) patch.sentences = msg.sentences;
            setTranscriptSegments((prev) => {
              const exists = prev.find((s) => s.paragraph_id === msg.paragraph_id);
              if (exists) {
                return prev.map((s) =>
                  s.paragraph_id === msg.paragraph_id ? { ...s, ...patch } : s
                );
              }
              return [...prev, { paragraph_id: msg.paragraph_id, text: msg.text, ...patch }];
            });
          }
          if (msg.type === "analysis_start") {
            // Silent Assistant started analyzing this segment/sentence — show spinner
            setAnalyzingSegmentIds((prev) => new Set([...prev, msg.segment_id]));
            const key: string | undefined = msg.sentence_id || msg.segment_id;
            if (key) setSentenceStatus((prev) => (prev[key] === "checked" ? prev : { ...prev, [key]: "pending" }));
          }
          if (msg.type === "analysis_done") {
            // Analysis finished (no result to show) — clear spinner, record status
            setAnalyzingSegmentIds((prev) => {
              const next = new Set(prev);
              next.delete(msg.segment_id);
              return next;
            });
            const key: string | undefined = msg.sentence_id || msg.segment_id;
            const status: CheckStatus =
              msg.status === "checked" || msg.status === "skipped" || msg.status === "timeout" || msg.status === "no_tags"
                ? msg.status : "checked";
            if (key) setSentenceStatus((prev) => (prev[key] === "checked" && status !== "checked" ? prev : { ...prev, [key]: status }));
          }
          if (msg.type === "analysis") {
            // Silent Assistant result — add card, annotate segment, clear spinner
            setAnalyzingSegmentIds((prev) => {
              const next = new Set(prev);
              next.delete(msg.segment_id);
              return next;
            });
            const check = upsertCheck(msg);
            // Legacy annotate: segment-level label (only when the payload is segment-scoped, i.e. v1)
            if (!msg.sentence_id || msg.sentence_id === msg.segment_id) {
              setTranscriptSegments((prev) =>
                prev.map((s) =>
                  s.paragraph_id === msg.segment_id
                    ? { ...s, label: check.label, confidence: check.confidence }
                    : s
                )
              );
            }
          }
          if (msg.type === "session") {
            const ack: SessionAck = {
              session_id: msg.session_id,
              scenario: msg.scenario,
              scenario_label: msg.scenario_label ?? msg.scenario,
              namespace: msg.namespace ?? "",
              analysis_mode: msg.analysis_mode ?? analysisModeRef.current,
              kb_docs: typeof msg.kb_docs === "number" ? msg.kb_docs : 0,
              roles: msg.roles ?? { me: "me", other: "other" },
              tag_vocab: Array.isArray(msg.tag_vocab) ? msg.tag_vocab : [],
            };
            setSessionAck(ack);
            resolvedScenarioRef.current = ack.scenario ?? null;
            // Adopt the server's effective analysis mode unless the user explicitly picked one.
            if (!analysisModeTouchedRef.current && ack.analysis_mode) setAnalysisModeState(ack.analysis_mode);
            // A suggestion for the scenario we now run is moot.
            setScenarioSuggestion((s) => (s && s.scenario === ack.scenario ? null : s));
            // Keep the chosen speaker role valid across a profile switch (role ids change per profile).
            const cur = myRoleRef.current;
            if (cur && cur !== ack.roles.me && cur !== ack.roles.other) {
              setMyRole(null);
            }
          }
          if (msg.type === "warning") {
            showWarning(msg.code ?? "warning", msg.message ?? "");
          }
          if (msg.type === "overloaded") {
            showWarning("overloaded", msg.message ?? "Server is overloaded — analysis may lag.");
          }
          if (msg.type === "entity" && msg.id) {
            const ent: DetectedEntity = msg;
            setEntities((prev) => {
              const i = prev.findIndex((e) => e.id === ent.id);
              if (i >= 0) { const next = [...prev]; next[i] = ent; return next; }
              return [...prev, ent];
            });
          }
          if (msg.type === "subject" && msg.id) {
            const sub: Subject = {
              id: msg.id, kind: msg.kind ?? "person", display_name: msg.display_name ?? "Unknown",
              matched_fields: Array.isArray(msg.matched_fields) ? msg.matched_fields : [],
              entity_ids: Array.isArray(msg.entity_ids) ? msg.entity_ids : [],
              confidence: msg.confidence, status: msg.status ?? "candidate", records_count: msg.records_count,
            };
            setSubjects((prev) => {
              const i = prev.findIndex((s) => s.id === sub.id);
              if (i >= 0) { const next = [...prev]; next[i] = sub; return next; }
              return [...prev, sub];
            });
          }
          if (msg.type === "record" && msg.id) {
            const rec: RecordHit = { ...msg, quotes: Array.isArray(msg.quotes) ? msg.quotes : [], title: msg.title ?? msg.doc_title ?? "Record" };
            setRecords((prev) => {
              const i = prev.findIndex((r) => r.id === rec.id);
              if (i >= 0) { const next = [...prev]; next[i] = rec; return next; }
              return [...prev, rec];
            });
          }
          if (msg.type === "scenario_suggest" && msg.scenario) {
            const sid: string = msg.scenario;
            const dismissed = dismissedSuggestionsRef.current.has(sid);
            const alreadyRunning = sid === resolvedScenarioRef.current || sid === (scenarioRef.current === "auto" ? null : scenarioRef.current);
            if (!dismissed && !alreadyRunning) {
              setScenarioSuggestion({ scenario: sid, confidence: Number(msg.confidence) || 0, reason: msg.reason });
            }
          }
          if (msg.type === "final") {
            const t = (msg.text ?? "").trim();
            setFullTranscript(t);
            transcriptForTagsRef.current = t;
            setPartial("");
            // Merge the final segments (incl. the trailing paragraph closed at EOS) so the last
            // spoken sentence doesn't vanish from the segment list, preserving any labels. (audit H3)
            if (Array.isArray(msg.segments)) {
              setTranscriptSegments((prev) => {
                const prevMap = new Map(prev.map((s) => [s.paragraph_id, s]));
                return msg.segments.map((s: { paragraph_id: string; text: string; role?: string | null; sentences?: any[] }) => {
                  const p = prevMap.get(s.paragraph_id);
                  return {
                    paragraph_id: s.paragraph_id,
                    text: s.text,
                    label: p?.label,
                    confidence: p?.confidence,
                    role: s.role ?? p?.role,
                    sentences: Array.isArray(s.sentences) ? s.sentences : p?.sentences,
                  };
                });
              });
            }
          }
          if (msg.type === "stored") {
            setWsError(null);
            const tid = msg.transcript_id;
            if (tid) {
              lastStoredTranscriptIdRef.current = tid;
              if (pendingTagsRef.current?.length) {
                updateTranscript(tid, { tags: pendingTagsRef.current }).catch(() => {});
                pendingTagsRef.current = null;
              }
              // Link this transcript to the active boardroom session (first time only)
              const bsid = boardroomSessionIdRef.current;
              if (bsid) {
                linkBoardroomTranscript(bsid, tid).catch(() => {});
              }
            }
          }
          if (msg.type === "error") {
            const m = msg.message || "Server error";
            if (m.includes("Reconnecting")) {
              stopMic(false);
              setTimeout(() => doStart(sessionNameRef.current, sessionLocationRef.current, true), RECONNECT_DELAY_MS);
              return;
            }
            console.error(m);
            handleError(m);
          }
        } catch {}
      };

      ws.onerror = () => handleError("WebSocket error");
      ws.onclose = (ev) => {
        const wasListening = listeningRef.current;
        const shouldReconnect =
          !userInitiatedCloseRef.current &&
          wasListening &&
          reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS;
        stopMic(false);
        if (shouldReconnect) {
          reconnectAttemptsRef.current += 1;
          setWsStatus("connecting");
          const delay = RECONNECT_DELAY_MS * reconnectAttemptsRef.current;
          setTimeout(() => doStart(sessionNameRef.current, sessionLocationRef.current, true), delay);
        } else {
          setWsStatus((s) => (s === "error" ? s : "idle"));
          if (!userInitiatedCloseRef.current && reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
            setWsError("Connection lost after multiple retries. Please restart.");
          }
        }
      };

      try {
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(() => reject(new Error("Connection timeout")), OPEN_TIMEOUT_MS);
          ws.addEventListener("open", () => {
            clearTimeout(t);
            resolve();
          }, { once: true });
          ws.addEventListener("error", () => {
            clearTimeout(t);
            reject(new Error("WebSocket failed"));
          }, { once: true });
        });
      } catch (e) {
        setWsError((e as Error)?.message || "Connection failed");
        setWsStatus("error");
        return;
      }

      const readyPromise = new Promise<number>((resolve, reject) => {
        const t = setTimeout(
          () => reject(new Error("Kyutai STT loading timeout (model may still be downloading)")),
          READY_TIMEOUT_MS
        );
        const check = (ev: MessageEvent) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "ready") {
              clearTimeout(t);
              ws.removeEventListener("message", check);
              resolve(msg.sample_rate ?? KYUTAI_SAMPLE_RATE);
            }
            if (msg.type === "error") {
              clearTimeout(t);
              ws.removeEventListener("message", check);
              reject(new Error(msg.message || "STT failed"));
            }
          } catch {}
        };
        ws.addEventListener("message", check);
      });

      let sampleRate: number;
      try {
        sampleRate = await readyPromise;
      } catch (e) {
        handleError((e as Error)?.message || "Kyutai STT not ready");
        return;
      }

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      } catch (micErr) {
        handleError("Microphone access denied or unavailable. Please allow microphone and try again.");
        return;
      }
      recRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate });
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      // Send the AudioContext's ACTUAL sample rate, not the requested 16k. The {sampleRate} hint
      // is not always honored (Safari/Android frequently stay at 48k); the backend resamples to
      // 16k, but only if told the true rate — otherwise audio is fed in 3x too fast = garbled. (audit H2)
      // Silent Assistant v2 start fields (additive; old servers ignore unknown keys).
      const chosenScenario = scenarioRef.current;
      const profile = chosenScenario !== "auto" ? findScenario(scenariosRef.current, chosenScenario) : undefined;
      const hintName = (subjectHintRef.current || "").trim();
      const participants = profile
        ? [
            { role: profile.roles.me },
            hintName ? { role: profile.roles.other, name: hintName } : { role: profile.roles.other },
          ]
        : undefined;
      ws.send(
        JSON.stringify({
          type: "start",
          auto_store: true,
          sample_rate: audioCtx.sampleRate,
          language: "en",
          name: name || undefined,
          location: location || undefined,
          namespace: kbNamespaceRef.current || undefined,
          analysis_always_surface: true,
          scenario: chosenScenario || "auto",
          participants,
          subject_hint: hintName ? { name: hintName } : undefined,
          analysis_mode: analysisModeRef.current,
        })
      );
      // Re-assert the speaker role after a reconnect (new server session starts as 'unknown').
      if (isReconnect && myRoleRef.current) {
        ws.send(JSON.stringify({ type: "speaker", role: myRoleRef.current }));
      }

      // NOTE: do NOT reset reconnectAttemptsRef here. Resetting right after the handshake means a
      // server that accepts then immediately drops the socket never reaches MAX_RECONNECT_ATTEMPTS
      // (infinite reconnect loop). It's reset only once the session is actually stable — i.e. when
      // the first 'partial'/'final' transcription arrives (see ws.onmessage). (M21)

      const src = audioCtx.createMediaStreamSource(stream);

      let workletNode: AudioWorkletNode;
      try {
        await audioCtx.audioWorklet.addModule('/pcm-processor.js');
        workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
      } catch (workletErr) {
        handleError("AudioWorklet failed to load. Please reload the page.");
        return;
      }
      processorRef.current = workletNode;

      workletNode.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || micMutedRef.current) return;
        const pcm16 = floatTo16BitPCM(new Float32Array(e.data));
        wsRef.current.send(pcm16.buffer);
      };

      src.connect(workletNode);
      workletNode.connect(audioCtx.destination);
      setListening(true);
      heartbeatIntervalRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "ping" }));
        }
      }, HEARTBEAT_INTERVAL_MS);

      // Boardroom mode: start MediaRecorder for full-quality audio capture
      if (boardroomModeRef.current && boardroomSessionIdRef.current) {
        try {
          const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : "audio/ogg";
          const recorder = new MediaRecorder(stream, { mimeType });
          // Record the real container so the backend stores/decodes the right format.
          boardroomFormatRef.current = (recorder.mimeType || mimeType).includes("ogg") ? "ogg" : "webm";
          boardroomChunksRef.current = [];
          boardroomChunkIndexRef.current = 0;
          recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
              boardroomChunksRef.current.push(e.data);
            }
          };
          recorder.start(5000); // 5-second chunks
          boardroomRecorderRef.current = recorder;
        } catch (e) {
          console.warn("Boardroom MediaRecorder failed:", e);
        }
      }
    },
    [stopMic, resetAssistantState, showWarning, upsertCheck]
  );

  const clearAndReset = useCallback(() => {
    userInitiatedCloseRef.current = true;
    reconnectAttemptsRef.current = 0;
    stopMic(true);
    setMicMuted(false);
    setFullTranscript("");
    setPartial("");
    setTranscriptSegments([]);
    resetAssistantState();
    setAnalyzingSegmentIds(new Set());
    setSelectedSegmentId(null);
    setSessionName("");
    setSessionLocation("");
    setSessionStartedAt(null);
    setCustomTags([]);
    setNewTagInput("");
    transcriptForTagsRef.current = "";
    lastStoredTranscriptIdRef.current = null;
    pendingTagsRef.current = null;
    setWsError(null);
    boardroomPollCancelRef.current = true;
    // Clean up boardroom recorder
    if (boardroomRecorderRef.current?.state !== "inactive") {
      boardroomRecorderRef.current?.stop();
    }
    boardroomRecorderRef.current = null;
    boardroomChunksRef.current = [];
    boardroomChunkIndexRef.current = 0;
    boardroomSessionIdRef.current = null;
    setBoardroomSession(null);
  }, [stopMic, resetAssistantState]);

  const startSession = useCallback(
    async (name: string, location: string) => {
      reconnectAttemptsRef.current = 0;
      setMicMuted(false);
      setSessionName(name);
      setSessionLocation(location);
      setSessionStartedAt(new Date());
      setCustomTags([]);
      setTranscriptSegments([]);
      resetAssistantState();
      setAnalyzingSegmentIds(new Set());
      setSelectedSegmentId(null);
      transcriptForTagsRef.current = "";
      lastStoredTranscriptIdRef.current = null;
      pendingTagsRef.current = null;
      setShowStartModal(false);

      // If boardroom mode, create a session first
      if (boardroomModeRef.current) {
        try {
          const { session_id } = await createBoardroomSession();
          boardroomSessionIdRef.current = session_id;
          setBoardroomSession({ id: session_id, status: "recording" });
        } catch (e) {
          console.warn("Failed to create boardroom session:", e);
        }
      }

      await doStart(name, location);
    },
    [doStart, resetAssistantState]
  );

  // ── Silent Assistant v2 controls ──────────────────────────────────────────

  // Load scenario profiles once (static fallback keeps the UI usable if the endpoint is missing).
  useEffect(() => {
    let cancelled = false;
    getScenarios()
      .then((list) => { if (!cancelled && Array.isArray(list) && list.length) setScenarios(list); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const setScenario = useCallback((id: ScenarioId) => {
    const next = (id || "auto") as ScenarioId;
    setScenarioState(next);
    saveScenarioPref(next);
    setScenarioSuggestion((s) => (s && s.scenario === next ? null : s));
    // Default analysis mode follows the profile unless the user picked one explicitly.
    if (!analysisModeTouchedRef.current) {
      const prof = next === "auto" ? undefined : findScenario(scenariosRef.current, next);
      setAnalysisModeState(prof ? prof.analysis_mode_default : "flags_and_records");
    }
    // Live: ask the server to switch profile (it re-sends `session`).
    wsSend({ type: "scenario", scenario: next });
  }, [wsSend]);

  const acceptScenarioSuggestion = useCallback(() => {
    const s = scenarioSuggestion;
    if (!s) return;
    setScenario(s.scenario);
    setScenarioSuggestion(null);
  }, [scenarioSuggestion, setScenario]);

  const dismissScenarioSuggestion = useCallback(() => {
    if (scenarioSuggestion) dismissedSuggestionsRef.current.add(String(scenarioSuggestion.scenario));
    setScenarioSuggestion(null);
  }, [scenarioSuggestion]);

  const setKbNamespace = useCallback((ns: string) => {
    setKbNamespaceState(ns ?? "");
  }, []);

  const setAnalysisMode = useCallback((m: AnalysisMode) => {
    analysisModeTouchedRef.current = true;
    setAnalysisModeState(m);
    // Not in PROTOCOL.md yet: mid-session mode switch. Harmless if the server ignores it.
    wsSend({ type: "analysis_mode", analysis_mode: m });
  }, [wsSend]);

  const setSpeakerRole = useCallback((role: Role | null) => {
    setMyRole(role);
    wsSend({ type: "speaker", role: role ?? null });
  }, [wsSend]);

  const answerSubject = useCallback((subjectId: string, action: "confirm" | "reject") => {
    setSubjects((prev) => prev.map((s) => (s.id === subjectId ? { ...s, status: action === "confirm" ? "confirmed" : "rejected" } : s)));
    wsSend({ type: "subject", subject_id: subjectId, action });
  }, [wsSend]);
  const confirmSubject = useCallback((id: string) => answerSubject(id, "confirm"), [answerSubject]);
  const rejectSubject = useCallback((id: string) => answerSubject(id, "reject"), [answerSubject]);

  const clearWarning = useCallback(() => {
    setWsWarning(null);
    if (warningTimerRef.current) { clearTimeout(warningTimerRef.current); warningTimerRef.current = null; }
  }, []);

  // Cards list = checks in arrival order (legacy consumers: AnalysisPanel, TTS summary, Boardroom).
  const analysisCards: AnalysisCard[] = useMemo(
    () => checkOrderRef.current.map((k) => checks[k]).filter(Boolean),
    [checks]
  );
  const actionItems = useMemo(() => deriveActionItems(analysisCards as SentenceCheck[]), [analysisCards]);

  const activeProfile = useMemo(() => {
    const sid = sessionAck?.scenario ?? (scenario === "auto" ? undefined : scenario);
    return sid ? findScenario(scenarios, sid) : undefined;
  }, [sessionAck, scenario, scenarios]);
  const tagVocab: TagSpec[] = useMemo(() => {
    if (sessionAck?.tag_vocab?.length) return sessionAck.tag_vocab;
    if (activeProfile?.tag_vocab?.length) return activeProfile.tag_vocab;
    // 'auto' before the ack: union of every profile's vocab so chips still render with colour.
    const seen = new Map<string, TagSpec>();
    for (const p of scenarios) for (const t of p.tag_vocab ?? []) if (!seen.has(t.id)) seen.set(t.id, t);
    return [...seen.values()];
  }, [sessionAck, activeProfile, scenarios]);
  const roles = useMemo(
    () => sessionAck?.roles ?? activeProfile?.roles ?? { me: "me", other: "other" },
    [sessionAck, activeProfile]
  );

  const endBoardroomSession = useCallback(async () => {
    const sid = boardroomSessionIdRef.current;
    if (!sid) return;

    setBoardroomUploading(true);
    try {
      // Stop recorder and collect final chunks
      if (boardroomRecorderRef.current?.state === "recording") {
        await new Promise<void>((resolve) => {
          boardroomRecorderRef.current!.onstop = () => resolve();
          boardroomRecorderRef.current!.stop();
        });
      }

      const chunks = [...boardroomChunksRef.current];
      boardroomChunksRef.current = [];

      // Upload chunks to backend (use the format the recorder actually produced)
      const fmt = boardroomFormatRef.current || "webm";
      for (let i = 0; i < chunks.length; i++) {
        try {
          await uploadBoardroomChunk(sid, i, chunks[i], fmt);
        } catch (e) {
          console.warn(`Chunk ${i} upload failed:`, e);
        }
      }

      // Trigger finalization
      await finalizeBoardroomSession(sid);
      setBoardroomSession((prev) => prev ? { ...prev, status: "processing" } : null);

      // Poll for status until transcribed (or cancelled via clearAndReset)
      boardroomPollCancelRef.current = false;
      const poll = async () => {
        let attempts = 0;
        while (attempts < 60) {
          await new Promise((r) => setTimeout(r, 3000));
          if (boardroomPollCancelRef.current) break;
          try {
            const s = await getBoardroomSession(sid);
            setBoardroomSession(s);
            if (s.status === "transcribed" || s.status === "analysed" || s.status === "error") {
              break;
            }
          } catch {}
          attempts++;
        }
      };
      poll().catch(() => {});
    } catch (e) {
      console.error("Boardroom session upload failed:", e);
    } finally {
      setBoardroomUploading(false);
      // Recording is finished — clear the active-session refs so a later session can't reuse this
      // id and the live-transcript 'stored' handler stops linking to it. (L29)
      boardroomSessionIdRef.current = null;
      boardroomRecorderRef.current = null;
      boardroomChunksRef.current = [];
    }
  }, []);

  const handleStopAndExtractTags = useCallback(async () => {
    const text = (transcriptForTagsRef.current || fullTranscript || "").trim();
    userInitiatedCloseRef.current = true;
    stopMic(true);
    if (text) {
      try {
        const { tags } = await getTranscriptTags(text);
        if (tags?.length) {
          setCustomTags(tags);
          pendingTagsRef.current = tags;
          const tid = lastStoredTranscriptIdRef.current;
          if (tid) {
            await updateTranscript(tid, { tags });
            pendingTagsRef.current = null;
          }
        }
      } catch {
        // ignore
      }
    }
  }, [stopMic, fullTranscript]);

  const openStartModal = useCallback(() => {
    setModalName("");
    setModalLocation("");
    setShowStartModal(true);
  }, []);

  const applyDefault = useCallback(() => {
    setModalName(defaultName());
    setModalLocation("default");
  }, [defaultName]);

  const removeTag = useCallback((tag: string) => {
    setCustomTags((prev) => prev.filter((x) => x !== tag));
  }, []);

  const addTagFromInput = useCallback(() => {
    const t = newTagInput.trim();
    if (t && !customTags.includes(t)) {
      setCustomTags((prev) => [...prev, t].slice(0, 20));
      setNewTagInput("");
    }
  }, [newTagInput, customTags]);

  // Resume AudioContext when tab becomes visible (browser suspends it when backgrounded).
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible" || !listeningRef.current) return;
      const ctx = audioCtxRef.current;
      if (ctx?.state === "suspended") {
        ctx.resume().catch(() => {});
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

  // Only stop when App unmounts (user closes tab). NOT when switching in-app tabs.
  useEffect(
    () => () => {
      stopMic(false);
    },
    [stopMic]
  );

  return {
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
    addTag: addTagFromInput,
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
    // Silent Assistant v2
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
    entities,
    subjects,
    records,
    actionItems,
    wsWarning,
    clearWarning,
    confirmSubject,
    rejectSubject,
    selectedSentenceId,
    setSelectedSentenceId,
  };
}
