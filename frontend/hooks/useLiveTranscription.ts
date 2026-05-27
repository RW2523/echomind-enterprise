import { useState, useRef, useCallback, useEffect } from "react";
import { transcribeWsUrl, getTranscriptTags, updateTranscript, createBoardroomSession, uploadBoardroomChunk, finalizeBoardroomSession, getBoardroomSession } from "../services/backend";
import type { AnalysisCard, TranscriptSegment, BoardroomSession } from "../types";

/** Kyutai STT sample rate (24kHz). Backend sends this in ready message. */
const KYUTAI_SAMPLE_RATE = 24000;
const OPEN_TIMEOUT_MS = 15000;
const READY_TIMEOUT_MS = 300000;
const HEARTBEAT_INTERVAL_MS = 20000;  // Keep connection alive when tab backgrounded (audio stops)
const RECONNECT_DELAY_MS = 1200;
const MAX_RECONNECT_ATTEMPTS = 5;

function floatTo16BitPCM(input: Float32Array) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function b64FromBytes(bytes: Uint8Array) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunkSize)) as any);
  }
  return btoa(binary);
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
}

export function useLiveTranscription(defaultName: () => string): UseLiveTranscriptionReturn {
  const [fullTranscript, setFullTranscript] = useState("");
  const [partial, setPartial] = useState("");
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[]>([]);
  const [analysisCards, setAnalysisCards] = useState<AnalysisCard[]>([]);
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

  // Boardroom mode state
  const [boardroomMode, setBoardroomMode] = useState(false);
  const [boardroomSession, setBoardroomSession] = useState<BoardroomSession | null>(null);
  const [boardroomUploading, setBoardroomUploading] = useState(false);
  const boardroomSessionIdRef = useRef<string | null>(null);
  const boardroomRecorderRef = useRef<MediaRecorder | null>(null);
  const boardroomChunksRef = useRef<Blob[]>([]);
  const boardroomChunkIndexRef = useRef(0);
  const boardroomModeRef = useRef(false);
  boardroomModeRef.current = boardroomMode;

  const wsRef = useRef<WebSocket | null>(null);
  const micMutedRef = useRef(false);
  micMutedRef.current = micMuted;
  const recRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
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
        setAnalysisCards([]);
        setAnalyzingSegmentIds(new Set());
        setSelectedSegmentId(null);
      }
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
            const t = msg.text ?? "";
            setFullTranscript(t);
            transcriptForTagsRef.current = t;
            // partial_text = text of the current live paragraph (not yet committed as a segment).
            // Falls back to full text when no segments exist yet (first ~2 s of speech).
            setPartial(msg.partial_text ?? "");
            // Update segment texts from partial payload
            if (Array.isArray(msg.segments)) {
              setTranscriptSegments((prev) => {
                const prevMap = new Map(prev.map((s) => [s.paragraph_id, s]));
                return msg.segments.map((s: { paragraph_id: string; text: string }) => ({
                  paragraph_id: s.paragraph_id,
                  text: s.text,
                  label: prevMap.get(s.paragraph_id)?.label,
                  confidence: prevMap.get(s.paragraph_id)?.confidence,
                }));
              });
            }
          }
          if (msg.type === "segment") {
            // A paragraph has been completed — add/update in segments list
            setTranscriptSegments((prev) => {
              const exists = prev.find((s) => s.paragraph_id === msg.paragraph_id);
              if (exists) {
                return prev.map((s) =>
                  s.paragraph_id === msg.paragraph_id ? { ...s, text: msg.text } : s
                );
              }
              return [...prev, { paragraph_id: msg.paragraph_id, text: msg.text }];
            });
          }
          if (msg.type === "analysis_start") {
            // Silent Assistant started analyzing this segment — show spinner
            setAnalyzingSegmentIds((prev) => new Set([...prev, msg.segment_id]));
          }
          if (msg.type === "analysis_done") {
            // Analysis finished (no result to show) — clear spinner
            setAnalyzingSegmentIds((prev) => {
              const next = new Set(prev);
              next.delete(msg.segment_id);
              return next;
            });
          }
          if (msg.type === "analysis") {
            // Silent Assistant result — add card, annotate segment, clear spinner
            setAnalyzingSegmentIds((prev) => {
              const next = new Set(prev);
              next.delete(msg.segment_id);
              return next;
            });
            const card: AnalysisCard = {
              id: msg.id,
              segment_id: msg.segment_id,
              segment_text: msg.segment_text,
              label: msg.label,
              confidence: msg.confidence,
              explanation: msg.explanation,
              source_chunks: msg.source_chunks || [],
            };
            setAnalysisCards((prev) => {
              // Replace if same segment already has a card
              const exists = prev.findIndex((c) => c.segment_id === msg.segment_id);
              if (exists >= 0) {
                const next = [...prev];
                next[exists] = card;
                return next;
              }
              return [...prev, card];
            });
            // Annotate the matching segment with the label
            setTranscriptSegments((prev) =>
              prev.map((s) =>
                s.paragraph_id === msg.segment_id
                  ? { ...s, label: msg.label, confidence: msg.confidence }
                  : s
              )
            );
          }
          if (msg.type === "final") {
            const t = (msg.text ?? "").trim();
            setFullTranscript(t);
            transcriptForTagsRef.current = t;
            setPartial("");
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

      ws.send(
        JSON.stringify({
          type: "start",
          auto_store: true,
          sample_rate: sampleRate,
          language: "en",
          name: name || undefined,
          location: location || undefined,
        })
      );

      // Reset reconnect counter once we've gotten past the WebSocket ready handshake
      reconnectAttemptsRef.current = 0;

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
      const src = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || micMutedRef.current) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(input);
        const b64 = b64FromBytes(new Uint8Array(pcm16.buffer));
        wsRef.current.send(JSON.stringify({ type: "audio", pcm16_b64: b64 }));
      };

      src.connect(processor);
      processor.connect(audioCtx.destination);
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
    [stopMic]
  );

  const clearAndReset = useCallback(() => {
    userInitiatedCloseRef.current = true;
    reconnectAttemptsRef.current = 0;
    stopMic(true);
    setMicMuted(false);
    setFullTranscript("");
    setPartial("");
    setTranscriptSegments([]);
    setAnalysisCards([]);
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
    // Clean up boardroom recorder
    if (boardroomRecorderRef.current?.state !== "inactive") {
      boardroomRecorderRef.current?.stop();
    }
    boardroomRecorderRef.current = null;
    boardroomChunksRef.current = [];
    boardroomChunkIndexRef.current = 0;
    boardroomSessionIdRef.current = null;
    setBoardroomSession(null);
  }, [stopMic]);

  const startSession = useCallback(
    async (name: string, location: string) => {
      reconnectAttemptsRef.current = 0;
      setMicMuted(false);
      setSessionName(name);
      setSessionLocation(location);
      setSessionStartedAt(new Date());
      setCustomTags([]);
      setTranscriptSegments([]);
      setAnalysisCards([]);
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
    [doStart]
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

      // Upload chunks to backend
      for (let i = 0; i < chunks.length; i++) {
        try {
          await uploadBoardroomChunk(sid, i, chunks[i], "webm");
        } catch (e) {
          console.warn(`Chunk ${i} upload failed:`, e);
        }
      }

      // Trigger finalization
      await finalizeBoardroomSession(sid);
      setBoardroomSession((prev) => prev ? { ...prev, status: "processing" } : null);

      // Poll for status until transcribed
      const poll = async () => {
        let attempts = 0;
        while (attempts < 60) {
          await new Promise((r) => setTimeout(r, 3000));
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
  };
}
