import { useState, useRef, useCallback, useEffect } from "react";
import { transcribeWsUrl, getTranscriptTags, updateTranscript } from "../services/backend";

/** Fallback sample rate if backend does not send one in the ready message. */
const FALLBACK_SAMPLE_RATE = 16000;
const OPEN_TIMEOUT_MS = 15000;
const READY_TIMEOUT_MS = 300000;
const HEARTBEAT_INTERVAL_MS = 25000;  // Keep connection alive when tab backgrounded (audio stops)
const RECONNECT_DELAY_MS = 800;

function floatTo16BitPCM(input: Float32Array) {
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
}

export function useLiveTranscription(defaultName: () => string): UseLiveTranscriptionReturn {
  const [fullTranscript, setFullTranscript] = useState("");
  const [partial, setPartial] = useState("");
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
      setFullTranscript("");
      setPartial("");
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
            setPartial("");
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
      ws.onclose = () => {
        const wasListening = listeningRef.current;
        const shouldReconnect = !userInitiatedCloseRef.current && wasListening;
        stopMic(false);
        setWsStatus((s) => (s === "error" ? s : "idle"));
        if (shouldReconnect) {
          setTimeout(() => doStart(sessionNameRef.current, sessionLocationRef.current, true), RECONNECT_DELAY_MS);
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
              resolve(msg.sample_rate ?? FALLBACK_SAMPLE_RATE);
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

      let requestedSampleRate: number;
      try {
        requestedSampleRate = await readyPromise;
      } catch (e) {
        handleError((e as Error)?.message || "STT not ready");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recRef.current = stream;

      // Create the AudioContext at the server-requested rate. Browsers support any rate
      // from 8 kHz to 96 kHz, but we read back the ACTUAL rate in case the browser clamped
      // it (some embedded/mobile environments), so the backend can resample correctly.
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: requestedSampleRate,
      });
      audioCtxRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      // Use the ACTUAL context rate – it may differ from the requested one.
      const actualSampleRate = audioCtx.sampleRate;

      ws.send(
        JSON.stringify({
          type: "start",
          auto_store: true,
          sample_rate: actualSampleRate,   // tell backend the TRUE capture rate
          language: "en",
          name: name || undefined,
          location: location || undefined,
        })
      );

      const src = audioCtx.createMediaStreamSource(stream);
      // Buffer size 2048 → 128 ms at 16 kHz – half the old 256 ms, giving the model
      // more granular chunks and less chance of silently dropping a long buffer.
      const processor = audioCtx.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || micMutedRef.current) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(input);
        // Send raw binary (Int16 PCM) – the backend binary path is faster and
        // avoids the base64-encode overhead inside the audio callback thread.
        wsRef.current.send(pcm16.buffer);
      };

      src.connect(processor);
      processor.connect(audioCtx.destination);
      setListening(true);
      heartbeatIntervalRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "ping" }));
        }
      }, HEARTBEAT_INTERVAL_MS);
    },
    [stopMic]
  );

  const clearAndReset = useCallback(() => {
    userInitiatedCloseRef.current = true;
    stopMic(true);
    setMicMuted(false);
    setFullTranscript("");
    setPartial("");
    setSessionName("");
    setSessionLocation("");
    setSessionStartedAt(null);
    setCustomTags([]);
    setNewTagInput("");
    transcriptForTagsRef.current = "";
    lastStoredTranscriptIdRef.current = null;
    pendingTagsRef.current = null;
    setWsError(null);
  }, [stopMic]);

  const startSession = useCallback(
    async (name: string, location: string) => {
      setMicMuted(false);
      setSessionName(name);
      setSessionLocation(location);
      setSessionStartedAt(new Date());
      setCustomTags([]);
      transcriptForTagsRef.current = "";
      lastStoredTranscriptIdRef.current = null;
      pendingTagsRef.current = null;
      setShowStartModal(false);
      await doStart(name, location);
    },
    [doStart]
  );

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
  };
}
