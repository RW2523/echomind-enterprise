import { useState, useRef, useCallback, useEffect } from "react";
import { voiceWsUrl } from "../services/backend";
import type { ConversationState } from "../components/Conversation/ChatState";
import { PersonaType } from "../types";
import type { AppSettings } from "../types";

const PERSONA_VOICE_PROMPTS: Record<string, string> = {
  [PersonaType.TEACHER]: (
    "You are EchoMind, an expert Teacher and Professor. " +
    "Explain topics clearly with analogies, step-by-step breakdowns, and an encouraging academic tone. " +
    "Adapt complexity to the learner's level. Keep voice responses concise and well-paced. " +
    "GUARDRAIL: Educate and inform. Refuse harmful or unethical requests politely."
  ),
  [PersonaType.FINANCIAL]: (
    "You are EchoMind, a DoD financial management advisor. " +
    "Be concise, precise, and cite sections when answering from documents. " +
    "For procedures give numbered steps; for comparisons use clear bullet points. " +
    "Never invent section numbers or page references—only cite what appears in context. " +
    "GUARDRAIL: Only answer financial management, DoD FMR, regulations, and compliance questions. " +
    "For anything else: 'I'm a financial advisor. I can only help with DoD FMR and regulatory topics.'"
  ),
  [PersonaType.FUNNY]: (
    "You are EchoMind, a warm, witty, and calming voice assistant. " +
    "Blend light humor with genuine helpfulness. Keep things positive and stress-free. " +
    "Give real, accurate answers—humor enhances but never replaces substance. " +
    "Keep voice responses concise and delightful. " +
    "GUARDRAIL: Always be kind and appropriate. Refuse harmful requests warmly."
  ),
  [PersonaType.LAWYER]: (
    "You are EchoMind, an experienced legal advisor. " +
    "Provide structured legal reasoning—cite laws, regulations, and document clauses precisely. " +
    "Use clear, professional language. Keep voice responses focused and actionable. " +
    "Always mention: 'This is informational, not formal legal advice.' " +
    "GUARDRAIL: Focus on legal analysis and compliance. " +
    "For unrelated topics: 'That falls outside my legal practice area.'"
  ),
  [PersonaType.AI_EXPERT]: (
    "You are EchoMind, a senior AI Expert and Software Engineering Manager. " +
    "Give direct, well-reasoned technical recommendations with clear trade-offs. " +
    "Think like both an engineer and a leader. Keep voice responses crisp and actionable. " +
    "GUARDRAIL: Focus on AI, software engineering, architecture, and tech leadership. " +
    "For unrelated topics: 'That's outside my technical domain. I'm here for AI and software topics.'"
  ),
  [PersonaType.GENERAL]: (
    "You are EchoMind, a friendly, helpful all-purpose voice assistant. " +
    "Answer any question clearly and directly, scaling depth to what's asked. " +
    "Draw on your broad knowledge and the user's documents and transcripts; never fabricate facts. " +
    "Keep voice responses concise and conversational. " +
    "GUARDRAIL: Be helpful, honest, and respectful. Refuse harmful requests politely."
  ),
  [PersonaType.ECHOMIND]: (
    "You are EchoMind, the voice guide to the EchoMind application itself. " +
    "Explain EchoMind's features and how it runs entirely on the NVIDIA DGX Spark—private, offline, and " +
    "GPU-accelerated, with no cloud. Main features: Knowledge Chat (RAG over your documents and transcripts), " +
    "Real-Time Transcription, Voice Conversation, and Boardroom. " +
    "Explain clearly and enthusiastically in plain language; keep voice responses concise. Be honest about anything you don't know. " +
    "GUARDRAIL: Focus on EchoMind and the DGX Spark. For unrelated topics: 'I'm the EchoMind guide—I focus on how this app and the DGX Spark work.'"
  ),
};

function buildPersonaVoicePrompt(persona: string | undefined, voiceContext: string): string {
  const basePersonaPrompt = persona && PERSONA_VOICE_PROMPTS[persona]
    ? PERSONA_VOICE_PROMPTS[persona]
    : "You are EchoMind, a helpful voice assistant. Be concise, helpful, and conversational.";
  const ctx = (voiceContext ?? "").trim();
  return ctx ? `${basePersonaPrompt}\n\n${ctx}` : basePersonaPrompt;
}

const LISTENING_THRESHOLD = 18;
const MIC_CHECK_MS = 150;
/** Consecutive samples above/below threshold before changing state (stops orb flicker) */
const MIC_HYSTERESIS = 4;

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function pcm16ToFloat32(pcmBytes: Uint8Array): Float32Array {
  const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
  const out = new Float32Array(pcmBytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

function resampleLinear(input: Float32Array, srcSr: number, dstSr: number): Float32Array {
  if (srcSr === dstSr) return input;
  const ratio = dstSr / srcSr;
  const n = Math.floor(input.length * ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i / ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const w = x - i0;
    out[i] = (1 - w) * input[i0] + w * input[i1];
  }
  return out;
}

/** Apply short fade-in/fade-out to reduce clicks at chunk boundaries. Modifies in place. */
function fadeBufferEdges(f32: Float32Array, sampleRate: number, fadeMs: number = 3): void {
  const n = Math.min(Math.floor((sampleRate * fadeMs) / 1000), Math.floor(f32.length / 2));
  if (n <= 0) return;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / (n + 1);
    f32[i] *= t;
    f32[f32.length - 1 - i] *= t;
  }
}

const WORKLET_CODE = `
  class Framer16k extends AudioWorkletProcessor {
    constructor() {
      super();
      this.ratio = 16000 / sampleRate;
      this.acc = 0;
      this.buf = [];
      this.frameSamples = 320;
    }
    _pushResampled(input) {
      let out = [];
      for (let i = 0; i < input.length; i++) {
        this.acc += this.ratio;
        while (this.acc >= 1.0) { out.push(input[i]); this.acc -= 1.0; }
      }
      return out;
    }
    process(inputs) {
      const input = inputs[0];
      if (!input || !input[0]) return true;
      const res = this._pushResampled(input[0]);
      for (let i = 0; i < res.length; i++) this.buf.push(res[i]);
      while (this.buf.length >= this.frameSamples) {
        const frame = this.buf.splice(0, this.frameSamples);
        const pcm16 = new Int16Array(this.frameSamples);
        for (let i = 0; i < this.frameSamples; i++) {
          const v = Math.max(-1, Math.min(1, frame[i]));
          pcm16[i] = (v * 32767) | 0;
        }
        this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
      }
      return true;
    }
  }
  registerProcessor('framer16k', Framer16k);
`;

export interface VoiceMessage {
  role: "user" | "assistant";
  text: string;
}

export interface UseVoiceConnectionReturn {
  state: ConversationState;
  userAnalyser: AnalyserNode | null;
  assistantAnalyser: AnalyserNode | null;
  /** Live transcript: what you said and what the assistant replied */
  voiceMessages: VoiceMessage[];
  /** Current assistant reply while streaming (partial text) */
  pendingAssistantText: string;
  applyContext: () => void;
  clearMemory: () => void;
  /** Continuous listening mode: only respond after wake word. */
  listenOnly: boolean;
  setListenOnly: (on: boolean) => void;
  /** Accumulated transcript while in listen-only (live updating until wake word). */
  listenBufferText: string;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  connecting: boolean;
  /** When true, mic is muted so the assistant can finish without interruption */
  micMuted: boolean;
  setMicMuted: (muted: boolean) => void;
  /** Error starting mic or connection (e.g. permission denied); cleared when user retries or disconnects */
  connectionError: string | null;
}

export interface UseVoiceConnectionOptions {
  settings?: AppSettings | null;
}

export function useVoiceConnection(options?: UseVoiceConnectionOptions): UseVoiceConnectionReturn {
  const settings = options?.settings;
  const [state, setState] = useState<ConversationState>({
    userOrb: "disconnected",
    assistantOrb: "disconnected",
    isConnected: false,
    interruptedAt: 0,
    listenOnly: false,
    backchannelText: "",
    partialTranscript: "",
  });
  const [connecting, setConnecting] = useState(false);
  const [userAnalyser, setUserAnalyser] = useState<AnalyserNode | null>(null);
  const [assistantAnalyser, setAssistantAnalyser] = useState<AnalyserNode | null>(null);
  const [voiceMessages, setVoiceMessages] = useState<VoiceMessage[]>([]);
  const [pendingAssistantText, setPendingAssistantText] = useState("");
  const [listenBufferText, setListenBufferText] = useState("");
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playbackGainRef = useRef<GainNode | null>(null);
  const playbackAnalyserRef = useRef<AnalyserNode | null>(null);
  const playQueueRef = useRef<{ f32: Float32Array; rate: number }[]>([]);
  const playingRef = useRef(false);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  // Gapless scheduled playback: each TTS phrase is scheduled back-to-back on the audio clock
  // (nextStartRef) instead of chained via onended, so there are no clicks/gaps between phrases.
  const nextStartRef = useRef(0);
  const scheduledSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const backchannelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const micAboveCountRef = useRef(0);
  const micBelowCountRef = useRef(0);
  const [micMuted, setMicMuted] = useState(false);
  const micMutedRef = useRef(false);
  micMutedRef.current = micMuted;
  const listenOnlyRef = useRef(false);
  listenOnlyRef.current = state.listenOnly;
  /** True only after WS open + mic/audio graph is ready (used to show connect errors vs. silent stop). */
  const voiceSessionReadyRef = useRef(false);
  /** User clicked Stop (or left Voice) while still connecting — do not treat close as a failure. */
  const userCancelledConnectRef = useRef(false);

  /** Stop ALL playback immediately and cleanly (used for barge-in / interruption).
   *  Kills every scheduled source so there is never overlapping/garbled audio, then
   *  restores gain so the next response plays normally. */
  const stopAllPlayback = useCallback((fade: boolean = true) => {
    const ctx = playbackCtxRef.current;
    const gain = playbackGainRef.current;
    if (ctx && gain && fade) {
      const now = ctx.currentTime;
      try {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        gain.gain.linearRampToValueAtTime(0.0001, now + 0.05);
      } catch (_) {}
    }
    // Detach onended (so a stopped source can't trigger more playback) and stop each source.
    scheduledSourcesRef.current.forEach((s) => {
      try { s.onended = null; s.stop(); } catch (_) {}
    });
    scheduledSourcesRef.current.clear();
    playQueueRef.current = [];
    currentSourceRef.current = null;
    nextStartRef.current = 0;
    playingRef.current = false;
    // Restore gain just after the fade so subsequent audio is audible.
    if (ctx && gain) {
      setTimeout(() => {
        try {
          gain.gain.cancelScheduledValues(ctx.currentTime);
          gain.gain.setValueAtTime(1, ctx.currentTime);
        } catch (_) {}
      }, 60);
    }
    setState((prev) => ({ ...prev, assistantOrb: "idle" }));
  }, []);

  const enqueuePlayback = useCallback(
    (pcmF32: Float32Array, sr: number, rate: number = 1) => {
      const ctx = playbackCtxRef.current;
      const analyser = playbackAnalyserRef.current;
      if (!ctx || !analyser) return;
      const targetSr = ctx.sampleRate;
      const f32 = resampleLinear(pcmF32, sr, targetSr);
      fadeBufferEdges(f32, targetSr, 3);
      const buf = ctx.createBuffer(1, f32.length, targetSr);
      buf.copyToChannel(f32 as Float32Array, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate;
      src.connect(analyser);

      // Schedule this phrase right after whatever is already queued on the audio clock.
      const now = ctx.currentTime;
      const startAt = Math.max(now + 0.02, nextStartRef.current);
      const playDur = buf.duration / (rate || 1);
      try {
        src.start(startAt);
      } catch (_) {
        return;
      }
      nextStartRef.current = startAt + playDur;
      currentSourceRef.current = src;
      scheduledSourcesRef.current.add(src);

      if (!playingRef.current) {
        playingRef.current = true;
        setState((prev) => ({ ...prev, assistantOrb: "speaking" }));
      }
      src.onended = () => {
        scheduledSourcesRef.current.delete(src);
        if (scheduledSourcesRef.current.size === 0) {
          // Nothing left scheduled -> assistant finished this turn.
          playingRef.current = false;
          nextStartRef.current = 0;
          currentSourceRef.current = null;
          setState((prev) => ({ ...prev, assistantOrb: "idle" }));
        }
      };
    },
    []
  );

  const connect = useCallback(async () => {
    setConnecting(true);
    setConnectionError(null);
    setMicMuted(false);

    if (!navigator.mediaDevices?.getUserMedia) {
      setConnectionError("Microphone not available. Use HTTPS and a supported browser.");
      setConnecting(false);
      return;
    }

    let stream: MediaStream;
    try {
      // Echo cancellation keeps the bot's own TTS out of the mic (prevents false barge-ins and
      // self-triggered backchannels); noise suppression + AGC clean the signal for better STT.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e: any) {
      const msg = e?.name === "NotAllowedError" || e?.message?.toLowerCase().includes("permission")
        ? "Microphone access denied. Allow mic in browser settings and try again."
        : e?.message || "Could not access microphone.";
      setConnectionError(msg);
      setConnecting(false);
      return;
    }

    micStreamRef.current = stream;
    voiceSessionReadyRef.current = false;
    userCancelledConnectRef.current = false;
    const ws = new WebSocket(voiceWsUrl());
    wsRef.current = ws;

    ws.onopen = async () => {
      try {
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioCtxRef.current = ctx;
        if (ctx.state === "suspended") {
          await ctx.resume();
        }
        const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
        await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
        const src = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.25;
        src.connect(analyser);
        setUserAnalyser(analyser);

        const worklet = new AudioWorkletNode(ctx, "framer16k");
        workletRef.current = worklet;
        worklet.port.onmessage = (ev: MessageEvent) => {
          if (wsRef.current?.readyState !== 1 || micMutedRef.current) return;
          const u8 = new Uint8Array(ev.data);
          const b64 = btoa(String.fromCharCode(...u8));
          wsRef.current?.send(JSON.stringify({ type: "audio_frame", ts: performance.now() / 1000, pcm16_b64: b64 }));
        };
        src.connect(worklet);

        const playCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        playbackCtxRef.current = playCtx;
        if (playCtx.state === "suspended") {
          await playCtx.resume();
        }
        const gain = playCtx.createGain();
        gain.gain.value = 1;
        playbackGainRef.current = gain;
        const pAnalyser = playCtx.createAnalyser();
        pAnalyser.fftSize = 2048;
        pAnalyser.smoothingTimeConstant = 0.25;
        playbackAnalyserRef.current = pAnalyser;
        pAnalyser.connect(gain);
        gain.connect(playCtx.destination);

        setAssistantAnalyser(pAnalyser);
        voiceSessionReadyRef.current = true;
        setState((prev) => ({
          ...prev,
          isConnected: true,
          userOrb: "idle",
          assistantOrb: "idle",
        }));
        micAboveCountRef.current = 0;
        micBelowCountRef.current = 0;
        const botName = (settings?.voiceBotName ?? "").trim();
        const userName = (settings?.voiceUserName ?? "").trim();
        let systemPrompt = buildPersonaVoicePrompt(settings?.persona, settings?.voiceContext ?? "");
        if (botName) {
          const userNameHint = userName ? `The user's name is ${userName}; use it when it fits naturally. ` : "";
          systemPrompt = (
            `You are ${botName}. Talk in a natural, conversational way—like a friendly voice assistant. ` +
            `When the user says your name or speaks to you, respond naturally. ` +
            `If they say "stop", pause speaking; if they say "start" or your name, continue. ` +
            `${userNameHint}` +
            systemPrompt
          );
        }
        ws.send(JSON.stringify({
          type: "set_context",
          system_prompt: systemPrompt,
          clear_memory: false,
          listen_only: false,
          piper_voice: settings?.voiceName ?? undefined,
          use_knowledge_base: true,
          persona: settings?.persona ?? undefined,
          context_window: settings?.contextWindow ?? undefined,
          voice_bot_name: botName || undefined,
          voice_user_name: userName || undefined,
        }));
      } catch (e: any) {
        console.error(e);
        const msg = e?.message || "Could not start voice. Try again.";
        setConnectionError(msg);
        if (micStreamRef.current) {
          micStreamRef.current.getTracks().forEach((t) => t.stop());
          micStreamRef.current = null;
        }
        setState((prev) => ({ ...prev, userOrb: "disconnected", assistantOrb: "disconnected" }));
      } finally {
        setConnecting(false);
      }
    };

    ws.onmessage = async (ev: MessageEvent) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data as string) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg.type === "event" && (msg.event === "BARGE_IN" || msg.event === "USER_SPEECH_START")) {
        stopAllPlayback(true);
        setState((prev) => ({ ...prev, interruptedAt: Date.now(), assistantOrb: "idle" }));
        return;
      }
      if (msg.type === "event" && msg.event === "SPEAKING") {
        setState((prev) => ({ ...prev, assistantOrb: "speaking" }));
        return;
      }
      if (msg.type === "event" && msg.event === "FILLER_SPEAKING") {
        // Lead phrase playing: use softer "thinking" orb, not full speaking bounce
        setState((prev) => ({ ...prev, assistantOrb: "filler" }));
        return;
      }
      if (msg.type === "event" && msg.event === "BACKCHANNEL") {
        // Brief backchannel ("Mm-hmm", "I see") during long user speech
        const bcText = typeof msg.text === "string" ? msg.text : "";
        setState((prev) => ({ ...prev, backchannelText: bcText }));
        // Auto-clear after 2.5 s — cancel any prior timer so an older one can't wipe a newer
        // backchannel early. (L34)
        if (backchannelTimerRef.current) clearTimeout(backchannelTimerRef.current);
        backchannelTimerRef.current = setTimeout(() => {
          backchannelTimerRef.current = null;
          setState((prev) => ({ ...prev, backchannelText: "" }));
        }, 2500);
        return;
      }
      if (msg.type === "partial_transcript" && typeof msg.text === "string") {
        setState((prev) => ({ ...prev, partialTranscript: msg.text as string }));
        return;
      }
      if (msg.type === "event" && msg.event === "BACK_TO_LISTENING") {
        setState((prev) => ({ ...prev, assistantOrb: "idle", backchannelText: "", partialTranscript: "" }));
        return;
      }
      if (msg.type === "memory_event" && msg.event === "listening_mode_on") {
        listenOnlyRef.current = true;
        setListenBufferText("");
        setState((prev) => ({ ...prev, listenOnly: true }));
        return;
      }
      if (msg.type === "memory_event" && msg.event === "listening_mode_off") {
        listenOnlyRef.current = false;
        setListenBufferText("");
        setState((prev) => ({ ...prev, listenOnly: false }));
        return;
      }
      if (msg.type === "listen_buffer" && typeof msg.text === "string") {
        setListenBufferText(String(msg.text).trim());
        return;
      }
      if (msg.type === "asr_final" && msg.text) {
        const userText = String(msg.text).trim();
        if (userText && !listenOnlyRef.current) {
          setVoiceMessages((prev) => [...prev, { role: "user", text: userText }]);
        }
        setState((prev) => ({ ...prev, userOrb: "idle", assistantOrb: listenOnlyRef.current ? "idle" : "thinking" }));
        return;
      }
      if (msg.type === "assistant_text_partial") {
        const text = typeof msg.text === "string" ? msg.text : "";
        setPendingAssistantText(text);
        setState((prev) => ({ ...prev, assistantOrb: "speaking" }));
        return;
      }
      if (msg.type === "assistant_text") {
        const text = typeof msg.text === "string" ? String(msg.text).trim() : "";
        if (text) {
          setVoiceMessages((prev) => [...prev, { role: "assistant", text }]);
        }
        setPendingAssistantText("");
        setState((prev) => ({ ...prev, assistantOrb: "speaking" }));
        return;
      }
      if (msg.type === "audio_out" && typeof msg.pcm16_b64 === "string") {
        try {
          const bytes = b64ToBytes(msg.pcm16_b64);
          const f32 = pcm16ToFloat32(bytes);
          enqueuePlayback(f32, (msg.sample_rate as number) || 24000, (msg.playback_rate as number) || 1);
        } catch (_) {
          // skip malformed audio chunk
        }
        return;
      }
    };

    const cleanupOnClose = () => {
      const sessionReady = voiceSessionReadyRef.current;
      voiceSessionReadyRef.current = false;
      scheduledSourcesRef.current.forEach((s) => { try { s.onended = null; s.stop(); } catch (_) {} });
      scheduledSourcesRef.current.clear();
      if (backchannelTimerRef.current) { clearTimeout(backchannelTimerRef.current); backchannelTimerRef.current = null; }
      playQueueRef.current = [];
      playingRef.current = false;
      nextStartRef.current = 0;
      if (workletRef.current) {
        try {
          workletRef.current.disconnect();
        } catch (_) {}
        workletRef.current = null;
      }
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
      if (playbackCtxRef.current) {
        playbackCtxRef.current.close().catch(() => {});
        playbackCtxRef.current = null;
      }
      playbackGainRef.current = null;
      playbackAnalyserRef.current = null;
      currentSourceRef.current = null;
      wsRef.current = null;
      setUserAnalyser(null);
      setAssistantAnalyser(null);
      setState((prev) => ({ ...prev, isConnected: false, userOrb: "disconnected", assistantOrb: "disconnected", listenOnly: false, backchannelText: "", partialTranscript: "" }));
      setVoiceMessages([]);
      setPendingAssistantText("");
      setListenBufferText("");
      listenOnlyRef.current = false;
      setMicMuted(false);
      if (!sessionReady && !userCancelledConnectRef.current) {
        setConnectionError((prev) => prev ?? "Could not connect to the voice service. Check that Docker `voice` is running and try again.");
      } else {
        setConnectionError(null);
      }
      userCancelledConnectRef.current = false;
      setConnecting(false);
    };

    ws.onclose = () => {
      cleanupOnClose();
    };
    ws.onerror = () => {
      setConnecting(false);
      setConnectionError((prev) => prev ?? "Voice WebSocket failed. Confirm the voice container is up and nginx proxies /voice/ws.");
    };
  }, [settings?.persona, settings?.voiceName, settings?.voiceBotName, settings?.voiceUserName, settings?.voiceContext, settings?.contextWindow, enqueuePlayback, stopAllPlayback]);

  const disconnect = useCallback(async () => {
    userCancelledConnectRef.current = true;
    setConnectionError(null);
    if (workletRef.current) {
      try {
        workletRef.current.disconnect();
      } catch (_) {}
      workletRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      await audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    scheduledSourcesRef.current.forEach((s) => { try { s.onended = null; s.stop(); } catch (_) {} });
    scheduledSourcesRef.current.clear();
    playQueueRef.current = [];
    playingRef.current = false;
    nextStartRef.current = 0;
    if (playbackCtxRef.current) {
      await playbackCtxRef.current.close();
      playbackCtxRef.current = null;
    }
    playbackGainRef.current = null;
    playbackAnalyserRef.current = null;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setUserAnalyser(null);
    setAssistantAnalyser(null);
    setMicMuted(false);
    setState((prev) => ({
      ...prev,
      isConnected: false,
      userOrb: "disconnected",
      assistantOrb: "disconnected",
      listenOnly: false,
    }));
  }, []);

  useEffect(() => {
    if (!state.isConnected || !userAnalyser) return;
    const data = new Uint8Array(userAnalyser.frequencyBinCount);
    const interval = setInterval(() => {
      userAnalyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const avg = sum / data.length;
      const above = avg > LISTENING_THRESHOLD;
      setState((prev) => {
        if (prev.assistantOrb === "speaking" || prev.assistantOrb === "thinking") return prev;
        if (above) {
          micBelowCountRef.current = 0;
          micAboveCountRef.current += 1;
          if (micAboveCountRef.current >= MIC_HYSTERESIS) return { ...prev, userOrb: "listening" as const };
        } else {
          micAboveCountRef.current = 0;
          micBelowCountRef.current += 1;
          if (micBelowCountRef.current >= MIC_HYSTERESIS) return { ...prev, userOrb: "idle" as const };
        }
        return prev;
      });
    }, MIC_CHECK_MS);
    return () => clearInterval(interval);
  }, [state.isConnected, userAnalyser]);

  const applyContext = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const botName = (settings?.voiceBotName ?? "").trim();
    const userName = (settings?.voiceUserName ?? "").trim();
    let sysPrompt = buildPersonaVoicePrompt(settings?.persona, settings?.voiceContext ?? "");
    if (botName) {
      const userNameHint = userName ? `The user's name is ${userName}; use it when it fits naturally. ` : "";
      sysPrompt = (
        `You are ${botName}. Talk in a natural, conversational way—like a friendly voice assistant. ` +
        `When the user says your name or speaks to you, respond naturally. ` +
        `If they say "stop", pause speaking; if they say "start" or your name, continue. ` +
        `${userNameHint}` +
        sysPrompt
      );
    }
    ws.send(JSON.stringify({
      type: "set_context",
      system_prompt: sysPrompt,
      clear_memory: false,
      listen_only: state.listenOnly,
      piper_voice: settings?.voiceName ?? undefined,
      use_knowledge_base: true,
      persona: settings?.persona ?? undefined,
      context_window: settings?.contextWindow ?? undefined,
      voice_bot_name: botName || undefined,
      voice_user_name: userName || undefined,
    }));
  }, [state.listenOnly, settings?.persona, settings?.voiceName, settings?.voiceContext, settings?.contextWindow, settings?.voiceBotName, settings?.voiceUserName]);

  const clearMemory = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "clear_memory" }));
    setVoiceMessages([]);
    setPendingAssistantText("");
  }, []);

  const setListenOnly = useCallback((on: boolean) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "set_context", listen_only: on }));
    }
    setState((prev) => ({ ...prev, listenOnly: on }));
  }, []);

  return {
    state,
    userAnalyser,
    assistantAnalyser,
    voiceMessages,
    pendingAssistantText,
    listenBufferText,
    applyContext,
    clearMemory,
    listenOnly: state.listenOnly,
    setListenOnly,
    connect,
    disconnect,
    connecting,
    micMuted,
    setMicMuted,
    connectionError,
  };
}
