import { useState, useRef, useCallback, useEffect } from "react";
import { voiceWsUrl } from "../services/backend";
import type { ConversationState, OrbState } from "../components/Conversation/ChatState";

const LISTENING_THRESHOLD = 15;
const MIC_CHECK_MS = 120;

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

export interface UseVoiceConnectionReturn {
  state: ConversationState;
  userAnalyser: AnalyserNode | null;
  assistantAnalyser: AnalyserNode | null;
  contextValue: string;
  setContextValue: (v: string) => void;
  applyContext: () => void;
  clearMemory: () => void;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  connecting: boolean;
  /** For demo: set user orb state */
  setUserOrbState: (s: OrbState) => void;
  /** For demo: set assistant orb state */
  setAssistantOrbState: (s: OrbState) => void;
  /** For demo: trigger interrupt */
  triggerInterrupt: () => void;
}

export function useVoiceConnection(): UseVoiceConnectionReturn {
  const [state, setState] = useState<ConversationState>({
    userOrb: "disconnected",
    assistantOrb: "disconnected",
    isConnected: false,
    interruptedAt: 0,
    showIntroTip: false,
  });
  const [contextValue, setContextValue] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [userAnalyser, setUserAnalyser] = useState<AnalyserNode | null>(null);
  const [assistantAnalyser, setAssistantAnalyser] = useState<AnalyserNode | null>(null);

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
  const micCheckRef = useRef<number>(0);

  const setUserOrbState = useCallback((s: OrbState) => {
    setState((prev) => ({ ...prev, userOrb: s }));
  }, []);
  const setAssistantOrbState = useCallback((s: OrbState) => {
    setState((prev) => ({ ...prev, assistantOrb: s }));
  }, []);
  const triggerInterrupt = useCallback(() => {
    setState((prev) => ({ ...prev, interruptedAt: Date.now(), assistantOrb: "idle" }));
  }, []);

  const pumpPlayback = useCallback(() => {
    const ctx = playbackCtxRef.current;
    const gain = playbackGainRef.current;
    if (!ctx || !gain || playQueueRef.current.length === 0) {
      playingRef.current = false;
      setState((prev) => ({ ...prev, assistantOrb: "idle" }));
      return;
    }
    playingRef.current = true;
    const item = playQueueRef.current.shift()!;
    const buf = ctx.createBuffer(1, item.f32.length, ctx.sampleRate);
    buf.copyToChannel(item.f32, 0);
    const analyser = playbackAnalyserRef.current!;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = item.rate;
    src.connect(analyser);
    currentSourceRef.current = src;
    src.onended = () => pumpPlayback();
    src.start();
  }, []);

  const enqueuePlayback = useCallback(
    (pcmF32: Float32Array, sr: number, rate: number = 1) => {
      const ctx = playbackCtxRef.current;
      if (!ctx) return;
      const targetSr = ctx.sampleRate;
      const f32 = resampleLinear(pcmF32, sr, targetSr);
      playQueueRef.current.push({ f32, rate });
      if (!playingRef.current) {
        setState((prev) => ({ ...prev, assistantOrb: "speaking" }));
        pumpPlayback();
      }
    },
    [pumpPlayback]
  );

  const smoothStop = useCallback(() => {
    const ctx = playbackCtxRef.current;
    const gain = playbackGainRef.current;
    if (!ctx || !gain) return;
    const now = ctx.currentTime;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setValueAtTime(gain.gain.value, now);
    gain.gain.linearRampToValueAtTime(0, now + 0.06);
    const src = currentSourceRef.current;
    if (src) src.stop(now + 0.07);
    setTimeout(() => {
      if (playbackCtxRef.current && playbackGainRef.current) {
        playbackGainRef.current.gain.setValueAtTime(1, playbackCtxRef.current.currentTime);
      }
      pumpPlayback();
    }, 80);
  }, [pumpPlayback]);

  const connect = useCallback(async () => {
    setConnecting(true);
    const ws = new WebSocket(voiceWsUrl());
    wsRef.current = ws;

    ws.onopen = async () => {
      try {
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioCtxRef.current = ctx;
        const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
        await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        micStreamRef.current = stream;
        const src = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.25;
        src.connect(analyser);
        setUserAnalyser(analyser);

        const worklet = new AudioWorkletNode(ctx, "framer16k");
        workletRef.current = worklet;
        worklet.port.onmessage = (ev: MessageEvent) => {
          if (wsRef.current?.readyState !== 1) return;
          const u8 = new Uint8Array(ev.data);
          const b64 = btoa(String.fromCharCode(...u8));
          wsRef.current?.send(JSON.stringify({ type: "audio_frame", ts: performance.now() / 1000, pcm16_b64: b64 }));
        };
        src.connect(worklet);

        const playCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        playbackCtxRef.current = playCtx;
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
        setState((prev) => ({
          ...prev,
          isConnected: true,
          userOrb: "idle",
          assistantOrb: "idle",
          showIntroTip: true,
        }));
        ws.send(JSON.stringify({ type: "set_context", system_prompt: contextValue, clear_memory: false }));
      } catch (e) {
        console.error(e);
        setState((prev) => ({ ...prev, userOrb: "disconnected", assistantOrb: "disconnected" }));
      } finally {
        setConnecting(false);
      }
    };

    ws.onmessage = async (ev: MessageEvent) => {
      const msg = JSON.parse(ev.data as string);
      if (msg.type === "event" && (msg.event === "BARGE_IN" || msg.event === "USER_SPEECH_START")) {
        playQueueRef.current = [];
        smoothStop();
        setState((prev) => ({ ...prev, interruptedAt: Date.now(), assistantOrb: "idle", showIntroTip: false }));
        return;
      }
      if (msg.type === "event" && (msg.event === "SPEAKING" || msg.event === "BACK_TO_LISTENING")) {
        setState((prev) => (prev.showIntroTip ? { ...prev, showIntroTip: false } : prev));
      }
      if (msg.type === "asr_final" && msg.text) {
        setState((prev) => ({ ...prev, userOrb: "idle", assistantOrb: "thinking", showIntroTip: false }));
        return;
      }
      if (msg.type === "assistant_text_partial" || msg.type === "assistant_text") {
        setState((prev) => ({ ...prev, assistantOrb: "speaking", showIntroTip: false }));
        return;
      }
      if (msg.type === "audio_out") {
        setState((prev) => (prev.showIntroTip ? { ...prev, showIntroTip: false } : prev));
        const bytes = b64ToBytes(msg.pcm16_b64);
        const f32 = pcm16ToFloat32(bytes);
        enqueuePlayback(f32, msg.sample_rate || 24000, msg.playback_rate || 1);
        return;
      }
    };

    ws.onclose = () => {
      setState((prev) => ({ ...prev, isConnected: false, userOrb: "disconnected", assistantOrb: "disconnected", showIntroTip: false }));
      setUserAnalyser(null);
      setAssistantAnalyser(null);
      setConnecting(false);
    };
    ws.onerror = () => setConnecting(false);
  }, [contextValue, enqueuePlayback, smoothStop]);

  const disconnect = useCallback(async () => {
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
    playQueueRef.current = [];
    playingRef.current = false;
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
    setState((prev) => ({
      ...prev,
      isConnected: false,
      userOrb: "disconnected",
      assistantOrb: "disconnected",
      showIntroTip: false,
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
      setState((prev) => {
        if (prev.assistantOrb === "speaking" || prev.assistantOrb === "thinking") return prev;
        return { ...prev, userOrb: avg > LISTENING_THRESHOLD ? "listening" : "idle" };
      });
    }, MIC_CHECK_MS);
    return () => clearInterval(interval);
  }, [state.isConnected, userAnalyser]);

  useEffect(() => {
    if (!state.isConnected || !state.showIntroTip) return;
    const t = setTimeout(() => {
      setState((prev) => (prev.showIntroTip ? { ...prev, showIntroTip: false } : prev));
    }, 8000);
    return () => clearTimeout(t);
  }, [state.isConnected, state.showIntroTip]);

  const applyContext = useCallback(() => {
    if (wsRef.current?.readyState !== 1) return;
    wsRef.current.send(JSON.stringify({ type: "set_context", system_prompt: contextValue, clear_memory: false }));
  }, [contextValue]);

  const clearMemory = useCallback(() => {
    if (wsRef.current?.readyState !== 1) return;
    wsRef.current.send(JSON.stringify({ type: "set_context", system_prompt: contextValue, clear_memory: true }));
  }, [contextValue]);

  return {
    state,
    userAnalyser,
    assistantAnalyser,
    contextValue,
    setContextValue,
    applyContext,
    clearMemory,
    connect,
    disconnect,
    connecting,
    setUserOrbState,
    setAssistantOrbState,
    triggerInterrupt,
  };
}
