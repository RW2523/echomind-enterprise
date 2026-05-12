import { useCallback, useRef, useState } from "react";
import { voiceWsUrl } from "../services/backend";
import type { AppSettings } from "../types";
import { PersonaType } from "../types";
import { b64ToBytes, fadeBufferEdges, pcm16ToFloat32, resampleLinear } from "../utils/voicePlayback";

function speakBrowser(text: string): Promise<void> {
  return new Promise((resolve, reject) => {
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1;
      u.onend = () => resolve();
      u.onerror = () => reject(new Error("Speech synthesis failed"));
      window.speechSynthesis.speak(u);
    } catch (e) {
      reject(e);
    }
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * One-shot Piper TTS via voice WebSocket (`speak_text_only`). No mic frames are sent — no auto-replies.
 * Falls back to `speechSynthesis` if the voice service is unavailable.
 */
async function speakThroughVoiceWs(text: string, settings: AppSettings): Promise<void> {
  const ws = new WebSocket(voiceWsUrl());

  await new Promise<void>((resolve, reject) => {
    const t = window.setTimeout(() => reject(new Error("Voice WebSocket open timeout")), 12000);
    ws.addEventListener(
      "open",
      () => {
        window.clearTimeout(t);
        resolve();
      },
      { once: true }
    );
    ws.addEventListener(
      "error",
      () => {
        window.clearTimeout(t);
        reject(new Error("Voice WebSocket error"));
      },
      { once: true }
    );
  });

  const playCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({
    sampleRate: 24000,
  });
  if (playCtx.state === "suspended") await playCtx.resume();

  const playQueue: { f32: Float32Array; rate: number }[] = [];
  let playing = false;

  const pump = (): void => {
    if (playQueue.length === 0) {
      playing = false;
      return;
    }
    playing = true;
    const item = playQueue.shift()!;
    const buf = playCtx.createBuffer(1, item.f32.length, playCtx.sampleRate);
    buf.copyToChannel(item.f32, 0);
    const src = playCtx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = item.rate;
    src.connect(playCtx.destination);
    src.onended = () => pump();
    src.start();
  };

  const enqueue = (pcmF32: Float32Array, sr: number, rate: number) => {
    const targetSr = playCtx.sampleRate;
    const f32 = resampleLinear(pcmF32, sr, targetSr);
    fadeBufferEdges(f32, targetSr, 3);
    playQueue.push({ f32, rate });
    if (!playing) pump();
  };

  const systemPrompt =
    "You are EchoMind. This connection is TTS-only: speak only when the client sends speak_text_only. " +
    "Do not respond to microphone audio.";
  ws.send(
    JSON.stringify({
      type: "set_context",
      system_prompt: systemPrompt,
      clear_memory: true,
      listen_only: true,
      use_knowledge_base: false,
      skip_intro: true,
      piper_voice: settings.voiceName ?? undefined,
      persona: settings.persona ?? PersonaType.FINANCIAL,
      context_window: settings.contextWindow ?? "all",
    })
  );

  await sleep(150);

  await new Promise<void>((resolve, reject) => {
    ws.onmessage = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
        if (msg.type === "audio_out" && typeof msg.pcm16_b64 === "string") {
          const bytes = b64ToBytes(msg.pcm16_b64);
          const f32 = pcm16ToFloat32(bytes);
          enqueue(f32, (msg.sample_rate as number) || 24000, (msg.playback_rate as number) || 1);
          return;
        }
        if (msg.type === "event" && msg.event === "BACK_TO_LISTENING") {
          resolve();
          return;
        }
        if (msg.type === "error") {
          reject(new Error(String((msg as { message?: string }).message ?? "voice error")));
        }
      } catch {
        /* ignore */
      }
    };
    ws.send(JSON.stringify({ type: "speak_text_only", text }));
  });

  await sleep(400);
  while (playQueue.length > 0 || playing) {
    await sleep(80);
  }
  await playCtx.close().catch(() => {});
  ws.close();
}

export function useAssistantSpeak() {
  const busyRef = useRef(false);
  const [speaking, setSpeaking] = useState(false);

  const speakApprovedText = useCallback(async (text: string, settings: AppSettings) => {
    const trimmed = (text || "").trim();
    if (!trimmed || busyRef.current) return;
    busyRef.current = true;
    setSpeaking(true);
    try {
      try {
        await speakThroughVoiceWs(trimmed, settings);
      } catch (e) {
        console.warn("[Assistant] Voice TTS unavailable, using browser speech:", e);
        await speakBrowser(trimmed);
      }
    } finally {
      busyRef.current = false;
      setSpeaking(false);
    }
  }, []);

  return { speakApprovedText, speaking };
}
