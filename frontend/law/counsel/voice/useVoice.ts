/**
 * Voice mode for the chat (owner's request 2026-11-02): speak with the assistant hands-free, as in the original EchoMind
 * voice conversation, ported from frontend/hooks/useVoiceConnection.ts. The voice service (voice/app, Law mode) hears
 * the question, asks it as a turn of this chat conversation — answered only from the matter's documents, with checked
 * quotes, and saved with its sources — and speaks the checked paragraphs as they arrive. Speaking over it stops it.
 *
 * Audio: 20 ms frames of 16 kHz PCM16 from an AudioWorklet (/law-voice-framer.js), sent as `audio_frame`; replies are
 * 24 kHz PCM16 chunks scheduled gaplessly on the audio clock and faded out on barge-in. Nothing is recorded in the
 * browser; the header's recording status follows the open microphone.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { captureStarted } from '../../app/capture';

export type VoicePhase = 'connecting' | 'listening' | 'hearing' | 'thinking' | 'speaking' | 'closed' | 'error';

export interface VoiceView {
  phase: VoicePhase;
  partial: string;         // what is being heard right now
  heard: string;           // the last question, as transcribed
  saying: string;          // what the assistant is saying (checked paragraphs, or a card's summary)
  holding: string;         // "Let me check the documents." while the answer is found
  error: string | null;
}

export const INITIAL: VoiceView = { phase: 'connecting', partial: '', heard: '', saying: '', holding: '', error: null };

type Msg = Record<string, unknown>;
const str = (v: unknown) => (typeof v === 'string' ? v : '');

/** How the view follows the voice service's messages (pure, so it can be tested without audio). */
export function reduce(view: VoiceView, msg: Msg): VoiceView {
  if (view.phase === 'closed' || view.phase === 'error') return view;
  switch (msg.type) {
    case 'hello': case 'context_ack': return view.phase === 'connecting' ? { ...view, phase: 'listening' } : view;
    case 'partial_transcript': return { ...view, partial: str(msg.text), phase: view.phase === 'speaking' ? view.phase : 'hearing' };
    case 'asr_final': return { ...view, heard: str(msg.text), partial: '', saying: '', holding: '', phase: 'thinking' };
    case 'assistant_text_partial': case 'assistant_text': return { ...view, saying: str(msg.text) || view.saying };
    case 'turn_saved': return { ...view, heard: '', saying: '', holding: '' };       // the saved turn is in the thread now
    case 'error':
      return msg.fatal ? { ...view, phase: 'error', error: 'Voice stopped: the speech service had a problem. Start voice again in a moment.' } : view;
    case 'event':
      switch (msg.event) {
        case 'USER_SPEECH_START': return { ...view, phase: 'hearing' };
        case 'FILLER_SPEAKING': return { ...view, holding: str(msg.text) };
        case 'BACK_TO_LISTENING': return { ...view, phase: view.phase === 'speaking' ? 'speaking' : 'listening', partial: '' };
        default: return view;
      }
    default: return view;
  }
}

function b64ToBytes(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength >> 1);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

function toBase64(buf: ArrayBuffer): string {
  const u8 = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}

export function voiceUrl(matterId: string, conversationId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/voice/ws?matter=${encodeURIComponent(matterId)}&conversation=${encodeURIComponent(conversationId)}`;
}

const CLOSE_TEXT: Record<number, string> = {
  1008: 'Voice is not available here: sign in again, or ask an administrator whether voice is on.',
  4401: 'Your session ended, so voice stopped. Sign in again to continue.',
};

export interface Voice {
  view: VoiceView;
  muted: boolean;
  setMuted: (m: boolean) => void;
  stop: () => void;
  micAnalyser: AnalyserNode | null;
  speechAnalyser: AnalyserNode | null;
}

export function useVoice({ matterId, conversationId, onTurnSaved }: {
  matterId: string; conversationId: string; onTurnSaved: (turnId: string) => void;
}): Voice {
  const [view, setView] = useState<VoiceView>(INITIAL);
  const [muted, setMutedState] = useState(false);
  const [micAnalyser, setMicAnalyser] = useState<AnalyserNode | null>(null);
  const [speechAnalyser, setSpeechAnalyser] = useState<AnalyserNode | null>(null);
  const mutedRef = useRef(false);
  const savedRef = useRef(onTurnSaved);
  savedRef.current = onTurnSaved;
  const stopRef = useRef<() => void>(() => {});

  useEffect(() => {
    let alive = true;
    let ws: WebSocket | null = null;
    let stream: MediaStream | null = null;
    let micCtx: AudioContext | null = null;
    let playCtx: AudioContext | null = null;
    let gain: GainNode | null = null;
    let analyser: AnalyserNode | null = null;
    let nextStart = 0;
    let holdUntil = 0;
    let ended: (() => void) | null = null;
    const sources = new Set<AudioBufferSourceNode>();
    let idle: ReturnType<typeof setTimeout> | null = null;

    const apply = (msg: Msg) => setView((v) => reduce(v, msg));
    const fail = (error: string) => setView((v) => (v.phase === 'closed' ? v : { ...v, phase: 'error', error }));

    // Stop everything that is playing, with a 50 ms fade (barge-in), as the original does.
    const silence = () => {
      if (idle) { clearTimeout(idle); idle = null; }
      const toStop = Array.from(sources);
      sources.clear();
      nextStart = 0;
      if (playCtx && gain) {
        const now = playCtx.currentTime;
        holdUntil = now + 0.07;
        try { gain.gain.cancelScheduledValues(now); gain.gain.setValueAtTime(gain.gain.value, now); gain.gain.linearRampToValueAtTime(0.0001, now + 0.05); } catch { /* closed */ }
        const g = gain, c = playCtx;
        setTimeout(() => {
          toStop.forEach((s) => { try { s.onended = null; s.stop(); } catch { /* already stopped */ } });
          try { g.gain.cancelScheduledValues(c.currentTime); g.gain.setValueAtTime(1, c.currentTime); } catch { /* closed */ }
        }, 60);
      }
    };

    const play = (f32: Float32Array, rate: number) => {
      if (!playCtx || !analyser) return;
      const buf = playCtx.createBuffer(1, f32.length, 24000);
      buf.copyToChannel(f32 as Float32Array<ArrayBuffer>, 0);
      const src = playCtx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate || 1;
      src.connect(analyser);
      const now = playCtx.currentTime;
      const at = Math.max(now + 0.02, nextStart, holdUntil);
      try { src.start(at); } catch { return; }
      nextStart = at + buf.duration / (rate || 1);
      sources.add(src);
      if (idle) { clearTimeout(idle); idle = null; }
      setView((v) => (v.phase === 'closed' || v.phase === 'error' ? v : { ...v, phase: 'speaking' }));
      src.onended = () => {
        sources.delete(src);
        if (sources.size) return;
        idle = setTimeout(() => {                      // a short gap between phrases is not the end of the reply
          idle = null;
          if (!sources.size) { nextStart = 0; setView((v) => (v.phase === 'speaking' ? { ...v, phase: 'listening' } : v)); }
        }, 250);
      };
    };

    const close = () => {
      silence();
      try { ws?.close(); } catch { /* closed */ }
      ws = null;
      stream?.getTracks().forEach((t) => t.stop());
      stream = null;
      if (ended) { ended(); ended = null; }
      micCtx?.close().catch(() => {});
      playCtx?.close().catch(() => {});
      micCtx = playCtx = null;
    };
    stopRef.current = () => { close(); if (alive) setView((v) => ({ ...v, phase: 'closed' })); };

    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) { fail('This browser cannot use the microphone here.'); return; }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      } catch {
        fail('The browser refused microphone access. Allow it and start voice again.');
        return;
      }
      if (!alive) { stream.getTracks().forEach((t) => t.stop()); return; }
      ended = captureStarted();                       // the header shows the microphone is open
      stream.getTracks().forEach((t) => t.addEventListener('ended', () => { if (ended) { ended(); ended = null; } }));
      try {
        micCtx = new AudioContext();
        await micCtx.audioWorklet.addModule('/law-voice-framer.js');
        const src = micCtx.createMediaStreamSource(stream);
        const mic = micCtx.createAnalyser();
        mic.fftSize = 1024;
        src.connect(mic);
        const framer = new AudioWorkletNode(micCtx, 'law-framer16k');
        src.connect(framer);
        playCtx = new AudioContext({ sampleRate: 24000 });
        gain = playCtx.createGain();
        analyser = playCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.connect(gain);
        gain.connect(playCtx.destination);
        if (alive) { setMicAnalyser(mic); setSpeechAnalyser(analyser); }
        ws = new WebSocket(voiceUrl(matterId, conversationId));
        framer.port.onmessage = (ev: MessageEvent) => {
          if (ws?.readyState === WebSocket.OPEN && !mutedRef.current) {
            ws.send(JSON.stringify({ type: 'audio_frame', ts: performance.now() / 1000, pcm16_b64: toBase64(ev.data as ArrayBuffer) }));
          }
        };
        ws.onopen = () => ws?.send(JSON.stringify({ type: 'set_context', listen_only: false }));
        ws.onmessage = (ev) => {
          let msg: Msg;
          try { msg = JSON.parse(String(ev.data)) as Msg; } catch { return; }
          if (msg.type === 'audio_out' && typeof msg.pcm16_b64 === 'string') {
            try { play(pcm16ToFloat32(b64ToBytes(msg.pcm16_b64)), Number(msg.playback_rate) || 1); } catch { /* a malformed chunk is skipped */ }
            return;
          }
          if (msg.type === 'event' && msg.event === 'USER_SPEECH_START') silence();
          if (msg.type === 'cancel') silence();
          if (msg.type === 'turn_saved' && typeof msg.turn_id === 'string') savedRef.current(msg.turn_id);
          apply(msg);
        };
        ws.onclose = (ev) => {
          if (!alive) return;
          close();
          setView((v) => (v.phase === 'closed' || v.phase === 'error' ? v
            : CLOSE_TEXT[ev.code] ? { ...v, phase: 'error', error: CLOSE_TEXT[ev.code] }
            : v.phase === 'connecting' ? { ...v, phase: 'error', error: 'The voice service could not be reached. Try again in a moment.' }
            : { ...v, phase: 'closed' }));
        };
      } catch {
        close();
        fail('Voice could not start in this browser.');
      }
    })();

    return () => { alive = false; close(); };
  }, [matterId, conversationId]);

  const setMuted = useCallback((m: boolean) => { mutedRef.current = m; setMutedState(m); }, []);
  const stop = useCallback(() => stopRef.current(), []);
  return { view, muted, setMuted, stop, micAnalyser, speechAnalyser };
}
