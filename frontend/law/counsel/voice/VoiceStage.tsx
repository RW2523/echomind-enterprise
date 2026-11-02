/**
 * The voice panel that takes the question box's place while voice mode is on (owner's request 2026-11-02): an orb that
 * follows who is speaking, what is being heard, what the assistant is saying, mute and end. The answers themselves
 * land in the conversation above as ordinary turns, with their sources, copy and rating.
 */
import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, PhoneOff } from 'lucide-react';
import { useVoice, type VoicePhase } from './useVoice';

const PHASE_TEXT: Record<VoicePhase, string> = {
  connecting: 'Starting voice…', listening: 'Listening', hearing: 'Listening', thinking: 'Checking the documents…',
  speaking: 'Speaking', closed: 'Voice ended', error: 'Voice stopped',
};

/** A calm orb whose size follows the voice that is active (the assistant's while it speaks, else the microphone). */
function Orb({ phase, mic, speech, muted }: { phase: VoicePhase; mic: AnalyserNode | null; speech: AnalyserNode | null; muted: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = phase === 'speaking' ? speech : muted ? null : mic;
    if (!node || !ref.current || typeof requestAnimationFrame !== 'function') { ref.current?.style.setProperty('--level', '0'); return; }
    const data = new Uint8Array(node.fftSize);
    let raf = 0;
    const tick = () => {
      node.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sum += v * v; }
      const level = Math.min(1, Math.sqrt(sum / data.length) * 4);
      ref.current?.style.setProperty('--level', level.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, mic, speech, muted]);
  return <div ref={ref} className="qc-orb" data-phase={phase} data-muted={muted || undefined} aria-hidden><span /></div>;
}

export default function VoiceStage({ matterId, matterNumber, conversationId, onTurnSaved, onClose }: {
  matterId: string; matterNumber: string; conversationId: string; onTurnSaved: (turnId: string) => void; onClose: () => void;
}) {
  const { view, muted, setMuted, stop, micAnalyser, speechAnalyser } = useVoice({ matterId, conversationId, onTurnSaved });
  const end = () => { stop(); onClose(); };
  const over = view.phase === 'closed' || view.phase === 'error';
  return (
    <section className="qc-voice" aria-label="Voice conversation" data-phase={view.phase}>
      <div className="qc-voice-main">
        <Orb phase={view.phase} mic={micAnalyser} speech={speechAnalyser} muted={muted} />
        <div className="qc-voice-text">
          <p className="qc-voice-state" role="status">{muted && !over ? 'Microphone muted' : PHASE_TEXT[view.phase]}</p>
          {view.partial && <p className="qc-voice-heard" data-live>“{view.partial}”</p>}
          {!view.partial && view.heard && <p className="qc-voice-heard">You: {view.heard}</p>}
          {view.phase === 'thinking' && view.holding && !view.saying && <p className="qc-voice-saying">{view.holding}</p>}
          {view.saying && <p className="qc-voice-saying">{view.saying}</p>}
          {view.error && <p className="qc-error" role="alert" style={{ margin: 0 }}>{view.error}</p>}
          {view.phase === 'listening' && !view.heard && !muted && (
            <p className="qc-meta" style={{ margin: 0 }}>Ask about {matterNumber}. Speak over the answer to interrupt it.</p>
          )}
        </div>
      </div>
      <div className="qc-voice-controls">
        {!over && (
          <button type="button" className="qc-btn" aria-pressed={muted} onClick={() => setMuted(!muted)}>
            {muted ? <MicOff aria-hidden /> : <Mic aria-hidden />}{muted ? 'Unmute' : 'Mute'}
          </button>
        )}
        <button type="button" className="qc-btn qc-voice-end" onClick={end}><PhoneOff aria-hidden />{over ? 'Close' : 'End voice'}</button>
      </div>
      <p className="qc-meta qc-voice-note">Answers come only from {matterNumber}'s documents and are saved in this chat with their sources.
        Your voice is transcribed on this server and not kept.</p>
    </section>
  );
}
