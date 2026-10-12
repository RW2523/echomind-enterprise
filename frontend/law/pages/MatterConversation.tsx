import React, { useEffect, useRef, useState } from 'react';
import { Mic, Send, Square, Volume2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { ApiError, postBytes, postForBlob } from '../api/client';
import { keys } from '../api/queries';
import { useMatter } from '../app/matter';
import { startMic, type Mic as Microphone } from '../app/mic';
import Toolbar from '../ui/Toolbar';
import { ErrorNote } from '../ui/primitives';
import { AnswerSheet, streamAnswer, type Answer } from './MatterResearch';

/**
 * S5.1 — the conversation assistant (D-062). Speak a question about the matter; this server transcribes it, the
 * research pipeline answers it from the matter's documents with checked quotes, and the answer is read aloud by
 * this server's voice. Nothing leaves the server and the recording is not kept. Typing works the same way.
 */
const MAX_SECONDS = 60;
const READ_ALOUD = 'law.conversation.readAloud';

type Turn = { id: number; answer: Answer; spoken: boolean };

function concat(frames: ArrayBuffer[]): ArrayBuffer {
  const out = new Uint8Array(frames.reduce((n, f) => n + f.byteLength, 0));
  let at = 0;
  for (const f of frames) { out.set(new Uint8Array(f), at); at += f.byteLength; }
  return out.buffer;
}

function savedReadAloud(): boolean {
  try { return localStorage.getItem(READ_ALOUD) !== 'off'; } catch { return true; }
}

export default function MatterConversation() {
  const matter = useMatter();
  const qc = useQueryClient();
  const base = `/api/legal/matters/${encodeURIComponent(matter.id)}/conversation`;
  const [turns, setTurns] = useState<Turn[]>([]);
  const [phase, setPhase] = useState<'idle' | 'listening' | 'transcribing' | 'answering'>('idle');
  const [seconds, setSeconds] = useState(0);
  const [typed, setTyped] = useState('');
  const [error, setError] = useState('');
  const [voiceNote, setVoiceNote] = useState('');
  const [readAloud, setReadAloud] = useState(savedReadAloud);
  const [playing, setPlaying] = useState(false);
  const mic = useRef<Microphone | null>(null);
  const frames = useRef<ArrayBuffer[]>([]);
  const timer = useRef<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const nextId = useRef(1);

  const stopAudio = () => { audio.current?.pause(); audio.current = null; setPlaying(false); };
  useEffect(() => () => { mic.current?.stop(); abort.current?.abort(); stopAudio(); if (timer.current) window.clearInterval(timer.current); }, []);
  useEffect(() => { try { localStorage.setItem(READ_ALOUD, readAloud ? 'on' : 'off'); } catch { /* private window */ } }, [readAloud]);

  const say = async (text: string) => {
    try {
      const wav = await postForBlob(`${base}/speak`, { text });
      if (typeof URL.createObjectURL !== 'function') return;
      const player = new Audio(URL.createObjectURL(wav));
      audio.current = player;
      player.onended = () => { setPlaying(false); audio.current = null; };
      setPlaying(true);
      await player.play();
    } catch (err) {
      setPlaying(false);
      if (err instanceof ApiError) setVoiceNote(err.message);
    }
  };

  const ask = async (question: string, spoken: boolean) => {
    const q = question.trim();
    if (!q) return;
    stopAudio();
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    const id = nextId.current++;
    setError(''); setPhase('answering');
    const put = (answer: Answer) => setTurns((ts) => [...ts.filter((t) => t.id !== id), { id, answer, spoken }]);
    try {
      const final = await streamAnswer(matter.id, { question: q, versions: 'signed' }, put, ctrl.signal);
      await qc.invalidateQueries({ queryKey: keys.research(matter.id) });
      if (readAloud && final.text && ['done', 'abstained'].includes(final.phase)) await say(final.text);
    } catch (err) {
      if (!ctrl.signal.aborted) setError(err instanceof ApiError ? (err.status === 404 ? 'This matter is no longer available to you.' : err.message)
        : 'The question could not be sent.');
    } finally {
      if (!ctrl.signal.aborted) setPhase('idle');
    }
  };

  const finish = async () => {
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    mic.current?.stop(); mic.current = null;
    const pcm = concat(frames.current);
    frames.current = [];
    if (pcm.byteLength < 3200) { setPhase('idle'); setError('Nothing was heard. Hold the microphone button a little longer, or type your question.'); return; }
    setPhase('transcribing');
    try {
      const r = await postBytes<{ text: string; seconds: number }>(`${base}/transcribe?sample_rate=16000`, pcm);
      if (!r.text.trim()) { setPhase('idle'); setError('No words were recognised. Try again, or type your question.'); return; }
      await ask(r.text, true);
    } catch (err) {
      setPhase('idle');
      setError(err instanceof ApiError ? err.message : 'The recording could not be sent.');
    }
  };

  const listen = async () => {
    stopAudio(); setError(''); frames.current = [];
    try {
      mic.current = await startMic((pcm) => { frames.current.push(pcm); });
    } catch {
      setError('The browser refused microphone access. Type your question instead.');
      return;
    }
    const began = Date.now();
    setSeconds(0); setPhase('listening');
    timer.current = window.setInterval(() => {
      const s = Math.floor((Date.now() - began) / 1000);
      setSeconds(s);
      if (s >= MAX_SECONDS) void finish();
    }, 250);
  };

  const busy = phase === 'transcribing' || phase === 'answering';
  return (
    <>
      <Toolbar crumb={matter.number} title="Conversation" />
      <div className="lw-content" style={{ display: 'grid', gap: 16, alignContent: 'start', maxWidth: 920 }}>
        <section className="lw-paper lw-sheet lw-talk" aria-label="Talk to this matter">
          <h2 className="lw-h2" style={{ margin: 0 }}>Talk to this matter</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>
            Ask out loud and hear the answer. Answers come only from this matter’s signed documents, with the exact words each statement relies on.
            Your voice is transcribed on this server and not kept.
          </p>
          <div className="lw-talk-row">
            {phase === 'listening' ? (
              <button type="button" className="lw-mic" data-live="" onClick={() => void finish()} aria-label="Stop and ask">
                <Square aria-hidden />
              </button>
            ) : (
              <button type="button" className="lw-mic" onClick={() => void listen()} disabled={busy} aria-label="Speak a question">
                <Mic aria-hidden />
              </button>
            )}
            <div role="status" aria-live="polite" className="lw-talk-state">
              {phase === 'listening' ? `Listening… ${seconds} s (up to ${MAX_SECONDS} s). Press again to ask.`
                : phase === 'transcribing' ? 'Transcribing your question…'
                : phase === 'answering' ? 'Reading the documents…'
                : playing ? 'Reading the answer aloud.' : 'Press the microphone and ask your question.'}
            </div>
            {playing && <button type="button" className="lw-btn lw-btn-sm" onClick={stopAudio}><Square aria-hidden />Stop reading</button>}
          </div>
          <form className="lw-talk-type" onSubmit={(e) => { e.preventDefault(); const q = typed; setTyped(''); void ask(q, false); }}>
            <label htmlFor="talk-q" className="lw-sr">Or type a question</label>
            <input id="talk-q" className="lw-input" placeholder="Or type a question" value={typed} onChange={(e) => setTyped(e.target.value)} disabled={busy} />
            <button type="submit" className="lw-btn" disabled={!typed.trim() || busy}><Send aria-hidden />Ask</button>
          </form>
          <label className="lw-check">
            <input type="checkbox" checked={readAloud} onChange={(e) => { setReadAloud(e.target.checked); if (!e.target.checked) stopAudio(); }} />
            <Volume2 aria-hidden /> Read answers aloud
          </label>
          {voiceNote && <p className="lw-meta" style={{ margin: 0 }}>{voiceNote}</p>}
        </section>
        {error && <ErrorNote>{error}</ErrorNote>}
        {turns.length > 0 && (
          <ol className="lw-talk-thread" aria-label="This conversation">
            {turns.map((t) => <TurnItem key={t.id} t={t} />)}
          </ol>
        )}
      </div>
    </>
  );
}

function TurnItem({ t }: { t: Turn }) {
  const [focus, setFocus] = useState<number | null>(null);
  return (
    <li>
      <p className="lw-meta lw-talk-q">{t.spoken ? 'Asked aloud' : 'Typed'}</p>
      <AnswerSheet a={t.answer} focus={focus} onFocus={setFocus} />
    </li>
  );
}
