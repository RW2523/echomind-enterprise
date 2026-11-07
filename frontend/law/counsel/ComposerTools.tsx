/**
 * Two tools of the question box (owner's request 2026-11-02, "like ChatGPT"):
 *
 * - Add a document: the file goes into the matter's library as a draft (the same import, checks and access rules as the
 *   Documents page), and the next answers use only it until it is taken off. Offered only to people whose role may add
 *   documents; the server checks again.
 * - Dictate: speak the question; the server transcribes it (the "Ask by voice" endpoint, not kept) and the words land in
 *   the box to be read and sent like typed ones. Shown only when voice is turned on for the firm; the header's recording
 *   status follows the microphone while it is open.
 */
import React, { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { FilePlus2, Mic, Square } from 'lucide-react';
import { api, ApiError, postBytes } from '../api/client';
import { keys } from '../api/queries';
import { startMic, type Mic as Microphone } from '../app/mic';

export const ACCEPTED = '.pdf,.docx,.txt,.md';
const ACCEPTED_RE = /\.(pdf|docx|txt|md)$/i;
export const MAX_MB = 100;

/** Add one file to the matter (as a draft) and report it; errors are said in words, never thrown. */
export async function addDocument(matterId: string, file: File): Promise<{ id: string; title: string } | { error: string }> {
  if (!ACCEPTED_RE.test(file.name)) return { error: 'Only PDF, Word (.docx), text and Markdown files can be added.' };
  if (file.size > MAX_MB * 1024 * 1024) return { error: `The file is larger than ${MAX_MB} MB.` };
  const form = new FormData();
  form.append('file', file);
  form.append('status', 'draft');
  form.append('title', file.name.replace(/\.[^.]+$/, '').slice(0, 200) || file.name);
  try {
    const r = await api.upload<{ document: { id: string; title: string } }>(
      `/api/legal/matters/${encodeURIComponent(matterId)}/documents/import`, form);
    return { id: r.document.id, title: r.document.title };
  } catch (e) {
    return { error: e instanceof ApiError ? e.message : 'The file could not be added.' };
  }
}

export function AddDocument({ matterId, matterNumber, onAdded, onError }: {
  matterId: string; matterNumber: string; onAdded: (docs: { id: string; title: string }[]) => void; onError: (message: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [adding, setAdding] = useState(false);
  const qc = useQueryClient();
  const pick = async (files: FileList | null) => {
    const list = Array.from(files ?? []).slice(0, MAX_AT_ONCE);
    if (input.current) input.current.value = '';
    if (!list.length) return;
    setAdding(true);
    const added: { id: string; title: string }[] = [];
    const failed: string[] = [];
    for (const file of list) {                            // one at a time, each through the same import
      const r = await addDocument(matterId, file);
      if ('error' in r) failed.push(list.length > 1 ? `${file.name}: ${r.error}` : r.error);
      else added.push(r);
    }
    setAdding(false);
    void qc.invalidateQueries({ queryKey: keys.documents(matterId) });
    if (added.length) onAdded(added);
    if (failed.length) onError(failed.join(' '));
  };
  return (
    <>
      <input ref={input} type="file" accept={ACCEPTED} multiple hidden onChange={(e) => void pick(e.target.files)} data-testid="add-document-input" />
      <button type="button" className="qc-pop-item" disabled={adding} onClick={() => input.current?.click()}>
        <FilePlus2 aria-hidden />{adding ? 'Adding…' : `Add documents to ${matterNumber}…`}
      </button>
    </>
  );
}

const MAX_SECONDS = 60;

function concat(frames: ArrayBuffer[]): ArrayBuffer {
  const out = new Uint8Array(frames.reduce((n, f) => n + f.byteLength, 0));
  let at = 0;
  for (const f of frames) { out.set(new Uint8Array(f), at); at += f.byteLength; }
  return out.buffer;
}

export function Dictate({ matterId, disabled, onText, onError }: {
  matterId: string; disabled: boolean; onText: (text: string) => void; onError: (message: string) => void;
}) {
  const [phase, setPhase] = useState<'idle' | 'listening' | 'transcribing'>('idle');
  const [seconds, setSeconds] = useState(0);
  const mic = useRef<Microphone | null>(null);
  const frames = useRef<ArrayBuffer[]>([]);
  const timer = useRef<number | null>(null);

  const stopTimer = () => { if (timer.current) { window.clearInterval(timer.current); timer.current = null; } };
  useEffect(() => () => { stopTimer(); mic.current?.stop(); mic.current = null; }, []);

  const finish = async () => {
    stopTimer();
    mic.current?.stop(); mic.current = null;
    const pcm = concat(frames.current);
    frames.current = [];
    if (pcm.byteLength < 3200) { setPhase('idle'); onError('Nothing was heard. Speak a little longer, or type your question.'); return; }
    setPhase('transcribing');
    try {
      const r = await postBytes<{ text: string }>(`/api/legal/matters/${encodeURIComponent(matterId)}/conversation/transcribe?sample_rate=16000`, pcm);
      if (r.text.trim()) onText(r.text.trim()); else onError('No words were recognised. Try again, or type your question.');
    } catch (e) {
      onError(e instanceof ApiError ? e.message : 'The recording could not be transcribed.');
    }
    setPhase('idle');
  };

  const listen = async () => {
    frames.current = [];
    try {
      mic.current = await startMic((pcm) => { frames.current.push(pcm); });
    } catch {
      onError('The browser refused microphone access. Type your question instead.');
      return;
    }
    const began = Date.now();
    setSeconds(0);
    setPhase('listening');
    timer.current = window.setInterval(() => {
      const s = Math.floor((Date.now() - began) / 1000);
      setSeconds(s);
      if (s >= MAX_SECONDS) void finish();
    }, 250);
  };

  if (phase === 'listening') {
    return (
      <button type="button" className="qc-tool qc-dictating" onClick={() => void finish()} aria-label={`Stop dictating (${seconds} seconds)`}>
        <span className="qc-rec-dot" aria-hidden /><Square aria-hidden />
        <span>Listening {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')}</span>
      </button>
    );
  }
  return (
    <button type="button" className="qc-tool" disabled={disabled || phase === 'transcribing'} onClick={() => void listen()}
      aria-label={phase === 'transcribing' ? 'Transcribing what you said' : 'Dictate your question'} title="Dictate (transcribed on this server, not kept)">
      <Mic aria-hidden />{phase === 'transcribing' && <span>Transcribing…</span>}
    </button>
  );
}

/** Files dropped on the question box (one at a time, the first file only). */
export function droppedFile(e: React.DragEvent): File | null {
  return e.dataTransfer?.files?.[0] ?? null;
}

/** Every file dropped, at most as many as an answer can be limited to. */
export const MAX_AT_ONCE = 10;
export function droppedFiles(e: React.DragEvent): File[] {
  return Array.from(e.dataTransfer?.files ?? []).slice(0, MAX_AT_ONCE);
}
