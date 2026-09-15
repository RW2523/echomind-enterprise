/**
 * Handraise = TTS read-out of Silent Assistant checks (server Piper audio, browser speech fallback).
 * Shared by the ••• overflow menu, the check cards and the card modal.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { AnalysisCard, SentenceCheck } from '../types';
import { speakText } from '../services/backend';
import { asCheck, sortChecksProblemsFirst } from '../utils/silentAssistant';

/** TTS-ready sentence for a check: server `phrase` if present, else composed from label/explanation. */
export function cardPhrase(card: AnalysisCard | SentenceCheck): string {
  const c = asCheck(card);
  if (c.phrase && c.phrase.trim()) return c.phrase.trim();
  const tagLabel = c.tags?.[0]?.label ?? c.label;
  return `This statement is ${tagLabel}. "${c.sentence_text || c.segment_text}". ${c.explanation ?? ''}`.trim();
}

interface Player { ctx: AudioContext; src: AudioBufferSourceNode }
let current: Player | null = null;

function stopAll() {
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) window.speechSynthesis.cancel();
  if (current) {
    try { current.src.stop(); } catch { /* already stopped */ }
    try { current.ctx.close(); } catch { /* ignore */ }
    current = null;
  }
}

function browserSpeak(text: string): Promise<void> {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) { resolve(); return; }
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.0;
    utt.pitch = 1.0;
    utt.onend = () => resolve();
    utt.onerror = () => resolve();
    window.speechSynthesis.speak(utt);
  });
}

async function playAudioB64(b64: string): Promise<void> {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(bytes.buffer);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  current = { ctx, src };
  return new Promise((resolve) => {
    src.onended = () => { if (current?.src === src) { try { ctx.close(); } catch { /* ignore */ } current = null; } resolve(); };
    src.start(0);
  });
}

export function useHandraise() {
  const [speaking, setSpeaking] = useState(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; stopAll(); }; }, []);

  const stop = useCallback(() => { stopAll(); if (mounted.current) setSpeaking(false); }, []);

  const speakCard = useCallback(async (card: AnalysisCard | SentenceCheck) => {
    stopAll();
    setSpeaking(true);
    const text = cardPhrase(card);
    try {
      const res = await speakText('card', { text });
      if (res.audio_b64) await playAudioB64(res.audio_b64);
      else await browserSpeak(res.text || text);
    } catch {
      await browserSpeak(text);
    } finally {
      if (mounted.current) setSpeaking(false);
    }
  }, []);

  const speakSummary = useCallback(async (cards: (AnalysisCard | SentenceCheck)[]) => {
    if (!cards.length) return;
    stopAll();
    setSpeaking(true);
    // Problems first, and prefer server phrases when we have them.
    const list = sortChecksProblemsFirst(cards.map(asCheck));
    try {
      const res = await speakText('summary', { cards: list });
      if (res.audio_b64) await playAudioB64(res.audio_b64);
      else await browserSpeak(res.text || list.map(cardPhrase).join('. '));
    } catch {
      await browserSpeak('Here is the analysis summary. ' + list.map(cardPhrase).join('. '));
    } finally {
      if (mounted.current) setSpeaking(false);
    }
  }, []);

  return { speaking, speakCard, speakSummary, stop };
}

export default useHandraise;
