import React, { useState, useEffect, useRef } from 'react';
import { ICONS } from '../constants';
import { defaultTranscriptName, transcribeWsUrl, getTranscriptTags, updateTranscript } from '../services/backend';

/** Kyutai STT sample rate (24kHz). Backend sends this in ready message; we use it for AudioContext and start payload. */
const KYUTAI_SAMPLE_RATE = 24000;

function floatTo16BitPCM(input: Float32Array) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function b64FromBytes(bytes: Uint8Array) {
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunkSize)) as any);
  }
  return btoa(binary);
}

function formatSessionDateTime(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${y}-${m}-${day} ${h}:${min}`;
}

const OPEN_TIMEOUT_MS = 15000;
// Kyutai model load can take 2–5 min on first run (download ~1GB + GPU load)
const READY_TIMEOUT_MS = 300000;

const LiveTranscription: React.FC = () => {
  const [fullTranscript, setFullTranscript] = useState('');
  const [partial, setPartial] = useState('');
  const [listening, setListening] = useState(false);
  const [wsStatus, setWsStatus] = useState<'idle' | 'connecting' | 'loading' | 'ready' | 'error'>('idle');
  const [wsError, setWsError] = useState<string | null>(null);

  // Start popup
  const [showStartModal, setShowStartModal] = useState(false);
  const [modalName, setModalName] = useState('');
  const [modalLocation, setModalLocation] = useState('');

  // Session metadata (editable bar) – set when user clicks Start
  const [sessionName, setSessionName] = useState('');
  const [sessionLocation, setSessionLocation] = useState('');
  const [sessionStartedAt, setSessionStartedAt] = useState<Date | null>(null);
  const [customTags, setCustomTags] = useState<string[]>([]);
  const [newTagInput, setNewTagInput] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const recRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const transcriptForTagsRef = useRef('');
  const lastStoredTranscriptIdRef = useRef<string | null>(null);
  const pendingTagsRef = useRef<string[] | null>(null);

  const openStartModal = () => {
    setModalName('');
    setModalLocation('');
    setShowStartModal(true);
  };

  const applyDefault = () => {
    setModalName(defaultTranscriptName());
    setModalLocation('default');
  };

  const startSession = async () => {
    const name = (modalName || '').trim() || defaultTranscriptName();
    const location = (modalLocation || '').trim() || 'default';
    setSessionName(name);
    setSessionLocation(location);
    setSessionStartedAt(new Date());
    setCustomTags([]);
    transcriptForTagsRef.current = '';
    lastStoredTranscriptIdRef.current = null;
    pendingTagsRef.current = null;
    setShowStartModal(false);
    await doStart(name, location);
  };

  const doStart = async (_name: string, _location: string) => {
    if (listening) return;
    setFullTranscript('');
    setPartial('');
    setWsError(null);
    setWsStatus('connecting');
    const ws = new WebSocket(transcribeWsUrl());
    wsRef.current = ws;

    const handleError = (err: string) => {
      setWsError(err);
      setWsStatus('error');
      stopMic(false);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'loading') setWsStatus('loading');
        if (msg.type === 'ready') setWsStatus('ready');
        if (msg.type === 'partial') {
          const t = msg.text ?? '';
          setFullTranscript(t);
          transcriptForTagsRef.current = t;
          setPartial('');
        }
        if (msg.type === 'segment') {}
        if (msg.type === 'final') {
          const t = (msg.text ?? '').trim();
          setFullTranscript(t);
          transcriptForTagsRef.current = t;
          setPartial('');
        }
        if (msg.type === 'stored') {
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
        if (msg.type === 'error') {
          console.error(msg.message);
          handleError(msg.message || 'Server error');
        }
      } catch {}
    };

    ws.onerror = () => handleError('WebSocket error');
    ws.onclose = () => {
      setWsStatus((s) => (s === 'error' ? s : 'idle'));
      stopMic(false);
    };

    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('Connection timeout')), OPEN_TIMEOUT_MS);
      ws.addEventListener('open', () => { clearTimeout(t); resolve(); }, { once: true });
      ws.addEventListener('error', () => { clearTimeout(t); reject(new Error('WebSocket failed')); }, { once: true });
    }).catch((e) => {
      setWsError(e?.message || 'Connection failed');
      setWsStatus('error');
      throw e;
    });

    const readyPromise = new Promise<number>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('Kyutai STT loading timeout (model may still be downloading)')), READY_TIMEOUT_MS);
      const check = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'ready') {
            clearTimeout(t);
            ws.removeEventListener('message', check);
            resolve(msg.sample_rate ?? KYUTAI_SAMPLE_RATE);
          }
          if (msg.type === 'error') {
            clearTimeout(t);
            ws.removeEventListener('message', check);
            reject(new Error(msg.message || 'STT failed'));
          }
        } catch {}
      };
      ws.addEventListener('message', check);
    });

    let sampleRate: number;
    try {
      sampleRate = await readyPromise;
    } catch (e) {
      handleError(e?.message || 'Kyutai STT not ready');
      return;
    }

    ws.send(JSON.stringify({ type: 'start', auto_store: true, sample_rate: sampleRate, language: 'en', name: _name || undefined, location: _location || undefined }));

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recRef.current = stream;

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate });
    audioCtxRef.current = audioCtx;
    const src = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = floatTo16BitPCM(input);
      const b64 = b64FromBytes(new Uint8Array(pcm16.buffer));
      wsRef.current.send(JSON.stringify({ type: 'audio', pcm16_b64: b64 }));
    };

    src.connect(processor);
    processor.connect(audioCtx.destination);
    setListening(true);
  };

  const stopMic = (sendStop: boolean) => {
    if (sendStop && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
    }
    processorRef.current?.disconnect();
    processorRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    recRef.current?.getTracks().forEach(t => t.stop());
    recRef.current = null;
    setListening(false);
  };

  const handleStopAndExtractTags = async () => {
    const text = (transcriptForTagsRef.current || fullTranscript || '').trim();
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
        // ignore tag extraction failure
      }
    }
  };

  const addTag = () => {
    const t = (newTagInput || '').trim();
    if (t && !customTags.includes(t)) {
      setCustomTags((prev) => [...prev, t].slice(0, 20));
      setNewTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setCustomTags((prev) => prev.filter((x) => x !== tag));
  };

  useEffect(() => () => { stopMic(false); wsRef.current?.close(); }, []);

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {/* Start modal */}
      {showStartModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setShowStartModal(false)}>
          <div className="rounded-2xl border border-white/20 bg-slate-900 shadow-xl max-w-md w-full p-5 space-y-4" onClick={(e) => e.stopPropagation()}>
            <div className="font-semibold text-white">Start transcription</div>
            <p className="text-sm text-slate-400">Name and location are saved with the transcript every 1 min and used in RAG (e.g. &quot;summary of last 5 mins in office&quot;).</p>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Name</label>
              <input
                type="text"
                value={modalName}
                onChange={(e) => setModalName(e.target.value)}
                placeholder="e.g. transcript_2025-02-12_14-30"
                className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
              <input
                type="text"
                value={modalLocation}
                onChange={(e) => setModalLocation(e.target.value)}
                placeholder="e.g. default or Office"
                className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
              />
            </div>
            <div className="flex flex-wrap gap-2 pt-2">
              <button type="button" onClick={applyDefault} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15">
                Default
              </button>
              <button type="button" onClick={() => setShowStartModal(false)} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-400 hover:bg-white/15">
                Cancel
              </button>
              <button type="button" onClick={startSession} className="rounded-xl px-4 py-2 text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30">
                Start
              </button>
            </div>
          </div>
        </div>
      )}

      <div
        className={`shrink-0 flex items-center gap-3 px-4 py-3 sm:px-5 sm:py-4 border-b transition-all duration-300 ${
          listening ? 'border-cyan-500/30 bg-cyan-500/5' : 'border-white/10'
        }`}
      >
        <div className="flex items-center gap-2">
          <div className={`relative flex items-center justify-center w-10 h-10 rounded-xl ${listening ? 'bg-cyan-500/20' : 'bg-white/5'}`}>
            <ICONS.Mic className={`w-5 h-5 ${listening ? 'text-cyan-400' : 'text-slate-400'}`} />
            {listening && (
              <span className="absolute inset-0 rounded-xl bg-cyan-400/20 animate-ping" style={{ animationDuration: '1.5s' }} />
            )}
          </div>
          <div>
            <div className="font-semibold">Real-Time Transcription</div>
            <div className="flex items-center gap-2 mt-0.5">
              {listening ? (
                <>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/20 border border-red-500/40 px-2.5 py-0.5 text-[10px] font-medium text-red-400 uppercase tracking-wider">Live</span>
                  <span className="flex items-center gap-1">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <span key={i} className="w-1 rounded-full bg-cyan-400/80 animate-listening-bar" style={{ height: 8, animationDelay: `${i * 0.12}s` }} />
                    ))}
                  </span>
                  <span className="text-[10px] text-cyan-400/90">Listening…</span>
                </>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-white/10 border border-white/10 px-2.5 py-0.5 text-[10px] font-medium text-slate-400 uppercase tracking-wider">Stopped</span>
              )}
            </div>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2 flex-wrap">
          {wsStatus === 'connecting' && <span className="text-xs text-slate-400">Connecting…</span>}
          {wsStatus === 'loading' && <span className="text-xs text-slate-400">Loading Kyutai STT… (first run may take 2–5 min)</span>}
          {wsError && <span className="text-xs text-red-400 max-w-[120px] sm:max-w-[200px] truncate" title={wsError}>{wsError}</span>}
          {!listening ? (
            <button type="button" onClick={openStartModal} disabled={wsStatus === 'connecting' || wsStatus === 'loading'} className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 transition-colors touch-manipulation">Start</button>
          ) : (
            <button type="button" onClick={handleStopAndExtractTags} className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors touch-manipulation">Stop</button>
          )}
        </div>
      </div>

      {/* Editable bar: name, location, date/time, custom tags - wraps on mobile */}
      {(listening || sessionName || sessionLocation || sessionStartedAt) && (
        <div className="shrink-0 px-3 sm:px-5 py-3 border-b border-white/10 bg-black/10 flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Name</span>
            <input type="text" value={sessionName} onChange={(e) => setSessionName(e.target.value)} placeholder="Transcript name" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-36 sm:w-48 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Location</span>
            <input type="text" value={sessionLocation} onChange={(e) => setSessionLocation(e.target.value)} placeholder="Location" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-24 sm:w-32 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Date & time</span>
            <span className="text-sm text-slate-300">{sessionStartedAt ? formatSessionDateTime(sessionStartedAt) : '—'}</span>
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Tags</span>
            {customTags.map((tag) => (
              <span key={tag} className="inline-flex items-center gap-1 rounded-lg bg-white/10 border border-white/10 px-2 py-1.5 text-xs text-white/90">
                {tag}
                <button type="button" onClick={() => removeTag(tag)} className="text-slate-400 hover:text-white leading-none p-0.5 touch-manipulation min-w-[28px]" aria-label={`Remove ${tag}`}>×</button>
              </span>
            ))}
            <input type="text" value={newTagInput} onChange={(e) => setNewTagInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())} placeholder="+ Add tag" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-20 sm:w-24 min-w-0 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
            <button type="button" onClick={addTag} className="rounded-lg px-3 py-2 text-xs font-medium bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 touch-manipulation min-h-[40px]">Add</button>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 p-4 sm:p-5 overflow-auto">
        <div className="rounded-2xl border border-white/10 bg-black/20 p-4 min-h-[280px] flex flex-col">
          <div className="text-xs font-semibold opacity-70 mb-3 shrink-0">Live transcript (auto-saved every 1 min to transcripts table + RAG; name, location &amp; time used for chat queries)</div>
          <div className="flex-1 min-h-0 text-sm whitespace-pre-wrap opacity-90 overflow-auto">
            {[fullTranscript, partial].filter(Boolean).join(' ') || '—'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveTranscription;
