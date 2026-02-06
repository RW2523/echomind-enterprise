import React, { useState, useEffect, useRef } from 'react';
import { ICONS } from '../constants';
import { polishTranscript, storeTranscript, getTranscriptTags, transcribeWsUrl } from '../services/backend';

const SR = 16000;

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

const OPEN_TIMEOUT_MS = 15000;
const READY_TIMEOUT_MS = 60000;

const LiveTranscription: React.FC = () => {
  const [fullTranscript, setFullTranscript] = useState('');
  const [partial, setPartial] = useState('');
  const [listening, setListening] = useState(false);
  const [polished, setPolished] = useState<string>('');
  const [wsStatus, setWsStatus] = useState<'idle' | 'connecting' | 'loading' | 'ready' | 'error'>('idle');
  const [wsError, setWsError] = useState<string | null>(null);
  const [previewTags, setPreviewTags] = useState<{ tags: string[]; conversation_type: string } | null>(null);
  const [tagsLoading, setTagsLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const recRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const start = async () => {
    if (listening) return;
    setFullTranscript(''); setPartial(''); setPolished(''); setWsError(null);
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
        // Partial and final carry the full transcript from the server — use as single source of truth to avoid duplication
        if (msg.type === 'partial') {
          setFullTranscript(msg.text ?? '');
          setPartial('');
        }
        if (msg.type === 'segment') {
          // Do not append: segment text is already included in the next/previous partial
        }
        if (msg.type === 'final') {
          setFullTranscript((msg.text ?? '').trim());
          setPartial('');
        }
        if (msg.type === 'error') {
          console.error(msg.message);
          handleError(msg.message || 'Server error');
        }
      } catch {
        // ignore parse errors
      }
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

    const readyPromise = new Promise<void>((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('STT loading timeout')), READY_TIMEOUT_MS);
      const check = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'ready') { clearTimeout(t); ws.removeEventListener('message', check); resolve(); }
          if (msg.type === 'error') { clearTimeout(t); ws.removeEventListener('message', check); reject(new Error(msg.message || 'STT failed')); }
        } catch { /* ignore */ }
      };
      ws.addEventListener('message', check);
    });

    try {
      await readyPromise;
    } catch (e) {
      handleError(e?.message || 'STT not ready');
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recRef.current = stream;

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SR });
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

  const polish = async () => {
    const raw = (fullTranscript || '').trim();
    if (!raw) return;
    try {
      const out = await polishTranscript(raw);
      setPolished(out.polished ?? '');
    } catch (e) {
      console.error(e);
      alert((e as Error)?.message || 'Polish failed');
    }
  };

  const store = async () => {
    const raw = (fullTranscript || '').trim();
    if (!raw) return;
    try {
      await storeTranscript(raw, polished || null);
      alert('Stored into EchoMind knowledge base.');
    } catch (e) {
      console.error(e);
      alert((e as Error)?.message || 'Store failed');
    }
  };

  const checkTags = async () => {
    const raw = (fullTranscript || '').trim();
    if (!raw) {
      setPreviewTags(null);
      return;
    }
    setTagsLoading(true);
    setPreviewTags(null);
    try {
      const out = await getTranscriptTags(raw);
      setPreviewTags({ tags: out.tags || [], conversation_type: out.conversation_type || 'casual' });
    } catch (e) {
      console.error(e);
      setPreviewTags({ tags: [], conversation_type: 'casual' });
    } finally {
      setTagsLoading(false);
    }
  };

  useEffect(() => () => { stopMic(false); wsRef.current?.close(); }, []);

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 sm:px-5 sm:py-4 border-b border-white/10">
        <div className="opacity-80"><ICONS.Mic className="w-5 h-5" /></div>
        <div className="font-semibold">Real-Time Transcription</div>
        <div className="ml-auto flex items-center gap-2 flex-wrap">
          {wsStatus === 'connecting' && <span className="text-xs text-slate-400">Connecting…</span>}
          {wsStatus === 'loading' && <span className="text-xs text-slate-400">Loading STT…</span>}
          {wsError && <span className="text-xs text-red-400 max-w-[200px] truncate" title={wsError}>{wsError}</span>}
          {!listening ? (
            <button onClick={start} disabled={wsStatus === 'connecting' || wsStatus === 'loading'} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50">Start</button>
          ) : (
            <button onClick={() => stopMic(true)} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15">Stop</button>
          )}
          <button onClick={polish} disabled={!fullTranscript.trim()} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50">Polish</button>
          <button onClick={store} disabled={!fullTranscript.trim()} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50">Store</button>
          <button onClick={checkTags} disabled={!fullTranscript.trim() || tagsLoading} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50">
            {tagsLoading ? '…' : 'Check tags'}
          </button>
        </div>
      </div>

      {previewTags && (
        <div className="shrink-0 px-4 sm:px-5 py-2 border-b border-white/10 bg-black/10 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">Type:</span>
          <span className="text-xs font-medium text-cyan-400 capitalize">{previewTags.conversation_type}</span>
          <span className="text-xs text-slate-500 mx-1">|</span>
          <span className="text-xs text-slate-400">Tags:</span>
          {previewTags.tags.length === 0 ? (
            <span className="text-xs text-slate-500">none</span>
          ) : (
            previewTags.tags.map((t, i) => (
              <span key={i} className="rounded-lg bg-white/10 border border-white/10 px-2 py-1 text-xs text-white/90">
                {t}
              </span>
            ))
          )}
        </div>
      )}

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-2 gap-4 p-4 sm:p-5 overflow-auto">
        <div className="rounded-2xl border border-white/10 bg-black/20 p-4 min-h-[280px] flex flex-col">
          <div className="text-xs font-semibold opacity-70 mb-3 shrink-0">Live transcript (updates as you speak)</div>
          <div className="flex-1 min-h-0 text-sm whitespace-pre-wrap opacity-90 overflow-auto">
            {[fullTranscript, partial].filter(Boolean).join(' ') || '—'}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/20 p-4 min-h-[280px] flex flex-col">
          <div className="text-xs font-semibold opacity-70 mb-3 shrink-0">Polished</div>
          <div className="flex-1 min-h-0 text-sm whitespace-pre-wrap opacity-90 overflow-auto">{polished || 'Click “Polish” after transcription.'}</div>
        </div>
      </div>
    </div>
  );
};

export default LiveTranscription;
