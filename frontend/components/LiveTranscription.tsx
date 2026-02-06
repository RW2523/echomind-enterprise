import React, { useState, useEffect, useRef } from 'react';
import { ICONS } from '../constants';
import { polishTranscript, storeTranscript, transcribeWsUrl } from '../services/backend';

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

const LiveTranscription: React.FC = () => {
  const [fullTranscript, setFullTranscript] = useState('');
  const [partial, setPartial] = useState('');
  const [listening, setListening] = useState(false);
  const [polished, setPolished] = useState<string>('');
  const wsRef = useRef<WebSocket | null>(null);
  const recRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const start = async () => {
    if (listening) return;
    setFullTranscript(''); setPartial(''); setPolished('');
    const ws = new WebSocket(transcribeWsUrl());
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'partial') setPartial(msg.text || '');
      if (msg.type === 'segment') {
        const t = (msg.text || '').trim();
        if (t) setFullTranscript(prev => (prev ? `${prev} ${t}` : t));
      }
      if (msg.type === 'final') {
        const t = (msg.text || '').trim();
        if (t) setFullTranscript(t);
        setPartial('');
      }
      if (msg.type === 'error') console.error(msg.message);
    };

    await new Promise<void>((res) => ws.addEventListener('open', () => res(), { once: true }));

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
    const out = await polishTranscript(raw);
    setPolished(out.polished);
  };

  const store = async () => {
    const raw = (fullTranscript || '').trim();
    if (!raw) return;
    await storeTranscript(raw, polished || null);
    alert('Stored into EchoMind knowledge base.');
  };

  useEffect(() => () => { stopMic(false); wsRef.current?.close(); }, []);

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 sm:px-5 sm:py-4 border-b border-white/10">
        <div className="opacity-80"><ICONS.Mic className="w-5 h-5" /></div>
        <div className="font-semibold">Real-Time Transcription</div>
        <div className="ml-auto flex gap-2">
          {!listening ? (
            <button onClick={start} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15">Start</button>
          ) : (
            <button onClick={() => stopMic(true)} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15">Stop</button>
          )}
          <button onClick={polish} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15">Polish</button>
          <button onClick={store} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15">Store</button>
        </div>
      </div>

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
