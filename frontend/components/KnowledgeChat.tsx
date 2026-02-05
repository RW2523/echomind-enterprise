import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, DocumentChunk } from '../types';
import { ICONS } from '../constants';
import Uploader from './Uploader';
import { createChat, askChat } from '../services/backend';

function mapCitations(citations: any[]): DocumentChunk[] {
  return (citations || []).map((c: any, i: number) => ({
    id: `cite_${i}_${c?.filename ?? 'doc'}`,
    docName: c?.filename ?? 'Unknown',
    content: c?.snippet ?? '',
    metadata: { section: `chunk ${c?.chunk_index ?? ''}`, timestamp: Date.now() }
  }));
}

const KnowledgeChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatId, setChatId] = useState<string>('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    (async () => {
      try {
        const { chat_id } = await createChat('EchoMind Chat');
        setChatId(chat_id);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  const send = async () => {
    const q = input.trim();
    if (!q || !chatId || busy) return;
    setBusy(true);
    const userMsg: ChatMessage = { id: `u_${Date.now()}`, role: 'user', content: q, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    try {
      const out = await askChat(chatId, q);
      const assistantMsg: ChatMessage = {
        id: `a_${Date.now()}`,
        role: 'assistant',
        content: out.answer,
        citations: mapCitations(out.citations),
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages(prev => [...prev, { id: `e_${Date.now()}`, role: 'assistant', content: err?.message || 'Request failed', timestamp: Date.now() }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-4 space-y-4">
        <Uploader onComplete={() => {}} />
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm">
          <div className="font-semibold mb-2">How this works</div>
          <div className="opacity-75">Upload docs → FAISS index + embeddings → RAG answers with citations.</div>
        </div>
      </div>

      <div className="col-span-12 lg:col-span-8 rounded-2xl border border-white/10 bg-white/5 flex flex-col h-[78vh]">
        <div className="px-5 py-4 border-b border-white/10 flex items-center gap-2">
          <div className="opacity-80">{ICONS.chat}</div>
          <div className="font-semibold">Intelligent Knowledge Chat</div>
          <div className="ml-auto text-xs opacity-60">{chatId ? 'Connected' : 'Connecting...'}</div>
        </div>

        <div className="flex-1 overflow-auto p-5 space-y-4">
          {messages.length === 0 && (
            <div className="text-sm opacity-70">Ask questions about uploaded documents. EchoMind will cite relevant chunks.</div>
          )}
          {messages.map(m => (
            <div key={m.id} className={`rounded-2xl p-4 border ${m.role === 'user' ? 'bg-white/10 border-white/10 ml-8' : 'bg-black/20 border-white/10 mr-8'}`}>
              <div className="text-xs opacity-60 mb-2">{m.role === 'user' ? 'You' : 'EchoMind'}</div>
              <div className="text-sm whitespace-pre-wrap">{m.content}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 text-xs opacity-80">
                  <div className="font-semibold mb-1">Sources</div>
                  <ul className="list-disc ml-5 space-y-1">
                    {m.citations.slice(0, 6).map(c => (
                      <li key={c.id}><span className="font-semibold">{c.docName}</span> — {c.metadata.section}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <div className="p-4 border-t border-white/10 flex gap-3">
          <input
            className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-sm outline-none"
            placeholder="Ask something..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') send(); }}
          />
          <button
            className="rounded-xl px-5 py-3 text-sm font-semibold bg-white/10 hover:bg-white/15 disabled:opacity-50"
            onClick={send}
            disabled={busy || !chatId}
          >
            {busy ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeChat;
