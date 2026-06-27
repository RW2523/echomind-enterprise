import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { askChat, askChatStream, createChat } from '../services/backend';

export const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] } }),
};

export const md = {
  p: ({ children }: any) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }: any) => <ul className="list-disc ml-4 mb-2 space-y-0.5">{children}</ul>,
  ol: ({ children }: any) => <ol className="list-decimal ml-4 mb-2 space-y-0.5">{children}</ol>,
  strong: ({ children }: any) => <strong className="text-white">{children}</strong>,
  h1: ({ children }: any) => <h3 className="text-base font-bold text-white mt-2 mb-1">{children}</h3>,
  h2: ({ children }: any) => <h3 className="text-sm font-bold text-white mt-2 mb-1">{children}</h3>,
  table: ({ children }: any) => <table className="w-full text-xs border-collapse my-2">{children}</table>,
  th: ({ children }: any) => <th className="border border-white/10 px-2 py-1 text-left bg-white/5">{children}</th>,
  td: ({ children }: any) => <td className="border border-white/10 px-2 py-1">{children}</td>,
};

export function useChatId(title: string) {
  const ref = useRef<string>('');
  useEffect(() => { createChat(title).then((r) => { ref.current = r.chat_id; }).catch(() => {}); }, []);
  return () => ref.current || 'vertical-demo';
}

export const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });

export const Ambient: React.FC = () => (
  <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
    <motion.div className="absolute -top-40 -right-40 w-[640px] h-[640px] rounded-full blur-[140px]" style={{ backgroundColor: 'var(--glow-1)' }}
      animate={{ scale: [1, 1.15, 1], opacity: [0.7, 1, 0.7] }} transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }} />
    <motion.div className="absolute -bottom-40 -left-40 w-[560px] h-[560px] rounded-full blur-[140px]" style={{ backgroundColor: 'var(--glow-2)' }}
      animate={{ scale: [1.1, 1, 1.1], opacity: [0.6, 0.9, 0.6] }} transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }} />
  </div>
);

type Link = { label: string; id: string; primary?: boolean };
export const Nav: React.FC<{ name: string; accent: string; links: Link[] }> = ({ name, accent, links }) => (
  <nav className="flex items-center justify-between px-5 sm:px-10 py-4 max-w-6xl mx-auto">
    <div className="flex items-center gap-2.5">
      <span className="w-8 h-8 rounded-xl flex items-center justify-center text-[#05070a] font-bold" style={{ backgroundColor: accent }}>E</span>
      <span className="font-bold text-white">{name}</span>
    </div>
    <div className="flex items-center gap-2 text-sm">
      {links.map((l) => (
        <button key={l.id} onClick={() => scrollTo(l.id)}
          className={l.primary ? 'px-3 py-1.5 rounded-lg font-semibold text-accent bg-accent/10 border border-accent/30 hover:bg-accent/20' : 'hidden sm:inline px-3 py-1.5 rounded-lg text-slate-300 hover:text-white'}>
          {l.label}
        </button>
      ))}
    </div>
  </nav>
);

export const Hero: React.FC<{ accent: string; eyebrow: string; titleA: string; titleHi: string; titleB: string; sub: string; ctas: Link[]; stats: { k: string; v: string }[] }> = ({ accent, eyebrow, titleA, titleHi, titleB, sub, ctas, stats }) => (
  <header className="max-w-6xl mx-auto px-5 sm:px-10 pt-14 sm:pt-24 pb-16 sm:pb-24">
    <motion.p variants={fadeUp} initial="hidden" animate="show" className="text-accent font-semibold tracking-widest uppercase text-xs mb-4">{eyebrow}</motion.p>
    <motion.h1 variants={fadeUp} initial="hidden" animate="show" custom={1} className="text-4xl sm:text-6xl font-black text-white leading-[1.05] max-w-3xl">
      {titleA}<span style={{ color: accent }}>{titleHi}</span>{titleB}
    </motion.h1>
    <motion.p variants={fadeUp} initial="hidden" animate="show" custom={2} className="mt-6 text-lg text-slate-400 max-w-2xl">{sub}</motion.p>
    <motion.div variants={fadeUp} initial="hidden" animate="show" custom={3} className="mt-9 flex flex-wrap gap-3">
      {ctas.map((c) => (
        <button key={c.id} onClick={() => scrollTo(c.id)}
          className={c.primary ? 'px-6 py-3 rounded-xl font-semibold text-[#05070a]' : 'px-6 py-3 rounded-xl font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10'}
          style={c.primary ? { backgroundColor: accent } : undefined}>{c.label}</button>
      ))}
    </motion.div>
    <motion.div variants={fadeUp} initial="hidden" animate="show" custom={4} className="mt-14 flex flex-wrap gap-8">
      {stats.map((s) => (<div key={s.v}><div className="text-3xl font-black" style={{ color: accent }}>{s.k}</div><div className="text-xs text-slate-500 uppercase tracking-wider">{s.v}</div></div>))}
    </motion.div>
  </header>
);

export const FeatureGrid: React.FC<{ features: { icon: string; title: string; body: string }[] }> = ({ features }) => (
  <section className="max-w-6xl mx-auto px-5 sm:px-10 py-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
    {features.map((f, i) => (
      <motion.div key={f.title} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} custom={i}
        className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 hover:border-accent/40 transition-colors">
        <div className="text-3xl mb-3">{f.icon}</div>
        <h3 className="font-bold text-white mb-1.5">{f.title}</h3>
        <p className="text-sm text-slate-400 leading-relaxed">{f.body}</p>
      </motion.div>
    ))}
  </section>
);

/** Generic signature widget: textarea -> RAG (askChat with buildPrompt) -> markdown result. */
export const AnalyzerWidget: React.FC<{
  id: string; accent: string; chatId: () => string; heading: string; sub: string; label: string;
  placeholder: string; sample: string; buttonLabel: string; buildPrompt: (text: string) => string;
}> = ({ id, accent, chatId, heading, sub, label, placeholder, sample, buttonLabel, buildPrompt }) => {
  const [text, setText] = useState(sample);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState('');
  const run = async () => {
    if (!text.trim() || busy) return;
    setBusy(true); setResult('');
    try { const out = await askChat(chatId(), buildPrompt(text.trim())); setResult(out.answer || 'No response.'); }
    catch { setResult('Sorry — the assistant is unavailable right now.'); }
    setBusy(false);
  };
  return (
    <section id={id} className="max-w-6xl mx-auto px-5 sm:px-10 py-16">
      <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-6">
        <h2 className="text-3xl font-black text-white">{heading}</h2>
        <p className="text-slate-400 mt-2">{sub}</p>
      </motion.div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <label className="text-sm font-semibold text-slate-300">{label}</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder={placeholder}
            className="w-full mt-3 rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-sm outline-none focus:border-accent/50 resize-none" />
          <div className="flex gap-2 mt-3">
            <button onClick={() => setText(sample)} className="px-4 py-2.5 rounded-xl text-sm font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10">Use sample</button>
            <button onClick={run} disabled={busy || !text.trim()} className="flex-1 px-5 py-2.5 rounded-xl text-sm font-semibold text-[#05070a] disabled:opacity-50" style={{ backgroundColor: accent }}>{busy ? 'Analyzing…' : buttonLabel}</button>
          </div>
        </motion.div>
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 min-h-[260px]">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">Result</h3>
          {busy && <p className="text-sm text-slate-500">Working on it…</p>}
          {!busy && !result && <p className="text-sm text-slate-500">Paste your text (or use the sample) and run to see the result, grounded in your knowledge base.</p>}
          {!busy && result && <div className="text-sm text-slate-300 leading-relaxed"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{result}</ReactMarkdown></div>}
        </motion.div>
      </div>
    </section>
  );
};

export const VerticalChat: React.FC<{ id: string; accent: string; chatId: () => string; persona?: string; heading: string; sub: string; starters: string[]; placeholder: string }> = ({ id, chatId, persona, heading, sub, starters, placeholder }) => {
  const [msgs, setMsgs] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);
  const send = async (text?: string) => {
    const q = (typeof text === 'string' ? text : input).trim();
    if (!q || busy) return;
    setInput('');
    setMsgs((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    setBusy(true);
    try {
      await askChatStream(chatId(), q, {
        onChunk: (t) => setMsgs((m) => { const c = [...m]; c[c.length - 1] = { role: 'assistant', content: c[c.length - 1].content + t }; return c; }),
        onDone: () => setBusy(false),
        onError: () => { setMsgs((m) => { const c = [...m]; c[c.length - 1] = { role: 'assistant', content: 'Sorry — unavailable right now.' }; return c; }); setBusy(false); },
      }, { persona });
    } catch { setBusy(false); }
  };
  return (
    <section id={id} className="max-w-3xl mx-auto px-5 sm:px-10 py-16">
      <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-5 text-center">
        <h2 className="text-3xl font-black text-white">{heading}</h2>
        <p className="text-slate-400 mt-2">{sub}</p>
      </motion.div>
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] flex flex-col h-[440px]">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {msgs.length === 0 && (
            <div className="flex flex-wrap gap-2 justify-center pt-6">
              {starters.map((p) => (<button key={p} onClick={() => send(p)} className="px-3 py-2 rounded-xl text-xs bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20">{p}</button>))}
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={i} className={`rounded-2xl p-3 text-sm ${m.role === 'user' ? 'bg-white/10 ml-10' : 'bg-black/30 mr-10'}`}>
              {m.role === 'assistant' ? (m.content ? <ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{m.content}</ReactMarkdown> : <span className="text-slate-500">…</span>) : <span className="whitespace-pre-wrap">{m.content}</span>}
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <div className="p-3 border-t border-white/10 flex gap-2">
          <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder={placeholder}
            className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-2.5 text-sm outline-none focus:border-accent/50" />
          <button onClick={() => send()} disabled={busy} className="px-5 py-2.5 rounded-xl text-sm font-semibold text-accent bg-accent/15 border border-accent/30 hover:bg-accent/25 disabled:opacity-50">{busy ? '…' : 'Send'}</button>
        </div>
      </div>
    </section>
  );
};

export const Footer: React.FC<{ name: string; note?: string }> = ({ name, note }) => (
  <footer className="max-w-6xl mx-auto px-5 sm:px-10 py-10 text-center text-xs text-slate-600 border-t border-white/5">
    {name} · private on-prem AI · synthetic demo data{note ? ` · ${note}` : ''}
  </footer>
);
