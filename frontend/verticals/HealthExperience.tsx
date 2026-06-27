import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { resolvePack } from '../packs';
import { askChat, askChatStream, createChat } from '../services/backend';

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] } }),
};

const md = {
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

const FEATURES = [
  { icon: '🩺', title: 'Ambient scribe', body: 'Turns a visit into a structured SOAP note — the clinician stays present with the patient.' },
  { icon: '💊', title: 'Formulary at hand', body: 'First-line treatments, dosing, and screening schedules answered from your own protocols.' },
  { icon: '⚠️', title: 'Safety net', body: 'Flags drug interactions and contraindications in the moment, grounded in your formulary.' },
  { icon: '🔒', title: 'HIPAA by design', body: 'Runs fully on-prem on the DGX. No PHI ever leaves the building — that’s the moat.' },
];
const STATS = [
  { k: '100%', v: 'on-premise' },
  { k: '0', v: 'PHI to the cloud' },
  { k: 'HIPAA', v: 'private by design' },
];

const HealthExperience: React.FC = () => {
  const pack = resolvePack();
  const accent = pack?.accent ?? '#10b981';
  const chatIdRef = useRef<string>('');
  useEffect(() => { createChat('Clinical copilot').then((r) => { chatIdRef.current = r.chat_id; }).catch(() => {}); }, []);
  const chatId = () => chatIdRef.current || 'health-demo';

  // ── Signature: drug interaction & dosing checker ──────────────────────────
  const [drugs, setDrugs] = useState<string[]>(['Lisinopril', 'Spironolactone']);
  const [drugInput, setDrugInput] = useState('');
  const [checkBusy, setCheckBusy] = useState(false);
  const [checkResult, setCheckResult] = useState('');
  const addDrug = () => { const d = drugInput.trim(); if (d && !drugs.includes(d)) { setDrugs([...drugs, d]); setDrugInput(''); } };
  const removeDrug = (d: string) => setDrugs(drugs.filter((x) => x !== d));

  const runCheck = async () => {
    if (drugs.length < 1 || checkBusy) return;
    setCheckBusy(true); setCheckResult('');
    const prompt =
      `For these medications: ${drugs.join(', ')}. Using our clinic formulary, list any clinically significant ` +
      `drug-drug interactions as a table (Drug A | Drug B | Effect | Action), and note relevant first-line dosing. ` +
      `If none are significant, say so. Be concise.`;
    try { const out = await askChat(chatId(), prompt); setCheckResult(out.answer || 'No response.'); }
    catch { setCheckResult('Sorry — the assistant is unavailable right now.'); }
    setCheckBusy(false);
  };

  // ── Clinical copilot chat ─────────────────────────────────────────────────
  const [msgs, setMsgs] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);
  const send = async (text?: string) => {
    const q = (typeof text === 'string' ? text : input).trim();
    if (!q || chatBusy) return;
    setInput('');
    setMsgs((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    setChatBusy(true);
    try {
      await askChatStream(chatId(), q, {
        onChunk: (t) => setMsgs((m) => { const c = [...m]; c[c.length - 1] = { role: 'assistant', content: c[c.length - 1].content + t }; return c; }),
        onDone: () => setChatBusy(false),
        onError: () => { setMsgs((m) => { const c = [...m]; c[c.length - 1] = { role: 'assistant', content: 'Sorry — unavailable right now.' }; return c; }); setChatBusy(false); },
      }, { persona: pack?.persona });
    } catch { setChatBusy(false); }
  };
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div className="h-full w-full overflow-y-auto bg-[#05070a] text-slate-200" style={{ height: '100dvh' }}>
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <motion.div className="absolute -top-40 -right-40 w-[640px] h-[640px] rounded-full blur-[140px]" style={{ backgroundColor: 'var(--glow-1)' }}
          animate={{ scale: [1, 1.15, 1], opacity: [0.7, 1, 0.7] }} transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.div className="absolute -bottom-40 -left-40 w-[560px] h-[560px] rounded-full blur-[140px]" style={{ backgroundColor: 'var(--glow-2)' }}
          animate={{ scale: [1.1, 1, 1.1], opacity: [0.6, 0.9, 0.6] }} transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }} />
      </div>

      <div className="relative z-10">
        <nav className="flex items-center justify-between px-5 sm:px-10 py-4 max-w-6xl mx-auto">
          <div className="flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-xl flex items-center justify-center text-[#05070a] font-bold" style={{ backgroundColor: accent }}>E</span>
            <span className="font-bold text-white">{pack?.name ?? 'EchoMind Health'}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <button onClick={() => scrollTo('checker')} className="hidden sm:inline px-3 py-1.5 rounded-lg text-slate-300 hover:text-white">Interaction checker</button>
            <button onClick={() => scrollTo('copilot')} className="px-3 py-1.5 rounded-lg font-semibold text-accent bg-accent/10 border border-accent/30 hover:bg-accent/20">Ask the assistant</button>
          </div>
        </nav>

        <header className="max-w-6xl mx-auto px-5 sm:px-10 pt-14 sm:pt-24 pb-16 sm:pb-24">
          <motion.p variants={fadeUp} initial="hidden" animate="show" className="text-accent font-semibold tracking-widest uppercase text-xs mb-4">Private clinical intelligence</motion.p>
          <motion.h1 variants={fadeUp} initial="hidden" animate="show" custom={1} className="text-4xl sm:text-6xl font-black text-white leading-[1.05] max-w-3xl">
            Your clinic’s AI — nothing leaves the <span style={{ color: accent }}>building</span>.
          </motion.h1>
          <motion.p variants={fadeUp} initial="hidden" animate="show" custom={2} className="mt-6 text-lg text-slate-400 max-w-2xl">
            An ambient scribe and clinical safety net — protocols, formulary, dosing, and interaction checks answered from your own knowledge base, fully on-premise and HIPAA-private.
          </motion.p>
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={3} className="mt-9 flex flex-wrap gap-3">
            <button onClick={() => scrollTo('checker')} className="px-6 py-3 rounded-xl font-semibold text-[#05070a]" style={{ backgroundColor: accent }}>Check an interaction</button>
            <button onClick={() => scrollTo('copilot')} className="px-6 py-3 rounded-xl font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10">Ask the assistant</button>
          </motion.div>
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={4} className="mt-14 flex flex-wrap gap-8">
            {STATS.map((s) => (
              <div key={s.v}><div className="text-3xl font-black" style={{ color: accent }}>{s.k}</div><div className="text-xs text-slate-500 uppercase tracking-wider">{s.v}</div></div>
            ))}
          </motion.div>
        </header>

        <section className="max-w-6xl mx-auto px-5 sm:px-10 py-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f, i) => (
            <motion.div key={f.title} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} custom={i}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 hover:border-accent/40 transition-colors">
              <div className="text-3xl mb-3">{f.icon}</div>
              <h3 className="font-bold text-white mb-1.5">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.body}</p>
            </motion.div>
          ))}
        </section>

        {/* signature: drug interaction & dosing checker */}
        <section id="checker" className="max-w-6xl mx-auto px-5 sm:px-10 py-16">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-6">
            <h2 className="text-3xl font-black text-white">Drug interaction & dosing checker</h2>
            <p className="text-slate-400 mt-2">Add medications and check against your clinic formulary in seconds.</p>
          </motion.div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <label className="text-sm font-semibold text-slate-300">Medications</label>
              <div className="flex flex-wrap gap-2 mt-3 mb-3">
                {drugs.map((d) => (
                  <span key={d} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm bg-accent/15 text-accent border border-accent/30">
                    {d}<button onClick={() => removeDrug(d)} className="text-accent/70 hover:text-white">×</button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input value={drugInput} onChange={(e) => setDrugInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addDrug()} placeholder="Add a medication…"
                  className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-2.5 text-sm outline-none focus:border-accent/50" />
                <button onClick={addDrug} className="px-4 py-2.5 rounded-xl text-sm font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10">Add</button>
              </div>
              <button onClick={runCheck} disabled={checkBusy || drugs.length === 0} className="mt-5 w-full px-5 py-3 rounded-xl text-sm font-semibold text-[#05070a] disabled:opacity-50" style={{ backgroundColor: accent }}>
                {checkBusy ? 'Checking formulary…' : 'Check interactions & dosing'}
              </button>
            </motion.div>
            <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 min-h-[260px]">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">Result</h3>
              {checkBusy && <p className="text-sm text-slate-500">Checking your formulary for interactions…</p>}
              {!checkBusy && !checkResult && <p className="text-sm text-slate-500">Add medications and run the check to see interactions, effects, and recommended actions from your formulary.</p>}
              {!checkBusy && checkResult && <div className="text-sm text-slate-300 leading-relaxed"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{checkResult}</ReactMarkdown></div>}
            </motion.div>
          </div>
        </section>

        <section id="copilot" className="max-w-3xl mx-auto px-5 sm:px-10 py-16">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-5 text-center">
            <h2 className="text-3xl font-black text-white">Ask the clinical assistant</h2>
            <p className="text-slate-400 mt-2">Grounded in your protocols & formulary. Decision support, not a substitute for judgment.</p>
          </motion.div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] flex flex-col h-[440px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {msgs.length === 0 && (
                <div className="flex flex-wrap gap-2 justify-center pt-6">
                  {['First-line treatment for stage 1 hypertension?', 'Adult colorectal screening schedule?', 'Summarize the latest patient visit note'].map((p) => (
                    <button key={p} onClick={() => send(p)} className="px-3 py-2 rounded-xl text-xs bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20">{p}</button>
                  ))}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`rounded-2xl p-3 text-sm ${m.role === 'user' ? 'bg-white/10 ml-10' : 'bg-black/30 mr-10'}`}>
                  {m.role === 'assistant' ? (m.content ? <ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{m.content}</ReactMarkdown> : <span className="text-slate-500">…</span>) : <span className="whitespace-pre-wrap">{m.content}</span>}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            <div className="p-3 border-t border-white/10 flex gap-2">
              <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder="Ask about protocols, dosing, screening…"
                className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-2.5 text-sm outline-none focus:border-accent/50" />
              <button onClick={() => send()} disabled={chatBusy} className="px-5 py-2.5 rounded-xl text-sm font-semibold text-accent bg-accent/15 border border-accent/30 hover:bg-accent/25 disabled:opacity-50">{chatBusy ? '…' : 'Send'}</button>
            </div>
          </div>
        </section>

        <footer className="max-w-6xl mx-auto px-5 sm:px-10 py-10 text-center text-xs text-slate-600 border-t border-white/5">
          {pack?.name ?? 'EchoMind Health'} · private on-prem AI · synthetic demo data · not medical advice
        </footer>
      </div>
    </div>
  );
};

export default HealthExperience;
