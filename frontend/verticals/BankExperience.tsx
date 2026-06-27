import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  { icon: '💬', title: 'Advisory chat', body: 'Answers product, rate, and policy questions from your bank’s own materials — with citations.' },
  { icon: '📊', title: 'Rates & products', body: 'Deposit accounts, loans, fees and eligibility, always grounded in your current sheet.' },
  { icon: '🛡️', title: 'Compliance built-in', body: 'Surfaces required KYC/AML disclosures in the moment, so nothing is missed.' },
  { icon: '🔒', title: 'Never leaves the building', body: 'Runs fully on-prem on the DGX. Customer financial data stays inside your walls.' },
];
const STATS = [
  { k: '100%', v: 'on-premise' },
  { k: '0', v: 'data sent to cloud' },
  { k: '24/7', v: 'private copilot' },
];
const LOAN_TYPES = ['Personal', 'Auto', 'Mortgage', 'Business'];

const BankExperience: React.FC = () => {
  const pack = resolvePack();
  const accent = pack?.accent ?? '#22c55e';
  const chatIdRef = useRef<string>('');

  useEffect(() => {
    createChat('Bank copilot').then((r) => { chatIdRef.current = r.chat_id; }).catch(() => {});
  }, []);

  const chatId = () => chatIdRef.current || 'bank-demo';

  // ── Loan & suitability wizard ─────────────────────────────────────────────
  const [step, setStep] = useState(0);
  const [loanType, setLoanType] = useState('Personal');
  const [amount, setAmount] = useState('25000');
  const [term, setTerm] = useState('36');
  const [income, setIncome] = useState('80000');
  const [wizBusy, setWizBusy] = useState(false);
  const [wizResult, setWizResult] = useState('');

  const runWizard = async () => {
    setWizBusy(true); setWizResult('');
    const prompt =
      `A customer wants a ${loanType} loan of $${amount} over ${term} months; annual income $${income}. ` +
      `Using our products, rates, and suitability guidelines, give: (1) the best-fit product with its rate/APR and fees, ` +
      `(2) a brief eligibility & suitability assessment, (3) the required disclosures to present. Be concise and use a short table for the product terms.`;
    try {
      const out = await askChat(chatId(), prompt);
      setWizResult(out.answer || 'No response.');
    } catch (e: any) {
      setWizResult('Sorry — the copilot is unavailable right now.');
    }
    setWizBusy(false);
  };

  // ── Copilot chat ──────────────────────────────────────────────────────────
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
    } catch (_) { setChatBusy(false); }
  };

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div className="h-full w-full overflow-y-auto bg-[#05070a] text-slate-200" style={{ height: '100dvh' }}>
      {/* animated ambient background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <motion.div
          className="absolute -top-40 -right-40 w-[640px] h-[640px] rounded-full blur-[140px]"
          style={{ backgroundColor: 'var(--glow-1)' }}
          animate={{ scale: [1, 1.15, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute -bottom-40 -left-40 w-[560px] h-[560px] rounded-full blur-[140px]"
          style={{ backgroundColor: 'var(--glow-2)' }}
          animate={{ scale: [1.1, 1, 1.1], opacity: [0.6, 0.9, 0.6] }}
          transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      <div className="relative z-10">
        {/* nav */}
        <nav className="flex items-center justify-between px-5 sm:px-10 py-4 max-w-6xl mx-auto">
          <div className="flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-xl flex items-center justify-center text-[#05070a] font-bold" style={{ backgroundColor: accent }}>E</span>
            <span className="font-bold text-white">{pack?.name ?? 'EchoMind Bank'}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <button onClick={() => scrollTo('wizard')} className="hidden sm:inline px-3 py-1.5 rounded-lg text-slate-300 hover:text-white">Suitability check</button>
            <button onClick={() => scrollTo('copilot')} className="px-3 py-1.5 rounded-lg font-semibold text-accent bg-accent/10 border border-accent/30 hover:bg-accent/20">Ask the copilot</button>
          </div>
        </nav>

        {/* hero */}
        <header className="max-w-6xl mx-auto px-5 sm:px-10 pt-14 sm:pt-24 pb-16 sm:pb-24">
          <motion.p variants={fadeUp} initial="hidden" animate="show" className="text-accent font-semibold tracking-widest uppercase text-xs mb-4">Private banking intelligence</motion.p>
          <motion.h1 variants={fadeUp} initial="hidden" animate="show" custom={1} className="text-4xl sm:text-6xl font-black text-white leading-[1.05] max-w-3xl">
            Your banking copilot, with compliance <span style={{ color: accent }}>built in</span>.
          </motion.h1>
          <motion.p variants={fadeUp} initial="hidden" animate="show" custom={2} className="mt-6 text-lg text-slate-400 max-w-2xl">
            Answer product, rate, and policy questions and surface required disclosures in the moment — grounded in your bank’s own materials, running entirely on-premise.
          </motion.p>
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={3} className="mt-9 flex flex-wrap gap-3">
            <button onClick={() => scrollTo('copilot')} className="px-6 py-3 rounded-xl font-semibold text-[#05070a]" style={{ backgroundColor: accent }}>Try the copilot</button>
            <button onClick={() => scrollTo('wizard')} className="px-6 py-3 rounded-xl font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10">Run a suitability check</button>
          </motion.div>
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={4} className="mt-14 flex flex-wrap gap-8">
            {STATS.map((s) => (
              <div key={s.v}>
                <div className="text-3xl font-black" style={{ color: accent }}>{s.k}</div>
                <div className="text-xs text-slate-500 uppercase tracking-wider">{s.v}</div>
              </div>
            ))}
          </motion.div>
        </header>

        {/* features */}
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

        {/* signature: loan & suitability wizard */}
        <section id="wizard" className="max-w-6xl mx-auto px-5 sm:px-10 py-16">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-6">
            <h2 className="text-3xl font-black text-white">Loan eligibility & suitability — in seconds</h2>
            <p className="text-slate-400 mt-2">Answers from your real products, rates, and compliance guidelines.</p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <div className="flex items-center gap-2 mb-5">
                {[0, 1, 2, 3].map((s) => (
                  <div key={s} className="flex-1 h-1.5 rounded-full" style={{ backgroundColor: s <= step ? accent : 'rgba(255,255,255,0.08)' }} />
                ))}
              </div>
              <AnimatePresence mode="wait">
                <motion.div key={step} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} transition={{ duration: 0.25 }}>
                  {step === 0 && (
                    <div>
                      <label className="text-sm font-semibold text-slate-300">Loan type</label>
                      <div className="grid grid-cols-2 gap-2 mt-3">
                        {LOAN_TYPES.map((t) => (
                          <button key={t} onClick={() => setLoanType(t)}
                            className={`py-3 rounded-xl border text-sm font-medium transition-colors ${loanType === t ? 'bg-accent/15 text-accent border-accent/40' : 'bg-white/5 text-slate-400 border-white/10 hover:text-white'}`}>{t}</button>
                        ))}
                      </div>
                    </div>
                  )}
                  {step === 1 && (
                    <div>
                      <label className="text-sm font-semibold text-slate-300">Loan amount (USD)</label>
                      <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} className="w-full mt-3 rounded-xl bg-black/30 border border-white/10 px-4 py-3 outline-none focus:border-accent/50" />
                    </div>
                  )}
                  {step === 2 && (
                    <div>
                      <label className="text-sm font-semibold text-slate-300">Term (months)</label>
                      <input type="number" value={term} onChange={(e) => setTerm(e.target.value)} className="w-full mt-3 rounded-xl bg-black/30 border border-white/10 px-4 py-3 outline-none focus:border-accent/50" />
                    </div>
                  )}
                  {step === 3 && (
                    <div>
                      <label className="text-sm font-semibold text-slate-300">Annual income (USD)</label>
                      <input type="number" value={income} onChange={(e) => setIncome(e.target.value)} className="w-full mt-3 rounded-xl bg-black/30 border border-white/10 px-4 py-3 outline-none focus:border-accent/50" />
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
              <div className="flex justify-between mt-6">
                <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0} className="px-4 py-2 rounded-xl text-sm text-slate-400 hover:text-white disabled:opacity-30">Back</button>
                {step < 3 ? (
                  <button onClick={() => setStep((s) => s + 1)} className="px-5 py-2 rounded-xl text-sm font-semibold text-[#05070a]" style={{ backgroundColor: accent }}>Next</button>
                ) : (
                  <button onClick={runWizard} disabled={wizBusy} className="px-5 py-2 rounded-xl text-sm font-semibold text-[#05070a] disabled:opacity-50" style={{ backgroundColor: accent }}>{wizBusy ? 'Assessing…' : 'Assess suitability'}</button>
                )}
              </div>
            </motion.div>

            <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 min-h-[260px]">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">Assessment</h3>
              {wizBusy && <p className="text-sm text-slate-500">Checking products, rates, and suitability rules…</p>}
              {!wizBusy && !wizResult && <p className="text-sm text-slate-500">Fill in the details and run the check to see the best-fit product, eligibility, and the disclosures to present.</p>}
              {!wizBusy && wizResult && (
                <div className="text-sm text-slate-300 leading-relaxed"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{wizResult}</ReactMarkdown></div>
              )}
            </motion.div>
          </div>
        </section>

        {/* copilot chat */}
        <section id="copilot" className="max-w-3xl mx-auto px-5 sm:px-10 py-16">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-5 text-center">
            <h2 className="text-3xl font-black text-white">Ask the banking copilot</h2>
            <p className="text-slate-400 mt-2">Grounded in your bank’s materials. Try a starter question.</p>
          </motion.div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] flex flex-col h-[440px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {msgs.length === 0 && (
                <div className="flex flex-wrap gap-2 justify-center pt-6">
                  {['What deposit accounts and rates do we offer?', 'What KYC disclosures are required for a new account?', 'Compare our personal vs auto loan rates'].map((p) => (
                    <button key={p} onClick={() => send(p)} className="px-3 py-2 rounded-xl text-xs bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20">{p}</button>
                  ))}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`rounded-2xl p-3 text-sm ${m.role === 'user' ? 'bg-white/10 ml-10' : 'bg-black/30 mr-10'}`}>
                  {m.role === 'assistant'
                    ? (m.content ? <ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{m.content}</ReactMarkdown> : <span className="text-slate-500">…</span>)
                    : <span className="whitespace-pre-wrap">{m.content}</span>}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            <div className="p-3 border-t border-white/10 flex gap-2">
              <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder="Ask about products, rates, disclosures…"
                className="flex-1 rounded-xl bg-black/30 border border-white/10 px-4 py-2.5 text-sm outline-none focus:border-accent/50" />
              <button onClick={() => send()} disabled={chatBusy} className="px-5 py-2.5 rounded-xl text-sm font-semibold text-accent bg-accent/15 border border-accent/30 hover:bg-accent/25 disabled:opacity-50">{chatBusy ? '…' : 'Send'}</button>
            </div>
          </div>
        </section>

        <footer className="max-w-6xl mx-auto px-5 sm:px-10 py-10 text-center text-xs text-slate-600 border-t border-white/5">
          {pack?.name ?? 'EchoMind Bank'} · private on-prem AI · synthetic demo data
        </footer>
      </div>
    </div>
  );
};

export default BankExperience;
