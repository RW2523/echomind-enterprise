import React from 'react';
import { resolvePack } from '../packs';
import { Ambient, Nav, Hero, FeatureGrid, AnalyzerWidget, VerticalChat, Footer, useChatId } from './_shared';

const FEATURES = [
  { icon: '📚', title: 'Cited research', body: 'Answers over your firm’s contracts and precedents, with clause-level citations.' },
  { icon: '✍️', title: 'Redline & review', body: 'Flags risky clauses and proposes standard fallback positions from your playbook.' },
  { icon: '⚠️', title: 'Risk radar', body: 'Surfaces missing terms, one-sided clauses, and conflicts before they cost you.' },
  { icon: '🔒', title: 'Privilege preserved', body: 'Runs fully on-prem — client matter and work product never leave the firm.' },
];
const STATS = [{ k: '100%', v: 'on-premise' }, { k: '0', v: 'client data to cloud' }, { k: 'privilege', v: 'preserved' }];
const SAMPLE = `8. Limitation of Liability. Each party's total liability shall not exceed the greater of USD 250,000 or fees paid in the prior 12 months. Neither party is liable for indirect or consequential damages.
9. Indemnification. Provider shall indemnify Client against third-party IP claims.
12. Termination. Either party may terminate for convenience on 30 days' notice.`;

const LawExperience: React.FC = () => {
  const pack = resolvePack();
  const accent = pack?.accent ?? '#6366f1';
  const chatId = useChatId('Legal associate');
  return (
    <div className="h-full w-full overflow-y-auto bg-[#05070a] text-slate-200" style={{ height: '100dvh' }}>
      <Ambient />
      <div className="relative z-10">
        <Nav name={pack?.name ?? 'EchoMind Law'} accent={accent}
          links={[{ label: 'Clause analyzer', id: 'analyzer' }, { label: 'Ask the associate', id: 'copilot', primary: true }]} />
        <Hero accent={accent} eyebrow="Private legal intelligence"
          titleA="Privileged AI that never leaves your " titleHi="firm" titleB="."
          sub="Research, summarize, and redline — grounded in your firm’s contracts and playbooks, with privilege preserved on a box in your office."
          ctas={[{ label: 'Analyze a clause', id: 'analyzer', primary: true }, { label: 'Ask the associate', id: 'copilot' }]} stats={STATS} />
        <FeatureGrid features={FEATURES} />
        <AnalyzerWidget id="analyzer" accent={accent} chatId={chatId}
          heading="Contract clause analyzer" sub="Paste a clause or excerpt; get risk flags and standard positions from your playbook."
          label="Contract excerpt" placeholder="Paste contract text…" sample={SAMPLE} buttonLabel="Flag risks & positions"
          buildPrompt={(t) => `Review this contract excerpt using our contract-review playbook. Produce a 'Risk Flags' table with columns Clause | Issue | Why it matters | Suggested position. Add the note: 'Informational analysis, not legal advice.'\n\nExcerpt:\n${t}`} />
        <VerticalChat id="copilot" accent={accent} chatId={chatId} persona={pack?.persona}
          heading="Ask the legal associate" sub="Grounded in your contracts & playbooks. Informational, not legal advice."
          starters={['Summarize the Master Services Agreement', 'What is the limitation of liability?', 'Flag the risky clauses to watch for']}
          placeholder="Ask about clauses, risks, positions…" />
        <Footer name={pack?.name ?? 'EchoMind Law'} note="not legal advice" />
      </div>
    </div>
  );
};
export default LawExperience;
