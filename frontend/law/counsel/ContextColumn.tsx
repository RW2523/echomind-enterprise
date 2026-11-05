/**
 * The column beside the conversation (owner's request 2026-11-05, the "Context · Sources · Actions" panel of the attached UI).
 * Nothing in it is generated: Context is the matter, its latest documents, questions worth asking (found by rules in its records
 * when the Council is on, otherwise the usual starters) and the way into a voice conversation; Sources are the passages this
 * conversation's answers cite, each reopened and re-verified by the server; Actions are what waits for the person, the
 * deadlines no one has confirmed and the inconsistencies still open. A part the person may not see is left out.
 */
import React, { useId, useState } from 'react';
import { Link } from 'react-router';
import { ArrowUpRight, AudioLines, ChevronRight, FileText, Lightbulb, Mic } from 'lucide-react';
import { usePanelReviews } from '../api/queries';
import type { Matter } from '../api/types';
import { useSession } from '../app/session';
import { usePanel } from './context';
import { Deadlines, Issues, LatestDocuments, WaitingForYou } from './MatterGlance';
import type { CitationRef } from './types';

type Tab = 'context' | 'sources' | 'actions';
const TABS: { key: Tab; label: string }[] = [{ key: 'context', label: 'Context' }, { key: 'sources', label: 'Sources' }, { key: 'actions', label: 'Actions' }];

export default function ContextColumn({ matter, several, sources, starters, onAsk, onVoice }: {
  matter: Matter | null; several: boolean; sources: CitationRef[]; starters: string[];
  onAsk: (q: string) => void; onVoice?: () => void;
}) {
  const [tab, setTab] = useState<Tab>('context');
  const id = useId();
  return (
    <aside className="qc-ctx" aria-label="Matter context">
      <div className="qc-ctx-tabs" role="tablist" aria-label="Beside the conversation">
        {TABS.map((t) => (
          <button key={t.key} type="button" role="tab" id={`${id}-${t.key}`} aria-selected={tab === t.key} aria-controls={`${id}-${t.key}-panel`}
            tabIndex={tab === t.key ? 0 : -1} className="qc-ctx-tab" onClick={() => setTab(t.key)}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
              const i = TABS.findIndex((x) => x.key === tab);
              const next = TABS[(i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length].key;
              setTab(next);
              document.getElementById(`${id}-${next}`)?.focus();
            }}>
            {t.label}{t.key === 'sources' && sources.length > 0 && <span className="qc-ctx-count" aria-label={`, ${sources.length}`}>{sources.length}</span>}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${id}-${tab}-panel`} aria-labelledby={`${id}-${tab}`} className="qc-ctx-body">
        {tab === 'context' && <ContextTab matter={matter} several={several} starters={starters} onAsk={onAsk} onVoice={onVoice} />}
        {tab === 'sources' && <SourcesTab sources={sources} />}
        {tab === 'actions' && (matter && !several ? <><WaitingForYou matter={matter} /><Deadlines matter={matter} /><Issues matter={matter} /></>
          : <p className="qc-meta qc-ctx-empty">{several ? 'Open one matter to see what waits for you on it.' : 'Choose a matter to see what waits for you on it.'}</p>)}
      </div>
    </aside>
  );
}

function ContextTab({ matter, several, starters, onAsk, onVoice }: {
  matter: Matter | null; several: boolean; starters: string[]; onAsk: (q: string) => void; onVoice?: () => void;
}) {
  const me = useSession();
  if (!matter || several) {
    return <p className="qc-meta qc-ctx-empty">{several ? 'Several matters are chosen: each answer says which one every source comes from.'
      : 'Choose a matter in the sidebar or the question box to see its documents and what waits for you.'}</p>;
  }
  const base = `/m/${encodeURIComponent(matter.number)}`;
  return (
    <>
      <section className="qc-card qc-ctx-card" aria-label="Matter">
        <div className="qc-card-head"><h3 className="qc-card-title">Matter</h3>
          <Link className="qc-icon-btn" to={`${base}/overview`} aria-label={`Open ${matter.number}`} title="Open the matter"><ArrowUpRight aria-hidden /></Link></div>
        <p className="qc-ctx-matter"><b>{matter.number}</b><span>{matter.title}</span><small>{matter.client.name}{matter.practice_area ? ` · ${matter.practice_area}` : ''}</small></p>
      </section>
      <LatestDocuments matter={matter} />
      {me.features?.panel ? <CouncilQuestions matter={matter} starters={starters} onAsk={onAsk} /> : <Starters starters={starters} onAsk={onAsk} />}
      {me.features?.conversation && onVoice && (
        <section className="qc-card qc-ctx-card qc-voice-card" aria-label="Talk by voice">
          <div className="qc-card-head"><h3 className="qc-card-title"><span className="qc-card-icon" aria-hidden><AudioLines /></span>Talk by voice</h3></div>
          <p className="qc-meta" style={{ margin: 0 }}>Speak your question; answers come from {matter.number}'s documents, are read aloud and saved in this chat.</p>
          <button type="button" className="qc-voice-orb" onClick={onVoice} aria-label={`Start a voice conversation about ${matter.number}`}><Mic aria-hidden /></button>
        </section>
      )}
    </>
  );
}

/** Questions found by rules in the matter's records (council/discovery.py); without any, the usual starters. */
function CouncilQuestions({ matter, starters, onAsk }: { matter: Matter; starters: string[]; onAsk: (q: string) => void }) {
  const q = usePanelReviews(matter.id);
  const found = q.data?.questions ?? [];
  if (!found.length) return <Starters starters={starters} onAsk={onAsk} />;
  const to = `/m/${encodeURIComponent(matter.number)}/panel`;
  return (
    <section className="qc-card qc-ctx-card" aria-label="Questions for the Council">
      <div className="qc-card-head"><h3 className="qc-card-title"><span className="qc-card-icon" aria-hidden><Lightbulb /></span>Questions for the Council</h3>
        <Link className="qc-linkbtn" to={to}>Review</Link></div>
      <ul className="qc-ctx-list">
        {found.map((x) => (
          <li key={x.id}><Link className="qc-ctx-row" to={to}><span>{x.title}</span><ChevronRight aria-hidden /></Link></li>
        ))}
      </ul>
      <p className="qc-meta" style={{ margin: 0 }}>Found in this matter's records; nothing runs until you convene one.</p>
    </section>
  );
}

function Starters({ starters, onAsk }: { starters: string[]; onAsk: (q: string) => void }) {
  return (
    <section className="qc-card qc-ctx-card" aria-label="Suggested questions">
      <div className="qc-card-head"><h3 className="qc-card-title"><span className="qc-card-icon" aria-hidden><Lightbulb /></span>Suggested questions</h3></div>
      <ul className="qc-ctx-list">
        {starters.map((s) => (
          <li key={s}><button type="button" className="qc-ctx-row" onClick={() => onAsk(s)}><span>{s}</span><ChevronRight aria-hidden /></button></li>
        ))}
      </ul>
    </section>
  );
}

function SourcesTab({ sources }: { sources: CitationRef[] }) {
  const panel = usePanel();
  if (!sources.length) return <p className="qc-meta qc-ctx-empty">The passages this conversation's answers cite appear here, each one opening at the cited words.</p>;
  return (
    <section className="qc-card qc-ctx-card" aria-label="Sources cited in this conversation">
      <ul className="qc-ctx-list">
        {sources.map((c) => (
          <li key={c.id}>
            <button type="button" className="qc-ctx-row qc-ctx-source" onClick={(e) => panel.open({ kind: 'citation', id: c.id }, e.currentTarget)}>
              <FileText aria-hidden />
              <span><b>{c.document_title}</b><small>{[c.version_number ? `version ${c.version_number}` : '', c.clause ? `clause ${c.clause}` : '',
                c.page ? `page ${c.page}` : ''].filter(Boolean).join(' · ')}</small></span>
              <ChevronRight aria-hidden />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
