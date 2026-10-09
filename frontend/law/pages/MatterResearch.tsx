import React, { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ApiError, postStream } from '../api/client';
import { keys, useResearchHistory } from '../api/queries';
import type { ResearchSource, ResearchTurn } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

type Phase = 'idle' | 'reading' | 'writing' | 'done' | 'abstained' | 'revoked' | 'error';
interface Answer { question: string; text: string; sources: ResearchSource[]; cited: number[]; phase: Phase; uncited?: boolean }

const pages = (s: ResearchSource) => (s.page_start ? (s.page_end && s.page_end !== s.page_start ? `pp. ${s.page_start}–${s.page_end}` : `p. ${s.page_start}`) : '');
const plain = (t: string) => t.replace(/(^|\s)#{1,6}\s+/g, '$1');
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** Research: questions answered only from this matter's documents, each statement tied to the
 *  clause it relies on. Laid out as a working note, not a chat. */
export default function MatterResearch() {
  const matter = useMatter();
  const qc = useQueryClient();
  const history = useResearchHistory(matter.id);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [error, setError] = useState('');
  const [focus, setFocus] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const busy = answer?.phase === 'reading' || answer?.phase === 'writing';

  const ask = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setError(''); setFocus(null);
    let state: Answer = { question: q, text: '', sources: [], cited: [], phase: 'reading' };
    const update = (patch: Partial<Answer>) => { state = { ...state, ...patch }; setAnswer(state); };
    update({});
    try {
      await postStream(`/api/legal/matters/${encodeURIComponent(matter.id)}/research/query`, { question: q }, (ev) => {
        if (ev.type === 'sources') update({ sources: ev.citations ?? [] });
        else if (ev.type === 'chunk') update({ text: state.text + (ev.text ?? ''), phase: 'writing' });
        else if (ev.type === 'done') update({ text: ev.answer ?? state.text, cited: (ev.citations ?? []).map((c: ResearchSource) => c.n),
          phase: ev.abstained ? 'abstained' : 'done', uncited: !!ev.uncited });
        else if (ev.type === 'scope_revoked') update({ text: '', sources: [], phase: 'revoked' });
        else if (ev.type === 'error') update({ phase: 'error' });
      }, ctrl.signal);
      setQuestion('');
      await qc.invalidateQueries({ queryKey: keys.research(matter.id) });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setAnswer(null);
      setError(err instanceof ApiError ? (err.status === 404 ? 'This matter is no longer available to you.' : err.message) : 'The question could not be sent.');
    }
  };

  const show = (t: ResearchTurn) => {
    setFocus(null);
    setAnswer({ question: t.question, text: t.answer ?? '', sources: t.citations, cited: t.citations.map((c) => c.n),
      phase: t.status === 'answered' ? 'done' : t.status === 'abstained' ? 'abstained' : t.status === 'scope_revoked' ? 'revoked' : 'error' });
  };

  return (
    <>
      <Toolbar crumb={matter.number} title="Research" />
      <div className="lw-content lw-research">
        <div style={{ display: 'grid', gap: 16, alignContent: 'start', minWidth: 0 }}>
          <form onSubmit={ask} className="lw-paper lw-sheet lw-ask">
            <label htmlFor="research-q" className="lw-h2">Ask about this matter</label>
            <p className="lw-meta" style={{ margin: '2px 0 10px' }}>Answers come only from this matter’s documents, with the passage each statement relies on.</p>
            <textarea id="research-q" rows={2} className="lw-input lw-serif" placeholder="e.g. What caps the supplier’s liability, and what is excluded?"
              value={question} onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask(e as unknown as React.FormEvent); }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
              <span className="lw-meta">⌘/Ctrl + Enter</span>
              <button type="submit" className="lw-btn lw-btn-primary" disabled={!question.trim() || busy}>{busy ? 'Working…' : 'Ask'}</button>
            </div>
          </form>
          {error && <ErrorNote>{error}</ErrorNote>}
          {answer && <AnswerSheet a={answer} focus={focus} onFocus={setFocus} />}
        </div>
        <aside className="lw-paper lw-research-history" aria-label="Your earlier questions">
          <div className="lw-card-h"><h2 className="lw-h2">Your earlier questions</h2></div>
          {history.isPending ? <div style={{ padding: 16 }}><Skeleton /></div>
            : !history.data?.length ? <p className="lw-meta" style={{ padding: '12px 20px', margin: 0 }}>Only you can see your questions.</p>
            : (
              <ol className="lw-history-list">
                {history.data.map((t) => (
                  <li key={t.id}>
                    <button type="button" className="lw-history-item" onClick={() => show(t)}>
                      <span className="lw-ellipsis">{t.question}</span>
                      <span className="lw-meta">{when(t.created_at)}{t.status === 'abstained' ? ' · not answered by the documents' : t.status === 'scope_revoked' ? ' · access ended' : ''}</span>
                    </button>
                  </li>
                ))}
              </ol>
            )}
        </aside>
      </div>
    </>
  );
}

function AnswerSheet({ a, focus, onFocus }: { a: Answer; focus: number | null; onFocus: (n: number) => void }) {
  const cited = a.sources.filter((s) => a.cited.includes(s.n));
  const consulted = a.sources.filter((s) => !a.cited.includes(s.n));
  return (
    <article className="lw-paper lw-sheet lw-answer" aria-label="Answer" aria-busy={a.phase === 'reading' || a.phase === 'writing'}>
      <div className="lw-eyebrow">Question</div>
      <h2 className="lw-serif lw-answer-q">{a.question}</h2>
      <div aria-live="polite">
        {a.phase === 'reading' && <p className="lw-meta">Reading this matter’s documents…</p>}
        {a.phase === 'revoked' && <p role="alert" className="lw-answer-text">Your access to this matter ended while the answer was being written. Nothing more is shown.</p>}
        {a.phase === 'error' && <p role="alert" className="lw-answer-text">The answer could not be completed. Try again.</p>}
        {a.phase === 'abstained' && (
          <div className="lw-answer-text">
            <p style={{ margin: 0 }}>The documents in this matter do not answer that.</p>
            <p className="lw-meta" style={{ margin: '6px 0 0', fontFamily: 'var(--sans)' }}>Nothing in them closely matches. Try other words, or check that the document you need is imported.</p>
          </div>
        )}
        {(a.phase === 'writing' || a.phase === 'done') && (
          <p className="lw-serif lw-answer-text">{withCitations(plain(a.text), a.sources, onFocus)}</p>
        )}
        {a.phase === 'done' && a.uncited && <StatusPill tone="attention">No source cited — check before relying on it</StatusPill>}
      </div>
      {cited.length > 0 && (
        <section aria-label="Sources" style={{ marginTop: 18 }}>
          <h3 className="lw-eyebrow" style={{ margin: '0 0 8px' }}>Sources</h3>
          <ol className="lw-sources">{cited.map((s) => <SourceItem key={s.n} s={s} active={focus === s.n} />)}</ol>
        </section>
      )}
      {a.phase !== 'revoked' && consulted.length > 0 && (a.phase === 'done' || a.phase === 'abstained') && (
        <details style={{ marginTop: 12 }}>
          <summary className="lw-meta" style={{ cursor: 'pointer' }}>Also consulted ({consulted.length})</summary>
          <ol className="lw-sources" style={{ marginTop: 8 }}>{consulted.map((s) => <SourceItem key={s.n} s={s} active={false} />)}</ol>
        </details>
      )}
    </article>
  );
}

function withCitations(text: string, sources: ResearchSource[], onFocus: (n: number) => void): React.ReactNode {
  const known = new Set(sources.map((s) => s.n));
  return text.split(/(\[\d{1,2}\])/g).map((part, i) => {
    const m = /^\[(\d{1,2})\]$/.exec(part);
    if (m && known.has(Number(m[1]))) {
      const n = Number(m[1]);
      return <button key={i} type="button" className="lw-cite" aria-label={`Source ${n}`} onClick={() => onFocus(n)}>{n}</button>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

function SourceItem({ s, active }: { s: ResearchSource; active: boolean }) {
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => { if (active) ref.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [active]);
  return (
    <li ref={ref} className="lw-source" data-active={active || undefined}>
      <span className="lw-cite lw-cite-static" aria-hidden="true">{s.n}</span>
      <div style={{ minWidth: 0 }}>
        <blockquote className="lw-serif">{plain(s.text)}</blockquote>
        <div className="lw-meta">{s.document_title} · v{s.version_number}{pages(s) ? ` · ${pages(s)}` : ''}</div>
      </div>
    </li>
  );
}
