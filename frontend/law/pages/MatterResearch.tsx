import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router';
import { Copy, Send } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError, postStream } from '../api/client';
import { keys, useResearchHistory } from '../api/queries';
import type { ResearchAmendment, ResearchConflict, ResearchEvidence, ResearchParagraph, ResearchSource, ResearchTurn, VersionPolicy } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { Starters } from '../ui/starters';
import { ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

type Phase = 'idle' | 'reading' | 'writing' | 'done' | 'abstained' | 'revoked' | 'error';
export interface Answer {
  question: string; text: string; sources: ResearchSource[]; cited: number[]; phase: Phase; uncited?: boolean; unsupported?: string[];
  failed?: string[]; paragraphs?: ResearchParagraph[]; evidence?: ResearchEvidence[]; conflicts?: ResearchConflict[];
  amendments?: ResearchAmendment[];
  status?: string | null; reason?: string | null; turnId?: string;
  asked?: string;                                        // S5.1b: the follow-up as asked, when it was made to stand alone
}

const pages = (s: { page_start?: number | null; page_end?: number | null }) =>
  (s.page_start ? (s.page_end && s.page_end !== s.page_start ? `pp. ${s.page_start}–${s.page_end}` : `p. ${s.page_start}`) : '');
const plain = (t: string) => t.replace(/(^|\s)#{1,6}\s+/g, '$1');
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const STATUS: Record<string, string> = { draft: 'Draft', final: 'Final', executed: 'Executed', received: 'Received' };
const VERIFIED: Record<ResearchEvidence['verification'], [string, 'verified' | 'attention' | 'conflict' | 'neutral']> = {
  verbatim: ['Quoted exactly', 'verified'],
  normalized: ['Quoted with small differences (spacing, quotes or case)', 'verified'],
  failed: ['Quote not found in the source', 'conflict'],
  not_quoted: ['No quote given — the closest passage is shown', 'neutral'],
};
const ABSTAINED: Record<string, string> = {
  no_passages: 'Nothing in this matter’s documents closely matches. Try other words, or check that the document you need is imported.',
  model: 'The passages found do not answer it. Ask more specifically, or read the passages it consulted below.',
  access_changed: 'This answer quoted a document you can no longer open, so it is not shown again.',
};

/** Research: questions answered only from this matter's documents, each statement tied to the exact
 *  words it relies on, checked before it is shown. Laid out as a working note, not a chat. */
/** Ask a question of the matter and follow the checked answer as it streams (S2.2; shared with S5.1). */
export async function streamAnswer(matterId: string, body: Record<string, unknown>, onChange: (a: Answer) => void, signal: AbortSignal): Promise<Answer> {
  let state: Answer = { question: String(body.question), text: '', sources: [], cited: [], phase: 'reading' };
  let checked = false;                                   // the server sends checked paragraphs (S2.2)
  const update = (patch: Partial<Answer>) => { state = { ...state, ...patch }; onChange(state); };
  update({});
  await postStream(`/api/legal/matters/${encodeURIComponent(matterId)}/research/query`, body, (ev) => {
    if (ev.type === 'understood') update(ev.rewritten ? { question: ev.question, asked: String(body.question) } : { question: ev.question });
    else if (ev.type === 'sources') update({ sources: ev.citations ?? [] });
    else if (ev.type === 'paragraph') {
      checked = true;
      const paragraphs = [...(state.paragraphs ?? []), ev as ResearchParagraph];
      update({ paragraphs, text: paragraphs.map((p) => p.text).join('\n\n'), phase: 'writing',
        unsupported: paragraphs.flatMap((p) => p.unsupported ?? []), failed: paragraphs.flatMap((p) => p.failed ?? []) });
    } else if (ev.type === 'chunk') { if (!checked) update({ text: state.text + (ev.text ?? ''), phase: 'writing' }); }
    else if (ev.type === 'done') update({ text: ev.answer ?? state.text, cited: (ev.citations ?? []).map((c: ResearchSource) => c.n),
      phase: ev.abstained ? 'abstained' : 'done', uncited: !!ev.uncited, unsupported: ev.unsupported ?? [], failed: ev.failed ?? [],
      evidence: ev.evidence ?? undefined, conflicts: ev.conflicts ?? [], status: ev.status ?? null, reason: ev.abstention_reason ?? null,
      turnId: ev.turn_id ?? undefined, amendments: ev.amendments ?? [] });
    else if (ev.type === 'scope_revoked') update({ text: '', sources: [], paragraphs: [], phase: 'revoked' });
    else if (ev.type === 'error') update({ phase: 'error' });
  }, signal);
  return state;
}

export default function MatterResearch() {
  const matter = useMatter();
  const qc = useQueryClient();
  const history = useResearchHistory(matter.id);
  const [question, setQuestion] = useState('');
  const [versions, setVersions] = useState<VersionPolicy>('signed');
  const [params, setParams] = useSearchParams();
  // S2.6: opened from an agenda item, research reads only that item's board papers until widened.
  const pack = (params.get('docs') ?? '').split(',').filter(Boolean);
  const about = params.get('about');
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
    try {
      await streamAnswer(matter.id, pack.length ? { question: q, versions, documents: pack } : { question: q, versions }, setAnswer, ctrl.signal);
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
    const done = ['answered', 'partial', 'unverified'].includes(t.status);
    setAnswer({ question: t.question, text: t.answer ?? '', sources: t.citations, cited: t.citations.map((c) => c.n), unsupported: t.unsupported ?? [],
      evidence: t.evidence?.length ? t.evidence : undefined, conflicts: t.conflicts ?? [], status: t.status, reason: t.abstention_reason, turnId: t.id,
      amendments: t.amendments ?? [],
      phase: done ? 'done' : t.status === 'abstained' ? 'abstained' : t.status === 'scope_revoked' ? 'revoked' : 'error' });
  };

  return (
    <>
      <Toolbar crumb={matter.number} title="Research" />
      <div className="lw-content lw-research">
        <div style={{ display: 'grid', gap: 16, alignContent: 'start', minWidth: 0 }}>
          <form onSubmit={ask} className="lw-paper lw-sheet lw-ask">
            <label htmlFor="research-q" className="lw-h2">Ask about this matter</label>
            <p className="lw-meta" style={{ margin: '2px 0 10px' }}>Answers come only from this matter’s documents, with the exact words each statement relies on.</p>
            {pack.length > 0 && (
              <p className="lw-pack" role="status">
                {`Only the board papers for “${about ?? 'this agenda item'}” (${pack.length} ${pack.length === 1 ? 'document' : 'documents'}). `}
                <button type="button" className="lw-linklike" onClick={() => setParams({}, { replace: true })}>Ask the whole matter instead</button>
              </p>
            )}
            <textarea id="research-q" rows={2} className="lw-input lw-serif" placeholder="e.g. What caps the supplier’s liability, and what is excluded?"
              value={question} onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask(e as unknown as React.FormEvent); }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
              <fieldset className="lw-segment" aria-label="Answer from">
                <legend className="lw-sr">Answer from</legend>
                {([['signed', 'Signed versions'], ['current', 'Latest drafts']] as [VersionPolicy, string][]).map(([v, label]) => (
                  <label key={v} data-on={versions === v || undefined}>
                    <input type="radio" name="versions" value={v} checked={versions === v} onChange={() => setVersions(v)} />{label}
                  </label>
                ))}
              </fieldset>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="lw-meta">⌘/Ctrl + Enter</span>
                <button type="submit" className="lw-btn lw-btn-primary" disabled={!question.trim() || busy}>{busy ? 'Working…' : 'Ask'}</button>
              </div>
            </div>
          </form>
          {!answer && !pack.length && (
            <Starters onPick={(q) => { setQuestion(q); document.getElementById('research-q')?.focus(); }} />
          )}
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
                      <span className="lw-meta">{when(t.created_at)}{t.status === 'abstained' ? ' · not answered by the documents'
                        : t.status === 'scope_revoked' ? ' · access ended' : t.status === 'partial' ? ' · some statements flagged'
                        : t.status === 'unverified' ? ' · not verified' : ''}</span>
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

export function AnswerSheet({ a, focus, onFocus }: { a: Answer; focus: number | null; onFocus: (n: number) => void }) {
  const matter = useMatter();
  const [copied, setCopied] = useState(false);
  const cited = a.sources.filter((s) => a.cited.includes(s.n));
  const consulted = a.sources.filter((s) => !a.cited.includes(s.n));
  const live = a.phase === 'writing' || a.phase === 'done';
  const marks: [string, 'unsupported' | 'failed'][] = a.phase === 'done' || a.paragraphs?.length
    ? [...(a.unsupported ?? []).map((s) => [s, 'unsupported'] as [string, 'unsupported']), ...(a.failed ?? []).map((s) => [s, 'failed'] as [string, 'failed'])]
    : [];
  const copy = async () => {
    await navigator.clipboard?.writeText(withReferences(a));
    setCopied(true);
  };
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
            <p className="lw-meta" style={{ margin: '6px 0 0', fontFamily: 'var(--sans)' }}>{ABSTAINED[a.reason ?? 'no_passages'] ?? ABSTAINED.no_passages}</p>
          </div>
        )}
        {live && (a.paragraphs?.length
          ? a.paragraphs.map((p) => <p key={p.index} className="lw-serif lw-answer-text">{withMarks(p.text, marks, a.sources, onFocus)}</p>)
          : a.text.split(/\n{2,}/).map((t, i) => <p key={i} className="lw-serif lw-answer-text">{withMarks(t, marks, a.sources, onFocus)}</p>))}
        {a.phase === 'done' && a.status === 'unverified' && <StatusPill tone="conflict">None of the quotes could be found in the sources. Do not rely on this answer.</StatusPill>}
        {a.phase === 'done' && a.status === 'partial' && (a.failed?.length ?? 0) > 0 && <StatusPill tone="attention">Some statements could not be checked against the sources — they are marked.</StatusPill>}
        {a.phase === 'done' && a.uncited && a.status !== 'unverified' && <StatusPill tone="attention">No source cited — check before relying on it</StatusPill>}
        {a.phase === 'done' && !a.uncited && !(a.failed?.length) && (a.unsupported?.length ?? 0) > 0 && (
          <StatusPill tone="attention">{a.unsupported!.length === 1
            ? '1 sentence cites no source — shown dimmed. Check it before relying on it.'
            : `${a.unsupported!.length} sentences cite no source — shown dimmed. Check them before relying on them.`}</StatusPill>
        )}
      </div>
      {a.phase === 'done' && (a.amendments ?? []).map((m) => (
        <div key={`${m.version_id}-${m.clause}-${m.by.version_id}`} role="alert" className="lw-conflict">
          <b>{m.by.signed
            ? `Clause ${m.clause} of ${m.document_title} was ${m.how} by ${m.by.document_title}${m.by.clause ? `, clause ${m.by.clause}` : ''} (signed)`
            : `Clause ${m.clause} of ${m.document_title} would be ${m.how} by ${m.by.document_title}, which is not signed`}</b>
          {m.by.effective?.words && <span className="lw-meta">It says: “{m.by.effective.words}”</span>}
          {m.by.signed && m.cited && !m.amendment_cited && <span className="lw-meta">This answer relies on the earlier wording. Read the amendment before relying on it.</span>}
          {m.by.char_start != null && m.by.char_end != null && (
            <Link className="lw-linklike lw-meta" to={`/m/${matter.number}/documents/${encodeURIComponent(m.by.document_id)}/read?v=${encodeURIComponent(m.by.version_id)}&at=${m.by.char_start}-${m.by.char_end}`}>Open the amendment</Link>
          )}
        </div>
      ))}
      {a.phase === 'done' && (a.conflicts ?? []).map((c) => (
        <div key={`${c.document_id}-${c.clause}-${c.other.id}`} role="alert" className="lw-conflict">
          <b>Clause {c.clause} reads differently in version {c.other.number} ({(STATUS[c.other.status ?? ''] ?? c.other.status ?? 'unknown').toLowerCase()})</b>
          <span className="lw-meta">{c.document_title} · this answer uses version {c.version.number}{c.version.status ? ` (${(STATUS[c.version.status] ?? c.version.status).toLowerCase()})` : ''}</span>
          <blockquote className="lw-serif">{plain(c.other.text)}</blockquote>
          <Link className="lw-linklike lw-meta" to={`/m/${matter.number}/documents/${encodeURIComponent(c.document_id)}/read?v=${encodeURIComponent(c.other.id)}`}>Open version {c.other.number}</Link>
        </div>
      ))}
      {a.evidence && a.evidence.length > 0 ? (
        <section aria-label="Evidence" style={{ marginTop: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '0 0 8px' }}>
            <h3 className="lw-eyebrow" style={{ margin: 0 }}>Evidence</h3>
            {a.phase === 'done' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={copy}><Copy />{copied ? 'Copied' : 'Copy with citations'}</button>}
          </div>
          <ol className="lw-sources">{a.evidence.map((e, i) => <EvidenceItem key={`${e.n}-${i}`} e={e} active={focus === e.n}
            source={a.sources.find((s) => s.n === e.n)} />)}</ol>
        </section>
      ) : cited.length > 0 && (
        <section aria-label="Sources" style={{ marginTop: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '0 0 8px' }}>
            <h3 className="lw-eyebrow" style={{ margin: 0 }}>Sources</h3>
            {a.phase === 'done' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={copy}><Copy />{copied ? 'Copied' : 'Copy with citations'}</button>}
          </div>
          <ol className="lw-sources">{cited.map((s) => <SourceItem key={s.n} s={s} active={focus === s.n} />)}</ol>
        </section>
      )}
      {a.phase !== 'revoked' && consulted.length > 0 && (a.phase === 'done' || a.phase === 'abstained') && (
        <details style={{ marginTop: 12 }}>
          <summary className="lw-meta" style={{ cursor: 'pointer' }}>Also consulted ({consulted.length})</summary>
          <ol className="lw-sources" style={{ marginTop: 8 }}>{consulted.map((s) => <SourceItem key={s.n} s={s} active={false} />)}</ol>
        </details>
      )}
      {(a.phase === 'done' || a.phase === 'abstained') && a.turnId && <SendForReview turnId={a.turnId} />}
    </article>
  );
}

/** S2.7: the answer, fixed as given, goes to a second lawyer as a research note (D-043). */
function SendForReview({ turnId }: { turnId: string }) {
  const matter = useMatter();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  if (!['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') || matter.status !== 'open') return null;
  const send = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ artifact: { id: string } }>(`/api/legal/matters/${encodeURIComponent(matter.id)}/research/${encodeURIComponent(turnId)}/review`);
      await qc.invalidateQueries({ queryKey: ['reviews'] });
      navigate(`/m/${matter.number}/notes/${encodeURIComponent(r.artifact.id)}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The answer could not be sent for review.');
      setBusy(false);
    }
  };
  return (
    <div className="lw-send-review">
      <button type="button" className="lw-btn lw-btn-sm" disabled={busy} onClick={send}><Send />{busy ? 'Sending…' : 'Send for review'}</button>
      <span className="lw-meta">A second lawyer approves this exact answer and its evidence before it is filed.</span>
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}

/** The answer as plain text with its references, for pasting into a note or an email. */
function withReferences(a: Answer): string {
  const where = (x: { clause?: string | null; page_start?: number | null; page_end?: number | null; line?: number | null }) =>
    [x.clause ? `cl. ${x.clause}` : '', pages(x), x.line ? `line ${x.line}` : ''].filter(Boolean).join(', ');
  const refs = a.evidence?.length
    ? a.evidence.map((e) => `[${e.n}] ${e.document_title}, version ${e.version_number}${e.version_status ? ` (${e.version_status})` : ''}` +
        `${where(e) ? `, ${where(e)}` : ''}: “${e.quote}”${e.verification === 'failed' ? ' (quote not found in the source)' : ''}`)
    : a.sources.filter((s) => a.cited.includes(s.n)).map((s) => `[${s.n}] ${s.document_title}, version ${s.version_number}${where(s) ? `, ${where(s)}` : ''}: “${plain(s.text)}”`);
  return `${a.text}\n\n${refs.join('\n')}`;
}

/** The answer with flagged sentences marked: those citing nothing (T08) and those whose quote is not
 *  in the source (S2.2). */
function withMarks(text: string, marks: [string, 'unsupported' | 'failed'][], sources: ResearchSource[], onFocus: (n: number) => void): React.ReactNode {
  const out: React.ReactNode[] = [];
  const found = marks.map(([s, kind]) => ({ s, kind, at: text.indexOf(s) })).filter((m) => m.at >= 0).sort((x, y) => x.at - y.at);
  let at = 0;
  found.forEach(({ s, kind, at: start }, i) => {
    if (start < at) return;
    if (start > at) out.push(<React.Fragment key={`t${i}`}>{withCitations(plain(text.slice(at, start)), sources, onFocus)}</React.Fragment>);
    out.push(
      <span key={`u${i}`} className="lw-unsupported" data-unsupported={kind === 'unsupported' ? '' : undefined} data-failed={kind === 'failed' ? '' : undefined}>
        <span>{kind === 'failed' ? plain(s).replace(/\s*(\[\d{1,2}\])+\s*\.?$/, '') : plain(s)}</span>
        <span className="lw-sr">{kind === 'failed' ? ' (quote not found in the source)' : ' (no source cited)'}</span>
      </span>,
    );
    at = start + s.length;
  });
  if (at < text.length) out.push(<React.Fragment key="end">{withCitations(plain(text.slice(at)), sources, onFocus)}</React.Fragment>);
  return out;
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

function EvidenceItem({ e, active, source }: { e: ResearchEvidence; active: boolean; source?: ResearchSource }) {
  const ref = useRef<HTMLLIElement>(null);
  const matter = useMatter();
  useEffect(() => { if (active) ref.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [active]);
  const [label, tone] = VERIFIED[e.verification];
  const meta = [e.document_title, `v${e.version_number}`, e.version_status ? STATUS[e.version_status] ?? e.version_status : '',
    e.clause ? `cl. ${e.clause}` : '', pages(e), e.line ? `line ${e.line}` : ''].filter(Boolean).join(' · ');
  const href = e.char_start != null && e.char_end != null
    ? `/m/${matter.number}/documents/${encodeURIComponent(e.document_id)}/read?v=${encodeURIComponent(e.version_id)}&at=${e.char_start}-${e.char_end}` : null;
  // A quote that could not be found still points at the passage the model cited, so the reviewer can check it there.
  const at = source ? [source.focus_start ?? source.char_start, source.focus_end ?? source.char_end] : [];
  const passage = !href && source && !e.removed
    ? `/m/${matter.number}/documents/${encodeURIComponent(source.document_id)}/read?v=${encodeURIComponent(source.version_id)}` +
      (at[0] != null && at[1] != null ? `&at=${at[0]}-${at[1]}` : '') : null;
  return (
    <li ref={ref} className="lw-source" data-active={active || undefined} data-verification={e.verification}>
      <span className="lw-cite lw-cite-static" aria-hidden="true">{e.n}</span>
      <div style={{ minWidth: 0, display: 'grid', gap: 4 }}>
        {e.removed ? <p className="lw-meta" style={{ margin: 0 }}>The document this quoted was deleted.</p>
          : <blockquote className="lw-serif">{`“${e.quote}”`}</blockquote>}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <StatusPill tone={tone}>{label}</StatusPill>
          {e.matched && <StatusPill tone="neutral">Reference added: the answer quoted this without one</StatusPill>}
          {e.signed === false && <StatusPill tone="attention">Not signed</StatusPill>}
        </div>
        <div className="lw-meta">{meta}</div>
        {href && <Link className="lw-linklike lw-meta" to={href} aria-label={`Open evidence ${e.n} in the document`}>Open in document</Link>}
        {passage && <Link className="lw-linklike lw-meta" to={passage} aria-label={`Open source ${e.n} in the document`}>Open the cited passage</Link>}
      </div>
    </li>
  );
}

function SourceItem({ s, active }: { s: ResearchSource; active: boolean }) {
  const ref = useRef<HTMLLIElement>(null);
  const matter = useMatter();
  useEffect(() => { if (active) ref.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [active]);
  const where = [pages(s), s.clause ? `cl. ${s.clause}` : '', s.line ? `line ${s.line}` : ''].filter(Boolean).join(' · ');
  const a = s.focus_start ?? s.char_start, b = s.focus_end ?? s.char_end;
  const href = `/m/${matter.number}/documents/${encodeURIComponent(s.document_id)}/read?v=${encodeURIComponent(s.version_id)}` +
    (a != null && b != null ? `&at=${a}-${b}` : '');
  return (
    <li ref={ref} className="lw-source" data-active={active || undefined}>
      <span className="lw-cite lw-cite-static" aria-hidden="true">{s.n}</span>
      <div style={{ minWidth: 0 }}>
        <blockquote className="lw-serif">{plain(s.text)}</blockquote>
        <div className="lw-meta">{s.document_title} · v{s.version_number}{where ? ` · ${where}` : ''}</div>
        {s.instructions && <div className="lw-meta lw-source-flag">Contains text addressed to an AI assistant. It was withheld from the answer.</div>}
        <Link className="lw-linklike lw-meta" to={href} aria-label={`Open source ${s.n} in the document`}>Open in document</Link>
      </div>
    </li>
  );
}
