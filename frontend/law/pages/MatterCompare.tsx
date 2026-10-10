import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError, postStream } from '../api/client';
import { comparePath, keys, useComparisons, useDocuments, useVersions } from '../api/queries';
import type {
  ChangeFlag, ComparePair, CompareQuote, CompareSide, Comparison, ConflictScan, DocVersion, FactKind, ScanFinding, ScanSide, VersionPolicy,
} from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

const STATUS: Record<string, string> = { draft: 'Draft', final: 'Final', executed: 'Executed', received: 'Received' };
const FLAG: Record<ChangeFlag, string> = { number: 'Number changed', obligation: 'Obligation changed', negation: 'Negation changed' };
const VERIFIED: Record<string, [string, 'verified' | 'conflict' | 'neutral']> = {
  verbatim: ['Quoted exactly', 'verified'],
  normalized: ['Quoted with small differences', 'verified'],
  failed: ['Quote not found', 'conflict'],
  not_quoted: ['No quote given', 'neutral'],
};
const FACT: Record<FactKind, string> = {
  money: 'Amounts differ', period: 'Periods differ', percent: 'Percentages differ', date: 'Dates differ',
  law: 'Governing law differs', entity: 'A company’s name differs', reference: 'The date of the agreement referred to differs',
};
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const plain = (t: string) => t.replace(/(^|\n)(?:#{1,6}|>)\s?/g, '$1').replace(/\*\*/g, '');
const versionName = (v: { number: number; status?: string | null }) => `version ${v.number}${v.status ? ` (${(STATUS[v.status] ?? v.status).toLowerCase()})` : ''}`;

/** Compare: two versions clause by clause, or every document in the matter checked against the others.
 *  What changed is computed from the documents' own text; the model only explains, with quotes that are checked. */
export default function MatterCompare() {
  const matter = useMatter();
  const [params, setParams] = useSearchParams();
  // The view answers at once; the address follows, so a reload or a shared link opens the same view.
  const [view, setViewState] = useState<'versions' | 'scan'>(params.get('view') === 'scan' ? 'scan' : 'versions');
  const setView = (v: 'versions' | 'scan') => { setViewState(v); setParams(v === 'scan' ? { view: 'scan' } : {}, { replace: true }); };
  const history = useComparisons(matter.id);
  const [opened, setOpened] = useState<Comparison | ConflictScan | null>(null);
  const [openError, setOpenError] = useState('');

  const open = async (id: string) => {
    setOpenError('');
    try {
      const c = (await api.get<{ comparison: Comparison | ConflictScan }>(`${comparePath(matter.id)}/${encodeURIComponent(id)}`)).comparison;
      setOpened(c);
      setView(c.kind === 'scan' ? 'scan' : 'versions');
    } catch (err) {
      setOpenError(err instanceof ApiError && err.status === 404 ? 'That comparison is no longer available to you.' : 'It could not be opened.');
    }
  };

  return (
    <>
      <Toolbar crumb={matter.number} title="Compare" />
      <div className="lw-content lw-research">
        <div style={{ display: 'grid', gap: 16, alignContent: 'start', minWidth: 0 }}>
          <fieldset className="lw-segment" aria-label="Compare" style={{ justifySelf: 'start' }}>
            <legend className="lw-sr">Compare</legend>
            {([['versions', 'Two versions'], ['scan', 'Across documents']] as const).map(([v, label]) => (
              <label key={v} data-on={view === v || undefined}>
                <input type="radio" name="compare-view" value={v} checked={view === v}
                  onChange={() => { setOpened(null); setView(v); }} />{label}
              </label>
            ))}
          </fieldset>
          {openError && <ErrorNote>{openError}</ErrorNote>}
          {view === 'versions'
            ? <VersionsCompare key={opened?.kind === 'versions' ? opened.id : 'new'} initial={opened?.kind === 'versions' ? opened : null} />
            : <ScanView key={opened?.kind === 'scan' ? opened.id : 'new'} initial={opened?.kind === 'scan' ? opened : null} />}
        </div>
        <aside className="lw-paper lw-research-history" aria-label="Your earlier comparisons">
          <div className="lw-card-h"><h2 className="lw-h2">Your earlier comparisons</h2></div>
          {history.isPending ? <div style={{ padding: 16 }}><Skeleton /></div>
            : !history.data?.length ? <p className="lw-meta" style={{ padding: '12px 20px', margin: 0 }}>Only you can see your comparisons.</p>
            : (
              <ol className="lw-history-list">
                {history.data.map((c) => (
                  <li key={c.id}>
                    <button type="button" className="lw-history-item" onClick={() => open(c.id)}>
                      <span className="lw-ellipsis">{c.kind === 'versions'
                        ? `${c.left.document_title} v${c.left.number} → ${c.right.document_title === c.left.document_title ? '' : `${c.right.document_title} `}v${c.right.number}`
                        : `Conflicts across ${c.documents} documents`}</span>
                      <span className="lw-meta">{when(c.created_at)} · {c.kind === 'versions'
                        ? `${c.summary.changed + c.summary.added + c.summary.removed} changes`
                        : `${c.conflicts} ${c.conflicts === 1 ? 'conflict' : 'conflicts'}`}</span>
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

// ── Two versions ─────────────────────────────────────────────────────────────────
interface Pick { doc: string; version: string }

function VersionPicker({ legend, value, onChange }: { legend: string; value: Pick; onChange: (p: Pick) => void }) {
  const matter = useMatter();
  const docs = useDocuments(matter.id);
  const versions = useVersions(matter.id, value.doc || null);
  const readable = (docs.data ?? []).filter((d) => d.current_version);
  const id = legend.toLowerCase().replace(/\W+/g, '-');
  useEffect(() => {
    if (value.doc && versions.data?.length && !versions.data.some((v) => v.id === value.version)) {
      onChange({ doc: value.doc, version: versions.data[versions.data.length - 1].id });
    }
  }, [value.doc, value.version, versions.data, onChange]);
  return (
    <fieldset className="lw-compare-pick">
      <legend className="lw-eyebrow">{legend}</legend>
      <label htmlFor={`${id}-doc`} className="lw-sr">{legend}: document</label>
      <select id={`${id}-doc`} className="lw-input" value={value.doc} onChange={(e) => onChange({ doc: e.target.value, version: '' })}>
        <option value="">Choose a document…</option>
        {readable.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}
      </select>
      <label htmlFor={`${id}-ver`} className="lw-sr">{legend}: version</label>
      <select id={`${id}-ver`} className="lw-input" value={value.version} disabled={!versions.data?.length}
        onChange={(e) => onChange({ ...value, version: e.target.value })}>
        {(versions.data ?? []).slice().reverse().map((v: DocVersion) => (
          <option key={v.id} value={v.id}>{`Version ${v.number}${v.status ? ` · ${STATUS[v.status] ?? v.status}` : ''} · ${v.filename}`}</option>
        ))}
      </select>
    </fieldset>
  );
}

function VersionsCompare({ initial }: { initial: Comparison | null }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const docs = useDocuments(matter.id);
  const [left, setLeft] = useState<Pick>({ doc: initial?.left.document_id ?? '', version: initial?.left.version_id ?? '' });
  const [right, setRight] = useState<Pick>({ doc: initial?.right.document_id ?? '', version: initial?.right.version_id ?? '' });
  const [result, setResult] = useState<Comparison | null>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [showAll, setShowAll] = useState(false);
  const [explaining, setExplaining] = useState<number[] | null>(null);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);

  // Default: the same document on both sides, once one has more than one version.
  useEffect(() => {
    if (left.doc || right.doc || !docs.data) return;
    const d = docs.data.find((x) => (x.current_version?.number ?? 0) > 1) ?? docs.data.find((x) => x.current_version);
    if (d) { setLeft({ doc: d.id, version: '' }); setRight({ doc: d.id, version: '' }); }
  }, [docs.data, left.doc, right.doc]);
  const leftVersions = useVersions(matter.id, left.doc || null);
  useEffect(() => {                         // left defaults to the latest signed version, else the first
    if (!left.doc || left.version || !leftVersions.data?.length) return;
    const signed = [...leftVersions.data].reverse().find((v) => v.status === 'executed');
    setLeft({ doc: left.doc, version: (signed ?? leftVersions.data[0]).id });
  }, [left, leftVersions.data]);

  const run = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ comparison: Comparison }>(comparePath(matter.id), { left_version_id: left.version, right_version_id: right.version });
      setResult(r.comparison);
      await qc.invalidateQueries({ queryKey: keys.comparisons(matter.id) });
    } catch (err) {
      setError(err instanceof ApiError ? (err.status === 404 ? 'One of these versions is not available to you.' : err.message) : 'The comparison could not be made.');
    } finally { setBusy(false); }
  };

  const explain = async (pairs?: number[]) => {
    if (!result) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    const wanted = pairs ?? result.pairs.filter((p) => ['changed', 'added', 'removed'].includes(p.kind) && !p.explanation).map((p) => p.index);
    setExplaining(wanted); setError('');
    let state = result;
    try {
      await postStream(`${comparePath(matter.id)}/${encodeURIComponent(result.id)}/explain`, pairs ? { pairs } : {}, (ev) => {
        if (ev.type === 'explanation') {
          state = { ...state, pairs: state.pairs.map((p) => (p.index === ev.index ? { ...p, explanation: ev } : p)) };
          setResult(state);
        } else if (ev.type === 'scope_revoked') { setError('Your access to this matter ended. Nothing more is shown.'); setResult(null); }
        else if (ev.type === 'error') setError(ev.message ?? 'The explanations could not be completed.');
      }, ctrl.signal);
    } catch (err) {
      if (!ctrl.signal.aborted) setError(err instanceof ApiError ? err.message : 'The explanations could not be completed.');
    } finally { if (!ctrl.signal.aborted) setExplaining(null); }
  };

  const shown = useMemo(() => (result?.pairs ?? []).filter((p) => showAll || p.kind !== 'unchanged' || p.moved || p.renumbered), [result, showAll]);
  const s = result?.summary;
  const material = (result?.pairs ?? []).filter((p) => ['changed', 'added', 'removed'].includes(p.kind));
  const ready = left.version && right.version && left.version !== right.version;

  return (
    <>
      <section className="lw-paper lw-sheet" aria-label="Choose two versions">
        <h2 className="lw-h2" style={{ margin: 0 }}>Compare two versions</h2>
        <p className="lw-meta" style={{ margin: '2px 0 12px' }}>Clauses are matched by their wording, so a renumbered or moved clause is found where it went.</p>
        <div className="lw-compare-picks">
          <VersionPicker legend="From" value={left} onChange={setLeft} />
          <VersionPicker legend="To" value={right} onChange={setRight} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
          <button type="button" className="lw-btn lw-btn-primary" disabled={!ready || busy} onClick={run}>{busy ? 'Comparing…' : 'Compare'}</button>
        </div>
      </section>
      {error && <ErrorNote>{error}</ErrorNote>}
      {result && s && (
        <article className="lw-paper lw-sheet" aria-label="Comparison" aria-busy={!!explaining}>
          <div className="lw-compare-head">
            <div style={{ minWidth: 0 }}>
              <div className="lw-eyebrow">Comparison</div>
              <h2 className="lw-serif lw-answer-q" style={{ margin: '2px 0 4px' }}>
                {result.left.document_title}, {versionName(result.left)} → {result.right.document_title === result.left.document_title ? '' : `${result.right.document_title}, `}{versionName(result.right)}
              </h2>
              <p className="lw-meta" style={{ margin: 0 }} role="status">
                {[s.changed && `${s.changed} changed`, s.added && `${s.added} added`, s.removed && `${s.removed} removed`, s.moved && `${s.moved} moved`,
                  s.renumbered && `${s.renumbered} renumbered`, s.formatting && `${s.formatting} formatting only`].filter(Boolean).join(' · ') || 'No differences'}
                {` · ${s.unchanged} unchanged`}
              </p>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <label className="lw-meta" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />Show unchanged
              </label>
              {material.length > 0 && (
                <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={!!explaining || material.every((p) => p.explanation)} onClick={() => explain()}>
                  {explaining ? `Explaining ${explaining.length}…` : 'Explain changes'}
                </button>
              )}
            </div>
          </div>
          <div className="lw-compare-cols" aria-hidden="true">
            <span className="lw-eyebrow">{versionName(result.left)}</span><span className="lw-eyebrow">{versionName(result.right)}</span>
          </div>
          {shown.length === 0 ? <p className="lw-meta">The two versions read the same.</p> : (
            <ol className="lw-compare-rows">
              {shown.map((p) => (
                <PairRow key={p.index} p={p} c={result} waiting={!!explaining?.includes(p.index) && !p.explanation}
                  onExplain={explaining ? undefined : () => explain([p.index])} />
              ))}
            </ol>
          )}
        </article>
      )}
    </>
  );
}

function sideLabel(x: CompareSide | null): string {
  if (!x) return '';
  return x.label ? (/^\d|^\(/.test(x.label) ? `cl. ${x.label}` : x.label) : x.clause ? `In cl. ${x.clause}` : x.kind === 'heading' ? 'Heading' : 'Paragraph';
}

function PairRow({ p, c, waiting, onExplain }: { p: ComparePair; c: Comparison; waiting: boolean; onExplain?: () => void }) {
  const matter = useMatter();
  const label = sideLabel(p.right) || sideLabel(p.left);
  const kind = { unchanged: 'Unchanged', formatting: 'Formatting only', changed: 'Changed', added: 'Added', removed: 'Removed' }[p.kind];
  const tone = p.kind === 'changed' ? 'attention' : p.kind === 'added' ? 'verified' : p.kind === 'removed' ? 'conflict' : 'neutral';
  const href = (v: { document_id: string; version_id: string }, x: CompareSide) =>
    `/m/${matter.number}/documents/${encodeURIComponent(v.document_id)}/read?v=${encodeURIComponent(v.version_id)}&at=${x.start}-${x.end}`;
  return (
    <li className="lw-compare-row" data-kind={p.kind} aria-label={`${label}: ${kind}`}>
      <div className="lw-compare-row-h">
        <b className="lw-tnum">{label}</b>
        <StatusPill tone={tone}>{kind}</StatusPill>
        {p.moved && <StatusPill tone="neutral">{`Moved${p.left?.label ? ` from ${p.left.label}` : ''}`}</StatusPill>}
        {p.renumbered && !p.moved && <StatusPill tone="neutral">{`Renumbered ${p.left?.label} → ${p.right?.label}`}</StatusPill>}
        {p.flags.map((f) => <StatusPill key={f} tone={f === 'number' ? 'neutral' : 'attention'}>{FLAG[f]}</StatusPill>)}
        {p.how === 'meaning' && <span className="lw-meta">Matched by meaning</span>}
        {onExplain && ['changed', 'added', 'removed'].includes(p.kind) && !p.explanation && (
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" style={{ marginLeft: 'auto' }} onClick={onExplain}>Explain</button>
        )}
      </div>
      <div className="lw-compare-cols">
        <div className="lw-compare-side" data-side="left">
          {p.left ? (
            <>
              <p className="lw-serif lw-compare-text">{marked(p.left.text, p.left_marks, 'del')}</p>
              <Link className="lw-linklike lw-meta" to={href(c.left, p.left)} aria-label={`Open ${sideLabel(p.left)} of ${versionName(c.left)} in the document`}>
                {['Open', p.left.page ? `p. ${p.left.page}` : '', p.left.line ? `line ${p.left.line}` : ''].filter(Boolean).join(' · ')} ↗
              </Link>
            </>
          ) : <p className="lw-meta lw-compare-none">Not in {versionName(c.left)}</p>}
        </div>
        <div className="lw-compare-side" data-side="right">
          {p.right ? (
            <>
              <p className="lw-serif lw-compare-text">{marked(p.right.text, p.right_marks, 'ins')}</p>
              <Link className="lw-linklike lw-meta" to={href(c.right, p.right)} aria-label={`Open ${sideLabel(p.right)} of ${versionName(c.right)} in the document`}>
                {['Open', p.right.page ? `p. ${p.right.page}` : '', p.right.line ? `line ${p.right.line}` : ''].filter(Boolean).join(' · ')} ↗
              </Link>
            </>
          ) : <p className="lw-meta lw-compare-none">Not in {versionName(c.right)}</p>}
        </div>
      </div>
      {waiting && <p className="lw-meta" role="status">Explaining…</p>}
      {p.explanation && <Explanation e={p.explanation} name={(n) => (n === 1 ? 'from' : 'to')} />}
    </li>
  );
}

/** Text with the removed (del) or inserted (ins) characters marked, each announced to screen readers. */
function marked(text: string, marks: [number, number][], tag: 'del' | 'ins'): React.ReactNode {
  const out: React.ReactNode[] = [];
  let at = 0;
  [...marks].sort((a, b) => a[0] - b[0]).forEach(([a, b], i) => {
    if (a < at) return;
    if (a > at) out.push(<React.Fragment key={`t${i}`}>{plain(text.slice(at, a))}</React.Fragment>);
    const Tag = tag;
    out.push(<Tag key={`m${i}`}><span className="lw-sr">{tag === 'del' ? 'removed: ' : 'added: '}</span>{text.slice(a, b)}</Tag>);
    at = b;
  });
  if (at < text.length) out.push(<React.Fragment key="end">{plain(text.slice(at))}</React.Fragment>);
  return out;
}

/** The model's explanation, its citations named (`name`), and each quote with how it was checked. */
function Explanation({ e, name }: {
  e: { text: string; status: string; evidence: (CompareQuote | ScanFinding['evidence'][number])[]; failed: string[] };
  name: (n: number) => string;
}) {
  const shown = e.text.replace(/\[(\d)\]/g, (m, n) => (n === '1' || n === '2' ? `[${name(Number(n))}]` : m));
  return (
    <div className="lw-compare-explain">
      <p className="lw-serif" style={{ margin: 0 }}>{shown || 'No explanation was given.'}</p>
      {e.status === 'unverified' && <StatusPill tone="conflict">None of its quotes could be found. Do not rely on this explanation.</StatusPill>}
      {e.status === 'partial' && <StatusPill tone="attention">Some statements could not be checked against the wording.</StatusPill>}
      {e.evidence.length > 0 && (
        <ul className="lw-compare-quotes" aria-label="Quotes in this explanation">
          {e.evidence.map((q, i) => {
            const [text, tone] = VERIFIED[q.verification] ?? VERIFIED.not_quoted;
            return (
              <li key={i} data-verification={q.verification}>
                <span className="lw-meta">{'side' in q ? (q.side === 'left' ? 'From' : 'To') : name(q.n)}</span>
                <q className="lw-serif">{q.quote}</q>
                <StatusPill tone={tone}>{text}</StatusPill>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

// ── Across documents ─────────────────────────────────────────────────────────────
function ScanView({ initial }: { initial: ConflictScan | null }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [versions, setVersions] = useState<VersionPolicy>(initial?.versions ?? 'signed');
  const [phase, setPhase] = useState<'idle' | 'reading' | 'checking' | 'done' | 'revoked' | 'error'>(initial ? 'done' : 'idle');
  const [docs, setDocs] = useState(initial?.documents ?? []);
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: initial?.candidates ?? 0 });
  const [found, setFound] = useState<ScanFinding[]>(initial?.findings ?? []);
  const [rejected, setRejected] = useState<ScanFinding[]>(initial?.rejected ?? []);
  const [hidden, setHidden] = useState(initial?.hidden ?? 0);
  const [error, setError] = useState('');
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const busy = phase === 'reading' || phase === 'checking';

  const run = async () => {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setPhase('reading'); setError(''); setFound([]); setRejected([]); setDocs([]); setHidden(0); setProgress({ done: 0, total: 0 });
    try {
      await postStream(`${comparePath(matter.id)}/scan`, { versions }, (ev) => {
        if (ev.type === 'documents') { setDocs(ev.documents); setProgress({ done: 0, total: ev.candidates }); setPhase('checking'); }
        else if (ev.type === 'checked') {
          setProgress({ done: ev.done, total: ev.total });
          (ev.finding.verdict === 'no_conflict' ? setRejected : setFound)((xs) => [...xs, ev.finding]);
        } else if (ev.type === 'done') setPhase('done');
        else if (ev.type === 'scope_revoked') { setFound([]); setRejected([]); setPhase('revoked'); }
        else if (ev.type === 'error') setPhase('error');
      }, ctrl.signal);
      await qc.invalidateQueries({ queryKey: keys.comparisons(matter.id) });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setPhase('error');
      setError(err instanceof ApiError ? (err.status === 404 ? 'This matter is no longer available to you.' : err.message) : 'The check could not be started.');
    }
  };
  const conflicts = found.filter((f) => f.verdict === 'conflict');
  const unclear = found.filter((f) => f.verdict === 'unclear');

  return (
    <>
      <section className="lw-paper lw-sheet" aria-label="Check for conflicts">
        <h2 className="lw-h2" style={{ margin: 0 }}>Conflicts across documents</h2>
        <p className="lw-meta" style={{ margin: '2px 0 12px' }}>
          Amounts, periods, dates, governing law, company names and agreement references are compared between related documents. Each possible conflict is then checked in context, with quotes from both.
        </p>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <fieldset className="lw-segment" aria-label="Check">
            <legend className="lw-sr">Check</legend>
            {([['signed', 'Signed versions'], ['current', 'Latest drafts']] as [VersionPolicy, string][]).map(([v, label]) => (
              <label key={v} data-on={versions === v || undefined}>
                <input type="radio" name="scan-versions" value={v} checked={versions === v} onChange={() => setVersions(v)} />{label}
              </label>
            ))}
          </fieldset>
          <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={run}>{busy ? 'Checking…' : 'Check for conflicts'}</button>
        </div>
      </section>
      {error && <ErrorNote>{error}</ErrorNote>}
      {phase !== 'idle' && (
        <article className="lw-paper lw-sheet" aria-label="Conflicts" aria-busy={busy}>
          <div aria-live="polite">
            {phase === 'reading' && <p className="lw-meta">Reading this matter’s documents…</p>}
            {phase === 'checking' && <p className="lw-meta" role="status">{`Checking ${progress.done} of ${progress.total} possible conflicts…`}</p>}
            {phase === 'revoked' && <p role="alert">Your access to this matter ended during the check. Nothing more is shown.</p>}
            {phase === 'error' && !error && <p role="alert">The check could not be completed. Try again.</p>}
            {phase === 'done' && (
              <p className="lw-meta" role="status" style={{ margin: 0 }}>
                {`${docs.length} documents read (${versions === 'signed' ? 'signed versions' : 'latest drafts'}) · ${progress.total} possible conflicts checked · `}
                {conflicts.length === 0 ? 'no conflicts found' : `${conflicts.length} ${conflicts.length === 1 ? 'conflict' : 'conflicts'} found`}
                {hidden > 0 && ` · ${hidden} hidden because you can no longer see a document they quote`}
              </p>
            )}
          </div>
          {phase === 'done' && progress.total === 0 && (
            <EmptyState title="Nothing to check">No two related documents state different amounts, periods, dates, governing law, names or agreement dates.</EmptyState>
          )}
          {(conflicts.length > 0 || unclear.length > 0) && (
            <ol className="lw-compare-rows" aria-label="Possible conflicts">
              {[...conflicts, ...unclear].map((f, i) => <FindingItem key={`${f.a.version_id}-${f.a.sentence_start}-${f.b.version_id}-${f.b.sentence_start}-${i}`} f={f} />)}
            </ol>
          )}
          {rejected.length > 0 && (
            <details style={{ marginTop: 14 }}>
              <summary className="lw-meta" style={{ cursor: 'pointer' }}>{`Checked and not in conflict (${rejected.length})`}</summary>
              <ol className="lw-compare-rows" style={{ marginTop: 8 }}>{rejected.map((f, i) => <FindingItem key={i} f={f} />)}</ol>
            </details>
          )}
        </article>
      )}
    </>
  );
}

function FindingItem({ f }: { f: ScanFinding }) {
  const verdict = f.verdict === 'conflict' ? <StatusPill tone="conflict">The documents say different things</StatusPill>
    : f.verdict === 'unclear' ? <StatusPill tone="attention">Not checked by the model — read both</StatusPill>
    : <StatusPill tone="neutral">Not in conflict</StatusPill>;
  return (
    <li className="lw-compare-row" data-kind={f.verdict} aria-label={FACT[f.kind]}>
      <div className="lw-compare-row-h">
        <b>{FACT[f.kind]}</b>{verdict}
        <span className="lw-meta">{f.method === 'rule' ? 'Found as written' : f.verdict === 'unclear' ? '' : 'Checked in context'}</span>
      </div>
      <div className="lw-compare-cols">
        <ScanQuote s={f.a} />
        <ScanQuote s={f.b} />
      </div>
      {f.subject && <p className="lw-meta" style={{ margin: 0 }}>{`What each passage governs: ${f.subject.replace(/^(yes|no)[,.:;]?\s*/i, '')}`}</p>}
      {f.method === 'model' && (f.explanation || f.evidence.length > 0) && (
        <Explanation e={{ text: f.explanation, status: f.failed.length ? 'partial' : 'answered', evidence: f.evidence, failed: f.failed }}
          name={(n) => (n === 1 ? f.a.document_title : f.b.document_title)} />
      )}
    </li>
  );
}

function ScanQuote({ s }: { s: ScanSide }) {
  const matter = useMatter();
  const a = s.sentence_start - s.context_start, b = s.sentence_end - s.context_start;
  const va = s.value_start - s.context_start, vb = s.value_end - s.context_start;
  const sentence = s.context.slice(a, b);
  return (
    <div className="lw-compare-side">
      <div className="lw-meta" style={{ marginBottom: 4 }}>
        {[s.document_title, `v${s.version_number}`, s.version_status ? STATUS[s.version_status] ?? s.version_status : '', s.clause ? `cl. ${s.clause}` : '',
          s.line ? `line ${s.line}` : ''].filter(Boolean).join(' · ')}
        {!s.signed && <> <StatusPill tone="attention">Not signed</StatusPill></>}
      </div>
      <p className="lw-serif lw-compare-text">
        {plain(sentence.slice(0, va - a))}<mark className="lw-compare-value">{sentence.slice(va - a, vb - a)}</mark>{plain(sentence.slice(vb - a))}
      </p>
      <Link className="lw-linklike lw-meta" to={`/m/${matter.number}/documents/${encodeURIComponent(s.document_id)}/read?v=${encodeURIComponent(s.version_id)}&at=${s.sentence_start}-${s.sentence_end}`}
        aria-label={`Open ${s.document_title} at this sentence`}>Open in document ↗</Link>
    </div>
  );
}
