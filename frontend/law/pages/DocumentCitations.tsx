import React, { useState } from 'react';
import { Link, useParams } from 'react-router';
import { ArrowLeft, Info, MessageSquarePlus, TextQuote } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, useCitationChecks, useDocument } from '../api/queries';
import type { CitationCheck, CitationItem, CitationStatus } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const STATUS: Record<CitationStatus, [string, Tone]> = {
  resolved: ['Found', 'verified'], missing: ['Missing clause', 'conflict'], not_found: ['Not found', 'attention'],
  accurate: ['Accurate', 'verified'], altered: ['Altered', 'conflict'], unverified: ['Unverified', 'neutral'],
};
const ORDER: Record<CitationStatus, number> = { missing: 0, altered: 0, not_found: 1, unverified: 2, resolved: 3, accurate: 3 };
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** A document's citations checked (S3.4, D-048): each clause reference resolved to its passage or
 *  flagged, each quotation compared word for word with what it quotes, and each external authority
 *  listed, never verified (P-009). Rules only: no model reads the document and nothing is looked up. */
export default function DocumentCitations() {
  const { docId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const doc = useDocument(matter.id, docId);
  const checks = useCitationChecks(matter.id, docId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const canRun = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const run = async () => {
    setBusy(true); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(docId)}/citation-check`, {});
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, docId), 'citation-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The check could not be run.'); }
    finally { setBusy(false); }
  };
  const latest = checks.data?.[0];
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={doc.data ? `${doc.data.title} — citations` : 'Citations'}>
        <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents/${encodeURIComponent(docId)}/read`}><ArrowLeft />Document</Link>
      </Toolbar>
      <div className="lw-content">
        {canRun && (
          <section className="lw-paper lw-sheet" aria-label="Check citations" aria-busy={busy}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={run}><TextQuote />{busy ? 'Reading the citations…' : latest ? 'Check again' : 'Check citations'}</button>
              <p className="lw-meta" style={{ margin: 0, flex: 1, minWidth: 240 }}>Clause references, quotations and cited authorities in the current version, checked against this matter's documents. Nothing leaves the firm.</p>
            </div>
            {error && <ErrorNote>{error}</ErrorNote>}
          </section>
        )}
        {checks.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : checks.isError ? <ErrorNote>The checks could not be loaded.</ErrorNote>
            : !latest ? <EmptyState title="Not checked yet">Check this document to see each clause it cites, each quotation compared with its source, and each authority it relies on.</EmptyState>
              : <CheckView check={latest} older={checks.data!.length - 1} />}
      </div>
    </>
  );
}

function CheckView({ check, older }: { check: CitationCheck; older: number }) {
  const s = check.summary;
  const of = (kind: CitationItem['kind']) => check.items.filter((i) => i.kind === kind)
    .sort((a, b) => ORDER[a.status] - ORDER[b.status] - (a.malformed.length ? 1 : 0) + (b.malformed.length ? 1 : 0) || a.start - b.start);
  const parts = [
    s.references && `${s.resolved} of ${plural(s.references, 'reference')} found${s.missing ? ` · ${s.missing} missing` : ''}`,
    s.quotes && `${s.accurate} of ${plural(s.quotes, 'quotation')} accurate${s.altered ? ` · ${s.altered} altered` : ''}`,
    s.not_found && `${s.not_found} not found`,
    s.authorities && `${plural(s.authorities, 'authority', 'authorities')}, none verified${s.malformed ? ` · ${s.malformed} cannot be real` : ''}`,
  ].filter(Boolean);
  return (
    <article className="lw-paper lw-sheet" aria-label="Citation check">
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{check.version_number ? `Citations in version ${check.version_number}` : 'Citations'}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{`Checked by ${check.created_by}, ${when(check.created_at)}${older ? ` · ${older} earlier ${older === 1 ? 'check' : 'checks'}` : ''}`}</p>
        </div>
        <StatusPill tone="neutral">Rules only · nothing looked up</StatusPill>
      </div>
      <p className="lw-meta" role="status" style={{ margin: '0 0 10px' }}>{parts.length ? parts.join(' · ') : 'This version cites no clauses, quotations or authorities.'}</p>
      {s.authorities > 0 && (
        <div className="lw-note" role="note" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 12 }}>
          <Info aria-hidden="true" size={16} style={{ flex: 'none', marginTop: 2 }} /><span>{check.notice}</span>
        </div>
      )}
      {(['quote', 'reference', 'authority'] as const).map((kind) => {
        const items = of(kind);
        if (!items.length) return null;
        const title = { quote: 'Quotations', reference: 'Clause references', authority: 'Authorities' }[kind];
        return (
          <section key={kind} aria-label={title} className="lw-cite-group">
            <h3 className="lw-h3" style={{ margin: '14px 0 2px' }}>{title}</h3>
            <ul className="lw-check-lines">{items.map((it) => <Item key={it.id} it={it} check={check} />)}</ul>
          </section>
        );
      })}
    </article>
  );
}

function Diff({ ops }: { ops: CitationItem['diff'] }) {
  return <p className="lw-redline lw-serif" style={{ margin: 0 }}>{ops.map(([op, t], i) =>
    op === 'del' ? <del key={i}>{t}</del> : op === 'ins' ? <ins key={i}>{t}</ins> : <React.Fragment key={i}>{t}</React.Fragment>)}</p>;
}

function Item({ it, check }: { it: CitationItem; check: CitationCheck }) {
  const matter = useMatter();
  const [noting, setNoting] = useState(false);
  const canNote = ['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '');
  const [label, tone] = STATUS[it.status];
  const here = `/m/${matter.number}/documents/${encodeURIComponent(check.document_id)}/read?v=${encodeURIComponent(check.version_id)}&at=${it.start}-${it.end}`;
  const t = it.target;
  const there = t ? `/m/${matter.number}/documents/${encodeURIComponent(t.document_id)}/read?v=${encodeURIComponent(t.version_id)}&at=${t.start}-${t.end}` : null;
  const own = t?.document_id === check.document_id;
  const missing = it.status === 'missing';
  const where = t && !missing ? `${t.clause ? `Clause ${t.clause}` : 'A passage'}${own ? ' of this document' : ` of ${t.title}`}` : null;
  return (
    <li aria-label={`${it.kind === 'quote' ? 'Quotation' : it.text}: ${it.malformed.length ? 'cannot be real' : label}`} data-class={it.status === 'missing' || it.status === 'altered' || it.malformed.length ? 'outside' : it.status}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        {it.kind === 'quote' ? <span className="lw-meta">{it.cited_as ? `Quoted as ${it.cited_as}` : 'Quoted without a clause'}</span> : <b className="lw-serif">{it.text}</b>}
        <StatusPill tone={tone}>{label}</StatusPill>
        {it.malformed.length > 0 && <StatusPill tone="conflict">Cannot be real</StatusPill>}
        <span className="lw-cite-actions">
          {canNote && !noting && <button type="button" className="lw-cite-act lw-meta" aria-label={`Add a note on ${it.kind === 'quote' ? 'this quotation' : it.text}`}
            onClick={() => setNoting(true)}><MessageSquarePlus aria-hidden="true" size={14} />Note</button>}
          <Link className="lw-cite-act lw-meta" to={here}>Open at the citation</Link>
        </span>
      </div>
      {it.kind === 'quote' && <blockquote className="lw-serif lw-check-quote">{`“${it.text}”`}</blockquote>}
      {it.status === 'altered' && it.diff.length > 0 && (
        <div className="lw-cite-diff" aria-label="What the source says">
          <p className="lw-meta" style={{ margin: '2px 0' }}>{`Compared with ${where}: struck words are the source's, underlined words the quotation's.`}</p>
          <Diff ops={it.diff} />
        </div>
      )}
      {where && it.status !== 'altered' && <p className="lw-meta" style={{ margin: '2px 0' }}>{it.kind === 'quote' ? `Matches ${where}.` : `${where}.`}</p>}
      {t && missing && !own && <p className="lw-meta" style={{ margin: '2px 0' }}>{'Searched '}<Link className="lw-linklike" to={`/m/${matter.number}/documents/${encodeURIComponent(t.document_id)}/read?v=${encodeURIComponent(t.version_id)}`}>{t.title}</Link>{` (version in force when checked).`}</p>}
      {t && there && !missing && (
        <details><summary className="lw-meta">Show the passage</summary>
          <blockquote className="lw-serif lw-check-quote">{t.passage}</blockquote>
          <Link className="lw-linklike lw-meta" to={there}>Open the passage</Link></details>
      )}
      {it.malformed.length > 0 && <p className="lw-meta" style={{ margin: '2px 0', color: 'var(--conflict)' }}>{`This citation cannot be real: ${it.malformed.join('; ')}`.replace(/\.?$/, '.')}</p>}
      {it.note && <p className="lw-meta" style={{ margin: '2px 0' }}>{it.note}</p>}
      <Notes it={it} check={check} open={canNote && noting} close={() => setNoting(false)} />
    </li>
  );
}

function Notes({ it, check, open, close }: { it: CitationItem; check: CitationCheck; open: boolean; close: () => void }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/citation-checks/${encodeURIComponent(check.id)}/notes`, { item: it.id, note: text.trim() });
      setText(''); close();
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, check.document_id), 'citation-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The note could not be saved.'); }
    finally { setBusy(false); }
  };
  return (
    <>
      {it.notes.map((n, i) => <p key={i} className="lw-meta" style={{ margin: '2px 0' }}><b>{n.by}</b>{`, ${when(n.at)}: ${n.note}`}</p>)}
      {open && (
        <form className="lw-check-decide" aria-label={`Note on ${it.kind === 'quote' ? 'this quotation' : it.text}`} onSubmit={add}>
          <input aria-label="Your note" className="lw-input lw-input-sm" autoFocus placeholder={it.kind === 'authority' ? 'Checked in a licensed source, or what to change' : 'A note for the team'}
            value={text} onChange={(e) => setText(e.target.value)} maxLength={1000} />
          <button type="submit" className="lw-btn lw-btn-sm" disabled={busy || !text.trim()}>Save note</button>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => { close(); setText(''); }}>Cancel</button>
        </form>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </>
  );
}
