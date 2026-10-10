import React, { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { ArrowLeft, BookA, FilePen, MessageSquarePlus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, useDefinitionChecks, useDocument } from '../api/queries';
import type { DefinitionCheck, DefinitionFinding, DefinitionFindingType } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const TYPE: Record<DefinitionFindingType, [string, Tone]> = {
  duplicate: ['Defined twice', 'conflict'], unused: ['Not used', 'neutral'], undefined: ['Not defined', 'attention'],
  inconsistent: ['Another form', 'attention'], before_definition: ['Used before defined', 'neutral'],
  gap: ['Gap', 'attention'], repeat: ['Repeated', 'conflict'], broken: ['Broken', 'conflict'], outdated: ['Outdated', 'attention'],
};
const RANK = { high: 0, medium: 1, low: 2 } as const;
const GROUPS = [['definition', 'Defined terms'], ['reference', 'Cross-references'], ['numbering', 'Numbering']] as const;
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const n = (k: number, one: string, many = `${one}s`) => `${k} ${k === 1 ? one : many}`;

/** A document read for its own consistency (S3.5, D-049): defined terms, numbering and cross-references,
 *  each at its exact place. Rules only. A correction is never made here: the ones a lawyer picks become a
 *  redline that a second lawyer approves before it is filed. */
export default function DocumentDefinitions() {
  const { docId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const doc = useDocument(matter.id, docId);
  const checks = useDefinitionChecks(matter.id, docId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const canRun = !!matter.my_role && matter.my_role !== 'break_glass';           // any member (research.query)
  const run = async () => {
    setBusy(true); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(docId)}/definitions-check`, {});
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, docId), 'definitions-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The check could not be run.'); }
    finally { setBusy(false); }
  };
  const latest = checks.data?.[0];
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={doc.data ? `${doc.data.title} — definitions and references` : 'Definitions and references'}>
        <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents/${encodeURIComponent(docId)}/read`}><ArrowLeft />Document</Link>
      </Toolbar>
      <div className="lw-content">
        {canRun && (
          <section className="lw-paper lw-sheet" aria-label="Check definitions and references" aria-busy={busy}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={run}><BookA />{busy ? 'Reading the document…' : latest ? 'Check again' : 'Check definitions and references'}</button>
              <p className="lw-meta" style={{ margin: 0, flex: 1, minWidth: 240 }}>Defined terms, clause numbering and cross-references in the current version. Nothing is changed unless a lawyer picks corrections and a second lawyer approves them.</p>
            </div>
            {error && <ErrorNote>{error}</ErrorNote>}
          </section>
        )}
        {checks.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : checks.isError ? <ErrorNote>The checks could not be loaded.</ErrorNote>
            : !latest ? <EmptyState title="Not checked yet">Check this document to see terms defined twice or never, numbering gaps, and references that no longer point where they say.</EmptyState>
              : <CheckView key={latest.id} check={latest} older={checks.data!.length - 1} />}
      </div>
    </>
  );
}

function CheckView({ check, older }: { check: DefinitionCheck; older: number }) {
  const matter = useMatter();
  const navigate = useNavigate();
  const [picks, setPicks] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const s = check.summary;
  const canPropose = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const parts = [
    s.duplicate && `${n(s.duplicate, 'term')} defined twice`, s.unused && `${s.unused} not used`, s.undefined && `${s.undefined} not defined`,
    s.inconsistent && `${s.inconsistent} in another form`, s.before_definition && `${s.before_definition} used before defined`,
    s.broken && n(s.broken, 'broken reference'), s.outdated && n(s.outdated, 'outdated reference'),
    s.gap && n(s.gap, 'numbering gap'), s.repeat && n(s.repeat, 'repeated number'),
  ].filter(Boolean);
  const chosen = Object.keys(picks).length;
  const propose = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ redline: { id: string } }>(`/api/legal/matters/${encodeURIComponent(matter.id)}/definitions-checks/${encodeURIComponent(check.id)}/redline`,
        { fixes: Object.entries(picks).map(([finding, fix]) => ({ finding, fix })) });
      navigate(`/m/${matter.number}/redlines/${encodeURIComponent(r.redline.id)}`);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The corrections could not be proposed.'); setBusy(false); }
  };
  return (
    <article className="lw-paper lw-sheet" aria-label="Definitions check">
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{check.version_number ? `Version ${check.version_number}` : 'This version'}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{`${n(s.definitions, 'defined term')} · checked by ${check.created_by}, ${when(check.created_at)}${older ? ` · ${older} earlier ${older === 1 ? 'check' : 'checks'}` : ''}`}</p>
        </div>
        <StatusPill tone="neutral">Rules only · advisory</StatusPill>
      </div>
      <p className="lw-meta" role="status" style={{ margin: '0 0 10px' }}>{parts.length ? parts.join(' · ') : 'Nothing found: every defined term is used, numbering runs on and every reference points to a clause that exists.'}</p>
      {GROUPS.map(([kind, title]) => {
        const items = check.findings.filter((f) => f.kind === kind).sort((a, b) => RANK[a.severity] - RANK[b.severity] || a.start - b.start);
        if (!items.length) return null;
        return (
          <section key={kind} aria-label={title}>
            <h3 className="lw-h3" style={{ margin: '14px 0 2px' }}>{title}</h3>
            <ul className="lw-check-lines">{items.map((f) => (
              <Finding key={f.id} f={f} check={check} pick={picks[f.id]} canPick={canPropose}
                onPick={(fix) => setPicks((p) => { const q = { ...p }; if (fix === null) delete q[f.id]; else q[f.id] = fix; return q; })} />))}</ul>
          </section>
        );
      })}
      {canPropose && check.findings.some((f) => f.fixes.length) && (
        <div className="lw-revise" role="group" aria-label="Corrections">
          <p className="lw-meta" style={{ margin: 0 }}>{chosen ? `${n(chosen, 'correction')} picked. They become a redline with tracked changes for a second lawyer to approve; the document is not changed until it is filed.`
            : 'Pick a correction under any finding that offers one.'}</p>
          <div><button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={!chosen || busy} onClick={propose}><FilePen />{busy ? 'Proposing…' : chosen ? `Propose ${n(chosen, 'correction')}` : 'Propose corrections'}</button></div>
          {error && <ErrorNote>{error}</ErrorNote>}
        </div>
      )}
    </article>
  );
}

function Finding({ f, check, pick, canPick, onPick }: { f: DefinitionFinding; check: DefinitionCheck; pick: number | undefined; canPick: boolean; onPick: (fix: number | null) => void }) {
  const matter = useMatter();
  const [noting, setNoting] = useState(false);
  const canNote = ['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '');
  const [label, tone] = TYPE[f.type];
  const at = (a: number, b: number) => `/m/${matter.number}/documents/${encodeURIComponent(check.document_id)}/read?v=${encodeURIComponent(check.version_id)}&at=${a}-${Math.max(b, a + 1)}`;
  const name = f.kind === 'numbering' ? f.message.split(':')[0] : f.text;
  return (
    <li aria-label={`${name}: ${label}`} data-class={f.severity === 'high' ? 'outside' : f.type}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        {f.kind !== 'numbering' && <b className="lw-serif">{f.text}</b>}
        <StatusPill tone={tone}>{label}</StatusPill>
        {f.where && f.type !== 'duplicate' && <span className="lw-meta">{f.where.charAt(0).toUpperCase() + f.where.slice(1)}</span>}
        <span className="lw-cite-actions">
          {canNote && !noting && <button type="button" className="lw-cite-act lw-meta" aria-label={`Add a note on ${name}`} onClick={() => setNoting(true)}>
            <MessageSquarePlus aria-hidden="true" size={14} />Note</button>}
          <Link className="lw-cite-act lw-meta" to={at(f.start, f.end)}>Open at the place</Link>
        </span>
      </div>
      <p className="lw-meta" style={{ margin: '2px 0' }}>{f.message}</p>
      {f.places.length > 1 && (
        <p className="lw-meta" style={{ margin: '2px 0' }}>{f.type === 'duplicate' ? 'Defined in ' : f.type === 'repeat' ? 'At ' : 'Also at '}{f.places.map((p, i) => (
          <React.Fragment key={i}>{i ? (i === f.places.length - 1 ? ' and ' : ', ') : ''}<Link className="lw-linklike" to={at(p.start, p.end)}>{p.where ?? `place ${i + 1}`}</Link></React.Fragment>))}</p>
      )}
      {f.fixes.length > 0 && canPick && (
        <fieldset className="lw-def-fixes">
          <legend className="lw-meta">Correction</legend>
          <label className="lw-meta"><input type="radio" name={`fix-${f.id}`} checked={pick === undefined} onChange={() => onPick(null)} /> None</label>
          {f.fixes.map((x, i) => (
            <label key={i} className="lw-meta"><input type="radio" name={`fix-${f.id}`} checked={pick === i} onChange={() => onPick(i)} />
              {` ${x.label}`}{x.original !== x.revised && <span className="lw-redline lw-serif"> <del>{x.original}</del>{x.revised && <> <ins>{x.revised}</ins></>}</span>}</label>
          ))}
        </fieldset>
      )}
      <Notes f={f} check={check} open={canNote && noting} close={() => setNoting(false)} />
    </li>
  );
}

function Notes({ f, check, open, close }: { f: DefinitionFinding; check: DefinitionCheck; open: boolean; close: () => void }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const add = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/definitions-checks/${encodeURIComponent(check.id)}/notes`, { finding: f.id, note: text.trim() });
      setText(''); close();
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, check.document_id), 'definitions-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The note could not be saved.'); }
    finally { setBusy(false); }
  };
  return (
    <>
      {f.notes.map((x, i) => <p key={i} className="lw-meta" style={{ margin: '2px 0' }}><b>{x.by}</b>{`, ${when(x.at)}: ${x.note}`}</p>)}
      {open && (
        <form className="lw-check-decide" aria-label={`Note on ${f.kind === 'numbering' ? f.message.split(':')[0] : f.text}`} onSubmit={add}>
          <input aria-label="Your note" className="lw-input lw-input-sm" autoFocus placeholder="A note for the team" value={text} onChange={(e) => setText(e.target.value)} maxLength={1000} />
          <button type="submit" className="lw-btn lw-btn-sm" disabled={busy || !text.trim()}>Save note</button>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => { close(); setText(''); }}>Cancel</button>
        </form>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </>
  );
}
