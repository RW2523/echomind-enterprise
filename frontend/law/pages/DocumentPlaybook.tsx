import React, { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { ArrowLeft, FilePen, ListChecks, ThumbsDown, ThumbsUp } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, useDocument, usePlaybookChecks, usePlaybooks } from '../api/queries';
import type { CheckClass, CheckLine, PlaybookCheck } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const CLASS: Record<CheckClass, [string, Tone]> = {
  standard: ['Standard', 'verified'], fallback: ['Fallback', 'attention'], outside: ['Outside the playbook', 'conflict'],
  not_found: ['Not found', 'neutral'], unclear: ['Unclear — a lawyer decides', 'attention'],
};
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** A document checked against a playbook (S3.2, D-046): each topic's clause, its class, the exact
 *  words it was judged on and the position it was compared with. Advisory: a lawyer confirms each line. */
export default function DocumentPlaybook() {
  const { docId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const doc = useDocument(matter.id, docId);
  const playbooks = usePlaybooks();
  const checks = usePlaybookChecks(matter.id, docId);
  const usable = (playbooks.data?.playbooks ?? []).filter((p) => p.positions.some((x) => x.status === 'approved'));
  const [chosen, setChosen] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const canRun = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const run = async () => {
    setBusy(true); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(docId)}/playbook-check`, { playbook_id: chosen });
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, docId), 'playbook-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The check could not be run.'); }
    finally { setBusy(false); }
  };
  const latest = checks.data?.[0];
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={doc.data ? `${doc.data.title} — playbook check` : 'Playbook check'}>
        <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents/${encodeURIComponent(docId)}/read`}><ArrowLeft />Document</Link>
      </Toolbar>
      <div className="lw-content">
        {canRun && (
          <section className="lw-paper lw-sheet" aria-label="Check against a playbook" aria-busy={busy}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <div style={{ minWidth: 280 }}>
                <Field label="Playbook">{(id) => <select id={id} className="lw-input" value={chosen} onChange={(e) => setChosen(e.target.value)}>
                  <option value="">Choose an approved playbook</option>{usable.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>}</Field>
              </div>
              <button type="button" className="lw-btn lw-btn-primary" disabled={!chosen || busy} onClick={run}><ListChecks />{busy ? 'Checking each topic…' : 'Check this document'}</button>
            </div>
            {playbooks.data && usable.length === 0 && <p className="lw-meta" style={{ margin: '8px 0 0' }}>No playbook has approved positions yet.</p>}
            {error && <ErrorNote>{error}</ErrorNote>}
          </section>
        )}
        {checks.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : checks.isError ? <ErrorNote>The checks could not be loaded.</ErrorNote>
            : !latest ? <EmptyState title="Not checked yet">Choose a playbook to see, topic by topic, where this document stands.</EmptyState>
              : <CheckView check={latest} older={checks.data!.length - 1} />}
      </div>
    </>
  );
}

function CheckView({ check, older }: { check: PlaybookCheck; older: number }) {
  const s = check.summary;
  return (
    <article className="lw-paper lw-sheet" aria-label="Playbook check">
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{check.playbook.name}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{`Version ${check.inputs.version_number} of the document · checked by ${check.created_by}, ${when(check.created_at)}${older ? ` · ${older} earlier ${older === 1 ? 'check' : 'checks'}` : ''}`}</p>
        </div>
        <StatusPill tone="neutral">Advisory — a lawyer decides</StatusPill>
      </div>
      <p className="lw-meta" role="status" style={{ margin: '0 0 10px' }}>{`${s.standard} standard · ${s.fallback} fallback · ${s.outside} outside the playbook · ${s.not_found} not found${s.unclear ? ` · ${s.unclear} unclear` : ''}`}</p>
      <ul className="lw-check-lines" aria-label="Topics">
        {check.results.map((r) => <Line key={r.topic} r={r} check={check} />)}
      </ul>
      <Revise check={check} />
    </article>
  );
}

/** S3.3: the lines outside the playbook revised to the firm's positions, as a redline to review. */
function Revise({ check }: { check: PlaybookCheck }) {
  const matter = useMatter();
  const navigate = useNavigate();
  const outside = check.results.filter((r) => r.classification === 'outside' && r.clause);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  if (!outside.length || !['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') || matter.status !== 'open') return null;
  const go = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ redline: { id: string } }>(`/api/legal/matters/${encodeURIComponent(matter.id)}/playbook-checks/${encodeURIComponent(check.id)}/redline`, { choices });
      navigate(`/m/${matter.number}/redlines/${encodeURIComponent(r.redline.id)}`);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The revisions could not be proposed.'); setBusy(false); }
  };
  return (
    <section className="lw-revise" aria-label="Revise to the playbook" aria-busy={busy}>
      <h3 className="lw-h3" style={{ margin: 0 }}>Revise to the playbook</h3>
      <p className="lw-meta" style={{ margin: '2px 0 8px' }}>Each clause outside the playbook is revised to the firm's approved wording, fitted only to this contract's own terms. You review the redline before anything is filed.</p>
      {outside.map((r) => r.position.fallbacks.length > 0 && (
        <label key={r.topic} className="lw-meta" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>{`${r.title}:`}
          <select className="lw-input lw-input-sm" style={{ maxWidth: 260 }} aria-label={`Position for ${r.title}`} value={choices[r.topic] ?? 'standard'}
            onChange={(e) => setChoices((c) => ({ ...c, [r.topic]: e.target.value }))}>
            <option value="standard">The standard</option>
            {r.position.fallbacks.map((f, i) => <option key={i} value={`fallback ${i + 1}`}>{`Fallback ${i + 1}${f.note ? ` (${f.note})` : ''}`}</option>)}
          </select></label>
      ))}
      <div><button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={go}><FilePen />{busy ? 'Proposing…' : `Propose ${outside.length} ${outside.length === 1 ? 'revision' : 'revisions'}`}</button></div>
      {error && <ErrorNote>{error}</ErrorNote>}
    </section>
  );
}

function Line({ r, check }: { r: CheckLine; check: PlaybookCheck }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [label, tone] = CLASS[r.classification];
  const canDecide = ['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '');
  const href = r.quote ? `/m/${matter.number}/documents/${encodeURIComponent(check.document_id)}/read?v=${encodeURIComponent(check.version_id)}&at=${r.quote.start}-${r.quote.end}` : null;
  const decide = async (decision: 'agree' | 'disagree') => {
    setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/playbook-checks/${encodeURIComponent(check.id)}/decisions`,
        { topic: r.topic, decision, reason: reason.trim() || null });
      await qc.invalidateQueries({ queryKey: [...keys.document(matter.id, check.document_id), 'playbook-checks'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The decision could not be saved.'); }
  };
  const compared = r.classification === 'fallback' && r.matched ? r.position.fallbacks[r.matched - 1]?.text : r.position.standard;
  return (
    <li aria-label={`${r.title}: ${label}`} data-class={r.classification}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <b>{r.title}</b>
        <StatusPill tone={tone}>{r.classification === 'fallback' && r.matched ? `Fallback ${r.matched}` : label}</StatusPill>
        {r.method === 'rule' && <span className="lw-meta">Decided by the figures</span>}
        {r.clause?.number && <span className="lw-meta">{`Clause ${r.clause.number}${r.clause.heading ? ` · ${r.clause.heading}` : ''}`}</span>}
      </div>
      {r.quote && <blockquote className="lw-serif lw-check-quote">{`“${r.quote.text}”`}{href && <> <Link className="lw-linklike lw-meta" to={href}>Open at the quote</Link></>}</blockquote>}
      {r.reason && <p className="lw-meta" style={{ margin: '2px 0' }}>{r.reason}</p>}
      <details><summary className="lw-meta">{`Compared with the ${r.classification === 'fallback' ? `fallback ${r.matched}` : 'standard'} (position version ${r.position.version})`}</summary>
        <p className="lw-serif lw-meta" style={{ margin: '4px 0' }}>{compared}</p></details>
      {r.decision ? (
        <p className="lw-meta" style={{ margin: '4px 0 0' }}>{`${r.decision.decision === 'agree' ? 'Agreed' : 'Disagreed'} by ${r.decision.by}, ${when(r.decision.at)}${r.decision.reason ? ` — ${r.decision.reason}` : ''}`}</p>
      ) : canDecide && (
        <div className="lw-check-decide" role="group" aria-label={`Decide on ${r.title}`}>
          <button type="button" className="lw-btn lw-btn-sm" onClick={() => decide('agree')}><ThumbsUp />Agree</button>
          <input aria-label={`Why you disagree on ${r.title}`} className="lw-input lw-input-sm" placeholder="Why you disagree" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={reason.trim().length < 3} onClick={() => decide('disagree')}><ThumbsDown />Disagree</button>
        </div>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </li>
  );
}
