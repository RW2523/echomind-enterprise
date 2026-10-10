import React, { useEffect, useState } from 'react';
import { Link } from 'react-router';
import { Check, Play, RotateCcw, X } from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { ddPath, useChecklists, useDdRun, useDdRuns } from '../api/queries';
import type { DdIssue, DdRunView } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const SEVERITY: Record<DdIssue['severity'], [string, Tone]> = { high: ['High', 'conflict'], medium: ['Medium', 'attention'], low: ['Low', 'neutral'] };
const STATUS: Record<DdIssue['status'], [string, Tone]> = {
  preliminary: ['Preliminary — not reviewed', 'attention'], verified: ['Verified', 'verified'], dismissed: ['Dismissed', 'neutral'],
};
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** Due diligence (S3.7, D-051): a matter's documents checked against an approved firm checklist: what is
 *  missing, what answers each item, and an issue register in which every finding quotes its sentence and
 *  stays preliminary until a reviewer verifies or dismisses it. */
export default function MatterDiligence() {
  const matter = useMatter();
  const qc = useQueryClient();
  const runs = useDdRuns(matter.id);
  const lists = useChecklists();
  const [chosen, setChosen] = useState('');
  const [runId, setRunId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const canRun = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const approved = (lists.data?.checklists ?? []).filter((c) => c.status === 'approved');
  useEffect(() => { if (!runId && runs.data?.length) setRunId(runs.data[0].id); }, [runs.data, runId]);
  const start = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ run: { id: string } }>(`${ddPath(matter.id)}/runs`, { checklist_id: chosen });
      setRunId(r.run.id);
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'dd-runs'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The run could not be started.'); }
    finally { setBusy(false); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Due diligence" />
      <div className="lw-content">
        {canRun && (
          <section className="lw-paper lw-sheet" aria-label="Start a due-diligence run" aria-busy={busy}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <div style={{ minWidth: 320 }}>
                <Field label="Checklist">{(id) => <select id={id} className="lw-input" value={chosen} onChange={(e) => setChosen(e.target.value)}>
                  <option value="">Choose an approved checklist</option>{approved.map((c) => <option key={c.id} value={c.id}>{`${c.name} (version ${c.version})`}</option>)}</select>}</Field>
              </div>
              <button type="button" className="lw-btn lw-btn-primary" disabled={!chosen || busy} onClick={start}><Play />{busy ? 'Starting…' : 'Run against this matter'}</button>
            </div>
            <p className="lw-meta" style={{ margin: '8px 0 0' }}>Every document you can see in this matter is matched to the checklist's items; the contracts are read for the questions each item asks. It runs in the background.</p>
            {error && <ErrorNote>{error}</ErrorNote>}
          </section>
        )}
        {runs.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : runs.isError ? <ErrorNote>The runs could not be loaded.</ErrorNote>
            : !runs.data.length ? <EmptyState title="No runs yet">Run an approved checklist to see what is missing and what each document raises.</EmptyState>
              : (
                <>
                  {runs.data.length > 1 && (
                    <nav aria-label="Runs" className="lw-meta" style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      {runs.data.map((r) => <button key={r.id} type="button" className="lw-cite-act" aria-current={r.id === runId ? 'true' : undefined}
                        onClick={() => setRunId(r.id)}>{`${r.checklist.name} v${r.checklist.version} · ${when(r.created_at)}`}</button>)}
                    </nav>
                  )}
                  {runId && <RunView runId={runId} />}
                </>
              )}
      </div>
    </>
  );
}

function RunView({ runId }: { runId: string }) {
  const matter = useMatter();
  const q = useDdRun(matter.id, runId);
  if (q.isPending) return <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>;
  if (q.isError) return <ErrorNote>This run could not be loaded.</ErrorNote>;
  const v: DdRunView = q.data;
  const s = v.summary;
  if (v.run.status === 'queued' || v.run.status === 'running') {
    return <div className="lw-paper lw-sheet" aria-busy="true"><p role="status" className="lw-meta" style={{ margin: 0 }}>{`Reading the matter's documents against ${v.run.checklist.name}…`}</p><Skeleton /></div>;
  }
  if (v.run.status === 'failed') return <ErrorNote>{`This run failed (${v.run.error ?? 'unknown error'}). Start it again.`}</ErrorNote>;
  const sections = Array.from(new Set(v.items.map((i) => i.section)));
  return (
    <>
      <article className="lw-paper lw-sheet" aria-label="Issue register">
        <div className="lw-minutes-h">
          <div>
            <h2 className="lw-h2" style={{ margin: 0 }}>{`${v.run.checklist.name}, version ${v.run.checklist.version}`}</h2>
            <p className="lw-meta" style={{ margin: '2px 0 0' }}>{`Run by ${v.run.created_by}, ${when(v.run.created_at)} · ${v.run.documents_read ?? 0} ${(v.run.documents_read ?? 0) === 1 ? 'document' : 'documents'} read`}</p>
          </div>
          <StatusPill tone="neutral">Rules only · preliminary until reviewed</StatusPill>
        </div>
        <p className="lw-meta" role="status" style={{ margin: '0 0 10px' }}>{[`${s.items - s.missing} of ${s.items} items answered`, `${s.missing} missing`,
          `${s.issues} ${s.issues === 1 ? 'issue' : 'issues'} (${s.high} high, ${s.medium} medium, ${s.low} low)`, `${s.verified} verified`, `${s.dismissed} dismissed`,
          `${s.preliminary} awaiting review`].join(' · ')}</p>
        {v.issues.length === 0 ? <p className="lw-meta">No issues were found in the documents that answer the checklist.</p>
          : <ul className="lw-check-lines" aria-label="Issues">{v.issues.map((i) => <Issue key={i.id} i={i} />)}</ul>}
      </article>
      <section className="lw-paper lw-sheet" aria-label="Missing documents">
        <h2 className="lw-h2" style={{ margin: 0 }}>Missing documents</h2>
        {v.missing.length === 0 ? <p className="lw-meta">Every item has a document that answers it.</p> : (
          <ul className="lw-check-lines">{v.missing.map((m) => <li key={m.item_id} aria-label={m.title}><b>{m.title}</b><span className="lw-meta">{m.section}</span></li>)}</ul>
        )}
      </section>
      <section className="lw-paper lw-sheet" aria-label="Documents for each item">
        <h2 className="lw-h2" style={{ margin: 0 }}>Documents for each item</h2>
        {sections.map((sec) => (
          <div key={sec}>
            <h3 className="lw-h3" style={{ margin: '12px 0 2px' }}>{sec}</h3>
            <ul className="lw-check-lines">{v.items.filter((i) => i.section === sec).map((i) => (
              <li key={i.item_id} aria-label={i.title} data-class={i.missing ? 'outside' : undefined}>
                <b>{i.title}</b>
                {i.missing ? <span className="lw-meta">Not found in this matter</span>
                  : <span className="lw-meta">{i.documents.map((d, k) => <React.Fragment key={d.id}>{k ? ', ' : ''}
                    <Link className="lw-linklike" to={`/m/${matter.number}/documents/${encodeURIComponent(d.id)}/read`}>{d.title}</Link></React.Fragment>)}</span>}
              </li>))}</ul>
          </div>
        ))}
      </section>
    </>
  );
}

function Issue({ i }: { i: DdIssue }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [mode, setMode] = useState<'' | 'dismiss' | 'verify' | 'trail'>('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const reviewer = ['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '');
  const [sev, sevTone] = SEVERITY[i.severity];
  const [label, tone] = STATUS[i.status];
  const at = `/m/${matter.number}/documents/${encodeURIComponent(i.document_id)}/read?v=${encodeURIComponent(i.version_id)}&at=${i.evidence.start}-${i.evidence.end}`;
  const send = async (action: 'verify' | 'dismiss' | 'reopen') => {
    setError('');
    try {
      await api.patch(`${ddPath(matter.id)}/issues/${encodeURIComponent(i.id)}`, { expected_version: i.version, action, reason: reason.trim() || null });
      setMode(''); setReason('');
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'dd-run'] });
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'dd-runs'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The decision could not be saved.'); }
  };
  return (
    <li aria-label={`${i.title}: ${label}`} data-class={i.severity === 'high' && i.status !== 'dismissed' ? 'outside' : i.status}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <b className="lw-serif">{i.title}</b>
        <StatusPill tone={sevTone}>{sev}</StatusPill>
        <StatusPill tone={tone}>{i.status !== 'preliminary' && i.reviewer ? `${label} by ${i.reviewer}` : label}</StatusPill>
        <span className="lw-cite-actions">
          <button type="button" className="lw-cite-act lw-meta" aria-expanded={mode === 'trail'} onClick={() => setMode(mode === 'trail' ? '' : 'trail')}>Trail</button>
          <Link className="lw-cite-act lw-meta" to={at}>Open the words</Link>
        </span>
      </div>
      <p className="lw-meta" style={{ margin: '2px 0' }}>{i.document_title}</p>
      {i.detail && <p style={{ margin: '2px 0' }}>{i.detail}</p>}
      <blockquote className="lw-serif lw-check-quote">{`“${i.evidence.quote}”`}</blockquote>
      {(i.related ?? []).map((r, k) => (
        <div key={k} className="lw-dd-related">
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{'Against '}<Link className="lw-linklike" to={`/m/${matter.number}/documents/${encodeURIComponent(r.document_id)}/read?v=${encodeURIComponent(r.version_id)}&at=${r.start}-${r.end}`}>{r.document_title}</Link></p>
          <blockquote className="lw-serif lw-check-quote">{`“${r.quote}”`}</blockquote>
        </div>
      ))}
      {reviewer && mode !== 'dismiss' && mode !== 'verify' && (
        <div className="lw-check-decide" role="group" aria-label={`Review ${i.title}`}>
          {i.status === 'preliminary' ? <>
            <button type="button" className="lw-btn lw-btn-sm" onClick={() => setMode('verify')}><Check />Verify</button>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setMode('dismiss')}><X />Dismiss</button>
          </> : <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => send('reopen')}><RotateCcw />Reopen</button>}
        </div>
      )}
      {(mode === 'verify' || mode === 'dismiss') && (
        <form className="lw-check-decide" aria-label={mode === 'verify' ? `Verify ${i.title}` : `Dismiss ${i.title}`} onSubmit={(e) => { e.preventDefault(); send(mode); }}>
          <input aria-label={mode === 'verify' ? 'Note (optional)' : 'Why it is not an issue'} className="lw-input lw-input-sm"
            placeholder={mode === 'verify' ? 'Note for the report (optional)' : 'Why it is not an issue'} value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="submit" className="lw-btn lw-btn-sm lw-btn-primary" disabled={mode === 'dismiss' && reason.trim().length < 3}>{mode === 'verify' ? 'Verify' : 'Dismiss'}</button>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => { setMode(''); setReason(''); }}>Cancel</button>
        </form>
      )}
      {mode === 'trail' && <Trail i={i} />}
      {error && <ErrorNote>{error}</ErrorNote>}
    </li>
  );
}

function Trail({ i }: { i: DdIssue }) {
  const matter = useMatter();
  const q = useQuery({ queryKey: ['matter', matter.id, 'dd-issue', i.id, i.version],
    queryFn: () => api.get<{ trail: { action: string; reason: string | null; by: string | null; at: string }[] }>(`${ddPath(matter.id)}/issues/${encodeURIComponent(i.id)}`) });
  if (q.isPending) return <Skeleton />;
  if (q.isError) return <ErrorNote>The trail could not be loaded.</ErrorNote>;
  const said: Record<string, string> = { found: 'found by the run', verify: 'verified', dismiss: 'dismissed', reopen: 'reopened' };
  return (
    <ol className="lw-meta" aria-label={`Trail of ${i.title}`} style={{ margin: '4px 0', paddingLeft: 18 }}>
      {q.data.trail.map((t, k) => <li key={k}>{`${t.action === 'found' ? 'Found by the run' : `${t.by ?? 'someone'} ${said[t.action] ?? t.action}`}, ${when(t.at)}${t.reason ? ` — ${t.reason}` : ''}`}</li>)}
    </ol>
  );
}

