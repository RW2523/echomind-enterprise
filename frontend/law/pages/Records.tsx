import React, { useState } from 'react';
import { CheckCircle2, ShieldPlus, XCircle } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, useDeletions, useHolds, useRetention } from '../api/queries';
import type { Deletion, Hold, RetentionPolicy } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import { ReportList } from '../ui/records';
import NotAvailable from './NotAvailable';

type Tab = 'deletions' | 'holds' | 'retention';
const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');
const DELETION: Record<Deletion['status'], [string, Tone]> = {
  requested: ['Awaiting approval', 'attention'], blocked: ['Blocked by legal hold', 'hold'], purged: ['Deleted', 'neutral'],
  rejected: ['Rejected — restored', 'neutral'], restored: ['Restored', 'neutral'],
};
const HOLD: Record<Hold['status'], [string, Tone]> = {
  requested: ['Awaiting approval', 'attention'], active: ['In force', 'hold'], release_requested: ['Release requested', 'attention'],
  released: ['Released', 'neutral'], rejected: ['Rejected', 'neutral'],
};
const OBJECT: Record<string, string> = { document: 'Document', meeting_transcript: 'Meeting transcript' };

/** Records: deletion approvals, legal holds and retention (records administrators, managing partners). */
export default function Records() {
  const me = useSession();
  const allowed = hasFirmRole(me, 'records_admin', 'managing_partner');
  const [tab, setTab] = useState<Tab>('deletions');
  if (!allowed) return <NotAvailable />;
  return (
    <>
      <Toolbar title="Records" />
      <div className="lw-content" style={{ maxWidth: 1040 }}>
        <div className="lw-seg" role="tablist" aria-label="Records">
          {([['deletions', 'Deletion requests'], ['holds', 'Legal holds'], ['retention', 'Retention']] as [Tab, string][]).map(([id, label]) => (
            <button key={id} type="button" role="tab" aria-selected={tab === id} className="lw-seg-item" onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <div role="tabpanel" style={{ display: 'grid', gap: 16 }}>
          {tab === 'deletions' && <Deletions me={me.user.username} />}
          {tab === 'holds' && <Holds me={me.user.username} />}
          {tab === 'retention' && <Retention me={me.user.username} canManage={hasFirmRole(me, 'records_admin')} canApprove={hasFirmRole(me, 'managing_partner')} />}
        </div>
      </div>
    </>
  );
}

// ── Deletions ──────────────────────────────────────────────────────────────────
function Deletions({ me }: { me: string }) {
  const qc = useQueryClient();
  const list = useDeletions(true);
  const [error, setError] = useState('');
  const act = async (d: Deletion, verb: 'approve' | 'reject') => {
    setError('');
    try { await api.post(`/api/legal/deletions/${encodeURIComponent(d.id)}/${verb}`); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The decision could not be recorded.'); }
    await qc.invalidateQueries({ queryKey: keys.deletions });
  };
  if (list.isPending) return <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>;
  if (list.isError) return <ErrorNote>The deletion requests could not be loaded.</ErrorNote>;
  const pending = list.data.filter((d) => d.status === 'requested' || d.status === 'blocked');
  const done = list.data.filter((d) => !pending.includes(d));
  return (
    <>
      <p className="lw-meta" style={{ margin: 0 }}>Approving destroys the item for good and checks that nothing is left. Material under a legal hold is never deleted. The audit trail is always kept.</p>
      {error && <ErrorNote>{error}</ErrorNote>}
      {pending.length === 0 ? <EmptyState title="Nothing waiting">Deletion requests from matter teams and retention appear here.</EmptyState> : pending.map((d) => (
        <section key={d.id} className="lw-paper lw-sheet" aria-label={d.object_label ?? 'Deletion request'} style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="lw-eyebrow">{OBJECT[d.object_type]} · {d.matter?.number} {d.matter?.title}</div>
              <h2 className="lw-h2" style={{ marginTop: 4 }}>{d.object_label}</h2>
              <p className="lw-meta" style={{ margin: '4px 0 0' }}>{d.origin === 'retention' ? 'Proposed by a retention policy' : `Requested by ${d.requested_by}`} · {when(d.requested_at)} — “{d.reason}”</p>
            </div>
            <StatusPill tone={DELETION[d.status][1]}>{DELETION[d.status][0]}</StatusPill>
          </div>
          <ReportList report={d.report} />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            {d.requested_by !== me
              ? <><button type="button" className="lw-btn" onClick={() => act(d, 'reject')}>Reject and restore</button>
                <button type="button" className="lw-btn lw-btn-danger-solid" onClick={() => act(d, 'approve')}>Approve deletion</button></>
              : <span className="lw-meta">You asked for this; someone else must decide.</span>}
          </div>
        </section>
      ))}
      {done.length > 0 && (
        <section className="lw-paper" aria-label="Decided">
          <table className="lw-table">
            <thead><tr><th>Item</th><th>Decision</th><th>Verification</th></tr></thead>
            <tbody>
              {done.map((d) => (
                <tr key={d.id}>
                  <td><div style={{ fontWeight: 600 }}>{d.object_label ?? `${OBJECT[d.object_type]} (title removed with it)`}</div>
                    <div className="lw-meta">{d.matter?.number} · requested by {d.requested_by}</div></td>
                  <td><StatusPill tone={DELETION[d.status][1]}>{DELETION[d.status][0]}</StatusPill>
                    <div className="lw-meta">{d.decided_by} · {when(d.decided_at)}</div></td>
                  <td>{d.verification ? (d.verification.ok
                    ? <StatusPill tone="verified" icon={<CheckCircle2 />}>Nothing left</StatusPill>
                    : <StatusPill tone="conflict" icon={<XCircle />}>Residue found: {Object.entries(d.verification.checks).filter(([, v]) => !v).map(([k]) => k.replace('_', ' ')).join(', ')}</StatusPill>)
                    : <span className="lw-meta">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

// ── Holds ──────────────────────────────────────────────────────────────────────
function Holds({ me }: { me: string }) {
  const qc = useQueryClient();
  const holds = useHolds(true);
  const [error, setError] = useState('');
  const act = async (path: string) => {
    setError('');
    try { await api.post(path); } catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
    await qc.invalidateQueries({ queryKey: keys.holds });
  };
  if (holds.isPending) return <div className="lw-paper lw-sheet"><Skeleton /></div>;
  if (holds.isError) return <ErrorNote>The holds could not be loaded.</ErrorNote>;
  return (
    <>
      <p className="lw-meta" style={{ margin: 0 }}>A hold preserves everything in its matters: nothing in them can be deleted by anyone, or by retention, until a release is approved by a second person.</p>
      {holds.data.can_request && <NewHold />}
      {error && <ErrorNote>{error}</ErrorNote>}
      {holds.data.holds.length === 0 && <EmptyState title="No legal holds" />}
      {holds.data.holds.map((h) => {
        const base = `/api/legal/holds/${encodeURIComponent(h.id)}`;
        return (
          <section key={h.id} className="lw-paper lw-sheet" aria-label={h.name} style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <h2 className="lw-h2">{h.name}</h2>
                <p className="lw-meta" style={{ margin: '2px 0 0' }}>Requested by {h.requested_by} · {when(h.requested_at)}{h.approved_by ? ` · approved by ${h.approved_by}` : ''}</p>
              </div>
              <StatusPill tone={HOLD[h.status][1]}>{HOLD[h.status][0]}</StatusPill>
            </div>
            <p style={{ margin: 0 }}>{h.reason}</p>
            <ul className="lw-chips">{h.matters.map((m) => <li key={m.id}><b className="lw-tnum">{m.number}</b> {m.title} <span className="lw-meta">· {m.client}</span></li>)}</ul>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              {h.status === 'requested' && h.requested_by !== me && <>
                <button type="button" className="lw-btn" onClick={() => act(`${base}/reject`)}>Reject</button>
                <button type="button" className="lw-btn lw-btn-primary" onClick={() => act(`${base}/approve`)}>Approve hold</button></>}
              {h.status === 'active' && holds.data.can_request && <button type="button" className="lw-btn" onClick={() => act(`${base}/release`)}>Request release</button>}
              {h.status === 'release_requested' && h.release_requested_by !== me && <button type="button" className="lw-btn lw-btn-primary" onClick={() => act(`${base}/release/approve`)}>Approve release</button>}
            </div>
          </section>
        );
      })}
    </>
  );
}

function NewHold() {
  const qc = useQueryClient();
  const [form, setForm] = useState({ name: '', reason: '' });
  const [q, setQ] = useState('');
  const [found, setFound] = useState<{ id: string; number: string; title: string }[]>([]);
  const [picked, setPicked] = useState<{ id: string; number: string; title: string }[]>([]);
  const [error, setError] = useState('');
  const search = async (text: string) => {
    setQ(text);
    if (text.trim().length < 2) { setFound([]); return; }
    try { setFound((await api.get<{ matters: { id: string; number: string; title: string }[] }>(`/api/legal/matter-directory?q=${encodeURIComponent(text.trim())}`)).matters); }
    catch { setFound([]); }
  };
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try {
      await api.post('/api/legal/holds', { ...form, matter_ids: picked.map((m) => m.id) });
      setForm({ name: '', reason: '' }); setPicked([]); setQ(''); setFound([]);
      await qc.invalidateQueries({ queryKey: keys.holds });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The hold could not be requested.'); }
  };
  return (
    <form onSubmit={submit} className="lw-paper lw-sheet" style={{ display: 'grid', gap: 12 }} aria-label="Request a legal hold">
      <h2 className="lw-h2">Request a legal hold</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <Field label="Name">{(id) => <input id={id} className="lw-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />}</Field>
        <Field label="Matters" hint="Search by number, title or client.">{(id) => <input id={id} className="lw-input" value={q} onChange={(e) => search(e.target.value)} />}</Field>
      </div>
      {found.length > 0 && (
        <ul className="lw-chips">{found.filter((m) => !picked.some((p) => p.id === m.id)).map((m) => (
          <li key={m.id}><button type="button" className="lw-linklike" onClick={() => setPicked([...picked, m])}>+ {m.number} {m.title}</button></li>))}</ul>
      )}
      {picked.length > 0 && <ul className="lw-chips" aria-label="Matters to hold">{picked.map((m) => (
        <li key={m.id}><b className="lw-tnum">{m.number}</b> {m.title} <button type="button" className="lw-linklike lw-meta" aria-label={`Remove ${m.number}`} onClick={() => setPicked(picked.filter((p) => p.id !== m.id))}>×</button></li>))}</ul>}
      <Field label="Reason" hint="Seen by the approver and by records staff.">{(id) => <textarea id={id} rows={2} className="lw-input" value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value })} />}</Field>
      {error && <ErrorNote>{error}</ErrorNote>}
      <div><button type="submit" className="lw-btn lw-btn-primary" disabled={!form.name.trim() || !form.reason.trim() || picked.length === 0}><ShieldPlus />Request hold</button></div>
    </form>
  );
}

// ── Retention ──────────────────────────────────────────────────────────────────
const POLICY: Record<RetentionPolicy['status'], [string, Tone]> = { draft: ['Draft', 'attention'], approved: ['Approved', 'verified'], retired: ['Retired', 'neutral'] };

function Retention({ me, canManage, canApprove }: { me: string; canManage: boolean; canApprove: boolean }) {
  const qc = useQueryClient();
  const data = useRetention(true);
  const [form, setForm] = useState({ object_type: 'document', days_after_close: 2555, note: '' });
  const [error, setError] = useState('');
  const act = async (fn: () => Promise<unknown>) => {
    setError('');
    try { await fn(); } catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
    await qc.invalidateQueries({ queryKey: keys.retention });
  };
  if (data.isPending) return <div className="lw-paper lw-sheet"><Skeleton /></div>;
  if (data.isError) return <ErrorNote>Retention could not be loaded.</ErrorNote>;
  const { policies, runs, apply_enabled } = data.data;
  return (
    <>
      <p className="lw-meta" style={{ margin: 0 }}>Retention never deletes anything by itself. A dry run shows what policies would select; applying them (switched {apply_enabled ? 'on' : 'off'} for this firm) only creates deletion requests for a person to approve. Held matters are always skipped.</p>
      {error && <ErrorNote>{error}</ErrorNote>}
      <section className="lw-paper" aria-label="Policies">
        <div className="lw-card-h"><h2 className="lw-h2">Policies</h2>
          {canManage && <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(() => api.post('/api/legal/retention/run', { mode: 'dry_run' }))}>Dry run now</button>
            <button type="button" className="lw-btn lw-btn-sm" disabled={!apply_enabled} title={apply_enabled ? '' : 'Switched off until counsel approves a retention policy (P-002)'}
              onClick={() => act(() => api.post('/api/legal/retention/run', { mode: 'apply' }))}>Apply</button></div>}</div>
        {policies.length === 0 ? <p className="lw-meta" style={{ padding: 20, margin: 0 }}>No policies yet.</p> : (
          <table className="lw-table">
            <thead><tr><th>Applies to</th><th>After the matter closes</th><th>Status</th><th><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>{policies.map((p) => (
              <tr key={p.id}>
                <td>{p.object_type === 'document' ? 'Documents' : 'Meeting transcripts'}{p.note && <div className="lw-meta">{p.note}</div>}</td>
                <td className="lw-tnum">{p.days_after_close} days</td>
                <td><StatusPill tone={POLICY[p.status][1]}>{POLICY[p.status][0]}</StatusPill><div className="lw-meta">drafted by {p.drafted_by}{p.approved_by ? ` · approved by ${p.approved_by}` : ''}</div></td>
                <td style={{ textAlign: 'right' }}>
                  {canApprove && p.status === 'draft' && p.drafted_by !== me && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => act(() => api.post(`/api/legal/retention/policies/${p.id}/approve`))}>Approve</button>}
                  {canManage && p.status !== 'retired' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => act(() => api.post(`/api/legal/retention/policies/${p.id}/retire`))}>Retire</button>}
                </td>
              </tr>))}</tbody>
          </table>
        )}
      </section>
      {canManage && (
        <form className="lw-paper lw-sheet" aria-label="Draft a policy" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 2fr auto', gap: 10, alignItems: 'end' }}
          onSubmit={(e) => { e.preventDefault(); void act(() => api.post('/api/legal/retention/policies', { ...form, note: form.note || null })); }}>
          <Field label="Applies to">{(id) => <select id={id} className="lw-input" value={form.object_type} onChange={(e) => setForm({ ...form, object_type: e.target.value })}>
            <option value="document">Documents</option><option value="meeting_transcript">Meeting transcripts</option></select>}</Field>
          <Field label="Days after closing">{(id) => <input id={id} type="number" min={0} className="lw-input" value={form.days_after_close} onChange={(e) => setForm({ ...form, days_after_close: Number(e.target.value) })} />}</Field>
          <Field label="Note">{(id) => <input id={id} className="lw-input" value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />}</Field>
          <button type="submit" className="lw-btn">Draft policy</button>
        </form>
      )}
      {runs.length > 0 && (
        <section className="lw-paper" aria-label="Runs">
          <div className="lw-card-h"><h2 className="lw-h2">Recent runs</h2></div>
          <table className="lw-table">
            <thead><tr><th>When</th><th>Kind</th><th>Selected</th><th>Held, skipped</th><th>Requests created</th></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.id}><td className="lw-meta">{when(r.run_at)} · {r.run_by}</td><td>{r.mode === 'apply' ? 'Applied' : 'Dry run'}</td>
                <td className="lw-tnum">{r.report.candidates}{r.report.draft_preview ? <span className="lw-meta"> (+{r.report.draft_preview} by drafts)</span> : null}</td>
                <td className="lw-tnum">{r.report.held_skipped}</td><td className="lw-tnum">{r.report.requests_created}</td></tr>))}</tbody>
          </table>
        </section>
      )}
    </>
  );
}
