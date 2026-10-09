import React, { useState } from 'react';
import { Siren } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation, useBreakGlass } from '../api/queries';
import type { BreakGlassRequest } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const STATUS: Record<BreakGlassRequest['status'], [string, Tone]> = {
  requested: ['Awaiting approval', 'attention'], active: ['Active', 'conflict'], rejected: ['Rejected', 'neutral'], ended: ['Ended', 'neutral'],
};
const time = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');

/** Break-glass: a firm admin asks for read-only access to a matter by number; a different managing
 *  partner decides. Never across an ethical wall, never for a highly restricted matter. */
export default function EmergencyAccess() {
  const me = useSession();
  const allowed = hasFirmRole(me, 'firm_admin', 'managing_partner');
  const list = useBreakGlass(allowed);
  const [form, setForm] = useState({ matter_number: '', reason: '', minutes: 60 });
  const [error, setError] = useState('');
  const request = useApiMutation(() => api.post('/api/legal/break-glass', form), [keys.breakGlass, keys.matters]);
  if (!allowed) return <NotAvailable />;
  const canRequest = hasFirmRole(me, 'firm_admin');
  const act = async (path: string) => {
    setError('');
    try { await api.post(path); await list.refetch(); } catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
  };
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await request.mutateAsync(undefined); setForm({ matter_number: '', reason: '', minutes: 60 }); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The request could not be sent.'); }
  };
  return (
    <>
      <Toolbar title="Emergency access" />
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>Read-only access to a matter you are not a member of, for a limited time, when its lead cannot be reached. A different managing partner must approve. Not available for highly restricted matters or across an ethical wall.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        {canRequest && (
          <form onSubmit={submit} className="lw-paper lw-sheet" style={{ display: 'grid', gap: 12 }}>
            <h2 className="lw-h2">Request access</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <Field label="Matter number">{(id) => <input id={id} className="lw-input lw-tnum" placeholder="M-215" value={form.matter_number} onChange={(e) => setForm({ ...form, matter_number: e.target.value })} />}</Field>
              <Field label="For">{(id) => <select id={id} className="lw-input" value={form.minutes} onChange={(e) => setForm({ ...form, minutes: Number(e.target.value) })}>
                {[30, 60, 120, 240].map((m) => <option key={m} value={m}>{m < 60 ? `${m} minutes` : `${m / 60} hour${m > 60 ? 's' : ''}`}</option>)}</select>}</Field>
            </div>
            <Field label="Reason" hint="Seen by the approving managing partner.">{(id) => <textarea id={id} rows={2} className="lw-input" value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value })} />}</Field>
            <div><button type="submit" className="lw-btn lw-btn-primary" disabled={!form.matter_number || !form.reason || request.isPending}><Siren />Request emergency access</button></div>
          </form>
        )}
        <section className="lw-paper" aria-label="Requests">
          <div className="lw-card-h"><h2 className="lw-h2">{list.data?.can_approve ? 'Requests' : 'Your requests'}</h2></div>
          {list.isPending ? <div style={{ padding: 20 }}><Skeleton /></div>
            : list.data!.requests.length === 0 ? <p className="lw-meta" style={{ padding: 20, margin: 0 }}>No requests.</p> : (
              <table className="lw-table">
                <thead><tr><th>Matter</th>{list.data!.can_approve && <th>Requested by</th>}<th>Reason</th><th>Status</th><th><span className="sr-only">Actions</span></th></tr></thead>
                <tbody>
                  {list.data!.requests.map((g) => {
                    const [label, tone] = STATUS[g.status];
                    const base = `/api/legal/break-glass/${encodeURIComponent(g.id)}`;
                    return (
                      <tr key={g.id}>
                        <td><div className="lw-tnum" style={{ fontWeight: 600 }}>{g.matter_number}</div>
                          {g.matter ? <div className="lw-meta">{g.matter.title} · {g.matter.client}</div> : list.data!.can_approve ? <div className="lw-meta">No matter has this number</div> : null}</td>
                        {list.data!.can_approve && <td>{g.requester}</td>}
                        <td>{g.reason}<div className="lw-meta">{g.minutes} min · asked {time(g.requested_at)}</div></td>
                        <td><StatusPill tone={tone}>{label}</StatusPill>{g.status === 'active' && <div className="lw-meta">until {time(g.expires_at)}</div>}</td>
                        <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                          {list.data!.can_approve && g.status === 'requested' && g.requester !== me.user.username && <>
                            <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => act(`${base}/approve`)}>Approve</button>
                            <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(`${base}/reject`)}>Reject</button></>}
                          {g.status === 'active' && <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => act(`${base}/end`)}>End now</button>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
        </section>
        {list.data && list.data.requests.length === 0 && !canRequest && <EmptyState title="Nothing to decide">Requests from firm administrators will appear here.</EmptyState>}
      </div>
    </>
  );
}
