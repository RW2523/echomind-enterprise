import React, { useMemo, useState } from 'react';
import { Plus, ShieldOff } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation, useFirmUsers, useMatterDirectory, useWalls } from '../api/queries';
import type { DirectoryMatter, Wall } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const STATUS: Record<Wall['status'], [string, Tone]> = {
  requested: ['Awaiting approval', 'attention'], active: ['In force', 'verified'],
  release_requested: ['Release awaiting approval', 'attention'], released: ['Released', 'neutral'], rejected: ['Rejected', 'neutral'],
};

/** Ethical walls (managing partners). Every decision needs a second managing partner. */
export default function Walls() {
  const me = useSession();
  const isMp = hasFirmRole(me, 'managing_partner');
  const walls = useWalls(isMp);
  const [creating, setCreating] = useState(false);
  const [exceptionFor, setExceptionFor] = useState<Wall | null>(null);
  const [error, setError] = useState('');
  if (!isMp) return <NotAvailable />;
  const act = async (path: string, method: 'post' | 'del' = 'post') => {
    setError('');
    try { await (method === 'post' ? api.post(path) : api.del(path)); await walls.refetch(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
  };
  const mine = (who?: string | null) => who === me.user.username;
  return (
    <>
      <Toolbar title="Ethical walls">
        <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><Plus />New wall</button>
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>A wall keeps the people it screens away from the matters and clients it protects — whatever their role on those matters. Walls, their release and exceptions each need a second managing partner.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        {walls.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : walls.data!.length === 0 ? <EmptyState title="No walls">Create a wall when someone must not see a matter or client.</EmptyState>
            : walls.data!.map((w) => {
              const [label, tone] = STATUS[w.status];
              const base = `/api/legal/walls/${encodeURIComponent(w.id)}`;
              return (
                <section key={w.id} className="lw-paper" aria-label={w.name}>
                  <div className="lw-card-h">
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}><ShieldOff size={18} /><h2 className="lw-h2">{w.name}</h2><StatusPill tone={tone}>{label}</StatusPill></div>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {w.status === 'requested' && !mine(w.requested_by) && <>
                        <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => act(`${base}/approve`)}>Approve</button>
                        <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(`${base}/reject`)}>Reject</button></>}
                      {w.status === 'active' && <>
                        <button type="button" className="lw-btn lw-btn-sm" onClick={() => setExceptionFor(w)}>Request exception</button>
                        <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(`${base}/release`)}>Request release</button></>}
                      {w.status === 'release_requested' && !mine(w.release_requested_by) &&
                        <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => act(`${base}/release/approve`)}>Approve release</button>}
                    </div>
                  </div>
                  <div className="lw-sheet" style={{ display: 'grid', gap: 12, paddingTop: 16 }}>
                    <p style={{ margin: 0 }}>{w.rationale}</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '8px 20px' }}>
                      <span className="lw-meta">Screened</span>
                      <span style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{w.screened.map((s) => <StatusPill key={s} tone="neutral">{s}</StatusPill>)}</span>
                      <span className="lw-meta">Protects</span>
                      <span style={{ display: 'grid', gap: 4 }}>{w.protected.map((p) => <span key={p.id}>{p.label}</span>)}</span>
                      <span className="lw-meta">Requested</span><span>{w.requested_by} · {w.requested_at.slice(0, 10)}</span>
                      {w.approved_by && <><span className="lw-meta">{w.status === 'rejected' ? 'Rejected' : 'Approved'}</span><span>{w.approved_by} · {w.approved_at?.slice(0, 10)}</span></>}
                    </div>
                    {w.exceptions.length > 0 && (
                      <table className="lw-table">
                        <thead><tr><th>Exception</th><th>Reason</th><th>Until</th><th>Status</th><th><span className="sr-only">Actions</span></th></tr></thead>
                        <tbody>
                          {w.exceptions.map((e) => (
                            <tr key={e.id}>
                              <td>{e.username}</td><td>{e.reason}</td><td className="lw-tnum">{e.expires_at.slice(0, 10)}</td>
                              <td>{e.revoked ? <StatusPill tone="neutral">Revoked</StatusPill> : e.approved ? <StatusPill tone="verified">Approved</StatusPill> : <StatusPill tone="attention">Awaiting approval</StatusPill>}</td>
                              <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                                {!e.approved && !e.revoked && !mine(e.requested_by) && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => act(`${base}/exceptions/${e.id}/approve`)}>Approve</button>}
                                {!e.revoked && <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => act(`${base}/exceptions/${e.id}`, 'del')}>Revoke</button>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </section>
              );
            })}
      </div>
      {creating && <NewWallDialog onClose={() => setCreating(false)} />}
      {exceptionFor && <ExceptionDialog wall={exceptionFor} onClose={() => setExceptionFor(null)} />}
    </>
  );
}

function NewWallDialog({ onClose }: { onClose: () => void }) {
  const users = useFirmUsers(true);
  const [q, setQ] = useState('');
  const directory = useMatterDirectory(q, true);
  const [form, setForm] = useState({ name: '', rationale: '' });
  const [screened, setScreened] = useState<string[]>([]);
  const [matters, setMatters] = useState<DirectoryMatter[]>([]);
  const [clients, setClients] = useState<{ id: string; name: string }[]>([]);
  const [error, setError] = useState('');
  const create = useApiMutation(() => api.post('/api/legal/walls', {
    ...form, screened, protected_matters: matters.map((m) => m.id), protected_clients: clients.map((c) => c.id),
  }), [keys.walls]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await create.mutateAsync(undefined); onClose(); } catch (err) { setError(err instanceof ApiError ? err.message : 'The wall could not be requested.'); }
  };
  const ready = form.name && form.rationale && screened.length && (matters.length || clients.length);
  return (
    <Dialog title="New ethical wall" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Name">{(id) => <input id={id} className="lw-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />}</Field>
        <Field label="Rationale" hint="Why the screen is needed; kept with the wall.">{(id) => <textarea id={id} className="lw-input" rows={2} value={form.rationale} onChange={(e) => setForm({ ...form, rationale: e.target.value })} />}</Field>
        <Field label="Screened people">{(id) => (
          <select id={id} className="lw-input" value="" onChange={(e) => e.target.value && setScreened([...new Set([...screened, e.target.value])])}>
            <option value="">Add a person…</option>
            {(users.data ?? []).map((u) => <option key={u.username} value={u.username}>{u.username}</option>)}
          </select>)}
        </Field>
        {screened.length > 0 && <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{screened.map((s) => (
          <button type="button" key={s} className="lw-pill lw-pill-neutral" style={{ border: 'none', cursor: 'pointer' }} onClick={() => setScreened(screened.filter((x) => x !== s))} aria-label={`Remove ${s}`}>{s} ×</button>))}</div>}
        <Field label="Protect" hint="Search the firm's matters; protect one matter or the whole client.">{(id) => <input id={id} className="lw-input" placeholder="Matter number, title or client" value={q} onChange={(e) => setQ(e.target.value)} />}</Field>
        {q && <div className="lw-paper" style={{ maxHeight: 180, overflow: 'auto', boxShadow: 'none', border: '1px solid var(--rule)' }}>
          {(directory.data ?? []).map((m) => (
            <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, padding: '6px 10px', borderBottom: '1px solid var(--rule)' }}>
              <span><b className="lw-tnum">{m.number}</b> · {m.title} <span className="lw-meta">({m.client.name})</span></span>
              <span style={{ whiteSpace: 'nowrap' }}>
                <button type="button" className="lw-btn lw-btn-sm" onClick={() => setMatters(matters.some((x) => x.id === m.id) ? matters : [...matters, m])}>Matter</button>
                <button type="button" className="lw-btn lw-btn-sm" onClick={() => setClients(clients.some((x) => x.id === m.client.id) ? clients : [...clients, m.client])}>Whole client</button>
              </span>
            </div>))}
        </div>}
        {(matters.length > 0 || clients.length > 0) && <ul style={{ margin: 0, paddingLeft: 18 }}>
          {matters.map((m) => <li key={m.id}>{m.number} · {m.title}</li>)}
          {clients.map((c) => <li key={c.id}>Client: {c.name} (all matters)</li>)}
        </ul>}
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!ready || create.isPending}>Request wall</button>
        </div>
      </form>
    </Dialog>
  );
}

function ExceptionDialog({ wall, onClose }: { wall: Wall; onClose: () => void }) {
  const clientIds = useMemo(() => wall.protected.filter((p) => p.type === 'client').map((p) => p.id), [wall]);
  const directory = useMatterDirectory('', clientIds.length > 0);
  const options = [
    ...wall.protected.filter((p) => p.type === 'matter').map((p) => ({ id: p.id, label: p.label })),
    ...(directory.data ?? []).filter((m) => clientIds.includes(m.client.id)).map((m) => ({ id: m.id, label: `${m.number} · ${m.title}` })),
  ];
  const max = new Date(Date.now() + 90 * 864e5).toISOString().slice(0, 10);
  const [form, setForm] = useState({ username: wall.screened[0] ?? '', matter_id: '', reason: '', until: '' });
  const [error, setError] = useState('');
  const create = useApiMutation(() => api.post(`/api/legal/walls/${encodeURIComponent(wall.id)}/exceptions`, {
    username: form.username, matter_id: form.matter_id, reason: form.reason, expires_at: new Date(`${form.until}T23:59:00`).toISOString(),
  }), [keys.walls]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await create.mutateAsync(undefined); onClose(); } catch (err) { setError(err instanceof ApiError ? err.message : 'The exception could not be requested.'); }
  };
  return (
    <Dialog title={`Exception to ${wall.name}`} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Person">{(id) => <select id={id} className="lw-input" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })}>{wall.screened.map((s) => <option key={s}>{s}</option>)}</select>}</Field>
        <Field label="Matter">{(id) => <select id={id} className="lw-input" value={form.matter_id} onChange={(e) => setForm({ ...form, matter_id: e.target.value })}>
          <option value="" disabled>Choose a protected matter</option>{options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}</select>}</Field>
        <Field label="Reason">{(id) => <textarea id={id} rows={2} className="lw-input" value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value })} />}</Field>
        <Field label="Until" hint="At most 90 days.">{(id) => <input id={id} type="date" className="lw-input" max={max} value={form.until} onChange={(e) => setForm({ ...form, until: e.target.value })} />}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.matter_id || !form.reason || !form.until || create.isPending}>Request exception</button>
        </div>
      </form>
    </Dialog>
  );
}
