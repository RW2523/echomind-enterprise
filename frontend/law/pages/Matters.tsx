import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Plus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation, useClients, useMatters } from '../api/queries';
import type { Client, Matter, Sensitivity } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, MatterStatusPill, roleLabel, SensitivityPill, Skeleton } from '../ui/primitives';

export default function Matters() {
  const me = useSession();
  const matters = useMatters();
  const [creating, setCreating] = useState(false);
  const canCreate = hasFirmRole(me, 'attorney', 'managing_partner');
  return (
    <>
      <Toolbar title="Matters">
        {canCreate && <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><Plus />New matter</button>}
      </Toolbar>
      <div className="lw-content">
        {matters.isPending ? <div className="lw-paper lw-sheet" style={{ display: 'grid', gap: 10 }}><Skeleton /><Skeleton /><Skeleton /></div>
          : matters.error ? <EmptyState title="Matters could not be loaded">Check the connection and try again.</EmptyState>
            : matters.data.length === 0 ? (
              <EmptyState title="No matters yet" action={canCreate ? <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><Plus />New matter</button> : undefined}>
                {canCreate ? 'Open a matter to start working with its documents, meetings and drafts.' : 'Matters you are a member of will appear here. The lead on a matter can add you.'}
              </EmptyState>
            ) : (
              <div className="lw-paper" style={{ overflowX: 'auto' }}>
                <table className="lw-table">
                  <thead><tr><th>Matter</th><th>Client</th><th>Status</th><th>Sensitivity</th><th>Your role</th></tr></thead>
                  <tbody>
                    {matters.data.map((m) => (
                      <tr key={m.id}>
                        <td><Link to={`/m/${encodeURIComponent(m.number)}/overview`} style={{ color: 'inherit', textDecoration: 'none' }}>
                          <div className="lw-tnum" style={{ fontWeight: 600, color: 'var(--accent)' }}>{m.number}</div><div>{m.title}</div></Link></td>
                        <td>{m.client.name}</td>
                        <td><MatterStatusPill value={m.status} /></td>
                        <td><SensitivityPill value={m.sensitivity} /></td>
                        <td>{roleLabel(m.my_role)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
      </div>
      {creating && <NewMatterDialog onClose={() => setCreating(false)} />}
    </>
  );
}

function NewMatterDialog({ onClose }: { onClose: () => void }) {
  const clients = useClients();
  const navigate = useNavigate();
  const [clientId, setClientId] = useState('');
  const [newClient, setNewClient] = useState({ name: '', code: '' });
  const [form, setForm] = useState({ number: '', title: '', practice_area: '', jurisdiction: '', sensitivity: 'restricted' as Sensitivity });
  const [error, setError] = useState('');
  const create = useApiMutation(async () => {
    let cid = clientId;
    if (cid === '__new') {
      cid = (await api.post<{ client: Client }>('/api/legal/clients', { name: newClient.name, code: newClient.code || undefined })).client.id;
    }
    return (await api.post<{ matter: Matter }>('/api/legal/matters', {
      client_id: cid, number: form.number, title: form.title, sensitivity: form.sensitivity,
      practice_area: form.practice_area || undefined, jurisdiction: form.jurisdiction || undefined,
    })).matter;
  }, [keys.matters, keys.clients]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try {
      const m = await create.mutateAsync(undefined);
      onClose();
      navigate(`/m/${encodeURIComponent(m.number)}/overview`);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The matter could not be created.'); }
  };
  const ready = form.number && form.title && clientId && (clientId !== '__new' || newClient.name);
  return (
    <Dialog title="New matter" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Client">{(id) => (
          <select id={id} className="lw-input" value={clientId} onChange={(e) => setClientId(e.target.value)}>
            <option value="" disabled>Choose a client</option>
            {(clients.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            <option value="__new">New client…</option>
          </select>)}
        </Field>
        {clientId === '__new' && (
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 10 }}>
            <Field label="Client name">{(id) => <input id={id} className="lw-input" value={newClient.name} onChange={(e) => setNewClient({ ...newClient, name: e.target.value })} />}</Field>
            <Field label="Code" hint="Optional">{(id) => <input id={id} className="lw-input" value={newClient.code} onChange={(e) => setNewClient({ ...newClient, code: e.target.value })} />}</Field>
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 10 }}>
          <Field label="Matter number">{(id) => <input id={id} className="lw-input lw-tnum" placeholder="M-101" value={form.number} onChange={(e) => setForm({ ...form, number: e.target.value })} />}</Field>
          <Field label="Title">{(id) => <input id={id} className="lw-input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />}</Field>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Practice area" hint="Optional">{(id) => <input id={id} className="lw-input" value={form.practice_area} onChange={(e) => setForm({ ...form, practice_area: e.target.value })} />}</Field>
          <Field label="Jurisdiction" hint="Optional">{(id) => <input id={id} className="lw-input" value={form.jurisdiction} onChange={(e) => setForm({ ...form, jurisdiction: e.target.value })} />}</Field>
        </div>
        <Field label="Sensitivity" hint="Only members can see a matter, whatever its sensitivity.">{(id) => (
          <select id={id} className="lw-input" value={form.sensitivity} onChange={(e) => setForm({ ...form, sensitivity: e.target.value as Sensitivity })}>
            <option value="standard">Standard</option><option value="restricted">Restricted</option><option value="highly_restricted">Highly restricted</option>
          </select>)}
        </Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!ready || create.isPending}>{create.isPending ? 'Opening…' : 'Open matter'}</button>
        </div>
      </form>
    </Dialog>
  );
}
