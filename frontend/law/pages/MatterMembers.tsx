import React, { useState } from 'react';
import { UserPlus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation, useMemberships } from '../api/queries';
import type { MatterRole, Membership } from '../api/types';
import { useMatter } from '../app/matter';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, ErrorNote, Field, roleLabel, Skeleton, StatusPill } from '../ui/primitives';

const ROLES: MatterRole[] = ['lead', 'attorney', 'paralegal', 'reviewer', 'read_only'];

export default function MatterMembers() {
  const matter = useMatter();
  const me = useSession();
  const members = useMemberships(matter.id);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');
  const isLead = matter.my_role === 'lead';
  const base = `/api/legal/matters/${encodeURIComponent(matter.id)}/memberships`;
  const invalidate = [keys.memberships(matter.id), keys.matters];
  const change = useApiMutation((v: { m: Membership; role: MatterRole }) => api.patch(`${base}/${v.m.id}`, { role: v.role }), invalidate);
  const remove = useApiMutation((m: Membership) => api.del(`${base}/${m.id}`), invalidate);
  const act = async (fn: () => Promise<unknown>) => {
    setError('');
    try { await fn(); } catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Members">
        {isLead && <button type="button" className="lw-btn lw-btn-primary" onClick={() => setAdding(true)}><UserPlus />Add member</button>}
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 960 }}>
        <p className="lw-meta" style={{ margin: 0 }}>Only these people can see that this matter exists. Removing someone takes effect immediately.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div className="lw-paper" style={{ overflowX: 'auto' }}>
          {members.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /></div> : (
            <table className="lw-table">
              <thead><tr><th>Person</th><th>Role</th><th>Export</th><th>Until</th>{isLead && <th><span className="sr-only">Actions</span></th>}</tr></thead>
              <tbody>
                {(members.data ?? []).map((m) => (
                  <tr key={m.id}>
                    <td><span style={{ fontWeight: 600 }}>{m.display_name}</span>{m.user_id === me.user.id && <span className="lw-meta"> · you</span>}</td>
                    <td>{isLead ? (
                      <select className="lw-input" style={{ width: 'auto', padding: '5px 8px' }} aria-label={`Role for ${m.display_name}`} value={m.role}
                        onChange={(e) => act(() => change.mutateAsync({ m, role: e.target.value as MatterRole }))}>
                        {ROLES.map((r) => <option key={r} value={r}>{roleLabel(r)}</option>)}
                      </select>) : roleLabel(m.role)}</td>
                    <td>{m.can_export ? <StatusPill tone="accent">Can export</StatusPill> : <span className="lw-meta">—</span>}</td>
                    <td className="lw-tnum lw-meta">{m.valid_to ? m.valid_to.slice(0, 10) : 'No end date'}</td>
                    {isLead && <td style={{ textAlign: 'right' }}>
                      <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => act(() => remove.mutateAsync(m))}>Remove</button>
                    </td>}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      {adding && <AddMemberDialog matterId={matter.id} onClose={() => setAdding(false)} />}
    </>
  );
}

function AddMemberDialog({ matterId, onClose }: { matterId: string; onClose: () => void }) {
  const [form, setForm] = useState({ username: '', role: 'attorney' as MatterRole, can_export: false, valid_to: '' });
  const [error, setError] = useState('');
  const add = useApiMutation(() => api.post(`/api/legal/matters/${encodeURIComponent(matterId)}/memberships`, {
    username: form.username.trim(), role: form.role, can_export: form.can_export,
    valid_to: form.valid_to ? new Date(`${form.valid_to}T23:59:59`).toISOString() : undefined,
  }), [keys.memberships(matterId), keys.matters]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await add.mutateAsync(undefined); onClose(); } catch (err) { setError(err instanceof ApiError ? err.message : 'The person could not be added.'); }
  };
  return (
    <Dialog title="Add a member" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Username" hint="A colleague in your firm with an active account.">{(id) => <input id={id} className="lw-input" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />}</Field>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Role">{(id) => (
            <select id={id} className="lw-input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as MatterRole })}>
              {ROLES.map((r) => <option key={r} value={r}>{roleLabel(r)}</option>)}
            </select>)}
          </Field>
          <Field label="Until" hint="Optional end date">{(id) => <input id={id} type="date" className="lw-input" value={form.valid_to} onChange={(e) => setForm({ ...form, valid_to: e.target.value })} />}</Field>
        </div>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input type="checkbox" checked={form.can_export} onChange={(e) => setForm({ ...form, can_export: e.target.checked })} />
          May request exports from this matter
        </label>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.username.trim() || add.isPending}>Add member</button>
        </div>
      </form>
    </Dialog>
  );
}
