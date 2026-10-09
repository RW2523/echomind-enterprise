import React, { useState } from 'react';
import { UserPlus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation, useFirmUsers } from '../api/queries';
import type { EnrollmentLink, FirmRole, FirmUser } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { CopyField, Dialog, ErrorNote, Field, roleLabel, Skeleton, StatusPill } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const FIRM_ROLES: FirmRole[] = ['firm_admin', 'managing_partner', 'attorney', 'paralegal', 'reviewer', 'records_admin', 'read_only'];
const linkUrl = (l: EnrollmentLink) => `${window.location.origin}${l.path}`;

/** Firm administration: accounts, firm roles, enrolment links. Firm admins see no matter content. */
export default function AdminUsers() {
  const me = useSession();
  const isAdmin = hasFirmRole(me, 'firm_admin');
  const users = useFirmUsers(isAdmin);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<FirmUser | null>(null);
  const [link, setLink] = useState<{ who: string; link: EnrollmentLink } | null>(null);
  const [error, setError] = useState('');
  if (!isAdmin) return <NotAvailable />;

  const act = async (fn: () => Promise<unknown>) => {
    setError('');
    try { await fn(); await users.refetch(); } catch (e) { setError(e instanceof ApiError ? e.message : 'The change could not be made.'); }
  };
  const u = (name: string) => `/api/auth/users/${encodeURIComponent(name)}`;
  return (
    <>
      <Toolbar title="Users & roles">
        <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><UserPlus />New account</button>
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Firm roles control firm-wide tasks. Access to a matter is granted by that matter's lead, not here. Removing a role signs the person out everywhere.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div className="lw-paper" style={{ overflowX: 'auto' }}>
          {users.isPending ? <div style={{ padding: 20 }}><Skeleton /></div> : (
            <table className="lw-table">
              <thead><tr><th>Account</th><th>Firm roles</th><th>Two-step</th><th>Status</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {(users.data ?? []).map((x) => (
                  <tr key={x.username}>
                    <td style={{ fontWeight: 600 }}>{x.username}</td>
                    <td><div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>{x.roles.length ? x.roles.map((r) => <StatusPill key={r} tone="neutral">{roleLabel(r)}</StatusPill>) : <span className="lw-meta">None</span>}</div></td>
                    <td>{x.mfa_enabled ? <StatusPill tone="verified">On</StatusPill> : <StatusPill tone="attention">Not set up</StatusPill>}</td>
                    <td>{x.status === 'active' ? 'Active' : <StatusPill tone="conflict">Disabled</StatusPill>}</td>
                    <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                      <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing(x)}>Roles</button>
                      {x.mfa_enabled
                        ? <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(async () => setLink({ who: x.username, link: (await api.post<{ enrollment: EnrollmentLink }>(`${u(x.username)}/reset-mfa`)).enrollment }))}>Reset two-step</button>
                        : <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(async () => setLink({ who: x.username, link: (await api.post<{ enrollment: EnrollmentLink }>(`${u(x.username)}/enrollment-link`)).enrollment }))}>Enrolment link</button>}
                      {x.username !== me.user.username && (x.status === 'active'
                        ? <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => act(() => api.post(`${u(x.username)}/disable`))}>Disable</button>
                        : <button type="button" className="lw-btn lw-btn-sm" onClick={() => act(() => api.post(`${u(x.username)}/enable`))}>Enable</button>)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      {creating && <NewAccountDialog onClose={() => setCreating(false)} onCreated={(who, l) => { setCreating(false); setLink({ who, link: l }); users.refetch(); }} />}
      {editing && <RolesDialog user={editing} self={editing.username === me.user.username} onClose={() => setEditing(null)} />}
      {link && (
        <Dialog title={`Enrolment link for ${link.who}`} onClose={() => setLink(null)}>
          <p style={{ margin: 0 }}>Give this one-time link to {link.who} through a channel you trust. It works once and expires in 72 hours; issuing a new link cancels this one.</p>
          <CopyField label="Link" value={linkUrl(link.link)} />
        </Dialog>
      )}
    </>
  );
}

function NewAccountDialog({ onClose, onCreated }: { onClose: () => void; onCreated: (who: string, l: EnrollmentLink) => void }) {
  const [form, setForm] = useState({ username: '', password: '', role: 'attorney' as FirmRole });
  const [error, setError] = useState('');
  const create = useApiMutation(() => api.post<{ enrollment: EnrollmentLink }>('/api/auth/users', form), [keys.users]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { const r = await create.mutateAsync(undefined); onCreated(form.username, r.enrollment); } catch (err) { setError(err instanceof ApiError ? err.message : 'The account could not be created.'); }
  };
  return (
    <Dialog title="New account" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Username">{(id) => <input id={id} className="lw-input" autoComplete="off" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />}</Field>
        <Field label="Initial password" hint="At least 12 characters. They must replace it at first sign-in.">{(id) => <input id={id} className="lw-input" type="password" autoComplete="new-password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />}</Field>
        <Field label="Firm role">{(id) => (
          <select id={id} className="lw-input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as FirmRole })}>
            {FIRM_ROLES.map((r) => <option key={r} value={r}>{roleLabel(r)}</option>)}
          </select>)}
        </Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.username || !form.password || create.isPending}>Create account</button>
        </div>
      </form>
    </Dialog>
  );
}

function RolesDialog({ user, self, onClose }: { user: FirmUser; self: boolean; onClose: () => void }) {
  const [roles, setRoles] = useState<FirmRole[]>(user.roles);
  const [error, setError] = useState('');
  const save = useApiMutation(() => api.put(`/api/legal/users/${encodeURIComponent(user.username)}/roles`, { roles }), [keys.users]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await save.mutateAsync(undefined); onClose(); } catch (err) { setError(err instanceof ApiError ? err.message : 'The roles could not be saved.'); }
  };
  const removing = user.roles.some((r) => !roles.includes(r));
  return (
    <Dialog title={`Firm roles for ${user.username}`} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 10 }}>
        {FIRM_ROLES.map((r) => (
          <label key={r} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={roles.includes(r)} disabled={self && r === 'firm_admin'}
              onChange={(e) => setRoles(e.target.checked ? [...roles, r] : roles.filter((x) => x !== r))} />
            {roleLabel(r)}
          </label>
        ))}
        {removing && <p className="lw-meta" style={{ margin: 0, color: 'var(--attention)' }}>Removing a role signs {user.username} out everywhere.</p>}
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={save.isPending}>Save roles</button>
        </div>
      </form>
    </Dialog>
  );
}
