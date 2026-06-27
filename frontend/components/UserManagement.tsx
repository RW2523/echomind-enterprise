import React, { useEffect, useState } from 'react';
import { getAuthConfig, getMe, listUsers, createUser, deleteUserAccount, AuthUser } from '../services/backend';

/** Admin-only account management. Renders nothing unless auth is enabled AND the current user is an admin. */
const UserManagement: React.FC = () => {
  const [show, setShow] = useState(false);
  const [me, setMe] = useState<AuthUser | null>(null);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [u, setU] = useState('');
  const [p, setP] = useState('');
  const [role, setRole] = useState('user');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const refresh = async () => { try { setUsers(await listUsers()); } catch (_) {} };

  useEffect(() => {
    (async () => {
      const cfg = await getAuthConfig();
      if (!cfg.auth_enabled) return;
      const m = await getMe();
      if (m && m.role === 'admin') { setMe(m); setShow(true); await refresh(); }
    })();
  }, []);

  if (!show) return null;

  const add = async () => {
    if (!u || !p || busy) return;
    setBusy(true); setErr('');
    try {
      await createUser(u, p, role);
      setU(''); setP(''); setRole('user');
      await refresh();
    } catch (e: any) { setErr(e?.message || 'Failed to add user'); }
    setBusy(false);
  };

  const del = async (name: string) => {
    if (!window.confirm(`Delete user "${name}"? This cannot be undone.`)) return;
    setErr('');
    try { await deleteUserAccount(name); await refresh(); } catch (e: any) { setErr(e?.message || 'Failed to delete user'); }
  };

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
      <div>
        <h3 className="text-sm font-bold text-slate-200">User Management</h3>
        <p className="text-xs text-slate-500 mt-1">Admin only. Add or remove accounts; users can sign in and use the app.</p>
      </div>

      <div className="flex flex-col gap-2">
        {users.map((usr) => (
          <div key={usr.id} className="flex items-center justify-between gap-2 rounded-xl bg-white/[0.03] border border-white/10 px-3 py-2">
            <div className="min-w-0 flex items-center gap-2">
              <span className="text-sm text-white truncate">{usr.username}</span>
              <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-accent/15 text-accent">{usr.role}</span>
            </div>
            {usr.username !== me?.username && (
              <button type="button" onClick={() => del(usr.username)} className="shrink-0 text-xs text-slate-400 hover:text-red-400 transition-colors">Delete</button>
            )}
          </div>
        ))}
        {users.length === 0 && <p className="text-xs text-slate-500">No users yet.</p>}
      </div>

      <div className="flex flex-col sm:flex-row gap-2">
        <input value={u} onChange={(e) => setU(e.target.value)} placeholder="Username" className="flex-1 rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent/50" />
        <input value={p} onChange={(e) => setP(e.target.value)} type="password" placeholder="Password" className="flex-1 rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent/50" />
        <select value={role} onChange={(e) => setRole(e.target.value)} className="rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm outline-none focus:border-accent/50">
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        <button type="button" onClick={add} disabled={busy || !u || !p} className="rounded-xl px-4 py-2 text-sm font-semibold bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 disabled:opacity-50 transition-colors">Add</button>
      </div>
      {err && <p className="text-xs text-red-400">{err}</p>}
    </div>
  );
};

export default UserManagement;
