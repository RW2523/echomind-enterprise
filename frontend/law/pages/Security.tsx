import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { api } from '../api/client';
import { keys, useApiMutation, useMySessions } from '../api/queries';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Skeleton, StatusPill } from '../ui/primitives';
import ChangePassword from './ChangePassword';

export default function Security() {
  const me = useSession();
  const sessions = useMySessions();
  const revoke = useApiMutation((id: string) => api.del(`/api/auth/sessions/${encodeURIComponent(id)}`), [keys.sessions]);
  return (
    <>
      <Toolbar title="Security" />
      <div className="lw-content" style={{ maxWidth: 880 }}>
        <section className="lw-paper lw-sheet" style={{ display: 'grid', gap: 8 }}>
          <h2 className="lw-h2">Two-step sign-in</h2>
          {me.session.mfa_enrolled
            ? <div><StatusPill tone="verified" icon={<ShieldCheck />}>On</StatusPill></div>
            : <p className="lw-meta" style={{ margin: 0 }}>Not set up. Ask your administrator for an enrolment link.</p>}
          <p className="lw-meta" style={{ margin: 0 }}>Lost your authenticator? Use a recovery code to sign in, then ask your administrator to reset it.</p>
        </section>
        <section className="lw-paper" aria-labelledby="sessions-h">
          <div className="lw-card-h"><h2 className="lw-h2" id="sessions-h">Where you're signed in</h2></div>
          {sessions.isPending ? <div style={{ padding: 20 }}><Skeleton /></div> : (
            <table className="lw-table">
              <thead><tr><th>Device</th><th>Signed in</th><th>Last active</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {(sessions.data ?? []).map((s) => (
                  <tr key={s.id}>
                    <td><div style={{ fontWeight: 600, maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.user_agent || 'Unknown device'}</div><div className="lw-meta">{s.ip}</div></td>
                    <td className="lw-tnum lw-meta">{s.created_at.replace('T', ' ').slice(0, 16)}</td>
                    <td className="lw-tnum lw-meta">{(s.last_seen_at || '').replace('T', ' ').slice(0, 16)}</td>
                    <td style={{ textAlign: 'right' }}>{s.current ? <StatusPill tone="accent">This device</StatusPill>
                      : <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => revoke.mutate(s.id)}>Sign out</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <ChangePassword onDone={() => sessions.refetch()} />
      </div>
    </>
  );
}
