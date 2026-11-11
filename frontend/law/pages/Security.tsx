import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { api } from '../api/client';
import { keys, useApiMutation, useMySessions } from '../api/queries';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Skeleton, StatusPill } from '../ui/primitives';
import ChangePassword from './ChangePassword';

/** "Chrome on Linux" from a browser's own string (2026-11-12, evaluation UI "Fix next": devices were shown as raw strings). */
export function device(ua?: string | null): string {
  if (!ua) return 'Unknown device';
  const browser = /Edg\//.test(ua) ? 'Edge' : /OPR\//.test(ua) ? 'Opera' : /Firefox\//.test(ua) ? 'Firefox'
    : /Chrome\//.test(ua) ? 'Chrome' : /Safari\//.test(ua) ? 'Safari' : /python|curl|httpx/i.test(ua) ? 'A script' : 'A browser';
  const system = /iPhone|iPad/.test(ua) ? 'iOS' : /Android/.test(ua) ? 'Android' : /Windows/.test(ua) ? 'Windows'
    : /Mac OS X|Macintosh/.test(ua) ? 'macOS' : /Linux/.test(ua) ? 'Linux' : '';
  return system ? `${browser} on ${system}` : browser;
}

/** An address inside the firm's own network is said to be so, not shown as a number nobody recognises. */
export function where(ip?: string | null): string {
  if (!ip) return '';
  return /^(?:10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|::1$|fc|fd)/i.test(ip) ? 'This network' : ip;
}

/** A time with its zone: "11 Nov 2026, 14:05 GMT". */
export function stamp(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString([], { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
}

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
                    <td><div style={{ fontWeight: 600 }} title={s.user_agent || undefined}>{device(s.user_agent)}</div><div className="lw-meta">{where(s.ip)}</div></td>
                    <td className="lw-tnum lw-meta">{stamp(s.created_at)}</td>
                    <td className="lw-tnum lw-meta">{s.last_seen_at ? stamp(s.last_seen_at) : ''}</td>
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
