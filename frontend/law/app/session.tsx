import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { keys, useMe } from '../api/queries';
import { ApiError } from '../api/client';
import { CSRF_COOKIE, readCookie, SESSION_EXPIRED_EVENT } from '../../services/session';
import type { Me } from '../api/types';
import ChangePassword from '../pages/ChangePassword';
import { Skeleton } from '../ui/primitives';

export const MeContext = React.createContext<Me | null>(null);
export const useSession = (): Me => {
  const me = React.useContext(MeContext);
  if (!me) throw new Error('useSession outside a signed-in route');
  return me;
};
export const hasFirmRole = (me: Me, ...roles: string[]) => roles.some((r) => me.user.roles.includes(r as any));

/** When the session ends anywhere (idle timeout, revocation, sign-out elsewhere), drop every cached
 *  response — matter content must not linger in memory — and re-check who is signed in. */
export function useSessionExpiry(): void {
  const qc = useQueryClient();
  useEffect(() => {
    const onExpired = () => { qc.removeQueries(); qc.invalidateQueries({ queryKey: keys.me }); };
    window.addEventListener(SESSION_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(SESSION_EXPIRED_EVENT, onExpired);
  }, [qc]);
}

/** Over HTTPS a session always comes with its readable CSRF cookie; without it there is no session to ask about, so the
 *  page goes straight to sign-in instead of asking the server and logging a 401 (a first visit to the open demo). Over
 *  plain HTTP (development, tests) the server is always asked. */
export function knownSignedOut(protocol: string, cookies: string): boolean {
  return protocol === 'https:' && !readCookie(CSRF_COOKIE, cookies);
}

export function RequireSession({ children }: { children: React.ReactNode }) {
  const signedOut = knownSignedOut(window.location.protocol, document.cookie);
  const me = useMe(!signedOut);
  const location = useLocation();
  if (signedOut) return <Navigate to={`/signin?next=${encodeURIComponent(location.pathname + location.search)}`} replace />;
  if (me.isPending) {
    return <div className="lw-content" aria-busy="true"><Skeleton height={48} /><Skeleton height={240} /></div>;
  }
  if (me.error || !me.data) {
    if (me.error && !(me.error instanceof ApiError && me.error.status === 401)) {
      return <div className="lw-content"><p role="alert">EchoMind could not reach the server. Check the connection and reload.</p></div>;
    }
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/signin?next=${next}`} replace />;
  }
  if (me.data.session.must_change_password) {
    return <ChangePassword forced />;
  }
  return <MeContext.Provider value={me.data}>{children}</MeContext.Provider>;
}
