/**
 * Browser-side session plumbing (EchoMind Law S1.1, harmless for other editions).
 *
 * - CSRF: the Law backend requires the per-session token from the readable `__Host-echomind_csrf`
 *   cookie in an `X-CSRF-Token` header on every same-origin state-changing request. Patching fetch
 *   once covers every call site (services, hooks and components alike).
 * - Expiry: a 401 from any /api/ call outside /api/auth/ means the session ended (idle timeout,
 *   revocation, sign-out elsewhere); the app is told so it can show the sign-in screen.
 */
export const CSRF_COOKIE = '__Host-echomind_csrf';
export const SESSION_EXPIRED_EVENT = 'echomind:session-expired';
const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export function readCookie(name: string, cookieString: string = document.cookie): string {
  for (const part of cookieString.split(';')) {
    const [k, ...rest] = part.trim().split('=');
    if (k === name) return decodeURIComponent(rest.join('='));
  }
  return '';
}

function sameOriginPath(url: string): string | null {
  try {
    const u = new URL(url, window.location.href);
    return u.origin === window.location.origin ? u.pathname : null;
  } catch {
    return null;
  }
}

let installed = false;

export function installSessionFetch(cookieSource: () => string = () => document.cookie): void {
  if (installed || typeof window === 'undefined' || typeof window.fetch !== 'function') return;
  installed = true;
  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const method = (init?.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    const path = sameOriginPath(url);
    let nextInit = init;
    if (path && MUTATING.has(method)) {
      const csrf = readCookie(CSRF_COOKIE, cookieSource());
      const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
      if (csrf && !headers.has('X-CSRF-Token')) headers.set('X-CSRF-Token', csrf);
      nextInit = { ...init, headers };
    }
    const response = await original(input, nextInit);
    if (response.status === 401 && path && path.startsWith('/api/') && !path.startsWith('/api/auth/')) {
      window.dispatchEvent(new CustomEvent(SESSION_EXPIRED_EVENT));
    }
    return response;
  };
}

/** Test hook: allow re-installing over a fresh mock fetch. */
export function _resetSessionFetchForTests(): void {
  installed = false;
}
