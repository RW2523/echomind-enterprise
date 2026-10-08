import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { _resetSessionFetchForTests, installSessionFetch, readCookie, SESSION_EXPIRED_EVENT } from './session';

describe('session fetch', () => {
  let calls: { url: string; init?: RequestInit }[];
  let status: number;

  beforeEach(() => {
    calls = [];
    status = 200;
    window.fetch = vi.fn(async (input: any, init?: RequestInit) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status });
    }) as any;
    // A __Host- cookie needs a secure page; jsdom (rightly) refuses it on http://localhost, so the
    // cookie string is injected.
    _resetSessionFetchForTests();
    installSessionFetch(() => 'theme=dark; __Host-echomind_csrf=abc%3D123');
  });
  afterEach(() => _resetSessionFetchForTests());

  const header = (i: number) => new Headers(calls[i].init?.headers).get('X-CSRF-Token');

  it('adds the CSRF header to same-origin state-changing requests only', async () => {
    await fetch('/api/chat/create', { method: 'POST', body: '{}' });
    await fetch('/api/docs/list');
    await fetch('https://elsewhere.example/api/x', { method: 'POST' });
    expect(header(0)).toBe('abc=123');
    expect(header(1)).toBeNull();
    expect(header(2)).toBeNull();
  });

  it('keeps headers the caller already set', async () => {
    await fetch('/api/docs/x', { method: 'DELETE', headers: { 'Content-Type': 'application/json' } });
    const h = new Headers(calls[0].init?.headers);
    expect(h.get('Content-Type')).toBe('application/json');
    expect(h.get('X-CSRF-Token')).toBe('abc=123');
  });

  it('announces an expired session on a 401 from the app API, not from /api/auth', async () => {
    const seen = vi.fn();
    window.addEventListener(SESSION_EXPIRED_EVENT, seen);
    status = 401;
    await fetch('/api/auth/login', { method: 'POST' });
    expect(seen).not.toHaveBeenCalled();
    await fetch('/api/docs/list');
    expect(seen).toHaveBeenCalledTimes(1);
    window.removeEventListener(SESSION_EXPIRED_EVENT, seen);
  });

  it('reads cookies by exact name', () => {
    expect(readCookie('b', 'ab=1; b=2; bb=3')).toBe('2');
    expect(readCookie('missing', 'a=1')).toBe('');
  });
});
