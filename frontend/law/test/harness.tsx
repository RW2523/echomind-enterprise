import React from 'react';
import { render } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import { lawRoutes, makeQueryClient } from '../app/LawApp';
import type { Matter, Me } from '../api/types';

type Reply = [number, unknown];
type Handler = (body: any) => Reply;

/** Route the app's fetch calls to in-memory handlers keyed "METHOD /path". Unknown calls fail loudly. */
export function mockApi(handlers: Record<string, Handler>) {
  const calls: { method: string; path: string; body: any }[] = [];
  globalThis.fetch = vi.fn(async (input: any, init?: RequestInit) => {
    const url = new URL(String(input), 'https://law.test');
    const method = (init?.method || 'GET').toUpperCase();
    const raw = init?.body;
    const body = raw instanceof FormData
      ? Object.fromEntries(Array.from(raw.entries()).map(([k, v]) => [k, typeof v === 'string' ? v : (v as File).name]))
      : raw instanceof ArrayBuffer ? { bytes: raw.byteLength }                      // audio (S5.1)
      : raw ? JSON.parse(String(raw)) : undefined;
    calls.push({ method, path: url.pathname + url.search, body });
    let h = handlers[`${method} ${url.pathname}${url.search}`] ?? handlers[`${method} ${url.pathname}`];
    // Opening a matter (GET /api/legal/matters/:id) answers like the server: from the person's own list.
    const open = /^\/api\/legal\/matters\/([^/]+)$/.exec(url.pathname);
    if (!h && method === 'GET' && open && handlers['GET /api/legal/matters']) {
      const [, list] = handlers['GET /api/legal/matters'](undefined) as [number, { matters: Matter[] }];
      const found = list.matters.find((m) => m.id === decodeURIComponent(open[1]));
      h = () => (found ? [200, { matter: found }] : [404, { detail: 'Not Found' }]);
    }
    if (!h) return new Response(JSON.stringify({ detail: `unmocked ${method} ${url.pathname}` }), { status: 599 });
    const [status, json] = h(body);
    if (json && typeof json === 'object' && Array.isArray((json as any).__ndjson)) {   // a streamed NDJSON answer
      const text = (json as any).__ndjson.map((e: unknown) => JSON.stringify(e)).join('\n') + '\n';
      return new Response(text, { status, headers: { 'Content-Type': 'application/x-ndjson' } });
    }
    if (json && typeof json === 'object' && typeof (json as any).__binary === 'string') {   // audio (S5.1)
      return new Response(new Uint8Array([82, 73, 70, 70]), { status, headers: { 'Content-Type': (json as any).__binary } });
    }
    return new Response(JSON.stringify(json), { status, headers: { 'Content-Type': 'application/json' } });
  }) as any;
  return calls;
}

export function renderLaw(path: string) {
  const queryClient = makeQueryClient();
  queryClient.setDefaultOptions({ queries: { retry: false, staleTime: Infinity } });
  const router = createMemoryRouter(lawRoutes, { initialEntries: [path] });
  const utils = render(<QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider>);
  return { ...utils, router, queryClient };
}

export const me = (roles: string[] = ['attorney'], extra: Partial<Me['session']> = {}): Me => ({
  user: { id: 'usr_alice', username: 'alice', role: 'user', roles: roles as any },
  session: { auth: 'session', mfa_enrolled: true, must_change_password: false, idle_expires_at: 0, abs_expires_at: 0, ...extra },
});

export const matter = (number: string, title: string, client = 'Alpha Holdings', extra: Partial<Matter> = {}): Matter => ({
  id: `id-${number}`, number, title, status: 'open', sensitivity: 'restricted', version: 1, created_at: '2026-10-08T00:00:00.000Z',
  client: { id: `c-${client}`, name: client }, my_role: 'lead', ...extra,
});
