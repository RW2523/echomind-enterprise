import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import axe from 'axe-core';
import type { Me } from '../api/types';
import { me, mockApi, renderLaw } from './harness';

/** D-077 — one-click demo sign-in: the demo's fictional people, no password or code; the pages say it is on. */
const PEOPLE = [{ username: 'alice', roles: ['attorney'] }, { username: 'harper', roles: ['attorney'] }, { username: 'nakamura', roles: ['managing_partner'] }];

describe('One-click demo sign-in (D-077)', () => {
  afterEach(() => cleanup());

  it("the owner's open demo: no sign-in step, the page enters as the named demo person by itself", async () => {
    sessionStorage.clear();
    const calls = mockApi({
      'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', demo_signin: true }],
      'GET /api/auth/demo': () => [200, { people: PEOPLE, auto: 'alice' }],
      'POST /api/auth/demo/alice': () => [200, { status: 'ok', user: { username: 'alice' } }],
      'GET /api/auth/me': () => [200, me()],
      'GET /api/legal/matters': () => [200, { matters: [] }],
    });
    const { router } = renderLaw('/signin');
    await waitFor(() => expect(router.state.location.pathname).toBe('/chat'));
    expect(calls.filter((c) => c.method === 'POST' && c.path === '/api/auth/demo/alice')).toHaveLength(1);
    expect(calls.some((c) => c.path === '/api/auth/login')).toBe(false);
  });

  it('the open demo still has a way in for other accounts (?manual=1), and does not loop if entering fails', async () => {
    sessionStorage.clear();
    const calls = mockApi({
      'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', demo_signin: true }],
      'GET /api/auth/demo': () => [200, { people: PEOPLE, auto: 'alice' }],
      'POST /api/auth/demo/alice': () => [403, { detail: 'This demo person is not active.' }],
    });
    renderLaw('/signin?manual=1');
    expect(await screen.findByRole('group', { name: 'Enter the demo as' })).toBeTruthy();
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
    cleanup();
    renderLaw('/signin');
    expect(await screen.findByText('This demo person is not active.')).toBeTruthy();       // tried once, says why
    cleanup();
    renderLaw('/signin');                                                                   // a minute later at the earliest
    expect(await screen.findByRole('group', { name: 'Enter the demo as' })).toBeTruthy();
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1);
  });

  it('lists the demo people; one click signs in as that person and opens the workspace', async () => {
    const calls = mockApi({
      'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', demo_signin: true }],
      'GET /api/auth/demo': () => [200, { people: PEOPLE }],
      'POST /api/auth/demo/harper': () => [200, { status: 'ok', user: { username: 'harper' } }],
      'GET /api/auth/me': () => [200, me()],
      'GET /api/legal/matters': () => [200, { matters: [] }],
    });
    const { router, container } = renderLaw('/signin');
    const people = await screen.findByRole('group', { name: 'Enter the demo as' });
    expect(screen.queryByLabelText('Password')).toBeNull();                      // only the people: no password or code form
    expect(screen.queryByLabelText('Authenticator code')).toBeNull();
    expect(screen.getByRole('button', { name: 'harper · Attorney' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'nakamura · Managing partner' })).toBeTruthy();
    expect(people.textContent).toMatch(/fictional people/i);
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
    fireEvent.click(screen.getByRole('button', { name: 'harper · Attorney' }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/chat'));
    expect(calls.some((c) => c.method === 'POST' && c.path === '/api/auth/demo/harper')).toBe(true);
    expect(screen.queryByLabelText('Password')).toBeNull();
  });

  it('keeps one small way in for other accounts (the administrator), which opens the password form', async () => {
    mockApi({
      'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', demo_signin: true }],
      'GET /api/auth/demo': () => [200, { people: PEOPLE }],
    });
    renderLaw('/signin');
    await screen.findByRole('group', { name: 'Enter the demo as' });
    fireEvent.click(screen.getByRole('button', { name: 'Other accounts' }));
    expect(screen.getByLabelText('Password')).toBeTruthy();
  });

  it('is not offered when the server has it off; the password stays', async () => {
    mockApi({ 'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law' }] });
    renderLaw('/signin');
    await screen.findByLabelText('Password');
    expect(screen.queryByRole('group', { name: 'Enter the demo as' })).toBeNull();
  });

  it('says on every page that demo sign-in is on', async () => {
    const who: Me = { ...me(['attorney']), features: { intake: false, intake_links: false, demo_signin: true } };
    mockApi({ 'GET /api/auth/me': () => [200, who], 'GET /api/legal/matters': () => [200, { matters: [] }], 'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }] });
    renderLaw('/home');
    expect((await screen.findByRole('status', { name: 'Demo sign-in' })).textContent).toMatch(/Demo sign-in is on/);
  });
});

describe('A first visit over HTTPS without a session', () => {
  it('goes straight to sign-in without asking the server (no 401 logged); with the CSRF cookie, or over HTTP, it asks', async () => {
    const { knownSignedOut } = await import('../app/session');
    expect(knownSignedOut('https:', '')).toBe(true);
    expect(knownSignedOut('https:', 'theme=dark')).toBe(true);
    expect(knownSignedOut('https:', '__Host-echomind_csrf=abc; theme=dark')).toBe(false);
    expect(knownSignedOut('http:', '')).toBe(false);
  });
});
