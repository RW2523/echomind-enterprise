import React from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import axe from 'axe-core';
import { keys } from '../api/queries';
import { useBlockMatterSwitch } from '../app/navigationGuard';
import { matter, me, mockApi, renderLaw } from './harness';

const M101 = matter('M-101', 'MSA renewal');
const M114 = matter('M-114', 'Series B diligence', 'Alpha Holdings', { sensitivity: 'standard', my_role: 'attorney' });

function signedIn(roles = ['attorney'], matters = [M101, M114], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(roles)],
    'GET /api/legal/matters': () => [200, { matters }],
    'GET /api/legal/clients': () => [200, { clients: [] }],
    ...extra,
  });
}

async function axeViolations(container: HTMLElement) {
  const r = await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } });
  return r.violations.map((v) => `${v.id}: ${v.nodes.length}`);
}

describe('Law workspace', () => {
  afterEach(() => cleanup());

  it('lists only the matters the server returns for this person', async () => {
    signedIn();
    renderLaw('/matters');
    expect(await screen.findByText('MSA renewal')).toBeTruthy();
    expect(screen.getByText('Series B diligence')).toBeTruthy();
    expect(screen.queryByText(/M-202/)).toBeNull();
  });

  it('shows the same "not available" page for a hidden or a non-existent matter', async () => {
    for (const path of ['/m/M-202/overview', '/m/NO-SUCH/overview']) {
      signedIn();
      renderLaw(path);
      expect(await screen.findByText("This item isn't available.")).toBeTruthy();
      expect(screen.getByText(/It may not exist, or you may not have access/)).toBeTruthy();
      cleanup();
    }
  });

  it('opens a matter by number and shows its overview', async () => {
    signedIn();
    renderLaw('/m/M-101/overview');
    expect(await screen.findByRole('heading', { name: 'MSA renewal' })).toBeTruthy();
    expect(screen.getByText('Restricted')).toBeTruthy();
  });

  it('leaves out what the person cannot use', async () => {
    signedIn(['paralegal'], [{ ...M101, my_role: 'read_only' }]);
    renderLaw('/m/M-101/overview');
    await screen.findByRole('heading', { name: 'MSA renewal' });
    expect(screen.queryByText('Users & roles')).toBeNull();          // not a firm admin
    expect(screen.queryByText('Edit details')).toBeNull();           // not the matter lead
    expect(screen.queryByText('Close matter')).toBeNull();
    cleanup();
    signedIn(['firm_admin', 'attorney']);
    renderLaw('/matters');
    expect(await screen.findByText('Users & roles')).toBeTruthy();
    expect(screen.getByRole('button', { name: /New matter/ })).toBeTruthy();
  });

  it('switching matter drops the previous matter\'s cached data', async () => {
    signedIn(undefined, undefined, { 'GET /api/legal/matters/id-M-101/memberships': () => [200, { memberships: [] }] });
    const { queryClient, router } = renderLaw('/m/M-101/members');
    await screen.findByText(/Only these people can see/);
    expect(queryClient.getQueryData(keys.memberships('id-M-101'))).toBeDefined();
    fireEvent.click(screen.getByRole('button', { name: /Switch matter/ }));
    fireEvent.click(await screen.findByRole('option', { name: /M-114/ }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/m/M-114/overview'));
    expect(queryClient.getQueryData(keys.memberships('id-M-101'))).toBeUndefined();
  });

  it('asks before switching away while something is in progress', async () => {
    signedIn();
    function Recording() { useBlockMatterSwitch('A recording is in progress'); return null; }
    const extra = render(<Recording />);
    const { router } = renderLaw('/m/M-101/overview');
    await screen.findByRole('heading', { name: 'MSA renewal' });
    fireEvent.click(screen.getByRole('button', { name: /Switch matter/ }));
    fireEvent.click(await screen.findByRole('option', { name: /M-114/ }));
    expect(await screen.findByText('A recording is in progress')).toBeTruthy();
    expect(router.state.location.pathname).toBe('/m/M-101/overview');
    fireEvent.click(screen.getByRole('button', { name: 'Switch matter' }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/m/M-114/overview'));
    extra.unmount();
  });

  it('sends a signed-out visitor to sign-in and keeps the destination', async () => {
    mockApi({ 'GET /api/auth/me': () => [401, { detail: 'Not authenticated' }] });
    const { router } = renderLaw('/m/M-101/overview');
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'));
    expect(router.state.location.search).toBe('?next=%2Fm%2FM-101%2Foverview');
  });

  it('makes a person with an administrator-set password choose their own first', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'], { must_change_password: true })] });
    renderLaw('/home');
    expect(await screen.findByText('Choose your own password')).toBeTruthy();
    expect(screen.queryByText('Your open matters')).toBeNull();
  });

  it('matters page and members page have no automatic accessibility violations', async () => {
    signedIn(undefined, undefined, {
      'GET /api/legal/matters/id-M-101/memberships': () => [200, { memberships: [
        { id: 'mm1', user_id: 'usr_alice', username: 'alice', display_name: 'alice', role: 'lead', can_export: true, valid_from: '', granted_at: '' },
      ] }],
    });
    const a = renderLaw('/matters');
    await screen.findByText('MSA renewal');
    expect(await axeViolations(a.container)).toEqual([]);
    cleanup();
    const b = renderLaw('/m/M-101/members');
    await screen.findByText(/Only these people can see/);
    expect(await axeViolations(b.container)).toEqual([]);
  });
});

describe('Sign-in', () => {
  afterEach(() => cleanup());

  function fillPassword() {
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'n.new' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'an initial passphrase' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));
  }

  it('binds an authenticator only through an enrolment link, then shows recovery codes once', async () => {
    const calls = mockApi({
      'POST /api/auth/login': () => [200, { status: 'mfa_enrollment_required' }],
      'POST /api/auth/mfa/enroll': () => [200, { secret: 'ABCDEFGHIJKLMNOP', otpauth_uri: 'otpauth://totp/EchoMind%20Law:n.new?secret=ABCDEFGHIJKLMNOP' }],
      'POST /api/auth/mfa/confirm': () => [200, { status: 'ok', recovery_codes: ['aaaaa-bbbbb', 'ccccc-ddddd'] }],
      'GET /api/auth/me': () => [200, me()],
      'GET /api/legal/matters': () => [200, { matters: [] }],
    });
    const { router } = renderLaw('/signin?enrol=link-123');
    fillPassword();
    expect(await screen.findByAltText('Authenticator QR code')).toBeTruthy();
    expect(screen.getByTestId('copy-value').textContent).toBe('ABCD EFGH IJKL MNOP');
    fireEvent.change(screen.getByLabelText('Authenticator code'), { target: { value: '123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Turn on two-step sign-in' }));
    expect((await screen.findByTestId('recovery-codes')).textContent).toContain('aaaaa-bbbbb');
    expect(calls.find((c) => c.path === '/api/auth/mfa/enroll')!.body).toEqual({ enrollment_token: 'link-123' });
    expect(calls.find((c) => c.path === '/api/auth/mfa/confirm')!.body).toEqual({ code: '123456', enrollment_token: 'link-123' });
    fireEvent.click(screen.getByRole('button', { name: "I've saved my recovery codes" }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/home'));
  });

  it('without a link, explains that an administrator must issue one', async () => {
    const calls = mockApi({ 'POST /api/auth/login': () => [200, { status: 'mfa_enrollment_required' }] });
    renderLaw('/signin');
    fillPassword();
    expect(await screen.findByText(/Ask your administrator for your enrolment link/)).toBeTruthy();
    expect(calls.some((c) => c.path === '/api/auth/mfa/enroll')).toBe(false);
  });

  it('never redirects outside the app after sign-in', async () => {
    mockApi({
      'POST /api/auth/login': () => [200, { status: 'mfa_required' }],
      'POST /api/auth/mfa/verify': () => [200, { status: 'ok' }],
      'GET /api/auth/me': () => [200, me()],
      'GET /api/legal/matters': () => [200, { matters: [] }],
    });
    const { router } = renderLaw('/signin?next=//evil.example/steal');
    fillPassword();
    fireEvent.change(await screen.findByLabelText('Authenticator code'), { target: { value: '123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/home'));
  });

  it('shows the server\'s message when sign-in fails', async () => {
    mockApi({ 'POST /api/auth/login': () => [401, { detail: 'Sign-in failed. Check your username and password, or try again later.' }] });
    renderLaw('/signin');
    fillPassword();
    expect((await screen.findByRole('alert')).textContent).toMatch(/Sign-in failed/);
  });

  it('has no automatic accessibility violations', async () => {
    mockApi({});
    const { container } = renderLaw('/signin');
    await screen.findByLabelText('Username');
    expect(await axeViolations(container)).toEqual([]);
  });
});

describe('monogram', () => {
  it('uses the distinctive words of a client name', async () => {
    const { monogram } = await import('../ui/MatterSwitcher');
    expect(monogram('Voss-Lindqvist AB')).toBe('VL');
    expect(monogram('Alder Finch Ltd')).toBe('AF');
    expect(monogram('Beta Shipping Corp.')).toBe('BS');
    expect(monogram('Acme')).toBe('A');
  });
});
