import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { me, mockApi, renderLaw } from './harness';

/** D-076 — single sign-on with the firm's identity provider, on the sign-in page. Off unless the server says so. */
describe('Signing in with the firm\'s account (D-076)', () => {
  afterEach(() => cleanup());

  it('offers the firm\'s account only when the server has one, and links to the server to start', async () => {
    mockApi({ 'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', oidc: { label: 'Example Firm account' } }] });
    renderLaw('/signin');
    const link = await screen.findByRole('link', { name: 'Sign in with Example Firm account' });
    expect(link.getAttribute('href')).toBe('/api/auth/oidc/start');
    expect(screen.getByLabelText('Username')).toBeTruthy();                       // the password stays available
    cleanup();
    mockApi({ 'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law' }] });
    renderLaw('/signin');
    await screen.findByLabelText('Username');
    expect(screen.queryByRole('link', { name: /^Sign in with / })).toBeNull();
  });

  it('back from the provider, asks for this server\'s code, then opens the workspace', async () => {
    mockApi({
      'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', oidc: { label: 'Example Firm account' } }],
      'POST /api/auth/mfa/verify': () => [200, { status: 'ok' }],
      'GET /api/auth/me': () => [200, me()],
      'GET /api/legal/matters': () => [200, { matters: [] }],
    });
    const { router } = renderLaw('/signin?step=code');
    fireEvent.change(await screen.findByLabelText('Authenticator code'), { target: { value: '123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/home'));
  });

  it('says plainly when the provider\'s sign-in did not work, without saying why', async () => {
    mockApi({ 'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', oidc: { label: 'Example Firm account' } }] });
    renderLaw('/signin?oidc_error=account');
    expect((await screen.findByRole('alert')).textContent).toMatch(/not set up here, or is not active\. Ask your administrator/);
    cleanup();
    mockApi({ 'GET /api/auth/config': () => [200, { auth_enabled: true, edition: 'law', oidc: { label: 'Example Firm account' } }] });
    renderLaw('/signin?oidc_error=failed');
    expect((await screen.findByRole('alert')).textContent).toMatch(/did not complete\. Try again, or sign in with your password/);
  });
});
