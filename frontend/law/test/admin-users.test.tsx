import { cleanup, fireEvent, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { me, mockApi, renderLaw } from './harness';

/** Commercial-readiness review F6 (2026-10-28): a managing partner is appointed by another managing partner. */
const USERS = [
  { username: 'alice', roles: ['managing_partner'], mfa_enabled: true, status: 'active' },
  { username: 'carol', roles: ['paralegal'], mfa_enabled: true, status: 'active' },
];

function world(roles: string[], calls: string[] = []) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(roles)],
    'GET /api/legal/matters': () => [200, { matters: [] }],
    'GET /api/legal/users': () => [200, { users: USERS }],
    'POST /api/legal/users/carol/appoint-managing-partner': () => { calls.push('appoint'); return [200, { user: USERS[1], sessions_revoked: 0 }]; },
  });
}

describe('Users & roles and managing partners', () => {
  afterEach(cleanup);
  it('a managing partner appoints another, after confirming, and does not manage accounts', async () => {
    const calls: string[] = [];
    world(['managing_partner'], calls);
    renderLaw('/admin/users');
    const row = (await screen.findByText('carol')).closest('tr')!;
    expect(screen.queryByRole('button', { name: /New account/ })).toBeNull();
    expect(within(row).queryByRole('button', { name: 'Roles' })).toBeNull();
    fireEvent.click(within(row).getByRole('button', { name: 'Appoint managing partner' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Appoint' }));
    await screen.findByText('carol');
    expect(calls).toEqual(['appoint']);
  });

  it('an administrator cannot tick managing partner once the firm has one, and is told why', async () => {
    world(['firm_admin']);
    renderLaw('/admin/users');
    const row = (await screen.findByText('carol')).closest('tr')!;
    expect(within(row).queryByRole('button', { name: 'Appoint managing partner' })).toBeNull();
    fireEvent.click(within(row).getByRole('button', { name: 'Roles' }));
    const dialog = await screen.findByRole('dialog', { name: 'Firm roles for carol' });
    const box = within(dialog).getByRole('checkbox', { name: 'Managing partner' }) as HTMLInputElement;
    expect(box.disabled).toBe(true);
    expect(screen.getByText(/appointed by another managing partner, not here/)).toBeTruthy();
  });
});
