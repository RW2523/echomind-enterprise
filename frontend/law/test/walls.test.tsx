import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { matter, me, mockApi, renderLaw } from './harness';

const wall = (over = {}) => ({
  id: 'w1', name: 'Conflict screen', rationale: 'Advised the counterparty', status: 'requested', requested_by: 'morgan',
  requested_at: '2026-10-08T10:00:00.000Z', screened: ['alice'], protected: [{ type: 'matter', id: 'id-M-101', label: 'M-101 · MSA renewal (Alpha)' }],
  exceptions: [], ...over,
});

describe('Ethical walls and emergency access screens', () => {
  afterEach(() => cleanup());

  it('are absent for people without the role', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/admin/walls');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
    expect(screen.queryByText('Ethical walls')).toBeNull();
    expect(screen.queryByText('Emergency access')).toBeNull();
  });

  it('offer approval only to a managing partner who did not ask', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, { ...me(['managing_partner']), user: { ...me().user, username: 'nakamura', roles: ['managing_partner'] } }],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/walls': () => [200, { walls: [wall(), wall({ id: 'w2', name: 'Own request', requested_by: 'nakamura' })] }],
      'POST /api/legal/walls/w1/approve': () => [200, { wall: wall({ status: 'active' }) }],
    });
    renderLaw('/admin/walls');
    const first = await screen.findByRole('region', { name: 'Conflict screen' });
    const own = screen.getByRole('region', { name: 'Own request' });
    expect(own.querySelector('button')?.textContent ?? '').not.toMatch(/Approve/);
    fireEvent.click(first.querySelector('button')!);
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/legal/walls/w1/approve')).toBe(true));
  });

  it('show a read-only banner on a matter opened through emergency access', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['firm_admin'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-215', 'Lease dispute', 'Beta', { my_role: 'break_glass' })] }],
      'GET /api/legal/break-glass': () => [200, { can_approve: false, requests: [{ id: 'g1', matter_number: 'M-215', status: 'active', minutes: 60,
        reason: 'x', requested_at: '2026-10-08T10:00:00.000Z', expires_at: '2026-10-08T11:00:00.000Z' }] }],
      'POST /api/legal/break-glass/g1/end': () => [200, { request: {} }],
    });
    renderLaw('/m/M-215/overview');
    expect(await screen.findByText('Emergency access.')).toBeTruthy();
    expect(screen.queryByText('Edit details')).toBeNull();
    fireEvent.click(await screen.findByRole('button', { name: 'End now' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/break-glass/g1/end')).toBe(true));
  });

  it('let a firm admin request emergency access by matter number', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['firm_admin'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/break-glass': () => [200, { can_approve: false, requests: [] }],
      'POST /api/legal/break-glass': () => [200, { request: { id: 'g9', status: 'requested' } }],
    });
    renderLaw('/admin/emergency-access');
    fireEvent.change(await screen.findByLabelText('Matter number'), { target: { value: 'M-215' } });
    fireEvent.change(screen.getByLabelText('Reason'), { target: { value: 'Lead unreachable; filing due today' } });
    fireEvent.click(screen.getByRole('button', { name: /Request emergency access/ }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/break-glass')?.body)
      .toEqual({ matter_number: 'M-215', reason: 'Lead unreachable; filing due today', minutes: 60 }));
  });
});
