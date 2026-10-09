import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { AuditEvent } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const ev = (seq: number, over: Partial<AuditEvent> = {}): AuditEvent => ({
  seq, ts: '2026-10-08T10:00:00.000Z', action: 'matter.read', outcome: 'allow', reason: null,
  actor: { username: 'alice', roles: ['attorney'] }, auth: 'session', session_ref: 'abcd1234abcd1234', ip: '10.0.0.5',
  object_type: 'matter', object_id: 'id-M-101', matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, details: null, ...over,
});
const TRAIL = [
  ev(5, { action: 'membership.add', details: { username: 'carol', role: 'paralegal' } }),
  ev(4, { action: 'matter.read', outcome: 'deny', reason: 'not_found', actor: { username: 'bob', roles: ['attorney'] },
    matter: { id: 'id-M-202', number: 'M-202', title: 'Board investigation' } }),
  ev(3, { action: 'auth.login', outcome: 'deny', reason: 'bad_password', matter: null, object_type: null }),
];

async function axeViolations(container: HTMLElement) {
  const r = await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } });
  return r.violations.map((v) => `${v.id}: ${v.nodes.length}`);
}

describe('Audit trail', () => {
  afterEach(() => cleanup());

  it('is absent for people without a governance role', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/admin/audit');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
    expect(screen.queryByText('Audit trail')).toBeNull();
  });

  it('reads as sentences, with refusals marked in words', async () => {
    mockApi({
      'GET /api/auth/me': () => [200, me(['managing_partner'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/audit': () => [200, { events: TRAIL, next_before: null, shows_matter_numbers: true }],
    });
    const { container } = renderLaw('/admin/audit');
    expect(await screen.findByText(/added carol to M-101 as Paralegal/)).toBeTruthy();
    expect(screen.getByText(/tried to view M-202/)).toBeTruthy();
    expect(screen.getByText('not visible to them')).toBeTruthy();
    expect(screen.getByText('wrong password')).toBeTruthy();
    expect(within(screen.getByRole('region', { name: 'Events' })).getAllByText('Refused')).toHaveLength(2);
    expect(await axeViolations(container)).toEqual([]);
  });

  it('tells a firm administrator that matters appear only as references', async () => {
    mockApi({
      'GET /api/auth/me': () => [200, me(['firm_admin'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/audit': () => [200, { events: [ev(1, { object_id: '7c3ee26e-c380', matter: { id: '7c3ee26e-c380-46c5-b412-30ceb29ee60e' } })],
        next_before: null, shows_matter_numbers: false }],
    });
    renderLaw('/admin/audit');
    expect(await screen.findByText(/each matter appears as an opaque reference/)).toBeTruthy();
    expect(screen.getByText(/viewed a matter/)).toBeTruthy();
    expect(screen.queryByText(/M-101/)).toBeNull();
    expect(screen.getByText('ref 7c3ee26e')).toBeTruthy();
  });

  it('filters on the server and pages back through older events', async () => {
    mockApi({
      'GET /api/auth/me': () => [200, me(['managing_partner'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/audit': () => [200, { events: [ev(9)], next_before: 9, shows_matter_numbers: true }],
    });
    const fetchSpy = globalThis.fetch as any;
    renderLaw('/admin/audit');
    await screen.findByText(/viewed M-101/);
    fireEvent.change(screen.getByLabelText('Outcome'), { target: { value: 'deny' } });
    await waitFor(() => expect(fetchSpy.mock.calls.some((c: any[]) => String(c[0]).includes('outcome=deny'))).toBe(true));
    fireEvent.click(await screen.findByRole('button', { name: 'Show older events' }));
    await waitFor(() => expect(fetchSpy.mock.calls.some((c: any[]) => String(c[0]).includes('before=9'))).toBe(true));
  });

  it('checks integrity and says plainly when the trail is broken', async () => {
    let ok = true;
    mockApi({
      'GET /api/auth/me': () => [200, me(['records_admin'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/audit': () => [200, { events: [ev(1)], next_before: null, shows_matter_numbers: true }],
      'GET /api/legal/audit/verify': () => [200, ok
        ? { ok: true, events: 1204, head: 'f'.repeat(64), head_seq: 1204, broken_at: null, problem: null, anchors_checked: 3, verified_at: '2026-10-08T10:42:00Z' }
        : { ok: false, events: 1204, head: 'f'.repeat(64), head_seq: 1204, broken_at: 312, problem: 'event_altered', anchors_checked: 3, verified_at: '2026-10-08T10:42:00Z' }],
    });
    renderLaw('/admin/audit');
    const box = await screen.findByRole('region', { name: 'Integrity' });
    fireEvent.click(within(box).getByRole('button', { name: 'Check now' }));
    expect(await within(box).findByText('Intact')).toBeTruthy();
    expect(within(box).getByText(/1,204 events checked against 3 daily anchors/)).toBeTruthy();
    ok = false;
    fireEvent.click(within(box).getByRole('button', { name: 'Check again' }));
    expect(await within(box).findByText('Broken')).toBeTruthy();
    expect(within(box).getByText(/changed after it was written\. First affected event: #312/)).toBeTruthy();
  });
});

describe('Matter activity', () => {
  afterEach(() => cleanup());

  it('is a tab for the lead only', async () => {
    mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'attorney' })] }],
    });
    renderLaw('/m/M-101/activity');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
    expect(screen.queryByRole('link', { name: 'Activity' })).toBeNull();
  });

  it('shows who did what on the matter', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/matters/id-M-101/activity': () => [200, { events: [ev(2, { actor: { username: 'admin2', roles: ['firm_admin'] }, reason: 'break_glass' })], next_before: null }],
    });
    const { container } = renderLaw('/m/M-101/activity');
    expect(await screen.findByText(/viewed M-101/)).toBeTruthy();
    expect(screen.getByText('Emergency access')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Activity' })).toBeTruthy();
    expect(calls.some((c) => c.path === '/api/legal/matters/id-M-101/activity')).toBe(true);
    expect(await axeViolations(container)).toEqual([]);
  });
});

describe('Opening a matter', () => {
  afterEach(() => cleanup());

  it('is a recorded read, and nothing is shown if it could not be recorded', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/matters/id-M-101': () => [500, { detail: 'This action could not be recorded in the audit trail.' }],
    });
    renderLaw('/m/M-101/overview');
    expect(await screen.findByText(/This matter could not be opened/)).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'MSA renewal' })).toBeNull();
    expect(calls.filter((c) => c.path === '/api/legal/matters/id-M-101')).toHaveLength(1);
  });
});
