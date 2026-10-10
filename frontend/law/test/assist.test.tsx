import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { AssistAlert, MeetingDetail, Participant } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

vi.mock('../app/mic', () => ({ startMic: vi.fn(async () => ({ sampleRate: 16000, stop: vi.fn() })) }));

class FakeSocket {
  static last: FakeSocket | null = null;
  static OPEN = 1;
  readyState = 1; binaryType = ''; sent: unknown[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  constructor(public url: string) { FakeSocket.last = this; }
  send(d: unknown) { this.sent.push(d); }
  close() { this.readyState = 3; }
  server(obj: unknown) { this.onmessage?.({ data: JSON.stringify(obj) }); }
}

const BASE = '/api/legal/matters/id-M-101/meetings';
const granted = (id: string, name: string, role: Participant['role']): Participant =>
  ({ id, name, role, consent: { state: 'granted', method: 'written', at: '2026-10-09T10:00:00.000Z', by: 'alice' } });
const ASSIST = { pack: 'legal_client_call' as const, chosen: null, label: 'Client call', rules: [] };
const meeting = (over: Partial<MeetingDetail> = {}): MeetingDetail => ({
  id: 'mt1', title: 'Renewal call', kind: 'client_call', scheduled_at: null, status: 'planned', created_at: '2026-10-09T09:00:00.000Z',
  consent: { granted: 2, needed: 2 }, assist: ASSIST,
  policy: { mode: 'all_party', methods: ['written'], max_age_hours: 12, statement: 'Recorded for M-101.', source: 'P-001' },
  participants: [granted('p1', 'Helena Marsh', 'attorney'), granted('p2', 'Thomas Reyes', 'counterparty')], captures: [], ...over,
});
const ALERT: AssistAlert = {
  id: 'a1', meeting_id: 'mt1', segment_id: 's1', pack: 'legal_client_call', rule: 'signed_terms', verdict: 'differs', severity: 'critical',
  title: 'Differs from signed agreement, cl. 4.1', explanation: 'The fee differs [1] [2].', confidence: 0.8, threshold: 0.7, method: 'model',
  status: 'open', heard_at: '2026-10-09T10:00:00.000Z', raised_at: '2026-10-09T10:00:02.000Z',
  said: { segment_id: 's1', start: 40, end: 50, quote: 'USD 18,000' },
  source: { document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v1', version_number: 1, version_status: 'executed', signed: true,
    clause: '4.1', start: 100, end: 160, quote: '4.1 Client shall pay a platform subscription fee of USD 18,500 per month.' },
};
const TELEMETRY = { alerts: 1, critical: 1, per_hour: 12, confirmed: 0, dismissed: 0, dismiss_rate: null, latency_p50_s: 2, latency_max_s: 2,
  threshold: 0.7, thresholds_approved: false };

function world(m: MeetingDetail, alerts: AssistAlert[] = [], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${BASE}`]: () => [200, { meetings: [m] }],
    [`GET ${BASE}/mt1`]: () => [200, { meeting: m }],
    [`GET ${BASE}/mt1/transcript`]: () => [200, { segments: [], marks: [], participants: [], finalised: null }],
    [`GET ${BASE}/mt1/alerts`]: () => [200, { alerts, assist: ASSIST, telemetry: TELEMETRY }],
    ...extra,
  });
}

describe('Silent Assist', () => {
  beforeEach(() => { (globalThis as any).WebSocket = FakeSocket; FakeSocket.last = null; });
  afterEach(() => cleanup());

  it('shows alerts live, with what was said, what it differs from, and that the thresholds await sign-off', async () => {
    world(meeting());
    const { container } = renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Start recording' }));
    await act(async () => { FakeSocket.last!.server({ type: 'ready', capture_id: 'c1', consent_snapshot: 'x', transcription: 'available' }); });
    const panel = await screen.findByRole('region', { name: 'Silent Assist' });
    expect(within(panel).getByText('Listening. Alerts appear here.')).toBeTruthy();
    expect(within(panel).getByText(/awaiting a lawyer’s sign-off/)).toBeTruthy();
    await act(async () => { FakeSocket.last!.server({ type: 'alert', ...ALERT }); });
    const card = within(panel).getByRole('listitem', { name: 'Differs from signed agreement, cl. 4.1' });
    expect(within(card).getByText('Differs')).toBeTruthy();
    expect(card.textContent).toContain('“USD 18,000”');
    expect(card.textContent).toContain('Harrowgate MSA v1 · Signed: “4.1 Client shall pay');
    expect(within(card).getByText('Checked in context · confidence 80% (threshold 70%)')).toBeTruthy();
    expect(within(card).getByRole('link', { name: /Open the passage/ }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=100-160');
    expect(card.textContent).not.toMatch(/\btrue\b|\bfalse\b/i);
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('confirms an alert, and dismisses one only with a reason', async () => {
    const calls = world(meeting({ status: 'ended' }), [ALERT, { ...ALERT, id: 'a2', rule: 'action', verdict: 'needs_review', severity: 'notice',
      title: 'Action taken on', source: null, method: 'rule', said: { segment_id: 's2', start: 0, end: 20, quote: 'I will send the redline' } }], {
      [`POST ${BASE}/mt1/alerts/a1/confirm`]: () => [200, { alert: { ...ALERT, status: 'confirmed', decided_by: 'alice' } }],
      [`POST ${BASE}/mt1/alerts/a2/dismiss`]: () => [200, { alert: { ...ALERT, id: 'a2', status: 'dismissed', decision_reason: 'On the task list' } }],
    });
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    const panel = await screen.findByRole('region', { name: 'Silent Assist' });
    const fee = within(panel).getByRole('listitem', { name: 'Differs from signed agreement, cl. 4.1' });
    fireEvent.click(within(fee).getByRole('button', { name: 'Confirm' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/mt1/alerts/a1/confirm`)).toBe(true));
    const action = within(panel).getByRole('listitem', { name: 'Action taken on' });
    fireEvent.click(within(action).getByRole('button', { name: 'Dismiss…' }));
    const dismiss = within(action).getByRole('button', { name: 'Dismiss' }) as HTMLButtonElement;
    expect(dismiss.disabled).toBe(true);
    fireEvent.change(within(action).getByLabelText('Why dismiss this alert?'), { target: { value: 'On the task list' } });
    fireEvent.click(dismiss);
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/mt1/alerts/a2/dismiss`)?.body).toEqual({ reason: 'On the task list' }));
    expect(within(panel).getByText(/1 alert · 12 an hour · 0 confirmed · 0 dismissed · median 2 s after the words/)).toBeTruthy();
  });

  it('chooses the pack for a meeting, or turns alerts off', async () => {
    const calls = world(meeting(), [], { [`PATCH ${BASE}/mt1/assist`]: () => [200, { assist: { ...ASSIST, pack: null, chosen: 'off' } }] });
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Prepare' }));
    const pack = await screen.findByLabelText('Pack') as HTMLSelectElement;
    expect(within(pack).getByRole('option', { name: 'The default for this kind of meeting (Client call)' })).toBeTruthy();
    fireEvent.change(pack, { target: { value: 'off' } });
    await waitFor(() => expect(calls.find((c) => c.method === 'PATCH')?.body).toEqual({ pack: 'off' }));
  });
});
