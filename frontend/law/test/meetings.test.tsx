import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { MeetingDetail, Participant } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const micStop = vi.fn();
vi.mock('../app/mic', () => ({ startMic: vi.fn(async () => ({ sampleRate: 16000, stop: micStop })) }));

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
const person = (id: string, name: string, state: Participant['consent']['state'] = 'missing'): Participant =>
  ({ id, name, role: 'client', consent: { state, method: state === 'granted' ? 'verbal_on_record' : null, at: state === 'granted' ? '2026-10-08T10:00:00.000Z' : null, by: 'alice' } });
const meeting = (people: Participant[], over: Partial<MeetingDetail> = {}): MeetingDetail => ({
  id: 'mt1', title: 'Call with Voss-Lindqvist', kind: 'client_call', scheduled_at: null, status: 'planned', created_at: '2026-10-08T09:00:00.000Z',
  consent: { granted: people.filter((p) => p.consent.state === 'granted').length, needed: people.length },
  policy: { mode: 'all_party', methods: ['verbal_on_record', 'written', 'host_attestation'], max_age_hours: 12,
    statement: 'This meeting will be recorded and transcribed by EchoMind for the firm, for matter M-101 only.', source: 'P-001' },
  participants: people, captures: [], ...over,
});

function world(m: MeetingDetail, extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${BASE}`]: () => [200, { meetings: [m] }],
    [`GET ${BASE}/mt1`]: () => [200, { meeting: m }],
    [`GET ${BASE}/mt1/transcript`]: () => [200, { segments: [] }],
    ...extra,
  });
}

describe('Meetings and consent', () => {
  beforeEach(() => { (globalThis as any).WebSocket = FakeSocket; FakeSocket.last = null; micStop.mockReset(); });
  afterEach(() => cleanup());

  it('lists meetings with how many have consented', async () => {
    world(meeting([person('p1', 'Alice Hart', 'granted'), person('p2', 'Petra Voss')]));
    renderLaw('/m/M-101/meetings');
    expect(await screen.findByRole('link', { name: 'Call with Voss-Lindqvist' })).toBeTruthy();
    expect(screen.getByText('1 of 2 consented')).toBeTruthy();
  });

  it('records consent only after the statement was read, and lets anyone withdraw it', async () => {
    const calls = world(meeting([person('p1', 'Alice Hart', 'granted'), person('p2', 'Petra Voss')]), {
      [`POST ${BASE}/mt1/participants/p2/consent`]: () => [200, { ok: true }],
      [`POST ${BASE}/mt1/participants/p1/consent/withdraw`]: () => [200, { ok: true }],
    });
    const { container } = renderLaw('/m/M-101/meetings/mt1');
    expect(await screen.findByText(/for matter M-101 only/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Record consent' }));
    const dialog = await screen.findByRole('dialog', { name: 'Consent of Petra Voss' });
    const save = within(dialog).getByRole('button', { name: 'Record consent' }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(within(dialog).getByLabelText('How was consent given?'), { target: { value: 'written' } });
    fireEvent.click(within(dialog).getByRole('checkbox'));
    fireEvent.click(save);
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/mt1/participants/p2/consent`)?.body).toEqual({ method: 'written', statement_read: true }));
    fireEvent.click(screen.getByRole('button', { name: 'Withdraw' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/mt1/participants/p1/consent/withdraw`)).toBe(true));
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('will not start recording until everyone has consented', async () => {
    world(meeting([person('p1', 'Alice Hart', 'granted'), person('p2', 'Petra Voss', 'withdrawn'), person('p3', 'Jon Hale', 'stale')]));
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    expect(screen.getByText('Still needed from: Petra Voss, Jon Hale.')).toBeTruthy();
    expect((screen.getByRole('button', { name: /Start recording/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(FakeSocket.last).toBeNull();
  });

  it('records, pauses, and stops at once when consent is withdrawn', async () => {
    world(meeting([person('p1', 'Alice Hart', 'granted'), person('p2', 'Petra Voss', 'granted')]));
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    fireEvent.click(screen.getByRole('button', { name: /Start recording/ }));
    const sock = FakeSocket.last!;
    expect(sock.url).toMatch(/\/api\/legal\/matters\/id-M-101\/meetings\/mt1\/capture\?sample_rate=16000$/);
    await act(async () => sock.server({ type: 'ready', capture_id: 'c1', consent_snapshot: 'f'.repeat(64), transcription: 'unavailable' }));
    expect(await screen.findByText(/Live transcription is not available on this server/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Pause/ }));
    expect(sock.sent).toContain(JSON.stringify({ type: 'pause' }));
    await act(async () => sock.server({ type: 'state', state: 'paused' }));
    expect(screen.getByText(/Paused — nothing is being captured/)).toBeTruthy();
    await act(async () => sock.server({ type: 'stopped', reason: 'consent_withdrawn' }));
    expect(await screen.findByText(/a participant withdrew consent\. Nothing after that moment was captured\./)).toBeTruthy();
    expect(micStop).toHaveBeenCalled();
  });

  it('explains a refused start', async () => {
    world(meeting([person('p1', 'Alice Hart', 'granted')]));
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    fireEvent.click(screen.getByRole('button', { name: /Start recording/ }));
    await act(async () => FakeSocket.last!.onclose?.({ code: 4412 }));
    expect(await screen.findByText('Everyone must consent before recording can start.')).toBeTruthy();
  });
});
