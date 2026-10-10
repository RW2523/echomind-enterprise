import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { MeetingDetail, MinutesVersion, Participant } from '../api/types';
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
const granted = (id: string, name: string): Participant =>
  ({ id, name, role: 'other', consent: { state: 'granted', method: 'written', at: '2026-10-09T10:00:00.000Z', by: 'alice' } });
const AGENDA = [{ id: 'a1', position: 1, title: 'Harrowgate renewal', documents: [{ id: 'd1', title: 'Harrowgate MSA' }] },
  { id: 'a2', position: 2, title: 'Dividend', documents: [] }];
const meeting = (over: Partial<MeetingDetail> = {}): MeetingDetail => ({
  id: 'mt1', title: 'Board meeting', kind: 'board', scheduled_at: null, status: 'ended', created_at: '2026-10-09T09:00:00.000Z',
  consent: { granted: 2, needed: 2 }, agenda: AGENDA,
  assist: { pack: 'legal_governance', chosen: null, label: 'Board or governance', rules: [] },
  policy: { mode: 'all_party', methods: ['written'], max_age_hours: 12, statement: 'Recorded for M-101.', source: 'P-001' },
  participants: [granted('p1', 'Karin Lindqvist'), granted('p2', 'Petra Voss')], captures: [], ...over,
});
const src = (line: number, quote: string, verification: 'verbatim' | 'failed' = 'verbatim') =>
  ({ segment_id: `s${line}`, line, time: '2026-10-09T10:00:00.000Z', speaker: 'Karin Lindqvist', agenda_item: 'a1', quote, verification });
const V1: MinutesVersion = {
  id: 'v1', version: 1, status: 'draft', content_sha256: 'a'.repeat(64), created_by: 'usr_bob', created_by_name: 'bob',
  created_at: '2026-10-09T11:00:00.000Z', origin: 'model', document_id: null, decisions: [], inputs: { batches: 2, failed_batches: 0 },
  content: {
    meeting: { id: 'mt1', title: 'Board meeting', kind: 'board', date: '2026-10-09' },
    attendees: [{ name: 'Karin Lindqvist', role: 'host' }, { name: 'Petra Voss', role: 'attorney' }],
    summary: { decisions: 1, actions: 1, unresolveds: 0, discussions: 1, not_verified: 1 }, gaps: [],
    sections: [
      { agenda_item: 'a1', title: 'Harrowgate renewal', items: [
        { id: 'm1', kind: 'decision', text: 'The board authorised management to negotiate the renewal.', status: 'verified',
          sources: [src(3, 'the board authorises management to negotiate')] },
        { id: 'm2', kind: 'action', text: 'Petra Voss will send the position by Friday.', owner: 'Petra Voss', deadline: 'Friday', status: 'verified',
          sources: [src(4, 'I will send the board position')] },
        { id: 'm3', kind: 'discussion', text: 'The cap was raised to USD 900,000.', status: 'not_verified', sources: [src(2, 'USD 900,000', 'failed')] },
      ] },
      { agenda_item: 'a2', title: 'Dividend', items: [] },
    ],
  },
};

function world(m: MeetingDetail, versions: MinutesVersion[] = [], extra = {}, user = 'usr_alice') {
  return mockApi({
    'GET /api/auth/me': () => [200, { ...me(['attorney']), user: { ...me(['attorney']).user, id: user } }],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${BASE}`]: () => [200, { meetings: [m] }],
    [`GET ${BASE}/mt1`]: () => [200, { meeting: m }],
    [`GET ${BASE}/mt1/transcript`]: () => [200, { segments: [], marks: [], participants: [], finalised: null }],
    [`GET ${BASE}/mt1/alerts`]: () => [200, { alerts: [], assist: m.assist, telemetry: null }],
    [`GET ${BASE}/mt1/minutes`]: () => [200, { versions }],
    'GET /api/legal/matters/id-M-101/documents': () => [200, { documents: [] }],
    ...extra,
  });
}

describe('Boardroom: agenda and minutes', () => {
  beforeEach(() => { (globalThis as any).WebSocket = FakeSocket; FakeSocket.last = null; });
  afterEach(() => cleanup());

  it('lists the agenda with its board papers, and researches only those papers', async () => {
    world(meeting({ status: 'planned' }));
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Prepare' }));
    const agenda = await screen.findByRole('list', { name: 'Agenda' });
    expect(within(agenda).getByText('Harrowgate MSA')).toBeTruthy();
    expect(within(agenda).getByRole('link', { name: /Research these papers/ }).getAttribute('href'))
      .toBe('/m/M-101/research?docs=d1&about=Harrowgate%20renewal');
  });

  it('research from an agenda item reads only its papers, and can be widened', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/matters/id-M-101/research': () => [200, { turns: [] }],
      'POST /api/legal/matters/id-M-101/research/query': () => [200, { __ndjson: [{ type: 'sources', citations: [] },
        { type: 'done', answer: 'x', citations: [], abstained: true, abstention_reason: 'no_passages' }] }],
    });
    renderLaw('/m/M-101/research?docs=d1&about=Harrowgate%20renewal');
    expect((await screen.findByRole('status')).textContent).toMatch(/Only the board papers for “Harrowgate renewal” \(1 document\)/);
    fireEvent.change(screen.getByLabelText('Ask about this matter'), { target: { value: 'What is the cap?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ question: 'What is the cap?', versions: 'signed', documents: ['d1'] }));
    fireEvent.click(screen.getByRole('button', { name: 'Ask the whole matter instead' }));
    await waitFor(() => expect(screen.queryByText(/Only the board papers/)).toBeNull());
  });

  it('marks the agenda item while recording', async () => {
    world(meeting({ status: 'planned' }));
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Start recording' }));
    await act(async () => { FakeSocket.last!.server({ type: 'ready', capture_id: 'c1', consent_snapshot: 'x', transcription: 'available' }); });
    const items = await screen.findByRole('group', { name: 'Agenda item' });
    fireEvent.click(within(items).getByRole('radio', { name: '2. Dividend' }));
    expect(FakeSocket.last!.sent).toContain(JSON.stringify({ type: 'agenda', item_id: 'a2' }));
  });

  it('drafts minutes from the transcript, each item with its words and flags, ready for review', async () => {
    let versions: MinutesVersion[] = [];
    const calls = world(meeting(), versions, { [`POST ${BASE}/mt1/minutes/draft`]: () => { versions = [V1]; return [200, { minutes: V1 }]; },
      [`GET ${BASE}/mt1/minutes`]: () => [200, { versions }] });
    const { container } = renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Minutes' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Draft minutes' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/mt1/minutes/draft`)).toBe(true));
    const renewal = await screen.findByRole('region', { name: 'Harrowgate renewal' });
    const decision = within(renewal).getByRole('listitem', { name: 'Decided: The board authorised management to negotiate the renewal.' });
    expect(decision.textContent).toMatch(/Karin Lindqvist/);
    const action = within(renewal).getByRole('listitem', { name: /^Action: Petra Voss will send/ });
    expect(action.textContent).toContain('Owner: Petra Voss · By: Friday');
    const flagged = within(renewal).getByRole('listitem', { name: 'Discussed: The cap was raised to USD 900,000.' });
    expect(flagged.textContent).toContain('Not verified against the transcript');
    expect(within(screen.getByRole('region', { name: 'Dividend' })).getByText('Nothing minuted.')).toBeTruthy();
    expect(screen.getByText('Draft — not approved')).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a second lawyer approves exactly the version they read; the author cannot', async () => {
    const calls = world(meeting(), [V1], { [`POST ${BASE}/mt1/minutes/v1/review`]: () => [200, { minutes: { ...V1, status: 'approved' } }] });
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Minutes' }));
    const review = await screen.findByRole('group', { name: 'Review these minutes' });
    expect((within(review).getByRole('button', { name: 'Request changes' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(within(review).getByRole('button', { name: 'Approve these minutes' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/mt1/minutes/v1/review`)?.body)
      .toEqual({ decision: 'approved', content_sha256: V1.content_sha256, reason: null }));
    cleanup();
    world(meeting(), [V1], {}, 'usr_bob');
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Minutes' }));
    expect(await screen.findByText('You made this version, so another lawyer approves it.')).toBeTruthy();
  });

  it('files approved minutes as a matter document', async () => {
    const approved = { ...V1, status: 'approved' as const, decisions: [{ decision: 'approved' as const, content_sha256: V1.content_sha256, by: 'dave', at: '2026-10-09T12:00:00.000Z' }] };
    const calls = world(meeting(), [approved], { [`POST ${BASE}/mt1/minutes/v1/file`]: () => [200, { document_id: 'd9', changed: true }] });
    renderLaw('/m/M-101/meetings/mt1');
    fireEvent.click(await screen.findByRole('tab', { name: 'Minutes' }));
    expect(await screen.findByText('Approved')).toBeTruthy();
    expect(screen.getByRole('list', { name: 'Review decisions' }).textContent).toMatch(/Approved by dave.*content aaaaaaaaaaaa…/);
    fireEvent.click(screen.getByRole('button', { name: 'File as a matter document' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/mt1/minutes/v1/file`)).toBe(true));
  });
});
