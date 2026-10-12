import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { MeetingDetail, Participant, TranscriptData } from '../api/types';
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
const granted = (id: string, name: string, role: Participant['role'] = 'client'): Participant =>
  ({ id, name, role, consent: { state: 'granted', method: 'verbal_on_record', at: '2026-10-09T10:00:00.000Z', by: 'alice' } });
const PEOPLE = [granted('p1', 'Helena Marsh', 'attorney'), granted('p2', 'Petra Voss')];
const meeting = (over: Partial<MeetingDetail> = {}): MeetingDetail => ({
  id: 'mt1', title: 'Renewal call', kind: 'client_call', scheduled_at: null, status: 'ended', created_at: '2026-10-09T09:00:00.000Z',
  consent: { granted: 2, needed: 2 },
  policy: { mode: 'all_party', methods: ['verbal_on_record', 'written', 'host_attestation'], max_age_hours: 12, statement: 'Recorded for M-101.', source: 'P-001' },
  participants: PEOPLE, captures: [], ...over,
});
const DATA: TranscriptData = {
  participants: [{ id: 'p1', name: 'Helena Marsh', role: 'attorney', removed: false }, { id: 'p2', name: 'Petra Voss', role: 'client', removed: false }],
  marks: [{ kind: 'gap', at: '2026-10-09T10:00:05.000Z', capture_id: 'c1', detail: 'transcription_error' }],
  finalised: null,
  segments: [
    { id: 's1', seq: 0, text: 'Can you confirm the cap?', raw_text: 'Can you confirm the cap?', speaker: 'Helena Marsh', speaker_participant_id: 'p1',
      wall_start: '2026-10-09T10:00:01.000Z', revised: false, revisions: [] },
    { id: 's2', seq: 1, text: 'Harrowgate Systems will pay USD 19,750.', raw_text: 'harrow gate systems will pay USD 19,570.', speaker: 'Petra Voss',
      speaker_participant_id: 'p2', wall_start: '2026-10-09T10:00:09.000Z', revised: true, revisions: [
        { id: 'r1', kind: 'text', text: 'Harrowgate Systems will pay USD 19,750.', reason: 'Checked against the redline', source: 'person', provenance: {},
          by: 'alice', at: '2026-10-09T11:00:00.000Z' }] },
  ],
};

function world(m: MeetingDetail, data: TranscriptData, extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${BASE}`]: () => [200, { meetings: [m] }],
    [`GET ${BASE}/mt1`]: () => [200, { meeting: m }],
    [`GET ${BASE}/mt1/transcript`]: () => [200, data],
    [`GET ${BASE}/mt1/transcript/suggestions`]: () => [200, { suggestions: [] }],
    ...extra,
  });
}

async function openRecordTab() {
  fireEvent.click(await screen.findByRole('tab', { name: 'Record' }));
}

describe('Transcripts', () => {
  beforeEach(() => { (globalThis as any).WebSocket = FakeSocket; FakeSocket.last = null; });
  afterEach(() => cleanup());

  it('shows each segment with its time and speaker, corrections with their history, and gaps', async () => {
    world(meeting(), DATA);
    const { container } = renderLaw('/m/M-101/meetings/mt1');
    await openRecordTab();
    const rows = await screen.findByRole('list', { name: 'Transcript segments' });
    const items = Array.from(rows.children) as HTMLElement[];          // the segments and gaps (not their history lists)
    expect(items).toHaveLength(3);
    expect(items[0].textContent).toMatch(/Helena Marsh.*Can you confirm the cap\?/);
    expect(items[1].textContent).toMatch(/Not transcribed: a transcription error at this point\./);
    expect(items[2].textContent).toMatch(/Petra Voss.*Harrowgate Systems will pay USD 19,750\./);
    expect(within(items[2]).getByText('Corrected once — as heard: “harrow gate systems will pay USD 19,570.”')).toBeTruthy();
    expect(within(items[2]).getByText(/Checked against the redline/)).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('says why a gap was left: speech recognition fell too far behind (D-066)', async () => {
    world(meeting(), { ...DATA, marks: [{ kind: 'gap', at: '2026-10-09T10:00:05.000Z', capture_id: 'c1', detail: 'transcription_behind' }] });
    renderLaw('/m/M-101/meetings/mt1');
    await openRecordTab();
    const rows = await screen.findByRole('list', { name: 'Transcript segments' });
    expect(rows.children[1].textContent).toMatch(/Not transcribed: speech recognition fell too far behind here\./);
  });

  it('a correction needs a reason, and keeps the original', async () => {
    const calls = world(meeting(), DATA, { [`POST ${BASE}/mt1/transcript/segments/s1/revisions`]: () => [200, { revisions: [{ id: 'r9', kind: 'speaker' }] }] });
    renderLaw('/m/M-101/meetings/mt1');
    await openRecordTab();
    fireEvent.click((await screen.findAllByRole('button', { name: /Correct the segment at/ }))[0]);
    const dialog = await screen.findByRole('dialog', { name: 'Correct this segment' });
    expect(within(dialog).getByText(/The original stays in the record/)).toBeTruthy();
    fireEvent.change(within(dialog).getByLabelText('Speaker'), { target: { value: 'p2' } });
    const save = within(dialog).getByRole('button', { name: 'Save correction' }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(within(dialog).getByLabelText('Reason'), { target: { value: 'Petra asked this' } });
    expect(save.disabled).toBe(false);
    fireEvent.click(save);
    await waitFor(() => expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ speaker_participant_id: 'p2', reason: 'Petra asked this' }));
  });

  it('offers the matter’s vocabulary with its source, and adds the transcript to research', async () => {
    const calls = world(meeting(), DATA, {
      [`GET ${BASE}/mt1/transcript/suggestions`]: () => [200, { suggestions: [{
        id: 'x1', segment_id: 's1', start: 0, end: 3, heard: 'harrow gate systems', suggested: 'Harrowgate Systems', kind: 'company', score: 0.9,
        text: 'Harrowgate Systems …', source: { kind: 'document', document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v1', version_number: 1 } }] }],
      [`POST ${BASE}/mt1/transcript/segments/s1/revisions`]: () => [200, { revisions: [{ id: 'r9', kind: 'text' }] }],
      [`POST ${BASE}/mt1/transcript/finalise`]: () => [200, { document_id: 'd9', version_number: 1, changed: true }],
      ['GET /api/legal/matters/id-M-101/documents']: () => [200, { documents: [] }],
    });
    renderLaw('/m/M-101/meetings/mt1');
    await openRecordTab();
    const vocab = await screen.findByRole('region', { name: 'Matter vocabulary' });
    expect(within(vocab).getByText('A company in the documents · Harrowgate MSA, v1')).toBeTruthy();
    fireEvent.click(within(vocab).getByRole('button', { name: 'Accept Harrowgate Systems for harrow gate systems' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/revisions'))?.body)
      .toEqual({ suggestion_id: 'x1', reason: 'Matter vocabulary: Harrowgate Systems (Harrowgate MSA)' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add to research' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/mt1/transcript/finalise`)).toBe(true));
  });

  it('marks who is speaking while recording, and says when transcription falls behind', async () => {
    world(meeting({ status: 'planned' }), { ...DATA, segments: [], marks: [] });
    renderLaw('/m/M-101/meetings/mt1');
    await openRecordTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Start recording' }));
    await act(async () => { FakeSocket.last!.server({ type: 'ready', capture_id: 'c1', consent_snapshot: 'abc', transcription: 'available' }); });
    const speaking = await screen.findByRole('group', { name: 'Speaking now' });
    fireEvent.click(within(speaking).getByRole('radio', { name: 'Petra Voss' }));
    expect(FakeSocket.last!.sent).toContain(JSON.stringify({ type: 'speaker', participant_id: 'p2' }));
    await act(async () => { FakeSocket.last!.server({ type: 'speaker', participant_id: 'p2' }); });
    expect((within(speaking).getByRole('radio', { name: 'Petra Voss' }) as HTMLInputElement).checked).toBe(true);
    fireEvent.keyDown(window, { altKey: true, code: 'Digit1' });
    expect(FakeSocket.last!.sent).toContain(JSON.stringify({ type: 'speaker', participant_id: 'p1' }));
    await act(async () => { FakeSocket.last!.server({ type: 'degraded', reason: 'behind', lag_s: 6.2 }); });
    expect(screen.getByRole('alert').textContent).toMatch(/Transcription is 6.2 s behind/);
    await act(async () => { FakeSocket.last!.server({ type: 'recovered' }); });
    expect(screen.queryByText(/Transcription is 6.2 s behind/)).toBeNull();
    await act(async () => { FakeSocket.last!.server({ type: 'degraded', reason: 'skipped' }); });      // D-066: too far behind
    expect(screen.getByRole('alert').textContent).toMatch(/some speech is not being transcribed\. The gap is marked/);
    await act(async () => { FakeSocket.last!.server({ type: 'recovered' }); });
    await act(async () => { FakeSocket.last!.server({ type: 'segment', id: 'n1', seq: 0, text: 'Hello.', speaker_participant_id: 'p2', wall_start: '2026-10-10T10:00:00Z' }); });
    expect(within(await screen.findByRole('list', { name: 'Transcript segments' })).getByText('Petra Voss')).toBeTruthy();
  });
});
