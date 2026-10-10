import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, SummaryContent, SummaryPoint, SummaryVersion } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S4.2 — summaries with page/line and timecode citations (OPT-02; D-055). */
const M = '/api/legal/matters/id-M-101';
const point = (x: Partial<SummaryPoint> & Pick<SummaryPoint, 'id' | 'text'>): SummaryPoint => ({
  quote: 'The Customer shall pay each invoice within thirty (30) days of its date.', verification: 'verbatim', start: 70, end: 140,
  cite: { clause: '5.2', line: 4 }, figures_unsupported: [], origin: 'model', ...x,
});
const CONTENT: SummaryContent = {
  title: 'Summary — Supply Agreement', document: { id: 'd1', title: 'Supply Agreement.txt', version_id: 'v1', version_number: 1 }, transcript: false,
  points: [
    point({ id: 'p1', text: 'The Customer pays each invoice within 30 days of its date.' }),
    point({ id: 'p2', text: 'The Customer pays within 45 days.', figures_unsupported: ['45'] }),
    point({ id: 'p3', text: 'Late payment carries interest.', quote: 'Interest accrues at 8%.', verification: 'failed', start: null, end: null, cite: null }),
  ],
  sections: 1, unverified: 1, figures_unsupported: 1, translation: { available: false, reason: 'No local translation model is approved yet.' },
};
const version = (x: Partial<SummaryVersion> = {}): SummaryVersion => ({
  id: 's1', version: 1, status: 'draft', content: CONTENT, content_sha256: 'c'.repeat(64), inputs: {}, created_by: 'usr_alice', created_by_name: 'alice',
  created_at: '2026-10-10T10:00:00.000Z', origin: 'model', document_id: null, decisions: [], kind: 'summary', title: CONTENT.title, subject_id: 'd1', ...x,
});
const withSummaries = (roles: string[] = ['attorney']): Me => ({ ...me(roles), features: { intake: false, intake_links: false, summaries: true } });

function world(v: SummaryVersion = version(), extra = {}, can = { approve: true, file: true, author: false }) {
  return mockApi({
    'GET /api/auth/me': () => [200, withSummaries()],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${M}/artifacts/${v.id}`]: () => [200, { artifact: v, versions: [{ id: v.id, version: v.version, status: v.status, created_by_name: 'alice', created_at: v.created_at, origin: v.origin }], can }],
    ...extra,
  });
}

describe('Summaries (S4.2)', () => {
  afterEach(() => cleanup());

  it('each point shows where its words are, links to them, and says what to check', async () => {
    const { container } = (world(), renderLaw('/m/M-101/summaries/s1'));
    const list = await screen.findByRole('list', { name: 'Key points' });
    const first = within(list).getByRole('listitem', { name: 'The Customer pays each invoice within 30 days of its date. (clause 5.2, l. 4)' });
    expect(within(first).getByRole('link', { name: 'clause 5.2, l. 4' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=70-140');
    expect(within(list).getByRole('listitem', { name: /45 days/ }).textContent).toContain('Check: 45 not in the quote');
    expect(within(list).getByRole('listitem', { name: /Late payment/ }).textContent).toContain('Not verified');
    expect(screen.getByText(/the original is unchanged/)).toBeTruthy();
    expect(screen.getByText(/Translation: not available/)).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a transcript point cites its time', async () => {
    world(version({ content: { ...CONTENT, transcript: true, points: [point({ id: 'p1', text: 'The client can accept EUR 40,000.', cite: { timecode: '10:03:40' } })] } }));
    renderLaw('/m/M-101/summaries/s1');
    expect(within(await screen.findByRole('list', { name: 'Key points' })).getByRole('link', { name: '10:03:40' })).toBeTruthy();
  });

  it('a reviewer rewords, reorders and removes points as a new version', async () => {
    const v2 = version({ id: 's2', version: 2, origin: 'person' });
    const calls = world(version(), {
      [`POST ${M}/summaries/s1/edit`]: () => [200, { summary: v2 }],
      [`GET ${M}/artifacts/s2`]: () => [200, { artifact: v2, versions: [], can: { approve: true, file: true, author: true } }],
    });
    renderLaw('/m/M-101/summaries/s1');
    fireEvent.click(await screen.findByRole('button', { name: 'Edit points' }));
    const edit = screen.getByRole('list', { name: 'Edit the points' });
    fireEvent.change(within(edit).getByLabelText('Point 1'), { target: { value: 'Invoices are payable within 30 days of their date.' } });
    fireEvent.click(within(edit).getByRole('button', { name: 'Remove point 3' }));
    fireEvent.click(within(edit).getByRole('button', { name: 'Move point 2 up' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save as a new version' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${M}/summaries/s1/edit`)).toBe(true));
    expect(calls.find((c) => c.path === `${M}/summaries/s1/edit`)!.body).toEqual({ points: [
      { id: 'p2', text: 'The Customer pays within 45 days.' }, { id: 'p1', text: 'Invoices are payable within 30 days of their date.' }] });
    await waitFor(() => expect(calls.some((c) => c.path === `${M}/artifacts/s2`)).toBe(true));
  });

  it('a matter\'s summaries, and summarising a document', async () => {
    let started = false;
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withSummaries()],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/summaries`]: () => [200, { summaries: [version()], can: { summarise: true } }],
      [`GET ${M}/documents`]: () => [200, { documents: [{ id: 'd2', title: 'Lease.txt', current_version: { id: 'v9', number: 1, index_status: 'indexed' } }] }],
      [`POST ${M}/documents/d2/summaries`]: () => { started = true; return [200, { job: { status: 'queued' } }]; },
      [`GET ${M}/documents/d2/summaries`]: () => [200, { summaries: [], job: { status: started ? 'running' : 'queued', error: null }, can: { summarise: true } }],
    });
    renderLaw('/m/M-101/summaries');
    const row = await screen.findByRole('listitem', { name: 'Summary — Supply Agreement' });
    expect(row.textContent).toContain('3 points');
    expect(row.textContent).toContain('1 not verified');
    fireEvent.change(screen.getByLabelText('Document'), { target: { value: 'd2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Summarise' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === `${M}/documents/d2/summaries`)).toBe(true));
    expect(await screen.findByRole('status')).toBeTruthy();
  });

  it('the matter navigation shows Summaries only with the feature on', async () => {
    world();
    renderLaw('/m/M-101/overview');
    expect(await screen.findByRole('link', { name: 'Summaries' })).toBeTruthy();
    cleanup();
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/parties`]: () => [200, { parties: [], can_manage: true }] });
    renderLaw('/m/M-101/overview');
    await screen.findByRole('navigation', { name: 'Matter' });
    expect(screen.queryByRole('link', { name: 'Summaries' })).toBeNull();
  });
});
