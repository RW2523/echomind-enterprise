import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { ArtifactView, PlaybookCheck, RedlineContent } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.3 — redlines to the firm's positions, as Word tracked changes (D-047). */
const M = '/api/legal/matters/id-M-101';
const SHA = 'e'.repeat(64);
const CHECK: PlaybookCheck = {
  id: 'c1', document_id: 'd1', version_id: 'v1', playbook: { id: 'p1', name: 'Commercial services' }, advisory: true, created_by: 'alice',
  created_at: '2026-10-10T10:00:00.000Z', inputs: { version_number: 1 }, summary: { standard: 0, fallback: 0, outside: 1, not_found: 0, unclear: 0 },
  results: [{ topic: 'payment', title: 'Payment terms', topic_label: 'Payment', classification: 'outside', matched: null, method: 'rule',
    position: { id: 'pos2', version: 1, standard: 'Invoices are payable within thirty (30) days.', fallbacks: [{ text: 'Within forty-five (45) days.', note: 'Strategic customers' }] },
    clause: { number: '9.1', heading: 'Payment', start: 10, end: 80 }, quote: { text: 'sixty (60) days', start: 50, end: 65, verification: 'verbatim' },
    reason: 'Outside the playbook: the clause says sixty (60) days; the position says thirty (30) days.', decision: null }],
};
const CONTENT: RedlineContent = {
  title: 'Redline — services-agreement', check_id: 'c1', playbook: { id: 'p1', name: 'Commercial services' }, summary: { changes: 1, fitted: 0 },
  document: { id: 'd1', title: 'services-agreement', version_id: 'v1', version_number: 1, filename: 'services-agreement.txt', mime: 'text/plain' },
  changes: [{ id: 'c1', topic: 'payment', title: 'Payment terms', clause: { number: '9.1', heading: 'Payment', start: 10, end: 80 },
    original: '9.1 The Customer shall pay within sixty (60) days.', revised: '9.1 Invoices are payable within thirty (30) days.',
    ops: [['eq', '9.1 '], ['del', 'The Customer shall pay'], ['ins', 'Invoices are payable'], ['eq', ' within '], ['del', 'sixty (60)'], ['ins', 'thirty (30)'], ['eq', ' days.']],
    reason: 'Sixty days is too long.', comment: 'Payment terms: revised.', fitted: false, fit_note: null,
    from: { kind: 'standard', n: null, position_id: 'pos2', position_version: 1, text: 'Invoices are payable within thirty (30) days.' } }],
};
const view = (status: 'draft' | 'approved' = 'draft', can = { approve: true, file: true, author: false }): ArtifactView<RedlineContent> => ({
  artifact: { id: 'r1', version: 1, status, content: CONTENT, content_sha256: SHA, inputs: {}, created_by: 'usr_alice', created_by_name: 'alice',
    created_at: '2026-10-10T10:05:00.000Z', origin: 'model', document_id: null, kind: 'redline', title: CONTENT.title,
    decisions: status === 'approved' ? [{ decision: 'approved', content_sha256: SHA, by: 'dave', at: '2026-10-10T11:00:00.000Z' }] : [] },
  versions: [{ id: 'r1', version: 1, status, created_by_name: 'alice', created_at: '2026-10-10T10:05:00.000Z', origin: 'model' }], can,
});

function world(extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'lead' })] }],
    ...extra,
  });
}

describe('Redlines (S3.3)', () => {
  afterEach(() => cleanup());

  it('the lines outside the playbook are revised, with the lawyer choosing a fallback', async () => {
    const calls = world({
      'GET /api/legal/playbooks': () => [200, { playbooks: [], topics: {}, can: { manage: false, approve: false } }],
      [`GET ${M}/documents/d1`]: () => [200, { document: { id: 'd1', title: 'services-agreement' } }],
      [`GET ${M}/documents/d1/playbook-checks`]: () => [200, { checks: [CHECK] }],
      [`POST ${M}/playbook-checks/c1/redline`]: () => [200, { redline: { id: 'r1' } }],
      [`GET ${M}/artifacts/r1`]: () => [200, view()],
    });
    renderLaw('/m/M-101/documents/d1/playbook');
    const revise = await screen.findByRole('region', { name: 'Revise to the playbook' });
    fireEvent.change(within(revise).getByLabelText('Position for Payment terms'), { target: { value: 'fallback 1' } });
    fireEvent.click(within(revise).getByRole('button', { name: 'Propose 1 revision' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/redline'))?.body).toEqual({ choices: { payment: 'fallback 1' } }));
    expect(await screen.findByRole('article', { name: 'Redline' })).toBeTruthy();
  });

  it('a redline shows what is struck and added, why, and offers the Word file with tracked changes', async () => {
    const calls = world({ [`GET ${M}/artifacts/r1`]: () => [200, view()], [`POST ${M}/redlines/r1/edit`]: () => [200, { redline: { id: 'r2' } }] });
    const { container } = renderLaw('/m/M-101/redlines/r1');
    const red = await screen.findByRole('article', { name: 'Redline' });
    const change = within(red).getByRole('listitem', { name: 'Payment terms, clause 9.1' });
    expect(Array.from(change.querySelectorAll('del')).map((d) => d.textContent)).toEqual(['The Customer shall pay', 'sixty (60)']);
    expect(Array.from(change.querySelectorAll('ins')).map((d) => d.textContent)).toEqual(['Invoices are payable', 'thirty (30)']);
    expect(change.textContent).toContain('Why: Sixty days is too long.');
    expect(change.textContent).toContain("to the firm's standard (position version 1)");
    expect(screen.getByRole('link', { name: 'Word, with tracked changes' }).getAttribute('href')).toBe(`${M}/redlines/r1/word`);
    expect(screen.getByText(/Not yet verified by a person in Microsoft Word/)).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
    fireEvent.click(screen.getByRole('button', { name: 'Edit the revisions' }));
    const form = screen.getByRole('form', { name: 'Edit the revisions' });
    fireEvent.click(within(form).getByRole('button', { name: 'Drop this change' }));
    fireEvent.click(within(form).getByRole('button', { name: 'Save as version 2' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/edit'))?.body).toEqual({ content_sha256: SHA, changes: [{ id: 'c1', remove: true }], reason: null }));
  });

  it('an approved redline is filed as the document\'s next version', async () => {
    const calls = world({ [`GET ${M}/artifacts/r1`]: () => [200, view('approved')], [`POST ${M}/artifacts/r1/file`]: () => [200, { document_id: 'd1', changed: true }] });
    renderLaw('/m/M-101/redlines/r1');
    expect((await screen.findByRole('article', { name: 'Redline' })).textContent).toMatch(/Approved by dave/);
    fireEvent.click(screen.getByRole('button', { name: 'File as the next version' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${M}/artifacts/r1/file`)).toBe(true));
  });
});
