import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Playbook, PlaybookCheck } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.2 — playbooks and playbook checks (D-046). */
const SHA = 'd'.repeat(64);
const PB = (status: 'draft' | 'approved' = 'approved', by = 'alice'): Playbook => ({
  id: 'p1', name: 'Commercial services (supplier)', description: null, jurisdiction: 'Sweden', created_by: 'alice', created_at: '2026-10-10T08:00:00.000Z',
  positions: [{ id: 'pos1', topic: 'liability_cap', topic_label: 'Limitation of liability', version: 1, status, title: 'Cap on liability',
    standard: 'Liability shall not exceed USD 250,000.', fallbacks: [{ text: 'Liability shall not exceed the fees paid in twelve months.', note: 'Partner sign-off' }],
    guidance: 'Never uncapped.', content_sha256: SHA, created_by: by, created_at: '2026-10-10T08:00:00.000Z', approved_by: status === 'approved' ? 'mp' : null,
    approved_at: status === 'approved' ? '2026-10-10T09:00:00.000Z' : null, owner_name: 'alice' }],
});
const TOPICS = { liability_cap: 'Limitation of liability', payment: 'Payment' };
const M = '/api/legal/matters/id-M-101';
const CHECK: PlaybookCheck = {
  id: 'c1', document_id: 'd1', version_id: 'v1', playbook: { id: 'p1', name: 'Commercial services (supplier)' }, advisory: true,
  created_by: 'alice', created_at: '2026-10-10T10:00:00.000Z', inputs: { version_number: 1 },
  summary: { standard: 0, fallback: 0, outside: 1, not_found: 1, unclear: 0 },
  results: [
    { topic: 'liability_cap', title: 'Cap on liability', topic_label: 'Limitation of liability', classification: 'outside', matched: null, method: 'rule',
      position: { id: 'pos1', version: 1, standard: 'Liability shall not exceed USD 250,000.', fallbacks: [] },
      clause: { number: '8.1', heading: 'Liability', start: 40, end: 120 }, quote: { text: 'shall not exceed USD 500,000', start: 60, end: 88, verification: 'verbatim' },
      reason: 'Outside the playbook: the clause says USD 500,000; the position says USD 250,000.', decision: null },
    { topic: 'payment', title: 'Payment terms', topic_label: 'Payment', classification: 'not_found', matched: null, method: 'model',
      position: { id: 'pos2', version: 1, standard: 'Thirty days.', fallbacks: [] }, clause: null, quote: null, reason: 'No clause of this document deals with the topic.', decision: null },
  ],
};
const AXE = { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } };

function world(roles: string[], extra = {}, myRole: 'lead' | 'attorney' = 'lead') {
  return mockApi({
    'GET /api/auth/me': () => [200, me(roles)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: myRole })] }],
    ...extra,
  });
}

describe('Playbooks (S3.2)', () => {
  afterEach(() => cleanup());

  it('shows each position with its fallbacks, and a managing partner approves a draft after confirming it', async () => {
    const calls = world(['managing_partner'], {
      'GET /api/legal/playbooks': () => [200, { playbooks: [PB('draft', 'bob')], topics: TOPICS, can: { manage: true, approve: true } }],
      'POST /api/legal/playbooks/p1/positions/pos1/approve': () => [200, { playbook: PB() }],
    });
    const { container } = renderLaw('/playbooks');
    const row = await screen.findByRole('listitem', { name: 'Limitation of liability: Cap on liability' });
    expect(row.textContent).toMatch(/Standard.*USD 250,000.*Fallback 1.*fees paid in twelve months.*— Partner sign-off.*Guidance.*Never uncapped/);
    expect(within(row).getByText('Version 1 waiting for approval')).toBeTruthy();
    fireEvent.click(within(row).getByRole('button', { name: 'Approve version 1' }));
    const dialog = await screen.findByRole('dialog', { name: 'Approve Cap on liability, version 1?' });
    expect(dialog.textContent).toContain(SHA);
    fireEvent.click(within(dialog).getByLabelText('I have reviewed this position and its fallbacks'));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Approve version 1' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/approve'))?.body).toEqual({ content_sha256: SHA }));
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });

  it('a precedent is read into proposals, and one becomes a position to approve', async () => {
    const calls = world(['attorney'], {
      'GET /api/legal/playbooks': () => [200, { playbooks: [PB()], topics: TOPICS, can: { manage: true, approve: false } }],
      'POST /api/legal/playbooks/p1/import': () => [200, { proposals: [{ clause: '1.1', title: 'Liability', text: '1.1 Liability shall not exceed USD 300,000.', topic: 'liability_cap' }] }],
      'POST /api/legal/playbooks/p1/positions': () => [200, { playbook: PB() }],
    });
    renderLaw('/playbooks');
    fireEvent.click(await screen.findByRole('button', { name: 'Read a precedent' }));
    const dialog = await screen.findByRole('dialog', { name: 'Read a precedent into Commercial services (supplier)' });
    fireEvent.change(within(dialog).getByLabelText('Precedent (Word, PDF or text)'), { target: { files: [new File(['x'], 'precedent.docx')] } });
    const proposal = await screen.findByRole('list', { name: 'Proposed positions' });
    fireEvent.click(within(proposal).getByRole('button', { name: 'Use as the standard position' }));
    const editor = await screen.findByRole('dialog', { name: 'Position in Commercial services (supplier)' });
    expect((within(editor).getByLabelText('Standard position') as HTMLTextAreaElement).value).toBe('Liability shall not exceed USD 300,000.');
    fireEvent.click(within(editor).getByRole('button', { name: 'Save as a draft version' }));
    await waitFor(() => expect(calls.find((c) => c.path === '/api/legal/playbooks/p1/positions')?.body).toMatchObject({ topic: 'liability_cap', title: 'Liability' }));
  });

  it('a document check shows each topic with its class, the words it was judged on, and asks a lawyer to decide', async () => {
    const calls = world(['attorney'], {
      'GET /api/legal/playbooks': () => [200, { playbooks: [PB()], topics: TOPICS, can: { manage: false, approve: false } }],
      [`GET ${M}/documents/d1`]: () => [200, { document: { id: 'd1', title: 'Services agreement' } }],
      [`GET ${M}/documents/d1/playbook-checks`]: () => [200, { checks: [CHECK] }],
      [`POST ${M}/documents/d1/playbook-check`]: () => [200, { check: CHECK }],
      [`POST ${M}/playbook-checks/c1/decisions`]: () => [200, { check: CHECK }],
    });
    const { container } = renderLaw('/m/M-101/documents/d1/playbook');
    const check = await screen.findByRole('article', { name: 'Playbook check' });
    expect(check.textContent).toContain('Advisory — a lawyer decides');
    const cap = within(check).getByRole('listitem', { name: 'Cap on liability: Outside the playbook' });
    expect(cap.textContent).toMatch(/Decided by the figures.*Clause 8\.1 · Liability.*“shall not exceed USD 500,000”/);
    expect(within(cap).getByRole('link', { name: 'Open at the quote' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=60-88');
    expect(within(check).getByRole('listitem', { name: 'Payment terms: Not found' })).toBeTruthy();
    const decide = within(cap).getByRole('group', { name: 'Decide on Cap on liability' });
    expect((within(decide).getByRole('button', { name: 'Disagree' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(within(decide).getByLabelText('Why you disagree on Cap on liability'), { target: { value: 'Agreed by the partner.' } });
    fireEvent.click(within(decide).getByRole('button', { name: 'Disagree' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/decisions'))?.body).toEqual({ topic: 'liability_cap', decision: 'disagree', reason: 'Agreed by the partner.' }));
    fireEvent.change(screen.getByLabelText('Playbook'), { target: { value: 'p1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Check this document' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/playbook-check'))?.body).toEqual({ playbook_id: 'p1' }));
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });
});
