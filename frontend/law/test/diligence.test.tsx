import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { DdChecklist, DdIssue, DdRunView } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.7 — due diligence (DOC-06; D-051). */
const M = '/api/legal/matters/id-M-101';
const LIST: DdChecklist = {
  id: 'c1', family_id: 'f1', version: 1, name: 'Share acquisition — core documents', expected_law: 'England and Wales', content_sha256: 'a'.repeat(64),
  status: 'approved', created_by: 'alice', created_at: '2026-10-10T09:00:00.000Z', approved_by: 'mp', approved_at: '2026-10-10T09:30:00.000Z',
  items: [{ id: 'contracts', section: 'Commercial', title: 'Material contracts', expect: ['supply agreement'], questions: ['change_of_control', 'renewal'] },
    { id: 'insurance', section: 'Insurance', title: 'Insurance policies', expect: ['insurance policy'], questions: [] }],
};
const issue = (x: Partial<DdIssue> & Pick<DdIssue, 'id' | 'kind' | 'severity' | 'title'>): DdIssue => ({
  run_id: 'r1', item_id: 'contracts', document_id: 'd1', document_title: 'Supply Agreement - Northwind.txt', version_id: 'v1', detail: null,
  evidence: { start: 300, end: 400, quote: '2.1 The Supplier may terminate this Agreement on written notice if the Customer undergoes a change of control.', clause: '2.1' },
  status: 'preliminary', reviewer: null, decided_at: null, version: 1, created_at: '2026-10-10T10:00:00.000Z', ...x,
});
const VIEW: DdRunView = {
  run: { id: 'r1', status: 'done', error: null, checklist: { id: 'c1', name: LIST.name, version: 1 }, created_by: 'alice', created_at: '2026-10-10T10:00:00.000Z',
    finished_at: '2026-10-10T10:00:05.000Z', documents_read: 4 },
  items: [{ item_id: 'contracts', title: 'Material contracts', section: 'Commercial', documents: [{ id: 'd1', title: 'Supply Agreement - Northwind.txt' }], missing: false },
    { item_id: 'insurance', title: 'Insurance policies', section: 'Insurance', documents: [], missing: true }],
  missing: [{ item_id: 'insurance', title: 'Insurance policies', section: 'Insurance' }],
  issues: [issue({ id: 'i1', kind: 'change_of_control', severity: 'high', title: 'Change of control (clause 2.1)', detail: 'A change of control lets the other party end the agreement.' }),
    issue({ id: 'i2', kind: 'renewal', severity: 'high', title: 'Automatic renewal (clause 1.2)', detail: 'The notice deadline (1 October 2026) has passed.', status: 'verified', reviewer: 'dave' })],
  summary: { items: 2, missing: 1, issues: 2, high: 2, medium: 0, low: 0, preliminary: 1, verified: 1, dismissed: 0 },
};

function world(role: 'lead' | 'read_only' = 'lead', extra = {}, firm: string[] = ['attorney']) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(firm)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role })] }],
    'GET /api/legal/diligence/checklists': () => [200, { checklists: [LIST], can: { manage: false, approve: false }, questions: [] }],
    [`GET ${M}/diligence/runs`]: () => [200, { runs: [{ ...VIEW.run, summary: VIEW.summary }] }],
    [`GET ${M}/diligence/runs/r1`]: () => [200, VIEW],
    ...extra,
  });
}

describe('Due diligence (S3.7)', () => {
  afterEach(() => cleanup());

  it('the register: issues with their words and status, what is missing, and what answers each item', async () => {
    const { container } = (world(), renderLaw('/m/M-101/diligence'));
    const reg = await screen.findByRole('article', { name: 'Issue register' });
    expect(within(reg).getByRole('status').textContent).toBe('1 of 2 items answered · 1 missing · 2 issues (2 high, 0 medium, 0 low) · 1 verified · 0 dismissed · 1 awaiting review');
    const coc = within(reg).getByRole('listitem', { name: 'Change of control (clause 2.1): Preliminary — not reviewed' });
    expect(coc.textContent).toContain('undergoes a change of control');
    expect(within(coc).getByRole('link', { name: 'Open the words' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=300-400');
    expect(within(reg).getByRole('listitem', { name: 'Automatic renewal (clause 1.2): Verified' }).textContent).toContain('Verified by dave');
    expect(within(screen.getByRole('region', { name: 'Missing documents' })).getByRole('listitem', { name: 'Insurance policies' })).toBeTruthy();
    expect(within(screen.getByRole('region', { name: 'Documents for each item' })).getByRole('link', { name: 'Supply Agreement - Northwind.txt' })).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a run starts from an approved checklist; a reviewer verifies, or dismisses with a reason', async () => {
    const calls = world('lead', {
      [`POST ${M}/diligence/runs`]: () => [200, { run: { id: 'r1', status: 'queued' } }],
      [`PATCH ${M}/diligence/issues/i1`]: () => [200, { issue: VIEW.issues[0] }],
    });
    renderLaw('/m/M-101/diligence');
    const reg = await screen.findByRole('article', { name: 'Issue register' });
    fireEvent.change(screen.getByLabelText('Checklist'), { target: { value: 'c1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run against this matter' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path.endsWith('/diligence/runs'))?.body).toEqual({ checklist_id: 'c1' }));
    const coc = within(reg).getByRole('listitem', { name: /^Change of control/ });
    fireEvent.click(within(coc).getByRole('button', { name: 'Dismiss' }));
    const form = within(coc).getByRole('form', { name: /^Dismiss/ });
    expect((within(form).getByRole('button', { name: 'Dismiss' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(within(form).getByLabelText('Why it is not an issue'), { target: { value: 'Waived by the supplier in writing' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Dismiss' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'PATCH')?.body).toEqual({ expected_version: 1, action: 'dismiss', reason: 'Waived by the supplier in writing' }));
  });

  it('a read-only member sees the register but cannot start runs or review', async () => {
    world('read_only');
    renderLaw('/m/M-101/diligence');
    const reg = await screen.findByRole('article', { name: 'Issue register' });
    expect(screen.queryByRole('region', { name: 'Start a due-diligence run' })).toBeNull();
    expect(within(reg).queryByRole('button', { name: 'Verify' })).toBeNull();
  });

  it('a managing partner approves a draft checklist by the hash they read; its author cannot', async () => {
    const draft: DdChecklist = { ...LIST, id: 'c2', version: 2, status: 'draft', created_by: 'nakamura', approved_by: null, approved_at: null, content_sha256: 'b'.repeat(64) };
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['managing_partner'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/diligence/checklists': () => [200, { checklists: [draft, LIST], can: { manage: true, approve: true }, questions: [] }],
      'POST /api/legal/diligence/checklists/c2/approve': () => [200, { checklist: { ...draft, status: 'approved' } }],
    });
    renderLaw('/checklists');
    const card = await screen.findByRole('region', { name: LIST.name });
    expect(within(card).getByText('Draft — not yet approved')).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Approve' }));
    const dialog = await screen.findByRole('dialog', { name: `Approve ${LIST.name}, version 2?` });
    fireEvent.click(within(dialog).getByLabelText('I have read every item of this version'));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Approve version 2' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/approve'))?.body).toEqual({ content_sha256: 'b'.repeat(64) }));
    cleanup();
    mockApi({
      'GET /api/auth/me': () => [200, me(['managing_partner'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/diligence/checklists': () => [200, { checklists: [{ ...draft, created_by: 'alice' }, LIST], can: { manage: true, approve: true }, questions: [] }],
    });
    renderLaw('/checklists');
    const own = await screen.findByRole('region', { name: LIST.name });
    expect(within(own).queryByRole('button', { name: 'Approve' })).toBeNull();
    expect(own.textContent).toContain('You wrote this version, so another managing partner approves it.');
  });
});
