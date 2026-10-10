import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Obligation } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.6 — obligations and deadlines (DOC-05; D-050). */
const M = '/api/legal/matters/id-M-101';
const base = (x: Partial<Obligation> & Pick<Obligation, 'id' | 'kind' | 'action'>): Obligation => ({
  document_id: 'd1', document_title: 'Master Services Agreement (DEMO)', version_id: 'v1', clause: '2.2', party: 'Either party', when: null,
  due_date: null, due_note: null, timezone: 'UTC', due_at: null, overdue: false, status: 'proposed', owner: null, source: 'rules',
  evidence: { start: 100, end: 180, quote: 'The words of the clause.' }, confirmed_by: null, confirmed_at: null, version: 1,
  created_by: 'alice', created_at: '2026-10-10T10:00:00.000Z', updated_at: '2026-10-10T10:00:00.000Z', ...x,
});
const OBS: Obligation[] = [
  base({ id: 'o1', kind: 'renewal', action: 'give written notice of non-renewal', when: 'at least ninety (90) days before the end of the then-current term',
    due_date: '2028-12-01', due_at: '2028-12-01T23:59:59+00:00',
    due_note: '90 days before the end of the initial term. The initial term is read as 36 months from 2 March 2026 (the Effective Date), ending on 1 March 2029 (counting the first day), so notice is due by 1 December 2028. Confirm how the days are counted before relying on it.' }),
  base({ id: 'o2', kind: 'notification', action: 'notify Client of any personal data breach affecting Client Data', clause: '6.1', party: 'Provider',
    when: 'within forty-eight (48) hours of becoming aware of it', due_note: 'Runs from an event (becoming aware); no date until it happens.',
    status: 'confirmed', confirmed_by: 'alice', owner: 'alice', version: 2 }),
  base({ id: 'o3', kind: 'payment', action: 'Invoices are payable', clause: '4.2', party: null, when: 'thirty (30) days from the invoice date',
    due_date: '2026-07-01', due_at: '2026-07-01T23:59:59+00:00', overdue: true, due_note: 'Set by alice: test date' }),
];

function world(role: 'lead' | 'read_only' = 'lead', extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role })] }],
    [`GET ${M}/obligations`]: () => [200, { obligations: OBS, timezone: 'UTC', calendar: false }],
    [`GET ${M}/documents`]: () => [200, { documents: [{ id: 'd1', title: 'Master Services Agreement (DEMO)', current_version: { index_status: 'indexed' } }] }],
    [`GET ${M}/memberships`]: () => [200, { memberships: [{ id: 'm1', user_id: 'u1', username: 'alice', display_name: 'Alice', role: 'lead' }] }],
    ...extra,
  });
}

describe('Obligations (S3.6)', () => {
  afterEach(() => cleanup());

  it('overdue first, then dated, then those running from an event; each with its words, its date and how the date was reached', async () => {
    const { container } = (world(), renderLaw('/m/M-101/obligations'));
    const list = await screen.findByRole('article', { name: 'Obligations' });
    expect(within(list).getByRole('status').textContent).toBe('3 open · 2 not yet confirmed · 1 overdue · next due 1 December 2028 · dates in UTC');
    expect(within(list).getAllByRole('region').map((r) => r.getAttribute('aria-label'))).toEqual(['Overdue', 'Due on a date', 'Running from an event']);
    const renewal = within(list).getByRole('listitem', { name: 'Give written notice of non-renewal: Proposed — not confirmed' });
    expect(renewal.textContent).toContain('Due 1 December 2028 (end of day, UTC)');
    expect(renewal.textContent).toContain('ending on 1 March 2029');
    expect(within(renewal).getByRole('link', { name: 'Open the words' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=100-180');
    expect(within(list).getByRole('listitem', { name: /^Notify Client/ }).textContent).toContain('Confirmed by alice');
    const late = within(within(list).getByRole('region', { name: 'Overdue' })).getByRole('listitem', { name: /^Invoices are payable/ });
    expect(within(late).getByText('Overdue')).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a lawyer confirms with an owner; dismissing needs a reason; an edit sends only what changed, with why', async () => {
    const calls = world('lead', { [`PATCH ${M}/obligations/o1`]: () => [200, { obligation: OBS[0] }] });
    renderLaw('/m/M-101/obligations');
    const renewal = await screen.findByRole('listitem', { name: 'Give written notice of non-renewal: Proposed — not confirmed' });
    fireEvent.click(within(renewal).getByRole('button', { name: 'Confirm' }));
    const form = await within(renewal).findByRole('form', { name: 'Confirm Give written notice of non-renewal' });
    await waitFor(() => expect(within(form).getAllByRole('option').length).toBe(2));
    fireEvent.change(within(form).getByRole('combobox'), { target: { value: 'alice' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Confirm this obligation' }));
    await waitFor(() => expect(calls.filter((c) => c.method === 'PATCH').at(-1)?.body).toEqual({ expected_version: 1, action: 'confirm', owner: 'alice' }));
    fireEvent.click(await within(renewal).findByRole('button', { name: 'Edit' }));
    const edit = within(renewal).getByRole('form', { name: 'Edit Give written notice of non-renewal' });
    fireEvent.change(within(edit).getByLabelText('Due date'), { target: { value: '2028-11-30' } });
    expect((within(edit).getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(within(edit).getByLabelText('Why (kept with the change)'), { target: { value: 'Counted as clear days' } });
    fireEvent.click(within(edit).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(calls.filter((c) => c.method === 'PATCH').at(-1)?.body).toEqual({ expected_version: 1, action: 'edit', reason: 'Counted as clear days', due_date: '2028-11-30' }));
  });

  it('a read-only member sees the obligations but cannot read documents, confirm or edit', async () => {
    world('read_only');
    renderLaw('/m/M-101/obligations');
    const list = await screen.findByRole('article', { name: 'Obligations' });
    expect(screen.queryByRole('region', { name: "Read a document's obligations" })).toBeNull();
    expect(within(list).queryByRole('button', { name: 'Confirm' })).toBeNull();
    expect(within(list).queryByRole('button', { name: 'Edit' })).toBeNull();
  });
});
