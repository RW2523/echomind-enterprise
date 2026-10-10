import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { CitationCheck, CitationItem, MatterRole } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.4 — the citation and quote checker (DOC-03, T07; D-048). */
const M = '/api/legal/matters/id-M-101';
const base: Pick<CitationItem, 'malformed' | 'note' | 'diff' | 'notes' | 'target'> = { malformed: [], note: null, diff: [], notes: [], target: null };
const ITEMS: CitationItem[] = [
  { ...base, id: 'r1', kind: 'reference', type: 'clause', text: 'clause 8.1 of the Master Services Agreement', start: 40, end: 84, status: 'resolved',
    target: { document_id: 'd2', version_id: 'v2', title: 'Master Services Agreement', clause: '8.1', start: 100, end: 160, passage: "8.1 Each party's total liability shall not exceed USD 250,000." } },
  { ...base, id: 'r2', kind: 'reference', type: 'clause', text: 'Clause 15.4 of the Master Services Agreement', start: 300, end: 344, status: 'missing',
    note: 'The Master Services Agreement has no clause 15.4.', target: { document_id: 'd2', version_id: 'v2', title: 'Master Services Agreement', clause: null, start: 0, end: 0, passage: '' } },
  { ...base, id: 'q1', kind: 'quote', type: 'quotation', text: 'The Customer shall pay each invoice within sixty days of receipt.', start: 200, end: 268,
    status: 'altered', cited_as: 'Clause 4.2 of the Master Services Agreement',
    target: { document_id: 'd2', version_id: 'v2', title: 'Master Services Agreement', clause: '4.2', start: 20, end: 90, passage: '4.2 The Customer shall pay each invoice within thirty days of receipt.' },
    diff: [['eq', 'The Customer shall pay each invoice within '], ['del', 'thirty'], ['ins', 'sixty'], ['eq', ' days of receipt.']] },
  { ...base, id: 'a1', kind: 'authority', type: 'us_code', text: '15 U.S.C. § 78j(b)', start: 400, end: 418, status: 'unverified' },
  { ...base, id: 'a2', kind: 'authority', type: 'us_case', text: 'Henderson v. Bramwell, 999 F.5th 1234 (14th Cir. 2031)', start: 500, end: 553, status: 'unverified',
    malformed: ['the year 2031 is in the future', 'F.5th is not a known reporter', 'there is no 14th Cir.'],
    notes: [{ note: 'No such case: removed.', by: 'alice', at: '2026-10-10T11:00:00.000Z' }] },
];
const CHECK: CitationCheck = {
  id: 'c1', document_id: 'd1', version_id: 'v1', version_number: 1, items: ITEMS, engine: 'citations-s3.4', external_lookup: false,
  summary: { references: 2, resolved: 1, missing: 1, quotes: 1, accurate: 0, altered: 1, not_found: 0, authorities: 2, malformed: 1 },
  notice: 'External authorities are listed but not verified: no licensed law library is connected (decision P-009).',
  created_by: 'alice', created_at: '2026-10-10T10:00:00.000Z',
};

function world(role: MatterRole = 'lead', checks: CitationCheck[] = [CHECK], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role })] }],
    [`GET ${M}/documents/d1`]: () => [200, { document: { id: 'd1', title: 'advice-memo' } }],
    [`GET ${M}/documents/d1/citation-checks`]: () => [200, { checks }],
    ...extra,
  });
}

describe('Citation checks (S3.4)', () => {
  afterEach(() => cleanup());

  it('shows what is wrong first: an altered quotation with the changed words, a missing clause, a citation that cannot be real', async () => {
    const { container } = (world(), renderLaw('/m/M-101/documents/d1/citations'));
    const check = await screen.findByRole('article', { name: 'Citation check' });
    expect(within(check).getByRole('status').textContent).toBe('1 of 2 references found · 1 missing · 0 of 1 quotation accurate · 1 altered · 2 authorities, none verified · 1 cannot be real');
    const quote = within(within(check).getByRole('region', { name: 'Quotations' })).getByRole('listitem', { name: 'Quotation: Altered' });
    expect(Array.from(quote.querySelectorAll('del')).map((d) => d.textContent)).toEqual(['thirty']);
    expect(Array.from(quote.querySelectorAll('ins')).map((d) => d.textContent)).toEqual(['sixty']);
    expect(quote.textContent).toContain('Compared with Clause 4.2 of Master Services Agreement');
    const refs = within(within(check).getByRole('region', { name: 'Clause references' })).getAllByRole('listitem');
    expect(refs.map((r) => r.getAttribute('aria-label'))).toEqual(['Clause 15.4 of the Master Services Agreement: Missing clause', 'clause 8.1 of the Master Services Agreement: Found']);
    expect(refs[0].textContent).toContain('has no clause 15.4');
    expect(within(refs[0]).getByRole('link', { name: 'Master Services Agreement' }).getAttribute('href')).toBe('/m/M-101/documents/d2/read?v=v2');
    expect(within(refs[0]).queryByText('Show the passage')).toBeNull();
    expect(within(refs[1]).getByRole('link', { name: 'Open the passage' }).getAttribute('href')).toBe('/m/M-101/documents/d2/read?v=v2&at=100-160');
    const auths = within(within(check).getByRole('region', { name: 'Authorities' })).getAllByRole('listitem');
    expect(auths[0].getAttribute('aria-label')).toBe('Henderson v. Bramwell, 999 F.5th 1234 (14th Cir. 2031): cannot be real');
    expect(auths[0].textContent).toContain('there is no 14th Cir.');
    expect(auths[0].textContent).not.toContain('Cir..');
    expect(auths[0].textContent).toContain('No such case: removed.');
    expect(auths.every((a) => a.textContent!.includes('Unverified'))).toBe(true);
    expect(within(check).getByRole('note').textContent).toContain('P-009');
    expect(within(auths[1]).getByRole('link', { name: 'Open at the citation' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=400-418');
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a lawyer runs a check and notes an item', async () => {
    const calls = world('lead', [], { [`POST ${M}/documents/d1/citation-check`]: () => [200, { check: CHECK }] });
    renderLaw('/m/M-101/documents/d1/citations');
    expect(await screen.findByText('Not checked yet')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Check citations' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === `${M}/documents/d1/citation-check`)).toBe(true));
  });

  it('notes are added by lawyers; a read-only member sees the check but cannot run it or note it', async () => {
    const calls = world('lead', [CHECK], { [`POST ${M}/citation-checks/c1/notes`]: () => [200, { check: CHECK }] });
    renderLaw('/m/M-101/documents/d1/citations');
    fireEvent.click(await screen.findByRole('button', { name: 'Add a note on 15 U.S.C. § 78j(b)' }));
    const form = screen.getByRole('form', { name: 'Note on 15 U.S.C. § 78j(b)' });
    fireEvent.change(within(form).getByLabelText('Your note'), { target: { value: 'Checked in a licensed source.' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Save note' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/notes'))?.body).toEqual({ item: 'a1', note: 'Checked in a licensed source.' }));
    cleanup();
    world('read_only');
    renderLaw('/m/M-101/documents/d1/citations');
    await screen.findByRole('article', { name: 'Citation check' });
    expect(screen.queryByRole('button', { name: /Check/ })).toBeNull();
    expect(screen.queryByRole('form')).toBeNull();
    expect(screen.queryByRole('button', { name: /^Add a note/ })).toBeNull();
  });
});
