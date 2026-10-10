import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import { matter, me, mockApi, renderLaw } from './harness';

const M = '/api/legal/matters/id-M-101';
const DOC = {
  id: 'd1', title: 'Harrowgate MSA', kind: 'contract', status: 'draft', version: 2, created_at: '2026-10-01T00:00:00Z', findings: {},
  current_version: { id: 'v2', number: 2, filename: 'msa-v2.md', mime: 'text/markdown', size_bytes: 10, sha256: 'x', uploaded_at: '2026-10-02T00:00:00Z',
    index_status: 'indexed', chunk_count: 4, status: 'draft' },
};
const VERSIONS = [
  { ...DOC.current_version, id: 'v1', number: 1, filename: 'msa-v1.md', status: 'executed' },
  DOC.current_version,
];
const L = { document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v1', number: 1, status: 'executed', filename: 'msa-v1.md' };
const R = { ...L, version_id: 'v2', number: 2, status: 'draft', filename: 'msa-v2.md' };
const side = (label: string, text: string, start = 0) => ({ start, end: start + text.length, label, clause: label, heading: null, kind: 'paragraph', text, page: null, line: 3 });
const COMPARISON = {
  id: 'c1', kind: 'versions', status: 'done', created_at: '2026-10-09T10:00:00Z', left: L, right: R, passes: ['wording'],
  summary: { unchanged: 1, formatting: 0, changed: 1, added: 1, removed: 0, moved: 1, renumbered: 1 },
  pairs: [
    { index: 0, kind: 'unchanged', moved: false, how: 'same', similarity: 1, renumbered: false, left: side('1.1', '1.1 "Services" means support.'),
      right: side('1.1', '1.1 "Services" means support.'), left_marks: [], right_marks: [], changes: [], flags: [] },
    { index: 1, kind: 'changed', moved: false, how: 'wording', similarity: 0.94, renumbered: false,
      left: side('4.1', '4.1 Client shall pay a fee of USD 18,500 per month.', 100), right: side('4.1', '4.1 Client shall pay a fee of USD 19,750 per month.', 120),
      left_marks: [[34, 40]], right_marks: [[34, 40]], changes: [{ from: '18,500', to: '19,750' }], flags: ['number'] },
    { index: 2, kind: 'added', moved: false, how: 'same', similarity: 1, renumbered: false, left: null,
      right: side('4.2', '4.2 The fee increases by three percent (3%) each year.', 200), left_marks: [], right_marks: [], changes: [], flags: [] },
    { index: 3, kind: 'unchanged', moved: true, how: 'moved', similarity: 1, renumbered: true, left: side('14.2', '14.2 Return confidential information.', 900),
      right: side('5.1', '5.1 Return confidential information.', 300), left_marks: [], right_marks: [], changes: [], flags: [] },
  ],
};

function world(extra: Record<string, (b: any) => [number, unknown]> = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${M}/documents`]: () => [200, { documents: [DOC] }],
    [`GET ${M}/documents/d1/versions`]: () => [200, { versions: VERSIONS }],
    [`GET ${M}/compare`]: () => [200, { comparisons: [] }],
    [`POST ${M}/compare`]: () => [200, { comparison: COMPARISON }],
    ...extra,
  });
}

describe('Compare', () => {
  afterEach(() => cleanup());

  it('compares the signed version with the latest draft, side by side, with changes marked', async () => {
    const calls = world();
    const { container } = renderLaw('/m/M-101/compare');
    const run = await screen.findByRole('button', { name: 'Compare' });
    await waitFor(() => expect((screen.getByLabelText('From: version') as HTMLSelectElement).value).toBe('v1'));   // the signed version
    expect((screen.getByLabelText('To: version') as HTMLSelectElement).value).toBe('v2');
    fireEvent.click(run);
    const result = await screen.findByRole('article', { name: 'Comparison' });
    expect(calls.find((c) => c.method === 'POST' && c.path === `${M}/compare`)!.body).toEqual({ left_version_id: 'v1', right_version_id: 'v2' });
    expect(within(result).getByRole('status').textContent).toMatch(/1 changed · 1 added · 1 moved · 1 renumbered · 1 unchanged/);
    const fee = within(result).getByRole('listitem', { name: 'cl. 4.1: Changed' });
    expect(fee.querySelector('del')!.textContent).toBe('removed: 18,500');
    expect(fee.querySelector('ins')!.textContent).toBe('added: 19,750');
    expect(within(fee).getByText('Number changed')).toBeTruthy();
    expect(within(result).getByRole('listitem', { name: 'cl. 4.2: Added' }).textContent).toMatch(/Not in version 1 \(executed\)/);
    expect(within(result).getByText('Moved from 14.2')).toBeTruthy();
    expect(within(result).queryByRole('listitem', { name: 'cl. 1.1: Unchanged' })).toBeNull();
    fireEvent.click(within(result).getByLabelText('Show unchanged'));
    expect(within(result).getByRole('listitem', { name: 'cl. 1.1: Unchanged' })).toBeTruthy();
    const open = within(fee).getByRole('link', { name: /Open cl\. 4\.1 of version 2/ });
    expect(open.getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v2&at=120-171');
    expect(await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })
      .then((r) => r.violations.map((v) => v.id))).toEqual([]);
  });

  it('explains a change with the quotes from each version and how they were checked', async () => {
    world({
      [`POST ${M}/compare/c1/explain`]: () => [200, { __ndjson: [
        { type: 'explanation', index: 1, text: 'The fee rises from [1] to [2].', status: 'partial', failed: [], unsupported: [], evidence: [
          { n: 1, side: 'left', quote: 'USD 18,500 per month', verification: 'verbatim', char_start: 126, char_end: 146 },
          { n: 2, side: 'right', quote: 'USD 20,000 per month', verification: 'failed', char_start: null, char_end: null }] },
        { type: 'done', explained: 1 }] }],
    });
    renderLaw('/m/M-101/compare');
    await waitFor(() => expect((screen.getByLabelText('From: version') as HTMLSelectElement).value).toBe('v1'));
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }));
    const fee = await screen.findByRole('listitem', { name: 'cl. 4.1: Changed' });
    fireEvent.click(within(fee).getByRole('button', { name: 'Explain' }));
    expect(await within(fee).findByText('The fee rises from [from] to [to].')).toBeTruthy();
    const quotes = within(fee).getByRole('list', { name: 'Quotes in this explanation' });
    expect(within(quotes).getByText('Quoted exactly')).toBeTruthy();
    expect(within(quotes).getByText('Quote not found')).toBeTruthy();
    expect(within(fee).getByText('Some statements could not be checked against the wording.')).toBeTruthy();
  });

  it('checks the documents for conflicts, with both quotes, and keeps what was set aside', async () => {
    const sideOf = (title: string, value: string, sentence: string, signed = true) => ({
      document_id: title, document_title: title, version_id: `${title}-v`, version_number: 1, version_status: signed ? 'executed' : 'draft', signed,
      clause: '4.2', page: null, line: 7, value, value_start: sentence.indexOf(value), value_end: sentence.indexOf(value) + value.length,
      sentence_start: 0, sentence_end: sentence.length, context_start: 0, context_end: sentence.length, context: sentence });
    const conflict = { kind: 'period', score: 0.8, verdict: 'conflict', method: 'model', subject: 'yes, invoice payment terms between Harrowgate and Voss-Lindqvist.',
      explanation: 'The MSA says [1] and the order form says [2].', failed: [],
      evidence: [{ n: 1, quote: 'thirty (30) days', verification: 'verbatim' }, { n: 2, quote: 'forty-five (45) days', verification: 'verbatim' }],
      a: sideOf('Harrowgate MSA', 'thirty (30) days', 'Invoices are payable net thirty (30) days.'),
      b: sideOf('Order Form No. 1', 'forty-five (45) days', 'Invoices are payable within forty-five (45) days.', false) };
    const setAside = { ...conflict, verdict: 'no_conflict', explanation: 'Different relationships.',
      a: sideOf('Engagement letter', 'thirty (30) days', 'Invoices are payable within thirty (30) days.') };
    const calls = world({
      [`POST ${M}/compare/scan`]: () => [200, { __ndjson: [
        { type: 'documents', documents: [{ document_id: 'a' }, { document_id: 'b' }, { document_id: 'c' }], candidates: 2 },
        { type: 'checked', done: 1, total: 2, finding: conflict },
        { type: 'checked', done: 2, total: 2, finding: setAside },
        { type: 'done', id: 's1', found: 1, unclear: 0, rejected: 1 }] }],
    });
    const { container } = renderLaw('/m/M-101/compare?view=scan');
    fireEvent.click(await screen.findByRole('radio', { name: 'Latest drafts' }));
    fireEvent.click(screen.getByRole('button', { name: 'Check for conflicts' }));
    const result = await screen.findByRole('article', { name: 'Conflicts' });
    expect(await within(result).findByText(/3 documents read \(latest drafts\) · 2 possible conflicts checked · 1 conflict found/)).toBeTruthy();
    expect(calls.find((c) => c.path === `${M}/compare/scan`)!.body).toEqual({ versions: 'current' });
    const found = within(result).getByRole('list', { name: 'Possible conflicts' });
    const item = within(found).getByRole('listitem', { name: 'Periods differ' });
    expect(within(item).getByText('The documents say different things')).toBeTruthy();
    expect(item.querySelectorAll('mark')[1].textContent).toBe('forty-five (45) days');
    expect(within(item).getByText('Not signed')).toBeTruthy();
    expect(within(item).getByText('Checked in context')).toBeTruthy();
    expect(within(item).getByText('What each passage governs: invoice payment terms between Harrowgate and Voss-Lindqvist.')).toBeTruthy();
    expect(within(item).getByText('The MSA says [Harrowgate MSA] and the order form says [Order Form No. 1].')).toBeTruthy();
    expect(within(within(item).getByRole('list', { name: 'Quotes in this explanation' })).getByText('Order Form No. 1')).toBeTruthy();
    expect(within(item).getByRole('link', { name: 'Open Order Form No. 1 at this sentence' }).getAttribute('href'))
      .toBe('/m/M-101/documents/Order%20Form%20No.%201/read?v=Order%20Form%20No.%201-v&at=0-49');
    expect(within(result).getByText('Checked and not in conflict (1)')).toBeTruthy();
    expect(await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })
      .then((r) => r.violations.map((v) => v.id))).toEqual([]);
  });

  it('says when access ends during a check, and reopens earlier comparisons', async () => {
    world({
      [`POST ${M}/compare/scan`]: () => [200, { __ndjson: [{ type: 'documents', documents: [], candidates: 3 }, { type: 'scope_revoked' }] }],
      [`GET ${M}/compare`]: () => [200, { comparisons: [{ id: 'c1', kind: 'versions', status: 'done', created_at: '2026-10-09T10:00:00Z',
        left: L, right: R, summary: COMPARISON.summary }] }],
      [`GET ${M}/compare/c1`]: () => [200, { comparison: COMPARISON }],
    });
    const { router } = renderLaw('/m/M-101/compare');
    const across = await screen.findByRole('radio', { name: 'Across documents' });
    fireEvent.click(across);
    expect((across as HTMLInputElement).checked).toBe(true);                      // at once, not after navigation
    await waitFor(() => expect(router.state.location.search).toBe('?view=scan'));
    fireEvent.click(await screen.findByRole('button', { name: 'Check for conflicts' }));
    expect(await screen.findByText(/Your access to this matter ended during the check/)).toBeTruthy();
    const earlier = screen.getByRole('complementary', { name: 'Your earlier comparisons' });
    fireEvent.click(await within(earlier).findByRole('button', { name: /Harrowgate MSA v1 → v2/ }));
    expect(await screen.findByRole('listitem', { name: 'cl. 4.1: Changed' })).toBeTruthy();
    expect((screen.getByRole('radio', { name: 'Two versions' }) as HTMLInputElement).checked).toBe(true);
  });
});
