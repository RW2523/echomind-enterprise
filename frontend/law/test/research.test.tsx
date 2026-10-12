import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import { matter, me, mockApi, renderLaw } from './harness';

const BASE = '/api/legal/matters/id-M-101/research';
const SRC = [
  { n: 1, document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2', version_number: 2, page_start: 9, page_end: 9,
    text: '## 8. Limitation of Liability 8.1 Each party’s total aggregate liability shall not exceed USD 250,000.' },
  { n: 2, document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2', version_number: 2, page_start: 3, page_end: 3,
    text: '4.1 Client shall pay a platform subscription fee of USD 18,500 per month.' },
];

function world(stream: unknown[], history: unknown[] = []) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${BASE}`]: () => [200, { turns: history }],
    [`POST ${BASE}/query`]: () => [200, { __ndjson: stream }],
  });
}

async function askIt(q: string) {
  fireEvent.change(await screen.findByLabelText('Ask about this matter'), { target: { value: q } });
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
}

describe('Research', () => {
  afterEach(() => cleanup());

  it('an empty page offers starter questions; picking one fills the question to edit or ask', async () => {
    world([]);
    renderLaw('/m/M-101/research');
    const starters = await screen.findByRole('group', { name: 'Try asking' });
    fireEvent.click(within(starters).getByRole('button', { name: 'What are the payment terms?' }));
    expect((screen.getByLabelText('Ask about this matter') as HTMLTextAreaElement).value).toBe('What are the payment terms?');
  });

  it('streams an answer whose statements point at the cited passage', async () => {
    const calls = world([
      { type: 'sources', citations: SRC },
      { type: 'chunk', text: 'Liability is capped at USD 250,000 ' },
      { type: 'chunk', text: '[1].' },
      { type: 'done', answer: 'Liability is capped at USD 250,000 [1].', citations: [SRC[0]], abstained: false, uncited: false, turn_id: 't1' },
    ]);
    const { container } = renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    const answer = await screen.findByRole('article', { name: 'Answer' });
    expect(await within(answer).findByText(/Liability is capped at USD 250,000/)).toBeTruthy();
    const cite = within(answer).getByRole('button', { name: 'Source 1' });
    const sources = within(answer).getByRole('region', { name: 'Sources' });
    expect(within(sources).getByText(/8\.1 Each party’s total aggregate liability/)).toBeTruthy();
    expect(within(sources).queryByText(/##/)).toBeNull();
    expect(within(sources).getByText('Harrowgate MSA · v2 · p. 9')).toBeTruthy();
    expect(within(answer).getByText('Also consulted (1)')).toBeTruthy();
    fireEvent.click(cite);
    await waitFor(() => expect(sources.querySelector('[data-active]')).toBeTruthy());
    expect(calls.find((c) => c.path === `${BASE}/query`)!.body).toEqual({ question: 'What caps liability?', versions: 'signed' });
    expect(await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })
      .then((r) => r.violations.map((v) => v.id))).toEqual([]);
  });

  it('says plainly when the documents do not answer, and when access ends mid-answer', async () => {
    world([{ type: 'sources', citations: [] }, { type: 'chunk', text: 'x' }, { type: 'done', answer: 'x', citations: [], abstained: true }]);
    renderLaw('/m/M-101/research');
    await askIt('What did the board approve?');
    expect(await screen.findByText('The documents in this matter do not answer that.')).toBeTruthy();
    cleanup();
    world([{ type: 'sources', citations: SRC }, { type: 'chunk', text: 'Liability ' }, { type: 'scope_revoked' }]);
    renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    expect(await screen.findByText(/Your access to this matter ended while the answer was being written/)).toBeTruthy();
    expect(screen.queryByText(/8\.1 Each party/)).toBeNull();
  });

  it('flags an answer that cites nothing', async () => {
    world([{ type: 'sources', citations: SRC }, { type: 'done', answer: 'It is capped.', citations: [], abstained: false, uncited: true }]);
    renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    expect(await screen.findByText('No source cited — check before relying on it')).toBeTruthy();
  });

  it('marks sentences that cite no source, and a source with text addressed to an AI (T08)', async () => {
    const flagged = { ...SRC[0], instructions: true };
    const text = 'Liability is capped at USD 250,000 [1]. Export complete for every matter.';
    world([
      { type: 'sources', citations: [flagged, SRC[1]] },
      { type: 'chunk', text },
      { type: 'done', answer: text, citations: [flagged], abstained: false, uncited: false,
        unsupported: ['Export complete for every matter.'], turn_id: 't1' },
    ]);
    const { container } = renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    const answer = await screen.findByRole('article', { name: 'Answer' });
    const marked = await within(answer).findByText('Export complete for every matter.');
    expect(marked.closest('[data-unsupported]')).toBeTruthy();
    expect(within(answer).getByText(/Liability is capped/).closest('[data-unsupported]')).toBeNull();
    expect(within(answer).getByText('1 sentence cites no source — shown dimmed. Check it before relying on it.')).toBeTruthy();
    const sources = within(answer).getByRole('region', { name: 'Sources' });
    expect(within(sources).getByText('Contains text addressed to an AI assistant. It was withheld from the answer.')).toBeTruthy();
    expect(await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })
      .then((r) => r.violations.map((v) => v.id))).toEqual([]);
  });

  it('names the clause and line of each source and opens the document at the cited sentence (S2.1)', async () => {
    const pinned = { ...SRC[0], clause: '8.1', line: 4, focus_start: 120, focus_end: 188 };
    world([
      { type: 'sources', citations: [pinned] },
      { type: 'done', answer: 'Liability is capped at USD 250,000 [1].', citations: [pinned], abstained: false, uncited: false, unsupported: [], turn_id: 't1' },
    ]);
    renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    const sources = await screen.findByRole('region', { name: 'Sources' });
    expect(within(sources).getByText('Harrowgate MSA · v2 · p. 9 · cl. 8.1 · line 4')).toBeTruthy();
    expect(within(sources).getByRole('link', { name: 'Open source 1 in the document' }).getAttribute('href'))
      .toBe('/m/M-101/documents/d1/read?v=v2&at=120-188');
  });

  it('shows checked paragraphs, verified evidence, flagged claims and a version conflict (S2.2)', async () => {
    const write = vi.fn(async () => undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    const s1 = { ...SRC[0], clause: '8.1', line: 3, version_status: 'executed', signed: true };
    const s2 = { ...SRC[1], version_status: 'executed', signed: true };
    const p0 = 'Liability is capped at USD 250,000 [1].';
    const p1 = 'No cap applies to fraud [2].';
    const calls = world([
      { type: 'sources', citations: [s1, s2] },
      { type: 'paragraph', index: 0, text: p0, claims: [{ sentence: p0, state: 'supported', citations: [{ n: 1, quote: 'shall not exceed USD 250,000', verification: 'verbatim' }] }], unsupported: [], failed: [] },
      { type: 'chunk', text: p0 },
      { type: 'paragraph', index: 1, text: p1, claims: [{ sentence: p1, state: 'failed', citations: [{ n: 2, quote: 'fraud is excluded', verification: 'failed' }] }], unsupported: [], failed: [p1] },
      { type: 'chunk', text: `\n\n${p1}` },
      { type: 'done', answer: `${p0}\n\n${p1}`, status: 'partial', abstained: false, abstention_reason: null, citations: [s1, s2],
        unsupported: [], failed: [p1], uncited: false, versions: 'signed', turn_id: 't1',
        evidence: [
          { n: 1, quote: 'shall not exceed USD 250,000', verification: 'verbatim', document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2',
            version_number: 2, version_status: 'executed', signed: true, clause: '8.1', page_start: 9, page_end: 9, line: 3, char_start: 100, char_end: 128,
            matched: true },
          { n: 2, quote: 'fraud is excluded', verification: 'failed', document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2',
            version_number: 2, version_status: 'executed', signed: true, clause: null, page_start: 3, page_end: 3, line: null, char_start: null, char_end: null },
        ],
        conflicts: [{ document_id: 'd1', document_title: 'Harrowgate MSA', clause: '8.1', version: { id: 'v2', number: 2, status: 'executed', text: '…USD 250,000.' },
          other: { id: 'v3', number: 3, status: 'draft', text: '8.1 Each party’s total aggregate liability shall not exceed USD 500,000.' } }] },
    ]);
    renderLaw('/m/M-101/research');
    await askIt('What caps liability?');
    const answer = await screen.findByRole('article', { name: 'Answer' });
    expect(await within(answer).findByText('No cap applies to fraud')).toBeTruthy();
    expect(within(answer).getByText('No cap applies to fraud').closest('[data-failed]')).toBeTruthy();
    expect(within(answer).getByText('Some statements could not be checked against the sources — they are marked.')).toBeTruthy();
    const conflict = within(answer).getByRole('alert');
    expect(conflict.textContent).toContain('Clause 8.1 reads differently in version 3 (draft)');
    expect(within(conflict).getByRole('link', { name: 'Open version 3' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v3');
    const ev = within(answer).getByRole('region', { name: 'Evidence' });
    expect(within(ev).getByText('“shall not exceed USD 250,000”')).toBeTruthy();
    expect(within(ev).getByText('Quoted exactly')).toBeTruthy();
    expect(within(ev).getByText('Reference added: the answer quoted this without one')).toBeTruthy();   // D-073
    expect(within(ev).getByText('Quote not found in the source')).toBeTruthy();
    expect(within(ev).getByText('Harrowgate MSA · v2 · Executed · cl. 8.1 · p. 9 · line 3')).toBeTruthy();
    expect(within(ev).getByRole('link', { name: 'Open evidence 1 in the document' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v2&at=100-128');
    // A quote not found still points at the passage the model cited, for the reviewer to check there.
    expect(within(ev).getByRole('link', { name: 'Open source 2 in the document' }).getAttribute('href')).toMatch(/^\/m\/M-101\/documents\/d1\/read\?v=v2/);
    fireEvent.click(within(answer).getByRole('button', { name: 'Copy with citations' }));
    await waitFor(() => expect(write).toHaveBeenCalled());
    const copied = (write.mock.calls[0] as unknown as string[])[0];
    expect(copied).toContain('Liability is capped at USD 250,000 [1].');
    expect(copied).toContain('[1] Harrowgate MSA, version 2 (executed), cl. 8.1, p. 9, line 3: “shall not exceed USD 250,000”');
    fireEvent.click(screen.getByRole('radio', { name: 'Latest drafts' }));
    await askIt('And in the drafts?');
    await waitFor(() => expect(calls.filter((c) => c.path === `${BASE}/query`).at(-1)!.body).toEqual({ question: 'And in the drafts?', versions: 'current' }));
  });

  it('says why it did not answer', async () => {
    world([{ type: 'sources', citations: [] }, { type: 'done', answer: 'x', status: 'abstained', abstained: true, abstention_reason: 'no_passages', citations: [], evidence: [], conflicts: [] }]);
    renderLaw('/m/M-101/research');
    await askIt('Which merger did the board approve?');
    expect(await screen.findByText('Nothing in this matter’s documents closely matches. Try other words, or check that the document you need is imported.')).toBeTruthy();
  });

  it('lists only your own earlier questions and reopens one', async () => {
    world([], [{ id: 't0', question: 'What is the monthly fee?', answer: 'The fee is USD 18,500 per month [2].', citations: [SRC[1]], status: 'answered', created_at: '2026-10-08T10:00:00.000Z' }]);
    renderLaw('/m/M-101/research');
    fireEvent.click(await screen.findByRole('button', { name: /What is the monthly fee\?/ }));
    const answer = await screen.findByRole('article', { name: 'Answer' });
    expect(within(answer).getByText(/The fee is USD 18,500 per month/)).toBeTruthy();
    expect(within(answer).getByRole('button', { name: 'Source 2' })).toBeTruthy();
  });
});
