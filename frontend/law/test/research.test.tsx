import { afterEach, describe, expect, it } from 'vitest';
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
    expect(calls.find((c) => c.path === `${BASE}/query`)!.body).toEqual({ question: 'What caps liability?' });
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

  it('lists only your own earlier questions and reopens one', async () => {
    world([], [{ id: 't0', question: 'What is the monthly fee?', answer: 'The fee is USD 18,500 per month [2].', citations: [SRC[1]], status: 'answered', created_at: '2026-10-08T10:00:00.000Z' }]);
    renderLaw('/m/M-101/research');
    fireEvent.click(await screen.findByRole('button', { name: /What is the monthly fee\?/ }));
    const answer = await screen.findByRole('article', { name: 'Answer' });
    expect(within(answer).getByText(/The fee is USD 18,500 per month/)).toBeTruthy();
    expect(within(answer).getByRole('button', { name: 'Source 2' })).toBeTruthy();
  });
});
