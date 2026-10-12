import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, SummaryContent, SummaryVersion } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** OPT-02 (D-070) — machine translation by the local model: beside the original, labelled, figures checked. */
const M = '/api/legal/matters/id-M-101';
const LABEL = "Machine translation by this server's model (qwen3), not reviewed by anyone. The original prevails: check it before relying on the translation.";
const withTranslation = (on = true): Me => ({ ...me(['attorney']), features: { intake: false, intake_links: false, summaries: true, translation: on } });
const DOC = { id: 'd1', title: 'Services Agreement.txt', status: 'executed', current_version: { id: 'v1', number: 1, index_status: 'indexed' } };
const axeClean = async (el: Element) =>
  (await axe.run(el, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id);

const para = (i: number, source: string, text: string, extra = {}) => ({ type: 'paragraph', i, source, text, withheld: false, figures_missing: [], ...extra });

function world(on = true) {
  return mockApi({
    'GET /api/auth/me': () => [200, withTranslation(on)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${M}/documents/d1`]: () => [200, { document: DOC }],
    [`GET ${M}/documents/d1/versions`]: () => [200, { versions: [] }],
    [`POST ${M}/documents/d1/translate`]: (b) => [200, { __ndjson: b.from ? [
      { type: 'start', language: { code: 'fr', name: 'French' }, paragraphs: 3, from: 2, label: LABEL },
      para(2, '3. Notices go to the registered office.', '3. Les notifications sont envoyées au siège social.'),
      { type: 'done', translated: 1 },
    ] : [
      { type: 'start', language: { code: 'fr', name: 'French' }, paragraphs: 3, from: 0, label: LABEL },
      para(0, '1. The Customer shall pay USD 19,750.', '1. Le Client paiera 19 750 USD.'),
      para(1, '2. Liability is capped at USD 250,000.', '2. La responsabilité est plafonnée.', { figures_missing: ['250,000'] }),
      { type: 'limit', next: 2, remaining: 1 },
      { type: 'done', translated: 2 },
    ] }],
  });
}

describe('Machine translation (OPT-02, D-070)', () => {
  afterEach(() => cleanup());

  it('translates a document beside its original, labelled, with missing figures flagged, and continues where it stopped', async () => {
    const calls = world();
    const { container } = renderLaw('/m/M-101/documents/d1/translate');
    fireEvent.change(await screen.findByRole('combobox', { name: 'Into' }), { target: { value: 'fr' } });
    fireEvent.click(screen.getByRole('button', { name: 'Translate' }));
    const rows = await screen.findByRole('list', { name: 'Original and translation' });
    await waitFor(() => expect(within(rows).getAllByRole('listitem')).toHaveLength(2));
    expect(screen.getByRole('note').textContent).toBe(LABEL);
    const [first, second] = within(rows).getAllByRole('listitem');
    expect(first.textContent).toContain('1. The Customer shall pay USD 19,750.');
    expect(first.textContent).toContain('1. Le Client paiera 19 750 USD.');
    expect(second.textContent).toMatch(/Check: 250,000 is not in the translation/);
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ language: 'fr' });
    fireEvent.click(screen.getByRole('button', { name: 'Continue (1 paragraph left)' }));
    await waitFor(() => expect(within(rows).getAllByRole('listitem')).toHaveLength(3));
    expect(calls.filter((c) => c.method === 'POST')[1].body).toEqual({ language: 'fr', from: 2 });
    expect(await axeClean(container)).toEqual([]);
  });

  it('is not offered when the firm has not turned it on', async () => {
    world(false);
    renderLaw('/m/M-101/documents/d1/translate');
    expect(await screen.findByText(/Translation is off on this server/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Translate' })).toBeNull();
  });

  it("translates a summary's points, each under its original, with the same label", async () => {
    const content: SummaryContent = {
      title: 'Summary — Services Agreement', document: { id: 'd1', title: 'Services Agreement.txt', version_id: 'v1', version_number: 1 }, transcript: false,
      points: [{ id: 'p1', text: 'The Customer pays USD 19,750.', quote: 'pay USD 19,750', verification: 'verbatim', start: 1, end: 5, cite: { line: 1 },
        figures_unsupported: [], origin: 'model' }],
      sections: 1, unverified: 0, figures_unsupported: 0, translation: { available: true, reason: '' },
    };
    const v: SummaryVersion = { id: 's1', version: 1, status: 'draft', content, content_sha256: 'c'.repeat(64), inputs: {}, created_by: 'u', created_by_name: 'alice',
      created_at: '2026-10-10T10:00:00.000Z', origin: 'model', document_id: null, decisions: [], kind: 'summary', title: content.title, subject_id: 'd1' };
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withTranslation()],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/artifacts/s1`]: () => [200, { artifact: v, versions: [{ id: 's1', version: 1, status: 'draft', created_by_name: 'alice', created_at: v.created_at, origin: 'model' }],
        can: { approve: true, file: true, author: false } }],
      [`POST ${M}/summaries/s1/translate`]: () => [200, { language: { code: 'es', name: 'Spanish' }, label: LABEL,
        points: [{ id: 'p1', translation: 'El Cliente paga 19.750 USD.', figures_missing: [] }] }],
    });
    renderLaw('/m/M-101/summaries/s1');
    fireEvent.change(await screen.findByRole('combobox', { name: 'Translate into' }), { target: { value: 'es' } });
    fireEvent.click(screen.getByRole('button', { name: 'Translate' }));
    const list = await screen.findByRole('list', { name: 'Key points' });
    expect(await within(list).findByText('El Cliente paga 19.750 USD.')).toBeTruthy();
    expect(screen.getByRole('note').textContent).toBe(LABEL);
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ language: 'es' });
  });
});
