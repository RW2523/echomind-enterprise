import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, PanelContent, PanelPerspective, PanelView } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S4.5 — four-perspective review panel (OPT-05; D-057). */
const M = '/api/legal/matters/id-M-101';
const ev = (quote: string, clause: string) => ({ n: 1, quote, verification: 'verbatim' as const, document_id: 'd1', document_title: 'MSA.txt', version_id: 'v1',
  version_number: 1, clause, line: 3, char_start: 40, char_end: 70 });
const persp = (role: PanelPerspective['role'], label: string, text: string, extra: Partial<PanelPerspective> = {}): PanelPerspective => ({
  role, label, text, evidence: [ev('limited to USD 250,000', '9.1')], unsupported: [], failed: [], missing_facts: [], abstained: false, ...extra });
const CONTENT: PanelContent = {
  title: 'Panel review — change of control', question: 'Can the Supplier terminate on a change of control?', specialist: 'commercial', area: 'commercial contracts',
  perspectives: [persp('deal', 'Deal lawyer', 'The cap stays at USD 250,000.'), persp('risk', 'Risk and compliance', 'A change of control lets the Supplier end the contract.', { unsupported: ['This is a real risk.'] }),
    persp('litigation', 'Litigator', 'A dispute would be under Delaware law.'), persp('specialist', 'Specialist (commercial contracts)', 'This is a standard commercial right.')],
  matrix: [{ issue: 'Whether the change-of-control right threatens the deal', positions: { deal: 'low risk', risk: 'real risk', litigation: 'not addressed', specialist: 'standard' },
    disagreement: true, summary: 'The exposure could reach USD 900,000.', figures_unsupported: ['900,000'] }],
  missing_facts: [], stopped: false, facts_given: [], notice: 'Four perspectives drafted by the same AI model to test the question from different angles. They are not independent experts.', sources: 1,
};
const view = (x: Partial<PanelView> = {}, content: Partial<PanelContent> = {}): PanelView => ({
  review: { id: 'r1', question: CONTENT.question, specialist: 'commercial', area: 'commercial contracts', status: 'done', facts: [], created_by: 'alice',
    created_at: '2026-10-10T10:00:00.000Z', finished_at: '2026-10-10T10:01:00.000Z', mine: false },
  artifact: { id: 'a1', version: 1, status: 'draft', content: { ...CONTENT, ...content }, content_sha256: 'd'.repeat(64), inputs: {}, created_by: 'usr_alice',
    created_by_name: 'alice', created_at: '2026-10-10T10:01:00.000Z', origin: 'model', document_id: null, decisions: [], kind: 'panel' },
  versions: [], can: { approve: true, facts: true, author: false }, ...x,
});
const withPanel = (roles: string[] = ['managing_partner']): Me => ({ ...me(roles), features: { intake: false, intake_links: false, panel: true } });

function world(v: PanelView, extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, withPanel()],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${M}/panel/r1`]: () => [200, v],
    ...extra,
  });
}

describe('Review panel (S4.5)', () => {
  afterEach(() => cleanup());

  it('shows four perspectives with their evidence, the matrix, and that they are not independent experts', async () => {
    const { container } = (world(view()), renderLaw('/m/M-101/panel/r1'));
    const grid = await screen.findByRole('region', { name: 'Perspectives' });
    expect(within(grid).getAllByRole('article').map((a) => a.getAttribute('aria-label'))).toEqual(['Deal lawyer', 'Risk and compliance', 'Litigator', 'Specialist (commercial contracts)']);
    const risk = within(grid).getByRole('article', { name: 'Risk and compliance' });
    expect(risk.textContent).toContain('Cites nothing: “This is a real risk.”');
    expect(within(risk).getByRole('link', { name: 'Open at the quote' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=40-70');
    const matrix = screen.getByRole('table', { name: 'Disagreement matrix' });
    const row = within(matrix).getAllByRole('row')[1];
    expect(row.textContent).toContain('They differ');
    expect(row.textContent).toContain('real risk');
    expect(row.textContent).toContain('Check: 900,000 is in no answer');
    expect(screen.getByText(/not independent experts/)).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Review this panel review' })).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a stopped panel asks for the missing facts and runs again; nothing can be approved meanwhile', async () => {
    const calls = world(view({ review: { ...view().review, status: 'needs_facts' } }, { stopped: true, missing_facts: ['whether the sale has been announced', 'the notice date'] }),
      { [`POST ${M}/panel/r1/facts`]: () => [200, { review: { id: 'r1', status: 'queued' } }] });
    renderLaw('/m/M-101/panel/r1');
    const stop = await screen.findByRole('region', { name: 'Facts needed' });
    expect(screen.queryByRole('group', { name: 'Review this panel review' })).toBeNull();
    fireEvent.change(within(stop).getByLabelText('Whether the sale has been announced'), { target: { value: 'Not yet; planned for March.' } });
    fireEvent.click(within(stop).getByRole('button', { name: 'Give these facts and run again' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/panel/r1/facts`)?.body).toEqual({
      facts: [{ question: 'whether the sale has been announced', answer: 'Not yet; planned for March.' }] }));
  });

  it('a lawyer who is not a partner sees that a partner decides', async () => {
    world(view({ can: { approve: false, facts: true, author: false } }));
    renderLaw('/m/M-101/panel/r1');
    expect(await screen.findByText(/Waiting for a partner/)).toBeTruthy();
    expect(screen.queryByRole('group', { name: 'Review this panel review' })).toBeNull();
  });

  it('asking the panel sends the question and the specialist area, and opens the review', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withPanel(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/panel`]: () => [200, { reviews: [], specialisms: { commercial: 'commercial contracts', employment: 'employment law' }, can: { ask: true, approve: false } }],
      [`POST ${M}/panel`]: () => [200, { review: { id: 'r1', status: 'queued' } }],
      [`GET ${M}/panel/r1`]: () => [200, view({ review: { ...view().review, status: 'running' }, artifact: null })],
    });
    renderLaw('/m/M-101/panel');
    const form = await screen.findByRole('form', { name: 'Ask the panel' });
    fireEvent.change(within(form).getByLabelText('Question'), { target: { value: 'Can the employer change the bonus scheme unilaterally?' } });
    fireEvent.change(within(form).getByLabelText('Specialist area'), { target: { value: 'employment' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Ask the panel' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === `${M}/panel`)?.body).toEqual({ question: 'Can the employer change the bonus scheme unilaterally?', specialist: 'employment' }));
    expect(await screen.findByRole('status')).toBeTruthy();                                // the panel is working
  });
});
