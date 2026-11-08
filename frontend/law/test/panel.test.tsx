import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, PanelContent, PanelPerspective, PanelReview, PanelView } from '../api/types';
import { councilFlow, evidenceInFocus, openingLine, stepIdeas } from '../ui/council';
import { matter, me, mockApi, renderLaw } from './harness';

/** S4.5 — the review panel, grown into the Council (2026-11-05) and redesigned (2026-11-07): the question with the
 *  perspectives around it, the discussion with its citations, the evidence, the flow and the result. */
const M = '/api/legal/matters/id-M-101';
const READER = `${M}/documents/d1/versions/v1/reader`;
const MSA_TEXT = 'MASTER SERVICES AGREEMENT. 9.1 Each party\'s total liability is limited to USD 250,000 in any year. 14.2 The Supplier may terminate on notice.';
const at = (q: string) => [MSA_TEXT.indexOf(q), MSA_TEXT.indexOf(q) + q.length] as const;
const ev = (quote: string, clause: string, n = 1, extra = {}) => ({ n, quote, verification: 'verbatim' as const, document_id: 'd1', document_title: 'MSA.txt', version_id: 'v1',
  version_number: 1, clause, line: 3, page_start: 4, char_start: at(quote)[0], char_end: at(quote)[1], ...extra });
const persp = (role: PanelPerspective['role'], label: string, text: string, extra: Partial<PanelPerspective> = {}): PanelPerspective => ({
  role, label, text, evidence: [ev('limited to USD 250,000', '9.1')], unsupported: [], failed: [], missing_facts: [], abstained: false, ...extra });
const CONTENT: PanelContent = {
  title: 'Panel review — change of control', question: 'Can the Supplier terminate on a change of control?', specialist: 'commercial', area: 'commercial contracts',
  perspectives: [persp('deal', 'Deal lawyer', 'The cap stays at USD 250,000 [1]. That suits a sale.'),
    persp('risk', 'Risk and compliance', 'A change of control lets the Supplier end the contract [2].', {
      unsupported: ['This is a real risk.'], evidence: [ev('may terminate on notice', '14.2', 2)], missing_facts: ['whether a sale is planned'] }),
    persp('litigation', 'Litigator', 'A dispute would be under Delaware law.'), persp('specialist', 'Specialist (commercial contracts)', 'This is a standard commercial right.')],
  matrix: [{ issue: 'Whether the change-of-control right threatens the deal', positions: { deal: 'low risk', risk: 'real risk', litigation: 'not addressed', specialist: 'standard' },
    disagreement: true, summary: 'The exposure could reach USD 900,000.', figures_unsupported: ['900,000'] },
  { issue: 'Governing law', positions: { deal: 'Delaware', risk: 'Delaware', litigation: 'Delaware', specialist: 'Delaware' }, disagreement: false, summary: null, figures_unsupported: [] }],
  missing_facts: ['whether a sale is planned'], stopped: false, facts_given: [], sources: 1,
  notice: 'Four perspectives drafted by the same AI model to test the question from different angles. They are not independent experts.',
};
const REVIEW: PanelReview = { id: 'r1', question: CONTENT.question, specialist: 'commercial', area: 'commercial contracts', status: 'done', facts: [], created_by: 'alice',
  created_at: '2026-10-10T10:00:00.000Z', finished_at: '2026-10-10T10:01:00.000Z', mine: false, depth: 'focused',
  roles: [{ role: 'deal', label: 'Deal lawyer' }, { role: 'risk', label: 'Risk and compliance' }, { role: 'litigation', label: 'Litigator' },
    { role: 'specialist', label: 'Specialist (commercial contracts)' }] };
const view = (x: Partial<PanelView> = {}, content: Partial<PanelContent> = {}): PanelView => ({
  review: REVIEW,
  artifact: { id: 'a1', version: 1, status: 'draft', content: { ...CONTENT, ...content }, content_sha256: 'd'.repeat(64), inputs: {}, created_by: 'usr_alice',
    created_by_name: 'alice', created_at: '2026-10-10T10:01:00.000Z', origin: 'model', document_id: null, decisions: [], kind: 'panel' },
  versions: [], can: { approve: true, facts: true, author: false }, ...x,
});
const COUNCIL: Partial<PanelContent> = {
  roles: ['lead', 'opposing', 'evidence', 'specialist'], depth: 'deep', outcome: 'answered', verified_quotes: 6,
  perspectives: [persp('lead', 'Lead counsel', 'The cap protects the Customer [1].'), persp('opposing', 'Opposing reading', 'The Supplier may end the contract.'),
    persp('evidence', 'Evidence investigator', 'The cap is stated once.'), persp('specialist', 'Specialist (commercial contracts)', 'A standard right.')],
  challenges: [{ ...persp('opposing', 'Opposing reading', 'The answer misses clause 14.2 [2].', { evidence: [ev('may terminate on notice', '14.2', 2)] }), target: 'lead', objects: true },
    { ...persp('evidence', 'Evidence investigator', 'NO OBJECTION', { evidence: [] }), target: 'lead', objects: false }],
  revised: persp('lead', 'Lead counsel', 'The cap protects the Customer, but the Supplier may terminate on notice.'),
  conclusion: 'The Supplier may terminate on a change of control; the cap is USD 250,000.',
  points: [{ text: 'Clause 14.2 needs only written notice.', roles: ['opposing', 'evidence'] }], gaps: ['Whether a sale is planned'], figures_unsupported: ['900,000'],
  matrix: [{ issue: 'Can the Supplier terminate?', positions: { lead: 'unlikely', opposing: 'yes, on notice' }, disagreement: true, summary: null, figures_unsupported: [] }],
  missing_facts: [],
};
const COUNCIL_REVIEW: PanelReview = { ...REVIEW, depth: 'deep', roles: [{ role: 'lead', label: 'Lead counsel' }, { role: 'opposing', label: 'Opposing reading' },
  { role: 'evidence', label: 'Evidence investigator' }, { role: 'specialist', label: 'Specialist (commercial contracts)' }] };
const withPanel = (roles: string[] = ['managing_partner']): Me => ({ ...me(roles), features: { intake: false, intake_links: false, panel: true } });
const AXE = { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } };

function world(v: PanelView, extra = {}, my = matter('M-101', 'MSA renewal')) {
  return mockApi({
    'GET /api/auth/me': () => [200, withPanel()],
    'GET /api/legal/matters': () => [200, { matters: [my] }],
    [`GET ${M}/panel/r1`]: () => [200, v],
    [`GET ${READER}`]: () => [200, { reader: { text: MSA_TEXT, pages: [{ number: 4, start: 0, end: MSA_TEXT.length, source: 'text' }], lines: [], clauses: [], terms: [], marks: [],
      engines: {}, version: { id: 'v1', number: 1, filename: 'MSA.txt', status: 'executed', index_status: 'indexed' } } }],
    [`GET ${M}/tasks`]: () => [200, { tasks: [] }],
    ...extra,
  });
}
const tab = (name: string) => fireEvent.click(screen.getByRole('tab', { name: new RegExp(`^${name}`) }));

describe('Council session: overview', () => {
  afterEach(() => cleanup());

  it('puts the question in the centre with each perspective around it, in its own opening words, and the Council\'s view below', async () => {
    const { container } = (world(view({}, { conclusion: 'The Supplier may terminate.', outcome: 'answered', challenges: [] })), renderLaw('/m/M-101/panel/r1'));
    const table = await screen.findByRole('region', { name: 'Perspectives' });
    expect(within(table).getByText('Can the Supplier terminate on a change of control?', { selector: '.cn-question-text' })).toBeTruthy();
    expect(within(table).getAllByRole('article').map((a) => a.getAttribute('aria-label'))).toEqual(['Deal lawyer', 'Risk and compliance', 'Litigator', 'Specialist (commercial contracts)']);
    const deal = within(table).getByRole('article', { name: 'Deal lawyer' });
    expect(deal.textContent).toContain('“The cap stays at USD 250,000.”');                    // the first verified sentence, markers gone
    expect(deal.textContent).toContain("How the documents affect the deal");
    const risk = within(table).getByRole('article', { name: 'Risk and compliance' });
    expect(risk.textContent).toContain('Says facts are missing');
    expect(risk.textContent).toContain('1 sentence without a source');
    expect(container.querySelector('img')).toBeNull();                                         // role marks, never faces
    expect(screen.getByRole('region', { name: "The Council's view" }).textContent).toContain('The Supplier may terminate.');
    expect(screen.getByText(/not independent experts/)).toBeTruthy();
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });

  it('a perspective\'s citations open its answer in the discussion', async () => {
    world(view());
    renderLaw('/m/M-101/panel/r1');
    const deal = within(await screen.findByRole('region', { name: 'Perspectives' })).getByRole('article', { name: 'Deal lawyer' });
    fireEvent.click(within(deal).getByRole('button', { name: 'Deal lawyer: 1 citation. Read the answer' }));
    expect(screen.getByRole('tab', { name: /^Discussion/ }).getAttribute('aria-selected')).toBe('true');
    expect(await screen.findByRole('article', { name: 'Deal lawyer: Initial view' })).toBeTruthy();
  });

  it('while the Council runs, each seat says whether it has answered, and nothing it wrote is shown before it is checked', async () => {
    world(view({ review: { ...REVIEW, status: 'running', progress: { step: 'views', done: ['deal', 'risk'], path: ['views'], at: null } }, artifact: null }));
    renderLaw('/m/M-101/panel/r1');
    const table = await screen.findByRole('region', { name: 'Perspectives' });
    expect(within(table).getByRole('article', { name: 'Deal lawyer' }).textContent).toContain('Answered; shown when the Council finishes');
    expect(within(table).getByRole('article', { name: 'Litigator' }).textContent).toContain('Answering now…');
    expect(within(table).getByRole('article', { name: 'Specialist (commercial contracts)' }).textContent).toContain('Waiting for its turn');
    expect(screen.getByRole('status').textContent).toContain('The steps below follow the run');
  });
});

describe('Council session: the discussion flow', () => {
  afterEach(() => cleanup());
  const steps = () => within(screen.getByRole('navigation', { name: 'Discussion flow' })).getAllByRole('listitem');

  it('follows the run\'s recorded step: answers counted, a focused run skips the responses', async () => {
    world(view({ review: { ...REVIEW, status: 'running', progress: { step: 'views', done: ['deal', 'risk'], path: ['views'], at: null } }, artifact: null }));
    renderLaw('/m/M-101/panel/r1');
    await screen.findByRole('navigation', { name: 'Discussion flow' });
    expect(steps().map((s) => s.getAttribute('data-state'))).toEqual(['done', 'current', 'next', 'skipped', 'next']);
    expect(steps()[1].getAttribute('aria-current')).toBe('step');
    expect(steps()[1].textContent).toContain('2 of 4 answered');
    expect(steps()[3].textContent).toContain('Not part of a focused run');
  });

  it('a step the run passed over is shown as skipped, with the reason', () => {
    const deep = { ...REVIEW, depth: 'deep' as const, status: 'running' as const, progress: { step: 'chair' as const, done: [], path: ['views' as const, 'challenge' as const, 'chair' as const], at: null } };
    expect(councilFlow(deep).map((s) => [s.state, s.detail])).toEqual([
      ['done', 'Asked by alice'], ['done', '4 of 4 answered'], ['done', 'Done'], ['skipped', 'Not needed: no objection cited anything new'], ['current', 'Comparing the verified answers']]);
    const lacking = { ...deep, progress: { ...deep.progress, path: ['views' as const, 'chair' as const] } };
    expect(councilFlow(lacking)[2]).toMatchObject({ state: 'skipped', detail: 'Skipped: most perspectives said facts were missing' });
    const failed = { ...deep, status: 'failed' as const, progress: { step: 'challenge' as const, done: [], path: ['views' as const, 'challenge' as const], at: null } };
    expect(councilFlow(failed).map((s) => s.state)).toEqual(['done', 'done', 'failed', 'next', 'next']);
  });

  it('a finished run reads its steps from the stored result', () => {
    const council = { ...CONTENT, ...COUNCIL } as PanelContent;
    expect(councilFlow(COUNCIL_REVIEW, council).map((s) => [s.label, s.state, s.detail])).toEqual([
      ['Question', 'done', 'Asked by alice'], ['Initial views', 'done', '4 of 4 answered'], ['Challenge', 'done', '1 objection'],
      ['Responses', 'done', 'Lead counsel replied'], ['Synthesis', 'done', 'Conclusion drawn']]);
    expect(councilFlow({ ...REVIEW, status: 'needs_facts' }, { ...CONTENT, stopped: true, outcome: 'stopped', challenges: [] }).slice(2).map((s) => s.state))
      .toEqual(['skipped', 'skipped', 'stopped']);
    expect(councilFlow(REVIEW, { ...CONTENT, outcome: 'inconclusive', challenges: [] })[2].detail).toBe('Skipped: fewer than two verified quotes');
  });

  it('each step opens its part of the discussion; the views are keyboard tabs', async () => {
    world(view({ review: COUNCIL_REVIEW }, COUNCIL));
    renderLaw('/m/M-101/panel/r1');
    const flow = await screen.findByRole('navigation', { name: 'Discussion flow' });
    fireEvent.click(within(flow).getByRole('button', { name: /^Challenge: done/ }));
    expect(screen.getByRole('tab', { name: /^Discussion/ }).getAttribute('aria-selected')).toBe('true');
    const overview = screen.getByRole('tab', { name: /^Overview/ });
    overview.focus();
    fireEvent.keyDown(screen.getByRole('tab', { name: /^Discussion/ }), { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: /^Evidence/ }).getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: /^Evidence/ }));
    fireEvent.keyDown(document.activeElement!, { key: 'End' });
    expect(screen.getByRole('tab', { name: /^Result/ }).getAttribute('aria-selected')).toBe('true');
  });
});

describe('Council session: discussion and evidence', () => {
  afterEach(() => cleanup());

  it('shows the answers, the challenge round, the reply and the chair in order, each citation opening the passage', async () => {
    const { container } = (world(view({ review: COUNCIL_REVIEW }, COUNCIL)), renderLaw('/m/M-101/panel/r1?view=discussion'));
    const views = await screen.findByRole('region', { name: 'Initial views' });
    const lead = within(views).getByRole('article', { name: 'Lead counsel: Initial view' });
    expect(within(screen.getByRole('region', { name: 'Challenge' })).getByRole('article', { name: 'Opposing reading: Objection to the Lead counsel answer' })).toBeTruthy();
    expect(within(screen.getByRole('region', { name: 'Challenge' })).getByRole('article', { name: /Evidence investigator: No objection to the Lead counsel answer/ })).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Responses' }).textContent).toContain('but the Supplier may terminate on notice');
    const chair = within(screen.getByRole('region', { name: 'Synthesis' })).getByRole('article', { name: 'The chair: synthesis' });
    expect(chair.textContent).toContain('The Supplier may terminate on a change of control');
    expect(within(chair).getByRole('list', { name: 'What matters most' }).textContent).toContain('Opposing reading');
    expect(chair.textContent).toContain('Check: 900,000 is in no answer');
    const regions = screen.getAllByRole('region').map((r) => r.getAttribute('aria-label') ?? r.getAttribute('aria-labelledby'));
    expect(regions.length).toBeGreaterThan(3);
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
    const chip = within(lead).getByRole('button', { name: 'Lead counsel, source 1: MSA.txt · clause 9.1 · p. 4. Open the passage' });
    chip.focus();
    fireEvent.click(chip);
    const dialog = await screen.findByRole('dialog', { name: 'Cited passage' });
    const passage = await within(dialog).findByLabelText('Passage');
    expect(passage.querySelector('mark')?.textContent).toBe('limited to USD 250,000');
    expect(passage.textContent).toContain('Each party\'s total liability is');
    expect(dialog.textContent).toContain('The quoted words were found in this passage, word for word.');
    expect(dialog.textContent).toContain('Lead counsel (initial view)');
    expect(within(dialog).getByRole('link', { name: 'Open at the quote' }).getAttribute('href')).toBe(`/m/M-101/documents/d1/read?v=v1&at=${at('limited to USD 250,000').join('-')}`);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }));
    await waitFor(() => expect(document.activeElement).toBe(chip));                       // focus goes back to the chip
  });

  it('marks what an answer does not support', async () => {
    world(view());
    renderLaw('/m/M-101/panel/r1?view=discussion');
    const risk = await screen.findByRole('article', { name: 'Risk and compliance: Initial view' });
    const uncited = risk.querySelector('details')!;
    expect(uncited.open).toBe(false);                                                   // collapsed, as on answer cards
    expect(within(uncited).getByText('Cites nothing: 1 sentence')).toBeTruthy();
    expect(uncited.textContent).toContain('“This is a real risk.”');
    expect(risk.textContent).toContain('Says it lacks: whether a sale is planned');
  });

  it('lists each cited document with its places, and which perspective used each', async () => {
    const { container } = (world(view({ review: COUNCIL_REVIEW }, COUNCIL)), renderLaw('/m/M-101/panel/r1?view=evidence'));
    const docs = await screen.findByRole('list', { name: 'Documents cited' });
    const msa = within(docs).getByRole('article', { name: 'MSA.txt' });
    const places = within(msa).getAllByRole('listitem').filter((li) => li.classList.contains('cn-place'));
    expect(places.map((p) => within(p).getByText(/clause/).textContent)).toEqual(['clause 9.1 · p. 4', 'clause 14.2 · p. 4']);
    expect(within(places[0]).getByRole('list', { name: 'Cited by' }).textContent).toContain('Lead counsel');
    expect(within(places[1]).getByRole('list', { name: 'Cited by' }).textContent).toContain('Opposing reading (objection)');
    expect(screen.getByRole('region', { name: 'Evidence in focus' }).textContent).toContain('1 document cited at 2 places');
    fireEvent.click(within(places[1]).getByRole('button', { name: 'Show the passage' }));
    expect((await within(await screen.findByRole('dialog', { name: 'Cited passage' })).findByLabelText('Passage')).querySelector('mark')?.textContent).toBe('may terminate on notice');
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });

  it('a quote inside another quote of the same version is one place, cited by both', () => {
    const c = { ...CONTENT, perspectives: [persp('deal', 'Deal lawyer', 'x', { evidence: [ev('limited to USD 250,000 in any year', '9.1')] }),
      persp('risk', 'Risk and compliance', 'y', { evidence: [ev('limited to USD 250,000', '9.1')] })] };
    const out = evidenceInFocus(c);
    expect(out.docs[0].places).toHaveLength(1);
    expect(out.docs[0].places[0].e.quote).toBe('limited to USD 250,000 in any year');
    expect(out.docs[0].places[0].by.map((u) => u.label)).toEqual(['Deal lawyer', 'Risk and compliance']);
  });

  it('a quote that was not found is counted, never listed as evidence', () => {
    const c = { ...CONTENT, perspectives: [persp('deal', 'Deal lawyer', 'x', { evidence: [ev('limited to USD 250,000', '9.1'), ev('nowhere', '1.1', 2, { verification: 'failed' })] })] };
    const out = evidenceInFocus(c);
    expect(out.docs[0].places).toHaveLength(1);
    expect(out.unverified).toBe(1);
  });
});

describe('Council session: the result', () => {
  afterEach(() => cleanup());

  it('gives the view, what matters most, where they differ, what is still missing, steps for the plan and the partner review', async () => {
    const { container } = (world(view({ review: COUNCIL_REVIEW }, { ...COUNCIL, missing_facts: ['whether a sale is planned'],
      perspectives: COUNCIL.perspectives!.map((p) => (p.role === 'opposing' ? { ...p, missing_facts: ['whether a sale is planned'] } : p)) })), renderLaw('/m/M-101/panel/r1?view=result'));
    const conclusion = await screen.findByRole('region', { name: 'Conclusion' });
    expect(conclusion.textContent).toContain('The Supplier may terminate on a change of control');
    expect(within(conclusion).getByRole('list', { name: 'What matters most' }).textContent).toContain('Clause 14.2 needs only written notice.Opposing readingEvidence investigator');
    const differ = screen.getByRole('region', { name: 'Where they differ' });
    expect(differ.textContent).toContain('Can the Supplier terminate?');
    expect(differ.textContent).toContain('Opposing reading: yes, on notice');
    expect(differ.textContent).toContain('Evidence investigator: not addressed');
    const head = within(within(differ).getByRole('table', { name: 'Disagreement matrix', hidden: true })).getAllByRole('columnheader', { hidden: true }).map((h) => h.textContent);
    expect(head).toEqual(['Issue', 'Lead counsel', 'Opposing reading', 'Evidence investigator', 'Specialist (commercial contracts)']);
    const open = screen.getByRole('region', { name: 'Facts still missing' });
    expect(open.textContent).toContain('Whether a sale is planned');
    expect(open.textContent).toContain('whether a sale is planned — Opposing reading');
    expect(screen.getByRole('group', { name: 'Review this panel review' })).toBeTruthy();
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });

  it('a next step is edited by the person and added to the matter\'s plan as theirs', async () => {
    const tasks: { title: string }[] = [];
    const calls = world(view({ review: COUNCIL_REVIEW }, COUNCIL), {
      [`GET ${M}/tasks`]: () => [200, { tasks: tasks.map((t, i) => ({ id: `t${i}`, matter_id: 'id-M-101', title: t.title, kind: 'task', status: 'open', source: 'person',
        created_by: 'usr_alice', created_by_name: 'alice', created_at: '', updated_at: '', version: 1, overdue: false })) }],
      [`POST ${M}/tasks`]: (b: { title: string }) => { tasks.push(b); return [200, { task: { id: 't0', title: b.title } }]; },
    });
    renderLaw('/m/M-101/panel/r1?view=result');
    const steps = await screen.findByRole('region', { name: 'Next steps' });
    const first = within(steps).getByLabelText('Step 1') as HTMLInputElement;
    expect(first.value).toBe('Confirm whether a sale is planned');
    expect(within(steps).getByLabelText('Step 2')).toHaveProperty('value', 'Check the figure 900,000 against the documents');
    fireEvent.change(first, { target: { value: 'Ask the client whether a sale is planned' } });
    fireEvent.click(within(steps).getByRole('button', { name: 'Add step 1 to the plan' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === `${M}/tasks`)?.body).toEqual({ title: 'Ask the client whether a sale is planned', kind: 'task' }));
    expect(await within(steps).findByText('In the plan')).toBeTruthy();
  });

  it('someone who may not plan sees the steps but adds none', async () => {
    world(view({ review: COUNCIL_REVIEW }, COUNCIL), {}, matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'reviewer' }));
    renderLaw('/m/M-101/panel/r1?view=result');
    const steps = await screen.findByRole('region', { name: 'Next steps' });
    expect(steps.textContent).toContain('Confirm whether a sale is planned');
    expect(within(steps).queryByRole('button')).toBeNull();
    expect(steps.textContent).toContain("The matter's lawyers and paralegals add steps to its plan.");
  });

  it('steps come by rule from what the run left open, once each', () => {
    expect(stepIdeas({ ...CONTENT, gaps: ['Whether a sale is planned'], missing_facts: ['whether a sale is planned', 'the notice date'], figures_unsupported: ['900,000'] })
      .map((x) => [x.text, x.from])).toEqual([
      ['Confirm whether a sale is planned', 'Not established by the documents'],
      ['Confirm the notice date', 'Asked for by the chair'],
      ['Check the figure 900,000 against the documents', "In the chair's view, but in no answer"]]);
    expect(openingLine(persp('deal', 'Deal lawyer', 'Uncited first. The cap is USD 250,000 [1]. More.', { unsupported: ['Uncited first.'] }))).toBe('The cap is USD 250,000.');
  });

  it('an inconclusive Council run says that no conclusion was drawn', async () => {
    world(view({}, { outcome: 'inconclusive', conclusion: null, points: [], challenges: [] }));
    renderLaw('/m/M-101/panel/r1?view=result');
    expect((await screen.findByRole('region', { name: 'Conclusion' })).textContent).toContain('No conclusion');
  });

  it('a lawyer who is not a partner sees that a partner decides', async () => {
    world(view({ can: { approve: false, facts: true, author: false } }));
    renderLaw('/m/M-101/panel/r1?view=result');
    expect(await screen.findByText(/Waiting for a partner/)).toBeTruthy();
    expect(screen.queryByRole('group', { name: 'Review this panel review' })).toBeNull();
  });
});

describe('Council session: missing facts', () => {
  afterEach(() => cleanup());

  it('a stopped Council asks for the missing facts and runs again; nothing can be approved meanwhile', async () => {
    const calls = world(view({ review: { ...REVIEW, status: 'needs_facts' } }, { stopped: true, outcome: 'stopped', missing_facts: ['whether the sale has been announced', 'the notice date'] }),
      { [`POST ${M}/panel/r1/facts`]: () => [200, { review: { id: 'r1', status: 'queued' } }] });
    renderLaw('/m/M-101/panel/r1');
    fireEvent.click(await screen.findByRole('button', { name: /^Answer them/ }));
    const stop = await screen.findByRole('region', { name: 'Facts needed' });
    expect(screen.queryByRole('group', { name: 'Review this panel review' })).toBeNull();
    fireEvent.change(within(stop).getByLabelText('Whether the sale has been announced'), { target: { value: 'Not yet; planned for March.' } });
    fireEvent.click(within(stop).getByRole('button', { name: 'Give these facts and run again' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/panel/r1/facts`)?.body).toEqual({
      facts: [{ question: 'whether the sale has been announced', answer: 'Not yet; planned for March.' }] }));
  });

  it('a stopped Council can continue without the facts, with a reason that is recorded', async () => {
    const calls = world(view({ review: { ...REVIEW, status: 'needs_facts' } }, { stopped: true, missing_facts: ['the notice date'] }),
      { [`POST ${M}/panel/r1/proceed`]: () => [200, { review: { id: 'r1', status: 'done' } }] });
    renderLaw('/m/M-101/panel/r1?view=result');
    const stop = await screen.findByRole('region', { name: 'Facts needed' });
    const go = within(stop).getByRole('button', { name: 'Continue without these facts' }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);                                                    // a reason first
    fireEvent.change(within(stop).getByLabelText('Or continue without them, because'), { target: { value: 'The client cannot say yet.' } });
    fireEvent.click(go);
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/panel/r1/proceed`)?.body).toEqual({ reason: 'The client cannot say yet.' }));
  });

  it('a Council that continued without facts says so, with who and why', async () => {
    world(view({}, { proceeded_without: { facts: ['the notice date'], by: 'alice', reason: 'The client cannot say yet.', at: '2026-10-10T11:00:00.000Z' } }));
    renderLaw('/m/M-101/panel/r1?view=result');
    const note = await screen.findByRole('region', { name: 'Continued without facts' });
    expect(note.textContent).toContain('alice chose to continue');
    expect(note.textContent).toContain('Left open: the notice date.');
  });
});

describe('Council home', () => {
  afterEach(() => cleanup());

  it('questions found in the records can be convened with the chosen area and depth, or dismissed with a reason', async () => {
    const question = { id: 'q1', title: 'How does Amendment No. 1 change clause 4.2 of the Services Agreement?', reason: 'Amendment No. 1 says clause 4.2 is deleted and replaced.',
      priority: 'high', detector: 'amended', status: 'suggested', created_at: '2026-11-05T10:00:00.000Z',
      sources: [{ document_title: 'Amendment No. 1', href: '/m/M-101/documents/d2/read?v=v2&at=10-90' }] };
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withPanel(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/panel`]: () => [200, { reviews: [], specialisms: { commercial: 'commercial contracts', employment: 'employment law' }, can: { ask: true, approve: false },
        discovery: true, questions: [question, { ...question, id: 'q2', title: 'What must be ready before 2026-11-15 under clause 4.2?', priority: 'medium', detector: 'deadline' }], more_questions: 1 }],
      [`GET ${M}/documents`]: () => [200, { documents: [] }],
      [`POST ${M}/council/questions/q1/convene`]: () => [200, { review: { id: 'r1', status: 'queued' }, question: { id: 'q1', status: 'convened' } }],
      [`POST ${M}/council/questions/q2/dismiss`]: () => [200, { question: { id: 'q2', status: 'dismissed' } }],
      [`POST ${M}/council/discover`]: () => [200, { queued: true }],
      [`GET ${M}/panel/r1`]: () => [200, view({ review: { ...REVIEW, status: 'running' }, artifact: null })],
    });
    const { container } = renderLaw('/m/M-101/panel');
    const found = await screen.findByRole('region', { name: 'Suggested questions' });
    const first = within(found).getByRole('listitem', { name: question.title });
    expect(first.textContent).toContain('High priority');
    expect(within(first).getByRole('link', { name: 'Amendment No. 1' }).getAttribute('href')).toBe('/m/M-101/documents/d2/read?v=v2&at=10-90');
    expect(found.textContent).toContain('1 more, shown when one of these is convened or dismissed.');
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
    const second = within(found).getByRole('listitem', { name: 'What must be ready before 2026-11-15 under clause 4.2?' });
    fireEvent.click(within(second).getByRole('button', { name: 'Dismiss…' }));
    const go = within(second).getByRole('button', { name: 'Dismiss' }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);                                                    // a reason first
    fireEvent.change(within(second).getByLabelText('Not worth asking, because'), { target: { value: 'Already in the delivery plan.' } });
    fireEvent.click(go);
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/council/questions/q2/dismiss`)?.body).toEqual({ reason: 'Already in the delivery plan.' }));
    fireEvent.click(within(found).getByRole('button', { name: 'Look again' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === `${M}/council/discover`)).toBe(true));
    fireEvent.change(screen.getByLabelText('Depth'), { target: { value: 'deep' } });
    fireEvent.click(within(first).getByRole('button', { name: 'Convene the Council' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/council/questions/q1/convene`)?.body).toEqual({ specialist: 'commercial', depth: 'deep' }));
    expect(await screen.findByRole('status')).toBeTruthy();                            // the session page, working
  });

  it('asking the Council sends the question and the specialist area, and opens the session', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withPanel(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/panel`]: () => [200, { reviews: [], specialisms: { commercial: 'commercial contracts', employment: 'employment law' }, can: { ask: true, approve: false } }],
      [`GET ${M}/documents`]: () => [200, { documents: [{ id: 'd1', title: 'Service Agreement.txt', current_version: { id: 'v1', number: 1, index_status: 'indexed' } },
        { id: 'd2', title: 'Handbook.txt', current_version: { id: 'v2', number: 1, index_status: 'indexed' } }] }],
      [`POST ${M}/panel`]: () => [200, { review: { id: 'r1', status: 'queued' } }],
      [`GET ${M}/panel/r1`]: () => [200, view({ review: { ...REVIEW, status: 'running' }, artifact: null })],
    });
    renderLaw('/m/M-101/panel');
    const form = await screen.findByRole('form', { name: 'Ask the Council' });
    fireEvent.change(within(form).getByLabelText('Question'), { target: { value: 'Can the employer change the bonus scheme unilaterally?' } });
    fireEvent.change(within(form).getByLabelText('Specialist area'), { target: { value: 'employment' } });
    fireEvent.click(await within(form).findByLabelText('Service Agreement.txt'));
    fireEvent.change(within(form).getByLabelText('Depth'), { target: { value: 'deep' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Ask the Council' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === `${M}/panel`)?.body).toEqual({
      question: 'Can the employer change the bonus scheme unilaterally?', specialist: 'employment', documents: ['d1'], depth: 'deep' }));
    expect(await screen.findByRole('status')).toBeTruthy();                                // the Council is working
  });

  it('lists the sessions with who sat on each and the step a running one is on', async () => {
    mockApi({
      'GET /api/auth/me': () => [200, withPanel(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${M}/panel`]: () => [200, { reviews: [{ ...REVIEW, status: 'running', progress: { step: 'challenge', done: [], path: ['views', 'challenge'], at: null } },
        { ...REVIEW, id: 'r2', question: 'Is the fee binding?', approval: 'approved', version: 2 }], specialisms: { commercial: 'commercial contracts' }, can: { ask: false, approve: false } }],
      [`GET ${M}/documents`]: () => [200, { documents: [] }],
    });
    renderLaw('/m/M-101/panel');
    const list = await screen.findByRole('list', { name: 'Council sessions' });
    const running = within(list).getByRole('listitem', { name: CONTENT.question });
    expect(running.textContent).toContain('Step 3 of 5 · Challenge');
    expect(within(running).getByRole('img', { name: 'Perspectives: Deal lawyer, Risk and compliance, Litigator, Specialist (commercial contracts)' })).toBeTruthy();
    expect(within(list).getByRole('listitem', { name: 'Is the fee binding?' }).textContent).toContain('Approved by a partner');
    expect(screen.queryByRole('form', { name: 'Ask the Council' })).toBeNull();
  });
});
