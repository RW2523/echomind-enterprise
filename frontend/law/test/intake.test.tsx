import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Intake, IntakeLinkPage, IntakeScript, IntakeView, Me } from '../api/types';
import { matter, me, mockApi, openAccountMenu, renderLaw } from './harness';

/** S4.1 — intake assistant with conflict screening (OPT-01; D-054). */
const SCRIPT: IntakeScript = {
  version: 'intake-script-1',
  questions: [
    { key: 'your_name', label: 'Your name', kind: 'text', required: true, max: 120, ask: 'What is your full name?', thanks: 'Thank you.' },
    { key: 'email', label: 'Email', kind: 'email', required: false, max: 160, ask: 'What email address should we use?', thanks: 'Noted.' },
    { key: 'phone', label: 'Telephone', kind: 'phone', required: false, max: 40, ask: 'And a telephone number?', thanks: 'Noted.' },
    { key: 'matter_type', label: 'Kind of matter', kind: 'choice', required: true, choices: { commercial: 'Commercial contracts', dispute: 'A dispute' },
      ask: 'Which of these best describes what you need help with?', thanks: 'Thank you.' },
    { key: 'other_parties', label: 'Other parties', kind: 'names', required: false, max: 160, max_items: 10, ask: 'Who else is involved?', thanks: 'We check these names before we act.' },
  ],
};
const DISCLAIMER = { version: 'intake-disclaimer-1', text: 'These questions collect administrative details only. This is not legal advice.' };
const BASE = '/api/legal/intake';
const intake = (x: Partial<Intake> = {}): Intake => ({
  id: 'i1', reference: 'INT-2026-0001', channel: 'phone', status: 'submitted', created_at: '2026-10-10T09:00:00.000Z', created_by: 'rita', updated_at: null,
  version: 3, disclaimer: { ...DISCLAIMER, accepted_at: '2026-10-10T09:01:00.000Z' }, script: SCRIPT,
  answers: { your_name: 'Hana Kowalczyk', email: 'hana@kestrel.example', phone: '', matter_type: 'commercial', other_parties: ['Orrin Shipping Ltd'] },
  screening: { result: 'possible_conflict', message: 'Possible conflict: refer to the conflicts team.', decision: null },
  consent: null, recording: null, replies: [], events: [{ action: 'created', actor: 'rita', at: '2026-10-10T09:00:00.000Z' }],
  matter_number: null, link: null, ...x,
});
const CAN = { manage: true, approve: false, conflicts: false, purge: false, convert: false, links: true };
const withFeatures = (roles: string[]): Me => ({ ...me(roles), features: { intake: true, intake_links: true } });

function world(view: IntakeView, roles: string[] = ['intake'], can = CAN, extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, withFeatures(roles)],
    'GET /api/legal/matters': () => [200, { matters: [] }],
    [`GET ${BASE}`]: () => [200, { intakes: [{ id: 'i1', reference: 'INT-2026-0001', channel: 'phone', status: view.intake.status, name: 'Hana Kowalczyk', organisation: null,
      screening: view.intake.screening, created_at: '2026-10-10T09:00:00.000Z', updated_at: null }], can, templates: [], script: SCRIPT }],
    [`GET ${BASE}/i1`]: () => [200, view],
    ...extra,
  });
}

describe('Intake (S4.1)', () => {
  afterEach(() => cleanup());

  it('front-desk staff see only whether to refer: no matches, no matter, no client', async () => {
    world({ intake: intake() });
    renderLaw('/enquiries');
    const row = await screen.findByRole('row', { name: /INT-2026-0001/ });
    expect(within(row).getByText('Refer to the conflicts team')).toBeTruthy();
    expect(within(row).getByText('Awaiting conflicts check')).toBeTruthy();
    expect(within(await openAccountMenu()).getByRole('menuitem', { name: 'Enquiries' })).toBeTruthy();   // the menu item, with the feature on
    cleanup();
    const { container } = renderLaw('/enquiries/i1');
    await screen.findByRole('region', { name: 'Answers' });
    expect(screen.queryByRole('region', { name: 'Conflicts check' })).toBeNull();
    expect(document.body.textContent).toContain('Possible conflict: refer to the conflicts team.');
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('the conflicts team sees the matches and decides, with a note', async () => {
    const calls = world({ intake: intake(), matches: [{ searched: 'Orrin Shipping Ltd', matched: 'Orrin Shipping Ltd', kind: 'adverse', score: 1, client: 'Beta Corp', matter_number: 'M-202' }] },
      ['conflicts'], { ...CAN, manage: false, conflicts: true }, { [`POST ${BASE}/i1/decision`]: () => [200, { intake: intake({ status: 'conflict' }) }] });
    renderLaw('/enquiries/i1');
    const panel = await screen.findByRole('region', { name: 'Conflicts check' });
    const match = within(panel).getByRole('row', { name: /Orrin Shipping Ltd/ });
    expect(match.textContent).toContain('Adverse party');
    expect(match.textContent).toContain('M-202');
    fireEvent.change(within(panel).getByLabelText('Note for the record'), { target: { value: 'Acting for the other side in M-202.' } });
    fireEvent.click(within(panel).getByRole('button', { name: 'Confirm a conflict' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/i1/decision`)).toBe(true));
    expect(calls.find((c) => c.path === `${BASE}/i1/decision`)!.body).toEqual({ decision: 'conflict', note: 'Acting for the other side in M-202.' });
  });

  it('the disclaimer comes first, then the script with the words to say', async () => {
    const open = intake({ status: 'open', disclaimer: { ...DISCLAIMER, accepted_at: null }, answers: {}, screening: { result: 'not_run', message: '', decision: null } });
    const calls = world({ intake: open }, ['intake'], CAN, { [`POST ${BASE}/i1/disclaimer`]: () => [200, { intake: open }] });
    renderLaw('/enquiries/i1');
    expect(await screen.findByText(DISCLAIMER.text)).toBeTruthy();
    expect(screen.queryByRole('form', { name: 'Intake questions' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'The enquirer accepts it' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/i1/disclaimer`)?.body).toEqual({ version: DISCLAIMER.version }));
  });

  it('staff submit the scripted answers; the browser asks for a way to reply first', async () => {
    const open = intake({ status: 'open', answers: {}, screening: { result: 'not_run', message: '', decision: null } });
    const calls = world({ intake: open }, ['intake'], CAN, { [`POST ${BASE}/i1/submit`]: () => [200, { intake: intake() }] });
    renderLaw('/enquiries/i1');
    const form = await screen.findByRole('form', { name: 'Intake questions' });
    expect(within(form).getByText('Ask: “What is your full name?”')).toBeTruthy();
    fireEvent.change(within(form).getByLabelText(/Your name/), { target: { value: 'Hana Kowalczyk' } });
    fireEvent.change(within(form).getByLabelText(/Kind of matter/), { target: { value: 'commercial' } });
    fireEvent.click(within(form).getByRole('button', { name: /Submit answers/ }));
    expect((await within(form).findByRole('alert')).textContent).toBe('Give an email address or a telephone number.');
    fireEvent.change(within(form).getByLabelText(/Email/), { target: { value: 'hana@kestrel.example' } });
    fireEvent.change(within(form).getByLabelText(/Other parties/), { target: { value: 'Orrin Shipping Ltd\n\n' } });
    fireEvent.click(within(form).getByRole('button', { name: /Submit answers/ }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/i1/submit`)).toBe(true));
    expect(calls.find((c) => c.path === `${BASE}/i1/submit`)!.body.answers).toMatchObject({ your_name: 'Hana Kowalczyk', email: 'hana@kestrel.example',
      matter_type: 'commercial', other_parties: ['Orrin Shipping Ltd'] });
  });

  it('a call recording waits for the caller\'s consent', async () => {
    const calls = world({ intake: intake() }, ['intake'], CAN, { [`POST ${BASE}/i1/consent`]: () => [200, { intake: intake({ consent: { method: 'verbal', recorded_by: 'rita', recorded_at: '2026-10-10T09:02:00.000Z' } }) }] });
    renderLaw('/enquiries/i1');
    const rec = await screen.findByRole('region', { name: 'Call recording' });
    expect(within(rec).queryByText(/Attach recording/)).toBeNull();
    fireEvent.click(within(rec).getByRole('button', { name: 'Record consent' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/i1/consent`)?.body).toEqual({ method: 'verbal' }));
  });

  it('a lawyer approves a reply written by someone else, after reading it', async () => {
    const draft = { id: 'r1', template: 'acknowledge' as const, body: 'Dear Hana Kowalczyk,\n\nThank you for contacting us.', body_sha256: 'b'.repeat(64), status: 'draft' as const,
      drafted_by: 'rita', drafted_at: '2026-10-10T09:05:00.000Z', approved_by: null, approved_at: null };
    const calls = world({ intake: intake({ replies: [draft] }) }, ['attorney'], { ...CAN, approve: true, convert: true },
      { [`POST ${BASE}/i1/replies/r1/approve`]: () => [200, { intake: intake() }] });
    renderLaw('/enquiries/i1');
    const replies = await screen.findByRole('region', { name: 'Replies' });
    expect(within(replies).getByText('Draft by rita — awaiting a lawyer')).toBeTruthy();
    fireEvent.click(within(replies).getByRole('button', { name: 'Approve and release' }));
    const dialog = await screen.findByRole('dialog');
    const ok = within(dialog).getByRole('button', { name: 'Approve and release' }) as HTMLButtonElement;
    expect(ok.disabled).toBe(true);
    fireEvent.click(within(dialog).getByRole('checkbox'));
    fireEvent.click(ok);
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/i1/replies/r1/approve`)?.body).toEqual({ body_sha256: 'b'.repeat(64) }));
    expect(screen.queryByRole('region', { name: 'Open a matter' })).toBeNull();               // not cleared yet
  });

  it('a cleared enquiry becomes a matter, and the page then links to it', async () => {
    let converted = false;
    const cleared = intake({ status: 'cleared', screening: { result: 'no_match', message: 'No match found.', decision: 'cleared' } });
    const done = intake({ status: 'converted', matter_number: 'M-301', screening: cleared.screening });
    const calls = mockApi({
      'GET /api/auth/me': () => [200, withFeatures(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      [`GET ${BASE}`]: () => [200, { intakes: [], can: { ...CAN, approve: true, convert: true }, templates: [], script: SCRIPT }],
      [`GET ${BASE}/i1`]: () => [200, { intake: converted ? done : cleared }],
      [`POST ${BASE}/i1/convert`]: () => { converted = true; return [200, { intake: done, matter: { id: 'm1', number: 'M-301' } }]; },
    });
    renderLaw('/enquiries/i1');
    fireEvent.click(within(await screen.findByRole('region', { name: 'Open a matter' })).getByRole('button', { name: 'Open a matter' }));
    const dialog = await screen.findByRole('dialog');
    expect((within(dialog).getByLabelText('Client') as HTMLInputElement).value).toBe('Hana Kowalczyk');
    fireEvent.change(within(dialog).getByLabelText('Matter number'), { target: { value: 'M-301' } });
    fireEvent.change(within(dialog).getByLabelText('Matter title'), { target: { value: 'Distribution renewal' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Open the matter' }));
    const opened = await screen.findByRole('region', { name: 'Matter opened' });
    expect(calls.filter((c) => c.path === '/api/legal/matters').length).toBeGreaterThan(1);    // the new matter is in the list it opens from
    expect(within(opened).getByRole('link', { name: 'M-301' }).getAttribute('href')).toBe('/m/M-301/overview');
    expect(calls.find((c) => c.path === `${BASE}/i1/convert`)!.body).toEqual({ client_name: 'Hana Kowalczyk', client_code: undefined, matter_number: 'M-301', matter_title: 'Distribution renewal' });
  });

  it('the enquirer\'s link: the disclaimer, then one question at a time, then thank you', async () => {
    const page = (x: Partial<IntakeLinkPage> = {}): IntakeLinkPage => ({ status: 'invited', reference: 'INT-2026-0002', disclaimer: { ...DISCLAIMER, accepted: false },
      script: SCRIPT, reply: null, ...x });
    const T = 'T'.repeat(43);
    const calls = mockApi({
      [`GET /api/intake-link/${T}`]: () => [200, page()],
      [`POST /api/intake-link/${T}/accept`]: () => [200, page({ disclaimer: { ...DISCLAIMER, accepted: true } })],
      [`POST /api/intake-link/${T}/submit`]: () => [200, page({ status: 'submitted', disclaimer: { ...DISCLAIMER, accepted: true } })],
    });
    const { container } = renderLaw(`/intake/${T}`);
    expect(await screen.findByText(DISCLAIMER.text)).toBeTruthy();
    expect(calls.some((c) => c.path === '/api/auth/me')).toBe(false);                         // no account, no session
    fireEvent.click(screen.getByRole('button', { name: 'I understand, continue' }));
    const q1 = await screen.findByRole('form', { name: 'Question 1 of 5' });
    fireEvent.click(within(q1).getByRole('button', { name: 'Next' }));
    expect((await within(q1).findByRole('alert')).textContent).toBe('Your name: this is needed.');
    fireEvent.change(within(q1).getByLabelText('What is your full name?'), { target: { value: 'Hana Kowalczyk' } });
    fireEvent.click(within(q1).getByRole('button', { name: 'Next' }));
    const q2 = await screen.findByRole('form', { name: 'Question 2 of 5' });
    expect(screen.getByRole('list', { name: 'Your answers so far' }).textContent).toContain('Hana Kowalczyk');
    fireEvent.change(within(q2).getByLabelText(/email address/), { target: { value: 'hana@kestrel.example' } });
    fireEvent.click(within(q2).getByRole('button', { name: 'Next' }));
    fireEvent.click(within(await screen.findByRole('form', { name: 'Question 3 of 5' })).getByRole('button', { name: 'Next' }));
    const q4 = await screen.findByRole('form', { name: 'Question 4 of 5' });
    fireEvent.change(within(q4).getByLabelText(/best describes/), { target: { value: 'dispute' } });
    fireEvent.click(within(q4).getByRole('button', { name: 'Next' }));
    fireEvent.click(within(await screen.findByRole('form', { name: 'Question 5 of 5' })).getByRole('button', { name: 'Next' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Send my answers' }));
    expect(await screen.findByText('Thank you')).toBeTruthy();
    expect(calls.find((c) => c.path.endsWith('/submit'))!.body.answers).toMatchObject({ your_name: 'Hana Kowalczyk', matter_type: 'dispute', other_parties: [] });
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('an expired link says so and offers nothing else', async () => {
    mockApi({ ['GET /api/intake-link/' + 'X'.repeat(43)]: () => [404, { detail: 'Not Found' }] });
    renderLaw(`/intake/${'X'.repeat(43)}`);
    expect((await screen.findByRole('alert')).textContent).toContain('not valid any more');
  });

  it('a matter\'s parties: its lawyers add and remove them with a reason', async () => {
    const P = '/api/legal/matters/id-M-101/parties';
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      [`GET ${P}`]: () => [200, { parties: [{ id: 'p1', name: 'Northwind Logistics Ltd', role: 'adverse', added_by: 'alice', added_at: '2026-10-10T09:00:00.000Z' }], can_manage: true }],
      [`POST ${P}`]: () => [200, { party: {} }],
      [`POST ${P}/p1/remove`]: () => [200, { parties: [] }],
    });
    renderLaw('/m/M-101/overview');
    const region = await screen.findByRole('region', { name: 'Parties' });
    expect(within(region).getByText('Northwind Logistics Ltd')).toBeTruthy();
    fireEvent.change(within(region).getByLabelText('Party name'), { target: { value: 'Orrin Shipping Ltd' } });
    fireEvent.change(within(region).getByLabelText('As'), { target: { value: 'related' } });
    fireEvent.click(within(region).getByRole('button', { name: 'Add party' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === P)?.body).toEqual({ name: 'Orrin Shipping Ltd', role: 'related' }));
    fireEvent.click(within(region).getByRole('button', { name: 'Remove Northwind Logistics Ltd' }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.change(within(dialog).getByLabelText('Reason'), { target: { value: 'Added in error' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${P}/p1/remove`)?.body).toEqual({ reason: 'Added in error' }));
  });

  it('without the feature, there is no Enquiries item', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['intake'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/home');
    expect(within(await openAccountMenu()).queryByRole('menuitem', { name: 'Enquiries' })).toBeNull();
  });
});
