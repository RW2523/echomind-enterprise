import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import { captureStarted, _resetCaptureForTests } from '../app/capture';
import { parseAction, parseComponent, parseEnvelope } from '../counsel/validate';
import { matter, me, mockApi, renderLaw } from './harness';

/** EchoMind Legal chat (docs/echomind-legal/02, 03, 04; AC02–AC05, AC07, AC11). */
const hex = (c: string) => c.repeat(32);
const CIT = `cit_${hex('a')}`;
const CMP = `cmp_${hex('1')}`;
const CMP2 = `cmp_${hex('2')}`;
const DSC = `dsc_${hex('d')}`;

const answerEnvelope = (extra: any[] = []) => ({
  schema_version: '1.0', message_id: 'turn_1', matter_id: 'id-M-101', summary: 'Answered from 1 source, 1 with a verified quote.',
  components: [{ id: CMP, version: 1, type: 'source_answer', props: {
    answer: 'The agreement requires delivery by May 15, 2026 [1].', state: 'answered',
    citations: [{ id: CIT, n: 1, document_id: 'doc1', document_title: 'Services Agreement', version_number: 1, version_status: 'executed', page: 2,
                  clause: '4.2', verified: true }],
    unsupported: [], notes: [] } }, ...extra],
  suggested_actions: [],
});

const comparison = (status = 'potential', can_review = true, version = 1) => ({
  id: CMP2, version, type: 'evidence_comparison', data_ref: CMP2, props: {
    discrepancy_id: DSC, kind: 'date', title: 'Potential inconsistency in dates: “May 12, 2026” and “May 15, 2026”', status,
    a: { document_id: 'wa', document_title: 'Witness A', version_id: 'va', version_number: 1, value: 'May 12, 2026',
         context: 'took place on May 12, 2026. The witness', value_start: 14, value_end: 26, reader_path: '/m/M-101/documents/wa/read?v=va&at=10-22', stale: false },
    b: { document_id: 'wb', document_title: 'Witness B', version_id: 'vb', version_number: 1, value: 'May 15, 2026',
         context: 'took place on May 15, 2026, on arrival', value_start: 14, value_end: 26, reader_path: '/m/M-101/documents/wb/read?v=vb&at=10-22', stale: false },
    explanation: 'Source [A] gives May 12, 2026 and [B] gives May 15, 2026.', caution: 'The two dates may describe different events.',
    method: 'model', history: status === 'potential' ? [] : [{ to_status: status, by: 'alice', at: '2026-10-26T12:00:00.000Z', note: 'Dispatch and arrival.' }],
    can_review,
  } });

const stream = (envelope: object, terminal = 'turn.completed') => ({ __ndjson: [
  { event: 'turn.started', data: { turn_id: 'turn_1', conversation_id: 'conv_1', matter_id: 'id-M-101', intent: 'ask_fact' } },
  { event: 'answer.delta', data: { text: 'The agreement requires delivery by May 15, 2026 [1].' } },
  { event: 'ui.final', data: envelope },
  { event: terminal, data: { turn_id: 'turn_1', status: 'completed' } },
] });

function world(handlers: Record<string, (body: any) => [number, unknown]> = {}, roles = ['attorney']) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(roles)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal'), matter('M-215', 'Lease dispute', 'Beta Corp')] }],
    'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }],
    'GET /api/legal/counsel/conversations': () => [200, { conversations: [] }],
    'POST /api/legal/counsel/conversations': (b) => [201, { conversation: { id: 'conv_1', matter_id: b.matter_id, title: 'New conversation' } }],
    ...handlers,
  });
}

async function ask(text: string) {
  const box = await screen.findByRole('textbox', { name: 'Ask EchoMind' });
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: 'Enter' });
}

const axeClean = async (el: Element) =>
  (await axe.run(el, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target).join(' ')}`);

beforeEach(() => { localStorage.clear(); _resetCaptureForTests(); });
afterEach(() => { cleanup(); localStorage.clear(); });

describe('The chat home', () => {
  it('quiet: one question, the matter, four plain starters, the matter\'s pages one click away, four destinations, private and not recording', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    world();
    const { container, router } = renderLaw('/');
    expect(await screen.findByRole('heading', { name: 'How can I help?' })).toBeTruthy();
    expect(router.state.location.pathname).toBe('/chat');
    expect(await screen.findByText('M-101 · MSA renewal', { selector: '.qc-context b' })).toBeTruthy();
    const starters = within(screen.getByRole('list', { name: 'Suggestions' })).getAllByRole('button');
    expect(starters.map((t) => t.textContent)).toEqual([
      'What needs my attention today?', 'Find conflicting dates or amounts', 'Build a chronology of events', 'Summarize this matter']);
    const pages = screen.getByRole('navigation', { name: 'This matter' });
    expect(within(pages).getAllByRole('link').map((a) => a.textContent)).toEqual(['Open M-101', 'Documents', 'Meetings', 'Drafting']);
    expect(screen.queryByRole('navigation', { name: 'Current matter' })).toBeNull();       // the rail holds four destinations only
    const rail = screen.getByRole('navigation', { name: 'Workspace' });
    expect(within(rail).getAllByRole('link').map((a) => a.textContent)).toEqual(['Chat', 'Matters', 'Drafts', 'Activity']);
    expect(within(rail).getByRole('button', { name: 'New chat' })).toBeTruthy();
    expect(screen.getByText('Not recording')).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Private/ })).toBeTruthy();
    expect(await axeClean(container)).toEqual([]);
    fireEvent.click(within(pages).getByRole('link', { name: 'Meetings' }));
    expect(router.state.location.pathname).toBe('/m/M-101/meetings');
  });

  it('with the conversation assistant on, the home links to it among the matter\'s pages', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    mockApi({
      'GET /api/auth/me': () => [200, { ...me(), features: { intake: false, intake_links: false, conversation: true, panel: true } }],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }],
      'GET /api/legal/counsel/conversations': () => [200, { conversations: [] }],
    });
    const { router } = renderLaw('/chat');
    const pages = await screen.findByRole('navigation', { name: 'This matter' });
    await waitFor(() => expect(within(pages).getAllByRole('link').map((a) => a.textContent)).toEqual(
      ['Open M-101', 'Documents', 'Meetings', 'Ask by voice', 'Drafting']));
    fireEvent.click(within(pages).getByRole('link', { name: 'Ask by voice' }));
    expect(router.state.location.pathname).toBe('/m/M-101/conversation');
  });

  it('the recording status follows the real capture, not the page', async () => {
    world();
    renderLaw('/chat');
    expect(await screen.findByText('Not recording')).toBeTruthy();
    let stop: () => void = () => {};
    act(() => { stop = captureStarted(); });
    expect(screen.getByText('Recording')).toBeTruthy();
    act(() => stop());
    expect(screen.getByText('Not recording')).toBeTruthy();
  });
});

describe('Asking', () => {
  it('starts a conversation in the chosen matter, streams, then shows the verified answer with its source', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const calls = world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(answerEnvelope())] });
    const { router } = renderLaw('/chat');
    await ask('When must Apex complete delivery?');
    expect(await screen.findByText('Supported by 1 verified quote')).toBeTruthy();
    expect(router.state.location.pathname).toBe('/chat/conv_1');
    expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/counsel/conversations')?.body).toEqual({ matter_id: 'id-M-101' });
    const turn = calls.find((c) => c.path === '/api/legal/counsel/conversations/conv_1/turns')!.body;
    expect(turn.text).toBe('When must Apex complete delivery?');
    expect(turn.client_request_id).toMatch(/^[0-9a-f]{32}$/);
    expect(screen.getByRole('button', { name: /Source 1: Services Agreement, page 2, clause 4.2\. Verified quote/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Open source 1: Services Agreement, page 2' })).toBeTruthy();
  });

  it('opens a source in the side panel, highlighted, and Escape returns focus to the chip', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(answerEnvelope())],
      [`GET /api/legal/counsel/citations/${CIT}`]: () => [200, { citation: {
        id: CIT, n: 1, matter_id: 'id-M-101', verification: 'verbatim', verified: true,
        document: { id: 'doc1', title: 'Services Agreement', kind: 'contract', status: 'executed' },
        version: { id: 'v1', number: 1, status: 'executed', filename: 'services-agreement.txt', sha256: 'f'.repeat(64) },
        quote: 'complete delivery by May 15, 2026',
        context: { text: '4.2 Delivery. Apex shall complete delivery by May 15, 2026.', highlight_start: 25, highlight_end: 58, offset: 0,
                   truncated_before: false, truncated_after: false },
        clause: '4.2', page: 2, line: 3, newer_version: null, reader_path: '/m/M-101/documents/doc1/read?v=v1&at=40-73' } }],
    });
    renderLaw('/chat');
    await ask('When must Apex complete delivery?');
    const chip = await screen.findByRole('button', { name: /Source 1: Services Agreement/ });
    chip.focus();
    fireEvent.click(chip);
    const panel = await screen.findByRole('complementary', { name: 'Source' });
    expect(await within(panel).findByText('complete delivery by May 15, 2026', { selector: 'mark' })).toBeTruthy();
    expect(within(panel).getByText(/Verified quote/)).toBeTruthy();
    expect(within(panel).getByRole('link', { name: 'Open full document' }).getAttribute('href')).toBe('/m/M-101/documents/doc1/read?v=v1&at=40-73');
    expect(calls.some((c) => c.path === `/api/legal/counsel/citations/${CIT}`)).toBe(true);
    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('complementary', { name: 'Source' })).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(chip));
  });

  it('a source that is no longer readable says so instead of showing anything', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(answerEnvelope())],
      [`GET /api/legal/counsel/citations/${CIT}`]: () => [409, { detail: 'The stored document no longer matches this citation, so it is not shown.' }],
    });
    renderLaw('/chat');
    await ask('When must Apex complete delivery?');
    fireEvent.click(await screen.findByRole('button', { name: /Source 1: Services Agreement/ }));
    expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'The stored document no longer matches this citation, so it is not shown.');
  });

  it('an abstention says no source was found and offers no sources', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const env = { ...answerEnvelope(), summary: "No source found in this matter's documents.",
                  components: [{ id: CMP, version: 1, type: 'source_answer', props: { answer: 'The documents in this matter do not answer that.',
                                 state: 'abstained', citations: [], unsupported: [], notes: [] } }] };
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(env, 'turn.abstained')] });
    renderLaw('/chat');
    await ask('What is the governing law of the Kestrel licence?');
    expect(await screen.findByText("No source found in this matter's documents")).toBeTruthy();
    expect(screen.queryByRole('group', { name: 'Sources' })).toBeNull();
    expect(screen.queryByRole('button', { name: /Open source/ })).toBeNull();
  });

  it('without a matter, asks which one, then answers in it', async () => {
    const calls = world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(answerEnvelope())] });
    renderLaw('/chat');
    await ask('What matters today?');
    const pick = await screen.findByRole('region', { name: 'Which matter is this about?' });
    fireEvent.click(within(pick).getByRole('button', { name: 'M-215 · Lease dispute' }));
    await screen.findByText('Supported by 1 verified quote');
    expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/counsel/conversations')?.body).toEqual({ matter_id: 'id-M-215' });
  });

  it('losing access mid-answer stops it and says why', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, { __ndjson: [
      { event: 'turn.started', data: { turn_id: 'turn_1' } }, { event: 'scope_revoked', data: { turn_id: 'turn_1' } }] }] });
    renderLaw('/chat');
    await ask('When must Apex complete delivery?');
    expect((await screen.findByRole('alert')).textContent).toMatch(/no longer have access to this matter/);
  });
});

describe('Trusted components only', () => {
  it('text is never markup, unknown components are inert, and actions outside the workspace are not offered', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const env = {
      ...answerEnvelope([{ id: `cmp_${hex('3')}`, version: 1, type: 'html', props: { html: '<img src=x onerror="alert(1)">' } }]),
      suggested_actions: [
        { id: 'evil', label: 'Open evil', action_type: 'open_page', path: 'https://evil.example/', document_ids: [] },
        { id: 'js', label: 'Run js', action_type: 'open_page', path: 'javascript:alert(1)', document_ids: [] },
        { id: 'ok', label: 'Open obligations', action_type: 'open_page', path: '/m/M-101/obligations', document_ids: [] },
      ],
    };
    env.components[0] = { ...env.components[0], props: { ...env.components[0].props, answer: 'Plain <img src=x onerror="alert(1)"> text [1].' } } as any;
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(env)] });
    const { container } = renderLaw('/chat');
    await ask('Anything');
    expect(await screen.findByText('This part of the answer could not be shown.')).toBeTruthy();
    expect(container.querySelector('img')).toBeNull();
    expect(screen.getByText(/Plain <img src=x onerror="alert\(1\)"> text/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Open evil' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Run js' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Open obligations' })).toBeTruthy();
  });

  it('the client re-checks the contract', () => {
    expect(parseComponent({ id: 'nope', type: 'notice', props: { tone: 'info', text: 'x' } }).type).toBe('inert');
    expect(parseComponent({ id: CMP, type: 'notice', props: { tone: 'info', text: 'x', onclick: 'x' } }).type).toBe('notice');   // extra fields are ignored, never rendered
    expect(parseComponent({ id: CMP, type: 'source_answer', props: { answer: 'x', state: 'abstained', citations: [answerEnvelope().components[0].props.citations[0]] } }).type).toBe('inert');
    expect(parseComponent({ ...comparison(), props: { ...comparison().props, a: { ...comparison().props.a, value_end: 999 } } }).type).toBe('inert');
    expect(() => parseAction({ id: 'x', label: 'x', action_type: 'open_page', path: '/m/M-1/../admin' })).toThrow();
    expect(() => parseAction({ id: 'x', label: 'x', action_type: 'ask' })).toThrow();
    const many = Array.from({ length: 5 }, (_, i) => ({ id: `cmp_${String(i).repeat(32)}`, type: 'notice', props: { tone: 'info', text: `${i}` } }));
    expect(parseEnvelope({ schema_version: '1.0', message_id: 'm', matter_id: 'x', components: many })!.components).toHaveLength(3);
    expect(parseEnvelope({ schema_version: '2.0', message_id: 'm', matter_id: 'x', components: [] })).toBeNull();
  });
});

describe('Potential inconsistencies', () => {
  const compareEnvelope = (comp: object) => ({ ...answerEnvelope(), summary: '1 potential inconsistency found.', components: [comp] });

  it('shows both sources with the uncertainty, and a lawyer resolves it with a reason', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(compareEnvelope(comparison()))],
      'POST /api/legal/counsel/actions': () => [200, { component: comparison('explained', true, 2) }],
    });
    const { container } = renderLaw('/chat');
    await ask('Find delivery date conflicts');
    const card = await screen.findByRole('region', { name: /Potential inconsistency in dates/ });
    expect(within(card).getByText('Potential')).toBeTruthy();
    expect(within(card).getByText('May 12, 2026', { selector: 'mark' })).toBeTruthy();
    expect(within(card).getByText('May 15, 2026', { selector: 'mark' })).toBeTruthy();
    expect(within(card).getByText('The two dates may describe different events.')).toBeTruthy();
    expect(card.textContent).not.toMatch(/\blie|lied|fraud/i);
    fireEvent.click(within(card).getByRole('button', { name: 'Explained' }));
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }));
    expect(within(card).getByRole('alert').textContent).toMatch(/Say briefly why/);
    expect(calls.some((c) => c.path === '/api/legal/counsel/actions')).toBe(false);
    fireEvent.change(within(card).getByRole('textbox', { name: /Explained: why/ }), { target: { value: 'Dispatch and arrival.' } });
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(within(card).getByText('Explained', { selector: '.qc-status-chip' })).toBeTruthy());
    const body = calls.find((c) => c.path === '/api/legal/counsel/actions')!.body;
    expect(body).toMatchObject({ component_id: CMP2, action_id: 'review', matter_id: 'id-M-101', expected_version: 1,
                                 parameters: { resolution: 'explained', note: 'Dispatch and arrival.' } });
    expect(body.idempotency_key).toMatch(/^[0-9a-f]{32}$/);
    expect(within(card).getByRole('list', { name: 'Review history' }).textContent).toMatch(/Explained by alice.*Dispatch and arrival/);
    expect(await axeClean(container)).toEqual([]);
  });

  it('someone else changed it first: the current state is shown, with why', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(compareEnvelope(comparison()))],
      'POST /api/legal/counsel/actions': () => [409, { detail: 'Someone changed this item while you were reading it. Here is its current state.',
                                                       code: 'VERSION_CONFLICT', component: comparison('dismissed', true, 2) }],
    });
    renderLaw('/chat');
    await ask('Find delivery date conflicts');
    const card = await screen.findByRole('region', { name: /Potential inconsistency in dates/ });
    fireEvent.click(within(card).getByRole('button', { name: 'Confirm inconsistency' }));
    fireEvent.change(within(card).getByRole('textbox'), { target: { value: 'Both say delivery.' } });
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }));
    const updated = await screen.findByRole('region', { name: /Potential inconsistency in dates/ });
    await waitFor(() => expect(within(updated).getByText('Dismissed', { selector: '.qc-status-chip' })).toBeTruthy());
    expect(within(updated).getByRole('alert').textContent).toMatch(/Someone changed this item/);
  });

  it('a read-only member sees it but is offered nothing to resolve it with', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(compareEnvelope(comparison('potential', false)))] });
    renderLaw('/chat');
    await ask('Find delivery date conflicts');
    const card = await screen.findByRole('region', { name: /Potential inconsistency in dates/ });
    expect(within(card).queryByRole('button', { name: 'Explained' })).toBeNull();
    expect(within(card).getByText(/Resolving it is for the matter's lawyers and reviewers/)).toBeTruthy();
  });

  it('a briefing shows at most three items; Review opens that item in the conversation', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const briefing = { ...answerEnvelope(), summary: '3 things in M-101 need your attention.', components: [{ id: CMP, version: 1, type: 'priority_list', props: {
      summary: '3 things in M-101 need your attention.', more: 1, items: [
        { id: 'discrepancies', kind: 'discrepancy', title: '1 potential inconsistency to review', detail: 'Witness A says “May 12, 2026”.',
          action: { id: 'review', label: 'Review', action_type: 'open_discrepancy', target_id: DSC, document_ids: [] } },
        { id: 'new_documents', kind: 'new_documents', title: '4 new documents', detail: 'Agreement, Witness A.',
          action: { id: 'summarize_new', label: 'What do they say?', action_type: 'ask', prompt: 'Summarize the key points of the new documents.', document_ids: ['doc1'] } },
        { id: 'obligations', kind: 'obligation', title: '1 deadline proposed', detail: 'Not yet confirmed.',
          action: { id: 'open_obligations', label: 'Confirm dates', action_type: 'open_page', path: '/m/M-101/obligations', document_ids: [] } },
      ] } }] };
    let n = 0;
    const calls = world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, n++ === 0 ? stream(briefing) : stream(compareEnvelope(comparison()))] });
    renderLaw('/chat');
    await ask('What matters today?');
    const list = await screen.findByRole('region', { name: '3 things in M-101 need your attention.' });
    expect(within(list).getAllByRole('listitem')).toHaveLength(3);
    expect(within(list).getByText('and 1 more')).toBeTruthy();
    fireEvent.click(within(list).getByRole('button', { name: 'Review' }));
    expect(await screen.findByRole('region', { name: /Potential inconsistency in dates/ })).toBeTruthy();
    const second = calls.filter((c) => c.path === '/api/legal/counsel/conversations/conv_1/turns')[1].body;
    expect(second.focus_discrepancy_id).toBe(DSC);
  });
});

describe('History and the other destinations', () => {
  it('reopens a conversation from the rail with its stored answers', async () => {
    world({
      'GET /api/legal/counsel/conversations': () => [200, { conversations: [{ id: 'conv_9', title: 'When must Apex complete delivery?', created_at: '', updated_at: '',
        matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, turns: 1 }] }],
      'GET /api/legal/counsel/conversations/conv_9': () => [200, { conversation: { id: 'conv_9', matter_id: 'id-M-101', title: 'x' }, turns: [
        { id: 'turn_1', seq: 1, question: 'When must Apex complete delivery?', intent: 'ask_fact', status: 'completed', created_at: '', envelope: answerEnvelope() }] }],
    });
    renderLaw('/chat');
    const recent = await screen.findByRole('navigation', { name: 'Recent conversations' });
    fireEvent.click(within(recent).getByRole('link', { name: /When must Apex complete delivery/ }));
    expect(await screen.findByText('Supported by 1 verified quote')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Matter: M-101 MSA renewal. Switch matter' })).toBeTruthy();
  });

  it('Drafts lists what waits for you and your own drafts', async () => {
    world({ 'GET /api/legal/reviews': () => [200, { count: 1, waiting: [{ kind: 'research_note', id: 'a1', title: 'Liability cap', matter: { number: 'M-101', title: 'MSA renewal' },
      by: 'bob', at: '2026-10-11T10:00:00Z', path: '/m/M-101/notes/a1' }], mine: [] }] });
    renderLaw('/drafts');
    const waiting = await screen.findByRole('list', { name: 'Waiting for you' });                  // the existing reviews queue, reused
    expect(within(waiting).getByRole('link', { name: /Liability cap/ }).getAttribute('href')).toBe('/m/M-101/notes/a1');
    expect(screen.getByRole('link', { name: 'Drafts, 1 waiting for you' })).toBeTruthy();
  });

  it('Activity shows your own actions and documents still being read', async () => {
    world({ 'GET /api/legal/counsel/activity': () => [200, { next_before: null, in_progress: [{ matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, documents: 2 }],
      events: [{ seq: 5, ts: '2026-10-26T10:00:00.000Z', action: 'matter.read', outcome: 'allow', actor: { username: 'alice', roles: ['attorney'] },
                 matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' } }] }] });
    renderLaw('/activity');
    expect(await screen.findByText(/2 documents being read in/)).toBeTruthy();
    expect(screen.getByRole('region', { name: 'What you did' })).toBeTruthy();
  });
});

describe('Citation numbers', () => {
  it('are shown 1, 2, 3 in the order the answer cites them, whatever research numbered its passages', async () => {
    const { displayNumbers } = await import('../counsel/cards');
    const c = (n: number) => ({ id: `cit_${String(n).repeat(32).slice(0, 32)}`, n, document_id: `d${n}`, document_title: `D${n}`, verified: true, removed: false });
    const labels = displayNumbers('First [5]. Then [1] and [5, 3].', [c(1), c(3), c(5)]);
    expect([...labels.entries()]).toEqual([[5, 1], [1, 2], [3, 3]]);
  });
});

describe('Chronology', () => {
  const side = (doc: string, title: string, value: string, context: string) => ({
    document_id: doc, document_title: title, version_id: `v-${doc}`, version_number: 1, value, context,
    value_start: context.indexOf(value), value_end: context.indexOf(value) + value.length,
    reader_path: `/m/M-101/documents/${doc}/read?v=v-${doc}&at=10-22`, stale: false });
  const ev = (i: number, date: string | undefined, dateText: string, extra: object = {}) => ({
    id: `ev${i}`, ...(date ? { date } : {}), date_text: dateText, certainty: date ? 'stated' : 'undated', kind: 'event',
    reported: false, superseded: false, text: `Sentence ${i} on ${dateText}.`, notes: [],
    source: side(`d${i}`, `Document ${i}`, dateText, `Sentence ${i} on ${dateText}.`), ...extra });
  const tl = (events: object[]) => ({ ...answerEnvelope(), summary: 'Chronology', components: [{ id: CMP, version: 1, type: 'timeline', props: {
    summary: `${events.length} events in M-101.`, more: 0, events,
    caution: 'Dates are shown as the documents state them; nothing is inferred.' } }] });

  it('lists dated events first with what each is, then on request the ones whose date is not stated; each opens its words', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const events = [
      ev(1, '2026-04-01', 'April 1, 2026', { kind: 'document_date' }),
      ev(2, '2026-05-12', 'May 12, 2026', { reported: true, certainty: 'uncertain', notes: ['Explained by alice: “Dispatch and arrival.”'] }),
      ev(3, '2026-05-15', 'May 15, 2026', { kind: 'deadline', superseded: true, status: 'Confirmed by a lawyer' }),
      ...Array.from({ length: 7 }, (_, k) => ev(10 + k, `2026-06-${String(k + 10)}`, `June ${k + 10}, 2026`)),
      ev(30, undefined, 'the 20th'),
    ];
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(tl(events))] });
    renderLaw('/chat');
    await ask('Build a chronology');
    const card = await screen.findByRole('region', { name: '11 events in M-101.' });
    const dated = within(card).getByRole('list', { name: 'Dated events' });
    expect(within(dated).getAllByRole('listitem')).toHaveLength(8);                           // a first view, the rest on request
    const witness = within(dated).getAllByRole('listitem')[1];
    expect(within(witness).getByText('Recollection')).toBeTruthy();
    expect(within(witness).getByText('Uncertain')).toBeTruthy();
    expect(within(witness).getByText(/Explained by alice/)).toBeTruthy();
    const deadline = within(dated).getAllByRole('listitem')[2];
    expect(['Deadline', 'Replaced', 'Confirmed by a lawyer'].every((t) => within(deadline).queryByText(t))).toBe(true);
    expect(within(card).queryByRole('list', { name: 'Events whose date is not stated' })).toBeNull();
    fireEvent.click(within(card).getByRole('button', { name: /Show all \(3 more, 1 without a stated date\)/ }));
    expect(within(card).getAllByRole('list', { name: 'Dated events' })[0].querySelectorAll('li.qc-event')).toHaveLength(10);
    const undated = within(card).getByRole('list', { name: 'Events whose date is not stated' });
    expect(within(undated).getByText('“the 20th”')).toBeTruthy();
    fireEvent.click(within(witness).getByRole('button', { name: 'Inspect Document 2, May 12, 2026' }));
    const panel = await screen.findByRole('complementary', { name: 'May 12, 2026: Document 2' });
    expect(within(panel).getByText('May 12, 2026', { selector: 'mark' })).toBeTruthy();
  });

  it('an event that claims a date while saying it has none is not shown', () => {
    const bad = { id: CMP, type: 'timeline', props: { summary: 's', caution: 'c', more: 0,
      events: [{ ...ev(1, '2026-05-12', 'May 12, 2026'), certainty: 'undated' }] } };
    expect(parseComponent(bad).type).toBe('inert');
    expect(parseComponent({ ...bad, props: { ...bad.props, events: [ev(1, '12/05/2026', 'x')] } }).type).toBe('inert');
  });
});

describe('A matter at a glance', () => {
  const compareEnvelope = (comp: object) => ({ ...answerEnvelope(), summary: '1 potential inconsistency found.', components: [comp] });
  const obl = (id: string, due: string | null, status: string, overdue = false) => ({
    id, document_id: 'doc1', document_title: 'Services Agreement', version_id: 'v1', clause: '4.2', kind: 'notice', party: null,
    action: `Deliver the units (${id})`, when: null, due_date: due, due_note: null, timezone: 'UTC', due_at: null, overdue, status, owner: null,
    source: 'rules', evidence: { start: 0, end: 5, quote: 'x' }, confirmed_by: null, confirmed_at: null, version: 1, created_by: null,
    created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z' });
  const doc = (id: string, title: string, at: string) => ({ id, title, kind: 'contract', status: 'executed', version: 1, created_at: at,
    updated_at: at, current_version: null, findings: {} });

  it('shows open inconsistencies, the next deadlines and the latest documents from the records; an inconsistency opens in the chat', async () => {
    const calls = world({
      'GET /api/legal/counsel/discrepancies': () => [200, { discrepancies: [
        { id: DSC, title: 'Potential inconsistency in dates', status: 'potential', kind: 'date', version: 1, updated_at: '2026-10-26T12:00:00Z' }] }],
      'GET /api/legal/matters/id-M-101/obligations': () => [200, { timezone: 'UTC', calendar: false,
        obligations: [obl('o1', '2026-05-20', 'confirmed'), obl('o2', '2026-05-15', 'proposed', true), obl('o3', null, 'proposed'), obl('o4', '2026-05-10', 'dismissed')] }],
      'GET /api/legal/matters/id-M-101/documents': () => [200, { documents: [doc('d1', 'Services Agreement', '2026-04-01T00:00:00Z'),
                                                                          doc('d2', 'Amendment No. 1', '2026-05-14T00:00:00Z')] }],
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(compareEnvelope(comparison()))],
    });
    const { router } = renderLaw('/m/M-101/overview');
    const glance = await screen.findByRole('region', { name: 'At a glance' });
    const deadlines = await within(glance).findByRole('region', { name: 'Next deadlines' });
    await waitFor(() => expect(within(deadlines).getAllByRole('listitem')).toHaveLength(2));        // undated and dismissed left out
    expect(within(deadlines).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      expect.stringMatching(/Deliver the units \(o2\).*Overdue/), expect.stringMatching(/Deliver the units \(o1\).*Confirmed/)]);
    const docs = await within(glance).findByRole('region', { name: 'Documents (2)' });
    const docLinks = within(within(docs).getByRole('list')).getAllByRole('link');
    expect(docLinks[0].textContent).toMatch(/^Amendment No\. 1/);                                    // the latest first
    expect(docLinks[1].getAttribute('href')).toBe('/m/M-101/documents/d1/read');
    expect(within(docs).getByRole('link', { name: 'All documents' }).getAttribute('href')).toBe('/m/M-101/documents');
    const issues = await within(glance).findByRole('region', { name: 'Open inconsistencies' });
    fireEvent.click(await within(issues).findByRole('button', { name: /^Dates/ }));
    await screen.findByRole('region', { name: /Potential inconsistency in dates: “May 12, 2026”/ });
    expect(router.state.location.pathname).toBe('/chat/conv_1');
    const turn = calls.find((c) => c.method === 'POST' && c.path.endsWith('/turns'));
    expect(turn?.body).toMatchObject({ text: 'Show this potential inconsistency', focus_discrepancy_id: DSC });
    expect(calls.filter((c) => c.path.endsWith('/turns'))).toHaveLength(1);                         // asked once
  });

  it('a part the person may not see is left out, not shown as an error', async () => {
    world({ 'GET /api/legal/counsel/discrepancies': () => [404, { detail: 'Not Found' }],
            'GET /api/legal/matters/id-M-101/obligations': () => [200, { obligations: [], timezone: 'UTC', calendar: false }],
            'GET /api/legal/matters/id-M-101/documents': () => [200, { documents: [] }] });
    renderLaw('/m/M-101/overview');
    const glance = await screen.findByRole('region', { name: 'At a glance' });
    expect(await within(glance).findByText(/No dated deadline is recorded/)).toBeTruthy();
    await waitFor(() => expect(within(glance).queryByRole('region', { name: 'Open inconsistencies' })).toBeNull());
  });
});

describe('Several matters in one conversation (owner, 2026-10-28)', () => {
  const twoWorld = (handlers: Record<string, (body: any) => [number, unknown]> = {}) => mockApi({
    'GET /api/auth/me': () => [200, me()],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal'), matter('M-215', 'Lease dispute', 'Beta Corp'),
                                                      matter('M-300', 'Board investigation', 'Gamma', { sensitivity: 'highly_restricted' })] }],
    'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }],
    'GET /api/legal/counsel/conversations': () => [200, { conversations: [] }],
    'POST /api/legal/counsel/conversations': (b) => [201, { conversation: { id: 'conv_1', matter_id: (b.matter_ids ?? [b.matter_id])[0],
                                                                             matter_ids: b.matter_ids ?? [b.matter_id], title: 'New conversation' } }],
    ...handlers,
  });
  const pick = async (name: RegExp) => {
    fireEvent.click(screen.getByRole('button', { name: /Change the matter|Pick the matter to ask about/ }));
    fireEvent.click(await screen.findByRole('option', { name }));
  };

  it('choosing a second matter in the question box asks across both, says they belong to different clients, and offers them side by side', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const calls = twoWorld({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(answerEnvelope())] });
    renderLaw('/chat');
    await screen.findByText('M-101 · MSA renewal', { selector: '.qc-context b' });
    await pick(/M-215/);
    expect(await screen.findByText('M-101, M-215', { selector: '.qc-context b' })).toBeTruthy();
    expect(document.querySelector('.qc-context')?.textContent).toMatch(/belong to different clients/);
    const starters = within(screen.getByRole('list', { name: 'Suggestions' })).getAllByRole('button').map((t) => t.textContent);
    expect(starters).toContain('Compare these cases side by side');
    expect(screen.queryByRole('navigation', { name: 'This matter' })).toBeNull();           // several matters: no one matter's pages
    expect(screen.queryByRole('button', { name: /Choose documents/ })).toBeNull();        // restricting to documents: one matter only
    await ask('What is the notice period?');
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/counsel/conversations')?.body)
      .toEqual({ matter_ids: ['id-M-101', 'id-M-215'] }));
  });

  it('a highly restricted matter is chosen on its own, and choosing another replaces it', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    twoWorld();
    renderLaw('/chat');
    await screen.findByText('M-101 · MSA renewal', { selector: '.qc-context b' });
    await pick(/M-300/);
    expect(await screen.findByText('M-300 · Board investigation', { selector: '.qc-context b' })).toBeTruthy();
    fireEvent.click(screen.getByRole('option', { name: /M-215/ }));
    await waitFor(() => expect(screen.getByText('M-215 · Lease dispute', { selector: '.qc-context b' })).toBeTruthy());
  });

  it('a chronology across matters can be narrowed by kind, by matter and by month', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const side = (n: number, text: string) => ({ document_id: `d${n}`, document_title: `Document ${n}`, version_id: `v${n}`, value: text,
      context: text, value_start: 0, value_end: text.length, reader_path: `/m/M-101/documents/d${n}/read`, stale: false });
    const e = (n: number, date: string, kind: string, matterNo: string, extra: object = {}) => ({ id: `ev${n}`, date, date_text: date,
      certainty: 'stated', kind, reported: false, superseded: false, text: `Event ${n}.`, notes: [], source: side(n, date), matter: matterNo, ...extra });
    const events = [e(1, '2026-04-01', 'document_date', 'M-101'), e(2, '2026-05-12', 'event', 'M-101', { reported: true }),
                    e(3, '2026-05-15', 'deadline', 'M-101'), e(4, '2026-05-20', 'deadline', 'M-215'), e(5, '2026-06-02', 'event', 'M-215')];
    const env = { ...answerEnvelope(), matters: [{ id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, { id: 'id-M-215', number: 'M-215', title: 'Lease dispute' }],
      components: [{ id: CMP, version: 1, type: 'timeline', props: { summary: '5 dated events in 2 matters.', more: 0, events, caution: 'c' } }] };
    twoWorld({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(env)] });
    renderLaw('/chat');
    await ask('Build a chronology of events');
    const card = await screen.findByRole('region', { name: '5 dated events in 2 matters.' });
    const count = () => card.querySelectorAll('li.qc-event').length;
    expect(count()).toBe(5);
    fireEvent.click(within(within(card).getByRole('group', { name: 'Show' })).getByRole('button', { name: /Deadlines/ }));
    expect(count()).toBe(2);
    fireEvent.click(within(within(card).getByRole('group', { name: 'Matter' })).getByRole('button', { name: 'M-215' }));
    expect(count()).toBe(1);
    expect(within(card).getByText(/Showing 1 of 5 events/)).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Show everything' }));
    expect(count()).toBe(5);
    fireEvent.click(within(within(card).getByRole('group', { name: 'Events by month' })).getByRole('button', { name: 'May 2026: 3 events' }));
    expect(count()).toBe(3);
    expect(within(card).getAllByText('M-215', { selector: '.qc-mtag' }).length).toBe(1);
  });

  it('matters side by side sort by any column, and each opens its own overview', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const row = (n: string, docs: number, open: number, due?: string) => ({ matter_id: `id-${n}`, number: n, title: `${n} title`, client: `Client ${n}`,
      status: 'open', sensitivity: 'restricted', documents: docs, open_inconsistencies: open, path: `/m/${n}/overview`,
      ...(due ? { next_deadline: due, deadline_state: 'Not confirmed', next_deadline_text: 'Deliver the units' } : {}) });
    const env = { ...answerEnvelope(), components: [{ id: CMP, version: 1, type: 'case_table', props: { summary: '2 matters side by side, 2 clients.',
      rows: [row('M-215', 9, 0), row('M-101', 4, 3, '2026-05-20')], caution: 'From the records.' } }] };
    twoWorld({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(env)] });
    const { router } = renderLaw('/chat');
    await ask('Compare these cases side by side');
    const card = await screen.findByRole('region', { name: '2 matters side by side, 2 clients.' });
    const first = () => within(card).getAllByRole('row')[1].querySelector('th button')?.textContent;
    expect(first()).toBe('M-101');                                        // by matter number
    fireEvent.click(within(card).getByRole('button', { name: /Documents/ }));
    expect(first()).toBe('M-101');                                        // 4 before 9
    fireEvent.click(within(card).getByRole('button', { name: /Documents/ }));
    expect(first()).toBe('M-215');
    expect(within(card).getByRole('columnheader', { name: /Documents/ }).getAttribute('aria-sort')).toBe('descending');
    expect(within(card).getByText('Not confirmed')).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Open M-215, M-215 title' }));
    expect(router.state.location.pathname).toBe('/m/M-215/overview');
  });

  it('resolving a potential inconsistency of another matter of the conversation acts under that matter', async () => {
    localStorage.setItem('echomind.counsel.matter', 'id-M-101');
    const cmp = { ...comparison(), matter_id: 'id-M-215' };
    const env = { ...answerEnvelope(), matters: [{ id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, { id: 'id-M-215', number: 'M-215', title: 'Lease dispute' }],
      components: [cmp] };
    const calls = twoWorld({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(env)],
      'POST /api/legal/counsel/actions': () => [200, { component: { ...comparison('explained', true, 2), matter_id: 'id-M-215' } }],
    });
    renderLaw('/chat');
    await ask('Find conflicting dates');
    const card = await screen.findByRole('region', { name: /Potential inconsistency in dates/ });
    expect(within(card).getByText('M-215', { selector: '.qc-mtag' })).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Explained' }));
    fireEvent.change(within(card).getByRole('textbox'), { target: { value: 'Dispatch and arrival.' } });
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(calls.find((c) => c.path === '/api/legal/counsel/actions')?.body.matter_id).toBe('id-M-215'));
  });
});

describe('where answers come from (review P1)', () => {
  it('claims no more than the configured model location supports', async () => {
    const { whereAnswersComeFrom } = await import('../counsel/where');
    const at = (model_location?: 'this_server' | 'firm_network' | 'external') =>
      whereAnswersComeFrom({ features: { intake: false, intake_links: false, model_location } } as never);
    expect(at('this_server').text).toContain('not sent to an outside AI service');
    expect(at('external').label).toBe('External model');
    expect(at('external').text).toContain('sent to that service');
    for (const w of [at('this_server'), at('firm_network'), at('external'), at()]) {
      expect(w.text).not.toMatch(/nothing leaves|no cloud|air.?gap|guarantee/i);
    }
    expect(at().chip).toBe('');
  });
});

describe('answer card honesty (review C1)', () => {
  it('drops a marker without a source and never says "supported" with no verified quote', async () => {
    const { render } = await import('@testing-library/react');
    const { ComponentView } = await import('../counsel/cards');
    const ctx = { openCitation: () => {}, go: () => {}, matterId: 'm1' } as never;
    const answer = (citations: unknown[]) => ({ id: 'cmp_' + 'a'.repeat(32), type: 'source_answer',
      props: { state: 'answered', answer: 'The cap is two times the fees [1] and nothing else [7].', citations, unsupported: [], notes: [] } }) as never;
    const cite = { n: 1, id: 'c1', document_id: 'd1', document_title: 'MSA', verified: true };
    const { container, unmount } = render(<ComponentView comp={answer([cite])} ctx={ctx} />);
    expect(container.textContent).not.toContain('[7]');
    expect(container.textContent).toContain('Supported by 1 verified quote');
    unmount();
    const none = render(<ComponentView comp={answer([{ ...cite, verified: false }])} ctx={ctx} />);
    expect(none.container.textContent).toContain('Not verified');
    expect(none.container.textContent).not.toContain('Supported by 0');
  });
});

describe('source panel errors (review S1)', () => {
  it('tells a source that is gone from one that did not load, and offers to try again', async () => {
    const { ApiError } = await import('../api/client');
    const { counselApi } = await import('../counsel/api');
    const { render, screen: s, fireEvent: fe, cleanup: done } = await import('@testing-library/react');
    const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query');
    const { MemoryRouter } = await import('react-router');
    const { PanelProvider, usePanel } = await import('../counsel/context');
    const { default: SidePanel } = await import('../counsel/SidePanel');
    const React = await import('react');
    let fail: unknown = new ApiError(404, 'Not Found');
    const spy = vi.spyOn(counselApi, 'citation').mockImplementation(async () => { throw fail; });
    function Open() { const p = usePanel(); React.useEffect(() => { p.open({ kind: 'citation', id: 'c1' } as never); }, []); return null; }
    const ui = () => render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><PanelProvider><Open /><SidePanel /></PanelProvider></MemoryRouter></QueryClientProvider>);
    ui();
    expect(await s.findByText(/no longer available to you/)).toBeTruthy();
    done();
    fail = new TypeError('Failed to fetch');
    ui();
    expect(await s.findByText(/could not be loaded/)).toBeTruthy();
    fail = undefined;
    spy.mockImplementation(async () => { throw new ApiError(404, 'x'); });
    fe.click(s.getByRole('button', { name: 'Try again' }));
    expect(await s.findByText(/no longer available to you/)).toBeTruthy();
    spy.mockRestore();
  });
});
