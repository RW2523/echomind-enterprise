import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { _resetCaptureForTests } from '../app/capture';
import { parseComponent } from '../counsel/validate';
import { matter, me, mockApi, renderLaw } from './harness';

/** The cards that work on the matter's records (docs/echomind-legal/03; owner's request 2026-11-02). */
const hex = (c: string) => c.repeat(32);
const CMP = `cmp_${hex('7')}`;
const T1 = `turn_${hex('1')}`;
const side = { document_id: 'doc1', document_title: 'Services Agreement', version_id: 'v1', value: 'USD 120,000', context: 'shall pay USD 120,000 within',
  value_start: 10, value_end: 21, reader_path: '/m/M-101/documents/doc1/read?v=v1&at=100-111', stale: false };

const env = (comp: object, turnId = T1) => ({ schema_version: '1.0', message_id: turnId, matter_id: 'id-M-101', summary: 's', components: [comp], suggested_actions: [] });
const stream = (comp: object, turnId = T1, parent: string | null = turnId === T1 ? null : T1) => ({ __ndjson: [
  { event: 'turn.started', data: { turn_id: turnId, conversation_id: 'conv_1', matter_id: 'id-M-101', intent: 'x', parent_turn_id: parent } },
  { event: 'ui.final', data: env(comp, turnId) },
  { event: 'turn.completed', data: { turn_id: turnId, status: 'completed' } },
] });

function world(handlers: Record<string, (body: any) => [number, unknown]> = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me()],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
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
const actions = (calls: { method: string; path: string; body: any }[]) => calls.filter((c) => c.path === '/api/legal/counsel/actions').map((c) => c.body);

beforeEach(() => { localStorage.clear(); _resetCaptureForTests(); localStorage.setItem('echomind.counsel.matter', 'id-M-101'); });
afterEach(() => { cleanup(); localStorage.clear(); vi.useRealTimers(); });

const item = (id: string, status: string, version = 1, extra = {}) => ({ id, title: `Obligation ${id}`, party: 'Apex', due_date: '2026-05-15', status,
  overdue: status === 'proposed', document_title: 'Services Agreement', clause: '4.2', path: '/m/M-101/documents/doc1/read?v=v1&at=1-9', version, ...extra });
const checklist = (items: object[], can = true) => ({ id: CMP, version: 1, type: 'checklist', data_ref: CMP, props: {
  summary: '2 open obligations in M-101.', items, more: 0, can_done: can, can_confirm: can, caution: 'From the register.', register_path: '/m/M-101/obligations' } });

describe('Checklist', () => {
  it('ticks an item done through the server and shows the register as it now stands; a date is confirmed only after asking', async () => {
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(checklist([item('o1', 'proposed'), item('o2', 'proposed', 1, { due_date: undefined, overdue: false })]))],
      'POST /api/legal/counsel/actions': (b) => [200, { component: checklist(b.parameters.action === 'done'
        ? [item('o1', 'proposed'), item('o2', 'done', 2, { due_date: undefined, overdue: false })]
        : [item('o1', 'confirmed', 2, { overdue: true }), item('o2', 'done', 2, { due_date: undefined, overdue: false })]) }],
    });
    renderLaw('/chat');
    await ask('What are the open obligations?');
    const box = await screen.findByRole('checkbox', { name: 'Obligation o2' });
    expect(screen.getByText('Overdue')).toBeTruthy();
    expect(screen.getByText('Date not confirmed')).toBeTruthy();
    fireEvent.click(box);
    await waitFor(() => expect((screen.getByRole('checkbox', { name: 'Obligation o2' }) as HTMLInputElement).checked).toBe(true));
    expect(actions(calls)[0]).toMatchObject({ component_id: CMP, action_id: 'obligation', expected_version: 1, parameters: { obligation_id: 'o2', action: 'done' } });
    fireEvent.click(screen.getByRole('button', { name: 'Confirm the date…' }));
    const confirm = screen.getByRole('group', { name: 'Confirm the date' });
    expect(confirm.textContent).toContain('It then counts as confirmed by you.');
    fireEvent.click(within(confirm).getByRole('button', { name: 'Confirm the date' }));
    expect(await screen.findByText('Confirmed by a lawyer')).toBeTruthy();
    expect(actions(calls)[1].parameters).toEqual({ obligation_id: 'o1', action: 'confirm' });
  });

  it('a reader sees the list without controls; a refused change is said in words', async () => {
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(checklist([item('o1', 'proposed')], false))],
    });
    renderLaw('/chat');
    await ask('What are the open obligations?');
    expect(((await screen.findByRole('checkbox', { name: 'Obligation o1' })) as HTMLInputElement).disabled).toBe(true);
    expect(screen.queryByRole('button', { name: 'Confirm the date…' })).toBeNull();
    expect(screen.getByText(/ticking them is for the matter's lawyers/)).toBeTruthy();
  });
});

const approvalItem = (state = 'waiting', extra = {}) => ({ artifact_id: 'art1', kind: 'Research note', title: 'When is delivery due?', version: 1, author: 'dave',
  created_at: '2026-11-02T10:00:00.000Z', preview: ['Delivery is due by May 15, 2026.'], gaps: 0, consequence: 'Approving marks this research note as reviewed. Nothing is filed or sent.',
  path: '/m/M-101/notes/art1', state, can_decide: state === 'waiting', ...extra });
const approvals = (items: object[]) => ({ id: CMP, version: 1, type: 'approval_request', data_ref: CMP, props: {
  summary: '1 item in M-101 waits for your decision.', items, more: 0, caution: 'Open each item and read it before deciding.' } });

describe('Approval request', () => {
  it('approving needs the person to say they read that version, and shows what it does and does not do', async () => {
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(approvals([approvalItem()]))],
      'POST /api/legal/counsel/actions': () => [200, { component: approvals([approvalItem('approved', { decided_by: 'alice' })]) }],
    });
    renderLaw('/chat');
    await ask('What is waiting for my approval?');
    fireEvent.click(await screen.findByRole('button', { name: 'Approve…' }));
    const form = screen.getByRole('form', { name: 'Approve When is delivery due?' });
    expect(form.textContent).toContain('Nothing is filed or sent.');
    const go = within(form).getByRole('button', { name: 'Approve version 1' }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);
    fireEvent.click(within(form).getByRole('checkbox', { name: 'I have read version 1 in full.' }));
    fireEvent.click(go);
    expect(await screen.findByText('Approved by alice')).toBeTruthy();
    expect(actions(calls)[0]).toMatchObject({ action_id: 'decide', parameters: { artifact_id: 'art1', decision: 'approved', read_it: true } });
  });

  it('asking for changes needs a reason; a draft with gaps cannot be approved', async () => {
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(approvals([approvalItem('waiting', { gaps: 2, kind: 'Legal draft' })]))],
      'POST /api/legal/counsel/actions': () => [200, { component: approvals([approvalItem('changes_requested', { decided_by: 'alice', gaps: 2, kind: 'Legal draft' })]) }],
    });
    renderLaw('/chat');
    await ask('What is waiting for my approval?');
    expect(((await screen.findByRole('button', { name: 'Approve…' })) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('2 gaps to fill before it can be approved.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Ask for changes…' }));
    fireEvent.click(screen.getByRole('button', { name: 'Send back' }));
    expect(await screen.findByText('Say what should change.')).toBeTruthy();
    expect(actions(calls)).toHaveLength(0);
    fireEvent.change(screen.getByRole('textbox', { name: 'What should change?' }), { target: { value: 'Name the signatory.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send back' }));
    expect(await screen.findByText('Sent back for changes by alice')).toBeTruthy();
    expect(actions(calls)[0].parameters).toMatchObject({ decision: 'changes_requested', reason: 'Name the signatory.' });
  });
});

const form = (extra = {}) => ({ id: CMP, version: 1, type: 'guided_questions', props: {
  purpose: 'draft', title: 'Draft from a firm template', intro: 'Three questions.', submit_label: 'Write the draft', note: 'A second lawyer approves.',
  questions: [
    { id: 'template', label: 'Template', kind: 'choice', default: 'tpl1', options: [{ value: 'tpl1', label: 'Engagement letter' }, { value: 'tpl2', label: 'Demand letter' }] },
    { id: 'title', label: 'Title of the draft', kind: 'text', default: 'Engagement letter for Harrowgate', max_length: 200 },
    { id: 'note', label: 'What the written passages should say', kind: 'text', default: 'Draft an engagement letter', required: false, max_length: 1000 }],
  ...extra } });
const preview = (version = 1, gaps = 1) => ({ id: `cmp_${hex('8')}`, version: 1, type: 'document_preview', data_ref: `cmp_${hex('8')}`, props: {
  artifact_id: 'art9', title: 'Letter to Harrowgate', template: 'Engagement letter', version, status: 'draft', gaps, sources: 3,
  fields: [{ label: 'Signatory', value: 'Needs input', status: gaps ? 'needs_input' : 'entered' }],
  blocks: [{ type: 'heading', text: 'Engagement letter', origin: 'template' }, { type: 'paragraph', text: 'Signed by [[NEEDS INPUT: signatory]].', origin: 'template' }],
  more_blocks: 0, review: 'In the review queue.', path: '/m/M-101/drafting/art9', can_edit: true } });

describe('A short form, then a draft beside the chat', () => {
  it('answers the questions (with their suggestions) and sends them as the next turn; the draft opens beside the chat', async () => {
    let n = 0;
    const calls = world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => { n += 1; return [200, n === 1 ? stream(form()) : stream(preview(), `turn_${hex('2')}`)]; },
      'GET /api/legal/matters/id-M-101/artifacts/art9': () => [200, { artifact: { id: 'art9', version: 1, status: 'draft', content_sha256: hex('a'), decisions: [],
        content: { title: 'Letter to Harrowgate', matter: { number: 'M-101', title: 'MSA renewal', client: 'Alpha' }, gaps: 1,
          fields: [{ key: 'signatory', label: 'Signatory', kind: 'text', source: 'person', value: null, status: 'needs_input', sources: [], suggestion: null }],
          blocks: [{ type: 'paragraph', origin: 'template', text: 'Signed by [[NEEDS INPUT: signatory]].', raw: '', sources: [] }],
          template: { name: 'Engagement letter' }, summary: {} }, created_by: 'u', created_at: 't', origin: 'model', inputs: {} },
        versions: [{ id: 'art9', version: 1, status: 'draft', created_at: 't', origin: 'model' }], can: { approve: false, file: false, author: true } }],
    });
    renderLaw('/chat');
    await ask('Draft an engagement letter for Harrowgate');
    const f = await screen.findByRole('form', { name: 'Draft from a firm template' });
    expect((within(f).getByRole('radio', { name: 'Engagement letter' }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(within(f).getByRole('radio', { name: 'Demand letter' }));
    fireEvent.change(within(f).getByRole('textbox', { name: 'Title of the draft' }), { target: { value: 'Letter to Harrowgate' } });
    fireEvent.click(within(f).getByRole('button', { name: 'Write the draft' }));
    await waitFor(() => expect(calls.filter((c) => c.path === '/api/legal/counsel/conversations/conv_1/turns')).toHaveLength(2));
    expect(await screen.findByRole('heading', { name: 'Letter to Harrowgate' })).toBeTruthy();
    const sent = calls.filter((c) => c.path === '/api/legal/counsel/conversations/conv_1/turns')[1].body;
    expect(sent.text).toBe('Write the draft “Letter to Harrowgate”');
    expect(sent.form).toEqual({ component_id: CMP, values: { template: 'tpl2', title: 'Letter to Harrowgate', note: 'Draft an engagement letter' } });
    expect(sent.after_turn_id).toBe(T1);
    expect(screen.getByText('signatory', { selector: 'mark' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Open beside the chat to correct' }));
    const panel = await screen.findByRole('complementary', { name: 'Letter to Harrowgate' });
    expect(await within(panel).findByRole('button', { name: 'Fill in or correct' })).toBeTruthy();
    expect(within(panel).getByText(/Nothing is filed or sent from here/)).toBeTruthy();
  });

  it('a required answer left empty keeps the form from being sent', async () => {
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(form())] });
    renderLaw('/chat');
    await ask('Draft an engagement letter');
    const f = await screen.findByRole('form', { name: 'Draft from a firm template' });
    fireEvent.change(within(f).getByRole('textbox', { name: 'Title of the draft' }), { target: { value: '  ' } });
    expect((within(f).getByRole('button', { name: 'Write the draft' }) as HTMLButtonElement).disabled).toBe(true);
    expect(within(f).getByText('“Title of the draft” is needed.')).toBeTruthy();
  });
});

describe('Panel, chart, graph, live meeting', () => {
  it('the panel card asks the server again while the panel works, then shows where the perspectives differ', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const running = { id: CMP, version: 1, type: 'council_summary', data_ref: CMP, props: { review_id: 'rv1', question: 'Is the fee binding?', area: 'commercial contracts',
      status: 'running', issues: [], missing_facts: [], path: '/m/M-101/panel/rv1', caution: 'Options, never a decision.' } };
    const done = { ...running, props: { ...running.props, status: 'done', approval: 'Waiting for a partner\'s approval',
      issues: [{ issue: 'Is the fixed fee binding?', disagreement: true, positions: [{ role: 'Deal lawyer', text: 'Yes, as agreed.' }, { role: 'Litigator', text: 'Arguably not.' }] }] } };
    let reads = 0;
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(running)],
      [`GET /api/legal/counsel/components/${CMP}`]: () => { reads += 1; return [200, { component: reads >= 2 ? done : running }]; },
    });
    renderLaw('/chat');
    await ask('Challenge our position on the fee');
    expect(await screen.findByText('The four perspectives are reading the documents…')).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(6500); });
    expect(await screen.findByText('They differ')).toBeTruthy();
    expect(screen.getByText('Arguably not.')).toBeTruthy();
    const n = reads;
    await act(async () => { await vi.advanceTimersByTimeAsync(6500); });
    expect(reads).toBe(n);                                              // done: no more asking
  });

  it('the chart draws stated amounts as bars, each opening its sentence; one currency, never added', async () => {
    const chart = { id: CMP, version: 1, type: 'numeric_chart', props: { summary: '2 amounts in USD stated in 1 document of M-101.', unit: 'USD', more: 0,
      other_units: ['EUR'], caution: 'Not added or converted.', bars: [
        { id: 'am1', shown: 'USD 120,000', value: 120000, document_title: 'Services Agreement', clause: '9.1', source: side },
        { id: 'am2', shown: 'USD 12,000', value: 12000, document_title: 'Side letter', source: { ...side, document_title: 'Side letter' } }] } };
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(chart)] });
    renderLaw('/chat');
    await ask('Chart the amounts in the documents');
    const bars = await screen.findByRole('list', { name: 'Amounts in USD' });
    expect(within(bars).getAllByRole('button').map((b) => b.getAttribute('aria-label'))).toEqual([
      'USD 120,000, Services Agreement, clause 9.1: open its sentence', 'USD 12,000, Side letter: open its sentence']);
    expect(screen.getByText(/Amounts in EUR are also stated; they are not shown here, and never converted/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Side letter' }));
    expect(within(bars).getAllByRole('button')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: 'All documents' }));
    fireEvent.click(within(bars).getAllByRole('button')[0]);
    expect(await screen.findByRole('complementary', { name: 'USD 120,000: Services Agreement' })).toBeTruthy();
  });

  it('the graph shows parties and the documents naming them; choosing a party lists where, each with its sentence', async () => {
    const graph = { id: CMP, version: 1, type: 'graph_excerpt', props: { summary: '2 parties named across 1 document of M-101.', more: 0, caution: 'Not legal relationships.',
      nodes: [{ id: 'p1', label: 'Morgan Company LLC', kind: 'party', role: 'Client' }, { id: 'p2', label: 'Apex Industries, Inc.', kind: 'party' },
              { id: 'd1', label: 'Services Agreement', kind: 'document' }],
      edges: [{ id: 'e1', source: 'p1', target: 'd1', label: 'party to', evidence: side }, { id: 'e2', source: 'p2', target: 'd1', label: 'named in', evidence: side }] } };
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(graph)] });
    renderLaw('/chat');
    await ask('How are the parties connected?');
    const node = await screen.findByRole('button', { name: 'Morgan Company LLC, Client: named in 1 document' });
    fireEvent.keyDown(node, { key: 'Enter' });
    const where = screen.getByRole('list', { name: 'Where Morgan Company LLC is named' });
    expect(where.textContent).toContain('Party to Services Agreement');
    fireEvent.click(within(where).getByRole('button', { name: 'Inspect' }));
    expect(await screen.findByRole('complementary', { name: 'Morgan Company LLC in Services Agreement' })).toBeTruthy();
    expect(screen.getByText('As a list')).toBeTruthy();
  });

  it('the meeting card says whether it is recording and keeps following only for a while', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const live = (segments: object[]) => ({ id: CMP, version: 1, type: 'live_transcript', data_ref: CMP, props: { meeting_id: 'mtg1', title: 'Client call (fictional)',
      state: 'recording', consent: 'Consent recorded for 2 of 2 participants.', segments, earlier: 0, path: '/m/M-101/meetings/mtg1', caution: 'Machine speech-to-text.' } });
    let reads = 0;
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(live([{ speaker: 'Alice', text: 'We expect delivery by the fifteenth.' }]))],
      [`GET /api/legal/counsel/components/${CMP}`]: () => { reads += 1; return [200, { component: live([{ speaker: 'Alice', text: 'We expect delivery by the fifteenth.' }, { text: 'The twelfth was dispatch.' }]) }]; },
    });
    renderLaw('/chat');
    await ask('Show the live transcript');
    expect(await screen.findByText('Recording', { selector: '[data-component="live_transcript"] .qc-status' })).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(4500); });
    expect(await screen.findByText('The twelfth was dispatch.')).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(11 * 60_000); });
    const n = reads;
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
    expect(reads).toBe(n);
    expect(screen.getByRole('button', { name: 'Keep following' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Open the meeting' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /stop|pause|record/i })).toBeNull();      // the chat never holds the microphone
  });
});

describe('The contract, in the browser', () => {
  it('a card that does not match becomes an inert placeholder', () => {
    const bad = [
      checklist([{ ...item('o1', 'proposed'), path: 'https://evil.test/' }]),
      { ...checklist([]), data_ref: undefined },
      form({ questions: [{ id: 'template', label: 'T', kind: 'choice', default: 'zzz', options: [{ value: 'a', label: 'A' }] }] }),
      form({ questions: [1, 2, 3, 4].map((i) => ({ id: `q${i}`, label: 'Q', kind: 'text' })) }),
      { id: CMP, version: 1, type: 'graph_excerpt', props: { summary: 's', caution: 'c', more: 0, nodes: [{ id: 'p1', label: 'A', kind: 'party' }],
        edges: [{ id: 'e1', source: 'p1', target: 'd9', label: 'named in', evidence: side }] } },
      { id: CMP, version: 1, type: 'numeric_chart', props: { summary: 's', unit: 'USD', caution: 'c', more: 0, other_units: [],
        bars: [{ id: 'b1', shown: 'x', value: Number.POSITIVE_INFINITY, document_title: 'D', source: side }] } },
      { ...preview(), props: { ...preview().props, path: '/admin' } },
    ];
    for (const b of bad) expect(parseComponent(b).type).toBe('inert');
    expect(parseComponent(checklist([item('o1', 'proposed')])).type).toBe('checklist');
    expect(parseComponent(preview()).type).toBe('document_preview');
  });
});
