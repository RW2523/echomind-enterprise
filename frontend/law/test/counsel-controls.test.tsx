import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { _resetCaptureForTests } from '../app/capture';
import { visiblePath } from '../counsel/ChatPage';
import { plainAnswer } from '../counsel/copy';
import { blocks } from '../counsel/format';
import type { ChatTurn } from '../counsel/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** Chat controls like other assistants (owner's request 2026-11-02): formatted answers, copy, ask again and edit as
 *  versions, rating, adding a document and dictating from the question box, and the history's search, rename and delete. */
const hex = (c: string) => c.repeat(32);
const CIT = `cit_${hex('a')}`;
const T1 = `turn_${hex('1')}`;
const T2 = `turn_${hex('2')}`;
const T3 = `turn_${hex('3')}`;

const envelope = (id: string, answer: string, cmp = hex('c')) => ({
  schema_version: '1.0', message_id: id, matter_id: 'id-M-101', summary: 'Answered from 1 source, 1 with a verified quote.',
  components: [{ id: `cmp_${cmp}`, version: 1, type: 'source_answer', props: {
    answer, state: 'answered',
    citations: [{ id: CIT, n: 1, document_id: 'doc1', document_title: 'Services Agreement', version_number: 1, page: 2, clause: '4.2', verified: true }],
    unsupported: [], notes: [] } }],
  suggested_actions: [],
});

const stream = (turnId: string, answer: string, parent: string | null = null, cmp = hex('c')) => ({ __ndjson: [
  { event: 'turn.started', data: { turn_id: turnId, conversation_id: 'conv_1', matter_id: 'id-M-101', intent: 'ask_fact', parent_turn_id: parent } },
  { event: 'answer.delta', data: { text: answer } },
  { event: 'ui.final', data: envelope(turnId, answer, cmp) },
  { event: 'turn.completed', data: { turn_id: turnId, status: 'completed' } },
] });

function world(handlers: Record<string, (body: any) => [number, unknown]> = {}, opts: { role?: string; features?: Record<string, boolean> } = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, { ...me(), features: { intake: false, intake_links: false, ...(opts.features ?? {}) } }],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: (opts.role ?? 'attorney') as any })] }],
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

const turnCalls = (calls: { method: string; path: string; body: any }[]) =>
  calls.filter((c) => c.method === 'POST' && c.path === '/api/legal/counsel/conversations/conv_1/turns').map((c) => c.body);

beforeEach(() => { localStorage.clear(); _resetCaptureForTests(); localStorage.setItem('echomind.counsel.matter', 'id-M-101'); });
afterEach(() => { cleanup(); localStorage.clear(); vi.restoreAllMocks(); });

describe('Answer formatting', () => {
  it('reads headings, lists and tables, and leaves everything else a paragraph', () => {
    expect(blocks('## Payment\n- Fee: USD 120,000 [1]\n- Due: 30 days\n\nAfter the list.\n\n1. First\n2. Second\n\n| Item | Amount |\n|---|---|\n| Fee | USD 1 |'))
      .toEqual([
        { kind: 'h', text: 'Payment' }, { kind: 'ul', items: ['Fee: USD 120,000 [1]', 'Due: 30 days'] }, { kind: 'p', text: 'After the list.' },
        { kind: 'ol', start: 1, items: ['First', 'Second'] }, { kind: 'table', head: ['Item', 'Amount'], rows: [['Fee', 'USD 1']] }]);
    expect(blocks('4.2 Delivery is due.\n9.1 Fees apply.')).toEqual([{ kind: 'p', text: '4.2 Delivery is due. 9.1 Fees apply.' }]);
    expect(blocks('- one\ncontinued\n\nText\n- two')).toEqual([{ kind: 'ul', items: ['one continued'] }, { kind: 'p', text: 'Text' }, { kind: 'ul', items: ['two'] }]);
    expect(blocks('| not | a table |\nno rule')).toEqual([{ kind: 'p', text: '| not | a table | no rule' }]);
    expect(blocks('**Delivery**\nBy May 15.')).toEqual([{ kind: 'h', text: 'Delivery' }, { kind: 'p', text: 'By May 15.' }]);
  });

  it('shows them as text with chips; a link, an image or markup in an answer stays words', async () => {
    const answer = '## Delivery\n- Apex must deliver by **May 15, 2026** [1].\n- A made-up source [7].\n\n| Party | Duty |\n|---|---|\n| Apex | Deliver [1] |\n\n'
      + 'See [the site](javascript:alert(1)) and ![x](http://evil.test/x.png) <img src=x onerror=alert(1)>.';
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(T1, answer)] });
    const { container } = renderLaw('/chat');
    await ask('When is delivery due?');
    expect(await screen.findByRole('heading', { name: 'Delivery', level: 4 })).toBeTruthy();
    const card = container.querySelector('[data-component="source_answer"]')!;
    const items = within(card as HTMLElement).getAllByRole('listitem');
    expect(items[0].textContent).toContain('Apex must deliver by May 15, 2026');
    expect(items[0].querySelector('strong')?.textContent).toBe('May 15, 2026');
    expect(items[1].textContent).not.toContain('[7]');                         // a marker without a source is dropped
    const table = within(card as HTMLElement).getByRole('table');
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Party', 'Duty']);
    expect(within(table).getByRole('button', { name: /Source 1: Services Agreement/ })).toBeTruthy();
    expect(card.querySelector('a, img, script')).toBeNull();
    expect(card.textContent).toContain('<img src=x onerror=alert(1)>');
    expect(card.textContent).not.toContain('**');
  });
});

describe('Copy', () => {
  it('copies the answer with the numbers the reader saw, its sources and whether each quote was verified', async () => {
    const write = vi.fn(async (_text: string) => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(T1, 'Delivery is due by **May 15, 2026** [1][1]. Fees apply [9].')] });
    renderLaw('/chat');
    await ask('When is delivery due?');
    fireEvent.click(await screen.findByRole('button', { name: 'Copy answer' }));
    await waitFor(() => expect(write).toHaveBeenCalledTimes(1));
    expect(write.mock.calls[0][0]).toBe('Delivery is due by May 15, 2026 [1]. Fees apply.\n\nSources:\n[1] Services Agreement · p. 2 · cl. 4.2 (verified quote)');
    expect(await screen.findByText('Copied to the clipboard.')).toBeTruthy();
  });

  it('a partly supported answer says so in the copy, and calculated dates are marked as calculated', () => {
    const text = plainAnswer({ answer: 'The term ends [1].', state: 'partial', unsupported: ['x'], notes: [], citations: [
      { id: CIT, n: 1, document_id: 'd', document_title: 'MSA', verified: false, removed: false }],
      calculation: { document_title: 'MSA', facts: [], derived: [{ label: 'Current term ends', display: '31 January 2027' }], assumptions: ['Calendar months.'], not_calculated: [] } });
    expect(text).toContain('Partly supported: some sentences have no verified source.');
    expect(text).toContain('Calculated, not stated in MSA:\n- Current term ends: 31 January 2027\n- Assumption: Calendar months.');
    expect(text).toContain('[1] MSA (quote not verified)');
  });
});

describe('Versions: ask again, edit, and what follows', () => {
  it('asks again as another version, moves between versions, and a next question follows the version shown', async () => {
    let n = 0;
    const calls = world({ 'POST /api/legal/counsel/conversations/conv_1/turns': (b) => {
      n += 1;
      if (n === 1) return [200, stream(T1, 'First answer [1].', null, hex('c'))];
      if (n === 2) return [200, stream(T2, 'Second answer [1].', null, hex('d'))];
      return [200, stream(T3, 'Follow-up answer [1].', b.after_turn_id, hex('e'))];
    } });
    renderLaw('/chat');
    await ask('When is delivery due?');
    expect(await screen.findByText(/First answer/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Ask again' }));
    expect(await screen.findByText(/Second answer/)).toBeTruthy();
    expect(screen.queryByText(/First answer/)).toBeNull();
    expect(turnCalls(calls)[1]).toMatchObject({ text: 'When is delivery due?', retry_of: T1 });
    expect(turnCalls(calls)[1].after_turn_id).toBeUndefined();
    const versions = screen.getByRole('group', { name: 'Version 2 of 2' });
    fireEvent.click(within(versions).getByRole('button', { name: 'Previous version' }));
    expect(await screen.findByText(/First answer/)).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Version 1 of 2' })).toBeTruthy();
    await ask('And the fees?');                                          // follows the version on screen
    expect(await screen.findByText(/Follow-up answer/)).toBeTruthy();
    expect(turnCalls(calls)[2]).toMatchObject({ text: 'And the fees?', after_turn_id: T1 });
    fireEvent.click(within(screen.getByRole('group', { name: 'Version 1 of 2' })).getByRole('button', { name: 'Next version' }));
    expect(await screen.findByText(/Second answer/)).toBeTruthy();
    expect(screen.queryByText(/Follow-up answer/)).toBeNull();           // it belongs to the other version
  });

  it('edits a question: the new wording is another version of that turn', async () => {
    let n = 0;
    const calls = world({ 'POST /api/legal/counsel/conversations/conv_1/turns': () => {
      n += 1;
      return [200, n === 1 ? stream(T1, 'Delivery answer [1].') : stream(T2, 'Fees answer [1].', null, hex('d'))];
    } });
    renderLaw('/chat');
    await ask('When is delivery due?');
    await screen.findByText(/Delivery answer/);
    fireEvent.click(screen.getByRole('button', { name: 'Edit question' }));
    const box = screen.getByRole('textbox', { name: 'Edit your question' });
    expect((box as HTMLTextAreaElement).value).toBe('When is delivery due?');
    fireEvent.change(box, { target: { value: 'What are the fees?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send the edited question' }));
    expect(await screen.findByText(/Fees answer/)).toBeTruthy();
    expect(turnCalls(calls)[1]).toMatchObject({ text: 'What are the fees?', retry_of: T1 });
    expect(screen.getByRole('group', { name: 'Version 2 of 2' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Edit question' }));
    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Edit your question' }), { key: 'Escape' });
    expect(screen.queryByRole('textbox', { name: 'Edit your question' })).toBeNull();
  });

  it('opens a saved conversation on the newest version of each step', async () => {
    world({ 'GET /api/legal/counsel/conversations/conv_1': () => [200, { conversation: { id: 'conv_1', matter_id: 'id-M-101', title: 'Delivery' }, turns: [
      { id: T1, seq: 1, parent_turn_id: null, question: 'When is delivery due?', intent: 'ask_fact', status: 'completed', created_at: 't',
        envelope: envelope(T1, 'Old answer [1].', hex('c')) },
      { id: T2, seq: 2, parent_turn_id: null, question: 'When is delivery due?', intent: 'ask_fact', status: 'completed', created_at: 't',
        envelope: envelope(T2, 'New answer [1].', hex('d')), feedback: { rating: 'up' } },
    ] }] });
    renderLaw('/chat/conv_1');
    expect(await screen.findByText(/New answer/)).toBeTruthy();
    expect(screen.queryByText(/Old answer/)).toBeNull();
    expect(screen.getByRole('group', { name: 'Version 2 of 2' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Helpful' }).getAttribute('aria-pressed')).toBe('true');
  });

  it('the path through versions picks the chosen one, else the newest, and stops at a turn never started', () => {
    const t = (key: string, parent: string | null, id?: string): ChatTurn => ({ key, id, parent, question: key, status: 'completed', text: '' });
    const turns = [t('a', null, 'A'), t('b', null, 'B'), t('c', 'A', 'C'), t('d', 'B', 'D'), t('e', 'D')];
    expect(visiblePath(turns, {}).map((s) => s.turn.key)).toEqual(['b', 'd', 'e']);
    expect(visiblePath(turns, { '': 'a' }).map((s) => s.turn.key)).toEqual(['a', 'c']);
    expect(visiblePath(turns, {})[0].versions.map((v) => v.key)).toEqual(['a', 'b']);
  });
});

describe('Rating an answer', () => {
  it('thumbs up, then thumbs down with a reason; pressing again removes it', async () => {
    const puts: any[] = [];
    world({
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(T1, 'Delivery answer [1].')],
      [`PUT /api/legal/counsel/turns/${T1}/feedback`]: (b) => { puts.push(b); return [200, { feedback: b.rating ? { rating: b.rating, ...(b.reason ? { reason: b.reason } : {}) } : null }]; },
    });
    renderLaw('/chat');
    await ask('When is delivery due?');
    const up = await screen.findByRole('button', { name: 'Helpful' });
    fireEvent.click(up);
    await waitFor(() => expect(up.getAttribute('aria-pressed')).toBe('true'));
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }));
    const form = await screen.findByRole('form', { name: 'What was wrong?' });
    fireEvent.click(within(form).getByRole('button', { name: 'Wrong or weak source' }));
    fireEvent.change(within(form).getByRole('textbox', { name: 'Say more (optional)' }), { target: { value: 'Clause 4.1 governs.' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Send' }));
    expect(await screen.findByText('Thank you. Your rating is saved with this answer.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }));
    await waitFor(() => expect(puts.length).toBe(4));
    expect(puts).toEqual([{ rating: 'up' }, { rating: 'down' }, { rating: 'down', reason: 'wrong_source', note: 'Clause 4.1 governs.' }, { rating: null }]);
    expect(screen.getByRole('button', { name: 'Not helpful' }).getAttribute('aria-pressed')).toBe('false');
  });
});

describe('The question box', () => {
  const docsList = (status: string) => [200, { documents: [{ id: 'docN', title: 'side-letter', kind: 'other', status: 'draft', version: 1,
    current_version: { id: 'v1', number: 1, filename: 'side-letter.txt', index_status: status } }] }] as [number, unknown];

  it('a contributor adds a document: it joins the matter as a draft, and asking waits until it has been read', async () => {
    let state = 'indexing';
    const calls = world({
      'GET /api/legal/matters/id-M-101/documents': () => docsList(state),
      'POST /api/legal/matters/id-M-101/documents/import': () => [202, { document: { id: 'docN', title: 'side-letter' } }],
      'POST /api/legal/counsel/conversations/conv_1/turns': () => [200, stream(T1, 'From the side letter [1].')],
    });
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: /Choose documents/ }));
    expect(await screen.findByRole('button', { name: 'Add a document to M-101…' })).toBeTruthy();
    const file = new File(['Side letter text'], 'side-letter.txt', { type: 'text/plain' });
    fireEvent.change(screen.getByTestId('add-document-input'), { target: { files: [file] } });
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/matters/id-M-101/documents/import')).toBe(true));
    expect(calls.find((c) => c.path === '/api/legal/matters/id-M-101/documents/import')!.body).toMatchObject({ file: 'side-letter.txt', status: 'draft', title: 'side-letter' });
    expect(await screen.findByText(/Wait until “side-letter” has been read/)).toBeTruthy();
    const box = screen.getByRole('textbox', { name: 'Ask EchoMind' });
    fireEvent.change(box, { target: { value: 'What does the side letter say?' } });
    expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true);
    state = 'indexed';
    await waitFor(() => expect(screen.queryByText(/Wait until/)).toBeNull(), { timeout: 4000 });
    fireEvent.keyDown(box, { key: 'Enter' });
    await screen.findByText(/From the side letter/);
    expect(turnCalls(calls)[0].document_ids).toEqual(['docN']);
  });

  it('a file of another type is refused before it is sent, and a read-only member is not offered adding', async () => {
    world({ 'GET /api/legal/matters/id-M-101/documents': () => docsList('indexed') }, { role: 'read_only' });
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: /Choose documents/ }));
    await screen.findByRole('dialog', { name: 'Answer only from these documents' });
    expect(screen.queryByRole('button', { name: /Add a document/ })).toBeNull();
    cleanup();
    const calls = world({ 'GET /api/legal/matters/id-M-101/documents': () => docsList('indexed') });
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: /Choose documents/ }));
    await screen.findByRole('button', { name: 'Add a document to M-101…' });
    fireEvent.change(screen.getByTestId('add-document-input'), { target: { files: [new File(['x'], 'macro.xlsm')] } });
    expect(await screen.findByText('Only PDF, Word (.docx), text and Markdown files can be added.')).toBeTruthy();
    expect(calls.some((c) => c.path.endsWith('/documents/import'))).toBe(false);
  });

  it('dictation is offered only where voice is turned on', async () => {
    world();
    renderLaw('/chat');
    await screen.findByRole('textbox', { name: 'Ask EchoMind' });
    expect(screen.queryByRole('button', { name: 'Dictate your question' })).toBeNull();
    cleanup();
    world({}, { features: { conversation: true } });
    renderLaw('/chat');
    expect(await screen.findByRole('button', { name: 'Dictate your question' })).toBeTruthy();
  });
});

describe('History: search, rename, delete', () => {
  const summary = (id: string, title: string, extra = {}) => ({ id, title, created_at: 't', updated_at: 't', turns: 1,
    matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, ...extra });

  it('searches the person\'s own conversations by title or question', async () => {
    const calls = world({
      'GET /api/legal/counsel/conversations': () => [200, { conversations: [summary('conv_a', 'Delivery'), summary('conv_b', 'Payment terms')] }],
      'GET /api/legal/counsel/conversations?q=late+fee': () => [200, { conversations: [summary('conv_a', 'Delivery', { match: 'Is there a late fee?' })] }],
    });
    renderLaw('/chat');
    const recent = await screen.findByRole('navigation', { name: 'Recent conversations' });
    expect(within(recent).getAllByRole('link').map((a) => a.querySelector('span')!.textContent)).toEqual(['Delivery', 'Payment terms']);
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search your conversations' }), { target: { value: 'late fee' } });
    expect(await screen.findByText('Is there a late fee?', {}, { timeout: 2000 })).toBeTruthy();
    const found = screen.getByRole('navigation', { name: 'Conversations matching late fee' });
    expect(within(found).getAllByRole('link')).toHaveLength(1);
    expect(calls.some((c) => c.path === '/api/legal/counsel/conversations?q=late+fee')).toBe(true);
  });

  it('renames and deletes from the conversation\'s menu; deleting the open one goes back to a new chat', async () => {
    let list = [summary('conv_1', 'Delivery')];
    const calls = world({
      'GET /api/legal/counsel/conversations': () => [200, { conversations: list }],
      'PATCH /api/legal/counsel/conversations/conv_1': (b) => { list = [summary('conv_1', b.title)]; return [200, { conversation: { id: 'conv_1', title: b.title } }]; },
      'DELETE /api/legal/counsel/conversations/conv_1': () => { list = []; return [200, null]; },     // 204 on the server
      'GET /api/legal/counsel/conversations/conv_1': () => [200, { conversation: { id: 'conv_1', matter_id: 'id-M-101', title: 'Delivery' }, turns: [] }],
    });
    const { router } = renderLaw('/chat/conv_1');
    fireEvent.click(await screen.findByRole('button', { name: 'More for Delivery' }));
    fireEvent.click(screen.getByRole('button', { name: 'Rename' }));
    const name = screen.getByRole('textbox', { name: 'Conversation name' });
    fireEvent.change(name, { target: { value: 'Delivery date' } });
    fireEvent.submit(name.closest('form')!);
    expect(await screen.findByRole('button', { name: 'More for Delivery date' })).toBeTruthy();
    expect(calls.find((c) => c.method === 'PATCH')!.body).toEqual({ title: 'Delivery date' });
    fireEvent.click(screen.getByRole('button', { name: 'More for Delivery date' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    const confirm = screen.getByRole('alertdialog', { name: 'Delete Delivery date?' });
    expect(confirm.textContent).toContain('The firm keeps its record');
    fireEvent.click(within(confirm).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/chat'));
    expect(calls.some((c) => c.method === 'DELETE' && c.path === '/api/legal/counsel/conversations/conv_1')).toBe(true);
  });
});
