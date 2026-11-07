import React from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { matter, me, mockApi, renderLaw } from './harness';
import { sortFiles } from '../ui/bulkUpload';
import { parseComponent } from '../counsel/validate';

const TRACKS = [
  { key: 'disputes', label: 'Disputes', stages: [
    { key: 'intake', label: 'Intake & conflicts', tasks: ['Run the conflict check'] },
    { key: 'pleadings', label: 'Pleadings', tasks: ['Draft the claim or defence', 'Serve the claim or defence'] },
    { key: 'hearing', label: 'Hearing / trial', tasks: [] }] },
  { key: 'advisory', label: 'Advisory', stages: [{ key: 'instruction', label: 'Instruction', tasks: [] }, { key: 'research', label: 'Research & analysis', tasks: [] }] },
];
const M150 = matter('M-150', 'Supply dispute', 'Alpha Holdings', { track: 'disputes', stage: 'pleadings', stage_since: '2026-11-01T00:00:00Z',
  description: 'Late deliveries under the supply agreement.' });
const UPDATE = { id: 'u1', matter_id: 'id-M-150', author_id: 'usr_alice', author: 'Alice Moreau', kind: 'call', body: 'Client accepts the new date.',
  occurred_on: '2026-11-04', created_at: '2026-11-04T10:00:00Z' };
const TASK = { id: 't1', matter_id: 'id-M-150', title: 'Serve the claim or defence', kind: 'filing', due_on: '2026-11-20', stage: 'pleadings', status: 'open',
  source: 'standard', created_by: 'usr_alice', created_by_name: 'Alice Moreau', created_at: '', updated_at: '', version: 1, overdue: false };

function signedIn(m = M150, extra: Record<string, (b: any) => [number, unknown]> = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [m] }],
    'GET /api/legal/clients': () => [200, { clients: [{ id: 'c-Alpha Holdings', name: 'Alpha Holdings', status: 'active', version: 1 }] }],
    'GET /api/legal/pipeline/tracks': () => [200, { tracks: TRACKS }],
    'GET /api/legal/matters/id-M-150/updates': () => [200, { updates: [UPDATE] }],
    'GET /api/legal/matters/id-M-150/tasks': () => [200, { tasks: [TASK] }],
    'GET /api/legal/matters/id-M-150/memberships': () => [200, { memberships: [] }],
    'GET /api/legal/matters/id-M-150/documents': () => [200, { documents: [] }],
    'GET /api/legal/matters/id-M-150/parties': () => [200, { parties: [], can_manage: true }],
    'GET /api/legal/matters/id-M-150/obligations': () => [200, { obligations: [] }],
    ...extra,
  });
}

describe('Case management', () => {
  afterEach(() => cleanup());

  it('shows the stage, the updates and the plan on the overview, and posts an update', async () => {
    const calls = signedIn(M150, { 'POST /api/legal/matters/id-M-150/updates': () => [200, { update: UPDATE }] });
    renderLaw('/m/M-150/overview');
    const stages = await screen.findByRole('list', { name: 'Disputes stages' });
    expect(within(stages).getAllByRole('listitem').map((li) => li.getAttribute('data-state'))).toEqual(['done', 'current', 'next']);
    expect(screen.getByRole('button', { name: /Move to Hearing \/ trial/ })).toBeTruthy();
    const feed = await screen.findByRole('list', { name: 'Updates' });
    expect(within(feed).getByText('Client accepts the new date.')).toBeTruthy();
    expect(within(await screen.findByRole('list', { name: 'Pleadings tasks' })).getByText('Serve the claim or defence')).toBeTruthy();
    expect(screen.getByText('Late deliveries under the supply agreement.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Post an update/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Post an update' });
    fireEvent.change(within(dialog).getByLabelText('What happened'), { target: { value: 'hearing' } });
    fireEvent.change(within(dialog).getByLabelText('Update'), { target: { value: 'The court listed the hearing.' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Post' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/legal/matters/id-M-150/updates')).toBe(true));
    const sent = calls.find((c) => c.method === 'POST' && c.path === '/api/legal/matters/id-M-150/updates')!.body;
    expect([sent.kind, sent.body]).toEqual(['hearing', 'The court listed the hearing.']);
  });

  it('a read-only member reads the updates and the plan but cannot post, plan or move the stage', async () => {
    signedIn({ ...M150, my_role: 'read_only' });
    renderLaw('/m/M-150/overview');
    await screen.findByRole('list', { name: 'Updates' });
    await screen.findByRole('list', { name: 'Pleadings tasks' });
    expect(screen.queryByRole('button', { name: /Post an update/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Add to the plan/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Move to/ })).toBeNull();
    expect(screen.queryByRole('checkbox')).toBeNull();
  });

  it('the pipeline shows the matters by stage and moves one with the matter\'s version', async () => {
    const row = { id: 'id-M-150', number: 'M-150', title: 'Supply dispute', client: 'Alpha Holdings', status: 'open', track: 'disputes', stage: 'pleadings',
      stage_label: 'Pleadings', days_in_stage: 6, my_role: 'lead', leads: ['Alice Moreau'], last_update: { kind: 'call', excerpt: 'Client accepts the new date.', occurred_on: '2026-11-04' },
      next_task: { title: 'Serve the claim', kind: 'filing', due_on: '2026-11-20' }, open_tasks: 1, overdue: 0 };
    const calls = signedIn(M150, { 'GET /api/legal/pipeline': () => [200, { matters: [row], tracks: TRACKS }],
                                   'POST /api/legal/matters/id-M-150/stage': () => [200, { matter: M150 }] });
    renderLaw('/pipeline');
    const col = await screen.findByRole('region', { name: 'Pleadings: 1' });
    expect(within(col).getByText('Supply dispute')).toBeTruthy();
    expect(within(col).getByText(/6 days in this stage/)).toBeTruthy();
    expect(screen.getByRole('region', { name: 'Intake & conflicts: 0' })).toBeTruthy();
    fireEvent.change(within(col).getByLabelText('Move M-150 to'), { target: { value: 'hearing' } });
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path.endsWith('/stage'))).toBe(true));
    expect(calls.find((c) => c.path.endsWith('/stage'))!.body).toEqual({ stage: 'hearing', version: 1 });
  });

  it('a new case is opened with its team and the other side, then offers its documents', async () => {
    const calls = signedIn(M150, {
      'GET /api/legal/matter-numbers/next?prefix=M-': () => [200, { number: 'M-151' }],
      'GET /api/legal/colleagues': () => [200, { colleagues: [{ username: 'carol', display_name: 'Carol Pike', lawyer: true }] }],
      'POST /api/legal/matters': () => [200, { matter: { ...M150, id: 'id-M-151', number: 'M-151' } }],
    });
    renderLaw('/matters/new');
    await waitFor(() => expect((screen.getByLabelText('Matter number') as HTMLInputElement).value).toBe('M-151'));
    fireEvent.change(screen.getByLabelText('Client'), { target: { value: 'c-Alpha Holdings' } });
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Late delivery claim' } });
    fireEvent.change(screen.getByLabelText('Practice area'), { target: { value: 'Commercial litigation' } });
    await waitFor(() => expect((screen.getByLabelText('Track') as HTMLSelectElement).value).toBe('disputes'));
    fireEvent.click(screen.getByRole('button', { name: /Next: people/ }));
    fireEvent.change(await screen.findByLabelText('Add a colleague'), { target: { value: 'carol' } });
    fireEvent.change(screen.getByLabelText('Party name'), { target: { value: 'Example Supplier plc' } });
    fireEvent.click(screen.getByRole('button', { name: /Add party/ }));
    fireEvent.click(screen.getByRole('button', { name: /Open the matter/ }));
    expect(await screen.findByText('M-151 is open')).toBeTruthy();
    const sent = calls.find((c) => c.method === 'POST' && c.path === '/api/legal/matters')!.body;
    expect(sent).toMatchObject({ client_id: 'c-Alpha Holdings', number: 'M-151', title: 'Late delivery claim', track: 'disputes', stage: 'intake',
      members: [{ username: 'carol', role: 'attorney' }], parties: [{ name: 'Example Supplier plc', role: 'adverse' }] });
    expect(screen.getByRole('button', { name: /Choose a folder/ })).toBeTruthy();
  });

  it('importing cases previews every row and opens only the ready ones', async () => {
    const rows = [
      { row: 2, number: 'M-301', title: 'Lease renewal', client: 'Alpha Holdings', client_id: 'c-Alpha Holdings', new_client: false, practice_area: null,
        jurisdiction: null, sensitivity: 'restricted', track: 'transactions', stage: 'diligence', stage_label: 'Due diligence', description: null, other_parties: [], errors: [] },
      { row: 3, number: 'M-101', title: 'Duplicate', client: 'Alpha Holdings', client_id: 'c-Alpha Holdings', new_client: false, practice_area: null,
        jurisdiction: null, sensitivity: 'restricted', track: 'advisory', stage: 'instruction', stage_label: 'Instruction', description: null, other_parties: [],
        errors: ['A matter with this number already exists.'] }];
    const calls = signedIn(M150, {
      'POST /api/legal/matters-import?preview=true': () => [200, { rows, ready: 1, created: [] }],
      'POST /api/legal/matters-import?preview=false': () => [200, { rows, ready: 1, created: [{ row: 2, id: 'x', number: 'M-301' }] }],
    });
    renderLaw('/matters');
    fireEvent.click(await screen.findByRole('button', { name: /Import cases/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Import cases' });
    const input = dialog.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(['number,title,client\n'], 'cases.csv', { type: 'text/csv' })] } });
    expect(await within(dialog).findByText('A matter with this number already exists.')).toBeTruthy();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Open 1 matter' }));
    expect(await within(dialog).findByText(/1 matter opened:/)).toBeTruthy();
    expect(calls.filter((c) => c.path.startsWith('/api/legal/matters-import')).map((c) => c.path))
      .toEqual(['/api/legal/matters-import?preview=true', '/api/legal/matters-import?preview=false']);
  });

  it('sends supported files and zips, skips other files and system files', () => {
    const f = (name: string, path = '') => Object.assign(new File(['x'], name), path ? { webkitRelativePath: path } : {});
    const { send, skipped } = sortFiles([f('a.pdf'), f('b.docx'), f('bundle.zip'), f('tool.exe'), f('.DS_Store'), f('x.txt', 'case/__MACOSX/x.txt')]);
    expect(send.map((x) => x.name)).toEqual(['a.pdf', 'b.docx', 'bundle.zip']);
    expect(skipped.map((x) => x.name)).toEqual(['tool.exe']);
  });

  it('the chat\'s case status card shows the stage, updates and tasks, and refuses a card that breaks the contract', async () => {
    const { ComponentView } = await import('../counsel/cards');
    const props = { summary: 'M-150 is at Pleadings', number: 'M-150', title: 'Supply dispute', track_label: 'Disputes',
      stages: ['Intake & conflicts', 'Pleadings', 'Hearing / trial'], stage_index: 1, days_in_stage: 6, open_tasks: 1, overdue: 0, as_of: '2026-11-07T10:00Z',
      can_post: true, path: '/m/M-150/overview',
      updates: [{ kind: 'call', kind_label: 'Call', occurred_on: '2026-11-04', author: 'Alice Moreau', text: 'Client accepts the new date.' }],
      tasks: [{ title: 'Serve the claim', kind: 'filing', due_on: '2026-11-20', overdue: false }] };
    const id = 'cmp_' + 'a'.repeat(32);
    const comp = parseComponent({ id, type: 'case_status', props });
    expect(comp.type).toBe('case_status');
    const go: string[] = [];
    render(<ComponentView comp={comp} ctx={{ go: (p: string) => go.push(p) } as never} />);
    const card = screen.getByRole('region', { name: 'Where M-150 stands' });
    expect(within(card).getByText('Client accepts the new date.')).toBeTruthy();
    expect(within(card).getByText('Serve the claim')).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: /Post an update/ }));
    expect(go).toEqual(['/m/M-150/overview']);
    expect(parseComponent({ id, type: 'case_status', props: { ...props, stage_index: 3 } }).type).toBe('inert');
    expect(parseComponent({ id, type: 'case_status', props: { ...props, path: 'https://example.com' } }).type).toBe('inert');
  });
});
