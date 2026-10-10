import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Deletion, Hold, LawDocument } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const DOCS = '/api/legal/matters/id-M-101/documents';
const LABELS = '/api/legal/matters/id-M-101/labels/document/d1';
const doc = (over: Partial<LawDocument> = {}): LawDocument => ({
  id: 'd1', title: 'Settlement memo', kind: 'memo', status: 'draft', version: 2, created_at: '2026-10-08T10:00:00.000Z', findings: {},
  label: null, label_suggested: true,
  current_version: { id: 'v1', number: 1, filename: 'settlement-memo.txt', mime: 'text/plain', size_bytes: 120, sha256: 'ab'.repeat(32),
    uploaded_by: 'alice', uploaded_at: '2026-10-08T10:00:00.000Z', index_status: 'indexed', chunk_count: 2 }, ...over,
});
const REPORT = { object: 'document' as const, versions: 1, passages: 2, findings: 0, labels: 1, files: 1, file_bytes: 120, index_vectors: 2,
  research_citations: 1, comparisons: 2, kept: ['The audit trail (identifiers only, never content)', 'This deletion record, without the document’s title'] };
const deletion = (over: Partial<Deletion> = {}): Deletion => ({
  id: 'del1', object_type: 'document', object_id: 'd1', object_label: 'Settlement memo', reason: 'Duplicate', origin: 'person', status: 'requested',
  requested_by: 'alice', requested_at: '2026-10-08T11:00:00.000Z', report: REPORT, verification: null,
  matter: { id: 'id-M-101', number: 'M-101', title: 'MSA renewal' }, ...over,
});

function documents(role: string, extra = {}, m = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role as any, ...m })] }],
    [`GET ${DOCS}`]: () => [200, { documents: [doc()] }],
    [`GET ${DOCS}/d1`]: () => [200, { document: doc() }],
    [`GET ${DOCS}/d1/versions`]: () => [200, { versions: [doc().current_version] }],
    [`GET ${LABELS}`]: () => [200, { current: null, history: [],
      suggestions: [{ label: 'privileged', kind: 'suggested', source: 'detector', reason_code: 'privileged_confidential', by: null, at: '2026-10-08T10:01:00.000Z' }] }],
    'GET /api/legal/matters/id-M-101/deletions': () => [200, { deletions: [] }],
    ...extra,
  });
}

async function openDetails() {
  fireEvent.click(await screen.findByRole('button', { name: 'Settlement memo' }));
  return screen.findByRole('complementary', { name: 'Document details' });
}

describe('Privilege labels', () => {
  afterEach(() => cleanup());

  it('shows the detector’s suggestion and lets a lawyer accept it', async () => {
    const calls = documents('attorney', { [`POST ${LABELS}`]: () => [200, { current: { label: 'privileged', kind: 'set', source: 'person', by: 'alice', at: '2026-10-08T12:00:00.000Z' }, suggestions: [], history: [] }] });
    renderLaw('/m/M-101/documents');
    const panel = await openDetails();
    expect(await within(panel).findByText(/the text contains “privileged and confidential”/)).toBeTruthy();
    fireEvent.click(within(panel).getByRole('button', { name: 'Accept' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === LABELS)?.body).toEqual({ label: 'privileged' }));
    await waitFor(() => expect(panel.querySelector('.lw-pill-privileged')?.textContent).toBe('Privileged'));
  });

  it('lets a paralegal suggest only, and a read-only member neither', async () => {
    documents('paralegal');
    renderLaw('/m/M-101/documents');
    let panel = await openDetails();
    expect(await within(panel).findByLabelText('Suggest label')).toBeTruthy();
    expect(within(panel).queryByRole('button', { name: 'Accept' })).toBeNull();
    cleanup();
    documents('read_only');
    renderLaw('/m/M-101/documents');
    panel = await openDetails();
    await within(panel).findByText(/Labels advise/);
    expect(within(panel).queryByRole('combobox')).toBeNull();
  });
});

describe('Deleting a document', () => {
  afterEach(() => cleanup());

  it('previews what goes and what stays, needs a reason, and can be restored', async () => {
    let pending: Deletion[] = [];
    const calls = documents('lead', {
      [`GET ${DOCS}/d1/deletion-preview`]: () => [200, { report: REPORT }],
      [`POST ${DOCS}/d1/delete`]: () => { pending = [deletion()]; return [200, { deletion: deletion() }]; },
      'GET /api/legal/matters/id-M-101/deletions': () => [200, { deletions: pending }],
      'POST /api/legal/deletions/del1/restore': () => { pending = []; return [200, { deletion: deletion({ status: 'restored' }) }]; },
    });
    const { container } = renderLaw('/m/M-101/documents');
    const panel = await openDetails();
    fireEvent.click(await within(panel).findByRole('button', { name: /Delete…/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Delete “Settlement memo”' });
    expect(await within(dialog).findByText('1 research answer quoting it (the quotes are removed)')).toBeTruthy();
    expect(within(dialog).getByText('2 saved comparisons quoting it (removed whole)')).toBeTruthy();
    expect(within(dialog).getByText('The audit trail (identifiers only, never content)')).toBeTruthy();
    const submit = within(dialog).getByRole('button', { name: /Request deletion/ }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
    fireEvent.change(within(dialog).getByLabelText('Reason'), { target: { value: 'Duplicate' } });
    fireEvent.click(submit);
    await waitFor(() => expect(calls.find((c) => c.path === `${DOCS}/d1/delete`)?.body).toEqual({ reason: 'Duplicate' }));
    const waiting = await screen.findByRole('region', { name: 'Waiting for deletion approval' });
    expect(within(waiting).getByText(/Requested by alice/)).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
    fireEvent.click(within(waiting).getByRole('button', { name: /Restore/ }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/deletions/del1/restore')).toBe(true));
  });

  it('is not offered under a legal hold, and says why', async () => {
    documents('lead', {}, { on_hold: true });
    renderLaw('/m/M-101/documents');
    const panel = await openDetails();
    expect(await within(panel).findByText(/Nothing in this matter can be deleted while the hold is in force/)).toBeTruthy();
    expect(within(panel).queryByRole('button', { name: /Delete…/ })).toBeNull();
  });
});

describe('Records', () => {
  afterEach(() => cleanup());

  function records(username = 'rhea', roles = ['records_admin'], extra = {}) {
    return mockApi({
      'GET /api/auth/me': () => [200, { ...me(roles), user: { ...me().user, username, roles: roles as any } }],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/deletions': () => [200, { deletions: [deletion(), deletion({ id: 'del0', object_label: null, status: 'purged', decided_by: 'rex',
        verification: { ok: true, checks: { database: true, index: true, files: true, research_quotes: true }, verified_at: '2026-10-08T12:00:00.000Z' } })] }],
      'POST /api/legal/deletions/del1/approve': () => [200, { deletion: deletion({ status: 'purged' }) }],
      ...extra,
    });
  }

  it('is absent without records authority', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/admin/records');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
  });

  it('shows a request with its report, approves it, and proves earlier purges', async () => {
    const calls = records();
    renderLaw('/admin/records');
    const card = await screen.findByRole('region', { name: 'Settlement memo' });
    expect(within(card).getByText('2 searchable passages')).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Approve deletion' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/deletions/del1/approve')).toBe(true));
    expect(screen.getByText('Nothing left')).toBeTruthy();
    expect(screen.getByText(/Document \(title removed with it\)/)).toBeTruthy();
  });

  it('never offers someone a decision on their own request', async () => {
    records('alice', ['managing_partner']);
    renderLaw('/admin/records');
    const card = await screen.findByRole('region', { name: 'Settlement memo' });
    expect(within(card).queryByRole('button', { name: 'Approve deletion' })).toBeNull();
    expect(within(card).getByText(/someone else must decide/)).toBeTruthy();
  });

  it('puts holds and their approval in different hands', async () => {
    const hold: Hold = { id: 'h1', name: 'Beta litigation hold', reason: 'Anticipated claim', status: 'requested', requested_by: 'rhea',
      requested_at: '2026-10-08T10:00:00.000Z', matters: [{ id: 'm', number: 'M-215', title: 'Lease dispute', client: 'Beta' }] };
    records('rhea', ['records_admin'], { 'GET /api/legal/holds': () => [200, { holds: [hold], can_request: true }] });
    renderLaw('/admin/records');
    fireEvent.click(await screen.findByRole('tab', { name: 'Legal holds' }));
    const card = await screen.findByRole('region', { name: 'Beta litigation hold' });
    expect(within(card).queryByRole('button', { name: 'Approve hold' })).toBeNull();
    expect(screen.getByRole('form', { name: 'Request a legal hold' })).toBeTruthy();
  });

  it('keeps applying retention switched off until it is enabled', async () => {
    records('rhea', ['records_admin'], { 'GET /api/legal/retention': () => [200, { policies: [], runs: [], apply_enabled: false }] });
    renderLaw('/admin/records');
    fireEvent.click(await screen.findByRole('tab', { name: 'Retention' }));
    expect((await screen.findByRole('button', { name: 'Apply' }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/never deletes anything by itself/)).toBeTruthy();
  });
});
