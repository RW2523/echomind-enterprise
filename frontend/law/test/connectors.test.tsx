import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Connector, LawDocument } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const conn = (over: Partial<Connector> = {}): Connector => ({
  id: 'k1', kind: 'local_folder', name: 'Firm DMS', status: 'active', sync_minutes: 15, stale_after_minutes: 60,
  last_sync_at: '2026-10-09T09:00:00.000Z', last_ok_at: '2026-10-09T09:00:00.000Z', last_status: 'partial', last_error: null,
  fresh: true, links: 2, documents: 7, runs: [{ started_at: '2026-10-09T09:00:00.000Z', status: 'partial', report: { imported: 5, acl_missing: 2, held: 0 } }], ...over,
});
const synced = (over: Partial<LawDocument> = {}): LawDocument => ({
  id: 'd1', title: 'msa', kind: 'other', status: 'received', version: 1, created_at: '2026-10-09T09:00:00.000Z', findings: {},
  source: { kind: 'connector', name: 'Firm DMS', state: 'current', path: 'msa.txt' },
  current_version: { id: 'v1', number: 1, filename: 'msa.txt', mime: 'text/plain', size_bytes: 10, sha256: 'ab'.repeat(32),
    uploaded_by: 'alice', uploaded_at: '2026-10-09T09:00:00.000Z', index_status: 'indexed', chunk_count: 1 }, ...over,
});

describe('Connectors', () => {
  afterEach(() => cleanup());

  it('shows what is available, what is out of date and what the last sync did', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['firm_admin'])],
      'GET /api/legal/matters': () => [200, { matters: [] }],
      'GET /api/legal/connectors': () => [200, { connectors: [conn(), conn({ id: 'k2', name: 'Archive', fresh: false, runs: [] })], providers: [
        { kind: 'local_folder', name: 'Folder on this server', available: true },
        { kind: 'dms', name: 'Firm document management system', available: false, note: 'Needs the firm’s choice of provider, credentials and network approval (P-010).' }] }],
      'POST /api/legal/connectors/k1/sync': () => [200, { run: { status: 'ok' } }],
    });
    const { container } = renderLaw('/admin/connectors');
    expect(await screen.findByText(/Needs the firm’s choice of provider/)).toBeTruthy();
    const dms = screen.getByRole('region', { name: 'Firm DMS' });
    expect(within(dms).getByText('5 imported')).toBeTruthy();
    expect(within(dms).getByText('2 skipped: no permissions file')).toBeTruthy();
    expect(within(screen.getByRole('region', { name: 'Archive' })).getByText(/Out of date — its documents are hidden/)).toBeTruthy();
    fireEvent.click(within(dms).getByRole('button', { name: /Sync now/ }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/connectors/k1/sync')).toBe(true));
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('is for firm administrators only', async () => {
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/admin/connectors');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
  });

  it('lets a matter’s lead link a folder and shows where documents came from', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/matters/id-M-101/documents': () => [200, { documents: [synced()] }],
      'GET /api/legal/matters/id-M-101/deletions': () => [200, { deletions: [] }],
      'GET /api/legal/matters/id-M-101/connector-links': () => [200, { links: [], connectors: [{ id: 'k1', name: 'Firm DMS', kind: 'local_folder' }] }],
      'POST /api/legal/matters/id-M-101/connector-links': () => [200, { link: { id: 'l1' } }],
    });
    renderLaw('/m/M-101/documents');
    expect(await screen.findByText('Synced')).toBeTruthy();
    const panel = await screen.findByRole('region', { name: 'Linked folders' });
    fireEvent.click(within(panel).getByRole('button', { name: /Link a folder/ }));
    fireEvent.change(within(panel).getByLabelText('Folder'), { target: { value: 'Harrowgate/MSA' } });
    fireEvent.click(within(panel).getByRole('button', { name: 'Link' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/matters/id-M-101/connector-links')?.body)
      .toEqual({ connector_id: 'k1', remote_path: 'Harrowgate/MSA' }));
    fireEvent.click(screen.getByRole('button', { name: /Import/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a document' });
    expect(within(dialog).queryByText(/A new version of “msa”/)).toBeNull();      // kept in the firm's system
  });
});
