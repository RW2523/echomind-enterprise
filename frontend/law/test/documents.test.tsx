import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { LawDocument } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const BASE = '/api/legal/matters/id-M-101/documents';
const doc = (over: Partial<LawDocument> = {}, status: 'queued' | 'indexed' | 'failed' = 'indexed', error: string | null = null): LawDocument => ({
  id: 'd1', title: 'Harrowgate MSA', kind: 'contract', status: 'final', version: 2, created_by: 'alice',
  created_at: '2026-10-08T10:00:00.000Z', updated_at: '2026-10-08T11:00:00.000Z', findings: { email: 1, phone: 1 },
  current_version: { id: 'v2', number: 2, filename: 'harrowgate-msa-v2.pdf', mime: 'application/pdf', size_bytes: 48213,
    sha256: 'ab'.repeat(32), pages: 14, uploaded_by: 'alice', uploaded_at: '2026-10-08T11:00:00.000Z',
    index_status: status, index_error: error, chunk_count: 31, indexed_at: null },
  ...over,
});

function world(role = 'lead', docs = [doc()], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role as any })] }],
    [`GET ${BASE}`]: () => [200, { documents: docs }],
    [`GET ${BASE}/d1`]: () => [200, { document: docs[0] }],
    [`GET ${BASE}/d1/versions`]: () => [200, { versions: [docs[0].current_version, { ...docs[0].current_version, id: 'v1', number: 1 }] }],
    ...extra,
  });
}

async function axeViolations(container: HTMLElement) {
  const r = await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } });
  return r.violations.map((v) => `${v.id}: ${v.nodes.length}`);
}

describe('Matter documents', () => {
  afterEach(() => cleanup());

  it('lists the library with plain search states, and read-only members cannot import', async () => {
    world('read_only', [doc(), doc({ id: 'd2', title: 'Side letter' }, 'queued')]);
    renderLaw('/m/M-101/documents');
    expect(await screen.findByRole('button', { name: 'Harrowgate MSA' })).toBeTruthy();
    expect(screen.getByText('Searchable')).toBeTruthy();
    expect(screen.getByText('Waiting')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Import/ })).toBeNull();
    expect(screen.getByRole('link', { name: 'Documents' })).toBeTruthy();
  });

  it('imports a file as multipart with its details', async () => {
    const calls = world('lead', [], { [`POST ${BASE}/import`]: () => [202, { document: doc({}, 'queued'), same_content_as: null }] });
    renderLaw('/m/M-101/documents');
    fireEvent.click(await screen.findByRole('button', { name: /Import a document/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a document' });
    const file = new File(['MASTER SERVICES AGREEMENT'], 'harrowgate-msa.txt', { type: 'text/plain' });
    fireEvent.change(within(dialog).getByLabelText('File'), { target: { files: [file] } });
    expect((within(dialog).getByLabelText('Title') as HTMLInputElement).value).toBe('harrowgate-msa');
    fireEvent.change(within(dialog).getByLabelText('Kind'), { target: { value: 'correspondence' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /^Import$/ }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/import`)).toBe(true));
    const sent = calls.find((c) => c.path === `${BASE}/import`)!.body;
    expect(sent).toEqual({ file: 'harrowgate-msa.txt', title: 'harrowgate-msa', kind: 'correspondence', status: 'draft' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('searches this matter only and says so plainly when nothing matches', async () => {
    let abstain = false;
    world('lead', [doc()], {
      [`POST ${BASE}/search`]: () => [200, abstain ? { passages: [], abstained: true } : { abstained: false, passages: [{
        chunk_id: 'c1', text: '## 9. Liability The aggregate liability of the Supplier shall not exceed two times the annual fees.', score: 4.2, score_kind: 'ce_logit',
        citation: { matter_id: 'id-M-101', document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2', version_number: 2, page_start: 9, page_end: 9 } }] }],
    });
    renderLaw('/m/M-101/documents');
    const box = await screen.findByLabelText('Search this matter’s documents');
    fireEvent.change(box, { target: { value: 'liability cap' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(await screen.findByText('1 passage from this matter')).toBeTruthy();
    expect(screen.getAllByText(/^liability$/i)[0].tagName).toBe('MARK');
    expect(screen.queryByText(/##/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Harrowgate MSA · v2 · p. 9' })).toBeTruthy();
    abstain = true;
    fireEvent.change(box, { target: { value: 'board resolution' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(await screen.findByText(/Nothing in this matter’s documents closely matches “board resolution”/)).toBeTruthy();
  });

  it('opens details as a recorded read, with personal data and versions', async () => {
    const calls = world('lead', [doc()]);
    const { container } = renderLaw('/m/M-101/documents');
    fireEvent.click(await screen.findByRole('button', { name: 'Harrowgate MSA' }));
    const panel = await screen.findByRole('complementary', { name: 'Document details' });
    expect(await within(panel).findByText(/1 email address · 1 phone number/)).toBeTruthy();
    expect(within(panel).getByRole('link', { name: 'Download version 1' }).getAttribute('href')).toBe(`${BASE}/d1/versions/v1/file`);
    expect(calls.some((c) => c.method === 'GET' && c.path === `${BASE}/d1`)).toBe(true);
    expect(await axeViolations(container)).toEqual([]);
  });

  it('a new version is imported with its own status (S2.2)', async () => {
    const calls = world('lead', [doc()], { [`POST ${BASE}/import`]: () => [202, { document: doc({}, 'queued'), same_content_as: null }] });
    renderLaw('/m/M-101/documents');
    fireEvent.click(await screen.findByRole('button', { name: /Import a document|Import/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Import a document' });
    fireEvent.change(within(dialog).getByLabelText('File'), { target: { files: [new File(['SIGNED'], 'msa-signed.pdf', { type: 'application/pdf' })] } });
    fireEvent.change(within(dialog).getByLabelText('This is'), { target: { value: 'd1' } });
    expect(within(dialog).queryByLabelText('Title')).toBeNull();
    fireEvent.change(within(dialog).getByLabelText(/^Status/), { target: { value: 'executed' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /^Import$/ }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/import`)).toBe(true));
    const sent = calls.find((c) => c.path === `${BASE}/import`)!.body;
    expect(sent.version_of).toBe('d1');
    expect(sent.status).toBe('executed');
  });

  it('explains a failed read and lets a contributor try again', async () => {
    const calls = world('attorney', [doc({}, 'failed', 'no_text')], { [`POST ${BASE}/d1/reindex`]: () => [202, { document: doc({}, 'queued') }] });
    renderLaw('/m/M-101/documents');
    fireEvent.click(await screen.findByRole('button', { name: 'Harrowgate MSA' }));
    const panel = await screen.findByRole('complementary', { name: 'Document details' });
    expect(await within(panel).findByText('No readable text, even with text recognition.')).toBeTruthy();   // S2.1: scans are now read
    fireEvent.click(within(panel).getByRole('button', { name: /Try again/ }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/d1/reindex`)).toBe(true));
  });
});
