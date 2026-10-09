import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { LawExport } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

const BASE = '/api/legal/matters/id-M-101/exports';
const exp = (over: Partial<LawExport> = {}): LawExport => ({
  id: 'x1', object_type: 'document', object_id: 'd1', title: 'Settlement terms', format: 'pdf', purpose: 'Disclosure to opposing counsel',
  status: 'draft', requested_by: 'alice', requested_at: '2026-10-09T09:00:00.000Z', downloads: 0, label: null,
  plan: { source: 'pdf', warnings: { image_pages: [] }, items: [
    { id: 'i1', kind: 'email', text: 'legal@harrowgate.example', count: 2, redact: true, reason: 'Email address', source: 'detector' },
    { id: 'i2', kind: 'matter_number', text: 'M-101', count: 1, redact: false, reason: 'This matter’s number (it is in the watermark)', source: 'detector' },
  ] }, ...over,
});

function world(username: string, role: string, e: LawExport, extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, { ...me(['attorney']), user: { ...me().user, username } }],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role as any })] }],
    [`GET ${BASE}`]: () => [200, { exports: [e] }],
    [`GET ${BASE}/x1`]: () => [200, { export: e }],
    [`GET ${BASE}/x1/preview`]: () => [200, { pages: 2 }],
    ...extra,
  });
}

describe('Exports', () => {
  afterEach(() => cleanup());

  it('lets the requester adjust redactions, then submit only once saved', async () => {
    const calls = world('alice', 'lead', exp(), {
      [`PUT ${BASE}/x1/plan`]: () => [200, { export: exp() }],
      [`POST ${BASE}/x1/submit`]: () => [200, { export: exp({ status: 'pending_approval' }) }],
    });
    const { container } = renderLaw('/m/M-101/exports/x1');
    const plan = await screen.findByRole('region', { name: 'Redactions' });
    expect(within(plan).getByText('Email addresses')).toBeTruthy();
    expect(await screen.findAllByRole('img', { name: /as it will be released/ })).toHaveLength(2);
    fireEvent.click(within(plan).getByRole('checkbox', { name: /M-101/ }));
    expect((screen.getByRole('button', { name: 'Submit for approval' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(within(plan).getByLabelText('Also remove'), { target: { value: 'USD 250,000' } });
    fireEvent.click(within(plan).getByRole('button', { name: /Save redactions/ }));
    await waitFor(() => expect(calls.find((c) => c.method === 'PUT')?.body).toEqual({ items: [{ id: 'i2', redact: true }], add: ['USD 250,000'] }));
    await waitFor(() => expect((screen.getByRole('button', { name: 'Submit for approval' }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${BASE}/x1/submit`)).toBe(true));
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('asks a second lawyer to confirm before releasing a privileged document', async () => {
    const calls = world('dave', 'attorney', exp({ status: 'pending_approval', label: 'privileged' }), {
      [`POST ${BASE}/x1/approve`]: () => [200, { export: exp({ status: 'ready' }) }],
    });
    renderLaw('/m/M-101/exports/x1');
    const approve = await screen.findByRole('button', { name: 'Approve and release' }) as HTMLButtonElement;
    expect(approve.disabled).toBe(true);
    fireEvent.click(screen.getByRole('checkbox', { name: /meant to leave the firm/ }));
    fireEvent.click(approve);
    await waitFor(() => expect(calls.find((c) => c.path === `${BASE}/x1/approve`)?.body).toEqual({ acknowledge_privilege: true }));
  });

  it('never offers the requester their own approval', async () => {
    world('alice', 'lead', exp({ status: 'pending_approval' }));
    renderLaw('/m/M-101/exports/x1');
    expect(await screen.findByText(/Waiting for another lead or attorney/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Approve and release' })).toBeNull();
    expect((screen.getByRole('checkbox', { name: /legal@harrowgate/ }) as HTMLInputElement).disabled).toBe(true);
  });

  it('shows the checks, the watermark and the download once released', async () => {
    world('alice', 'lead', exp({ status: 'ready', decided_by: 'dave', watermark: 'CONFIDENTIAL — M-101 — alice — 2026-10-09T09:30Z',
      expires_at: '2026-10-10T09:30:00.000Z', verification: { ok: true, checks: { redactions_absent: true, metadata_clean: true, structure_clean: true, watermark: true } } }));
    renderLaw('/m/M-101/exports/x1');
    expect((await screen.findByRole('link', { name: /Download/ })).getAttribute('href')).toBe(`${BASE}/x1/file`);
    expect(screen.getByText(/Redacted text is gone from text, metadata and every part of the file/)).toBeTruthy();
    expect(screen.getByText('CONFIDENTIAL — M-101 — alice — 2026-10-09T09:30Z')).toBeTruthy();
  });

  it('starts from a document and is not offered to read-only members', async () => {
    const calls = mockApi({
      'GET /api/auth/me': () => [200, me(['attorney'])],
      'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
      'GET /api/legal/matters/id-M-101/documents': () => [200, { documents: [{ id: 'd1', title: 'Settlement terms', kind: 'contract', status: 'final', version: 1,
        created_at: '2026-10-09T08:00:00.000Z', findings: {}, current_version: { id: 'v1', number: 1, filename: 't.pdf', mime: 'application/pdf', size_bytes: 9,
        sha256: 'ab'.repeat(32), uploaded_at: '2026-10-09T08:00:00.000Z', index_status: 'indexed', chunk_count: 1 } }] }],
      'GET /api/legal/matters/id-M-101/documents/d1': () => [200, { document: { id: 'd1', title: 'Settlement terms', kind: 'contract', status: 'final', version: 1,
        created_at: '2026-10-09T08:00:00.000Z', findings: {}, current_version: { id: 'v1', number: 1, filename: 't.pdf', mime: 'application/pdf', size_bytes: 9,
        sha256: 'ab'.repeat(32), uploaded_at: '2026-10-09T08:00:00.000Z', index_status: 'indexed', chunk_count: 1 } } }],
      'GET /api/legal/matters/id-M-101/documents/d1/versions': () => [200, { versions: [] }],
      'GET /api/legal/matters/id-M-101/labels/document/d1': () => [200, { current: null, suggestions: [], history: [] }],
      'GET /api/legal/matters/id-M-101/deletions': () => [200, { deletions: [] }],
      [`POST ${BASE}`]: () => [200, { export: exp() }],
      [`GET ${BASE}/x1`]: () => [200, { export: exp() }],
      [`GET ${BASE}/x1/preview`]: () => [200, { pages: 1 }],
    });
    renderLaw('/m/M-101/documents');
    fireEvent.click(await screen.findByRole('button', { name: 'Settlement terms' }));
    fireEvent.click(await screen.findByRole('button', { name: /Export…/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Export “Settlement terms”' });
    fireEvent.change(within(dialog).getByLabelText('Purpose'), { target: { value: 'Disclosure' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /Prepare export/ }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === BASE)?.body).toEqual({ object_type: 'document', object_id: 'd1', format: 'pdf', purpose: 'Disclosure' }));
    expect(await screen.findByRole('region', { name: 'Redactions' })).toBeTruthy();
    cleanup();
    world('carol', 'read_only', exp());
    renderLaw('/m/M-101/exports');
    expect(await screen.findByText("This item isn't available.")).toBeTruthy();
  });
});
