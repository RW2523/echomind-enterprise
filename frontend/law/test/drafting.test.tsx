import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { ArtifactView, DraftContent, FirmTemplate } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.1 — firm templates and drafting without invention (D-045). */
const M = '/api/legal/matters/id-M-101';
const SHA = 'c'.repeat(64);
const TEMPLATE = (over: Partial<FirmTemplate['versions'][number]> = {}): FirmTemplate => ({
  id: 't1', name: 'Engagement letter', kind: 'engagement_letter', preset: 'engagement_letter', created_by: 'standard preset', created_at: '2026-10-10T08:00:00.000Z',
  versions: [{ id: 'tv1', version: 1, status: 'draft', body: '# Engagement letter\n\nDear {{client_contact}},', content_sha256: SHA, styles: null,
    fields: [{ key: 'client_contact', label: 'Client contact', kind: 'party', source: 'documents', question: 'Who is the contact?' }],
    created_by: 'standard preset', created_at: '2026-10-10T08:00:00.000Z', approved_by: null, approved_at: null, ...over }],
});
const SRC = { document_id: 'd1', document_title: 'Engagement request', version_id: 'v1', version_number: 1, version_status: 'final',
  quote: 'The client contact is Petra Voss', verification: 'verbatim' as const, char_start: 10, char_end: 42 };
const CONTENT: DraftContent = {
  title: 'Engagement letter — Harrowgate', matter: { number: 'M-101', title: 'MSA renewal', client: 'Alpha Holdings' }, gaps: 1,
  template: { id: 't1', name: 'Engagement letter', kind: 'engagement_letter', version: 1, version_id: 'tv1', content_sha256: SHA },
  summary: { fields: 3, sourced: 1, needs_input: 1, written: 1, unsupported: 0 },
  fields: [
    { key: 'client_contact', label: 'Client contact', kind: 'party', source: 'documents', value: 'Petra Voss', status: 'sourced', sources: [SRC] },
    { key: 'fee', label: 'Fee', kind: 'money', source: 'documents', value: null, status: 'needs_input', sources: [],
      suggestion: { value: 'USD 25,000', reason: 'the value is not in the quoted words' } },
    { key: 'matter_number', label: 'Matter number', kind: 'text', source: 'matter', value: 'M-101', status: 'matter', sources: [] },
  ],
  blocks: [
    { type: 'heading', level: 1, origin: 'template', text: 'Engagement letter' },
    { type: 'paragraph', origin: 'template', text: 'Dear Petra Voss,' },
    { type: 'paragraph', origin: 'written', text: 'The firm will review the redline.', instruction: 'Describe the scope', sources: [SRC], unsupported: [] },
    { type: 'paragraph', origin: 'template', text: 'Our fee for matter M-101 is [[NEEDS INPUT: Fee]].' },
  ],
};
const draftView = (over: Partial<DraftContent> = {}, status: 'draft' | 'approved' = 'draft', can = { approve: true, file: true, author: false }): ArtifactView<DraftContent> => ({
  artifact: { id: 'a1', version: 1, status, content: { ...CONTENT, ...over }, content_sha256: SHA, inputs: {}, created_by: 'usr_alice', created_by_name: 'alice',
    created_at: '2026-10-10T09:00:00.000Z', origin: 'model', document_id: null, kind: 'legal_draft', title: CONTENT.title,
    decisions: status === 'approved' ? [{ decision: 'approved', content_sha256: SHA, by: 'dave', at: '2026-10-10T10:00:00.000Z' }] : [] },
  versions: [{ id: 'a1', version: 1, status, created_by_name: 'alice', created_at: '2026-10-10T09:00:00.000Z', origin: 'model' }], can,
});

function world(roles: string[], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(roles)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'lead' })] }],
    ...extra,
  });
}
const AXE = { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } };

describe('Templates and drafting (S3.1)', () => {
  afterEach(() => cleanup());

  it('a managing partner approves a template version after confirming its wording, bound to its hash', async () => {
    const calls = world(['managing_partner'], {
      'GET /api/legal/templates': () => [200, { templates: [TEMPLATE()], can: { manage: true, approve: true } }],
      'POST /api/legal/templates/t1/versions/tv1/approve': () => [200, { template: TEMPLATE({ status: 'approved' }) }],
    });
    const { container } = renderLaw('/templates');
    const card = await screen.findByRole('region', { name: 'Engagement letter' });
    expect(within(card).getByText('No approved version')).toBeTruthy();
    fireEvent.click(within(card).getByRole('button', { name: 'Approve version 1' }));
    const dialog = await screen.findByRole('dialog', { name: 'Approve version 1 of Engagement letter?' });
    expect(dialog.textContent).toContain(SHA);
    const go = within(dialog).getByRole('button', { name: 'Approve version 1' }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);
    fireEvent.click(within(dialog).getByLabelText('I have reviewed this wording and its fields'));
    fireEvent.click(go);
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/approve'))?.body).toEqual({ content_sha256: SHA }));
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
  });

  it('a new template matches its fields to the wording before it is saved', async () => {
    const calls = world(['attorney'], {
      'GET /api/legal/templates': () => [200, { templates: [], can: { manage: true, approve: false } }],
      'POST /api/legal/templates': () => [200, { template: TEMPLATE() }],
    });
    renderLaw('/templates');
    fireEvent.click(await screen.findByRole('button', { name: 'New template' }));
    const dialog = await screen.findByRole('dialog', { name: 'New template' });
    fireEvent.change(within(dialog).getByLabelText('Name'), { target: { value: 'Short memo' } });
    fireEvent.change(within(dialog).getByLabelText('Wording'), { target: { value: '# Memo\n\nTo {{to}} about {{subject}}.' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Match the fields to the wording' }));
    expect((within(dialog).getByLabelText('Key of field 1') as HTMLInputElement).value).toBe('to');
    expect((within(dialog).getByLabelText('Key of field 2') as HTMLInputElement).value).toBe('subject');
    fireEvent.change(within(dialog).getByLabelText('Source of to'), { target: { value: 'person' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save as a draft version' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST')?.body).toMatchObject({
      name: 'Short memo', kind: 'custom', body: '# Memo\n\nTo {{to}} about {{subject}}.',
      fields: [{ key: 'to', source: 'person' }, { key: 'subject', source: 'documents' }] }));
  });

  it('a new draft is made only from an approved template, from the chosen documents', async () => {
    const approved = { ...TEMPLATE({ status: 'approved' }) };
    const calls = world(['attorney'], {
      'GET /api/legal/templates': () => [200, { templates: [approved, { ...TEMPLATE(), id: 't2', name: 'Unapproved' }], can: { manage: false, approve: false } }],
      [`GET ${M}/drafts`]: () => [200, { drafts: [] }],
      [`GET ${M}/documents`]: () => [200, { documents: [{ id: 'd1', title: 'Engagement request', status: 'final' }] }],
      [`POST ${M}/drafts`]: () => [200, { draft: { id: 'a1' } }],
      [`GET ${M}/artifacts/a1`]: () => [200, draftView()],
    });
    renderLaw('/m/M-101/drafting');
    fireEvent.click(await screen.findByRole('button', { name: 'New draft' }));
    const dialog = await screen.findByRole('dialog', { name: 'New draft' });
    await within(dialog).findByRole('option', { name: 'Engagement letter (Engagement letter, version 1)' });
    const select = within(dialog).getByLabelText('Template') as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual(['Choose an approved template', 'Engagement letter (Engagement letter, version 1)']);
    fireEvent.change(select, { target: { value: 't1' } });
    expect((within(dialog).getByLabelText('Title') as HTMLInputElement).value).toBe('Engagement letter — MSA renewal');
    fireEvent.click(within(dialog).getByLabelText('Engagement request'));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Draft' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ template_id: 't1', title: 'Engagement letter — MSA renewal', documents: ['d1'], note: null }));
    expect(await screen.findByRole('article', { name: 'Draft document' })).toBeTruthy();
  });

  it('the draft shows each gap, where every field came from, and no approval until the gaps are filled', async () => {
    const calls = world(['attorney'], {
      [`GET ${M}/artifacts/a1`]: () => [200, draftView()],
      [`POST ${M}/drafts/a1/edit`]: () => [200, { draft: { id: 'a2' } }],
    });
    const { container } = renderLaw('/m/M-101/drafting/a1');
    const doc = await screen.findByRole('article', { name: 'Draft document' });
    expect(within(doc).getByText('Fee').tagName).toBe('MARK');
    expect(doc.textContent).toContain('1 gap to fill');
    const fields = screen.getByRole('list', { name: 'Fields' });
    const contact = within(fields).getByRole('listitem', { name: 'Client contact: Petra Voss' });
    expect(contact.textContent).toContain('From the documents');
    expect(within(contact).getByRole('link', { name: 'Open at the quote' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=10-42');
    expect(within(fields).getByRole('listitem', { name: 'Fee: needs input' }).textContent).toContain('The model suggested “USD 25,000”, not used: the value is not in the quoted words.');
    expect(screen.queryByRole('group', { name: 'Review this draft' })).toBeNull();
    expect(screen.getByText('Approval opens once the 1 gap is filled.')).toBeTruthy();
    expect((await axe.run(container, AXE)).violations.map((v) => v.id)).toEqual([]);
    fireEvent.click(screen.getByRole('button', { name: 'Fill in or correct' }));
    const form = screen.getByRole('form', { name: 'Fill in or correct the draft' });
    fireEvent.change(within(form).getByLabelText(/^Fee/), { target: { value: 'USD 18,000' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Save as version 2' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/edit'))?.body).toEqual({ content_sha256: SHA, fields: [{ key: 'fee', value: 'USD 18,000' }], blocks: [], reason: null }));
    await waitFor(() => expect(calls.some((c) => c.method === 'GET' && c.path === `${M}/artifacts/a2`)).toBe(true));   // opens the new version
  });

  it('an approved draft is filed as a Word document', async () => {
    const calls = world(['attorney'], {
      [`GET ${M}/artifacts/a1`]: () => [200, draftView({ gaps: 0 }, 'approved')],
      [`POST ${M}/artifacts/a1/file`]: () => [200, { document_id: 'd9', changed: true }],
    });
    renderLaw('/m/M-101/drafting/a1');
    expect((await screen.findByRole('article', { name: 'Draft document' })).textContent).toMatch(/Approved by dave/);
    fireEvent.click(screen.getByRole('button', { name: 'File as a Word document' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${M}/artifacts/a1/file`)).toBe(true));
  });
});
