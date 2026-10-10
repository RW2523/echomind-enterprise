import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { ArtifactView, LawExport, ResearchNoteContent, ReviewQueue } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S2.7 — review and approval of AI drafts: the queue, research notes, and approvals carried into exports (D-043). */

const M = '/api/legal/matters/id-M-101';
const SHA = 'b'.repeat(64);
const NOTE: ResearchNoteContent = {
  question: 'What is the liability cap?', answer: 'Liability is capped at USD 250,000 [1].', status: 'answered', abstention_reason: null,
  versions: 'signed', unsupported: [], conflicts: [], sources: [], asked_by: 'alice', asked_at: '2026-10-10T09:00:00.000Z',
  evidence: [{ n: 1, quote: 'capped at USD 250,000', verification: 'verbatim', document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v1',
    version_number: 1, version_status: 'executed', clause: '12.1', page_start: 3, line: 4, char_start: 120, char_end: 141 }],
};
const view = (over: Partial<ArtifactView<ResearchNoteContent>['artifact']> = {}, can = { approve: true, file: true, author: false }): ArtifactView<ResearchNoteContent> => ({
  artifact: { id: 'a1', version: 1, status: 'draft', content: NOTE, content_sha256: SHA, inputs: {}, created_by: 'usr_alice', created_by_name: 'alice',
    created_at: '2026-10-10T09:01:00.000Z', origin: 'model', document_id: null, decisions: [], kind: 'research_note', title: NOTE.question, ...over },
  versions: [{ id: 'a1', version: 1, status: 'draft', created_by_name: 'alice', created_at: '2026-10-10T09:01:00.000Z', origin: 'model' }], can,
});
const QUEUE: ReviewQueue = {
  count: 2,
  waiting: [
    { kind: 'research_note', id: 'a1', title: 'What is the liability cap?', version: 1, matter: { number: 'M-101', title: 'MSA renewal' }, by: 'alice',
      at: '2026-10-10T09:01:00.000Z', path: '/m/M-101/notes/a1' },
    { kind: 'export', id: 'e1', title: 'Side letter (PDF)', matter: { number: 'M-101', title: 'MSA renewal' }, by: 'alice',
      at: '2026-10-10T09:05:00.000Z', path: '/m/M-101/exports/e1' }],
  mine: [{ kind: 'minutes', id: 'a9', title: 'Board meeting', version: 2, matter: { number: 'M-101', title: 'MSA renewal' }, by: 'dave',
    at: '2026-10-10T08:00:00.000Z', path: '/m/M-101/meetings/mt1?tab=minutes', state: 'changes_requested', reason: 'Add the quorum.', decided_by: 'erin' }],
};

function world(extra = {}, user = 'usr_dave') {
  return mockApi({
    'GET /api/auth/me': () => [200, { ...me(['attorney']), user: { ...me(['attorney']).user, id: user } }],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'attorney' })] }],
    'GET /api/legal/reviews': () => [200, QUEUE],
    ...extra,
  });
}

describe('Reviews (S2.7)', () => {
  afterEach(() => cleanup());

  it('the queue lists what waits for you and what of yours waits, and the navigation counts it', async () => {
    world();
    const { container } = renderLaw('/reviews');
    const waiting = await screen.findByRole('list', { name: 'Waiting for you' });
    const note = within(waiting).getByRole('listitem', { name: 'Research note: What is the liability cap?' });
    expect(within(note).getByRole('link').getAttribute('href')).toBe('/m/M-101/notes/a1');
    expect(note.textContent).toMatch(/M-101 · MSA renewal · From alice/);
    expect(within(waiting).getByRole('link', { name: /Side letter \(PDF\)/ }).getAttribute('href')).toBe('/m/M-101/exports/e1');
    const mine = screen.getByRole('list', { name: 'Yours, with others' });
    expect(mine.textContent).toContain('Sent back by erin: “Add the quorum.”');
    expect(await screen.findByRole('link', { name: 'Reviews, 2 waiting for you' })).toBeTruthy();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a research note shows the answer fixed as given, with evidence opening at the quote', async () => {
    world({ [`GET ${M}/artifacts/a1`]: () => [200, view()] });
    const { container } = renderLaw('/m/M-101/notes/a1');
    const note = await screen.findByRole('article', { name: 'Research note' });
    expect(within(note).getByRole('heading', { name: 'What is the liability cap?' })).toBeTruthy();
    expect(note.textContent).toContain('Draft — not approved');
    expect(note.textContent).toContain('Answered from the documents');
    const ev = within(note).getByRole('region', { name: 'Evidence' });
    expect(ev.textContent).toMatch(/\[1\] “capped at USD 250,000”.*Harrowgate MSA, version 1 \(executed\), clause 12\.1, page 3, line 4 · Found as written/);
    expect(within(ev).getByRole('link', { name: 'Open at the quote' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=120-141');
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a second lawyer approves the exact version after confirming they read it; asking for changes needs a reason', async () => {
    const calls = world({ [`GET ${M}/artifacts/a1`]: () => [200, view()],
      [`POST ${M}/artifacts/a1/review`]: () => [200, { artifact: view().artifact }] });
    renderLaw('/m/M-101/notes/a1');
    const review = await screen.findByRole('group', { name: 'Review this note' });
    const back = within(review).getByRole('button', { name: 'Request changes' }) as HTMLButtonElement;
    expect(back.disabled).toBe(true);
    fireEvent.change(within(review).getByLabelText('What should change?'), { target: { value: 'Cite the side letter too.' } });
    fireEvent.click(back);
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/artifacts/a1/review`)?.body)
      .toEqual({ decision: 'changes_requested', content_sha256: SHA, reason: 'Cite the side letter too.' }));
    fireEvent.click(within(review).getByRole('button', { name: 'Approve this note' }));
    const dialog = await screen.findByRole('dialog', { name: 'Approve version 1?' });
    expect(dialog.textContent).toContain(SHA);
    fireEvent.click(within(dialog).getByLabelText('I have read this version and checked its sources'));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Approve version 1' }));
    await waitFor(() => expect(calls.filter((c) => c.path === `${M}/artifacts/a1/review`).map((c) => c.body.decision)).toEqual(['changes_requested', 'approved']));
  });

  it('the author is told another lawyer approves; an approved note is filed and then opened', async () => {
    world({ [`GET ${M}/artifacts/a1`]: () => [200, view({}, { approve: false, file: true, author: true })] }, 'usr_alice');
    renderLaw('/m/M-101/notes/a1');
    expect(await screen.findByText('You made this version, so another lawyer approves it.')).toBeTruthy();
    cleanup();
    const approved = view({ status: 'approved', decisions: [{ decision: 'approved', content_sha256: SHA, by: 'dave', at: '2026-10-10T10:00:00.000Z' }] });
    let filed: string | null = null;
    const calls = world({ [`GET ${M}/artifacts/a1`]: () => [200, filed ? view({ ...approved.artifact, document_id: filed }) : approved],
      [`POST ${M}/artifacts/a1/file`]: () => { filed = 'd9'; return [200, { document_id: 'd9', changed: true }]; } }, 'usr_alice');
    renderLaw('/m/M-101/notes/a1');
    expect((await screen.findByRole('article', { name: 'Research note' })).textContent).toMatch(/Approved by dave on .* · binds content bbbbbbbbbbbb…/);
    fireEvent.click(screen.getByRole('button', { name: 'File as a matter document' }));
    await waitFor(() => expect(calls.some((c) => c.path === `${M}/artifacts/a1/file`)).toBe(true));
    expect((await screen.findByRole('link', { name: 'Open the filed note' })).getAttribute('href')).toBe('/m/M-101/documents/d9/read');
  });

  it('a research answer is sent for review and opens as a note', async () => {
    const calls = world({
      [`GET ${M}/research`]: () => [200, { turns: [] }],
      [`POST ${M}/research/query`]: () => [200, { __ndjson: [{ type: 'sources', citations: [] },
        { type: 'done', answer: 'Liability is capped.', citations: [], abstained: false, status: 'answered', turn_id: 't1', evidence: [] }] }],
      [`POST ${M}/research/t1/review`]: () => [200, { artifact: { id: 'a1' }, changed: true }],
      [`GET ${M}/artifacts/a1`]: () => [200, view()],
    }, 'usr_alice');
    renderLaw('/m/M-101/research');
    fireEvent.change(await screen.findByLabelText('Ask about this matter'), { target: { value: 'What is the liability cap?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Send for review' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === `${M}/research/t1/review`)).toBe(true));
    expect(await screen.findByRole('article', { name: 'Research note' })).toBeTruthy();
  });

  it('an export of a filed note shows the approval it carries; a changed source is explained, not hidden', async () => {
    const base: LawExport = {
      id: 'e1', object_type: 'document', object_id: 'd9', title: 'Research note — What is the liability cap?', format: 'pdf', purpose: 'To the client',
      status: 'pending_approval', requested_by: 'alice', requested_at: '2026-10-10T11:00:00.000Z', downloads: 0, label: null,
      plan: { items: [], source: 'text', warnings: { image_pages: [] },
        approval: { kind: 'research_note', version: 1, approved_by: 'dave', approved_at: '2026-10-10T10:00:00.000Z', content_sha256: SHA } },
    };
    world({ [`GET ${M}/exports/e1`]: () => [200, { export: base }], [`GET ${M}/exports/e1/preview`]: () => [200, { pages: 1 }] });
    renderLaw('/m/M-101/exports/e1');
    expect(await screen.findByText(/Filed from a research note \(version 1\) approved by dave on .* · content bbbbbbbbbbbb…\. A changed copy cannot be exported\./)).toBeTruthy();
    cleanup();
    world({ [`GET ${M}/exports/e1`]: () => [200, { export: { ...base, status: 'failed', verification: { ok: false, source_changed: true } } }],
      [`GET ${M}/exports/e1/preview`]: () => [200, { pages: 1 }] });
    renderLaw('/m/M-101/exports/e1');
    expect((await screen.findByRole('alert')).textContent).toMatch(/document changed after this export was requested/);
  });
});
