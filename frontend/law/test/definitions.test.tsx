import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { ArtifactView, DefinitionCheck, DefinitionFinding, RedlineContent } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S3.5 — defined terms and cross-references (DOC-04; D-049). */
const M = '/api/legal/matters/id-M-101';
const f = (x: Partial<DefinitionFinding> & Pick<DefinitionFinding, 'id' | 'kind' | 'type' | 'text' | 'message'>): DefinitionFinding =>
  ({ severity: 'medium', start: 10, end: 14, places: [], fixes: [], notes: [], ...x });
const CHECK: DefinitionCheck = {
  id: 'k1', document_id: 'd1', version_id: 'v1', version_number: 1, engine: 'definitions-s3.5', advisory: true, created_by: 'alice', created_at: '2026-10-10T10:00:00.000Z',
  summary: { definitions: 8, duplicate: 1, unused: 1, undefined: 0, inconsistent: 1, before_definition: 0, gap: 1, repeat: 0, broken: 0, outdated: 1 },
  findings: [
    f({ id: 'f1', kind: 'definition', type: 'duplicate', severity: 'high', text: 'Fees', message: '“Fees” is defined twice, with different meanings.', conflicting: true,
      places: [{ start: 248, end: 252, where: 'clause 1.2' }, { start: 490, end: 494, where: 'clause 1.5' }] }),
    f({ id: 'f2', kind: 'definition', type: 'unused', severity: 'low', text: 'Territory', message: '“Territory” is defined but never used.',
      fixes: [{ label: 'Delete the definition', start: 545, end: 586, original: '1.6 "Territory" means the United Kingdom.', revised: '', clause: '1.6' }] }),
    f({ id: 'f3', kind: 'definition', type: 'inconsistent', severity: 'low', text: 'fees', message: '“fees” is used where the defined term is “Fees”.', start: 705, end: 709,
      fixes: [{ label: 'Use “Fees”', start: 705, end: 709, original: 'fees', revised: 'Fees', clause: '2.2' }] }),
    f({ id: 'f4', kind: 'reference', type: 'outdated', text: 'clause 4 (Liability)', message: 'Clause 4 is headed Confidentiality, not Liability.', start: 828, end: 848,
      fixes: [{ label: 'Keep clause 4; name it Confidentiality', start: 828, end: 848, original: 'clause 4 (Liability)', revised: 'clause 4 (Confidentiality)', clause: '2.3' },
        { label: 'Point to clause 9 (Liability)', start: 828, end: 848, original: 'clause 4 (Liability)', revised: 'clause 9 (Liability)', clause: '2.3' }] }),
    f({ id: 'f5', kind: 'numbering', type: 'gap', severity: 'low', text: '4.3', message: 'Clause 4.2 is missing: the numbering goes from 4.1 to 4.3.', start: 954, end: 957 }),
  ],
};

function world(role: 'lead' | 'read_only' = 'lead', extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: role })] }],
    [`GET ${M}/documents/d1`]: () => [200, { document: { id: 'd1', title: 'services-agreement' } }],
    [`GET ${M}/documents/d1/definitions-checks`]: () => [200, { checks: [CHECK] }],
    ...extra,
  });
}

describe('Definitions and references (S3.5)', () => {
  afterEach(() => cleanup());

  it('findings by group, each at its place; picked corrections become a redline', async () => {
    const calls = world('lead', {
      [`POST ${M}/definitions-checks/k1/redline`]: () => [200, { redline: { id: 'r9' } }],
      [`GET ${M}/artifacts/r9`]: () => [404, { detail: 'Not Found' }],
    });
    const { container } = renderLaw('/m/M-101/documents/d1/definitions');
    const check = await screen.findByRole('article', { name: 'Definitions check' });
    expect(within(check).getByRole('status').textContent).toBe('1 term defined twice · 1 not used · 1 in another form · 1 outdated reference · 1 numbering gap');
    const terms = within(within(check).getByRole('region', { name: 'Defined terms' })).getAllByRole('listitem');
    expect(terms.map((t) => t.getAttribute('aria-label'))).toEqual(['Fees: Defined twice', 'Territory: Not used', 'fees: Another form']);
    expect(terms[0].textContent).toContain('Defined in clause 1.2 and clause 1.5');
    expect(within(terms[0]).getByRole('link', { name: 'clause 1.5' }).getAttribute('href')).toBe('/m/M-101/documents/d1/read?v=v1&at=490-494');
    const gap = within(within(check).getByRole('region', { name: 'Numbering' })).getByRole('listitem', { name: 'Clause 4.2 is missing: Gap' });
    expect(within(gap).queryByRole('radio')).toBeNull();                      // never renumbered automatically
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
    const ref = within(within(check).getByRole('region', { name: 'Cross-references' })).getByRole('listitem', { name: 'clause 4 (Liability): Outdated' });
    fireEvent.click(within(ref).getByRole('radio', { name: /Point to clause 9 \(Liability\)/ }));
    fireEvent.click(within(terms[2]).getByRole('radio', { name: /Use “Fees”/ }));
    fireEvent.click(within(check).getByRole('button', { name: 'Propose 2 corrections' }));
    await waitFor(() => expect(calls.find((c) => c.path.endsWith('/redline'))?.body).toEqual({ fixes: [{ finding: 'f4', fix: 1 }, { finding: 'f3', fix: 0 }] }));
  });

  it('a read-only member can check but not pick corrections or add notes', async () => {
    world('read_only');
    renderLaw('/m/M-101/documents/d1/definitions');
    const check = await screen.findByRole('article', { name: 'Definitions check' });
    expect(screen.getByRole('button', { name: 'Check again' })).toBeTruthy();
    expect(within(check).queryByRole('radio')).toBeNull();
    expect(within(check).queryByRole('button', { name: /Propose/ })).toBeNull();
    expect(within(check).queryByRole('button', { name: /^Add a note/ })).toBeNull();
  });

  it('a corrections redline says where it came from and changes only the words named', async () => {
    const content: RedlineContent = {
      title: 'Corrections — services-agreement', origin: 'definitions', check_id: 'k1', playbook: null, summary: { changes: 1, fitted: 0 },
      document: { id: 'd1', title: 'services-agreement', version_id: 'v1', version_number: 1, filename: 'services-agreement.txt', mime: 'text/plain' },
      changes: [{ id: 'c1', topic: 'inconsistent', title: 'Use “Fees”', clause: { number: '2.2', heading: null, start: 672, end: 747 },
        original: '2.2 The Customer shall pay the fees within thirty Business Days of invoice.', revised: '2.2 The Customer shall pay the Fees within thirty Business Days of invoice.',
        ops: [['eq', '2.2 The Customer shall pay the '], ['del', 'fees'], ['ins', 'Fees'], ['eq', ' within thirty Business Days of invoice.']],
        reason: '“fees” is used where the defined term is “Fees”.', comment: '', fitted: false, fit_note: null,
        from: { kind: 'fix', n: null, position_id: null, position_version: null, text: 'Use “Fees”', findings: ['f3'] } }],
    };
    const view: ArtifactView<RedlineContent> = {
      artifact: { id: 'r9', version: 1, status: 'draft', content, content_sha256: 'a'.repeat(64), inputs: {}, created_by: 'usr_alice', created_by_name: 'alice',
        created_at: '2026-10-10T10:05:00.000Z', origin: 'person', document_id: null, kind: 'redline', title: content.title, decisions: [] },
      versions: [{ id: 'r9', version: 1, status: 'draft', created_by_name: 'alice', created_at: '2026-10-10T10:05:00.000Z', origin: 'person' }],
      can: { approve: false, file: false, author: true },
    };
    world('lead', { [`GET ${M}/artifacts/r9`]: () => [200, view] });
    renderLaw('/m/M-101/redlines/r9');
    const red = await screen.findByRole('article', { name: 'Redline' });
    const change = within(red).getByRole('listitem', { name: 'Use “Fees”, clause 2.2' });
    expect(change.textContent).toContain('Clause 2.2 · correction');
    expect(change.textContent).not.toContain('position version');
    expect(Array.from(change.querySelectorAll('ins')).map((x) => x.textContent)).toEqual(['Fees']);
    expect(red.textContent).toContain('corrections picked by alice · definitions and references');
    expect(screen.getByRole('link', { name: 'Definitions check' }).getAttribute('href')).toBe('/m/M-101/documents/d1/definitions');
  });
});
