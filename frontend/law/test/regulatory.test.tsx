import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, RegulatoryMatch, RegulatoryUpdate } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S4.4 (D-068) — regulatory updates: public law fetched on the host, matched to open matters, decided by a lawyer. */
const withRegulatory = (roles: string[] = ['attorney'], on = true): Me =>
  ({ ...me(roles), features: { intake: false, intake_links: false, regulatory: on } });

const UK: RegulatoryUpdate = {
  id: 'u1', source: 'legislation.gov.uk', source_name: 'legislation.gov.uk', external_id: 'uksi/2026/901', jurisdiction: 'UK',
  kind: 'UK Statutory Instrument', title: 'The Employment Rights Act 2025 (Commencement No. 3) Regulations 2026', summary: '',
  url: 'https://www.legislation.gov.uk/uksi/2026/901/made', published: '2026-10-09', subjects: [], licence: 'OGL-3.0',
};
const US: RegulatoryUpdate = {
  id: 'u2', source: 'federalregister.gov', source_name: 'Federal Register', external_id: '2026-20111', jurisdiction: 'US', kind: 'Final rule',
  title: 'Residential Lease Disclosures', summary: 'Landlords must disclose fees to a tenant.', published: '2026-10-08',
  url: 'https://www.federalregister.gov/documents/2026/10/08/2026-20111/residential-lease-disclosures', subjects: ['Housing and Urban Development Department'],
  licence: 'public-domain',
};
const MATCH: RegulatoryMatch = { update: UK, terms: ['employment'], status: 'new', decided_by: null, decided_at: null, note: null, version: 1 };
const axeClean = async (el: Element) =>
  (await axe.run(el, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id);

function world(who: Me = withRegulatory(), role: 'lead' | 'read_only' = 'lead') {
  return mockApi({
    'GET /api/auth/me': () => [200, who],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-302', 'Brightwater Clinics', 'Brightwater', { my_role: role, practice_area: 'Employment', jurisdiction: 'England & Wales' })] }],
    'GET /api/legal/regulatory': () => [200, { since: '2026-09-12', updates: [{ ...UK, matters: [{ id: 'id-M-302', number: 'M-302', title: 'Brightwater Clinics', status: 'new' }] }, { ...US, matters: [] }] }],
    'GET /api/legal/regulatory?mine=1': () => [200, { since: '2026-09-12', updates: [{ ...UK, matters: [{ id: 'id-M-302', number: 'M-302', title: 'Brightwater Clinics', status: 'new' }] }] }],
    'GET /api/legal/matters/id-M-302/regulatory': () => [200, { matches: [MATCH] }],
    'POST /api/legal/matters/id-M-302/regulatory/u1': (b) => [200, { match: { ...MATCH, status: b.status, note: b.note ?? null, decided_by: 'alice', decided_at: '2026-10-12T09:00:00.000Z', version: 2 } }],
  });
}

describe('Regulatory updates (S4.4, D-068)', () => {
  afterEach(() => cleanup());

  it('lists new public law with its source and licence, the matters it may concern, and links to the official page', async () => {
    const calls = world();
    const { container } = renderLaw('/regulatory');
    const list = await screen.findByRole('list', { name: 'Regulatory updates' });
    const [uk, us] = within(list).getAllByRole('listitem').filter((li) => li.parentElement === list);
    const official = within(uk).getByRole('link', { name: /The Employment Rights Act 2025/ });
    expect(official.getAttribute('href')).toBe(UK.url);
    expect(official.getAttribute('rel')).toBe('noopener noreferrer');
    expect(uk.textContent).toMatch(/Real public law · Open Government Licence v3\.0/);
    expect(within(uk).getByRole('link', { name: 'M-302 Brightwater Clinics' }).getAttribute('href')).toBe('/m/M-302/regulatory');
    expect(us.textContent).toMatch(/Federal Register · public domain/);
    expect(us.textContent).toMatch(/Landlords must disclose fees/);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Only updates that may concern my matters' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/regulatory?mine=1')).toBe(true));
    expect(await axeClean(container)).toEqual([]);
  });

  it('on a matter, a lawyer marks an update reviewed or not relevant, and not relevant needs a reason', async () => {
    const calls = world();
    const { container } = renderLaw('/m/M-302/regulatory');
    const item = await screen.findByRole('article', { name: /The Employment Rights Act 2025/ });
    expect(item.textContent).toMatch(/Matched on: employment/);
    fireEvent.click(within(item).getByRole('button', { name: 'Not relevant…' }));
    const save = within(item).getByRole('button', { name: 'Save: not relevant' }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(within(item).getByRole('textbox', { name: 'Why it does not concern this matter' }), { target: { value: 'Commencement only.' } });
    fireEvent.click(save);
    await waitFor(() => expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ status: 'not_relevant', note: 'Commencement only.', version: 1 }));
    expect(await within(item).findByText(/Not relevant · alice/)).toBeTruthy();
    expect(await axeClean(container)).toEqual([]);
  });

  it('a read-only member sees the matches but cannot decide', async () => {
    world(withRegulatory(), 'read_only');
    renderLaw('/m/M-302/regulatory');
    const item = await screen.findByRole('article', { name: /The Employment Rights Act 2025/ });
    expect(within(item).queryByRole('button', { name: 'Mark reviewed' })).toBeNull();
  });

  it('appears in the sidebar and on the matter only when the firm has turned it on', async () => {
    world(withRegulatory(['attorney'], false));
    renderLaw('/m/M-302/overview');
    const work = await screen.findByRole('region', { name: 'Work on this matter' });
    expect(within(work).queryByRole('link', { name: /Regulatory/ })).toBeNull();
    cleanup();
    world();
    renderLaw('/m/M-302/overview');
    expect(within(await screen.findByRole('region', { name: 'Work on this matter' })).getByRole('link', { name: /Regulatory/ })).toBeTruthy();
    expect(screen.getAllByRole('link', { name: 'Regulatory updates' }).length).toBeGreaterThan(0);
  });
});
