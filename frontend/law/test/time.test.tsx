import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me, TimeEntry } from '../api/types';
import { me, mockApi, openAccountMenu, renderLaw } from './harness';

/** S4.3 — time-entry drafts from audited activity (OPT-03; D-056). */
const entry = (x: Partial<TimeEntry> = {}): TimeEntry => ({
  id: 't1', work_date: '2026-10-05', matter: { id: 'm1', number: 'M-101', title: 'MSA renewal', client: 'Alpha Holdings' }, source: 'suggested',
  minutes: 42, hours: '0.7', minutes_suggested: 42, narrative: 'Document review (3); legal research (1); drafting (2)', narrative_suggested: 'Document review (3); legal research (1); drafting (2)',
  sessions: [{ from: '09:00', to: '09:20' }, { from: '11:00', to: '11:05' }], counts: {}, edited: false, status: 'draft', approved_at: null, updated_at: null, version: 1, ...x,
});
const withTime = (): Me => ({ ...me(['attorney']), features: { intake: false, intake_links: false, time_entries: true } });

function world(entries: TimeEntry[], extra = {}) {
  return mockApi({
    'GET /api/auth/me': () => [200, withTime()],
    'GET /api/legal/matters': () => [200, { matters: [] }],
    'GET /api/legal/time': () => [200, { entries, timezone: 'Europe/London', unit_minutes: 6 }],
    ...extra,
  });
}

describe('Time entries (S4.3)', () => {
  afterEach(() => cleanup());

  it('shows each suggested draft with its sittings, and asks the server for suggestions for the chosen days', async () => {
    const calls = world([entry()], { 'POST /api/legal/time/suggest': () => [200, { entries: [entry()], timezone: 'Europe/London' }] });
    const { container } = renderLaw('/time');
    const item = await screen.findByRole('listitem', { name: /^M-101 on / });
    expect(item.textContent).toContain('09:00–09:20, 11:00–11:05');
    expect((within(item).getByLabelText('Hours') as HTMLInputElement).value).toBe('0.7');
    expect(within(item).getByText('Suggested draft')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Suggest from my activity' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/legal/time/suggest')).toBe(true));
    expect(Object.keys(calls.find((c) => c.path === '/api/legal/time/suggest')!.body).sort()).toEqual(['from', 'to']);
    expect(screen.getByRole('link', { name: /Download approved/ }).getAttribute('aria-disabled')).toBe('true');
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('an edit is saved in minutes, then the entry is approved', async () => {
    const calls = world([entry()], { 'PATCH /api/legal/time/t1': () => [200, { entry: entry({ minutes: 60, edited: true, version: 2 }) }] });
    renderLaw('/time');
    const item = await screen.findByRole('listitem', { name: /^M-101 on / });
    expect(within(item).queryByRole('button', { name: 'Approve' })).toBeTruthy();
    fireEvent.change(within(item).getByLabelText('Hours'), { target: { value: '1.0' } });
    fireEvent.change(within(item).getByLabelText('Narrative'), { target: { value: 'Reviewed the MSA and drafted the renewal letter.' } });
    expect(within(item).queryByRole('button', { name: 'Approve' })).toBeNull();                  // save the change first
    fireEvent.click(within(item).getByRole('button', { name: 'Save changes' }));
    await waitFor(() => expect(calls.find((c) => c.method === 'PATCH')?.body).toEqual({ minutes: 60, narrative: 'Reviewed the MSA and drafted the renewal letter.', version: 1 }));
  });

  it('an approved entry is locked, can be reopened, and can be downloaded', async () => {
    const calls = world([entry({ status: 'approved', approved_at: '2026-10-06T09:00:00.000Z', version: 3 })],
      { 'POST /api/legal/time/t1/reopen': () => [200, { entry: entry({ version: 4 }) }] });
    renderLaw('/time');
    const item = await screen.findByRole('listitem', { name: /^M-101 on / });
    expect((within(item).getByLabelText('Hours') as HTMLInputElement).disabled).toBe(true);
    const link = screen.getByRole('link', { name: /Download approved/ });
    expect(link.getAttribute('aria-disabled')).toBe('false');
    expect(link.getAttribute('href')).toMatch(/^\/api\/legal\/time\/export\.csv\?from=\d{4}-\d{2}-\d{2}&to=\d{4}-\d{2}-\d{2}$/);
    fireEvent.click(within(item).getByRole('button', { name: 'Reopen' }));
    await waitFor(() => expect(calls.find((c) => c.path === '/api/legal/time/t1/reopen')?.body).toEqual({ version: 3 }));
  });

  it('a discarded entry is hidden, and without the feature there is no Time item', async () => {
    world([entry({ id: 't2', status: 'discarded' })]);
    renderLaw('/time');
    expect(await screen.findByText('No entries for these days')).toBeTruthy();
    expect(within(await openAccountMenu()).getByRole('menuitem', { name: 'Time' })).toBeTruthy();
    cleanup();
    mockApi({ 'GET /api/auth/me': () => [200, me(['attorney'])], 'GET /api/legal/matters': () => [200, { matters: [] }] });
    renderLaw('/home');
    expect(within(await openAccountMenu()).queryByRole('menuitem', { name: 'Time' })).toBeNull();
  });
});
