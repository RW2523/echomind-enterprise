import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** D-065 — Home as a dashboard, the command palette, and each matter's tools. */
const withFeatures = (roles: string[], features: Partial<NonNullable<Me['features']>> = {}): Me =>
  ({ ...me(roles), features: { intake: false, intake_links: false, conversation: true, summaries: true, panel: true, ...features } });

function world(who: Me = withFeatures(['attorney'])) {
  return mockApi({
    'GET /api/auth/me': () => [200, who],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal'), matter('M-301', 'Northwind v Calder Freight')] }],
    'GET /api/legal/reviews': () => [200, { waiting: [{ kind: 'note', id: 'n1', title: 'Research note: liability cap', matter: { number: 'M-101', title: 'MSA renewal' },
      by: 'bob', at: '2026-10-11T10:00:00Z', path: '/m/M-101/notes/n1' }], mine: [], count: 1 }],
    'GET /api/legal/matters/id-M-101/research': () => [200, { turns: [] }],
    'GET /api/legal/matters/id-M-301/research': () => [200, { turns: [] }],
    'GET /api/legal/matters/id-M-301/conversation': () => [200, {}],
    'GET /api/legal/matters/id-M-101/parties': () => [200, { parties: [] }],
  });
}

const axeClean = async (el: Element) =>
  (await axe.run(el, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id);

describe('Home, the command palette and the matter tools (D-065)', () => {
  afterEach(() => cleanup());

  it('Home greets, counts, lists the open matters as cards and what is waiting for a decision', async () => {
    world();
    const { container } = renderLaw('/home');
    expect(await screen.findByRole('heading', { name: /^Good (morning|afternoon|evening), Alice$/ })).toBeTruthy();
    const stats = screen.getByRole('group', { name: 'At a glance' });
    await waitFor(() => expect(within(stats).getByText('Open matters').nextSibling?.textContent).toBe('2'));
    expect(screen.getByRole('heading', { name: 'Your open matters' })).toBeTruthy();
    expect(screen.getByRole('link', { name: 'M-301 Northwind v Calder Freight' }).getAttribute('href')).toBe('/m/M-301/overview');
    const waiting = screen.getByRole('region', { name: 'Waiting for you' });
    expect(await within(waiting).findByRole('link', { name: /Research note: liability cap/ })).toBeTruthy();
    expect(await axeClean(container)).toEqual([]);
  });

  it('a quick action asks which matter, then opens that tool on it', async () => {
    world();
    renderLaw('/home');
    fireEvent.click(await screen.findByRole('button', { name: /Talk to a matter/ }));
    const palette = await screen.findByRole('dialog', { name: 'Search or jump to' });
    const input = within(palette).getByRole('combobox', { name: 'Talk to which matter?' });
    fireEvent.change(input, { target: { value: 'northwind' } });
    expect(within(palette).getAllByRole('option').map((o) => o.textContent)).toEqual([expect.stringContaining('M-301 · Northwind v Calder Freight')]);
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(await screen.findByRole('region', { name: 'Talk to this matter' })).toBeTruthy();
  });

  it('Ctrl+K opens the palette anywhere; a matter searched for brings its tools; Escape closes it', async () => {
    world();
    const { container } = renderLaw('/home');
    await screen.findByRole('heading', { name: /Good/ });
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
    const palette = await screen.findByRole('dialog', { name: 'Search or jump to' });
    expect(await axeClean(container)).toEqual([]);
    const input = within(palette).getByRole('combobox');
    fireEvent.change(input, { target: { value: 'M-301 research' } });
    const options = within(palette).getAllByRole('option');
    expect(options[0].textContent).toContain('M-301 · Research');
    fireEvent.keyDown(input, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Search or jump to' })).toBeNull());
    fireEvent.click(screen.getByRole('button', { name: /Search or jump to/ }));
    expect(await screen.findByRole('dialog', { name: 'Search or jump to' })).toBeTruthy();
  });

  it('a matter overview offers its tools as tiles, only the ones the firm has turned on', async () => {
    world(withFeatures(['attorney'], { conversation: false, panel: false }));
    renderLaw('/m/M-101/overview');
    const work = await screen.findByRole('region', { name: 'Work on this matter' });
    const names = within(work).getAllByRole('link').map((a) => a.querySelector('strong')?.textContent);
    expect(names).toContain('Research');
    expect(names).toContain('Meetings');
    expect(names).not.toContain('Conversation');
    expect(names).not.toContain('Review panel');
  });
});
