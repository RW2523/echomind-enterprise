import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { pageTitle } from '../counsel/CounselShell';
import { me, mockApi, renderLaw } from './harness';

/** Keyboard and screen-reader use of the shell (commercial-readiness review A3, N1, A4; 2026-10-28). */
function world() {
  return mockApi({
    'GET /api/auth/me': () => [200, me(['attorney'])],
    'GET /api/legal/matters': () => [200, { matters: [] }],
    'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }],
    'GET /api/legal/counsel/conversations': () => [200, { conversations: [] }],
  });
}

describe('keyboard', () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('the account menu takes focus when it opens and answers the arrow keys', async () => {
    world();
    renderLaw('/activity');
    fireEvent.click(await screen.findByRole('button', { name: /Account and settings/ }));
    const menu = await screen.findByRole('menu', { name: 'Account and settings' });
    const items = [...menu.querySelectorAll<HTMLElement>('[role="menuitem"]')];
    await waitFor(() => expect(document.activeElement).toBe(items[0]));
    fireEvent.keyDown(document, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(items[1]);
    fireEvent.keyDown(document, { key: 'End' });
    expect(document.activeElement).toBe(items[items.length - 1]);
    fireEvent.keyDown(document, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(items[0]);                                   // wraps around
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('on a narrow screen the closed drawer is out of reach, and Escape closes the open one', async () => {
    vi.stubGlobal('matchMedia', (q: string) => ({ matches: q.includes('max-width: 860px'), media: q, addEventListener() {}, removeEventListener() {} }));
    world();
    renderLaw('/activity');
    const rail = await screen.findByRole('complementary', { hidden: true, name: 'Sidebar' });
    expect(rail.hasAttribute('inert')).toBe(true);
    const open = screen.getByRole('button', { name: 'Open menu' });
    fireEvent.click(open);
    expect(rail.hasAttribute('inert')).toBe(false);
    await waitFor(() => expect(rail.contains(document.activeElement)).toBe(true));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(rail.hasAttribute('inert')).toBe(true);
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Open menu' }));
  });

  it('the collapsed rail keeps every link named by its text, with no label that would clash with a field', async () => {
    world();
    renderLaw('/activity');
    fireEvent.click(await screen.findByRole('button', { name: 'Collapse the sidebar' }));
    const rail = screen.getByRole('navigation', { name: 'Workspace' });
    for (const name of ['Chat', 'Matters', 'Drafts', 'Activity']) {
      const link = within(rail).getByRole('link', { name });
      expect(link.getAttribute('aria-label')).toBeNull();
    }
    expect(within(rail).getByRole('button', { name: 'New chat' })).toBeTruthy();
  });

  it('every page has its own title', () => {
    expect(pageTitle('/chat')).toBe('Chat — EchoMind Legal');
    expect(pageTitle('/chat/abc')).toBe('Chat — EchoMind Legal');
    expect(pageTitle('/m/MAT-1042/obligations')).toBe('MAT-1042 · Obligations — EchoMind Legal');
    expect(pageTitle('/admin/users')).toBe('Users & roles — EchoMind Legal');
    expect(pageTitle('/drafts')).toBe('Drafts — EchoMind Legal');
  });
});
