import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import axe from 'axe-core';
import type { Me } from '../api/types';
import { matter, me, mockApi, renderLaw } from './harness';

/** S5.1 — the conversation assistant (D-062). */
vi.mock('../app/mic', () => ({
  startMic: vi.fn(async (onFrame: (pcm: ArrayBuffer) => void) => {
    onFrame(new Int16Array(16000).buffer);                                   // one second of audio
    return { stop: vi.fn(), sampleRate: 16000 };
  }),
}));

const M = '/api/legal/matters/id-M-101';
const SRC = [{ n: 1, document_id: 'd1', document_title: 'Harrowgate MSA', version_id: 'v2', version_number: 2, page_start: 9, page_end: 9,
  text: '8.1 Each party’s total aggregate liability shall not exceed USD 250,000.' }];
const STREAM = [
  { type: 'sources', citations: SRC },
  { type: 'chunk', text: 'Liability is capped at USD 250,000 [1].' },
  { type: 'done', answer: 'Liability is capped at USD 250,000 [1].', citations: SRC, status: 'answered', abstained: false, evidence: [], turn_id: 't1' },
];
const withConversation = (on = true): Me => ({ ...me(['attorney']), features: { intake: false, intake_links: false, conversation: on } });

function world(extra: Record<string, (b: any) => [number, unknown]> = {}, on = true) {
  return mockApi({
    'GET /api/auth/me': () => [200, withConversation(on)],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal')] }],
    [`GET ${M}/research`]: () => [200, { turns: [] }],
    [`POST ${M}/research/query`]: () => [200, { __ndjson: STREAM }],
    [`POST ${M}/conversation/transcribe?sample_rate=16000`]: () => [200, { text: 'What is the liability cap?', seconds: 1 }],
    [`POST ${M}/conversation/speak`]: () => [200, { __binary: 'audio/wav' }],
    ...extra,
  });
}

describe('Conversation (S5.1)', () => {
  beforeEach(() => {
    (URL as any).createObjectURL = vi.fn(() => 'blob:answer');
    HTMLMediaElement.prototype.play = vi.fn(async () => undefined);
    HTMLMediaElement.prototype.pause = vi.fn();
    try { localStorage.clear(); } catch { /* none */ }
  });
  afterEach(() => cleanup());

  it('a spoken question is transcribed on the server, answered from the documents and read aloud', async () => {
    const calls = world();
    const { container } = renderLaw('/m/M-101/conversation');
    fireEvent.click(await screen.findByRole('button', { name: 'Speak a question' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Stop and ask' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/conversation/transcribe?sample_rate=16000`)?.body).toEqual({ bytes: 32000 }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/research/query`)?.body).toEqual({ question: 'What is the liability cap?', versions: 'signed' }));
    const thread = await screen.findByRole('list', { name: 'This conversation' });
    expect(within(thread).getByRole('heading', { name: 'What is the liability cap?' })).toBeTruthy();
    expect(within(thread).getByText('Asked aloud')).toBeTruthy();
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/conversation/speak`)?.body).toEqual({ text: 'Liability is capped at USD 250,000 [1].' }));
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    expect((await axe.run(container, { rules: { 'color-contrast': { enabled: false }, region: { enabled: false } } })).violations.map((v) => v.id)).toEqual([]);
  });

  it('a typed question works too, and reading aloud can be turned off', async () => {
    const calls = world();
    renderLaw('/m/M-101/conversation');
    fireEvent.click(await screen.findByLabelText(/Read answers aloud/));
    fireEvent.change(screen.getByLabelText('Or type a question'), { target: { value: 'What is the fee?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));
    await waitFor(() => expect(calls.find((c) => c.path === `${M}/research/query`)?.body).toEqual({ question: 'What is the fee?', versions: 'signed' }));
    expect(await screen.findByText('Typed')).toBeTruthy();
    expect(calls.some((c) => c.path.endsWith('/conversation/speak'))).toBe(false);
  });

  it('without speech recognition on the server it says so and typing still works', async () => {
    world({ [`POST ${M}/conversation/transcribe?sample_rate=16000`]: () => [503, { detail: 'Speech recognition is not installed on this server. Type your question instead.' }] });
    renderLaw('/m/M-101/conversation');
    fireEvent.click(await screen.findByRole('button', { name: 'Speak a question' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Stop and ask' }));
    expect(await screen.findByText(/Speech recognition is not installed on this server/)).toBeTruthy();
    expect((screen.getByLabelText('Or type a question') as HTMLInputElement).disabled).toBe(false);
  });

  it('is in the matter menu only when the firm has turned it on', async () => {
    world({}, false);
    renderLaw('/m/M-101/research');
    const nav = await screen.findByRole('navigation', { name: 'Matter' });
    expect(within(nav).queryByRole('link', { name: 'Conversation' })).toBeNull();
    cleanup();
    world();
    renderLaw('/m/M-101/research');
    expect(within(await screen.findByRole('navigation', { name: 'Matter' })).getByRole('link', { name: 'Conversation' })).toBeTruthy();
  });
});
