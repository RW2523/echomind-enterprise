import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { _resetCaptureForTests } from '../app/capture';
import { INITIAL, reduce } from '../counsel/voice/useVoice';
import { matter, me, mockApi, renderLaw } from './harness';

/** Voice mode in the chat (owner's request 2026-11-02): the original EchoMind voice conversation, speaking turns of
 *  this conversation that are answered from the matter's documents and saved with their sources. */
const hex = (c: string) => c.repeat(32);
const T1 = `turn_${hex('1')}`;

class FakeSocket {
  static all: FakeSocket[] = [];
  static OPEN = 1;
  url: string; readyState = 1; sent: string[] = [];
  onopen: (() => void) | null = null; onmessage: ((e: { data: string }) => void) | null = null; onclose: ((e: { code: number }) => void) | null = null;
  constructor(url: string) { this.url = url; FakeSocket.all.push(this); }
  send(s: string) { this.sent.push(s); }
  close() { this.readyState = 3; }
  say(msg: object) { act(() => { this.onmessage?.({ data: JSON.stringify(msg) }); }); }
}

function fakeAudio() {
  const node = () => ({ connect() {}, disconnect() {}, fftSize: 1024, getByteTimeDomainData() {}, port: { onmessage: null } });
  class Ctx {
    currentTime = 0; sampleRate = 24000; destination = {}; audioWorklet = { addModule: vi.fn(async () => {}) };
    createMediaStreamSource() { return node(); }
    createAnalyser() { return node(); }
    createGain() { return { ...node(), gain: { value: 1, cancelScheduledValues() {}, setValueAtTime() {}, linearRampToValueAtTime() {} } }; }
    createBuffer(_c: number, n: number) { return { duration: n / 24000, copyToChannel() {} }; }
    createBufferSource() { return { ...node(), buffer: null, playbackRate: { value: 1 }, start() {}, stop() {}, onended: null }; }
    close() { return Promise.resolve(); }
  }
  (globalThis as any).AudioContext = Ctx;
  (globalThis as any).AudioWorkletNode = class { port = { onmessage: null }; connect() {} disconnect() {} };
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop() {}, addEventListener() {} }] })) } });
}

function world(handlers: Record<string, (body: any) => [number, unknown]> = {}, conversation = true) {
  return mockApi({
    'GET /api/auth/me': () => [200, { ...me(), features: { intake: false, intake_links: false, conversation } }],
    'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'attorney' as any })] }],
    'GET /api/legal/reviews': () => [200, { waiting: [], mine: [], count: 0 }],
    'GET /api/legal/counsel/conversations': () => [200, { conversations: [] }],
    'POST /api/legal/counsel/conversations': (b) => [201, { conversation: { id: 'conv_1', matter_id: b.matter_id, title: 'New conversation' } }],
    ...handlers,
  });
}

const saved = { conversation: { id: 'conv_1', matter_id: 'id-M-101', title: 'When is delivery due?' }, turns: [
  { id: T1, seq: 1, parent_turn_id: null, question: 'When is delivery due?', intent: 'ask_fact', status: 'completed', created_at: 't',
    envelope: { schema_version: '1.0', message_id: T1, matter_id: 'id-M-101', summary: 's', suggested_actions: [], components: [
      { id: `cmp_${hex('c')}`, version: 1, type: 'source_answer', props: { answer: 'Delivery is due by May 15, 2026 [1].', state: 'answered', unsupported: [], notes: [],
        citations: [{ id: `cit_${hex('a')}`, n: 1, document_id: 'd1', document_title: 'Services Agreement', clause: '4.2', verified: true }] } }] } }] };

beforeEach(() => {
  localStorage.clear(); _resetCaptureForTests(); localStorage.setItem('echomind.counsel.matter', 'id-M-101');
  FakeSocket.all = [];
  (globalThis as any).WebSocket = FakeSocket;
});
afterEach(() => { cleanup(); localStorage.clear(); });

describe('Read aloud', () => {
  it('reads an answer in the server\'s voice, stops when pressed again, and says when it cannot', async () => {
    let said: any = null;
    let available = true;
    (URL as any).createObjectURL = vi.fn(() => 'blob:answer');
    (URL as any).revokeObjectURL = vi.fn();
    const play = vi.fn(async () => undefined);
    HTMLMediaElement.prototype.play = play;
    HTMLMediaElement.prototype.pause = vi.fn();
    world({ 'GET /api/legal/counsel/conversations/conv_1': () => [200, saved],
            'POST /api/legal/matters/id-M-101/conversation/speak': (b) => { said = b; return available ? [200, { __binary: 'audio/wav' }]
              : [503, { detail: 'Reading aloud is not available on this server.' }]; } });
    renderLaw('/chat/conv_1');
    const read = await screen.findByRole('button', { name: 'Read aloud' });
    fireEvent.click(read);
    expect(await screen.findByRole('button', { name: 'Stop reading' })).toBeTruthy();
    expect(said).toEqual({ text: 'Delivery is due by May 15, 2026 [1].' });                 // the server removes the source marks
    await waitFor(() => expect(play).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', { name: 'Stop reading' }));
    expect(await screen.findByRole('button', { name: 'Read aloud' })).toBeTruthy();
    available = false;
    fireEvent.click(screen.getByRole('button', { name: 'Read aloud' }));
    expect(await screen.findByText('Reading aloud is not available on this server.')).toBeTruthy();
  });

  it('is offered only where voice is on', async () => {
    world({ 'GET /api/legal/counsel/conversations/conv_1': () => [200, saved] }, false);
    renderLaw('/chat/conv_1');
    await screen.findByRole('button', { name: 'Copy answer' });
    expect(screen.queryByRole('button', { name: 'Read aloud' })).toBeNull();
  });
});

describe('What the voice panel shows', () => {
  it('follows the voice service: listening, the words being heard, checking, what it says', () => {
    let v = reduce(INITIAL, { type: 'hello' });
    expect(v.phase).toBe('listening');
    v = reduce(v, { type: 'partial_transcript', text: 'when is deliv' });
    expect(v).toMatchObject({ phase: 'hearing', partial: 'when is deliv' });
    v = reduce(v, { type: 'asr_final', text: 'When is delivery due?' });
    expect(v).toMatchObject({ phase: 'thinking', heard: 'When is delivery due?', partial: '' });
    v = reduce(v, { type: 'event', event: 'FILLER_SPEAKING', text: 'Let me check the documents.' });
    expect(v.holding).toBe('Let me check the documents.');
    v = reduce(v, { type: 'assistant_text_partial', text: 'Delivery is due by May 15, 2026.' });
    expect(v.saying).toBe('Delivery is due by May 15, 2026.');
    expect(reduce(v, { type: 'turn_saved', turn_id: T1 })).toMatchObject({ heard: '', saying: '' });   // the thread shows it now
    expect(reduce(v, { type: 'error', fatal: true, where: 'stt' }).phase).toBe('error');
    expect(reduce(v, { type: 'error', where: 'tts' })).toBe(v);                 // a passing problem does not end voice
  });
});

describe('Voice mode in the chat', () => {
  it('is offered only where voice is on, with an empty question box', async () => {
    world({}, false);
    renderLaw('/chat');
    await screen.findByRole('textbox', { name: 'Ask EchoMind' });
    expect(screen.queryByRole('button', { name: 'Talk with voice' })).toBeNull();
    cleanup();
    world();
    renderLaw('/chat');
    const box = await screen.findByRole('textbox', { name: 'Ask EchoMind' });
    expect(await screen.findByRole('button', { name: 'Talk with voice' })).toBeTruthy();
    fireEvent.change(box, { target: { value: 'typed' } });
    expect(screen.queryByRole('button', { name: 'Talk with voice' })).toBeNull();   // Send takes its place
  });

  it('without a chosen matter, asks which one first and then starts', async () => {
    fakeAudio();
    localStorage.removeItem('echomind.counsel.matter');
    world({ 'GET /api/legal/matters': () => [200, { matters: [matter('M-101', 'MSA renewal', 'Alpha Holdings', { my_role: 'attorney' as any }),
                                                           matter('M-114', 'Series B', 'Beta Ltd', { my_role: 'attorney' as any })] }] });
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: 'Talk with voice' }));
    const pick = await screen.findByRole('region', { name: 'Which matter is this about?' });
    expect(FakeSocket.all).toHaveLength(0);
    fireEvent.click(within(pick).getByRole('button', { name: 'M-101 · MSA renewal' }));
    await waitFor(() => expect(FakeSocket.all.length).toBe(1));
    expect(FakeSocket.all[0].url).toMatch(/\?matter=id-M-101&conversation=conv_1$/);
  });

  it('a matter\'s "Talk by voice" opens voice mode in a new chat about that matter', async () => {
    fakeAudio();
    localStorage.removeItem('echomind.counsel.matter');
    world();
    const { router } = renderLaw('/m/M-101/conversation');
    expect(await screen.findByRole('region', { name: 'Voice conversation' })).toBeTruthy();
    await waitFor(() => expect(FakeSocket.all.length).toBe(1));
    expect(router.state.location.pathname).toBe('/chat/conv_1');
  });

  it('shows what is asked and answered in the thread as it is spoken, then the saved turn in its place', async () => {
    fakeAudio();
    world({ 'GET /api/legal/counsel/conversations/conv_1': () => [200, saved] });
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: 'Talk with voice' }));
    await waitFor(() => expect(FakeSocket.all.length).toBe(1));
    const ws = FakeSocket.all[0];
    act(() => ws.onopen?.());
    ws.say({ type: 'hello' });
    ws.say({ type: 'asr_final', text: 'When is delivery due?' });
    const live = await screen.findByRole('article', { name: 'Being answered by voice' });
    expect(live.textContent).toContain('When is delivery due?');
    expect(live.textContent).toContain("Searching this matter's documents…");
    ws.say({ type: 'assistant_text_partial', text: 'Delivery is due by May 15' });
    expect(live.textContent).toContain('Delivery is due by May 15');
    ws.say({ type: 'turn_saved', turn_id: T1, conversation_id: 'conv_1' });
    expect(await screen.findByRole('button', { name: /Source 1: Services Agreement/ })).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole('article', { name: 'Being answered by voice' })).toBeNull());
    expect(screen.getAllByText('When is delivery due?')).toHaveLength(1);                     // shown once, as the saved turn
  });

  it('starts a conversation, speaks in it, and shows each saved answer in the thread with its sources', async () => {
    fakeAudio();
    const calls = world({ 'GET /api/legal/counsel/conversations/conv_1': () => [200, saved] });
    const { router } = renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: 'Talk with voice' }));
    const panel = await screen.findByRole('region', { name: 'Voice conversation' });
    await waitFor(() => expect(FakeSocket.all.length).toBe(1));
    expect(router.state.location.pathname).toBe('/chat/conv_1');
    expect(calls.find((c) => c.method === 'POST' && c.path === '/api/legal/counsel/conversations')!.body).toEqual({ matter_id: 'id-M-101' });
    const ws = FakeSocket.all[0];
    expect(ws.url).toMatch(/\/voice\/ws\?matter=id-M-101&conversation=conv_1$/);
    act(() => ws.onopen?.());
    expect(JSON.parse(ws.sent[0])).toEqual({ type: 'set_context', listen_only: false });       // never ambient listening
    ws.say({ type: 'hello' });
    expect(panel.textContent).toContain('Listening');
    expect(screen.getByText('Recording')).toBeTruthy();                       // the header shows the open microphone
    ws.say({ type: 'partial_transcript', text: 'when is delivery' });
    expect(panel.textContent).toContain('“when is delivery”');
    ws.say({ type: 'asr_final', text: 'When is delivery due?' });
    expect(panel.textContent).toContain('Checking the documents…');
    ws.say({ type: 'assistant_text_partial', text: 'Delivery is due by May 15, 2026.' });
    ws.say({ type: 'turn_saved', turn_id: T1, conversation_id: 'conv_1' });
    expect(await screen.findByRole('button', { name: /Source 1: Services Agreement/ })).toBeTruthy();
    expect(screen.queryByRole('textbox', { name: 'Ask EchoMind' })).toBeNull();     // the panel takes the box's place
    fireEvent.click(screen.getByRole('button', { name: 'Mute' }));
    expect(panel.textContent).toContain('Microphone muted');
    fireEvent.click(screen.getByRole('button', { name: 'End voice' }));
    expect(await screen.findByRole('textbox', { name: 'Ask EchoMind' })).toBeTruthy();
    await waitFor(() => expect(screen.getByText('Not recording')).toBeTruthy());
  });

  it('says plainly when the browser refuses the microphone or the service refuses the session', async () => {
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => { throw new Error('denied'); }) } });
    world();
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: 'Talk with voice' }));
    expect(await screen.findByText('The browser refused microphone access. Allow it and start voice again.')).toBeTruthy();
    expect(FakeSocket.all).toHaveLength(0);
    cleanup();
    fakeAudio();
    world();
    renderLaw('/chat');
    fireEvent.click(await screen.findByRole('button', { name: 'Talk with voice' }));
    await waitFor(() => expect(FakeSocket.all.length).toBe(1));
    act(() => FakeSocket.all[0].onclose?.({ code: 4401 }));
    expect(await screen.findByText('Your session ended, so voice stopped. Sign in again to continue.')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy();
  });
});
