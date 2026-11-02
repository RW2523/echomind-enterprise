import { afterEach, describe, expect, it, vi } from 'vitest';
import { getEdition } from '../../services/backend';

/** A blip on the first request must not open the standard app in place of EchoMind Legal (2026-11-02). */
describe('Which product opens', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

  it('asks again after a network or server error, and only then assumes the standard edition', async () => {
    vi.useFakeTimers();
    const fetch = vi.fn()
      .mockRejectedValueOnce(new TypeError('network changed'))
      .mockResolvedValueOnce(new Response('', { status: 502 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ edition: 'law', auth_enabled: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const edition = getEdition();
    await vi.runAllTimersAsync();
    expect(await edition).toBe('law');
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it('an older backend without the endpoint is the standard edition at once; one never reached is too, in the end', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })));
    expect(await getEdition()).toBe('standard');
    vi.useFakeTimers();
    const down = vi.fn().mockRejectedValue(new TypeError('offline'));
    vi.stubGlobal('fetch', down);
    const edition = getEdition(3);
    await vi.runAllTimersAsync();
    expect(await edition).toBe('standard');
    expect(down).toHaveBeenCalledTimes(3);
  });
});
