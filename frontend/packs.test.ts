import { afterEach, describe, expect, it, vi } from 'vitest';

// resolvePack() caches its answer per page load, so each case imports a fresh module.
async function packFor(url: string) {
  window.history.replaceState(null, '', url);
  vi.resetModules();
  const mod = await import('./packs');
  return { pack: mod.resolvePack(), ns: mod.getActiveNamespace() };
}

function setHost(hostname: string) {
  vi.spyOn(window, 'location', 'get').mockReturnValue({ ...window.location, hostname, search: '' } as Location);
}

describe('resolvePack', () => {
  afterEach(() => vi.restoreAllMocks());

  it('selects the Law pack from ?vertical=law and scopes requests to its namespace', async () => {
    const { pack, ns } = await packFor('/?vertical=law');
    expect(pack?.id).toBe('law');
    expect(ns).toBe('law');
  });

  it('ignores unknown verticals and falls back to the main app (whole KB)', async () => {
    const { pack, ns } = await packFor('/?vertical=unknown');
    expect(pack).toBeNull();
    expect(ns).toBe('');
  });

  it('resolves the pack from the first subdomain label', async () => {
    setHost('law.echomind-ajace.com');
    vi.resetModules();
    const { resolvePack } = await import('./packs');
    expect(resolvePack()?.id).toBe('law');
  });

  it('never treats a raw IP or www as a vertical', async () => {
    for (const host of ['192.168.1.20', 'www.echomind-ajace.com']) {
      setHost(host);
      vi.resetModules();
      const { resolvePack } = await import('./packs');
      expect(resolvePack()).toBeNull();
      vi.restoreAllMocks();
    }
  });
});
