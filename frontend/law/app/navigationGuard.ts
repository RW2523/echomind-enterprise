import { useEffect } from 'react';

/**
 * Things that must not be dropped silently when the person switches matter or signs out: an active
 * recording (S1.8), an unsaved draft, an upload in progress. Components register a reason while
 * active; the matter switcher asks for confirmation if any are registered.
 */
const blockers = new Map<number, string>();
let nextId = 1;

export function getBlockers(): string[] {
  return [...blockers.values()];
}

export function useBlockMatterSwitch(reason: string | null): void {
  useEffect(() => {
    if (!reason) return;
    const id = nextId++;
    blockers.set(id, reason);
    return () => { blockers.delete(id); };
  }, [reason]);
}
