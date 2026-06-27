import { PersonaType } from './types';

/**
 * Vertical "packs" — the building-block model. Each pack maps a subdomain (or a ?vertical=
 * override for local testing) to an isolated KB namespace (backend Step 1), a default persona,
 * and light branding. To take a pack live, add a Cloudflare public hostname (<id>.<domain>)
 * pointing at the frontend and upload that vertical's docs with the matching namespace.
 */
export interface VerticalPack {
  id: string;
  namespace: string;   // backend KB namespace — isolates this vertical's documents
  name: string;        // brand title shown in the header
  tagline: string;
  persona: PersonaType;
  accent: string;      // hex accent for theming
}

export const PACKS: Record<string, VerticalPack> = {
  health: {
    id: 'health', namespace: 'health', name: 'EchoMind Health',
    tagline: 'Private clinical assistant', persona: PersonaType.GENERAL, accent: '#10b981',
  },
  law: {
    id: 'law', namespace: 'law', name: 'EchoMind Law',
    tagline: 'Private legal associate', persona: PersonaType.LAWYER, accent: '#6366f1',
  },
  meetings: {
    id: 'meetings', namespace: 'meetings', name: 'EchoMind Meeting Rooms',
    tagline: 'Private boardroom intelligence', persona: PersonaType.AI_EXPERT, accent: '#06b6d4',
  },
  retail: {
    id: 'retail', namespace: 'retail', name: 'EchoMind Retail',
    tagline: 'AI sales associate', persona: PersonaType.GENERAL, accent: '#f59e0b',
  },
  bank: {
    id: 'bank', namespace: 'bank', name: 'EchoMind Bank',
    tagline: 'Private banking copilot', persona: PersonaType.FINANCIAL, accent: '#22c55e',
  },
};

let _activePack: VerticalPack | null | undefined; // undefined = not yet resolved

/**
 * Resolve the active vertical from ?vertical= (testing) or the subdomain.
 * Returns null for the main app (apex domain, www, localhost, raw IP). Cached after first call.
 */
export function resolvePack(): VerticalPack | null {
  if (_activePack !== undefined) return _activePack;
  let key = '';
  try {
    const params = new URLSearchParams(window.location.search);
    key = (params.get('vertical') || '').trim().toLowerCase();
    if (!key) {
      const labels = window.location.hostname.toLowerCase().split('.');
      const first = labels[0];
      const lastLabel = labels[labels.length - 1] || '';
      const isIp = /^\d+$/.test(lastLabel);
      // health.echomind-ajace.com -> "health"; ignore apex / www / localhost / IPs
      if (labels.length >= 3 && first && first !== 'www' && !isIp) key = first;
    }
  } catch (_) { /* SSR / no window */ }
  _activePack = key && PACKS[key] ? PACKS[key] : null;
  return _activePack;
}

/** KB namespace to send with backend requests. "" = whole KB (main app, unchanged behavior). */
export function getActiveNamespace(): string {
  const p = resolvePack();
  return p ? p.namespace : '';
}
