import { PersonaType, AppView } from './types';

/**
 * Vertical "packs" — the building-block model. Each pack maps a subdomain (or a ?vertical=
 * override for local testing) to an isolated KB namespace (backend Step 1), a default persona,
 * tailored naming/content, and a full color theme. To take a pack live, add a Cloudflare public
 * hostname (<id>.<domain>) pointing at the frontend and upload that vertical's docs with the
 * matching namespace.
 */
export interface PackContent {
  /** Per-view nav label overrides (e.g. "Knowledge Chat" -> "Clinical Chat"). */
  navLabels?: Partial<Record<AppView, string>>;
  /** Chat empty-state headline + blurb. */
  welcomeTitle: string;
  welcomeBlurb: string;
  /** Clickable starter prompts shown in the empty chat (mapped to the seeded demo data). */
  suggestedPrompts: string[];
}

export interface PackTheme {
  /** "r g b" triples consumed by the Tailwind `accent` / `accent2` colors via CSS vars. */
  accentRgb: string;
  accent2Rgb: string;
  /** Full rgba() strings for the two ambient background glows. */
  glow1: string;
  glow2: string;
}

export interface VerticalPack {
  id: string;
  namespace: string;   // backend KB namespace — isolates this vertical's documents
  name: string;        // brand title shown in the sidebar + header
  tagline: string;
  persona: PersonaType;
  accent: string;      // hex accent (header dot, quick uses)
  icon: string;        // emoji shown in the chat empty state
  theme: PackTheme;
  content: PackContent;
}

export const PACKS: Record<string, VerticalPack> = {
  health: {
    id: 'health', namespace: 'health', name: 'EchoMind Health',
    tagline: 'Private clinical assistant', persona: PersonaType.CLINICAL, accent: '#10b981', icon: '🩺',
    theme: { accentRgb: '16 185 129', accent2Rgb: '13 148 136', glow1: 'rgba(16,185,129,0.12)', glow2: 'rgba(13,148,136,0.10)' },
    content: {
      navLabels: { [AppView.KNOWLEDGE_CHAT]: 'Clinical Chat', [AppView.TRANSCRIPTION]: 'Visit Scribe', [AppView.VOICE_CONVERSATION]: 'Voice Assistant' },
      welcomeTitle: 'Your private clinical assistant',
      welcomeBlurb: 'Ask about protocols, formulary, drug interactions, and patient notes — answered only from your clinic’s knowledge base, fully on-prem.',
      suggestedPrompts: [
        'What is first-line treatment for stage 1 hypertension?',
        'Are there drug interactions with warfarin?',
        'Summarize the latest patient visit note',
        'What is the adult colorectal screening schedule?',
      ],
    },
  },
  law: {
    id: 'law', namespace: 'law', name: 'EchoMind Law',
    tagline: 'Private legal associate', persona: PersonaType.LAWYER, accent: '#6366f1', icon: '⚖️',
    theme: { accentRgb: '99 102 241', accent2Rgb: '79 70 229', glow1: 'rgba(99,102,241,0.12)', glow2: 'rgba(129,140,248,0.10)' },
    content: {
      navLabels: { [AppView.KNOWLEDGE_CHAT]: 'Case Chat', [AppView.TRANSCRIPTION]: 'Deposition Notes', [AppView.VOICE_CONVERSATION]: 'Voice Counsel' },
      welcomeTitle: 'Your private legal associate',
      welcomeBlurb: 'Research, summarize, and redline — grounded in your firm’s contracts and playbooks, with privilege preserved on-prem.',
      suggestedPrompts: [
        'Summarize the Master Services Agreement',
        'What is the limitation of liability in the contract?',
        'Flag the risky clauses to watch for',
        'What are our standard fallback positions?',
      ],
    },
  },
  meetings: {
    id: 'meetings', namespace: 'meetings', name: 'EchoMind Meeting Rooms',
    tagline: 'Private boardroom intelligence', persona: PersonaType.MEETING, accent: '#06b6d4', icon: '🗂️',
    theme: { accentRgb: '6 182 212', accent2Rgb: '14 165 233', glow1: 'rgba(6,182,212,0.12)', glow2: 'rgba(56,189,248,0.10)' },
    content: {
      navLabels: { [AppView.KNOWLEDGE_CHAT]: 'Briefing Chat', [AppView.TRANSCRIPTION]: 'Meeting Capture', [AppView.VOICE_CONVERSATION]: 'Voice Assistant' },
      welcomeTitle: 'Private boardroom intelligence',
      welcomeBlurb: 'Capture meetings, surface decisions and action items, and brief from your company knowledge — never leaving the building.',
      suggestedPrompts: [
        'Summarize the Q3 product strategy meeting',
        'What action items were assigned, and to whom?',
        'What decisions were made?',
        'What is our remote-work policy?',
      ],
    },
  },
  retail: {
    id: 'retail', namespace: 'retail', name: 'EchoMind Retail',
    tagline: 'AI sales associate', persona: PersonaType.RETAIL, accent: '#f59e0b', icon: '🛍️',
    theme: { accentRgb: '245 158 11', accent2Rgb: '234 88 12', glow1: 'rgba(245,158,11,0.12)', glow2: 'rgba(234,88,12,0.10)' },
    content: {
      navLabels: { [AppView.KNOWLEDGE_CHAT]: 'Product Chat', [AppView.TRANSCRIPTION]: 'Floor Notes', [AppView.VOICE_CONVERSATION]: 'Voice Concierge' },
      welcomeTitle: 'Your AI sales associate',
      welcomeBlurb: 'Answer product, inventory, and financing questions instantly from your catalog — at the counter, kiosk, or showroom.',
      suggestedPrompts: [
        'What products do we carry?',
        'Compare the financing options',
        'What is the warranty on our products?',
        'What vehicles are in inventory right now?',
      ],
    },
  },
  bank: {
    id: 'bank', namespace: 'bank', name: 'EchoMind Bank',
    tagline: 'Private banking copilot', persona: PersonaType.BANKING, accent: '#22c55e', icon: '🏦',
    theme: { accentRgb: '34 197 94', accent2Rgb: '16 185 129', glow1: 'rgba(34,197,94,0.12)', glow2: 'rgba(16,185,129,0.10)' },
    content: {
      navLabels: { [AppView.KNOWLEDGE_CHAT]: 'Advisory Chat', [AppView.TRANSCRIPTION]: 'Meeting Notes', [AppView.VOICE_CONVERSATION]: 'Voice Banker' },
      welcomeTitle: 'Your private banking copilot',
      welcomeBlurb: 'Answer product, rate, and policy questions and surface required disclosures in the moment — grounded in your bank’s own materials.',
      suggestedPrompts: [
        'What deposit accounts do we offer?',
        'What are the current loan rates?',
        'What KYC disclosures are required?',
        'Walk me through the loan suitability checks',
      ],
    },
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

/** Apply the active pack's color theme to CSS variables on :root. No-op (keeps defaults) for the main app. */
export function applyPackTheme(): void {
  const p = resolvePack();
  if (!p || typeof document === 'undefined') return;
  const r = document.documentElement;
  r.style.setProperty('--accent-rgb', p.theme.accentRgb);
  r.style.setProperty('--accent2-rgb', p.theme.accent2Rgb);
  r.style.setProperty('--glow-1', p.theme.glow1);
  r.style.setProperty('--glow-2', p.theme.glow2);
}
