/**
 * Silent Assistant v2 — shared helpers: tone -> Tailwind classes, static fallback profiles
 * (mirrors backend/app/silent_assistant/profiles.py so the UI works before the first
 * `session` ack / when GET /scenarios is unavailable), sorting & derivation utilities.
 */
import type {
  ActionItem,
  AnalysisCard,
  AnalysisLabel,
  CheckStatus,
  RecordKind,
  Scenario,
  ScenarioId,
  SentenceCheck,
  TagSpec,
  TagTone,
} from '../types';

// ── Tone -> Tailwind classes (all literal so Tailwind's scanner keeps them) ────

export interface ToneClasses {
  bg: string;      // soft background
  border: string;  // border colour
  text: string;    // foreground text
  badge: string;   // filled chip
  bar: string;     // solid bar / dot
  mark: string;    // <mark> highlight
}

export const TONE_CLASSES: Record<string, ToneClasses> = {
  green:  { bg: 'bg-emerald-500/10', border: 'border-emerald-500/40', text: 'text-emerald-300', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', bar: 'bg-emerald-500', mark: 'bg-emerald-400/30 text-emerald-100' },
  red:    { bg: 'bg-rose-500/10',    border: 'border-rose-500/40',    text: 'text-rose-300',    badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',          bar: 'bg-rose-500',    mark: 'bg-rose-400/30 text-rose-100' },
  yellow: { bg: 'bg-amber-300/5',    border: 'border-amber-300/50',   text: 'text-amber-200',   badge: 'bg-transparent text-amber-200 border-amber-300/60',        bar: 'bg-amber-300',   mark: 'bg-amber-300/30 text-amber-50' },
  blue:   { bg: 'bg-sky-500/10',     border: 'border-sky-500/40',     text: 'text-sky-300',     badge: 'bg-sky-500/20 text-sky-300 border-sky-500/40',             bar: 'bg-sky-500',     mark: 'bg-sky-400/30 text-sky-100' },
  violet: { bg: 'bg-violet-500/10',  border: 'border-violet-500/40',  text: 'text-violet-300',  badge: 'bg-violet-500/20 text-violet-300 border-violet-500/40',    bar: 'bg-violet-500',  mark: 'bg-violet-400/30 text-violet-100' },
  indigo: { bg: 'bg-indigo-500/10',  border: 'border-indigo-500/40',  text: 'text-indigo-300',  badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',    bar: 'bg-indigo-500',  mark: 'bg-indigo-400/30 text-indigo-100' },
  teal:   { bg: 'bg-teal-500/10',    border: 'border-teal-500/40',    text: 'text-teal-300',    badge: 'bg-teal-500/20 text-teal-300 border-teal-500/40',          bar: 'bg-teal-500',    mark: 'bg-teal-400/30 text-teal-100' },
  orange: { bg: 'bg-orange-500/10',  border: 'border-orange-500/40',  text: 'text-orange-300',  badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',    bar: 'bg-orange-500',  mark: 'bg-orange-400/30 text-orange-100' },
  amber:  { bg: 'bg-amber-500/10',   border: 'border-amber-500/40',   text: 'text-amber-300',   badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',       bar: 'bg-amber-500',   mark: 'bg-amber-400/30 text-amber-100' },
  grey:   { bg: 'bg-slate-500/10',   border: 'border-slate-500/40',   text: 'text-slate-300',   badge: 'bg-slate-500/20 text-slate-300 border-slate-500/40',       bar: 'bg-slate-500',   mark: 'bg-slate-400/30 text-slate-100' },
  cyan:   { bg: 'bg-cyan-500/10',    border: 'border-cyan-500/40',    text: 'text-cyan-300',    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',          bar: 'bg-cyan-500',    mark: 'bg-cyan-400/30 text-cyan-100' },
};

export const NEUTRAL_TONE: ToneClasses = {
  bg: 'bg-white/5', border: 'border-white/15', text: 'text-slate-300',
  badge: 'bg-white/10 text-slate-300 border-white/15', bar: 'bg-slate-400', mark: 'bg-white/20 text-white',
};

export function toneClasses(tone?: TagTone | string | null): ToneClasses {
  return (tone && TONE_CLASSES[tone]) || NEUTRAL_TONE;
}

// ── Static fallback tag definitions (from profiles.py) ────────────────────────

export const FALLBACK_TAGS: Record<string, TagSpec> = {
  'supported':          { id: 'supported',          label: 'Supported',          tone: 'green',  proof: 'quote',  description: 'Confirmed by a source; the quote is the proof.' },
  'contradicted':       { id: 'contradicted',       label: 'Wrong',              tone: 'red',    proof: 'quote',  description: 'Conflicts with a source; the quote shows the correct value.' },
  'unverified':         { id: 'unverified',         label: 'Unverified',         tone: 'yellow', proof: 'none',   description: 'A claim, but no source confirms or denies it.' },
  'record-found':       { id: 'record-found',       label: 'Record found',       tone: 'blue',   proof: 'record', description: 'A record mentioning this person/identifier was pulled.' },
  'personal-detail':    { id: 'personal-detail',    label: 'Personal detail',    tone: 'violet', proof: 'span',   description: 'Name, ID, phone, DOB or similar was spoken; used to look up records.' },
  'contract-clause':    { id: 'contract-clause',    label: 'Contract',           tone: 'indigo', proof: 'quote',  description: 'Matches a clause in a contract/agreement/terms document.' },
  'policy':             { id: 'policy',             label: 'Policy',             tone: 'teal',   proof: 'quote',  description: 'Matches a policy / regulation / SOP statement.' },
  'risk':               { id: 'risk',               label: 'Risk',               tone: 'orange', proof: 'rule',   description: 'Risky claim or promise per a source or a domain rule.' },
  'violating':          { id: 'violating',          label: 'Violation',          tone: 'red',    proof: 'rule',   description: 'Breaks a rule or policy in the sources.' },
  'disclosure-missing': { id: 'disclosure-missing', label: 'Disclosure missing', tone: 'orange', proof: 'rule',   description: 'A required disclosure/step was not given.' },
  'related-case':       { id: 'related-case',       label: 'Related case',       tone: 'indigo', proof: 'quote',  description: 'A related matter or precedent in the sources.' },
  'action-item':        { id: 'action-item',        label: 'Action item',        tone: 'amber',  proof: 'span',   description: 'Something someone must do.' },
  'commitment':         { id: 'commitment',         label: 'Commitment',         tone: 'amber',  proof: 'span',   description: 'A promise made to the other party.' },
  'question':           { id: 'question',           label: 'Question',           tone: 'grey',   proof: 'span',   description: 'A question that may need a record lookup.' },
  'decision':           { id: 'decision',           label: 'Decision',           tone: 'cyan',   proof: 'span',   description: 'A decision taken.' },
  'reference':          { id: 'reference',          label: 'Reference',          tone: 'cyan',   proof: 'quote',  description: 'Related material in the knowledge base.' },
};

const pick = (...ids: string[]): TagSpec[] => ids.map((i) => FALLBACK_TAGS[i]).filter(Boolean);
const COMMON = ['supported', 'contradicted', 'unverified', 'record-found', 'personal-detail', 'action-item', 'commitment', 'question', 'reference'];

/** Mirrors profiles.py `public()` output; used until GET /api/transcribe/scenarios answers. */
export const FALLBACK_SCENARIOS: Scenario[] = [
  {
    id: 'customer_care', label: 'Customer call',
    description: 'Support / customer-care agent speaking with a caller. Verifies what the agent says against policy & T&Cs; pulls the caller\'s records from stated details.',
    default_namespace: '', roles: { me: 'agent', other: 'caller' }, analysis_mode_default: 'flags_and_records',
    tag_vocab: pick(...COMMON, 'policy', 'contract-clause', 'risk'),
  },
  {
    id: 'legal', label: 'Legal consultation',
    description: 'Lawyer speaking with a client. Pulls the client\'s matter/previous records and related cases; flags contract clauses, risks and violations with quotes.',
    default_namespace: 'law', roles: { me: 'lawyer', other: 'client' }, analysis_mode_default: 'flags_and_records',
    tag_vocab: pick(...COMMON, 'contract-clause', 'related-case', 'violating', 'risk', 'policy', 'decision'),
  },
  {
    id: 'banking', label: 'Banking',
    description: 'Bank agent speaking with a client. Pulls KYC/account/product records; flags missing disclosures, mis-selling and policy violations with quotes.',
    default_namespace: 'bank', roles: { me: 'banker', other: 'client' }, analysis_mode_default: 'flags_and_records',
    tag_vocab: pick(...COMMON, 'disclosure-missing', 'violating', 'risk', 'policy', 'contract-clause'),
  },
  {
    id: 'general', label: 'General conversation',
    description: 'Two or more people talking (meeting, interview, discussion). Every claim is checked against the knowledge base; action items and decisions are captured.',
    default_namespace: '', roles: { me: 'speaker_a', other: 'speaker_b' }, analysis_mode_default: 'flags_only',
    tag_vocab: pick('supported', 'contradicted', 'unverified', 'record-found', 'action-item', 'commitment', 'decision', 'risk', 'reference', 'question'),
  },
];

export const AUTO_SCENARIO: Scenario = {
  id: 'auto', label: 'Auto-detect',
  description: 'Let the assistant pick the scenario from what is being said (you can switch any time).',
  default_namespace: '', roles: { me: 'me', other: 'other' }, analysis_mode_default: 'flags_and_records',
  tag_vocab: Object.values(FALLBACK_TAGS),
};

export const SCENARIO_ICONS: Record<string, string> = {
  auto: '✨', customer_care: '🎧', legal: '⚖️', banking: '🏦', general: '💬',
};

/** Human label for a role id ('agent' -> 'Agent', 'speaker_a' -> 'Speaker A'). */
export function roleLabel(role?: string | null): string {
  if (!role) return 'Unknown';
  return role.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function findScenario(list: Scenario[], id: ScenarioId | string | null | undefined): Scenario | undefined {
  if (!id) return undefined;
  if (id === 'auto') return AUTO_SCENARIO;
  return list.find((s) => s.id === id) ?? FALLBACK_SCENARIOS.find((s) => s.id === id);
}

/** Look up a TagSpec by id from a vocab (falls back to the static table, then a neutral spec). */
export function tagSpecFor(vocab: TagSpec[] | undefined, id: string, inline?: { label?: string; tone?: TagTone }): TagSpec {
  const v = vocab?.find((t) => t.id === id) ?? FALLBACK_TAGS[id];
  if (v) return { ...v, label: inline?.label ?? v.label, tone: inline?.tone ?? v.tone };
  return { id, label: inline?.label ?? roleLabel(id), tone: inline?.tone ?? 'grey', proof: 'none' };
}

// ── Check helpers ─────────────────────────────────────────────────────────────

/** Coerce any AnalysisCard (legacy or v2) into a SentenceCheck with safe defaults. */
export function asCheck(card: AnalysisCard | SentenceCheck): SentenceCheck {
  const c = card as Partial<SentenceCheck> & AnalysisCard;
  return {
    ...c,
    sentence_id: c.sentence_id ?? c.segment_id,
    sentence_text: c.sentence_text ?? c.segment_text ?? '',
    tags: Array.isArray(c.tags) ? c.tags : [],
    evidence: Array.isArray(c.evidence) ? c.evidence : [],
    record_ids: Array.isArray(c.record_ids) ? c.record_ids : [],
    searched_docs: Array.isArray(c.searched_docs) ? c.searched_docs : [],
    source_chunks: Array.isArray(c.source_chunks) ? c.source_chunks : [],
    confidence: typeof c.confidence === 'number' ? c.confidence : 0,
    explanation: c.explanation ?? '',
    status: c.status ?? 'checked',
  };
}

export const PROBLEM_TAGS = new Set(['contradicted', 'violating', 'risk', 'disclosure-missing']);
const PROBLEM_LABELS = new Set<AnalysisLabel>(['Contradicted', 'Violating', 'Risky Statement']);

export function checkTagIds(check: SentenceCheck): string[] {
  const ids = check.tags?.map((t) => t.tag) ?? [];
  if (!ids.length && check.verdict) ids.push(check.verdict);
  return ids;
}

export function isProblem(check: SentenceCheck): boolean {
  if (check.verdict === 'contradicted') return true;
  if (checkTagIds(check).some((t) => PROBLEM_TAGS.has(t))) return true;
  return PROBLEM_LABELS.has(check.label);
}

/** Problems first, then supported, then unverified/reference/other. Lower = earlier. */
export function checkRank(check: SentenceCheck): number {
  if (check.verdict === 'contradicted' || checkTagIds(check).includes('contradicted') || check.label === 'Contradicted') return 0;
  const ids = checkTagIds(check);
  if (ids.includes('violating') || check.label === 'Violating') return 1;
  if (ids.includes('disclosure-missing')) return 2;
  if (ids.includes('risk') || check.label === 'Risky Statement') return 3;
  if (check.verdict === 'supported' || ids.includes('supported') || check.label === 'Supported') return 4;
  if (ids.includes('record-found') || ids.includes('contract-clause') || ids.includes('policy') || ids.includes('related-case')) return 5;
  if (check.verdict === 'unverified' || ids.includes('unverified') || check.label === 'Unverified') return 6;
  return 7;
}

export function sortChecksProblemsFirst(checks: SentenceCheck[]): SentenceCheck[] {
  return checks
    .map((c, i) => ({ c, i }))
    .sort((a, b) => checkRank(a.c) - checkRank(b.c) || b.i - a.i) // newest first within a rank
    .map((x) => x.c);
}

/** Primary tag id used to colour a check (problem tag > verdict > first tag > legacy label). */
export function primaryTagId(check: SentenceCheck): string | null {
  const ids = checkTagIds(check);
  for (const p of ['contradicted', 'violating', 'disclosure-missing', 'risk']) if (ids.includes(p)) return p;
  if (check.verdict) return check.verdict;
  if (ids.length) return ids[0];
  return legacyLabelToTag(check.label);
}

export function legacyLabelToTag(label?: AnalysisLabel | string | null): string | null {
  switch (label) {
    case 'Supported': return 'supported';
    case 'Contradicted': return 'contradicted';
    case 'Unverified': return 'unverified';
    case 'Violating': return 'violating';
    case 'Risky Statement': return 'risk';
    case 'Relevant': return 'reference';
    default: return null;
  }
}

/** Legacy `label` from verdict/tags — same mapping as PROTOCOL.md (used when the server omits it). */
export function deriveLegacyLabel(check: Partial<SentenceCheck>): AnalysisLabel {
  const ids = (check.tags ?? []).map((t) => t.tag);
  if (check.verdict === 'supported') return 'Supported';
  if (check.verdict === 'contradicted') return 'Contradicted';
  if (ids.includes('violating')) return 'Violating';
  if (ids.includes('risk') || ids.includes('disclosure-missing')) return 'Risky Statement';
  if (check.verdict === 'unverified') return 'Unverified';
  return 'Relevant';
}

const ACTION_TAGS = new Set(['action-item', 'commitment', 'decision']);

export function deriveActionItems(checks: SentenceCheck[]): ActionItem[] {
  const out: ActionItem[] = [];
  for (const c of checks) {
    for (const t of c.tags ?? []) {
      if (ACTION_TAGS.has(t.tag)) {
        out.push({ id: `${c.id}:${t.tag}`, sentence_id: c.sentence_id, tag: t.tag, role: c.role, text: c.sentence_text || c.segment_text, check: c });
      }
    }
  }
  return out;
}

export function statusGlyph(status?: CheckStatus | null): { glyph: string; title: string; cls: string } {
  switch (status) {
    case 'pending': return { glyph: '…', title: 'Checking', cls: 'text-cyan-400 animate-pulse' };
    case 'checked': return { glyph: '✓', title: 'Checked', cls: 'text-emerald-400' };
    case 'skipped': return { glyph: '·', title: 'Skipped (small talk)', cls: 'text-slate-600' };
    case 'timeout': return { glyph: '⏱', title: 'Timed out', cls: 'text-amber-400' };
    case 'no_tags': return { glyph: '·', title: 'Nothing to flag', cls: 'text-slate-600' };
    default: return { glyph: '', title: '', cls: '' };
  }
}

// ── Records ───────────────────────────────────────────────────────────────────

export const RECORD_KIND_META: Record<string, { label: string; icon: string; tone: TagTone }> = {
  customer_file: { label: 'Customer file', icon: '🗂️', tone: 'blue' },
  contract:      { label: 'Contracts', icon: '📜', tone: 'indigo' },
  ticket:        { label: 'Tickets', icon: '🎫', tone: 'orange' },
  previous_call: { label: 'Previous calls', icon: '📞', tone: 'violet' },
  matter:        { label: 'Matter files', icon: '⚖️', tone: 'indigo' },
  related_case:  { label: 'Related cases', icon: '📚', tone: 'indigo' },
  account:       { label: 'Accounts', icon: '💳', tone: 'green' },
  product:       { label: 'Products', icon: '📦', tone: 'teal' },
  policy:        { label: 'Policies', icon: '📋', tone: 'teal' },
  kyc:           { label: 'KYC', icon: '🪪', tone: 'violet' },
  document:      { label: 'Documents', icon: '📄', tone: 'cyan' },
};

export function recordKindMeta(kind: RecordKind | string): { label: string; icon: string; tone: TagTone } {
  return RECORD_KIND_META[kind] ?? { label: roleLabel(kind), icon: '📄', tone: 'grey' };
}

// ── Misc ──────────────────────────────────────────────────────────────────────

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      return true;
    } catch {
      return false;
    }
  }
}

/** Format "Doc title · p.3 · Section" for evidence/record rows. */
export function sourceLine(doc_title?: string | null, page?: number | null, section_path?: string | null): string {
  const parts: string[] = [];
  if (doc_title) parts.push(doc_title);
  if (page != null) parts.push(`p.${page}`);
  if (section_path) parts.push(section_path);
  return parts.join(' · ');
}
