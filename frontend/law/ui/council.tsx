/**
 * The Council's interface (redesign 2026-11-07; the owner's reference image, adapted to what the product does).
 * - Who sits on a Council: a role mark (the role's icon) and its name, never a face.
 * - The discussion flow (Question, Initial views, Challenge, Responses, Synthesis) as the run itself records it
 *   (`review.progress` while it runs, the stored result after).
 * - Answers with citation chips that open the cited passage, re-read from the document on the server each time.
 * - The evidence each perspective used, and next steps a person writes into the matter's plan.
 * Nothing here scores, ranks or rates anything: every state is a word taken from the stored run.
 */
import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router';
import {
  ArrowRightLeft, BookOpen, Check, CircleDashed, FileSearch, Gavel, Handshake, Landmark, Minus, Scale, ScrollText, ShieldAlert, Users2, X,
  type LucideIcon,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import type { PanelContent, PanelPerspective, PanelReview, PanelRole, ReaderData, ResearchEvidence } from '../api/types';
import { Dialog } from './primitives';
import '../theme/council.css';

// ── Who sits on the Council ──────────────────────────────────────────────────────────────────────────────────────────
/** What each role looks at: the same words the run gives the model (panel/run.py LIBRARY), shortened. */
export const ROLE_META: Record<PanelRole, { icon: LucideIcon; about: string }> = {
  deal: { icon: Handshake, about: "How the documents affect the deal and the client's commercial aims" },
  risk: { icon: ShieldAlert, about: 'Exposure, obligations the client must meet, regulatory and compliance points' },
  litigation: { icon: Gavel, about: 'Disputes, remedies, notice, governing law and how a court would read the words' },
  specialist: { icon: BookOpen, about: 'The points particular to {area}' },
  lead: { icon: Scale, about: 'The strongest position for the client that the documents support' },
  opposing: { icon: ArrowRightLeft, about: 'The strongest reading against the client the same documents allow; it speaks for no real party' },
  evidence: { icon: FileSearch, about: 'Whether each factual point rests on the documents, and what contradicts it' },
  contract: { icon: ScrollText, about: 'What the clauses, amendments and versions require, with their conditions and limits' },
};
export const aboutRole = (role: PanelRole, area: string) => (ROLE_META[role]?.about ?? '').replace('{area}', area || 'the chosen area');

/** A role's mark: its icon in a walnut disc. Decorative; the role's name is always written beside it. */
export function RoleMark({ role, size = 'md' }: { role: PanelRole | 'chair'; size?: 'sm' | 'md' | 'lg' }) {
  const Icon = role === 'chair' ? Landmark : ROLE_META[role]?.icon ?? Users2;
  return <span className="cn-mark" data-size={size} data-role={role} aria-hidden><Icon /></span>;
}

// ── Reading an answer ────────────────────────────────────────────────────────────────────────────────────────────────
const MARKER = /\s*\[\d{1,2}\]/g;
export const isVerified = (e: ResearchEvidence) => e.verification === 'verbatim' || e.verification === 'normalized';

/** An answer without its uncited sentences and those whose quote failed: what the challengers and the chair read. */
export function verifiedText(p: Pick<PanelPerspective, 'text' | 'unsupported' | 'failed'>): string {
  let t = p.text ?? '';
  for (const s of [...(p.unsupported ?? []), ...(p.failed ?? [])]) if (s) t = t.split(s).join(' ');
  return t.replace(/[ \t]+/g, ' ').trim();
}

/** The first sentence of the verified answer, without citation markers: how a perspective opens, in its own words. */
export function openingLine(p: PanelPerspective, max = 190): string {
  const plain = verifiedText(p).replace(MARKER, '').replace(/\s+/g, ' ').trim();
  const first = plain.split(/(?<=[.!?])\s+(?=[A-Z“"(])/)[0] ?? '';
  if (first.length <= max) return first;
  const cut = first.slice(0, max);
  return `${cut.slice(0, Math.max(cut.lastIndexOf(' '), max - 30)).replace(/[,;:]$/, '')}…`;
}

export const verifiedCount = (p: Pick<PanelPerspective, 'evidence'>) => (p.evidence ?? []).filter(isVerified).length;
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Where a cited place is: document, clause, page. */
export function placeOf(e: ResearchEvidence, withTitle = true): string {
  const page = e.page_start ? (e.page_end && e.page_end !== e.page_start ? `pp. ${e.page_start}–${e.page_end}` : `p. ${e.page_start}`) : '';
  return [withTitle ? e.document_title : '', e.clause ? `clause ${e.clause}` : '', page].filter(Boolean).join(' · ');
}

export const VERIFICATION_WORDS: Record<ResearchEvidence['verification'], string> = {
  verbatim: 'The quoted words were found in this passage, word for word.',
  normalized: 'The quoted words were found in this passage, allowing for spacing and punctuation.',
  failed: 'The quoted words were not found in this passage.',
  not_quoted: 'The answer points here without quoting it.',
};

/** Text with its [n] markers as citation chips; each opens the cited passage. */
export function CitedText({ text, evidence, onOpen, who }: {
  text: string; evidence: ResearchEvidence[]; onOpen: (e: ResearchEvidence, from: HTMLElement) => void; who: string;
}) {
  const parts = text.split(/(\[\d{1,2}\])/);
  return (
    <>
      {parts.map((part, i) => {
        const m = /^\[(\d{1,2})\]$/.exec(part);
        if (!m) return <React.Fragment key={i}>{part}</React.Fragment>;
        const n = Number(m[1]);
        const e = evidence.find((x) => x.n === n && isVerified(x)) ?? evidence.find((x) => x.n === n);
        if (!e) return null;                                    // a marker with nothing behind it is not shown as a source
        return (
          <button key={i} type="button" className="cn-cite" data-verified={isVerified(e) ? 'true' : 'false'}
            aria-label={`${who}, source ${n}: ${placeOf(e)}. Open the passage`} onClick={(ev) => onOpen(e, ev.currentTarget)}>{n}</button>
        );
      })}
    </>
  );
}

// ── The cited passage ────────────────────────────────────────────────────────────────────────────────────────────────
export interface UsedBy { role: PanelRole; label: string; kind: 'view' | 'objection' | 'reply' }
const KIND_WORD: Record<UsedBy['kind'], string> = { view: 'initial view', objection: 'objection', reply: 'reply' };
export const readerHref = (number: string, e: ResearchEvidence) => (e.char_start != null && e.char_end != null
  ? `/m/${encodeURIComponent(number)}/documents/${encodeURIComponent(e.document_id)}/read?v=${encodeURIComponent(e.version_id)}&at=${e.char_start}-${e.char_end}`
  : `/m/${encodeURIComponent(number)}/documents/${encodeURIComponent(e.document_id)}/read?v=${encodeURIComponent(e.version_id)}`);

function around(text: string, a: number, b: number, reach = 360) {
  const s = Math.max(0, a - reach), t = Math.min(text.length, b + reach);
  const start = s > 0 ? text.indexOf(' ', s) + 1 || s : 0;
  const endSpace = text.lastIndexOf(' ', t);
  const end = t < text.length && endSpace > b ? endSpace : t;
  return { before: text.slice(start, a), hit: text.slice(a, b), after: text.slice(b, end), cutBefore: start > 0, cutAfter: end < text.length };
}

/** The passage a citation points at, read again from the document version on the server each time it is opened (access
 *  is checked there, and it is a recorded read), with the quoted words in gold; who used it; and the way into the full
 *  document at that place. */
export function PassageDialog({ matterId, number, e, usedBy, onClose }: {
  matterId: string; number: string; e: ResearchEvidence; usedBy: UsedBy[]; onClose: () => void;
}) {
  const reader = useQuery({
    queryKey: ['council', 'passage', matterId, e.document_id, e.version_id], retry: false, staleTime: 0, gcTime: 0, refetchOnWindowFocus: false,
    queryFn: async () => (await api.get<{ reader: ReaderData }>(`/api/legal/matters/${encodeURIComponent(matterId)}/documents/${
      encodeURIComponent(e.document_id)}/versions/${encodeURIComponent(e.version_id)}/reader`)).reader,
  });
  const text = reader.data?.text;
  const span = text != null && e.char_start != null && e.char_end != null && e.char_end <= text.length ? around(text, e.char_start, e.char_end) : null;
  const page = e.page_start ?? (reader.data && e.char_start != null ? reader.data.pages.find((p) => e.char_start! >= p.start && e.char_start! < p.end)?.number : undefined);
  const status = reader.error instanceof ApiError ? reader.error.status : 0;
  return (
    <Dialog title="Cited passage" onClose={onClose} width={640}>
      <div className="cn-passage-wrap">
        <p className="cn-verif" data-verified={isVerified(e) ? 'true' : 'false'}>
          {isVerified(e) ? <Check aria-hidden /> : <CircleDashed aria-hidden />}{VERIFICATION_WORDS[e.verification]}
        </p>
        {reader.isPending ? <p className="lw-meta" aria-busy="true">Opening the document…</p>
          : span ? (
            <blockquote className="cn-passage" aria-label="Passage">
              {span.cutBefore && '…'}{span.before}<mark>{span.hit}</mark>{span.after}{span.cutAfter && '…'}
            </blockquote>
          ) : (
            <>
              <blockquote className="cn-passage" aria-label="Quoted words"><mark>{e.quote}</mark></blockquote>
              <p className="lw-meta" role="note">{status === 403 || status === 404 ? 'The document is no longer open to you, so only the quoted words are shown.'
                : reader.isError ? 'The surrounding text could not be opened here; the quoted words are shown as they were checked.'
                  : 'No exact place is stored for this quote; the quoted words are shown as they were checked.'}</p>
            </>
          )}
        <dl className="cn-facts">
          <dt>Document</dt><dd>{e.document_title}</dd>
          <dt>Version</dt><dd>{`${e.version_number}${e.version_status ? ` · ${e.version_status === 'executed' ? 'signed' : e.version_status}` : ''}`}</dd>
          {e.clause && <><dt>Clause</dt><dd>{e.clause}</dd></>}
          {page != null && <><dt>Page</dt><dd>{page}</dd></>}
          {usedBy.length > 0 && <><dt>Cited by</dt><dd><ul className="cn-usedby">{usedBy.map((u, i) => (
            <li key={i}><RoleMark role={u.role} size="sm" />{`${u.label} (${KIND_WORD[u.kind]})`}</li>))}</ul></dd></>}
        </dl>
        <p style={{ margin: 0 }}><Link className="lw-btn lw-btn-sm" to={readerHref(number, e)} onClick={onClose}>Open at the quote</Link></p>
      </div>
    </Dialog>
  );
}

/** Opens a passage and gives focus back to the chip that opened it when the dialog closes. */
export function usePassage() {
  const [open, setOpen] = useState<ResearchEvidence | null>(null);
  const from = useRef<HTMLElement | null>(null);
  return {
    open,
    show: (e: ResearchEvidence, el?: HTMLElement | null) => { from.current = el ?? null; setOpen(e); },
    close: () => { setOpen(null); const el = from.current; if (el) window.setTimeout(() => el.isConnected && el.focus(), 0); },
  };
}

// ── Evidence in focus ────────────────────────────────────────────────────────────────────────────────────────────────
export interface CitedPlace { key: string; e: ResearchEvidence; by: UsedBy[] }
export interface CitedDocument { id: string; title: string; places: CitedPlace[]; roles: PanelRole[] }

const placeKey = (e: ResearchEvidence) => `${e.version_id}:${e.char_start ?? ''}-${e.char_end ?? ''}:${e.char_start == null ? e.quote : ''}`;

/** Every verified quote of the run, by document and place, with who cited it; quotes not found are counted, not listed. */
export function evidenceInFocus(c: PanelContent): { docs: CitedDocument[]; unverified: number; byPlace: Map<string, UsedBy[]> } {
  const docs = new Map<string, CitedDocument>();
  const byPlace = new Map<string, UsedBy[]>();
  let unverified = 0;
  const add = (p: PanelPerspective, kind: UsedBy['kind']) => {
    for (const e of p.evidence ?? []) {
      if (!isVerified(e)) { unverified += 1; continue; }
      const d = docs.get(e.document_id) ?? { id: e.document_id, title: e.document_title, places: [], roles: [] };
      docs.set(e.document_id, d);
      const key = placeKey(e);
      let place = d.places.find((x) => x.key === key);
      if (!place) { place = { key, e, by: [] }; d.places.push(place); }
      if (!place.by.some((u) => u.role === p.role && u.kind === kind)) place.by.push({ role: p.role, label: p.label, kind });
      if (!d.roles.includes(p.role)) d.roles.push(p.role);
      byPlace.set(key, place.by);
    }
  };
  c.perspectives.forEach((p) => add(p, 'view'));
  (c.challenges ?? []).filter((x) => x.objects).forEach((x) => add(x, 'objection'));
  if (c.revised) add(c.revised, 'reply');
  const list = [...docs.values()];
  for (const d of list) {
    // A quote inside another quote of the same version (the same sentence with or without its full stop) is one place.
    d.places.sort((a, b) => (a.e.char_start ?? 0) - (b.e.char_start ?? 0) || (b.e.char_end ?? 0) - (a.e.char_end ?? 0));
    const kept: CitedPlace[] = [];
    for (const pl of d.places) {
      const host = pl.e.char_start == null || pl.e.char_end == null ? undefined : kept.find((k) => k.e.version_id === pl.e.version_id
        && k.e.char_start != null && k.e.char_end != null && pl.e.char_start! >= k.e.char_start && pl.e.char_end! <= k.e.char_end);
      if (!host) { kept.push(pl); continue; }
      for (const u of pl.by) if (!host.by.some((h) => h.role === u.role && h.kind === u.kind)) host.by.push(u);
      byPlace.set(pl.key, host.by);
    }
    d.places = kept;
  }
  list.sort((a, b) => b.roles.length - a.roles.length || b.places.length - a.places.length || a.title.localeCompare(b.title));
  return { docs: list, unverified, byPlace };
}
export const usedByOf = (byPlace: Map<string, UsedBy[]>, e: ResearchEvidence) => byPlace.get(placeKey(e)) ?? [];

// ── The discussion flow ──────────────────────────────────────────────────────────────────────────────────────────────
export type StepKey = 'question' | 'views' | 'challenge' | 'reply' | 'chair';
export type StepState = 'done' | 'current' | 'next' | 'skipped' | 'stopped' | 'failed';
export interface FlowStep { key: StepKey; label: string; state: StepState; detail: string }
export const STEP_LABEL: Record<StepKey, string> = { question: 'Question', views: 'Initial views', challenge: 'Challenge', reply: 'Responses', chair: 'Synthesis' };
const RUN_STEPS = ['views', 'challenge', 'reply', 'chair'] as const;

/** The five steps as the run records them: its progress while it works, its stored result once it has finished. */
export function councilFlow(review: PanelReview, c?: PanelContent | null): FlowStep[] {
  const step = (key: StepKey, state: StepState, detail: string): FlowStep => ({ key, label: STEP_LABEL[key], state, detail });
  const roles = review.roles ?? [];
  const first = roles[0]?.label ?? c?.perspectives[0]?.label ?? 'first';
  const deep = (review.depth ?? c?.depth) === 'deep';
  const total = roles.length || c?.perspectives.length || 4;
  const asked = step('question', 'done', review.created_by ? `Asked by ${review.created_by}` : 'Asked');
  const focusedReply = step('reply', 'skipped', 'Not part of a focused run');
  const progress = review.progress;
  const working = review.status === 'queued' || review.status === 'running';

  if (working || review.status === 'failed' || review.status === 'not_allowed' || !c) {
    const at = working ? (review.status === 'queued' ? null : progress?.step ?? 'views') : progress?.step ?? null;
    const path = progress?.path ?? [];
    const idx = at && at !== 'finished' ? RUN_STEPS.indexOf(at) : -1;
    const ended = !working;
    return [asked, ...RUN_STEPS.map((k, i): FlowStep => {
      if (k === 'reply' && !deep) return focusedReply;
      if (idx < 0) {
        if (ended) return step(k, i === 0 ? 'failed' : 'next', i === 0 ? (review.status === 'not_allowed' ? 'No longer allowed' : 'Could not finish') : '');
        return step(k, 'next', i === 0 ? 'Waiting to start' : '');
      }
      if (i < idx) {
        if (path.length === 0 || path.includes(k)) return step(k, 'done', k === 'views' ? `${total} of ${total} answered` : 'Done');
        return step(k, 'skipped', k === 'challenge' ? 'Skipped: most perspectives said facts were missing'
          : path.includes('challenge') ? 'Not needed: no objection cited anything new' : 'Skipped with the challenge');
      }
      if (i > idx) return step(k, 'next', '');
      if (ended) return step(k, 'failed', review.status === 'not_allowed' ? 'No longer allowed' : 'Could not finish');
      const done = progress?.done ?? [];
      const detail = k === 'views' ? `${done.length} of ${total} answered`
        : k === 'challenge' ? (done.length ? `${plural(done.length, 'challenge')} in` : `Testing the ${first} answer`)
          : k === 'reply' ? `${first} is answering the objections` : 'Comparing the verified answers';
      return step(k, 'current', detail);
    })];
  }

  const views = c.perspectives;
  const answered = views.filter((v) => !v.abstained).length;
  const challenges = c.challenges ?? [];
  const objections = challenges.filter((x) => x.objects).length;
  const inconclusive = c.outcome === 'inconclusive';
  const stopped = c.outcome === 'stopped' || (c.stopped && !c.outcome);
  const legacy = c.challenges === undefined;
  const viewsStep = step('views', 'done', answered === views.length ? `${views.length} of ${views.length} answered`
    : answered === 0 ? 'No answer found in the documents' : `${answered} of ${views.length} answered`);
  const challengeStep = challenges.length
    ? step('challenge', 'done', objections ? plural(objections, 'objection') : 'No objection')
    : step('challenge', 'skipped', legacy ? 'Not part of this review' : inconclusive ? 'Skipped: fewer than two verified quotes'
      : 'Skipped: facts were missing');
  const replyStep = c.revised ? step('reply', 'done', `${c.revised.label} replied`)
    : !deep ? focusedReply
      : step('reply', 'skipped', objections ? 'Not needed: no objection cited anything new' : 'Not needed: nothing to answer');
  const chairStep = c.conclusion ? step('chair', 'done', 'Conclusion drawn')
    : c.stopped ? step('chair', 'stopped', 'Stopped: facts needed')
      : c.proceeded_without ? step('chair', 'done', 'Continued without some facts')
        : inconclusive ? step('chair', 'skipped', 'No conclusion drawn')
          : stopped ? step('chair', 'done', 'No conclusion drawn')
            : step('chair', 'done', c.matrix.length ? 'Comparison drawn' : 'Done');
  return [asked, viewsStep, challengeStep, replyStep, chairStep];
}

/** The flow as a stepper. Each step opens the part of the discussion it stands for. */
export function FlowStepper({ steps, onPick }: { steps: FlowStep[]; onPick: (key: StepKey) => void }) {
  const current = steps.find((s) => s.state === 'current' || s.state === 'stopped' || s.state === 'failed');
  const ICON: Record<StepState, React.ReactNode> = {
    done: <Check aria-hidden />, current: <span className="cn-step-pulse" aria-hidden />, next: null, skipped: <Minus aria-hidden />,
    stopped: <span aria-hidden>!</span>, failed: <X aria-hidden />,
  };
  const WORD: Record<StepState, string> = { done: 'done', current: 'in progress', next: 'not started', skipped: 'skipped', stopped: 'stopped', failed: 'did not finish' };
  return (
    <nav className="cn-flow" aria-label="Discussion flow">
      <ol>
        {steps.map((s, i) => (
          <li key={s.key} data-state={s.state} aria-current={current?.key === s.key ? 'step' : undefined}>
            <button type="button" className="cn-step" onClick={() => onPick(s.key)} aria-label={`${s.label}: ${WORD[s.state]}${s.detail ? `. ${s.detail}` : ''}`}>
              <span className="cn-step-dot">{ICON[s.state] ?? i + 1}</span>
              <span className="cn-step-text"><span className="cn-step-label">{s.label}</span>{s.detail && <span className="cn-step-detail">{s.detail}</span>}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

// ── Next steps for the matter's plan ─────────────────────────────────────────────────────────────────────────────────
export interface StepIdea { text: string; from: string }
const lowerFirst = (s: string) => (/^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);

/** Steps a person may add to the plan, taken by rule from what the run left open: the gaps the chair named, the facts the
 *  perspectives asked for, and figures no answer gave. A person reads and edits each before it goes into the plan. */
export function stepIdeas(c: PanelContent): StepIdea[] {
  const out: StepIdea[] = [];
  const seen = new Set<string>();
  const add = (text: string, from: string) => {
    const t = text.replace(/\s+/g, ' ').trim().replace(/[.;]$/, '').slice(0, 300);
    if (t && !seen.has(t.toLowerCase())) { seen.add(t.toLowerCase()); out.push({ text: t, from }); }
  };
  const confirm = (s: string) => (/^(whether|if|when|who|what|which|how|why|the|a|an)\b/i.test(s) ? `Confirm ${lowerFirst(s)}` : `Confirm: ${s}`);
  for (const g of c.gaps ?? []) add(confirm(g), 'Not established by the documents');
  for (const m of c.missing_facts ?? []) {
    const by = c.perspectives.filter((p) => p.missing_facts.some((x) => x.toLowerCase() === m.toLowerCase())).map((p) => p.label);
    add(confirm(m), by.length ? `Asked for by ${by.join(', ')}` : 'Asked for by the chair');
  }
  for (const f of c.figures_unsupported ?? []) add(`Check the figure ${f} against the documents`, "In the chair's view, but in no answer");
  return out.slice(0, 6);
}

// ── Small pieces ─────────────────────────────────────────────────────────────────────────────────────────────────────
/** Keyboard tabs (WAI-ARIA tabs pattern: arrows, Home and End move between them). */
export function Tabs<T extends string>({ label, tabs, value, onChange, idBase }: {
  label: string; tabs: { id: T; label: string; count?: number }[]; value: T; onChange: (id: T) => void; idBase: string;
}) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const move = (e: React.KeyboardEvent, i: number) => {
    const n = tabs.length;
    const to = e.key === 'ArrowRight' ? (i + 1) % n : e.key === 'ArrowLeft' ? (i - 1 + n) % n : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1;
    if (to < 0) return;
    e.preventDefault();
    onChange(tabs[to].id);
    refs.current[tabs[to].id]?.focus();
  };
  return (
    <div className="cn-tabs" role="tablist" aria-label={label}>
      {tabs.map((t, i) => (
        <button key={t.id} ref={(el) => { refs.current[t.id] = el; }} type="button" role="tab" id={`${idBase}-tab-${t.id}`} aria-selected={value === t.id}
          aria-controls={`${idBase}-panel-${t.id}`} tabIndex={value === t.id ? 0 : -1} className="cn-tab" onClick={() => onChange(t.id)} onKeyDown={(e) => move(e, i)}>
          {t.label}{t.count != null && t.count > 0 && <span className="cn-tab-count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

export function useStableId(prefix: string) {
  const id = useId();
  return useMemo(() => `${prefix}${id.replace(/[^a-zA-Z0-9]/g, '')}`, [id, prefix]);
}

export function useScrollTo(target: string | null, done: () => void) {
  useEffect(() => {
    if (!target) return;
    const el = document.getElementById(target);
    if (el) {
      el.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
      el.focus?.({ preventScroll: true });
    }
    done();
  }, [target, done]);
}
