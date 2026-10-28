/**
 * The trusted component registry (docs/echomind-legal/03): each validated component type maps to exactly one card
 * below; anything else is the inert placeholder. Every string from the server is rendered as text (React escapes it);
 * nothing is parsed as HTML or markdown, and only the workspace's own pages can be opened.
 */
import React, { useMemo, useState } from 'react';
import {
  AlertTriangle, ArrowDown, ArrowUp, BadgeCheck, CalendarDays, CheckCircle2, CircleDashed, Columns3, FileSearch, FileX, Info, ListChecks, Lock,
  Scale,
} from 'lucide-react';
import { counselApi, requestId } from './api';
import { isWorkspacePath, parseComponent } from './validate';
import type {
  CaseRow, CaseTableProps, CitationRef, ComparisonSide, DiscrepancyStatus, EvidenceComparisonProps, NoticeProps, PriorityListProps,
  SafeComponent, SourceAnswerProps, TimelineEvent, TimelineProps, UiAction,
} from './types';

export interface CardContext {
  matterId: string;
  matterNumber?: (matterId: string) => string | undefined;   // in a conversation about several matters
  busy: boolean;
  ask: (text: string, opts?: { document_ids?: string[]; focus?: string }) => void;
  openCitation: (c: CitationRef, opener: HTMLElement | null) => void;
  openSide: (side: ComparisonSide, label: string, opener: HTMLElement | null) => void;
  go: (path: string) => void;
  replace: (component: SafeComponent) => void;
}

/** Run a suggested or item action. Only the four contract action types exist; an unknown one does nothing. */
export function runAction(a: UiAction, ctx: CardContext, opener: HTMLElement | null) {
  switch (a.action_type) {
    case 'ask': if (a.prompt) ctx.ask(a.prompt, { document_ids: a.document_ids }); break;
    case 'open_discrepancy': if (a.target_id) ctx.ask('Show this potential inconsistency', { focus: a.target_id }); break;
    case 'open_page': if (isWorkspacePath(a.path)) ctx.go(a.path); break;
    case 'open_source': break;          // sources open from their citation chips, which carry the opaque id
  }
  void opener;
}

export function ActionButtons({ actions, ctx }: { actions: UiAction[]; ctx: CardContext }) {
  if (!actions.length) return null;
  return (
    <div className="qc-actions">
      {actions.map((a) => (
        <button key={a.id} type="button" className="qc-btn" disabled={ctx.busy && a.action_type !== 'open_page'}
          onClick={(e) => runAction(a, ctx, e.currentTarget)}>{a.label}</button>
      ))}
    </div>
  );
}

// ── Answer text with inline citation chips ─────────────────────────────────────
const MARK = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;

/** Display numbers: sources numbered 1, 2, 3 in the order the answer first cites them (research numbers its passages). */
export function displayNumbers(text: string, citations: CitationRef[]): Map<number, number> {
  const known = new Set(citations.map((c) => c.n));
  const order = new Map<number, number>();
  for (const m of text.matchAll(MARK)) {
    for (const x of (m[1] ?? '').split(',')) {
      const n = Number(x.trim());
      if (known.has(n) && !order.has(n)) order.set(n, order.size + 1);
    }
  }
  for (const c of [...citations].sort((a, b) => a.n - b.n)) if (!order.has(c.n)) order.set(c.n, order.size + 1);
  return order;
}

export function CitationChip({ n, c, ctx, pending, label }: { n: number; c?: CitationRef; ctx?: CardContext; pending?: boolean; label?: number }) {
  const shown = label ?? n;
  if (!c || pending || !ctx) {
    return <span className="qc-cite" data-pending="true" aria-label={`Source ${shown}, not checked yet`}>{shown}</span>;
  }
  if (c.removed) {
    return <button type="button" className="qc-cite" disabled aria-label={`Source ${shown}: the document was deleted`}>{shown}</button>;
  }
  const where = [c.document_title, c.page ? `page ${c.page}` : '', c.clause ? `clause ${c.clause}` : ''].filter(Boolean).join(', ');
  return (
    <button type="button" className="qc-cite" data-verified={c.verified ? 'true' : 'false'} title={where}
      aria-label={`Source ${shown}: ${where}. ${c.verified ? 'Verified quote' : 'Not verified'}`}
      onClick={(e) => ctx.openCitation(c, e.currentTarget)}>
      {shown}{c.verified ? <BadgeCheck aria-hidden /> : null}
    </button>
  );
}

/** Paragraphs of plain text; "[n]" markers become chips only for sources this answer actually has. */
export function AnswerText({ text, citations, ctx, pending, labels }: {
  text: string; citations: CitationRef[]; ctx?: CardContext; pending?: boolean; labels?: Map<number, number>;
}) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  const paragraphs = text.replace(/\*\*(.+?)\*\*/g, '$1').split(/\n{2,}/).filter((p) => p.trim());
  return (
    <div className="qc-answer">
      {paragraphs.map((p, i) => {
        const parts: React.ReactNode[] = [];
        let last = 0;
        let prev: number | null = null;                   // "[1][1]" (two quotes from one source) shows one chip
        for (const m of p.matchAll(MARK)) {
          // A finished answer shows only markers that have a source; one without (a number the model made up) is dropped,
          // never left as text that looks like a citation (review C1). While it streams, every marker waits for its chip.
          const ns = m[1].split(',').map((x) => Number(x.trim())).filter((n) => pending || byN.has(n));
          const between = p.slice(last, m.index);
          if (between.trim()) prev = null;
          parts.push(between);
          ns.forEach((n) => {
            if (n !== prev) parts.push(<CitationChip key={`${m.index}-${n}`} n={n} c={byN.get(n)} ctx={ctx} pending={pending} label={labels?.get(n)} />);
            prev = n;
          });
          last = (m.index ?? 0) + m[0].length;
        }
        parts.push(p.slice(last));
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}

// ── source_answer ──────────────────────────────────────────────────────────────
function AnswerState({ p }: { p: SourceAnswerProps }) {
  const verified = p.citations.filter((c) => c.verified && !c.removed).length;
  // "Supported" needs at least one verified quote (review C1); an answer with none is said to be unverified.
  if (p.state === 'answered' && verified === 0) return <span className="qc-state" data-tone="attention"><CircleDashed aria-hidden />Not verified: no quote could be matched to the documents</span>;
  if (p.state === 'abstained') return <span className="qc-state" data-tone="neutral"><FileX aria-hidden />No source found in this matter's documents</span>;
  if (p.state === 'unverified') return <span className="qc-state" data-tone="attention"><CircleDashed aria-hidden />Not verified: no quote could be matched to the documents</span>;
  if (p.state === 'partial') return <span className="qc-state" data-tone="attention"><AlertTriangle aria-hidden />Partly supported: some sentences have no verified source</span>;
  return <span className="qc-state" data-tone="verified"><BadgeCheck aria-hidden />Supported by {verified} verified quote{verified === 1 ? '' : 's'}</span>;
}

function SourceAnswer({ p, ctx }: { p: SourceAnswerProps; ctx: CardContext }) {
  const labels = displayNumbers(p.answer, p.citations);
  const unique = Array.from(new Map(p.citations.map((c) => [c.n, c])).values())
    .sort((a, b) => (labels.get(a.n) ?? a.n) - (labels.get(b.n) ?? b.n));
  return (
    <div className="qc-card qc-answer-card" data-component="source_answer">
      <AnswerState p={p} />
      {p.state === 'abstained'
        ? <p className="qc-answer" style={{ margin: 0 }}>The documents in this matter do not answer that. Try naming the document, clause or party.</p>
        : <AnswerText text={p.answer} citations={p.citations} ctx={ctx} labels={labels} />}
      {unique.length > 0 && (
        <div className="qc-sources" role="group" aria-label="Sources">
          {unique.map((c) => (
            <button key={c.id} type="button" className="qc-source" data-verified={c.verified ? 'true' : 'false'} disabled={c.removed}
              onClick={(e) => ctx.openCitation(c, e.currentTarget)}
              aria-label={c.removed ? `Source ${labels.get(c.n) ?? c.n}: the document was deleted`
                : `Open source ${labels.get(c.n) ?? c.n}: ${c.document_title}${c.page ? `, page ${c.page}` : ''}${c.matter ? `, ${c.matter}` : ''}`}>
              {c.verified ? <BadgeCheck aria-hidden /> : <CircleDashed aria-hidden />}
              <b>{labels.get(c.n) ?? c.n}</b>
              {c.matter && <MatterTag number={c.matter} />}
              <span>{c.removed ? 'Deleted document' : `${c.document_title}${c.page ? ` · p. ${c.page}` : ''}${c.clause ? ` · cl. ${c.clause}` : ''}`}</span>
            </button>
          ))}
        </div>
      )}
      {p.notes.map((n, i) => <p key={i} className="qc-note" role="note"><AlertTriangle size={15} aria-hidden />{n}</p>)}
      {p.unsupported.length > 0 && (
        <details className="qc-disclosure">
          <summary>Sentences without a verified source ({p.unsupported.length})</summary>
          <ul>{p.unsupported.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function CardIcon({ children }: { children: React.ReactNode }) {
  return <span className="qc-card-icon" aria-hidden>{children}</span>;
}

/** The matter something comes from, in a conversation about several. */
function MatterTag({ number }: { number: string }) {
  return <span className="qc-mtag" title={`From ${number}`}>{number}</span>;
}

// ── priority_list ──────────────────────────────────────────────────────────────
function PriorityList({ p, ctx }: { p: PriorityListProps; ctx: CardContext }) {
  return (
    <section className="qc-card" data-component="priority_list" aria-label={p.summary}>
      <h3 className="qc-card-title"><CardIcon><ListChecks aria-hidden /></CardIcon>{p.summary}</h3>
      {p.items.length === 0 ? (
        <p className="qc-state" data-tone="verified"><CheckCircle2 aria-hidden />{p.empty || 'Nothing needs your attention.'}</p>
      ) : (
        <ol className="qc-items">
          {p.items.map((it, i) => (
            <li key={it.id} className="qc-item" data-kind={it.kind}>
              <span className="qc-item-n" aria-hidden>{i + 1}</span>
              <div className="qc-item-body">
                <strong>{it.matter && <><MatterTag number={it.matter} />{' '}</>}{it.title}</strong>
                {it.detail && <span className="qc-meta">{it.detail}</span>}
              </div>
              <button type="button" className="qc-btn" disabled={ctx.busy && it.action.action_type !== 'open_page'}
                onClick={(e) => runAction(it.action, ctx, e.currentTarget)}>{it.action.label}</button>
            </li>
          ))}
        </ol>
      )}
      {p.more > 0 && (p.more_action
        ? <ActionButtons actions={[{ ...p.more_action, label: `${p.more_action.label} (${p.more})` }]} ctx={ctx} />
        : <p className="qc-meta" style={{ margin: 0 }}>and {p.more} more</p>)}
    </section>
  );
}

// ── evidence_comparison ────────────────────────────────────────────────────────
const STATUS_LABEL: Record<DiscrepancyStatus, string> = {
  potential: 'Potential', in_review: 'In review', explained: 'Explained', confirmed: 'Confirmed', dismissed: 'Dismissed',
};
const RESOLUTIONS: { to: Exclude<DiscrepancyStatus, 'potential'>; label: string; needsNote: boolean }[] = [
  { to: 'in_review', label: 'Mark in review', needsNote: false },
  { to: 'explained', label: 'Explained', needsNote: true },
  { to: 'confirmed', label: 'Confirm inconsistency', needsNote: true },
  { to: 'dismissed', label: 'Dismiss', needsNote: true },
];

function Side({ s, label, ctx }: { s: ComparisonSide; label: 'A' | 'B'; ctx: CardContext }) {
  const a = Math.min(s.value_start, s.context.length);
  const b = Math.min(Math.max(s.value_end, a), s.context.length);
  const where = [s.version_number ? `version ${s.version_number}${s.version_status === 'executed' ? ' (signed)' : ''}` : '',
    s.page ? `p. ${s.page}` : '', s.clause ? `cl. ${s.clause}` : ''].filter(Boolean).join(' · ');
  return (
    <div className="qc-side">
      <div className="qc-side-head">
        <span><span className="qc-side-label">{label}</span> <b>{s.document_title}</b></span>
      </div>
      {where && <span className="qc-meta">{where}</span>}
      <p className="qc-quote">…{s.context.slice(0, a)}<mark>{s.context.slice(a, b)}</mark>{s.context.slice(b)}…</p>
      {s.stale && <span className="qc-note" role="note"><AlertTriangle size={14} aria-hidden />A newer version of this document is now in use.</span>}
      <div className="qc-actions">
        <button type="button" className="qc-btn" onClick={(e) => ctx.openSide(s, `${label}: ${s.document_title}`, e.currentTarget)}>
          <FileSearch aria-hidden />Inspect
        </button>
      </div>
    </div>
  );
}

function EvidenceComparison({ comp, ctx }: { comp: Extract<SafeComponent, { type: 'evidence_comparison' }>; ctx: CardContext }) {
  const p: EvidenceComparisonProps = comp.props;
  const [choice, setChoice] = useState<(typeof RESOLUTIONS)[number] | null>(null);
  const [note, setNote] = useState('');
  const [key, setKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const start = (r: (typeof RESOLUTIONS)[number]) => { setChoice(r); setNote(''); setKey(requestId()); setMessage(null); };
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!choice || (choice.needsNote && note.trim().length < 3)) { setMessage('Say briefly why, so the next reader knows how this was resolved.'); return; }
    setSaving(true);
    const r = await counselApi.action({ component_id: comp.id, action_id: 'review', matter_id: comp.matter_id ?? ctx.matterId, expected_version: comp.version,
      parameters: { resolution: choice.to, ...(note.trim() ? { note: note.trim() } : {}) }, idempotency_key: key });
    setSaving(false);
    if (r.ok) {
      setChoice(null);
      ctx.replace(parseComponent(r.component));
      return;
    }
    if (r.status === 409 && r.component) {
      ctx.replace(parseComponent(r.component));
      setChoice(null);
    }
    setMessage(r.status === 403 ? 'Resolving this is for the matter\'s lawyers and reviewers.' : r.message);
  };

  return (
    <section className="qc-card" data-component="evidence_comparison" aria-label={p.title}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <h3 className="qc-card-title"><CardIcon><Scale aria-hidden /></CardIcon>{p.title}
          {comp.matter_id && ctx.matterNumber?.(comp.matter_id) && <MatterTag number={ctx.matterNumber(comp.matter_id)!} />}</h3>
        <span className="qc-status-chip" data-status={p.status}>{STATUS_LABEL[p.status]}</span>
      </div>
      <div className="qc-compare">
        <Side s={p.a} label="A" ctx={ctx} />
        <Side s={p.b} label="B" ctx={ctx} />
      </div>
      {p.explanation && <p style={{ margin: 0 }}><span className="qc-meta">Why it was raised: </span>{p.explanation}</p>}
      <p className="qc-note" data-tone="info" role="note"><Info size={15} aria-hidden />{p.caution}</p>
      <p className="qc-meta" style={{ margin: 0 }}>
        {p.method === 'rule' ? 'Found by a rule that reads both documents.' : 'Proposed by the model after reading both passages.'} Nothing is decided until a person reviews it.
      </p>
      {p.history.length > 0 && (
        <ul className="qc-history-list" aria-label="Review history">
          {p.history.map((h, i) => (
            <li key={i}><b>{STATUS_LABEL[h.to_status]}</b> by {h.by}, {new Date(h.at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}
              {h.note ? <>: “{h.note}”</> : null}</li>
          ))}
        </ul>
      )}
      {p.can_review ? (
        choice ? (
          <form className="qc-review-form" onSubmit={submit}>
            <label htmlFor={`note-${comp.id}`} style={{ fontWeight: 600, fontSize: 14 }}>
              {choice.label}{choice.needsNote ? ': why?' : ' (a note is optional)'}
            </label>
            <textarea id={`note-${comp.id}`} value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} autoFocus
              placeholder={choice.to === 'explained' ? 'For example: dispatch on one date, arrival on the other, per the shipment log.' : ''} />
            <div className="qc-actions">
              <button type="submit" className="qc-btn qc-btn-primary" disabled={saving}>{saving ? 'Saving…' : 'Save'}</button>
              <button type="button" className="qc-btn" onClick={() => { setChoice(null); setMessage(null); }}>Cancel</button>
            </div>
          </form>
        ) : (
          <div className="qc-actions" role="group" aria-label="Resolve">
            {RESOLUTIONS.filter((r) => r.to !== p.status && !(r.to === 'in_review' && p.status !== 'potential')).map((r) => (
              <button key={r.to} type="button" className="qc-btn" onClick={() => start(r)}>{r.label}</button>
            ))}
          </div>
        )
      ) : <p className="qc-meta" style={{ margin: 0 }}>You can read this. Resolving it is for the matter's lawyers and reviewers.</p>}
      {message && <p className="qc-error" role="alert">{message}</p>}
    </section>
  );
}

// ── timeline ───────────────────────────────────────────────────────────────────
const FIRST_EVENTS = 8;
const KIND_LABEL: Record<TimelineEvent['kind'], string | null> = {
  event: null, deadline: 'Deadline', document_date: 'Document date', obligation: 'Deadline (register)',
};

function EventRow({ e, ctx }: { e: TimelineEvent; ctx: CardContext }) {
  const s = e.source;
  const where = [s.document_title, s.clause ? `cl. ${s.clause}` : '', s.page ? `p. ${s.page}` : ''].filter(Boolean).join(' · ');
  const tags = [KIND_LABEL[e.kind], e.reported ? 'Recollection' : null, e.certainty === 'uncertain' ? 'Uncertain' : null,
    e.superseded ? 'Replaced' : null, e.status ?? null].filter((t): t is string => !!t);
  return (
    <li className="qc-event" data-certainty={e.certainty} data-kind={e.kind} data-reported={e.reported ? 'true' : undefined}
      data-superseded={e.superseded ? 'true' : undefined}>
      <div className="qc-event-date">
        {e.date ? <time dateTime={e.date}>{e.date_text}</time> : <span>“{e.date_text}”</span>}
      </div>
      <span className="qc-event-rail" aria-hidden><span className="qc-event-dot" /></span>
      <div className="qc-event-body">
        {(tags.length > 0 || e.matter) && <div className="qc-tags">{e.matter && <MatterTag number={e.matter} />}
          {tags.map((t) => <span key={t} className="qc-tag" data-tag={t}>{t}</span>)}</div>}
        <p className="qc-quote" style={{ fontSize: 14.5 }}>{e.text}</p>
        <span className="qc-meta">{where} · <button type="button" className="qc-linkbtn"
          onClick={(ev) => ctx.openSide(s, `${e.date_text}: ${s.document_title}`, ev.currentTarget)}
          aria-label={`Inspect ${s.document_title}, ${e.date_text}`}>Inspect</button></span>
        {e.notes.map((n, i) => <p key={i} className="qc-note" role="note" style={{ fontSize: 13 }}><Info size={14} aria-hidden />{n}</p>)}
      </div>
    </li>
  );
}

/** Dated events under their year, as a list a screen reader reads in order. */
function DatedEvents({ events, ctx }: { events: TimelineEvent[]; ctx: CardContext }) {
  const rows: React.ReactNode[] = [];
  let year: string | undefined;
  for (const e of events) {
    const y = e.date?.slice(0, 4);
    if (y !== year) rows.push(<li key={`y-${y}`} className="qc-year" aria-hidden>{y}</li>);
    year = y;
    rows.push(<EventRow key={e.id} e={e} ctx={ctx} />);
  }
  return <ol className="qc-timeline" aria-label="Dated events">{rows}</ol>;
}

/** What a reader can narrow the chronology to. Filtering only hides events: nothing is recomputed or inferred. */
const EVENT_FILTERS: { key: string; label: string; test: (e: TimelineEvent) => boolean }[] = [
  { key: 'all', label: 'All', test: () => true },
  { key: 'deadline', label: 'Deadlines', test: (e) => e.kind === 'deadline' || e.kind === 'obligation' },
  { key: 'document_date', label: 'Document dates', test: (e) => e.kind === 'document_date' },
  { key: 'recollection', label: 'Recollections', test: (e) => e.reported },
  { key: 'uncertain', label: 'Uncertain', test: (e) => e.certainty === 'uncertain' },
];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const monthLabel = (ym: string) => `${MONTHS[Number(ym.slice(5, 7)) - 1]} ${ym.slice(0, 4)}`;

/** Dated events per month, as bars; choosing a bar shows that month only. */
function MonthChart({ events, month, onMonth }: { events: TimelineEvent[]; month: string | null; onMonth: (m: string | null) => void }) {
  const counts = new Map<string, number>();
  for (const e of events) if (e.date) counts.set(e.date.slice(0, 7), (counts.get(e.date.slice(0, 7)) ?? 0) + 1);
  const months = [...counts.keys()].sort();
  if (months.length < 2 || months.length > 36) return null;
  const top = Math.max(...counts.values());
  return (
    <div className="qc-chart" role="group" aria-label="Events by month">
      {months.map((m) => (
        <button key={m} type="button" className="qc-bar" aria-pressed={month === m} onClick={() => onMonth(month === m ? null : m)}
          aria-label={`${monthLabel(m)}: ${counts.get(m)} event${counts.get(m) === 1 ? '' : 's'}`}>
          <span className="qc-bar-fill" style={{ height: `${Math.max(10, Math.round((100 * counts.get(m)!) / top))}%` }} />
          <span className="qc-bar-n" aria-hidden>{counts.get(m)}</span>
          <span className="qc-bar-label" aria-hidden>{monthLabel(m)}</span>
        </button>
      ))}
    </div>
  );
}

function Timeline({ p, ctx }: { p: TimelineProps; ctx: CardContext }) {
  const [all, setAll] = useState(false);
  const [kind, setKind] = useState('all');
  const [matter, setMatter] = useState<string | null>(null);
  const [month, setMonth] = useState<string | null>(null);
  const matters = useMemo(() => [...new Set(p.events.map((e) => e.matter).filter((m): m is string => !!m))], [p.events]);
  const test = EVENT_FILTERS.find((f) => f.key === kind)!.test;
  const narrowed = kind !== 'all' || matter !== null || month !== null;
  const keep = (e: TimelineEvent) => test(e) && (!matter || e.matter === matter) && (!month || !!e.date?.startsWith(month));
  const dated = p.events.filter((e) => e.certainty !== 'undated' && keep(e));
  const undated = month ? [] : p.events.filter((e) => e.certainty === 'undated' && keep(e));
  const open = all || narrowed;                           // a narrowed view shows every match
  const shownDated = open ? dated : dated.slice(0, FIRST_EVENTS);
  const hidden = dated.length - shownDated.length + (open ? 0 : undated.length);
  const allDated = p.events.filter((e) => e.certainty !== 'undated' && test(e) && (!matter || e.matter === matter));
  return (
    <section className="qc-card" data-component="timeline" aria-label={p.summary}>
      <h3 className="qc-card-title"><CardIcon><CalendarDays aria-hidden /></CardIcon>{p.summary}</h3>
      {p.events.length === 0 ? <p className="qc-meta" style={{ margin: 0 }}>No dates found in the documents you can read.</p> : (
        <>
          {p.events.length >= 4 && (
            <div className="qc-filters">
              <div className="qc-chips" role="group" aria-label="Show">
                {EVENT_FILTERS.map((f) => {
                  const n = p.events.filter((e) => f.test(e) && (!matter || e.matter === matter)).length;
                  return (f.key === 'all' || n > 0) && (
                    <button key={f.key} type="button" className="qc-chip" aria-pressed={kind === f.key} onClick={() => setKind(f.key)}>
                      {f.label} <span className="qc-chip-n">{n}</span>
                    </button>);
                })}
              </div>
              {matters.length > 1 && (
                <div className="qc-chips" role="group" aria-label="Matter">
                  <button type="button" className="qc-chip" aria-pressed={matter === null} onClick={() => setMatter(null)}>All matters</button>
                  {matters.map((m) => (
                    <button key={m} type="button" className="qc-chip" aria-pressed={matter === m} onClick={() => setMatter(matter === m ? null : m)}>{m}</button>
                  ))}
                </div>
              )}
              <MonthChart events={allDated} month={month} onMonth={setMonth} />
              {narrowed && (
                <p className="qc-meta" style={{ margin: 0 }} aria-live="polite">
                  Showing {dated.length + undated.length} of {p.events.length} events{month ? ` in ${monthLabel(month)}` : ''}.{' '}
                  <button type="button" className="qc-linkbtn" onClick={() => { setKind('all'); setMatter(null); setMonth(null); }}>Show everything</button>
                </p>
              )}
            </div>
          )}
          {shownDated.length > 0 && <DatedEvents events={shownDated} ctx={ctx} />}
          {open && undated.length > 0 && (
            <>
              <h4 className="qc-card-title" style={{ fontSize: 15 }}>Date not stated</h4>
              <ol className="qc-timeline" aria-label="Events whose date is not stated">{undated.map((e) => <EventRow key={e.id} e={e} ctx={ctx} />)}</ol>
            </>
          )}
          {!shownDated.length && !(open && undated.length) && <p className="qc-meta" style={{ margin: 0 }}>No event matches.</p>}
          {hidden > 0 && (
            <button type="button" className="qc-btn" onClick={() => setAll(true)}>
              Show all ({hidden} more{undated.length && !open ? `, ${undated.length} without a stated date` : ''})
            </button>
          )}
          {p.more > 0 && <p className="qc-meta" style={{ margin: 0 }}>{p.more} more not shown: ask about a narrower period or document.</p>}
        </>
      )}
      <p className="qc-note" data-tone="info" role="note"><Info size={15} aria-hidden />{p.caution}</p>
    </section>
  );
}

// ── case_table: several matters side by side ───────────────────────────────────
type SortKey = 'number' | 'client' | 'documents' | 'open' | 'deadline';
const COLUMNS: { key: SortKey; label: string; value: (r: CaseRow) => string | number }[] = [
  { key: 'number', label: 'Matter', value: (r) => r.number },
  { key: 'client', label: 'Client', value: (r) => r.client.toLowerCase() },
  { key: 'documents', label: 'Documents', value: (r) => r.documents },
  { key: 'open', label: 'Open inconsistencies', value: (r) => r.open_inconsistencies ?? -1 },
  { key: 'deadline', label: 'Next deadline', value: (r) => r.next_deadline ?? '9999' },
];
const DAY = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });

function CaseTable({ p, ctx }: { p: CaseTableProps; ctx: CardContext }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'number', dir: 1 });
  const col = COLUMNS.find((c) => c.key === sort.key)!;
  const rows = [...p.rows].sort((a, b) => {
    const x = col.value(a), y = col.value(b);
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
  });
  return (
    <section className="qc-card" data-component="case_table" aria-label={p.summary}>
      <h3 className="qc-card-title"><CardIcon><Columns3 aria-hidden /></CardIcon>{p.summary}</h3>
      <div className="qc-table-wrap">
        <table className="qc-table">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.key} scope="col" aria-sort={sort.key === c.key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
                  <button type="button" onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? (s.dir === 1 ? -1 : 1) : 1 }))}>
                    {c.label}{sort.key === c.key && (sort.dir === 1 ? <ArrowUp aria-hidden /> : <ArrowDown aria-hidden />)}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.matter_id}>
                <th scope="row">
                  <button type="button" className="qc-linkbtn" onClick={() => ctx.go(r.path)} aria-label={`Open ${r.number}, ${r.title}`}>{r.number}</button>
                  {r.sensitivity !== 'standard' && <Lock className="qc-row-lock" aria-label={r.sensitivity === 'highly_restricted' ? 'Highly restricted' : 'Restricted'} />}
                  <span className="qc-meta" style={{ display: 'block' }}>{r.title}</span>
                  {r.status !== 'open' && <span className="qc-tag">{r.status === 'closed' ? 'Closed' : 'Archived'}</span>}
                </th>
                <td>{r.client}</td>
                <td className="qc-num">{r.documents}</td>
                <td className="qc-num">{r.open_inconsistencies === undefined ? '—'
                  : r.open_inconsistencies ? <span className="qc-tag" data-tag="Uncertain">{r.open_inconsistencies}</span> : <span>0</span>}</td>
                <td>{r.next_deadline ? <>
                  <time dateTime={r.next_deadline} style={{ fontWeight: 600 }}>{DAY(r.next_deadline)}</time>
                  {r.deadline_state && <span className="qc-tag" data-tag={r.deadline_state === 'Confirmed by a lawyer' ? r.deadline_state : 'Uncertain'}
                    style={{ marginLeft: 6 }}>{r.deadline_state}</span>}
                  {r.next_deadline_text && <span className="qc-meta" style={{ display: 'block' }}>{r.next_deadline_text}</span>}
                </> : <span className="qc-meta">None recorded</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="qc-note" data-tone="info" role="note"><Info size={15} aria-hidden />{p.caution}</p>
    </section>
  );
}

// ── notice, unavailable, inert ─────────────────────────────────────────────────
function Notice({ p }: { p: NoticeProps }) {
  return <p className="qc-note" data-tone={p.tone === 'caution' ? undefined : 'info'} data-component="notice" style={{ fontSize: 15 }}>
    {p.tone === 'caution' ? <AlertTriangle size={16} aria-hidden /> : <Info size={16} aria-hidden />}{p.text}
  </p>;
}

export function ComponentView({ comp, ctx }: { comp: SafeComponent; ctx: CardContext }) {
  switch (comp.type) {
    case 'source_answer': return <SourceAnswer p={comp.props} ctx={ctx} />;
    case 'priority_list': return <PriorityList p={comp.props} ctx={ctx} />;
    case 'evidence_comparison': return <EvidenceComparison comp={comp} ctx={ctx} />;
    case 'timeline': return <Timeline p={comp.props} ctx={ctx} />;
    case 'case_table': return <CaseTable p={comp.props} ctx={ctx} />;
    case 'notice': return <Notice p={comp.props} />;
    case 'unavailable': return <p className="qc-inert" data-component="unavailable">{comp.props.text}</p>;
    case 'task_progress': return <p className="qc-meta"><ListChecks size={14} aria-hidden /> {comp.props.label}</p>;
    default: return <p className="qc-inert" data-component="inert">This part of the answer could not be shown.</p>;
  }
}
