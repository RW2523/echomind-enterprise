/**
 * Cards that work on the matter's records (docs/echomind-legal/03 registry; owner's request 2026-11-02): a checklist of
 * open obligations, decisions waiting for this person, a short form before something is made, a draft written from the
 * chat, the review panel's run, amounts as a chart, the parties and the documents naming them, and a meeting being
 * recorded. Every string is shown as text. A change goes through POST /counsel/actions, where the server checks the
 * person's right, the card's origin and the record's version; a card drawn from changing records refreshes itself from
 * GET /counsel/components/{id}, never from anything the model wrote.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, BarChart3, Building2, CalendarClock, Check, CheckCircle2, ClipboardCheck, FilePen, FileSearch, FileText, Gauge, Info, ListChecks,
  MessageSquarePlus, Mic, Network, Scale, Users,
} from 'lucide-react';
import { counselApi, requestId } from './api';
import { parseComponent } from './validate';
import type {
  ApprovalItem, ChecklistItem, GraphEdge, GuidedQuestion, SafeComponent,
} from './types';
import type { CardContext } from './cards';

type Of<T extends SafeComponent['type']> = Extract<SafeComponent, { type: T }>;

const DAY = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
const WHEN = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
};
const GAP = /(\[\[NEEDS INPUT:[^\]]*\]\])/;

function Icon({ children }: { children: React.ReactNode }) {
  return <span className="qc-card-icon" aria-hidden>{children}</span>;
}

function Caution({ text }: { text: string }) {
  return <p className="qc-note" data-tone="info" role="note"><Info size={15} aria-hidden />{text}</p>;
}

/** Ask the server to rebuild a card every `ms` while `on`, for at most `forMs` (then the reader may follow on). */
function usePolling(on: boolean, ms: number, forMs: number, ctx: CardContext, id: string): { following: boolean; follow: () => void } {
  const [until, setUntil] = useState(() => Date.now() + forMs);
  const following = on && Date.now() < until;
  useEffect(() => {
    if (!on || !ctx.refresh) return;
    const t = window.setInterval(() => {
      if (Date.now() > until) { window.clearInterval(t); setUntil(0); return; }
      void ctx.refresh?.(id);
    }, ms);
    return () => window.clearInterval(t);
  }, [on, ms, until, id, ctx.refresh]);
  return { following, follow: () => setUntil(Date.now() + forMs) };
}

async function act(comp: SafeComponent, ctx: CardContext, actionId: 'obligation' | 'decide', version: number, parameters: Record<string, unknown>) {
  const r = await counselApi.action({ component_id: comp.id, action_id: actionId, matter_id: comp.matter_id ?? ctx.matterId,
                                      expected_version: version, parameters, idempotency_key: requestId() });
  if (r.component) ctx.replace(parseComponent(r.component));
  return r;
}

// ── checklist ──────────────────────────────────────────────────────────────────
function CheckRow({ it, comp, ctx, canDone, canConfirm }: { it: ChecklistItem; comp: Of<'checklist'>; ctx: CardContext; canDone: boolean; canConfirm: boolean }) {
  const [saving, setSaving] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const done = it.status === 'done';
  const id = `chk-${comp.id}-${it.id}`;
  const change = async (action: 'done' | 'reopen' | 'confirm') => {
    setSaving(true); setMessage(null);
    const r = await act(comp, ctx, 'obligation', it.version, { obligation_id: it.id, action });
    setSaving(false); setConfirming(false);
    if (!r.ok) setMessage(r.status === 403 ? 'Your role in this matter does not include this change.' : r.message);
  };
  const meta = [it.party, it.due_date ? `due ${DAY(it.due_date)}` : it.when, it.owner ? `owner ${it.owner}` : ''].filter(Boolean).join(' · ');
  return (
    <li className="qc-task" data-status={it.status} data-overdue={it.overdue || undefined}>
      <input id={id} type="checkbox" checked={done} disabled={!canDone || saving || it.status === 'dismissed'}
        onChange={() => void change(done ? 'reopen' : 'done')} aria-describedby={`${id}-meta`} />
      <div className="qc-task-body">
        <label htmlFor={id}>{it.title}</label>
        {meta && <span id={`${id}-meta`} className="qc-meta">{meta}</span>}
        <div className="qc-tags">
          {it.overdue && <span className="qc-tag" data-tag="Overdue">Overdue</span>}
          {it.status === 'proposed' && it.due_date && <span className="qc-tag" data-tag="Uncertain">Date not confirmed</span>}
          {it.status === 'confirmed' && <span className="qc-tag" data-tag="Confirmed by a lawyer">Confirmed by a lawyer</span>}
          {it.status === 'dismissed' && <span className="qc-tag">Dismissed</span>}
        </div>
        <span className="qc-meta">{it.document_title}{it.clause ? ` · cl. ${it.clause}` : ''} · <button type="button" className="qc-linkbtn"
          onClick={() => ctx.go(it.path)} aria-label={`Open the words of ${it.document_title} for: ${it.title}`}>Open the words</button></span>
        {it.status === 'proposed' && it.due_date && canConfirm && (confirming ? (
          <div className="qc-confirm" role="group" aria-label="Confirm the date">
            <span>Confirm {DAY(it.due_date)} against the document's words? It then counts as confirmed by you.</span>
            <button type="button" className="qc-btn qc-btn-primary" disabled={saving} onClick={() => void change('confirm')}>Confirm the date</button>
            <button type="button" className="qc-btn" onClick={() => setConfirming(false)}>Cancel</button>
          </div>
        ) : <button type="button" className="qc-btn qc-btn-sm" onClick={() => setConfirming(true)}>Confirm the date…</button>)}
        {message && <p className="qc-error" role="alert">{message}</p>}
      </div>
    </li>
  );
}

function Checklist({ comp, ctx }: { comp: Of<'checklist'>; ctx: CardContext }) {
  const p = comp.props;
  return (
    <section className="qc-card" data-component="checklist" aria-label={p.summary}>
      <h3 className="qc-card-title"><Icon><ListChecks aria-hidden /></Icon>{p.summary}</h3>
      {p.items.length === 0
        ? <p className="qc-state" data-tone="verified"><CheckCircle2 aria-hidden />Nothing open in the register.</p>
        : <ul className="qc-tasks">{p.items.map((it) => (
          <CheckRow key={it.id} it={it} comp={comp} ctx={ctx} canDone={p.can_done} canConfirm={p.can_confirm} />))}</ul>}
      <p className="qc-meta" style={{ margin: 0 }}>
        {p.more > 0 ? `${p.more} more in the ` : 'All of them in the '}
        <button type="button" className="qc-linkbtn" onClick={() => ctx.go(p.register_path)}>obligations register</button>.
        {!p.can_done && ' You can read these; ticking them is for the matter\'s lawyers and paralegals.'}
      </p>
      <Caution text={p.caution} />
    </section>
  );
}

// ── approval_request ───────────────────────────────────────────────────────────
const STATE_TEXT: Record<ApprovalItem['state'], string | null> = {
  waiting: null, approved: 'Approved', changes_requested: 'Sent back for changes', changed: 'Changed since it was shown: open it to review the current version',
  gone: 'No longer available',
};

function ApprovalRow({ it, comp, ctx }: { it: ApprovalItem; comp: Of<'approval_request'>; ctx: CardContext }) {
  const [mode, setMode] = useState<'idle' | 'approve' | 'changes'>('idle');
  const [readIt, setReadIt] = useState(false);
  const [reason, setReason] = useState('');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'changes' && reason.trim().length < 3) { setMessage('Say what should change.'); return; }
    setSaving(true); setMessage(null);
    const r = await act(comp, ctx, 'decide', it.version, { artifact_id: it.artifact_id, decision: mode === 'approve' ? 'approved' : 'changes_requested',
                                                          ...(mode === 'changes' ? { reason: reason.trim() } : {}), read_it: readIt });
    setSaving(false);
    if (r.ok) { setMode('idle'); return; }
    setMessage(r.status === 403 ? 'Deciding this is for another lawyer of the matter.' : r.message);
  };
  const state = STATE_TEXT[it.state];
  return (
    <li className="qc-approval" data-state={it.state}>
      <div><strong>{it.kind}: {it.title}</strong>
        <span className="qc-meta" style={{ display: 'block' }}>Version {it.version} · by {it.author}{it.created_at ? ` · ${WHEN(it.created_at)}` : ''}</span></div>
      {it.preview.length > 0 && <blockquote className="qc-preview" aria-label="How it begins">{it.preview.map((t, i) => <p key={i}><Gapped text={t} /></p>)}</blockquote>}
      {it.gaps > 0 && <p className="qc-note" role="note"><AlertTriangle size={15} aria-hidden />{it.gaps} gap{it.gaps === 1 ? '' : 's'} to fill before it can be approved.</p>}
      {state && <p className="qc-state" data-tone={it.state === 'approved' ? 'verified' : 'neutral'}>{state}{it.decided_by ? ` by ${it.decided_by}` : ''}</p>}
      <div className="qc-actions">
        <button type="button" className="qc-btn" onClick={() => ctx.go(it.path)}><FileText aria-hidden />Open to read</button>
        {it.can_decide && mode === 'idle' && <>
          <button type="button" className="qc-btn" disabled={it.gaps > 0} onClick={() => { setMode('approve'); setReadIt(false); setMessage(null); }}>Approve…</button>
          <button type="button" className="qc-btn" onClick={() => { setMode('changes'); setReason(''); setMessage(null); }}>Ask for changes…</button>
        </>}
      </div>
      {mode !== 'idle' && (
        <form className="qc-review-form" onSubmit={(e) => void submit(e)} aria-label={mode === 'approve' ? `Approve ${it.title}` : `Ask for changes to ${it.title}`}>
          {mode === 'approve' ? <>
            <p style={{ margin: 0 }}>{it.consequence}</p>
            <label className="qc-readit"><input type="checkbox" checked={readIt} onChange={(e) => setReadIt(e.target.checked)} />
              I have read version {it.version} in full.</label>
          </> : <label style={{ display: 'grid', gap: 6, fontWeight: 600, fontSize: 14 }}>What should change?
            <textarea value={reason} maxLength={500} onChange={(e) => setReason(e.target.value)} autoFocus /></label>}
          <div className="qc-actions">
            <button type="submit" className="qc-btn qc-btn-primary" disabled={saving || (mode === 'approve' && !readIt)}>
              {saving ? 'Saving…' : mode === 'approve' ? `Approve version ${it.version}` : 'Send back'}</button>
            <button type="button" className="qc-btn" onClick={() => setMode('idle')}>Cancel</button>
          </div>
        </form>
      )}
      {message && <p className="qc-error" role="alert">{message}</p>}
    </li>
  );
}

function ApprovalRequest({ comp, ctx }: { comp: Of<'approval_request'>; ctx: CardContext }) {
  const p = comp.props;
  return (
    <section className="qc-card" data-component="approval_request" aria-label={p.summary}>
      <h3 className="qc-card-title"><Icon><ClipboardCheck aria-hidden /></Icon>{p.summary}</h3>
      {p.items.length > 0 && <ol className="qc-approvals">{p.items.map((it) => <ApprovalRow key={it.artifact_id} it={it} comp={comp} ctx={ctx} />)}</ol>}
      {p.more > 0 && <p className="qc-meta" style={{ margin: 0 }}>and {p.more} more: <button type="button" className="qc-linkbtn" onClick={() => ctx.go('/drafts')}>open Drafts</button></p>}
      <Caution text={p.caution} />
    </section>
  );
}

// ── guided_questions ───────────────────────────────────────────────────────────
function Question({ q, value, onChange, idBase }: { q: GuidedQuestion; value: string; onChange: (v: string) => void; idBase: string }) {
  const id = `${idBase}-${q.id}`;
  if (q.kind === 'choice' && q.options.length <= 5) {
    return (
      <fieldset className="qc-q-field">
        <legend>{q.label}</legend>
        {q.options.map((o) => (
          <label key={o.value} className="qc-option">
            <input type="radio" name={id} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} />
            <span>{o.label}{o.detail && <span className="qc-meta"> · {o.detail}</span>}</span>
          </label>
        ))}
      </fieldset>
    );
  }
  if (q.kind === 'choice') {
    return (
      <label className="qc-q-field" htmlFor={id}><span>{q.label}</span>
        <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
          {q.options.map((o) => <option key={o.value} value={o.value}>{o.label}{o.detail ? ` (${o.detail})` : ''}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="qc-q-field" htmlFor={id}><span>{q.label}{q.required ? '' : ' (optional)'}</span>
      {q.max_length > 200
        ? <textarea id={id} value={value} maxLength={q.max_length} rows={3} onChange={(e) => onChange(e.target.value)} />
        : <input id={id} value={value} maxLength={q.max_length} onChange={(e) => onChange(e.target.value)} />}
    </label>
  );
}

function GuidedQuestions({ comp, ctx }: { comp: Of<'guided_questions'>; ctx: CardContext }) {
  const p = comp.props;
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(p.questions.map((q) => [q.id, q.default ?? ''])));
  const [sent, setSent] = useState(false);
  const missing = p.questions.find((q) => q.required && !values[q.id]?.trim());
  const label = p.purpose === 'draft'
    ? `Write the draft “${(values.title ?? '').trim() || 'Draft'}”`
    : `Ask the review panel: ${(values.question ?? '').trim()}`;
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (missing || ctx.busy || !ctx.submitForm) return;
    setSent(true);
    ctx.submitForm(comp, values, label.slice(0, 300));
  };
  return (
    <section className="qc-card" data-component="guided_questions" aria-label={p.title}>
      <h3 className="qc-card-title"><Icon>{p.purpose === 'draft' ? <FilePen aria-hidden /> : <Users aria-hidden />}</Icon>{p.title}</h3>
      <p className="qc-meta" style={{ margin: 0 }}>{p.intro}</p>
      <form className="qc-guided" onSubmit={submit} aria-label={p.title}>
        {p.questions.map((q) => <Question key={q.id} q={q} value={values[q.id] ?? ''} idBase={comp.id}
          onChange={(v) => { setSent(false); setValues((x) => ({ ...x, [q.id]: v })); }} />)}
        <div className="qc-actions">
          <button type="submit" className="qc-btn qc-btn-primary" disabled={!!missing || ctx.busy || sent}>{sent ? 'Sent' : p.submit_label}</button>
        </div>
        {missing && <p className="qc-meta" style={{ margin: 0 }}>“{missing.label}” is needed.</p>}
      </form>
      <Caution text={p.note} />
    </section>
  );
}

// ── document_preview ───────────────────────────────────────────────────────────
function Gapped({ text }: { text: string }) {
  return <>{text.split(GAP).map((piece, i) => (GAP.test(piece)
    ? <mark key={i} className="qc-gap">{piece.replace(/^\[\[NEEDS INPUT:\s*/, '').replace(/\]\]$/, '')}</mark> : <React.Fragment key={i}>{piece}</React.Fragment>))}</>;
}

const DRAFT_STATUS: Record<string, string> = { draft: 'Draft', approved: 'Approved', superseded: 'Superseded', filed: 'Filed' };

function DocumentPreview({ comp, ctx }: { comp: Of<'document_preview'>; ctx: CardContext }) {
  const p = comp.props;
  const matterId = comp.matter_id ?? ctx.matterId;
  return (
    <section className="qc-card" data-component="document_preview" aria-label={p.title}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <h3 className="qc-card-title"><Icon><FilePen aria-hidden /></Icon>{p.title}</h3>
        <span className="qc-status-chip" data-status={p.status === 'approved' || p.status === 'filed' ? 'explained' : 'potential'}>
          {DRAFT_STATUS[p.status]} · version {p.version}</span>
      </div>
      <p className="qc-meta" style={{ margin: 0 }}>From the template {p.template} · {p.sources} cited source{p.sources === 1 ? '' : 's'}
        {p.gaps ? ` · ${p.gaps} gap${p.gaps === 1 ? '' : 's'} to fill` : ''}</p>
      <div className="qc-draft qc-draft-preview">
        {p.blocks.map((b, i) => b.type === 'heading'
          ? <h4 key={i} className="qc-draft-h">{b.text}</h4>
          : <p key={i} data-origin={b.origin}><Gapped text={b.text} /></p>)}
        {p.more_blocks > 0 && <p className="qc-meta">…and {p.more_blocks} more paragraph{p.more_blocks === 1 ? '' : 's'}.</p>}
      </div>
      {p.fields.some((f) => f.status === 'needs_input') && (
        <p className="qc-note" role="note"><AlertTriangle size={15} aria-hidden />Needs input: {p.fields.filter((f) => f.status === 'needs_input').map((f) => f.label).join(', ')}.</p>
      )}
      <p className="qc-meta" style={{ margin: 0 }}>{p.review}</p>
      <div className="qc-actions">
        <button type="button" className="qc-btn qc-btn-primary"
          onClick={(e) => ctx.openDraft?.(matterId, p.artifact_id, p.title, p.can_edit, e.currentTarget, () => { void ctx.refresh?.(comp.id); })}>
          <FileSearch aria-hidden />{p.can_edit ? 'Open beside the chat to correct' : 'Open beside the chat'}</button>
        <button type="button" className="qc-btn" onClick={() => ctx.go(p.path)}><FileText aria-hidden />Open in Drafting</button>
      </div>
    </section>
  );
}

// ── council_summary ────────────────────────────────────────────────────────────
const PANEL_STATE: Record<string, string> = {
  queued: 'Waiting to start…', running: 'The perspectives are reading the documents and challenging the first answer…', done: 'Done', failed: 'The panel could not finish.',
  needs_facts: 'The panel needs facts the documents do not give.', not_allowed: 'The panel did not run for this matter.',
};

function CouncilSummary({ comp, ctx }: { comp: Of<'council_summary'>; ctx: CardContext }) {
  const p = comp.props;
  const working = p.status === 'queued' || p.status === 'running';
  const { following, follow } = usePolling(working, 3000, 5 * 60_000, ctx, comp.id);
  return (
    <section className="qc-card" data-component="council_summary" aria-label={`Review panel: ${p.question}`}>
      <h3 className="qc-card-title"><Icon><Scale aria-hidden /></Icon>Review panel · {p.area}</h3>
      <p style={{ margin: 0 }}>“{p.question}”</p>
      <p className="qc-state" data-tone={p.status === 'done' ? 'verified' : p.status === 'failed' ? 'attention' : 'neutral'} role="status">
        {PANEL_STATE[p.status]}{working && !following && <> <button type="button" className="qc-linkbtn" onClick={follow}>Check again</button></>}</p>
      {p.outcome === 'inconclusive' && <p style={{ margin: 0 }}>No conclusion: fewer than two of the answers' quotes could be verified in the documents.</p>}
      {p.conclusion && (
        <div className="qc-conclusion" aria-label="The chair's conclusion">
          <p className="qc-meta" style={{ margin: 0 }}>The chair's conclusion</p>
          <p style={{ margin: '2px 0 0' }}><b>{p.conclusion}</b></p>
          {p.figures_unsupported.length > 0 && <p className="qc-meta" style={{ margin: '4px 0 0' }}>{`Check: ${p.figures_unsupported.join(', ')} is in no answer.`}</p>}
        </div>
      )}
      {p.points.length > 0 && <ul className="qc-history-list" aria-label="What matters most">{p.points.map((x, i) => <li key={i}>{x}</li>)}</ul>}
      {p.gaps.length > 0 && <p className="qc-meta" style={{ margin: 0 }}>{`Not established by the documents: ${p.gaps.join('; ')}.`}</p>}
      {p.issues.length > 0 && (
        <div className="qc-table-wrap">
          <table className="qc-table qc-panel-table">
            <thead><tr><th scope="col">Issue</th><th scope="col">Positions</th></tr></thead>
            <tbody>{p.issues.map((i, k) => (
              <tr key={k}>
                <th scope="row">{i.issue}{i.disagreement && <span className="qc-tag" data-tag="Uncertain" style={{ marginLeft: 6 }}>They differ</span>}</th>
                <td><ul className="qc-positions">{i.positions.map((x, j) => <li key={j}><b>{x.role}:</b> {x.text}</li>)}</ul></td>
              </tr>))}
            </tbody>
          </table>
        </div>
      )}
      {p.missing_facts.length > 0 && (
        <div><p className="qc-meta" style={{ margin: 0 }}>Facts the panel asks for:</p><ul className="qc-history-list">{p.missing_facts.map((m, i) => <li key={i}>{m}</li>)}</ul></div>
      )}
      {p.approval && <p className="qc-meta" style={{ margin: 0 }}>{p.approval}</p>}
      <div className="qc-actions"><button type="button" className="qc-btn" onClick={() => ctx.go(p.path)}><FileText aria-hidden />Open the full review</button></div>
      <Caution text={p.caution} />
    </section>
  );
}

// ── numeric_chart ──────────────────────────────────────────────────────────────
function NumericChart({ comp, ctx }: { comp: Of<'numeric_chart'>; ctx: CardContext }) {
  const p = comp.props;
  const [doc, setDoc] = useState<string | null>(null);
  const [largest, setLargest] = useState(false);
  const docs = useMemo(() => [...new Set(p.bars.map((b) => b.document_title))], [p.bars]);
  const bars = p.bars.filter((b) => !doc || b.document_title === doc);
  const shown = largest ? [...bars].sort((a, b) => b.value - a.value) : bars;
  const top = Math.max(1, ...p.bars.map((b) => b.value));
  return (
    <section className="qc-card" data-component="numeric_chart" aria-label={p.summary}>
      <h3 className="qc-card-title"><Icon><BarChart3 aria-hidden /></Icon>{p.summary}</h3>
      {p.bars.length === 0 ? null : (
        <>
          <div className="qc-filters">
            {docs.length > 1 && (
              <div className="qc-chips" role="group" aria-label="Document">
                <button type="button" className="qc-chip" aria-pressed={doc === null} onClick={() => setDoc(null)}>All documents</button>
                {docs.map((d) => <button key={d} type="button" className="qc-chip" aria-pressed={doc === d} onClick={() => setDoc(doc === d ? null : d)}>{d}</button>)}
              </div>
            )}
            <div className="qc-chips" role="group" aria-label="Order">
              <button type="button" className="qc-chip" aria-pressed={!largest} onClick={() => setLargest(false)}>As in the documents</button>
              <button type="button" className="qc-chip" aria-pressed={largest} onClick={() => setLargest(true)}>Largest first</button>
            </div>
          </div>
          <ol className="qc-hbars" aria-label={`Amounts in ${p.unit}`}>
            {shown.map((b) => (
              <li key={b.id}>
                <button type="button" className="qc-hbar" onClick={(e) => ctx.openSide(b.source, `${b.shown}: ${b.document_title}`, e.currentTarget)}
                  aria-label={`${b.shown}, ${b.document_title}${b.clause ? `, clause ${b.clause}` : ''}: open its sentence`}>
                  <span className="qc-hbar-label">{b.shown}</span>
                  <span className="qc-hbar-track" aria-hidden><span className="qc-hbar-fill" style={{ width: `${Math.max(2, Math.round((100 * b.value) / top))}%` }} /></span>
                  <span className="qc-hbar-where">{b.document_title}{b.clause ? ` · cl. ${b.clause}` : ''}</span>
                </button>
              </li>
            ))}
          </ol>
          {p.more > 0 && <p className="qc-meta" style={{ margin: 0 }}>{p.more} more amount{p.more === 1 ? '' : 's'} in {p.unit} not shown: ask about one document.</p>}
          {p.other_units.length > 0 && <p className="qc-meta" style={{ margin: 0 }}>Amounts in {p.other_units.join(', ')} are also stated; they are not shown here, and never converted.</p>}
        </>
      )}
      <Caution text={p.caution} />
    </section>
  );
}

// ── graph_excerpt ──────────────────────────────────────────────────────────────
const ROW_H = 46;
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function GraphExcerpt({ comp, ctx }: { comp: Of<'graph_excerpt'>; ctx: CardContext }) {
  const p = comp.props;
  const partiesN = p.nodes.filter((n) => n.kind === 'party');
  const docsN = p.nodes.filter((n) => n.kind === 'document');
  const [chosen, setChosen] = useState<string | null>(null);
  const rows = Math.max(partiesN.length, docsN.length, 1);
  const H = rows * ROW_H + 8;
  const y = (i: number, n: number) => 4 + (i + 0.5) * (H - 8) / Math.max(n, 1);
  const pos = new Map<string, number>([...partiesN.map((n, i) => [n.id, y(i, partiesN.length)] as const),
                                       ...docsN.map((n, i) => [n.id, y(i, docsN.length)] as const)]);
  const label = new Map(p.nodes.map((n) => [n.id, n.label]));
  const edgesOf = (id: string) => p.edges.filter((e) => e.source === id);
  const chosenEdges: GraphEdge[] = chosen ? edgesOf(chosen) : [];
  return (
    <section className="qc-card" data-component="graph_excerpt" aria-label={p.summary}>
      <h3 className="qc-card-title"><Icon><Network aria-hidden /></Icon>{p.summary}</h3>
      {partiesN.length > 0 && (
        <>
          <svg className="qc-graph" viewBox={`0 0 640 ${H}`} role="img" aria-label="Parties on the left, the documents that name them on the right">
            {p.edges.map((e) => {
              const a = pos.get(e.source) ?? 0, b = pos.get(e.target) ?? 0;
              const on = !chosen || e.source === chosen;
              return <path key={e.id} d={`M 232 ${a} C 320 ${a}, 320 ${b}, 408 ${b}`} className="qc-graph-edge"
                data-on={on || undefined} data-kind={e.label === 'party to' ? 'party' : 'named'} />;
            })}
            {partiesN.map((n) => (
              <g key={n.id} transform={`translate(4 ${(pos.get(n.id) ?? 0) - 17})`} className="qc-graph-node" data-on={!chosen || chosen === n.id || undefined}
                role="button" tabIndex={0} aria-pressed={chosen === n.id} aria-label={`${n.label}${n.role ? `, ${n.role}` : ''}: named in ${edgesOf(n.id).length} document${edgesOf(n.id).length === 1 ? '' : 's'}`}
                onClick={() => setChosen(chosen === n.id ? null : n.id)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setChosen(chosen === n.id ? null : n.id); } }}>
                <rect width="228" height="34" rx="8" />
                <text x="12" y="21">{clip(n.label, n.role ? 22 : 30)}</text>
                {n.role && <text x="218" y="21" textAnchor="end" className="qc-graph-role">{n.role}</text>}
                <title>{n.label}</title>
              </g>
            ))}
            {docsN.map((n) => (
              <g key={n.id} transform={`translate(408 ${(pos.get(n.id) ?? 0) - 17})`} className="qc-graph-doc" aria-hidden>
                <rect width="228" height="34" rx="8" />
                <text x="12" y="21">{clip(n.label, 30)}</text>
                <title>{n.label}</title>
              </g>
            ))}
          </svg>
          <p className="qc-meta" style={{ margin: 0 }}>Choose a party to see where each document names it. Solid lines: named as a party; dashed: mentioned.</p>
          {chosen && (
            <ul className="qc-graph-list" aria-label={`Where ${label.get(chosen)} is named`}>
              {chosenEdges.map((e) => (
                <li key={e.id}><b>{e.label === 'party to' ? 'Party to' : 'Named in'}</b> {label.get(e.target)}{e.evidence.clause ? `, cl. ${e.evidence.clause}` : ''} ·{' '}
                  <button type="button" className="qc-linkbtn" onClick={(ev) => ctx.openSide(e.evidence, `${label.get(chosen)} in ${label.get(e.target)}`, ev.currentTarget)}>Inspect</button></li>
              ))}
            </ul>
          )}
          <details className="qc-disclosure qc-graph-text">
            <summary>As a list</summary>
            <ul>{partiesN.map((n) => <li key={n.id}><b>{n.label}</b>{n.role ? ` (${n.role})` : ''}: {edgesOf(n.id).map((e) => `${e.label} ${label.get(e.target)}`).join('; ')}</li>)}</ul>
          </details>
        </>
      )}
      {p.more > 0 && <p className="qc-meta" style={{ margin: 0 }}>{p.more} more compan{p.more === 1 ? 'y' : 'ies'} named less often are not shown.</p>}
      <Caution text={p.caution} />
    </section>
  );
}

// ── live_transcript ────────────────────────────────────────────────────────────
function LiveTranscript({ comp, ctx }: { comp: Of<'live_transcript'>; ctx: CardContext }) {
  const p = comp.props;
  const live = p.state !== 'not_recording';
  const { following, follow } = usePolling(live, 4000, 10 * 60_000, ctx, comp.id);
  const [seen, setSeen] = useState(() => new Date());
  const last = useRef(p.segments.length);
  useEffect(() => { if (p.segments.length !== last.current) { last.current = p.segments.length; setSeen(new Date()); } }, [p.segments.length]);
  return (
    <section className="qc-card" data-component="live_transcript" aria-label={`Meeting: ${p.title}`}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <h3 className="qc-card-title"><Icon><Mic aria-hidden /></Icon>{p.title}</h3>
        <span className="qc-status" data-state={p.state === 'recording' ? 'recording' : undefined} role="status">
          <span className="qc-dot" aria-hidden />{p.state === 'recording' ? 'Recording' : p.state === 'paused' ? 'Paused' : 'Not recording'}</span>
      </div>
      <p className="qc-meta" style={{ margin: 0 }}>{p.consent}</p>
      {p.segments.length === 0 ? <p className="qc-meta" style={{ margin: 0 }}>Nothing has been transcribed yet.</p> : (
        <ol className="qc-transcript" aria-label="Latest transcript">
          {p.earlier > 0 && <li className="qc-meta">{p.earlier} earlier segment{p.earlier === 1 ? '' : 's'} on the meeting page</li>}
          {p.segments.map((g, i) => (
            <li key={i}>{g.speaker && <b>{g.speaker}: </b>}{g.text}{g.at && <span className="qc-meta"> · {WHEN(g.at)}</span>}</li>
          ))}
        </ol>
      )}
      {live && <p className="qc-meta" style={{ margin: 0 }}>{following ? `Following live · updated ${seen.toLocaleTimeString()}` : 'Stopped following. '}
        {!following && <button type="button" className="qc-linkbtn" onClick={follow}>Keep following</button>}</p>}
      <div className="qc-actions"><button type="button" className="qc-btn" onClick={() => ctx.go(p.path)}><Building2 aria-hidden />Open the meeting</button></div>
      <Caution text={p.caution} />
    </section>
  );
}

// ── case_status: where the matter stands (its stage, its people's latest updates, its next tasks) ──
const TASK_KIND: Record<string, string> = { key_date: 'Key date', hearing: 'Hearing', filing: 'Filing deadline' };
function CaseStatus({ comp, ctx }: { comp: Of<'case_status'>; ctx: CardContext }) {
  const p = comp.props;
  return (
    <section className="qc-card" data-component="case_status" aria-label={`Where ${p.number} stands`}>
      <h3 className="qc-card-title"><Icon><Gauge aria-hidden /></Icon>{p.number} · {p.title}</h3>
      <ol className="qc-stages" aria-label={`${p.track_label} stages`}>
        {p.stages.map((s, i) => (
          <li key={s} data-state={i < p.stage_index ? 'done' : i === p.stage_index ? 'current' : 'next'} aria-current={i === p.stage_index ? 'step' : undefined}>
            <span aria-hidden>{i < p.stage_index ? <Check size={12} /> : i + 1}</span>{s}
          </li>
        ))}
      </ol>
      <p className="qc-meta" style={{ margin: 0 }}>{p.track_label} track · {p.stages[p.stage_index]}{p.days_in_stage != null ? ` for ${p.days_in_stage} day${p.days_in_stage === 1 ? '' : 's'}` : ''}</p>
      <div className="qc-case-cols">
        <div>
          <h4 className="qc-case-head">Latest updates</h4>
          {p.updates.length === 0 ? <p className="qc-meta" style={{ margin: 0 }}>No updates have been posted yet.</p> : (
            <ol className="qc-case-feed">
              {p.updates.map((u, i) => (
                <li key={i}><span className="qc-meta"><b>{u.kind_label}</b> · {DAY(u.occurred_on)} · {u.author}</span>
                  <span>{u.text}</span>{u.document_title && <span className="qc-meta">Document: {u.document_title}</span>}</li>
              ))}
            </ol>
          )}
        </div>
        <div>
          <h4 className="qc-case-head">Next in the plan{p.open_tasks ? ` (${p.open_tasks} open${p.overdue ? `, ${p.overdue} overdue` : ''})` : ''}</h4>
          {p.tasks.length === 0 ? <p className="qc-meta" style={{ margin: 0 }}>Nothing is planned.</p> : (
            <ul className="qc-case-tasks">
              {p.tasks.map((t, i) => (
                <li key={i} data-overdue={t.overdue ? 'true' : undefined}>
                  <span>{t.title}</span>
                  <span className="qc-meta">{TASK_KIND[t.kind] ? `${TASK_KIND[t.kind]} · ` : ''}{t.due_on ? <><CalendarClock size={12} aria-hidden /> {t.overdue ? 'Overdue · ' : ''}{DAY(t.due_on)}</> : 'No date'}{t.assignee ? ` · ${t.assignee}` : ''}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="qc-actions">
        <button type="button" className="qc-btn" onClick={() => ctx.go(p.path)}>{p.can_post ? <><MessageSquarePlus aria-hidden />Post an update</> : <><Building2 aria-hidden />Open the case</>}</button>
      </div>
      <p className="qc-meta" style={{ margin: 0 }}>From the matter's updates and plan, as its people wrote them · {WHEN(p.as_of)}</p>
    </section>
  );
}

/** The cards of this file, or null for a type cards.tsx shows itself. */
export function WorkCard({ comp, ctx }: { comp: SafeComponent; ctx: CardContext }): React.ReactElement | null {
  switch (comp.type) {
    case 'checklist': return <Checklist comp={comp} ctx={ctx} />;
    case 'approval_request': return <ApprovalRequest comp={comp} ctx={ctx} />;
    case 'guided_questions': return <GuidedQuestions comp={comp} ctx={ctx} />;
    case 'document_preview': return <DocumentPreview comp={comp} ctx={ctx} />;
    case 'council_summary': return <CouncilSummary comp={comp} ctx={ctx} />;
    case 'numeric_chart': return <NumericChart comp={comp} ctx={ctx} />;
    case 'graph_excerpt': return <GraphExcerpt comp={comp} ctx={ctx} />;
    case 'live_transcript': return <LiveTranscript comp={comp} ctx={ctx} />;
    case 'case_status': return <CaseStatus comp={comp} ctx={ctx} />;
    default: return null;
  }
}
