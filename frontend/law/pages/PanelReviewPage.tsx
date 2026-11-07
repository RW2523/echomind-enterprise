import React, { useCallback, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router';
import { ArrowLeft, ArrowRight, Check, CircleDashed, FileText, ListPlus, MessagesSquare, Scale, Send } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, keys, panelPath, usePanelReview, useTasks } from '../api/queries';
import type { ArtifactVersion, Matter, PanelChallenge, PanelContent, PanelPerspective, PanelReview, PanelRole, ResearchEvidence } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';
import {
  aboutRole, CitedText, councilFlow, evidenceInFocus, FlowStepper, isVerified, openingLine, PassageDialog, placeOf, plural, readerHref, RoleMark,
  stepIdeas, Tabs, usedByOf, usePassage, useScrollTo, useStableId, verifiedCount, type FlowStep, type StepKey,
} from '../ui/council';
import { PANEL_STATUS } from './MatterPanel';

type View = 'overview' | 'discussion' | 'evidence' | 'result';
const VIEWS: View[] = ['overview', 'discussion', 'evidence', 'result'];
const NOTICE = 'Perspectives drafted by the same AI model to test the question from different angles. They are not independent experts and '
  + 'not legal advice; a partner reviews them before anyone relies on them.';
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** One Council session (S4.5 review panel, grown into the Council 2026-11-05, redesigned 2026-11-07): the question, the
 *  discussion flow as the run records it, and four views — the perspectives around the question, the discussion with its
 *  citations, the evidence they used, and the result with what is still open, steps for the plan and the partner's review. */
export default function PanelReviewPage() {
  const { reviewId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const q = usePanelReview(matter.id, reviewId);
  const [params, setParams] = useSearchParams();
  const asked = params.get('view') as View | null;
  const view: View = asked && VIEWS.includes(asked) ? asked : 'overview';
  const setView = useCallback((v: View) => setParams((p) => { const n = new URLSearchParams(p); n.set('view', v); return n; }, { replace: true }), [setParams]);
  const [jump, setJump] = useState<string | null>(null);
  const clearJump = useCallback(() => setJump(null), []);
  useScrollTo(jump, clearJump);
  const passage = usePassage();
  const idBase = useStableId('cn');
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] }), qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${encodeURIComponent(matter.number)}/panel`}><ArrowLeft />Council</Link>;
  const content = (q.data?.artifact?.content as PanelContent | undefined) ?? null;
  const places = useMemo(() => (content ? evidenceInFocus(content) : null), [content]);

  if (q.isPending) return <><Toolbar crumb={`${matter.number} · Council`} title="Council session">{back}</Toolbar><div className="lw-content"><Skeleton height={200} /></div></>;
  if (q.isError) return <><Toolbar crumb={`${matter.number} · Council`} title="Council session">{back}</Toolbar><div className="lw-content"><ErrorNote>This Council session is not available.</ErrorNote></div></>;

  const { review, artifact: a, can } = q.data;
  const working = review.status === 'queued' || review.status === 'running';
  const c = working ? null : content;                        // while it runs again, the earlier version is not shown as this run's
  const steps = councilFlow(review, c);
  const [label, tone] = a?.status === 'approved' && !working ? ['Approved by a partner', 'verified' as const] : PANEL_STATUS[review.status];
  const go = (v: View, target?: string) => { setView(v); if (target) window.setTimeout(() => setJump(target), 0); };
  const pick = (key: StepKey) => (key === 'question' ? go('overview') : go('discussion', `${idBase}-stage-${key}`));
  const messages = c ? c.perspectives.length + (c.challenges?.length ?? 0) + (c.revised ? 1 : 0) + 1 : 0;
  const area = review.area ?? review.specialist;
  const open = (e: ResearchEvidence, el?: HTMLElement | null) => passage.show(e, el);

  return (
    <>
      <Toolbar crumb={`${matter.number} · Council`} title="Council session">{back}</Toolbar>
      <div className="lw-content cn-page">
        <section className="lw-paper lw-sheet cn-head" aria-label="Question">
          <div className="cn-head-top">
            <div style={{ minWidth: 0 }}>
              <p className="cn-eyebrow">{['Council', area, review.depth === 'deep' ? 'Deep' : 'Focused'].filter(Boolean).join(' · ')}</p>
              <h2 className="cn-question-h">{review.question}</h2>
            </div>
            <StatusPill tone={tone}>{label}</StatusPill>
          </div>
          <p className="cn-meta-line">
            <span>{review.documents && review.documents.length
              ? `Consulted: ${review.documents.map((d) => d.title ?? 'a document no longer available').join('; ')}` : "Consulted: all of the matter's documents"}</span>
            <span>{`Asked by ${review.created_by ?? 'someone'}, ${when(review.created_at)}`}</span>
            {a && !working && <span>{`Version ${a.version}`}</span>}
          </p>
          {working && <p className="lw-meta" role="status" style={{ margin: 0 }}>{review.status === 'queued' ? 'Waiting to start. ' : ''}The answers are being drafted, challenged and checked; this takes a minute or two. The steps below follow the run.</p>}
          {c?.stopped && (
            <div className="cn-alert">
              <span><b>The Council stopped:</b> it needs facts the documents do not give.</span>
              <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => go('result', `${idBase}-facts`)}>Answer them<ArrowRight /></button>
            </div>
          )}
          <FlowStepper steps={steps} onPick={pick} />
          <p className="cn-notice">{c?.notice ?? NOTICE}</p>
        </section>

        <Tabs idBase={idBase} label="Council session" value={view} onChange={setView} tabs={[
          { id: 'overview', label: 'Overview' }, { id: 'discussion', label: 'Discussion', count: messages },
          { id: 'evidence', label: 'Evidence', count: places?.docs.length }, { id: 'result', label: 'Result' }]} />

        <div className="cn-panel" role="tabpanel" id={`${idBase}-panel-${view}`} aria-labelledby={`${idBase}-tab-${view}`} tabIndex={0}>
          {view === 'overview' && <Overview review={review} c={c} area={area} onSeat={(role) => go('discussion', `${idBase}-msg-${role}`)} onView={go} idBase={idBase} />}
          {view === 'discussion' && <Discussion review={review} c={c} steps={steps} idBase={idBase} onCite={open} />}
          {view === 'evidence' && <Evidence c={c} number={matter.number} onOpen={open} working={working} />}
          {view === 'result' && (c && a
            ? <Result review={review} a={a} c={c} can={can} matter={matter} idBase={idBase} refresh={refresh} />
            : <Waiting working={working} what="The result" />)}
        </div>
      </div>
      {passage.open && places && (
        <PassageDialog matterId={matter.id} number={matter.number} e={passage.open} usedBy={usedByOf(places.byPlace, passage.open)} onClose={passage.close} />
      )}
    </>
  );
}

function Waiting({ working, what }: { working: boolean; what: string }) {
  return (
    <div className="lw-paper lw-sheet" style={{ textAlign: 'center', padding: '36px 24px' }}>
      <p style={{ margin: 0, fontWeight: 600 }}>{working ? `${what} appears when the Council finishes.` : `${what} is not available for this session.`}</p>
      {working && <p className="lw-meta" style={{ margin: '4px 0 0' }}>Each answer is checked against the documents before it is shown.</p>}
    </div>
  );
}

// ── Overview: the question with the perspectives around it ──────────────────────────────────────────────────────────
function Overview({ review, c, area, onSeat, onView, idBase }: {
  review: PanelReview; c: PanelContent | null; area: string; onSeat: (role: PanelRole) => void; onView: (v: View, target?: string) => void; idBase: string;
}) {
  const seats: { role: PanelRole; label: string }[] = c ? c.perspectives.map((p) => ({ role: p.role, label: p.label })) : review.roles ?? [];
  const progress = review.progress;
  const working = review.status === 'queued' || review.status === 'running';
  const pastViews = !!progress?.step && progress.step !== 'views';
  const current = working && progress?.step === 'views' ? seats.find((s) => !progress.done.includes(s.role))?.role : undefined;
  return (
    <>
      {c && <ViewStrip c={c} onView={onView} idBase={idBase} />}
      <section className="cn-table" data-seats={Math.min(seats.length, 4)} aria-label="Perspectives">
        <div className="cn-question">
          <span className="cn-seal" aria-hidden><Scale /></span>
          <p className="cn-eyebrow">The question</p>
          <p className="cn-question-text">{review.question}</p>
          <p className="lw-meta" style={{ margin: 0 }}>{`${plural(seats.length, 'perspective')} · ${area}`}</p>
        </div>
        {seats.slice(0, 4).map((s) => {
          const p = c?.perspectives.find((x) => x.role === s.role);
          const ch = c?.challenges?.find((x) => x.role === s.role);
          const target = ch ? c?.perspectives.find((x) => x.role === ch.target)?.label ?? 'first' : '';
          const n = p ? verifiedCount(p) : 0;
          const state = !working ? null : progress?.done.includes(s.role) || pastViews ? 'done' : current === s.role ? 'current' : 'waiting';
          return (
            <article key={s.role} className="cn-seat" aria-label={s.label}>
              <div className="cn-seat-head">
                <RoleMark role={s.role} size="lg" />
                <div style={{ minWidth: 0 }}>
                  <h3 className="cn-seat-name">{s.label}</h3>
                  <p className="cn-seat-about">{aboutRole(s.role, area)}</p>
                </div>
              </div>
              {p ? (p.abstained || !openingLine(p)
                ? <p className="cn-seat-line" data-quiet="">Found nothing in the documents to answer this.</p>
                : <p className="cn-seat-line">{`“${openingLine(p)}”`}</p>)
                : state && (
                  <p className="cn-seat-state" data-state={state}>
                    {state === 'done' ? <><Check aria-hidden />Answered; shown when the Council finishes</>
                      : state === 'current' ? <><CircleDashed aria-hidden />Answering now…</> : 'Waiting for its turn'}
                  </p>)}
              {p && (
                <div className="cn-seat-foot">
                  {n > 0 && <button type="button" className="cn-chip" data-tone="gold" onClick={() => onSeat(s.role)}
                    aria-label={`${s.label}: ${plural(n, 'citation')}. Read the answer`}><FileText aria-hidden />{plural(n, 'citation')}</button>}
                  {ch && <span className="cn-chip">{ch.objects ? `Objected to the ${target} answer` : 'No objection'}</span>}
                  {p.missing_facts.length > 0 && <span className="cn-chip" data-tone="attention">Says facts are missing</span>}
                  {p.unsupported.length > 0 && <span className="cn-chip">{`${plural(p.unsupported.length, 'sentence')} without a source`}</span>}
                </div>
              )}
            </article>
          );
        })}
      </section>
    </>
  );
}

function ViewStrip({ c, onView, idBase }: { c: PanelContent; onView: (v: View, target?: string) => void; idBase: string }) {
  const text = c.conclusion ? c.conclusion
    : c.stopped ? 'No view yet: the Council stopped because the documents leave facts open.'
      : c.outcome === 'inconclusive' ? 'No conclusion: fewer than two of the answers’ quotes could be verified in the documents.'
        : 'No single conclusion was drawn; the result shows where the perspectives agree and differ.';
  return (
    <section className="cn-view" aria-label="The Council's view">
      <RoleMark role="chair" />
      <p className="cn-eyebrow">The Council’s view</p>
      <p className="cn-view-text">{text}</p>
      <div className="cn-view-actions">
        {(c.figures_unsupported?.length ?? 0) > 0 && <StatusPill tone="conflict">{`Check: ${c.figures_unsupported!.join(', ')} is in no answer`}</StatusPill>}
        <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => onView('result', c.stopped ? `${idBase}-facts` : undefined)}>
          {c.stopped ? 'Answer the facts' : 'See the result'}<ArrowRight /></button>
        <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => onView('discussion')}><MessagesSquare />Read the discussion</button>
      </div>
    </section>
  );
}

// ── Discussion ──────────────────────────────────────────────────────────────────────────────────────────────────────
function Stage({ id, n, title, sub, children }: { id: string; n: number; title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="cn-stage" id={id} tabIndex={-1} aria-labelledby={`${id}-h`}>
      <h3 className="cn-stage-h" id={`${id}-h`}><span className="cn-stage-n" aria-hidden>{n}</span>{title}</h3>
      {sub && <p className="cn-stage-sub">{sub}</p>}
      {children}
    </section>
  );
}

function Message({ id, role, label, tag, p, onCite, kind = 'view' }: {
  id?: string; role: PanelRole; label: string; tag: string; p: PanelPerspective; onCite: (e: ResearchEvidence, el: HTMLElement) => void; kind?: string;
}) {
  const sources = p.evidence.filter((e, i, all) => e.verification !== 'failed'
    && all.findIndex((x) => x.n === e.n && x.verification !== 'failed') === i);            // one chip per source passage
  return (
    <li id={id} tabIndex={id ? -1 : undefined}>
      <article className="cn-msg" data-kind={kind} aria-label={`${label}: ${tag}`}>
        <RoleMark role={role} />
        <div className="cn-bubble">
          <div className="cn-msg-head"><b>{label}</b><span className="cn-tag">{tag}</span></div>
          <p className="cn-msg-text"><CitedText text={p.text} evidence={p.evidence} onOpen={onCite} who={label} /></p>
          {sources.length > 0 && (
            <ul className="cn-cites" aria-label={`${label}: sources`}>
              {sources.map((e, i) => (
                <li key={i}><button type="button" className="cn-chip" data-tone={isVerified(e) ? 'gold' : undefined} onClick={(ev) => onCite(e, ev.currentTarget)}
                  aria-label={`Source ${e.n}: ${placeOf(e)}. Open the passage`}><b>{e.n}</b><span>{placeOf(e)}</span></button></li>
              ))}
            </ul>
          )}
          {p.unsupported.length > 0 && (
            <details className="cn-uncited">
              <summary>{`Cites nothing: ${plural(p.unsupported.length, 'sentence')}`}</summary>
              <ul>{p.unsupported.map((x, i) => <li key={i}>{`“${x}”`}</li>)}</ul>
            </details>
          )}
          {p.failed.length > 0 && <p className="cn-note">{`Quote not found in the source: ${p.failed.length}`}</p>}
          {p.missing_facts.length > 0 && <p className="cn-note"><b>Says it lacks:</b>{` ${p.missing_facts.join('; ')}`}</p>}
        </div>
      </article>
    </li>
  );
}

function Pending({ role, label, text }: { role: PanelRole | 'chair'; label: string; text: string }) {
  return (
    <li>
      <article className="cn-msg" data-kind="pending" aria-label={`${label}: ${text}`}>
        <RoleMark role={role} />
        <div className="cn-bubble"><div className="cn-msg-head"><b>{label}</b><span className="cn-tag">{text}</span></div></div>
      </article>
    </li>
  );
}

function Discussion({ review, c, steps, idBase, onCite }: {
  review: PanelReview; c: PanelContent | null; steps: FlowStep[]; idBase: string; onCite: (e: ResearchEvidence, el: HTMLElement) => void;
}) {
  const step = (k: StepKey) => steps.find((s) => s.key === k)!;
  const sid = (k: StepKey) => `${idBase}-stage-${k}`;
  if (!c) {
    const roles = review.roles ?? [];
    const p = review.progress;
    const words = (k: StepKey) => { const s = step(k); return s.state === 'current' ? s.detail : s.state === 'skipped' ? s.detail : s.state === 'done' ? 'Done' : s.state === 'failed' ? s.detail : 'Not started'; };
    return (
      <>
        <Stage id={sid('views')} n={1} title="Initial views" sub="Each perspective answers alone from the matter's documents, without seeing the others. The answers are shown when they have all been checked.">
          <ol className="cn-feed">{roles.map((r) => (
            <Pending key={r.role} role={r.role} label={r.label} text={p?.done.includes(r.role) || (p?.step && p.step !== 'views') ? 'Answered'
              : step('views').state === 'current' && roles.find((x) => !p?.done.includes(x.role))?.role === r.role ? 'Answering now…' : 'Waiting'} />))}</ol>
        </Stage>
        <Stage id={sid('challenge')} n={2} title="Challenge" sub={words('challenge')}><></></Stage>
        <Stage id={sid('reply')} n={3} title="Responses" sub={words('reply')}><></></Stage>
        <Stage id={sid('chair')} n={4} title="Synthesis" sub={words('chair')}><></></Stage>
      </>
    );
  }
  const first = c.perspectives[0]?.label ?? 'first';
  const challenges: PanelChallenge[] = c.challenges ?? [];
  const label = (r: PanelRole) => c.perspectives.find((p) => p.role === r)?.label ?? r;
  return (
    <>
      <Stage id={sid('views')} n={1} title="Initial views" sub="Each perspective answered alone from the matter's documents, without seeing the others. Each chip opens the passage it cites.">
        <ol className="cn-feed">{c.perspectives.map((p) => <Message key={p.role} id={`${idBase}-msg-${p.role}`} role={p.role} label={p.label} tag="Initial view" p={p} onCite={onCite} />)}</ol>
      </Stage>
      <Stage id={sid('challenge')} n={2} title="Challenge" sub={challenges.length
        ? `The challengers read only the verified sentences of the ${first} answer and objected where the sources show it wrong or incomplete.` : step('challenge').detail}>
        {challenges.length > 0 && (
          <ol className="cn-feed">{challenges.map((x) => {
            const target = c.perspectives.find((p) => p.role === x.target)?.label ?? 'first';
            return x.objects
              ? <Message key={x.role} role={x.role} label={x.label} tag={`Objection to the ${target} answer`} p={x} onCite={onCite} kind="objection" />
              : <Pending key={x.role} role={x.role} label={x.label} text={`No objection to the ${target} answer that the sources support`} />;
          })}</ol>
        )}
      </Stage>
      <Stage id={sid('reply')} n={3} title="Responses" sub={c.revised ? `The ${first} answer replied to the objections, checked like any answer.` : step('reply').detail}>
        {c.revised && <ol className="cn-feed"><Message role={c.revised.role} label={c.revised.label} tag="Reply to the objections" p={c.revised} onCite={onCite} kind="reply" /></ol>}
      </Stage>
      <Stage id={sid('chair')} n={4} title="Synthesis" sub="The chair read only the verified sentences of the answers and objections, never the sources, and adds nothing of its own.">
        <ol className="cn-feed">
          <li>
            <article className="cn-msg" data-kind="chair" aria-label="The chair: synthesis">
              <RoleMark role="chair" />
              <div className="cn-bubble">
                <div className="cn-msg-head"><b>The chair</b><span className="cn-tag">Synthesis</span></div>
                {c.conclusion ? <p className="cn-msg-text">{c.conclusion}</p>
                  : <p className="cn-note">{c.stopped ? 'No conclusion yet: the Council stopped for facts the documents do not give.'
                    : c.outcome === 'inconclusive' ? 'No conclusion: fewer than two of the answers’ quotes could be verified, so the chair drew none.'
                      : 'No single conclusion; see where the perspectives agree and differ in the result.'}</p>}
                {(c.figures_unsupported?.length ?? 0) > 0 && <p style={{ margin: 0 }}><StatusPill tone="conflict">{`Check: ${c.figures_unsupported!.join(', ')} is in no answer`}</StatusPill></p>}
                {(c.points?.length ?? 0) > 0 && (
                  <ul className="cn-points" aria-label="What matters most">{c.points!.map((x, i) => (
                    <li key={i}><span>{x.text}</span>{x.roles.length > 0 && <span className="cn-by">{x.roles.map((r) => <span key={r}><RoleMark role={r} size="sm" />{label(r)}</span>)}</span>}</li>))}</ul>
                )}
                {(c.gaps?.length ?? 0) > 0 && <p className="cn-note"><b>Not established by the documents:</b>{` ${c.gaps!.join('; ')}.`}</p>}
              </div>
            </article>
          </li>
        </ol>
      </Stage>
    </>
  );
}

// ── Evidence in focus ───────────────────────────────────────────────────────────────────────────────────────────────
function Evidence({ c, number, onOpen, working }: { c: PanelContent | null; number: string; onOpen: (e: ResearchEvidence, el: HTMLElement) => void; working: boolean }) {
  const data = useMemo(() => (c ? evidenceInFocus(c) : null), [c]);
  if (!c || !data) return <Waiting working={working} what="The evidence" />;
  const total = data.docs.reduce((n, d) => n + d.places.length, 0);
  const seated = [...c.perspectives.map((p) => ({ role: p.role, label: p.label }))];
  return (
    <>
      <section className="cn-section" aria-label="Evidence in focus">
        <h3 className="cn-section-h">Evidence in focus</h3>
        <p className="cn-section-sub">{data.docs.length
          ? `${plural(data.docs.length, 'document')} cited at ${plural(total, 'place')}; each quote below was found in its document.`
          : 'No quote in this session was found in the documents.'}{data.unverified ? ` ${plural(data.unverified, 'quote')} could not be found in ${data.unverified === 1 ? 'its source and is' : 'their sources and are'} not listed.` : ''}</p>
        <ul className="cn-legend" aria-label="Perspectives">{seated.map((s) => <li key={s.role}><RoleMark role={s.role} size="sm" />{s.label}</li>)}</ul>
      </section>
      {data.docs.length > 0 && (
        <ul className="cn-docs" aria-label="Documents cited">
          {data.docs.map((d) => (
            <li key={d.id}>
              <article className="cn-doc" aria-label={d.title}>
                <div className="cn-doc-head">
                  <h4 className="cn-doc-title"><FileText aria-hidden /><span>{d.title}</span></h4>
                  <span className="cn-doc-roles">{`Cited by ${plural(d.roles.length, 'perspective')}`}</span>
                </div>
                <ul className="cn-places">
                  {d.places.map((pl) => (
                    <li key={pl.key} className="cn-place">
                      <div style={{ display: 'grid', gap: 6, minWidth: 0 }}>
                        <p className="cn-place-where">{placeOf(pl.e, false) || `Version ${pl.e.version_number}`}</p>
                        <p className="cn-place-quote">“<mark>{pl.e.quote}</mark>”</p>
                        <ul className="cn-usedby" aria-label="Cited by">{pl.by.map((u, i) => (
                          <li key={i}><RoleMark role={u.role} size="sm" />{u.kind === 'view' ? u.label : `${u.label} (${u.kind === 'objection' ? 'objection' : 'reply'})`}</li>))}</ul>
                      </div>
                      <div className="cn-place-actions">
                        <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={(ev) => onOpen(pl.e, ev.currentTarget)}>Show the passage</button>
                        <Link className="lw-btn lw-btn-sm" to={readerHref(number, pl.e)}>Open at the quote</Link>
                      </div>
                    </li>
                  ))}
                </ul>
              </article>
            </li>
          ))}
        </ul>
      )}
      {c.facts_given.length > 0 && (
        <section className="cn-section" aria-label="Facts you gave">
          <h3 className="cn-section-h">Facts you gave (not from the documents)</h3>
          <dl className="lw-intake-answers">{c.facts_given.map((f) => <React.Fragment key={f.question}><dt>{f.question}</dt><dd>{f.answer}</dd></React.Fragment>)}</dl>
        </section>
      )}
    </>
  );
}

// ── Result ──────────────────────────────────────────────────────────────────────────────────────────────────────────
function Result({ review, a, c, can, matter, idBase, refresh }: {
  review: PanelReview; a: ArtifactVersion<PanelContent>; c: PanelContent; can: { approve: boolean; facts: boolean; author: boolean }; matter: Matter;
  idBase: string; refresh: () => Promise<unknown>;
}) {
  return (
    <>
      {c.stopped && <MissingFacts id={`${idBase}-facts`} c={c} canGive={can.facts} path={`${panelPath(matter.id)}/${encodeURIComponent(review.id)}`} onDone={refresh} />}
      {c.proceeded_without && (
        <section className="cn-section" aria-label="Continued without facts">
          <h3 className="cn-section-h">Continued without some facts</h3>
          <p style={{ margin: 0 }}>{`${c.proceeded_without.by} chose to continue on ${when(c.proceeded_without.at)}: “${c.proceeded_without.reason}”`}</p>
          <p className="lw-meta" style={{ margin: 0 }}>{`Left open: ${c.proceeded_without.facts.join('; ')}.`}</p>
        </section>
      )}
      <div className="cn-result-grid">
        <div style={{ display: 'grid', gap: 16 }}>
          {!c.stopped && <Conclusion c={c} />}
          <Differences c={c} />
        </div>
        <div style={{ display: 'grid', gap: 16 }}>
          {!c.stopped && <StillOpen c={c} />}
          <NextSteps c={c} matter={matter} />
          <section className="cn-section" aria-label="Partner review">
            <h3 className="cn-section-h">Partner review</h3>
            <ReviewBanner v={a} made={`Version ${a.version} · asked by ${review.created_by ?? 'someone'}`} />
            {!c.stopped && (can.approve || can.author
              ? <ApprovalDialog v={a} what="this panel review" endpoint={`${artifactPath(matter.id, a.id)}/review`} canApprove={can.approve || can.author} isAuthor={can.author} onDone={refresh} />
              : a.status === 'draft' && <p className="lw-meta" style={{ margin: 0 }}>Waiting for a partner: the matter's lead or a managing partner.</p>)}
            <ReviewDecisions decisions={a.decisions} />
          </section>
        </div>
      </div>
    </>
  );
}

function Conclusion({ c }: { c: PanelContent }) {
  const label = (r: PanelRole) => c.perspectives.find((p) => p.role === r)?.label ?? r;
  if (c.outcome === 'inconclusive') {
    return (
      <section className="cn-section" aria-label="Conclusion">
        <h3 className="cn-section-h">No conclusion</h3>
        <p style={{ margin: 0 }}>Fewer than two of the answers' quotes could be verified in the documents, so nothing was challenged and the chair drew no conclusion.</p>
      </section>
    );
  }
  if (!c.conclusion && !(c.points?.length)) return null;
  return (
    <section className="cn-section" aria-label="Conclusion">
      <div className="cn-view" style={{ boxShadow: 'none', padding: 0, background: 'transparent' }}>
        <RoleMark role="chair" />
        <h3 className="cn-eyebrow">The Council’s view</h3>
        {c.conclusion && <p className="cn-view-text">{c.conclusion}</p>}
      </div>
      {(c.figures_unsupported?.length ?? 0) > 0 && <p style={{ margin: 0 }}><StatusPill tone="conflict">{`Check: ${c.figures_unsupported!.join(', ')} is in no answer`}</StatusPill></p>}
      {(c.points?.length ?? 0) > 0 && (
        <>
          <h4 className="cn-eyebrow" style={{ marginTop: 4 }}>What matters most</h4>
          <ul className="cn-points" aria-label="What matters most">{c.points!.map((x, i) => (
            <li key={i}><span>{x.text}</span>{x.roles.length > 0 && <span className="cn-by" aria-label="Supported by">{x.roles.map((r) => <span key={r}><RoleMark role={r} size="sm" />{label(r)}</span>)}</span>}</li>))}</ul>
        </>
      )}
      <p className="cn-note">Drawn from the verified sentences of the answers and objections; options to consider, not a decision.</p>
    </section>
  );
}

/** The chair sometimes names a perspective by its key ("the 'deal' perspective"); show its name instead. */
function named(text: string, c: PanelContent): string {
  return text.replace(/['‘"“](deal|risk|litigation|specialist|lead|opposing|evidence|contract)['’"”]/g,
    (m, key: string) => c.perspectives.find((p) => p.role === key)?.label ?? m);
}

function Differences({ c }: { c: PanelContent }) {
  if (c.matrix.length === 0) return null;
  const differ = c.matrix.filter((m) => m.disagreement);
  return (
    <section className="cn-section" aria-label="Where they differ">
      <h3 className="cn-section-h">Where they differ</h3>
      <p className="cn-section-sub">Drawn by the chair from the answers' verified sentences only. Check each against the answer it summarises.</p>
      {differ.length === 0 ? <p style={{ margin: 0 }}>The perspectives did not differ in substance on the issues the chair compared.</p> : (
        <ul className="cn-differs">
          {differ.map((m, i) => (
            <li key={i} className="cn-differ">
              <h4 className="cn-differ-h">{m.issue}</h4>
              {m.summary && <p className="cn-note" style={{ marginTop: -4 }}>{named(m.summary, c)}</p>}
              {m.figures_unsupported.length > 0 && <p style={{ margin: 0 }}><StatusPill tone="conflict">{`Check: ${m.figures_unsupported.join(', ')} is in no answer`}</StatusPill></p>}
              <ul className="cn-positions">{c.perspectives.map((p) => (
                <li key={p.role}><RoleMark role={p.role} size="sm" /><span><b>{p.label}:</b>{` ${m.positions[p.role] ?? 'not addressed'}`}</span></li>))}</ul>
            </li>
          ))}
        </ul>
      )}
      <details className="cn-details">
        <summary>{`All ${plural(c.matrix.length, 'issue')} as a table`}</summary>
        <div className="cn-table-wrap">
          <table className="lw-table" aria-label="Disagreement matrix">
            <thead><tr><th scope="col">Issue</th>{c.perspectives.map((p) => <th key={p.role} scope="col">{p.label}</th>)}</tr></thead>
            <tbody>{c.matrix.map((m, i) => (
              <tr key={i} data-disagreement={m.disagreement ? 'yes' : 'no'}>
                <td><b>{m.issue}</b>{m.disagreement && <div><StatusPill tone="attention">They differ</StatusPill></div>}
                  {m.summary && <div className="lw-meta">{named(m.summary, c)}</div>}
                  {m.figures_unsupported.length > 0 && <div><StatusPill tone="conflict">{`Check: ${m.figures_unsupported.join(', ')} is in no answer`}</StatusPill></div>}</td>
                {c.perspectives.map((p) => <td key={p.role}>{m.positions[p.role] ?? 'not addressed'}</td>)}
              </tr>))}</tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

function StillOpen({ c }: { c: PanelContent }) {
  const gaps = c.gaps ?? [];
  const asked = c.missing_facts.map((m) => ({ m, by: c.perspectives.filter((p) => p.missing_facts.some((x) => x.toLowerCase() === m.toLowerCase())).map((p) => p.label) }));
  return (
    <section className="cn-section" aria-label="Facts still missing">
      <h3 className="cn-section-h">Facts still missing</h3>
      {gaps.length === 0 && asked.length === 0 ? <p className="cn-note">The chair named no gaps, and no perspective said it lacked facts.</p> : (
        <>
          {gaps.length > 0 && (<><h4 className="cn-eyebrow">Not established by the documents</h4>
            <ul className="cn-open-list">{gaps.map((g, i) => <li key={i}>{g}</li>)}</ul></>)}
          {asked.length > 0 && (<><h4 className="cn-eyebrow">Facts the perspectives asked for</h4>
            <ul className="cn-open-list">{asked.map(({ m, by }, i) => <li key={i}>{m}{by.length > 0 && <span className="lw-meta">{` — ${by.join(', ')}`}</span>}</li>)}</ul></>)}
        </>
      )}
    </section>
  );
}

/** Next steps a person writes into the matter's plan (`POST /legal/matters/{m}/tasks`), each read and edited first. */
function NextSteps({ c, matter }: { c: PanelContent; matter: Matter }) {
  const ideas = useMemo(() => stepIdeas(c), [c]);
  const canPlan = matter.status === 'open' && ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '');
  const tasks = useTasks(matter.id);
  const qc = useQueryClient();
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [own, setOwn] = useState('');
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState('');
  const planned = (title: string) => (tasks.data ?? []).some((t) => t.status !== 'cancelled' && t.title.trim().toLowerCase() === title.trim().toLowerCase());
  const add = async (i: number, title: string) => {
    setBusy(i); setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/tasks`, { title: title.trim().slice(0, 300), kind: 'task' });
      await Promise.all([qc.invalidateQueries({ queryKey: keys.tasks(matter.id) }), qc.invalidateQueries({ queryKey: keys.pipeline })]);
      if (i < 0) setOwn('');
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The step could not be added to the plan.'); } finally { setBusy(null); }
  };
  return (
    <section className="cn-section" aria-label="Next steps">
      <h3 className="cn-section-h"><ListPlus aria-hidden style={{ width: 18, height: 18 }} />Next steps</h3>
      <p className="cn-section-sub">{ideas.length
        ? 'Suggested by rule from what this session left open. Edit a step, then add it to the matter’s plan; it is written as yours.'
        : 'This session left nothing open that suggests a step. You can still add your own to the plan.'}</p>
      {canPlan ? (
        <ul className="cn-plan">
          {ideas.map((x, i) => {
            const value = drafts[i] ?? x.text;
            const done = planned(value);
            return (
              <li key={i} className="cn-plan-row">
                <Field label={`Step ${i + 1}`}>{(id) => <textarea id={id} className="lw-input" rows={2} maxLength={300} value={value} onChange={(e) => setDrafts({ ...drafts, [i]: e.target.value })} />}</Field>
                <div className="cn-plan-foot">
                  <p className="cn-plan-from">{x.from}</p>
                  {done ? <span className="cn-plan-added"><Check aria-hidden />In the plan</span>
                    : <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy !== null || !value.trim()} onClick={() => void add(i, value)} aria-label={`Add step ${i + 1} to the plan`}><ListPlus />Add to the plan</button>}
                </div>
              </li>
            );
          })}
          <li className="cn-plan-row">
            <Field label="Your own step">{(id) => <textarea id={id} className="lw-input" rows={1} maxLength={300} value={own} placeholder="For example: send the client a note on clause 14.2" onChange={(e) => setOwn(e.target.value)} />}</Field>
            <div className="cn-plan-foot">
              <span />
              <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy !== null || !own.trim()} onClick={() => void add(-1, own)} aria-label="Add your own step to the plan"><ListPlus />Add to the plan</button>
            </div>
          </li>
        </ul>
      ) : (
        <>
          {ideas.length > 0 && <ul className="cn-open-list">{ideas.map((x, i) => <li key={i}>{x.text}</li>)}</ul>}
          <p className="cn-note">{matter.status === 'open' ? "The matter's lawyers and paralegals add steps to its plan." : 'This matter is closed; its plan no longer changes.'}</p>
        </>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      <p className="cn-note" style={{ margin: 0 }}><Link className="lw-linklike" to={`/m/${encodeURIComponent(matter.number)}`}>Open the matter's plan</Link></p>
    </section>
  );
}

function MissingFacts({ id, c, canGive, path, onDone }: { id: string; c: PanelContent; canGive: boolean; path: string; onDone: () => Promise<unknown> }) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const filled = c.missing_facts.filter((m) => (answers[m] ?? '').trim());
  const [why, setWhy] = useState('');
  const send = async () => {
    setBusy(true); setError('');
    try { await api.post(`${path}/facts`, { facts: filled.map((m) => ({ question: m, answer: answers[m].trim() })) }); await onDone(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The facts could not be sent.'); } finally { setBusy(false); }
  };
  const proceed = async () => {
    setBusy(true); setError('');
    try { await api.post(`${path}/proceed`, { reason: why.trim() }); await onDone(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The Council could not continue.'); } finally { setBusy(false); }
  };
  return (
    <section className="cn-section lw-panel-stop" id={id} tabIndex={-1} aria-label="Facts needed">
      <h3 className="cn-section-h">The Council stopped: it needs facts the documents do not give</h3>
      <p className="cn-section-sub" style={{ marginTop: -4 }}>Answer what you can. The Council then runs again, with your answers marked as yours, not the documents'. Nothing here is approved until it does.</p>
      {canGive ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {c.missing_facts.map((m) => (
            <Field key={m} label={m[0].toUpperCase() + m.slice(1)}>{(fid) => <input id={fid} className="lw-input" maxLength={1000} value={answers[m] ?? ''}
              onChange={(e) => setAnswers({ ...answers, [m]: e.target.value })} />}</Field>
          ))}
          {error && <ErrorNote>{error}</ErrorNote>}
          <div><button type="button" className="lw-btn lw-btn-primary" disabled={busy || filled.length === 0} onClick={send}><Send />Give these facts and run again</button></div>
          <div className="lw-panel-proceed">
            <Field label="Or continue without them, because" hint="Recorded with your name; the partner sees what was left open.">
              {(fid) => <input id={fid} className="lw-input" maxLength={500} value={why} onChange={(e) => setWhy(e.target.value)} />}
            </Field>
            <div><button type="button" className="lw-btn" disabled={busy || !why.trim()} onClick={proceed}>Continue without these facts</button></div>
          </div>
        </div>
      ) : <ul className="cn-open-list">{c.missing_facts.map((m) => <li key={m}>{m}</li>)}</ul>}
    </section>
  );
}
