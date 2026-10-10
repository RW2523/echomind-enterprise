import React, { useState } from 'react';
import { Link, useParams } from 'react-router';
import { ArrowLeft, Send } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, panelPath, usePanelReview } from '../api/queries';
import type { PanelContent, PanelPerspective, PanelRole, ResearchEvidence } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';
import { PANEL_STATUS } from './MatterPanel';

const ROLES: PanelRole[] = ['deal', 'risk', 'litigation', 'specialist'];

/** One panel review (S4.5, D-057). */
export default function PanelReviewPage() {
  const { reviewId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const q = usePanelReview(matter.id, reviewId);
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] }), qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/panel`}><ArrowLeft />Review panel</Link>;
  if (q.isPending) return <><Toolbar crumb={`${matter.number} · Review panel`} title="Panel review">{back}</Toolbar><div className="lw-content"><Skeleton height={200} /></div></>;
  if (q.isError) return <><Toolbar crumb={`${matter.number} · Review panel`} title="Panel review">{back}</Toolbar><div className="lw-content"><ErrorNote>This panel review is not available.</ErrorNote></div></>;
  const { review, artifact: a, can } = q.data;
  const working = review.status === 'queued' || review.status === 'running';
  const [label, tone] = PANEL_STATUS[review.status];
  const c = a?.content as PanelContent | undefined;
  return (
    <>
      <Toolbar crumb={`${matter.number} · Review panel`} title="Panel review">{back}</Toolbar>
      <div className="lw-content" style={{ maxWidth: 1180 }}>
        <section className="lw-paper lw-sheet" aria-label="Question">
          <p className="lw-eyebrow" style={{ margin: 0 }}>{`Question · specialist: ${review.area ?? review.specialist}`}</p>
          <h2 className="lw-h2 lw-serif" style={{ margin: '2px 0 8px' }}>{review.question}</h2>
          <p className="lw-meta" style={{ margin: '0 0 8px' }}>{review.documents && review.documents.length
            ? `Consulted: ${review.documents.map((d) => d.title ?? 'a document no longer available').join('; ')}` : "Consulted: all of the matter's documents"}</p>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <StatusPill tone={a?.status === 'approved' ? 'verified' : tone}>{a?.status === 'approved' ? 'Approved by a partner' : label}</StatusPill>
            {working && <span className="lw-meta" role="status">Five answers are being drafted and checked; this takes about a minute.</span>}
          </div>
          {c && <p className="lw-panel-notice">{c.notice}</p>}
        </section>
        {c && a && (
          <>
            {c.stopped && <MissingFacts c={c} canGive={can.facts} path={`${panelPath(matter.id)}/${encodeURIComponent(review.id)}/facts`} onDone={refresh} />}
            {c.facts_given.length > 0 && (
              <section className="lw-paper lw-sheet" aria-label="Facts you gave">
                <h3 className="lw-h3" style={{ marginTop: 0 }}>Facts you gave (not from the documents)</h3>
                <dl className="lw-intake-answers">{c.facts_given.map((f) => <React.Fragment key={f.question}><dt>{f.question}</dt><dd>{f.answer}</dd></React.Fragment>)}</dl>
              </section>
            )}
            <section className="lw-panel-grid" aria-label="Perspectives">
              {c.perspectives.map((p) => <Perspective key={p.role} p={p} number={matter.number} />)}
            </section>
            {c.matrix.length > 0 && (
              <section className="lw-paper lw-sheet" aria-label="Where they agree and differ">
                <h3 className="lw-h3" style={{ marginTop: 0 }}>Where they agree and differ</h3>
                <p className="lw-meta" style={{ margin: '0 0 8px' }}>Drawn by the moderator from the four answers only. Check each against the answer it summarises.</p>
                <div style={{ overflowX: 'auto' }}>
                  <table className="lw-table" aria-label="Disagreement matrix">
                    <thead><tr><th>Issue</th>{c.perspectives.map((p) => <th key={p.role}>{p.label}</th>)}</tr></thead>
                    <tbody>{c.matrix.map((m, i) => (
                      <tr key={i} data-disagreement={m.disagreement ? 'yes' : 'no'}>
                        <td><b>{m.issue}</b>{m.disagreement && <div><StatusPill tone="attention">They differ</StatusPill></div>}
                          {m.summary && <div className="lw-meta">{m.summary}</div>}
                          {m.figures_unsupported.length > 0 && <div><StatusPill tone="conflict">{`Check: ${m.figures_unsupported.join(', ')} is in no answer`}</StatusPill></div>}</td>
                        {ROLES.map((r) => <td key={r}>{m.positions[r]}</td>)}
                      </tr>))}</tbody>
                  </table>
                </div>
              </section>
            )}
            <section className="lw-paper lw-sheet" aria-label="Partner review">
              <ReviewBanner v={a} made={`Version ${a.version} · asked by ${review.created_by ?? 'someone'}`} />
              {!c.stopped && (can.approve || can.author
                ? <ApprovalDialog v={a} what="this panel review" endpoint={`${artifactPath(matter.id, a.id)}/review`} canApprove={can.approve || can.author} isAuthor={can.author} onDone={refresh} />
                : a.status === 'draft' && <p className="lw-meta" style={{ margin: 0 }}>Waiting for a partner: the matter's lead or a managing partner.</p>)}
              <ReviewDecisions decisions={a.decisions} />
            </section>
          </>
        )}
      </div>
    </>
  );
}

function Perspective({ p, number }: { p: PanelPerspective; number: string }) {
  return (
    <article className="lw-paper lw-sheet lw-panel-card" aria-label={p.label}>
      <h3 className="lw-h3" style={{ margin: 0 }}>{p.label}</h3>
      <p className="lw-serif lw-panel-answer">{p.text}</p>
      {p.unsupported.length > 0 && <p className="lw-meta" style={{ margin: 0 }}>{`Cites nothing: ${p.unsupported.map((s) => `“${s}”`).join(' ')}`}</p>}
      {p.failed.length > 0 && <p className="lw-meta" style={{ margin: 0 }}>{`Quote not found in the source: ${p.failed.length}`}</p>}
      {p.evidence.length > 0 && (
        <ol className="lw-note-evidence" aria-label={`${p.label}: evidence`}>{p.evidence.map((e, i) => <Evidence key={i} e={e} number={number} />)}</ol>
      )}
      {p.missing_facts.length > 0 && <p className="lw-meta" style={{ margin: 0 }}>{`Says it lacks: ${p.missing_facts.join('; ')}`}</p>}
    </article>
  );
}

function Evidence({ e, number }: { e: ResearchEvidence; number: string }) {
  const href = e.char_start != null && e.char_end != null
    ? `/m/${number}/documents/${encodeURIComponent(e.document_id)}/read?v=${encodeURIComponent(e.version_id)}&at=${e.char_start}-${e.char_end}` : null;
  const where = [e.document_title, e.clause ? `clause ${e.clause}` : '', e.line ? `line ${e.line}` : ''].filter(Boolean).join(', ');
  return (
    <li data-verification={e.verification}>
      <span className="lw-serif">{`“${e.quote}”`}</span>
      <div className="lw-meta">{where}{href && <> · <Link className="lw-linklike" to={href}>Open at the quote</Link></>}</div>
    </li>
  );
}

function MissingFacts({ c, canGive, path, onDone }: { c: PanelContent; canGive: boolean; path: string; onDone: () => Promise<unknown> }) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const filled = c.missing_facts.filter((m) => (answers[m] ?? '').trim());
  const send = async () => {
    setBusy(true); setError('');
    try { await api.post(path, { facts: filled.map((m) => ({ question: m, answer: answers[m].trim() })) }); await onDone(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The facts could not be sent.'); } finally { setBusy(false); }
  };
  return (
    <section className="lw-paper lw-sheet lw-panel-stop" aria-label="Facts needed">
      <h3 className="lw-h3" style={{ marginTop: 0 }}>The panel stopped: it needs facts the documents do not give</h3>
      <p className="lw-meta" style={{ margin: '0 0 8px' }}>Answer what you can. The panel then runs again, with your answers marked as yours, not the documents'. Nothing here is approved until it does.</p>
      {canGive ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {c.missing_facts.map((m) => (
            <Field key={m} label={m[0].toUpperCase() + m.slice(1)}>{(id) => <input id={id} className="lw-input" maxLength={1000} value={answers[m] ?? ''}
              onChange={(e) => setAnswers({ ...answers, [m]: e.target.value })} />}</Field>
          ))}
          {error && <ErrorNote>{error}</ErrorNote>}
          <div><button type="button" className="lw-btn lw-btn-primary" disabled={busy || filled.length === 0} onClick={send}><Send />Give these facts and run again</button></div>
        </div>
      ) : <ul>{c.missing_facts.map((m) => <li key={m}>{m}</li>)}</ul>}
    </section>
  );
}
