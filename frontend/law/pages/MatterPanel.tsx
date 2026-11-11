import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { RefreshCw, Users2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { panelPath, useDocuments, usePanelReviews } from '../api/queries';
import type { CouncilQuestion, PanelReview, PanelRole } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import { RoleMark, STEP_LABEL } from '../ui/council';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
export const PANEL_STATUS: Record<PanelReview['status'], [string, Tone]> = {
  queued: ['Waiting to start', 'neutral'], running: ['The Council is working', 'accent'], done: ['Ready for a partner', 'attention'],
  needs_facts: ['Stopped: facts needed', 'conflict'], failed: ['Could not finish', 'conflict'], not_allowed: ['No longer allowed', 'neutral'],
};
const DEFAULT_SEATS: { role: PanelRole; label: string }[] = [
  { role: 'deal', label: 'Deal lawyer' }, { role: 'risk', label: 'Risk and compliance' }, { role: 'litigation', label: 'Litigator' },
  { role: 'specialist', label: 'Specialist in the chosen area' }];
const HOW: [string, string][] = [
  ['Question', 'You ask; the perspectives are chosen by rule from your words.'],
  ['Initial views', 'Each answers alone from the documents, citing them.'],
  ['Challenge', 'The first answer is tested against the sources.'],
  ['Responses', 'At "deep", it replies to the objections.'],
  ['Synthesis', 'A chair sets out the view and where they differ.'],
];
const STEP_NO = { views: 2, challenge: 3, reply: 4, chair: 5 } as const;

/** The Council (the S4.5 review panel, grown into the Council 2026-11-05, redesigned 2026-11-07): put one question to
 *  several perspectives chosen for it, each answering alone from this matter's documents and citing them; the first answer
 *  is challenged, and a chair gives the conclusion and where they differ. Drafted by the same model; a partner reviews it. */
export default function MatterPanel() {
  const matter = useMatter();
  const q = usePanelReviews(matter.id);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [question, setQuestion] = useState('');
  const [area, setArea] = useState('commercial');
  const [depth, setDepth] = useState<'focused' | 'deep'>('focused');
  const [chosen, setChosen] = useState<string[]>([]);
  const docs = useDocuments(matter.id);
  const ready = (docs.data ?? []).filter((d) => d.current_version?.index_status === 'indexed');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const open = (id: string) => navigate(`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(id)}`);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      const r = await api.post<{ review: { id: string } }>(panelPath(matter.id), { question, specialist: area, documents: chosen, depth });
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] });
      open(r.review.id);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The question could not be sent.'); } finally { setBusy(false); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Council" />
      <div className="lw-content cn-home">
        <section className="lw-paper lw-sheet cn-intro" aria-labelledby="cn-intro-h">
          <div style={{ display: 'grid', gap: 6 }}>
            <p className="cn-eyebrow">{`${matter.number} · Council`}</p>
            <h2 className="cn-intro-h" id="cn-intro-h">Put a question to the Council</h2>
            <p className="lw-meta" style={{ margin: 0, maxWidth: '80ch' }}>Several perspectives answer one question from this matter's documents, each alone and citing them. For a challenge the Council seats lead counsel, an opposing reading and an evidence investigator; otherwise the four below. If the answer depends on facts the documents do not give, it stops and asks you.</p>
          </div>
          {q.data?.can.ask && (
            <form className="cn-ask" aria-label="Ask the Council" onSubmit={submit}>
              <Field label="Question">{(id) => <textarea id={id} className="lw-input" rows={3} maxLength={2000} value={question}
                placeholder="For example: Can the Supplier terminate if the Customer undergoes a change of control?" onChange={(e) => setQuestion(e.target.value)} />}</Field>
              {ready.length > 1 && (
                <fieldset className="lw-panel-docs">
                  <legend className="lw-meta">Only these documents (optional; all of the matter's otherwise)</legend>
                  {ready.map((d) => (
                    <label key={d.id} className="lw-check"><input type="checkbox" checked={chosen.includes(d.id)}
                      onChange={(e) => setChosen(e.target.checked ? [...chosen, d.id] : chosen.filter((x) => x !== d.id))} />{d.title}</label>
                  ))}
                </fieldset>
              )}
              <div className="cn-ask-row">
                <Field label="Specialist area">{(id) => (
                  <select id={id} className="lw-input" value={area} onChange={(e) => setArea(e.target.value)}>
                    {Object.entries(q.data.specialisms).map(([k, label]) => <option key={k} value={k}>{label[0].toUpperCase() + label.slice(1)}</option>)}
                  </select>)}</Field>
                <Field label="Depth">{(id) => (
                  <select id={id} className="lw-input" value={depth} onChange={(e) => setDepth(e.target.value === 'deep' ? 'deep' : 'focused')}>
                    <option value="focused">Focused: answers and one challenge round</option>
                    <option value="deep">Deep: the first answer also replies to the objections</option>
                  </select>)}</Field>
                <button type="submit" className="lw-btn lw-btn-primary" disabled={busy || question.trim().length < 10}><Users2 />Ask the Council</button>
              </div>
            </form>
          )}
          <details className="cn-how">
            <summary>How the Council works</summary>
          <ul className="cn-seats-row" aria-label="The usual perspectives">
            {DEFAULT_SEATS.map((s) => <li key={s.role}><RoleMark role={s.role} />{s.label}</li>)}
          </ul>
          <ol className="cn-howto" aria-label="How a session runs">
            {HOW.map(([t, d]) => <li key={t}><b>{t}</b>{d}</li>)}
          </ol>
          </details>
          {error && <ErrorNote>{error}</ErrorNote>}
          <p className="cn-notice">Drafted by the same AI model: they are not independent experts, and a partner reviews them before anyone relies on them.</p>
        </section>
        {q.data?.discovery && (
          <SuggestedQuestions questions={q.data.questions ?? []} more={q.data.more_questions ?? 0} canAct={q.data.can.ask} matterId={matter.id}
            onConvened={open} area={area} depth={depth} refresh={() => qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] })} />
        )}
        <section aria-labelledby="cn-sessions-h" style={{ display: 'grid', gap: 10 }}>
          <h2 className="lw-h2" id="cn-sessions-h" style={{ margin: 0 }}>Sessions</h2>
          {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
            : q.isError ? <ErrorNote>The Council's sessions could not be loaded.</ErrorNote>
              : q.data.reviews.length === 0 ? <EmptyState title="No Council sessions yet">{q.data.can.ask ? 'Ask a question above.' : 'A lawyer on the matter asks the Council.'}</EmptyState>
                : (
                  <ul className="cn-sessions" aria-label="Council sessions">
                    {q.data.reviews.map((r) => {
                      const [label, tone] = r.approval === 'approved' ? ['Approved by a partner', 'verified' as Tone] : PANEL_STATUS[r.status];
                      const step = r.status === 'running' && r.progress?.step && r.progress.step !== 'finished' ? r.progress.step : null;
                      return (
                        <li key={r.id} className="cn-session" aria-label={r.question}>
                          <Link className="cn-session-q" to={`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(r.id)}`}>{r.question}</Link>
                          <StatusPill tone={tone}>{label}</StatusPill>
                          <div className="cn-session-meta">
                            {(r.roles?.length ?? 0) > 0 && (
                              <span className="cn-session-marks" role="img" aria-label={`Perspectives: ${r.roles!.map((x) => x.label).join(', ')}`}>
                                {r.roles!.map((x) => <RoleMark key={x.role} role={x.role} size="sm" />)}</span>
                            )}
                            {step && <span>{`Step ${STEP_NO[step]} of 5 · ${STEP_LABEL[step]}`}</span>}
                            <span>{`${r.area ?? r.specialist} · asked by ${r.created_by ?? 'someone'}, ${day(r.created_at)}${r.version ? ` · version ${r.version}` : ''}`}</span>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
        </section>
      </div>
    </>
  );
}

/** Questions EchoMind found in the matter's own records (council/discovery.py): at most three, each with its sources. A
 *  person convenes one (a Council run asked in its words) or dismisses it with a reason; nothing runs by itself. */
function SuggestedQuestions({ questions, more, canAct, matterId, onConvened, area, depth, refresh }: {
  questions: CouncilQuestion[]; more: number; canAct: boolean; matterId: string; onConvened: (reviewId: string) => void;
  area: string; depth: 'focused' | 'deep'; refresh: () => Promise<unknown>;
}) {
  const [dismissing, setDismissing] = useState<string | null>(null);
  const [why, setWhy] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const base = `/api/legal/matters/${encodeURIComponent(matterId)}/council`;
  const act = async (fn: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await fn(); } catch (e) { setError(e instanceof ApiError ? e.message : 'That did not work. Try again.'); } finally { setBusy(false); }
  };
  const convene = (id: string) => act(async () => {
    const r = await api.post<{ review: { id: string } }>(`${base}/questions/${encodeURIComponent(id)}/convene`, { specialist: area, depth });
    await refresh(); onConvened(r.review.id);
  });
  const dismiss = (id: string) => act(async () => {
    await api.post(`${base}/questions/${encodeURIComponent(id)}/dismiss`, { reason: why.trim() });
    setDismissing(null); setWhy(''); await refresh();
  });
  const look = () => act(async () => { await api.post(`${base}/discover`, {}); await refresh(); });
  return (
    <section className="lw-paper lw-sheet" aria-label="Suggested questions" style={{ display: 'grid', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <h2 className="lw-h2" style={{ margin: 0 }}>Questions worth putting to the Council</h2>
        {canAct && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={look}><RefreshCw />Look again</button>}
      </div>
      <p className="lw-meta" style={{ margin: 0 }}>Found by rules in this matter's records: unresolved inconsistencies, amended clauses, deadlines no one has confirmed. Nothing runs until you convene one, with the specialist area and depth chosen above.</p>
      {questions.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>Nothing found in the records you can see.</p> : (
        <ul className="cn-suggest" aria-label="Questions found">
          {questions.map((x) => (
            <li key={x.id} aria-label={x.title}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <b className="lw-serif" style={{ fontSize: 15.5 }}>{x.title}</b>
                <StatusPill tone={x.priority === 'high' ? 'attention' : 'neutral'}>{x.priority === 'high' ? 'High priority' : 'Medium priority'}</StatusPill>
              </div>
              <span className="lw-meta">{x.reason}</span>
              {x.sources.length > 0 && (
                <div className="lw-meta">From: {x.sources.map((s, i) => (
                  <React.Fragment key={i}>{i > 0 && '; '}{s.href ? <Link className="lw-linklike" to={s.href}>{s.document_title ?? 'a document'}</Link> : s.document_title}</React.Fragment>))}</div>
              )}
              {canAct && (dismissing === x.id ? (
                <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap', marginTop: 6 }}>
                  <Field label="Not worth asking, because" hint="Kept with your name; it is not suggested again until its sources change.">
                    {(id) => <input id={id} className="lw-input" maxLength={500} value={why} onChange={(e) => setWhy(e.target.value)} />}</Field>
                  <button type="button" className="lw-btn" disabled={busy || !why.trim()} onClick={() => dismiss(x.id)}>Dismiss</button>
                  <button type="button" className="lw-btn lw-btn-ghost" onClick={() => { setDismissing(null); setWhy(''); }}>Cancel</button>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                  <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={() => convene(x.id)}><Users2 />Convene the Council</button>
                  <button type="button" className="lw-btn lw-btn-sm" disabled={busy} onClick={() => setDismissing(x.id)}>Dismiss…</button>
                </div>
              ))}
            </li>
          ))}
        </ul>
      )}
      {more > 0 && <p className="lw-meta" style={{ margin: 0 }}>{`${more} more, shown when one of these is convened or dismissed.`}</p>}
      {error && <ErrorNote>{error}</ErrorNote>}
    </section>
  );
}
