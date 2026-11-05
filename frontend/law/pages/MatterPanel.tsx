import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { RefreshCw, Users2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { panelPath, useDocuments, usePanelReviews } from '../api/queries';
import type { CouncilQuestion, PanelReview } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
export const PANEL_STATUS: Record<PanelReview['status'], [string, Tone]> = {
  queued: ['Waiting to start', 'neutral'], running: ['The panel is working', 'accent'], done: ['Ready for a partner', 'attention'],
  needs_facts: ['Stopped: facts needed', 'conflict'], failed: ['Could not finish', 'conflict'], not_allowed: ['No longer allowed', 'neutral'],
};

/** The review panel (S4.5, D-057), grown into the Council (2026-11-05): one question answered from the angles its rules
 *  choose (four by default: deal, risk and compliance, litigation and a specialist area), each citing the matter's
 *  documents; the first answer is challenged, and a chair gives the conclusion and where they differ. Drafted by the same
 *  model; a partner reviews it before anyone relies on it. */
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
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      const r = await api.post<{ review: { id: string } }>(panelPath(matter.id), { question, specialist: area, documents: chosen, depth });
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] });
      navigate(`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(r.review.id)}`);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The question could not be sent.'); } finally { setBusy(false); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Council" />
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Several perspectives on one question, chosen for it — for a challenge, lead counsel, an opposing reading and an evidence investigator; otherwise a deal lawyer, risk and compliance, a litigator and a specialist — each answering alone from this matter's documents and citing them. The first answer is then challenged, and a chair gives the conclusion and where they differ. If the answer depends on facts the documents do not give, the panel stops and asks you. These are drafted by the same AI model: they are not independent experts, and a partner reviews them before anyone relies on them.</p>
        {q.data?.discovery && (
          <SuggestedQuestions questions={q.data.questions ?? []} more={q.data.more_questions ?? 0} canAct={q.data.can.ask} matterId={matter.id}
            onConvened={(id) => navigate(`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(id)}`)} area={area} depth={depth}
            refresh={() => qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] })} />
        )}
        {q.data?.can.ask && (
          <form className="lw-paper lw-sheet" aria-label="Ask the panel" onSubmit={submit} style={{ display: 'grid', gap: 10 }}>
            <Field label="Question">{(id) => <textarea id={id} className="lw-input" rows={3} maxLength={2000} value={question} onChange={(e) => setQuestion(e.target.value)} />}</Field>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <Field label="Specialist area">{(id) => (
                <select id={id} className="lw-input" value={area} onChange={(e) => setArea(e.target.value)}>
                  {Object.entries(q.data.specialisms).map(([k, label]) => <option key={k} value={k}>{label[0].toUpperCase() + label.slice(1)}</option>)}
                </select>)}</Field>
              <Field label="Depth">{(id) => (
                <select id={id} className="lw-input" value={depth} onChange={(e) => setDepth(e.target.value === 'deep' ? 'deep' : 'focused')}>
                  <option value="focused">Focused: answers and one challenge round</option>
                  <option value="deep">Deep: the first answer also replies to the objections</option>
                </select>)}</Field>
              <button type="submit" className="lw-btn lw-btn-primary" disabled={busy || question.trim().length < 10}><Users2 />Ask the panel</button>
            </div>
            {ready.length > 1 && (
              <fieldset className="lw-panel-docs">
                <legend className="lw-meta">Only these documents (optional; all of the matter's otherwise)</legend>
                {ready.map((d) => (
                  <label key={d.id} className="lw-check"><input type="checkbox" checked={chosen.includes(d.id)}
                    onChange={(e) => setChosen(e.target.checked ? [...chosen, d.id] : chosen.filter((x) => x !== d.id))} />{d.title}</label>
                ))}
              </fieldset>
            )}
          </form>
        )}
        {error && <ErrorNote>{error}</ErrorNote>}
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>The panel reviews could not be loaded.</ErrorNote>
            : q.data.reviews.length === 0 ? <EmptyState title="No panel reviews yet">{q.data.can.ask ? 'Ask a question above.' : 'A lawyer on the matter asks the panel.'}</EmptyState>
              : (
                <ul className="lw-check-lines lw-paper lw-sheet" aria-label="Panel reviews">
                  {q.data.reviews.map((r) => {
                    const [label, tone] = r.approval === 'approved' ? ['Approved by a partner', 'verified' as Tone] : PANEL_STATUS[r.status];
                    return (
                      <li key={r.id} aria-label={r.question}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                          <Link to={`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(r.id)}`}><b>{r.question}</b></Link>
                          <StatusPill tone={tone}>{label}</StatusPill>
                        </div>
                        <span className="lw-meta">{`${r.area ?? r.specialist} · asked by ${r.created_by ?? 'someone'}, ${day(r.created_at)}${r.version ? ` · version ${r.version}` : ''}`}</span>
                      </li>
                    );
                  })}
                </ul>
              )}
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
    <section className="lw-paper lw-sheet" aria-label="Suggested questions">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <h2 className="lw-h2" style={{ margin: 0 }}>Questions worth putting to the Council</h2>
        {canAct && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={look}><RefreshCw />Look again</button>}
      </div>
      <p className="lw-meta" style={{ margin: '2px 0 8px' }}>Found by rules in this matter's records: unresolved inconsistencies, amended clauses, deadlines no one has confirmed. Nothing runs until you convene one, with the specialist area and depth chosen below.</p>
      {questions.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>Nothing found in the records you can see.</p> : (
        <ul className="lw-check-lines" aria-label="Questions found">
          {questions.map((x) => (
            <li key={x.id} aria-label={x.title}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <b>{x.title}</b>
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
      {more > 0 && <p className="lw-meta" style={{ margin: '6px 0 0' }}>{`${more} more, shown when one of these is convened or dismissed.`}</p>}
      {error && <ErrorNote>{error}</ErrorNote>}
    </section>
  );
}
