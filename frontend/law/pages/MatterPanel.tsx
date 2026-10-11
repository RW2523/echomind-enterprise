import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Users2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { panelPath, useDocuments, usePanelReviews } from '../api/queries';
import type { PanelReview } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
export const PANEL_STATUS: Record<PanelReview['status'], [string, Tone]> = {
  queued: ['Waiting to start', 'neutral'], running: ['The panel is working', 'accent'], done: ['Ready for a partner', 'attention'],
  needs_facts: ['Stopped: facts needed', 'conflict'], failed: ['Could not finish', 'conflict'], not_allowed: ['No longer allowed', 'neutral'],
};

/** The review panel (S4.5, D-057): one question answered from four angles (deal, risk and compliance, litigation and a
 *  specialist area), each citing the matter's documents, compared by a moderator. Drafted by the same model; a partner
 *  reviews it before anyone relies on it. */
export default function MatterPanel() {
  const matter = useMatter();
  const q = usePanelReviews(matter.id);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [question, setQuestion] = useState('');
  const [area, setArea] = useState('commercial');
  const [chosen, setChosen] = useState<string[]>([]);
  const docs = useDocuments(matter.id);
  const ready = (docs.data ?? []).filter((d) => d.current_version?.index_status === 'indexed');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      const r = await api.post<{ review: { id: string } }>(panelPath(matter.id), { question, specialist: area, documents: chosen });
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'panel'] });
      navigate(`/m/${encodeURIComponent(matter.number)}/panel/${encodeURIComponent(r.review.id)}`);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The question could not be sent.'); } finally { setBusy(false); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Review panel" />
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Four perspectives on one question — a deal lawyer, risk and compliance, a litigator and a specialist — each answering alone from this matter's documents and citing them. A moderator lays out where they agree and differ. If the answer depends on facts the documents do not give, the panel stops and asks you. These are drafted by the same AI model: they are not independent experts, and a partner reviews them before anyone relies on them.</p>
        {q.data?.can.ask && (
          <form className="lw-paper lw-sheet" aria-label="Ask the panel" onSubmit={submit} style={{ display: 'grid', gap: 10 }}>
            <Field label="Question">{(id) => <textarea id={id} className="lw-input" rows={3} maxLength={2000} value={question} onChange={(e) => setQuestion(e.target.value)} />}</Field>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <Field label="Specialist area">{(id) => (
                <select id={id} className="lw-input" value={area} onChange={(e) => setArea(e.target.value)}>
                  {Object.entries(q.data.specialisms).map(([k, label]) => <option key={k} value={k}>{label[0].toUpperCase() + label.slice(1)}</option>)}
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
