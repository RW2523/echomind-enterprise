import React, { useState } from 'react';
import { Check, RotateCcw, X } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { regulatoryPath, useMatterRegulatory } from '../api/queries';
import type { RegulatoryMatch } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill } from '../ui/primitives';
import { licenceLabel, longDate, OfficialTitle, UpdateMeta } from './Regulatory';

const DECIDERS = ['lead', 'attorney', 'paralegal', 'reviewer'];

/** A matter's regulatory updates (S4.4, D-068): the new legislation and rules matched to it, with the words that
 *  matched. A lawyer marks each reviewed, or not relevant with a reason; the decision is kept with who and when. */
export default function MatterRegulatory() {
  const matter = useMatter();
  const q = useMatterRegulatory(matter.id);
  const canDecide = DECIDERS.includes(matter.my_role ?? '');
  const open = (q.data?.matches ?? []).filter((m) => m.status === 'new');
  const decided = (q.data?.matches ?? []).filter((m) => m.status !== 'new');
  return (
    <>
      <Toolbar crumb={matter.number} title="Regulatory" />
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>
          {`New legislation and rules that may concern this matter: its jurisdiction (${matter.jurisdiction || 'not set'}) matches the update's, and a word from its practice area (${matter.practice_area || 'not set'}) appears in the update's title, summary or agencies. A match is a prompt to look, not advice; mark each reviewed, or not relevant with a reason.`}
        </p>
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>{q.error instanceof ApiError ? q.error.message : 'The updates could not be loaded.'}</ErrorNote>
            : q.data.matches.length === 0 ? <EmptyState title="Nothing matched yet">Updates are imported each morning and matched to open matters.</EmptyState>
              : (
                <>
                  <h2 className="lw-h2">{`To review (${open.length})`}</h2>
                  {open.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>Every update has been decided.</p>
                    : open.map((m) => <MatchCard key={m.update.id} m={m} canDecide={canDecide} />)}
                  {decided.length > 0 && <h2 className="lw-h2">{`Decided (${decided.length})`}</h2>}
                  {decided.map((m) => <MatchCard key={m.update.id} m={m} canDecide={canDecide} />)}
                </>
              )}
      </div>
    </>
  );
}

function MatchCard({ m: initial, canDecide }: { m: RegulatoryMatch; canDecide: boolean }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [m, setM] = useState(initial);
  const [asking, setAsking] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  React.useEffect(() => setM(initial), [initial]);
  const titleId = `reg-${m.update.id}`;
  const decide = async (status: RegulatoryMatch['status'], note?: string) => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ match: Omit<RegulatoryMatch, 'update'> }>(`${regulatoryPath(matter.id)}/${encodeURIComponent(m.update.id)}`,
        { status, ...(note ? { note } : {}), version: m.version });
      setM({ ...m, ...r.match }); setAsking(false); setReason('');
      void qc.invalidateQueries({ queryKey: ['regulatory'] });
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The decision could not be saved.'); } finally { setBusy(false); }
  };
  return (
    <article className="lw-paper lw-sheet lw-reg-item" aria-labelledby={titleId}>
      <UpdateMeta u={m.update} />
      <span id={titleId}><OfficialTitle u={m.update} /></span>
      {m.update.summary && <p className="lw-reg-summary">{m.update.summary}</p>}
      <p className="lw-meta" style={{ margin: 0 }}>{`Matched on: ${m.terms.join(', ')}`}</p>
      {m.status !== 'new' && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <StatusPill tone={m.status === 'reviewed' ? 'verified' : 'neutral'}>
            {`${m.status === 'reviewed' ? 'Reviewed' : 'Not relevant'} · ${m.decided_by ?? 'someone'}${m.decided_at ? ` · ${longDate(m.decided_at.slice(0, 10))}` : ''}`}
          </StatusPill>
          {m.note && <span className="lw-meta">{m.note}</span>}
        </div>
      )}
      {canDecide && (m.status === 'new' ? (
        asking ? (
          <div style={{ display: 'grid', gap: 8 }}>
            <label className="lw-meta" htmlFor={`${titleId}-why`}>Why it does not concern this matter</label>
            <textarea id={`${titleId}-why`} className="lw-input" rows={2} maxLength={1000} value={reason} onChange={(e) => setReason(e.target.value)} />
            <div style={{ display: 'flex', gap: 8 }}>
              <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy || !reason.trim()} onClick={() => decide('not_relevant', reason.trim())}>Save: not relevant</button>
              <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setAsking(false)}>Cancel</button>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={() => decide('reviewed')}><Check />Mark reviewed</button>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={() => setAsking(true)}><X />Not relevant…</button>
          </div>
        )
      ) : (
        <div><button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={() => decide('new')}><RotateCcw />Reopen</button></div>
      ))}
      {error && <ErrorNote>{error}</ErrorNote>}
      <span className="lw-meta" style={{ fontSize: 12 }}>{licenceLabel(m.update)}</span>
    </article>
  );
}
