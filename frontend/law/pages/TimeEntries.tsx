import React, { useState } from 'react';
import { Check, Download, RotateCcw, Sparkles, X } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { useTimeEntries } from '../api/queries';
import type { TimeEntry } from '../api/types';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';

const today = () => new Date().toISOString().slice(0, 10);
const weekAgo = () => new Date(Date.now() - 6 * 864e5).toISOString().slice(0, 10);
const longDay = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' });

/** Time-entry drafts (S4.3, D-056): suggested from your own audited work on each matter, per day, in 0.1-hour units. You
 *  edit and approve each one; approved entries can be downloaded as CSV. Nothing is sent to a billing system. */
export default function TimeEntries() {
  const [range, setRange] = useState({ from: weekAgo(), to: today() });
  const q = useTimeEntries(range.from, range.to);
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const refresh = () => qc.invalidateQueries({ queryKey: ['time'] });
  const suggest = async () => {
    setBusy(true); setError('');
    try { await api.post('/api/legal/time/suggest', { from: range.from, to: range.to }); await refresh(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The suggestions could not be made.'); } finally { setBusy(false); }
  };
  const entries = (q.data?.entries ?? []).filter((e) => e.status !== 'discarded');
  const approved = entries.filter((e) => e.status === 'approved');
  const total = (list: TimeEntry[]) => (list.reduce((n, e) => n + e.minutes, 0) / 60).toFixed(1);
  return (
    <>
      <Toolbar title="Time">
        <a className="lw-btn lw-btn-sm lw-btn-ghost" href={`/api/legal/time/export.csv?from=${range.from}&to=${range.to}`} aria-disabled={approved.length === 0}
          onClick={(e) => { if (approved.length === 0) e.preventDefault(); }}><Download />Download approved (CSV)</a>
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Suggestions come from your own work recorded in the audit trail: on each matter, each day, the whole minutes from your first to your last action in a sitting (gaps under 15 minutes), plus 6 minutes, in 0.1-hour units. The narrative names the kinds of work, never content. Check, edit and approve each entry; only approved entries are downloaded. Nothing is sent to a billing system.</p>
        <section className="lw-paper lw-sheet" aria-label="Days">
          <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
            <Field label="From">{(id) => <input id={id} type="date" className="lw-input" value={range.from} max={range.to} onChange={(e) => setRange({ ...range, from: e.target.value })} />}</Field>
            <Field label="To">{(id) => <input id={id} type="date" className="lw-input" value={range.to} min={range.from} onChange={(e) => setRange({ ...range, to: e.target.value })} />}</Field>
            <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={suggest}><Sparkles />{busy ? 'Reading your activity…' : 'Suggest from my activity'}</button>
          </div>
          {q.data && <p className="lw-meta lw-tnum" style={{ margin: '8px 0 0' }}>{`${entries.length} ${entries.length === 1 ? 'entry' : 'entries'} · ${total(entries)} h, of which ${total(approved)} h approved · times in ${q.data.timezone}`}</p>}
        </section>
        {error && <ErrorNote>{error}</ErrorNote>}
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>{q.error instanceof ApiError ? q.error.message : 'Your time entries could not be loaded.'}</ErrorNote>
            : entries.length === 0 ? <EmptyState title="No entries for these days">Choose the days and ask for suggestions from your activity.</EmptyState>
              : <ul className="lw-time-list" aria-label="Time entries">{entries.map((e) => <Entry key={e.id} e={e} onChange={refresh} />)}</ul>}
      </div>
    </>
  );
}

function Entry({ e, onChange }: { e: TimeEntry; onChange: () => Promise<unknown> }) {
  const [hours, setHours] = useState((e.minutes / 60).toFixed(1));
  const [narrative, setNarrative] = useState(e.narrative);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  React.useEffect(() => { setHours((e.minutes / 60).toFixed(1)); setNarrative(e.narrative); }, [e.minutes, e.narrative, e.version]);
  const minutes = Math.round(parseFloat(hours) * 60);
  const changed = minutes !== e.minutes || narrative.trim() !== e.narrative;
  const send = async (path: string, body: unknown, method: 'post' | 'patch' = 'post') => {
    setBusy(true); setError('');
    try { await (method === 'patch' ? api.patch(path, body) : api.post(path, body)); await onChange(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The entry could not be saved.'); } finally { setBusy(false); }
  };
  const base = `/api/legal/time/${encodeURIComponent(e.id)}`;
  const locked = e.status === 'approved';
  const label = `${e.matter.number} on ${longDay(e.work_date)}`;
  return (
    <li className="lw-paper lw-sheet lw-time-entry" aria-label={label}>
      <div className="lw-minutes-h">
        <div>
          <b className="lw-tnum">{e.matter.number}</b> <span>{e.matter.title}</span>
          <div className="lw-meta">{`${longDay(e.work_date)} · ${e.matter.client} · ${e.sessions.map((s) => `${s.from}–${s.to}`).join(', ')}`}</div>
        </div>
        <StatusPill tone={locked ? 'verified' : e.edited ? 'accent' : 'neutral'}>{locked ? 'Approved' : e.edited ? 'Edited draft' : 'Suggested draft'}</StatusPill>
      </div>
      <div className="lw-time-fields">
        <Field label="Hours" hint={e.minutes_suggested != null && e.minutes_suggested !== e.minutes ? `Suggested: ${(e.minutes_suggested / 60).toFixed(1)} h` : undefined}>
          {(id) => <input id={id} className="lw-input lw-tnum" type="number" step="0.1" min="0.1" max="24" disabled={locked} value={hours} onChange={(x) => setHours(x.target.value)} />}
        </Field>
        <Field label="Narrative" hint={e.narrative_suggested && e.narrative_suggested !== e.narrative ? `Suggested: ${e.narrative_suggested}` : undefined}>
          {(id) => <textarea id={id} className="lw-input" rows={2} maxLength={1000} disabled={locked} value={narrative} onChange={(x) => setNarrative(x.target.value)} />}
        </Field>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {!locked && changed && <button type="button" className="lw-btn lw-btn-sm" disabled={busy || !(minutes >= 6)}
          onClick={() => send(base, { minutes, narrative: narrative.trim(), version: e.version }, 'patch')}>Save changes</button>}
        {!locked && !changed && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={() => send(`${base}/approve`, { version: e.version })}><Check />Approve</button>}
        {!locked && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={() => send(`${base}/discard`, { version: e.version })}><X />Discard</button>}
        {locked && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy} onClick={() => send(`${base}/reopen`, { version: e.version })}><RotateCcw />Reopen</button>}
      </div>
    </li>
  );
}
