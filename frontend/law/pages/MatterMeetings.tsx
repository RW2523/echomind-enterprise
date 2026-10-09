import React, { useState } from 'react';
import { Link } from 'react-router';
import { CalendarPlus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, meetingsPath, useMeetings } from '../api/queries';
import type { MeetingDetail, MeetingKind } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';

export const MEETING_KINDS: [MeetingKind, string][] = [['client_call', 'Client call'], ['internal', 'Internal'], ['negotiation', 'Negotiation'],
  ['interview', 'Interview'], ['board', 'Board meeting'], ['hearing', 'Hearing'], ['other', 'Other']];
export const kindLabel = (k: string) => MEETING_KINDS.find(([v]) => v === k)?.[1] ?? k;
const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—');

/** The matter's meetings. Recording needs everyone's consent first, so it is planned here. */
export default function MatterMeetings() {
  const matter = useMatter();
  const meetings = useMeetings(matter.id);
  const [creating, setCreating] = useState(false);
  const canPlan = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  return (
    <>
      <Toolbar crumb={matter.number} title="Meetings">
        {canPlan && <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><CalendarPlus />New meeting</button>}
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>A meeting can be recorded only once every participant has consented. Anyone on the matter can record a withdrawal, which stops recording at once.</p>
        {meetings.isPending ? <div className="lw-paper lw-sheet" aria-busy="true"><Skeleton /><Skeleton /></div>
          : meetings.isError ? <ErrorNote>The meetings could not be loaded.</ErrorNote>
          : meetings.data.length === 0 ? (
            <EmptyState title="No meetings yet" action={canPlan ? <button type="button" className="lw-btn lw-btn-primary" onClick={() => setCreating(true)}><CalendarPlus />Plan a meeting</button> : undefined}>
              Plan a meeting to list who takes part and record their consent before anything is captured.
            </EmptyState>
          ) : (
            <section className="lw-paper" aria-label="Meetings">
              <table className="lw-table">
                <thead><tr><th>Meeting</th><th>Kind</th><th>Scheduled</th><th>Consent</th><th>Status</th></tr></thead>
                <tbody>
                  {meetings.data.map((m) => {
                    const all = m.consent.needed > 0 && m.consent.granted === m.consent.needed;
                    return (
                      <tr key={m.id}>
                        <td><Link className="lw-linklike" to={m.id}>{m.title}</Link></td>
                        <td>{kindLabel(m.kind)}</td>
                        <td className="lw-meta">{when(m.scheduled_at)}</td>
                        <td><StatusPill tone={all ? 'verified' : 'attention'}>{m.consent.granted} of {m.consent.needed} consented</StatusPill></td>
                        <td>{m.status === 'ended' ? <StatusPill tone="neutral">Ended</StatusPill> : <StatusPill tone="accent">Planned</StatusPill>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          )}
      </div>
      {creating && <NewMeetingDialog matterId={matter.id} onClose={() => setCreating(false)} />}
    </>
  );
}

function NewMeetingDialog({ matterId, onClose }: { matterId: string; onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ title: '', kind: 'client_call' as MeetingKind, scheduled_at: '' });
  const [error, setError] = useState('');
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try {
      await api.post<{ meeting: MeetingDetail }>(meetingsPath(matterId), {
        title: form.title.trim(), kind: form.kind, scheduled_at: form.scheduled_at ? new Date(form.scheduled_at).toISOString() : null });
      await qc.invalidateQueries({ queryKey: keys.meetings(matterId) });
      onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The meeting could not be created.'); }
  };
  return (
    <Dialog title="Plan a meeting" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Title">{(id) => <input id={id} className="lw-input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />}</Field>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Kind">{(id) => <select id={id} className="lw-input" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as MeetingKind })}>
            {MEETING_KINDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          <Field label="Scheduled (optional)">{(id) => <input id={id} type="datetime-local" className="lw-input" value={form.scheduled_at} onChange={(e) => setForm({ ...form, scheduled_at: e.target.value })} />}</Field>
        </div>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.title.trim()}>Create</button>
        </div>
      </form>
    </Dialog>
  );
}
