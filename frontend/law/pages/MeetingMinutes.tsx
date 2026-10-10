import React, { useState } from 'react';
import { Link } from 'react-router';
import { Check, FileOutput, PenLine, Sparkles } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, meetingsPath, useMe, useMinutes } from '../api/queries';
import type { MeetingDetail, MinutesItem, MinutesVersion } from '../api/types';
import { useMatter } from '../app/matter';
import { ErrorNote, StatusPill, type Tone } from '../ui/primitives';

const KIND: Record<MinutesItem['kind'], [string, Tone]> = {
  decision: ['Decided', 'verified'], action: ['Action', 'accent'], unresolved: ['Open', 'attention'], discussion: ['Discussed', 'neutral'],
};
const clock = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '');
const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** Minutes: drafted from the whole transcript, every item tied to the words it rests on; changed only
 *  as a new version; approved by a second lawyer for exactly the content they read; then filed. */
export default function MeetingMinutes({ m, canDraft, canApprove }: { m: MeetingDetail; canDraft: boolean; canApprove: boolean }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const me = useMe();
  const versions = useMinutes(matter.id, m.id, m.status === 'ended');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [editing, setEditing] = useState(false);
  const base = `${meetingsPath(matter.id)}/${encodeURIComponent(m.id)}/minutes`;
  const latest = versions.data?.[0];
  const approved = versions.data?.find((v) => v.status === 'approved');
  const refresh = () => qc.invalidateQueries({ queryKey: [...keys.meeting(matter.id, m.id), 'minutes'] });
  const run = async (what: string, fn: () => Promise<unknown>) => {
    setBusy(what); setError('');
    try { await fn(); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'That could not be done.'); }
    finally { setBusy(''); }
  };

  if (m.status !== 'ended') {
    return <section className="lw-paper lw-sheet" aria-label="Minutes"><p className="lw-meta" style={{ margin: 0 }}>Minutes are drafted after the meeting ends, from its whole transcript.</p></section>;
  }
  const mine = latest && me.data && latest.created_by === me.data.user.id;
  return (
    <section className="lw-paper lw-sheet" aria-label="Minutes" aria-busy={!!busy}>
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>Minutes</h2>
          {latest && <p className="lw-meta" style={{ margin: '2px 0 0' }}>{`Version ${latest.version} · ${latest.origin === 'model' ? 'drafted from the transcript' : `edited by ${latest.created_by_name}`} · ${day(latest.created_at)}`}</p>}
        </div>
        {latest && <StatusPill tone={latest.status === 'approved' ? 'verified' : 'attention'}>{latest.status === 'approved' ? 'Approved' : 'Draft — not approved'}</StatusPill>}
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {!latest && (
        <div style={{ display: 'grid', gap: 8 }}>
          <p className="lw-meta" style={{ margin: 0 }}>Drafted from the whole transcript as corrected. Every item cites the words it rests on, which are checked; nothing is final until a second lawyer approves it.</p>
          {canDraft && <div><button type="button" className="lw-btn lw-btn-primary" disabled={!!busy} onClick={() => run('draft', () => api.post(`${base}/draft`))}>
            <Sparkles />{busy === 'draft' ? 'Drafting from the whole transcript…' : 'Draft minutes'}</button></div>}
        </div>
      )}
      {latest && !editing && <MinutesView v={latest} />}
      {latest && editing && <MinutesEditor v={latest} base={base} onClose={() => setEditing(false)} onSaved={async () => { setEditing(false); await refresh(); }} />}
      {latest && !editing && (
        <div className="lw-minutes-actions">
          {canDraft && latest.status !== 'superseded' && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing(true)}><PenLine />Edit as a new version</button>}
          {canDraft && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={!!busy} onClick={() => run('draft', () => api.post(`${base}/draft`))}>
            <Sparkles />{busy === 'draft' ? 'Drafting…' : 'Draft again'}</button>}
          {approved && (approved.document_id
            ? <Link className="lw-btn lw-btn-sm" to={`/m/${matter.number}/documents/${encodeURIComponent(approved.document_id)}/read`}><FileOutput />Open the filed minutes</Link>
            : canDraft && <button type="button" className="lw-btn lw-btn-sm" disabled={!!busy} onClick={() => run('file', () => api.post(`${base}/${approved.id}/file`))}>
              <FileOutput />{busy === 'file' ? 'Filing…' : 'File as a matter document'}</button>)}
        </div>
      )}
      {latest && latest.status === 'draft' && !editing && (
        <Review v={latest} base={base} canApprove={canApprove} isAuthor={!!mine} onDone={refresh} />
      )}
      {latest && latest.decisions.length > 0 && (
        <ol className="lw-minutes-decisions" aria-label="Review decisions">
          {latest.decisions.map((d, i) => (
            <li key={i} className="lw-meta">{`${d.decision === 'approved' ? 'Approved' : 'Changes requested'} by ${d.by ?? 'someone'}, ${day(d.at)}`}
              {d.reason ? ` — ${d.reason}` : ''}{d.decision === 'approved' ? ` · content ${d.content_sha256.slice(0, 12)}…` : ''}</li>
          ))}
        </ol>
      )}
    </section>
  );
}

function MinutesView({ v }: { v: MinutesVersion }) {
  const c = v.content;
  const counts = c.summary;
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <p className="lw-meta" style={{ margin: 0 }}>
        {`${counts.decisions ?? 0} decided · ${counts.actions ?? 0} actions · ${counts.unresolveds ?? 0} open · ${counts.discussions ?? 0} discussed`}
        {counts.not_verified ? ` · ${counts.not_verified} not verified against the transcript` : ''}
        {v.inputs.failed_batches ? ` · ${v.inputs.failed_batches} part(s) of the transcript could not be drafted — check them` : ''}
      </p>
      <p className="lw-meta" style={{ margin: 0 }}>Present: {c.attendees.map((a) => `${a.name} (${a.role})`).join('; ')}</p>
      {c.sections.map((sec, k) => (
        <section key={k} aria-label={sec.title} className="lw-minutes-section">
          <h3 className="lw-h3">{sec.title}</h3>
          {sec.items.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>Nothing minuted.</p> : (
            <ul className="lw-minutes-items">
              {sec.items.map((it) => <Item key={it.id} it={it} />)}
            </ul>
          )}
        </section>
      ))}
      {c.gaps.length > 0 && <p className="lw-meta" style={{ margin: 0 }}>{`Not transcribed at ${c.gaps.map(clock).join(', ')}: these minutes cannot cover what was said there.`}</p>}
    </div>
  );
}

function Item({ it }: { it: MinutesItem }) {
  const [label, tone] = KIND[it.kind];
  const pins = it.sources.filter((s) => s.verification !== 'failed');
  return (
    <li data-status={it.status} aria-label={`${label}: ${it.text}`}>
      <div className="lw-minutes-item-h">
        <StatusPill tone={tone}>{label}</StatusPill>
        <span className="lw-serif">{it.text}</span>
      </div>
      {it.kind === 'action' && <div className="lw-meta">{`Owner: ${it.owner ?? 'not stated'} · By: ${it.deadline ?? 'not stated'}`}</div>}
      <div className="lw-meta">
        {pins.length > 0 && <span>{pins.map((s) => `${clock(s.time)} ${s.speaker ?? ''}`.trim()).join(' · ')}</span>}
        {it.status === 'not_verified' && <span className="lw-minutes-flag"> · Not verified against the transcript — check before relying on it</span>}
        {it.status === 'added_by_person' && <span> · Added by {it.edited_by}</span>}
        {it.edited_by && it.status !== 'added_by_person' && <span> · Edited by {it.edited_by}</span>}
        {it.by_rule && <span> · Taken word for word from the transcript</span>}
      </div>
      {it.sources.length > 0 && (
        <details className="lw-minutes-sources">
          <summary className="lw-meta">The words it rests on</summary>
          <ul>{it.sources.map((s, i) => (
            <li key={i} className="lw-meta" data-verification={s.verification}>{`${clock(s.time)} ${s.speaker ?? ''}: “${s.quote}”`}
              {s.verification === 'failed' ? ' — not found in that line' : ''}</li>
          ))}</ul>
        </details>
      )}
    </li>
  );
}

function Review({ v, base, canApprove, isAuthor, onDone }: { v: MinutesVersion; base: string; canApprove: boolean; isAuthor: boolean; onDone: () => void }) {
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  if (!canApprove) return <p className="lw-meta" style={{ margin: 0 }}>Waiting for a lead, attorney or reviewer on this matter to approve.</p>;
  if (isAuthor) return <p className="lw-meta" style={{ margin: 0 }}>You made this version, so another lawyer approves it.</p>;
  const decide = async (decision: 'approved' | 'changes_requested') => {
    setBusy(true); setError('');
    try { await api.post(`${base}/${v.id}/review`, { decision, content_sha256: v.content_sha256, reason: reason.trim() || null }); setReason(''); onDone(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The decision could not be saved.'); }
    finally { setBusy(false); }
  };
  return (
    <div className="lw-minutes-review" role="group" aria-label="Review these minutes">
      <p className="lw-meta" style={{ margin: 0 }}>{`Approving binds exactly this version (content ${v.content_sha256.slice(0, 12)}…). Any later change needs approval again.`}</p>
      <label className="lw-sr" htmlFor="minutes-reason">What should change?</label>
      <input id="minutes-reason" className="lw-input lw-input-sm" placeholder="What should change? (for a request)" value={reason} onChange={(e) => setReason(e.target.value)} />
      <div style={{ display: 'flex', gap: 6 }}>
        <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={() => decide('approved')}><Check />Approve these minutes</button>
        <button type="button" className="lw-btn lw-btn-sm" disabled={busy || reason.trim().length < 3} onClick={() => decide('changes_requested')}>Request changes</button>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}

function MinutesEditor({ v, base, onClose, onSaved }: { v: MinutesVersion; base: string; onClose: () => void; onSaved: () => void }) {
  const items = v.content.sections.flatMap((s) => s.items);
  const [edits, setEdits] = useState<Record<string, { text?: string; owner?: string; deadline?: string; remove?: boolean }>>({});
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (id: string, patch: object) => setEdits((e) => ({ ...e, [id]: { ...e[id], ...patch } }));
  const save = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    const changes = Object.entries(edits).map(([id, ch]) => ({ id, ...ch })).filter((x) => Object.keys(x).length > 1);
    try { await api.post(`${base}/${v.id}/edit`, { content_sha256: v.content_sha256, items: changes, add: [], reason: reason.trim() }); onSaved(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The changes could not be saved.'); setBusy(false); }
  };
  return (
    <form onSubmit={save} style={{ display: 'grid', gap: 10 }} aria-label="Edit the minutes">
      <p className="lw-meta" style={{ margin: 0 }}>Your changes make a new version. Each item keeps the words it rests on; what you change is marked with your name.</p>
      {items.map((it) => (
        <fieldset key={it.id} className="lw-minutes-edit" data-removed={edits[it.id]?.remove || undefined}>
          <legend className="lw-meta">{KIND[it.kind][0]}</legend>
          <label className="lw-sr" htmlFor={`t-${it.id}`}>Text of {it.id}</label>
          <textarea id={`t-${it.id}`} className="lw-input lw-serif" rows={2} defaultValue={it.text} onChange={(e) => set(it.id, { text: e.target.value })} />
          {it.kind === 'action' && (
            <div style={{ display: 'flex', gap: 6 }}>
              <input aria-label={`Owner of ${it.id}`} className="lw-input lw-input-sm" defaultValue={it.owner ?? ''} placeholder="Owner" onChange={(e) => set(it.id, { owner: e.target.value })} />
              <input aria-label={`Deadline of ${it.id}`} className="lw-input lw-input-sm" defaultValue={it.deadline ?? ''} placeholder="By" onChange={(e) => set(it.id, { deadline: e.target.value })} />
            </div>
          )}
          <label className="lw-meta"><input type="checkbox" onChange={(e) => set(it.id, { remove: e.target.checked })} /> Remove this item</label>
        </fieldset>
      ))}
      <label className="lw-meta" htmlFor="edit-reason">Why these changes?</label>
      <input id="edit-reason" className="lw-input" value={reason} onChange={(e) => setReason(e.target.value)} />
      {error && <ErrorNote>{error}</ErrorNote>}
      <div style={{ display: 'flex', gap: 6 }}>
        <button type="submit" className="lw-btn lw-btn-primary" disabled={busy || reason.trim().length < 3}>Save as a new version</button>
        <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
      </div>
    </form>
  );
}
