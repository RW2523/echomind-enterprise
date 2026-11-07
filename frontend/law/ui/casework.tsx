/**
 * Case management on a matter's overview (2026-11-07): where it stands in its track, what has happened on it, and its plan.
 * Everything here is written by the matter's people and kept: an update is never changed (a correction is a new update that
 * names it), each task change is recorded, and a stage move is posted to the updates too. Nothing is generated.
 */
import React, { useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { CalendarClock, Check, ChevronRight, ListChecks, MessageSquarePlus, Pencil, Plus, RotateCcw } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useDocuments, useMemberships, useTasks, useTracks, useUpdates } from '../api/queries';
import type { CaseTask, CaseUpdate, Matter, TaskKind, TrackDef, UpdateKind } from '../api/types';
import { Dialog, EmptyState, ErrorNote, Field } from './primitives';

export const UPDATE_KIND: Record<UpdateKind, string> = {
  note: 'Note', call: 'Call', meeting: 'Meeting', filing: 'Filing', hearing: 'Hearing', order: 'Court order',
  instruction: 'Client instruction', milestone: 'Milestone', stage: 'Stage',
};
export const TASK_KIND: Record<TaskKind, string> = { task: 'Task', key_date: 'Key date', hearing: 'Hearing', filing: 'Filing deadline' };
const POSTABLE = (Object.keys(UPDATE_KIND) as UpdateKind[]).filter((k) => k !== 'stage');
export const day = (iso?: string | null) => (iso ? new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '');
const today = () => new Date().toISOString().slice(0, 10);
const path = (m: Matter, rest: string) => `/api/legal/matters/${encodeURIComponent(m.id)}/${rest}`;
const can = (m: Matter, roles: string[]) => m.status === 'open' && roles.includes(m.my_role ?? '');

export function trackOf(tracks: TrackDef[] | undefined, m: Pick<Matter, 'track'>): TrackDef | undefined {
  return tracks?.find((t) => t.key === (m.track ?? 'advisory'));
}

/** The track's stages with the current one marked; the lawyers on the matter move it on. */
export function StageStepper({ matter }: { matter: Matter }) {
  const tracks = useTracks();
  const qc = useQueryClient();
  const [moving, setMoving] = useState(false);
  const [error, setError] = useState('');
  const track = trackOf(tracks.data, matter);
  if (!track) return null;
  const index = Math.max(0, track.stages.findIndex((s) => s.key === matter.stage));
  const next = track.stages[index + 1];
  const mayMove = can(matter, ['lead', 'attorney']);
  const move = async (stage: string, note?: string, trackKey?: string, on?: string) => {
    setError('');
    try {
      await api.post(path(matter, 'stage'), { stage, version: matter.version, ...(note ? { note } : {}), ...(trackKey ? { track: trackKey } : {}),
                                              ...(on && on !== today() ? { effective_on: on } : {}) });
      await Promise.all([qc.invalidateQueries({ queryKey: keys.matters }), qc.invalidateQueries({ queryKey: keys.matter(matter.id) }),
                         qc.invalidateQueries({ queryKey: keys.pipeline })]);
      setMoving(false);
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The stage could not be changed.'); }
  };
  return (
    <section className="lw-paper lw-sheet" aria-labelledby="stage-title" style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <h2 className="lw-h2" id="stage-title" style={{ margin: 0 }}>Stage</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{track.label} track{matter.stage_since ? ` · at ${track.stages[index].label} since ${day(matter.stage_since)}` : ''}</p>
        </div>
        {mayMove && (
          <div style={{ display: 'flex', gap: 8 }}>
            {next && <button type="button" className="lw-btn lw-btn-primary" onClick={() => void move(next.key)}>Move to {next.label}<ChevronRight /></button>}
            <button type="button" className="lw-btn" onClick={() => setMoving(true)}>Change stage…</button>
          </div>
        )}
      </div>
      <ol className="lw-stages" aria-label={`${track.label} stages`}>
        {track.stages.map((s, i) => (
          <li key={s.key} data-state={i < index ? 'done' : i === index ? 'current' : 'next'} aria-current={i === index ? 'step' : undefined}>
            <span className="lw-stage-dot" aria-hidden>{i < index ? <Check /> : i + 1}</span>
            <span className="lw-stage-label">{s.label}</span>
          </li>
        ))}
      </ol>
      {error && <ErrorNote>{error}</ErrorNote>}
      {moving && <MoveDialog matter={matter} tracks={tracks.data ?? []} onClose={() => setMoving(false)} onMove={move} />}
    </section>
  );
}

function MoveDialog({ matter, tracks, onClose, onMove }: { matter: Matter; tracks: TrackDef[]; onClose: () => void;
  onMove: (stage: string, note?: string, track?: string, on?: string) => Promise<void> }) {
  const [trackKey, setTrackKey] = useState<string>(matter.track ?? 'advisory');
  const track = tracks.find((t) => t.key === trackKey)!;
  const [stage, setStage] = useState<string>(trackKey === matter.track ? matter.stage ?? track.stages[0].key : track.stages[0].key);
  const [note, setNote] = useState('');
  const [on, setOn] = useState(today());
  const same = trackKey === matter.track && stage === matter.stage;
  return (
    <Dialog title="Change the stage" onClose={onClose}
      footer={<><button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
        <button type="button" className="lw-btn lw-btn-primary" disabled={same}
          onClick={() => void onMove(stage, note.trim() || undefined, trackKey !== matter.track ? trackKey : undefined, on)}>Move</button></>}>
      <div style={{ display: 'grid', gap: 12 }}>
        <Field label="Track">{(id) => (
          <select id={id} className="lw-input" value={trackKey} onChange={(e) => { setTrackKey(e.target.value); setStage(tracks.find((t) => t.key === e.target.value)!.stages[0].key); }}>
            {tracks.map((t) => <option key={t.key} value={t.key}>{t.label}</option>)}
          </select>)}</Field>
        <Field label="Stage">{(id) => (
          <select id={id} className="lw-input" value={stage} onChange={(e) => setStage(e.target.value)}>
            {track.stages.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>)}</Field>
        <Field label="Moved on" hint="Today, or the day it happened, for a case brought in with its history.">{(id) => (
          <input id={id} type="date" className="lw-input" max={today()} value={on} onChange={(e) => setOn(e.target.value)} />)}</Field>
        <Field label="Note for the updates" hint="Optional. Without one, the update says which stage it moved from and to.">{(id) => (
          <textarea id={id} className="lw-input" rows={3} maxLength={1000} value={note} onChange={(e) => setNote(e.target.value)} />)}</Field>
      </div>
    </Dialog>
  );
}

/** What has happened on the matter, newest first. A corrected update is shown under its correction, struck through. */
export function CaseUpdates({ matter }: { matter: Matter }) {
  const q = useUpdates(matter.id);
  const [posting, setPosting] = useState(false);
  const [correcting, setCorrecting] = useState<CaseUpdate | null>(null);
  const mayPost = can(matter, ['lead', 'attorney', 'paralegal']);
  const all = q.data ?? [];
  const byId = useMemo(() => new Map(all.map((u) => [u.id, u])), [all]);
  const shown = all.filter((u) => !u.corrected_by);
  return (
    <section className="lw-paper lw-sheet" aria-labelledby="updates-title" style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <h2 className="lw-h2" id="updates-title" style={{ margin: 0 }}>Case updates</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>What has happened on this matter, posted by the people working on it. Updates are kept as written; a mistake is corrected, not erased.</p>
        </div>
        {mayPost && <button type="button" className="lw-btn lw-btn-primary" onClick={() => setPosting(true)}><MessageSquarePlus />Post an update</button>}
      </div>
      {q.isPending ? <p className="lw-meta" style={{ margin: 0 }}>Loading…</p>
        : q.isError ? <ErrorNote>The updates could not be loaded.</ErrorNote>
          : shown.length === 0 ? <EmptyState title="No updates yet">{mayPost ? 'Post the first one: a call, a filing, a hearing, the client\'s instructions.' : 'The matter\'s lawyers post them.'}</EmptyState>
            : (
              <ol className="lw-feed" aria-label="Updates">
                {shown.map((u) => {
                  const before = u.corrects_id ? byId.get(u.corrects_id) : undefined;
                  return (
                    <li key={u.id} data-kind={u.kind}>
                      <div className="lw-feed-head">
                        <span className="lw-feed-kind">{UPDATE_KIND[u.kind]}</span>
                        <time className="lw-meta" dateTime={u.occurred_on}>{day(u.occurred_on)}</time>
                        <span className="lw-meta">· {u.author}{u.corrects_id ? ' · correction' : ''}</span>
                        {mayPost && u.kind !== 'stage' && (
                          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" style={{ marginLeft: 'auto' }} onClick={() => setCorrecting(u)}
                            aria-label={`Correct the ${UPDATE_KIND[u.kind].toLowerCase()} of ${day(u.occurred_on)}`}><Pencil />Correct</button>
                        )}
                      </div>
                      <p className="lw-feed-body">{u.body}</p>
                      {u.document_title && <p className="lw-meta" style={{ margin: 0 }}>Document: {u.document_title}</p>}
                      {before && <p className="lw-meta lw-feed-was"><s>{before.body}</s> <span>(as first posted by {before.author})</span></p>}
                    </li>
                  );
                })}
              </ol>
            )}
      {posting && <PostUpdateDialog matter={matter} onClose={() => setPosting(false)} />}
      {correcting && <PostUpdateDialog matter={matter} correcting={correcting} onClose={() => setCorrecting(null)} />}
    </section>
  );
}

function PostUpdateDialog({ matter, correcting, onClose }: { matter: Matter; correcting?: CaseUpdate; onClose: () => void }) {
  const qc = useQueryClient();
  const docs = useDocuments(matter.id);
  const [kind, setKind] = useState<UpdateKind>(correcting?.kind ?? 'note');
  const [body, setBody] = useState(correcting?.body ?? '');
  const [on, setOn] = useState(correcting?.occurred_on ?? today());
  const [doc, setDoc] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setBusy(true);
    try {
      if (correcting) await api.post(path(matter, `updates/${encodeURIComponent(correcting.id)}/correct`), { body, kind, occurred_on: on });
      else await api.post(path(matter, 'updates'), { kind, body, occurred_on: on, ...(doc ? { document_id: doc } : {}) });
      await Promise.all([qc.invalidateQueries({ queryKey: keys.updates(matter.id) }), qc.invalidateQueries({ queryKey: keys.pipeline })]);
      onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The update could not be posted.'); }
    finally { setBusy(false); }
  };
  return (
    <Dialog title={correcting ? 'Correct the update' : 'Post an update'} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        {correcting && <p className="lw-meta" style={{ margin: 0 }}>The original stays on record, struck through under the correction.</p>}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="What happened">{(id) => (
            <select id={id} className="lw-input" value={kind} onChange={(e) => setKind(e.target.value as UpdateKind)}>
              {POSTABLE.map((k) => <option key={k} value={k}>{UPDATE_KIND[k]}</option>)}
            </select>)}</Field>
          <Field label="On">{(id) => <input id={id} type="date" className="lw-input" max={today()} value={on} onChange={(e) => setOn(e.target.value)} />}</Field>
        </div>
        <Field label="Update">{(id) => <textarea id={id} className="lw-input" rows={5} maxLength={4000} required value={body} onChange={(e) => setBody(e.target.value)}
          placeholder="For example: Spoke to the client; they accept the revised delivery date and want the amendment signed this week." />}</Field>
        {!correcting && (docs.data?.length ?? 0) > 0 && (
          <Field label="Document" hint="Optional: the document this update is about.">{(id) => (
            <select id={id} className="lw-input" value={doc} onChange={(e) => setDoc(e.target.value)}>
              <option value="">None</option>
              {docs.data!.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}
            </select>)}</Field>
        )}
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!body.trim() || busy}>{correcting ? 'Post the correction' : 'Post'}</button>
        </div>
      </form>
    </Dialog>
  );
}

/** The plan of work: tasks, key dates, hearings and filing deadlines, entered by a person, grouped by stage. */
export function CasePlan({ matter }: { matter: Matter }) {
  const q = useTasks(matter.id);
  const tracks = useTracks();
  const qc = useQueryClient();
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<CaseTask | null>(null);
  const [showDone, setShowDone] = useState(false);
  const [ticking, setTicking] = useState<Set<string>>(new Set());      // ticked, waiting for the server
  const [error, setError] = useState('');
  const mayPlan = can(matter, ['lead', 'attorney', 'paralegal']);
  const track = trackOf(tracks.data, matter);
  const stageLabel = (k?: string | null) => track?.stages.find((s) => s.key === k)?.label ?? 'Other';
  const current = track?.stages.find((s) => s.key === matter.stage);
  const tasks = q.data ?? [];
  const open = tasks.filter((t) => t.status === 'open');
  const closed = tasks.filter((t) => t.status !== 'open');
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: keys.tasks(matter.id) }), qc.invalidateQueries({ queryKey: keys.pipeline })]);
  const change = async (t: CaseTask, body: Record<string, unknown>) => {
    setError('');
    if (body.status === 'done') setTicking((s) => new Set(s).add(t.id));
    try { await api.patch(path(matter, `tasks/${encodeURIComponent(t.id)}`), { ...body, version: t.version }); await refresh(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The task could not be changed.'); }
    finally { setTicking((s) => { const n = new Set(s); n.delete(t.id); return n; }); }
  };
  const standard = async () => {
    setError('');
    try { await api.post(path(matter, 'plan/standard-tasks'), {}); await refresh(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The standard tasks could not be added.'); }
  };
  const missing = current ? current.tasks.filter((title) => !tasks.some((t) => t.status !== 'cancelled' && t.title.toLowerCase() === title.toLowerCase())) : [];
  const groups = (track?.stages ?? []).map((s) => ({ key: s.key, label: s.label, items: open.filter((t) => t.stage === s.key) }))
    .concat([{ key: '_', label: 'Other', items: open.filter((t) => !track?.stages.some((s) => s.key === t.stage)) }]).filter((g) => g.items.length);
  return (
    <section className="lw-paper lw-sheet" aria-labelledby="plan-title" style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <h2 className="lw-h2" id="plan-title" style={{ margin: 0 }}>Plan</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>Tasks, key dates, hearings and filing deadlines, entered by the team. Dates here are set by a person, never read in by the AI.</p>
        </div>
        {mayPlan && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {missing.length > 0 && <button type="button" className="lw-btn" onClick={() => void standard()}><ListChecks />Add the standard tasks for {current!.label}</button>}
            <button type="button" className="lw-btn lw-btn-primary" onClick={() => setAdding(true)}><Plus />Add to the plan</button>
          </div>
        )}
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {q.isPending ? <p className="lw-meta" style={{ margin: 0 }}>Loading…</p>
        : q.isError ? <ErrorNote>The plan could not be loaded.</ErrorNote>
          : open.length === 0 ? <EmptyState title="Nothing planned">{mayPlan ? (missing.length ? `Start with the standard tasks for ${current!.label}, or add your own.` : 'Add a task, a key date or a hearing.') : 'The matter\'s team keeps the plan.'}</EmptyState>
            : groups.map((g) => (
              <div key={g.key} style={{ display: 'grid', gap: 6 }}>
                <h3 className="lw-meta" style={{ margin: 0, fontSize: 12.5, textTransform: 'uppercase', letterSpacing: '.04em' }}>{g.label}</h3>
                <ul className="lw-tasks" aria-label={`${g.label} tasks`}>
                  {g.items.map((t) => (
                    <li key={t.id} data-overdue={t.overdue ? 'true' : undefined}>
                      {mayPlan
                        ? <input type="checkbox" aria-label={`Mark “${t.title}” done`} checked={ticking.has(t.id)} disabled={ticking.has(t.id)}
                            onChange={() => void change(t, { status: 'done' })} />
                        : <span aria-hidden className="lw-task-box" />}
                      <span className="lw-task-title">{t.title}</span>
                      <span className="lw-meta lw-task-meta">
                        {t.kind !== 'task' && <span className="lw-chip">{TASK_KIND[t.kind]}</span>}
                        {t.due_on && <span data-overdue={t.overdue ? 'true' : undefined}><CalendarClock aria-hidden />{t.overdue ? 'Overdue · ' : ''}{day(t.due_on)}</span>}
                        {t.assignee && <span>{t.assignee}</span>}
                      </span>
                      {mayPlan && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setEditing(t)} aria-label={`Edit “${t.title}”`}><Pencil /></button>}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
      {closed.length > 0 && (
        <details open={showDone} onToggle={(e) => setShowDone((e.target as HTMLDetailsElement).open)}>
          <summary className="lw-meta">Done and cancelled ({closed.length})</summary>
          <ul className="lw-tasks" aria-label="Done and cancelled">
            {closed.map((t) => (
              <li key={t.id} data-done="true">
                <span className="lw-task-title"><s>{t.title}</s></span>
                <span className="lw-meta lw-task-meta">{t.status === 'done' ? 'Done' : 'Cancelled'} · {stageLabel(t.stage)}</span>
                {mayPlan && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => void change(t, { status: 'open' })}><RotateCcw />Reopen</button>}
              </li>
            ))}
          </ul>
        </details>
      )}
      {adding && <TaskDialog matter={matter} track={track} onClose={() => setAdding(false)} onSaved={refresh} />}
      {editing && <TaskDialog matter={matter} track={track} task={editing} onClose={() => setEditing(null)} onSaved={refresh} />}
    </section>
  );
}

function TaskDialog({ matter, track, task, onClose, onSaved }: { matter: Matter; track?: TrackDef; task?: CaseTask; onClose: () => void; onSaved: () => Promise<unknown> }) {
  const members = useMemberships(matter.id);
  const [form, setForm] = useState({ title: task?.title ?? '', kind: (task?.kind ?? 'task') as TaskKind, due_on: task?.due_on ?? '',
                                     assignee_id: task?.assignee_id ?? '', stage: task?.stage ?? matter.stage ?? '' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setBusy(true);
    try {
      if (task) {
        await api.patch(path(matter, `tasks/${encodeURIComponent(task.id)}`), {
          title: form.title, kind: form.kind, stage: form.stage || undefined, version: task.version,
          ...(form.due_on ? { due_on: form.due_on } : { clear_due: true }),
          ...(form.assignee_id ? { assignee_id: form.assignee_id } : { clear_assignee: true }) });
      } else {
        await api.post(path(matter, 'tasks'), { title: form.title, kind: form.kind, stage: form.stage || undefined,
          ...(form.due_on ? { due_on: form.due_on } : {}), ...(form.assignee_id ? { assignee_id: form.assignee_id } : {}) });
      }
      await onSaved();
      onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The task could not be saved.'); }
    finally { setBusy(false); }
  };
  const cancel = async () => {
    if (!task) return;
    setError('');
    try { await api.patch(path(matter, `tasks/${encodeURIComponent(task.id)}`), { status: 'cancelled', version: task.version }); await onSaved(); onClose(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The task could not be cancelled.'); }
  };
  return (
    <Dialog title={task ? 'Edit the task' : 'Add to the plan'} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="What">{(id) => <input id={id} className="lw-input" maxLength={300} required value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })}
          placeholder="For example: File the reply to the defence" />}</Field>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Kind">{(id) => (
            <select id={id} className="lw-input" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as TaskKind })}>
              {(Object.keys(TASK_KIND) as TaskKind[]).map((k) => <option key={k} value={k}>{TASK_KIND[k]}</option>)}
            </select>)}</Field>
          <Field label="Due" hint="Optional">{(id) => <input id={id} type="date" className="lw-input" value={form.due_on} onChange={(e) => setForm({ ...form, due_on: e.target.value })} />}</Field>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Who">{(id) => (
            <select id={id} className="lw-input" value={form.assignee_id} onChange={(e) => setForm({ ...form, assignee_id: e.target.value })}>
              <option value="">Nobody yet</option>
              {(members.data ?? []).map((m) => <option key={m.user_id} value={m.user_id}>{m.display_name}</option>)}
            </select>)}</Field>
          {track && <Field label="Stage">{(id) => (
            <select id={id} className="lw-input" value={form.stage} onChange={(e) => setForm({ ...form, stage: e.target.value })}>
              {track.stages.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>)}</Field>}
        </div>
        {(form.kind === 'hearing' || form.kind === 'filing') && <p className="lw-meta" style={{ margin: 0 }}>Check the date against the court's order or the rules before relying on it.</p>}
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          {task ? <button type="button" className="lw-btn lw-btn-ghost" onClick={() => void cancel()}>Cancel this task</button> : <span />}
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Close</button>
            <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.title.trim() || busy}>Save</button>
          </div>
        </div>
      </form>
    </Dialog>
  );
}
