/**
 * The pipeline (case management, 2026-11-07): the open matters a person may see, in columns by stage, one track at a time.
 * Each card shows the client, lead, days in the stage, the next task and the latest update; the lawyers on a matter move it
 * with "Move to…" (no dragging needed). On a narrow screen the columns stack.
 */
import React, { useState } from 'react';
import { Link } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { CalendarClock, MessageSquareText, Plus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, usePipeline } from '../api/queries';
import type { Matter, PipelineRow, TrackKey } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { UPDATE_KIND, day } from '../ui/casework';
import { EmptyState, ErrorNote, Skeleton } from '../ui/primitives';

export default function Pipeline() {
  const me = useSession();
  const q = usePipeline();
  const qc = useQueryClient();
  const [trackKey, setTrackKey] = useState<TrackKey | ''>('');
  const [error, setError] = useState('');
  const canCreate = hasFirmRole(me, 'attorney', 'managing_partner');
  const rows = q.data?.matters ?? [];
  const tracks = q.data?.tracks ?? [];
  const counts = Object.fromEntries(tracks.map((t) => [t.key, rows.filter((r) => r.track === t.key).length]));
  const active = (trackKey || tracks.find((t) => counts[t.key])?.key || tracks[0]?.key) as TrackKey | undefined;
  const track = tracks.find((t) => t.key === active);
  const move = async (r: PipelineRow, stage: string) => {
    setError('');
    try {
      const m = (await api.get<{ matter: Matter }>(`/api/legal/matters/${encodeURIComponent(r.id)}`)).matter;
      await api.post(`/api/legal/matters/${encodeURIComponent(r.id)}/stage`, { stage, version: m.version });
      await Promise.all([qc.invalidateQueries({ queryKey: keys.pipeline }), qc.invalidateQueries({ queryKey: keys.matters }),
                         qc.invalidateQueries({ queryKey: keys.matter(r.id) })]);
    } catch (e) { setError(e instanceof ApiError ? e.message : `${r.number} could not be moved.`); }
  };
  return (
    <>
      <Toolbar title="Pipeline">
        <Link to="/matters" className="lw-btn lw-btn-ghost">List</Link>
        {canCreate && <Link to="/matters/new" className="lw-btn lw-btn-primary"><Plus />New matter</Link>}
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 'none' }}>
        {q.isPending ? <div className="lw-paper lw-sheet" style={{ display: 'grid', gap: 10 }}><Skeleton /><Skeleton /></div>
          : q.isError ? <EmptyState title="The pipeline could not be loaded">Check the connection and try again.</EmptyState>
            : rows.length === 0 ? <EmptyState title="No open matters">Matters you work on appear here by stage.</EmptyState>
              : (
                <>
                  <div className="lw-seg lw-board-tabs" role="tablist" aria-label="Track">
                    {tracks.map((t) => (
                      <button key={t.key} type="button" role="tab" aria-selected={t.key === active} className="lw-seg-item"
                        onClick={() => setTrackKey(t.key)}>{t.label} <span className="lw-meta">({counts[t.key] ?? 0})</span></button>
                    ))}
                  </div>
                  {error && <ErrorNote>{error}</ErrorNote>}
                  {track && (
                    <div className="lw-board" aria-label={`${track.label} pipeline`}>
                      {track.stages.map((s) => {
                        const items = rows.filter((r) => r.track === track.key && r.stage === s.key);
                        return (
                          <section key={s.key} className="lw-board-col" aria-label={`${s.label}: ${items.length}`}>
                            <h2 className="lw-board-head">{s.label} <span className="lw-meta">{items.length}</span></h2>
                            {items.length === 0 ? <p className="lw-meta lw-board-empty">None</p> : items.map((r) => (
                              <article key={r.id} className="lw-board-card" data-overdue={r.overdue ? 'true' : undefined}>
                                <Link to={`/m/${encodeURIComponent(r.number)}/overview`} className="lw-board-link">
                                  <span className="lw-tnum lw-board-num">{r.number}</span>
                                  <span className="lw-board-title">{r.title}</span>
                                </Link>
                                <p className="lw-meta" style={{ margin: 0 }}>{r.client}{r.leads.length ? ` · ${r.leads.join(', ')}` : ''}</p>
                                <p className="lw-meta" style={{ margin: 0 }}>{r.days_in_stage != null ? `${r.days_in_stage} day${r.days_in_stage === 1 ? '' : 's'} in this stage` : ''}
                                  {r.open_tasks ? ` · ${r.open_tasks} open task${r.open_tasks === 1 ? '' : 's'}` : ''}{r.overdue ? ` · ${r.overdue} overdue` : ''}</p>
                                {r.next_task && (
                                  <p className="lw-board-line" data-overdue={r.overdue ? 'true' : undefined}>
                                    <CalendarClock aria-hidden />{r.next_task.due_on ? `${day(r.next_task.due_on)}: ` : 'Next: '}{r.next_task.title}
                                  </p>
                                )}
                                {r.last_update && (
                                  <p className="lw-board-line" title={r.last_update.excerpt}>
                                    <MessageSquareText aria-hidden />{UPDATE_KIND[r.last_update.kind]}, {day(r.last_update.occurred_on)}: {r.last_update.excerpt}
                                  </p>
                                )}
                                {['lead', 'attorney'].includes(r.my_role ?? '') && (
                                  <select className="lw-input lw-board-move" aria-label={`Move ${r.number} to`} value=""
                                    onChange={(e) => e.target.value && void move(r, e.target.value)}>
                                    <option value="">Move to…</option>
                                    {track.stages.filter((x) => x.key !== s.key).map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
                                  </select>
                                )}
                              </article>
                            ))}
                          </section>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
      </div>
    </>
  );
}
