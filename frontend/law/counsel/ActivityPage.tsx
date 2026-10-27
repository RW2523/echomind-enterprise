/**
 * Activity (docs/echomind-legal/04): documents still being read in your matters, and what you did, from the audit trail
 * (your own events only; a matter you can no longer open shows without its name).
 */
import React from 'react';
import { Link } from 'react-router';
import { useInfiniteQuery } from '@tanstack/react-query';
import AuditList from '../ui/AuditList';
import { counselApi } from './api';

export default function ActivityPage() {
  const q = useInfiniteQuery({
    queryKey: ['counsel', 'activity'], initialPageParam: 0 as number, retry: false,
    queryFn: ({ pageParam }) => counselApi.activity(pageParam || undefined),
    getNextPageParam: (last) => last.next_before ?? undefined,
  });
  const first = q.data?.pages[0];
  const events = q.data?.pages.flatMap((p) => p.events) ?? [];
  return (
    <div className="qc-page">
      <h1>Activity</h1>
      {q.isPending && <p className="qc-meta" aria-busy="true">Loading…</p>}
      {q.isError && <p className="qc-error" role="alert">Activity could not be loaded. Try again.</p>}
      {first && first.in_progress.length > 0 && (
        <section className="qc-card" aria-label="In progress">
          <h2 className="qc-card-title">In progress</h2>
          <ul className="qc-list">
            {first.in_progress.map((p) => (
              <li key={p.matter.id}>
                <span>{p.documents} document{p.documents === 1 ? '' : 's'} being read in <Link to={`/m/${encodeURIComponent(p.matter.number)}/documents`}>{p.matter.number} · {p.matter.title}</Link></span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {first && (
        <section aria-label="What you did">
          <h2 className="qc-card-title" style={{ marginBottom: 8 }}>What you did</h2>
          {events.length ? <AuditList events={events} /> : <p className="qc-meta">Nothing yet.</p>}
          {q.hasNextPage && (
            <button type="button" className="qc-btn" disabled={q.isFetchingNextPage} onClick={() => void q.fetchNextPage()}>
              {q.isFetchingNextPage ? 'Loading…' : 'Show earlier'}
            </button>
          )}
        </section>
      )}
    </div>
  );
}
