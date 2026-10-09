import React from 'react';
import { ApiError } from '../api/client';
import { useMatterActivity } from '../api/queries';
import { useMatter } from '../app/matter';
import AuditList from '../ui/AuditList';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton } from '../ui/primitives';
import NotAvailable from './NotAvailable';

/** Everything recorded about one matter, for its lead: who opened it, changed it, or was refused. */
export default function MatterActivity() {
  const matter = useMatter();
  if (matter.my_role !== 'lead') return <NotAvailable />;
  return <ActivityFor matterId={matter.id} number={matter.number} />;
}

function ActivityFor({ matterId, number }: { matterId: string; number: string }) {
  const activity = useMatterActivity(matterId);
  const events = activity.data?.pages.flatMap((p) => p.events) ?? [];
  return (
    <>
      <Toolbar crumb={number} title="Activity" />
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>Who viewed or changed this matter, including attempts that were refused. Only the matter’s lead sees this page, and viewing it is recorded too.</p>
        {activity.isError ? <ErrorNote>{activity.error instanceof ApiError ? activity.error.message : 'Activity could not be loaded.'}</ErrorNote>
          : activity.isPending ? <div className="lw-paper lw-sheet" aria-busy="true"><Skeleton /><Skeleton /></div>
          : events.length === 0 ? <EmptyState title="No activity yet" />
          : (
            <section className="lw-paper" aria-label="Activity">
              <AuditList events={events} />
              {activity.hasNextPage && (
                <div style={{ padding: '12px 20px', borderTop: '1px solid var(--rule)' }}>
                  <button type="button" className="lw-btn lw-btn-sm" disabled={activity.isFetchingNextPage} onClick={() => activity.fetchNextPage()}>Show older activity</button>
                </div>
              )}
            </section>
          )}
      </div>
    </>
  );
}
