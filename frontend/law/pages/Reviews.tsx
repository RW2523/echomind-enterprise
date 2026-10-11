import React from 'react';
import { Link } from 'react-router';
import { useReviews } from '../api/queries';
import type { ReviewItem, ReviewKind } from '../api/types';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const KIND: Record<ReviewKind, [string, Tone]> = {
  minutes: ['Minutes', 'accent'], research_note: ['Research note', 'accent'], legal_draft: ['Draft', 'accent'], redline: ['Redline', 'accent'], summary: ['Summary', 'accent'], panel: ['Panel review', 'privileged'], export: ['Export', 'privileged'],
  deletion: ['Deletion', 'conflict'], hold: ['Legal hold', 'hold'], wall: ['Ethical wall', 'neutral'], emergency_access: ['Emergency access', 'conflict'],
};
const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');

/** Reviews (S2.7, D-043): everything waiting for your decision — AI drafts to approve, exports to
 *  release, and the firm decisions you hold — and what of yours waits for someone else. */
export default function Reviews() {
  const q = useReviews();
  return (
    <>
      <Toolbar title="Reviews" />
      <div className="lw-content" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(260px, 1fr)', alignItems: 'start' }}>
        <section className="lw-paper" aria-labelledby="rv-waiting">
          <div className="lw-card-h"><h2 className="lw-h2" id="rv-waiting">Waiting for you</h2>
            {q.data && <span className="lw-meta lw-tnum">{q.data.count}</span>}</div>
          {q.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /></div>
            : q.isError ? <div style={{ padding: 20 }}><ErrorNote>The reviews could not be loaded.</ErrorNote></div>
              : q.data.waiting.length === 0
                ? <div style={{ padding: 20 }}><EmptyState title="Nothing waits for you">Drafts, exports and firm decisions that need you appear here.</EmptyState></div>
                : <ul className="lw-queue" aria-label="Waiting for you">{q.data.waiting.map((i) => <Row key={`${i.kind}-${i.id}`} i={i} />)}</ul>}
        </section>
        <section className="lw-paper" aria-labelledby="rv-mine">
          <div className="lw-card-h"><h2 className="lw-h2" id="rv-mine">Yours, with others</h2></div>
          {q.data && (q.data.mine.length === 0
            ? <p className="lw-meta" style={{ padding: '14px 20px 20px', margin: 0 }}>Nothing of yours is waiting for someone else.</p>
            : <ul className="lw-queue" aria-label="Yours, with others">{q.data.mine.map((i) => <Row key={`${i.kind}-${i.id}`} i={i} mine />)}</ul>)}
        </section>
      </div>
    </>
  );
}

function Row({ i, mine = false }: { i: ReviewItem; mine?: boolean }) {
  const [label, tone] = KIND[i.kind];
  const back = i.state === 'changes_requested';
  return (
    <li aria-label={`${label}: ${i.title}`}>
      <StatusPill tone={tone}>{label}</StatusPill>
      <div style={{ minWidth: 0 }}>
        <Link to={i.path}>{i.title}{i.version ? <span className="lw-meta"> · version {i.version}</span> : null}</Link>
        <div className="lw-meta">
          {i.matter ? `${i.matter.number} · ${i.matter.title} · ` : ''}
          {mine ? (back ? `Sent back by ${i.decided_by ?? 'a reviewer'}: “${i.reason ?? ''}”` : 'Waiting for a second lawyer')
            : `${i.by ? `From ${i.by}` : ''}`}
        </div>
      </div>
      <span className="lw-meta lw-tnum" style={{ whiteSpace: 'nowrap' }}>{when(i.at)}</span>
    </li>
  );
}
