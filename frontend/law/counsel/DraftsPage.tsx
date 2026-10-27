/**
 * Drafts (docs/echomind-legal/04): what waits for your decision, and your own AI drafts with where each stands. It reads
 * the existing reviews queue, which lists only matters you can open; each item opens its own page to read and decide.
 */
import React from 'react';
import { Link } from 'react-router';
import { useReviews } from '../api/queries';
import type { ReviewItem } from '../api/types';

const KIND: Record<string, string> = {
  minutes: 'Minutes', research_note: 'Research note', legal_draft: 'Draft', redline: 'Redline', summary: 'Summary', panel: 'Review panel',
  export: 'Export', deletion: 'Deletion', hold: 'Legal hold', wall: 'Ethical wall', emergency_access: 'Emergency access',
};
const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');

function Row({ it, mine }: { it: ReviewItem; mine?: boolean }) {
  const label = `${KIND[it.kind] ?? it.kind}: ${it.title}`;
  const target = it.path.split('?')[0] ?? '';
  const internal = target.startsWith('/m/') || target.startsWith('/admin/') || target === '/reviews';
  return (
    <li>
      <span style={{ minWidth: 0 }}>
        {internal ? <Link to={it.path}><b>{label}</b></Link> : <b>{label}</b>}
        <span className="qc-meta" style={{ display: 'block' }}>
          {it.matter ? `${it.matter.number} · ${it.matter.title} · ` : ''}
          {mine ? (it.state === 'changes_requested' ? `Changes asked for by ${it.decided_by ?? 'a reviewer'}` : 'Waiting for a second lawyer') : `From ${it.by ?? 'someone'}`}
          {it.at ? ` · ${when(it.at)}` : ''}
        </span>
        {mine && it.reason && <span className="qc-meta" style={{ display: 'block' }}>“{it.reason}”</span>}
      </span>
    </li>
  );
}

export default function DraftsPage() {
  const q = useReviews();
  return (
    <div className="qc-page">
      <h1>Drafts</h1>
      <p className="qc-meta" style={{ margin: 0 }}>Drafts are made in a matter's Drafting page from your firm's approved templates. Nothing leaves a matter until a second lawyer approves it.</p>
      {q.isPending && <p className="qc-meta" aria-busy="true">Loading…</p>}
      {q.isError && <p className="qc-error" role="alert">Drafts could not be loaded. Try again.</p>}
      {q.data && (
        <>
          <section className="qc-card" aria-label="Waiting for your review">
            <h2 className="qc-card-title">Waiting for your review ({q.data.waiting.length})</h2>
            {q.data.waiting.length ? <ul className="qc-list">{q.data.waiting.map((it) => <Row key={`${it.kind}-${it.id}`} it={it} />)}</ul>
              : <p className="qc-meta" style={{ margin: 0 }}>Nothing is waiting for you.</p>}
          </section>
          <section className="qc-card" aria-label="Your drafts">
            <h2 className="qc-card-title">Your drafts ({q.data.mine.length})</h2>
            {q.data.mine.length ? <ul className="qc-list">{q.data.mine.map((it) => <Row key={`${it.kind}-${it.id}`} it={it} mine />)}</ul>
              : <p className="qc-meta" style={{ margin: 0 }}>You have no drafts waiting for review.</p>}
          </section>
        </>
      )}
    </div>
  );
}
