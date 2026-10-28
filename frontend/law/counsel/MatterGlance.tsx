/**
 * A matter at a glance, above its tools: the inconsistencies still open, the next deadlines and the latest documents.
 * Every line comes from the matter's own records (the chat's discrepancy register, the obligations register, the document
 * library) and opens where it can be acted on; nothing here is generated. A part the person may not see is left out.
 */
import React from 'react';
import { Link, useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { CalendarClock, ChevronRight, ClipboardCheck, Files, Scale } from 'lucide-react';
import { ApiError } from '../api/client';
import { useDocuments, useObligations, useReviews } from '../api/queries';
import type { Matter, Obligation } from '../api/types';
import { counselApi } from './api';
import { useChatMatter } from './context';
import type { DiscrepancyStatus } from './types';

const ISSUE_STATUS: Record<DiscrepancyStatus, string> = {
  potential: 'Potential', in_review: 'In review', explained: 'Explained', confirmed: 'Confirmed', dismissed: 'Dismissed',
};
const DUE_STATE: Record<Obligation['status'], string> = {
  proposed: 'Not confirmed', confirmed: 'Confirmed', done: 'Done', dismissed: 'Dismissed',
};
/** "Potential inconsistency in dates: “May 20” and “May 15”" reads "Dates: “May 20” and “May 15”" beside its status chip. */
const shortTitle = (t: string) => { const r = t.replace(/^potential inconsistency in\s+/i, ''); return r.charAt(0).toUpperCase() + r.slice(1); };
const day = (iso: string) => new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });

function Panel({ icon, title, to, more, children }: { icon: React.ReactNode; title: string; to?: string; more?: string; children: React.ReactNode }) {
  return (
    <section className="qc-card qc-glance" aria-label={title}>
      <div className="qc-card-head">
        <h3 className="qc-card-title"><span className="qc-card-icon" aria-hidden>{icon}</span>{title}</h3>
        {to && more && <Link className="qc-linkbtn" to={to}>{more}</Link>}
      </div>
      {children}
    </section>
  );
}

/** A panel the person may not see (403/404) is left out; one that could not load says so instead of disappearing (review M2). */
const refused = (e: unknown) => e instanceof ApiError && (e.status === 403 || e.status === 404);
const Failed = ({ what }: { what: string }) => <p className="qc-meta" role="status" style={{ margin: 0 }}>{what} could not be loaded. Reload the page to try again.</p>;
const DOC_STATUS: Record<string, string> = { executed: 'signed', draft: 'draft', received: 'received', final: 'final' };
const READING: Record<string, string> = { queued: 'Waiting to be read', indexing: 'Being read', failed: 'Could not be read' };

function Issues({ matter }: { matter: Matter }) {
  const navigate = useNavigate();
  const { setMatterId } = useChatMatter();
  const q = useQuery({ queryKey: ['counsel', 'discrepancies', matter.id], retry: false, staleTime: 15_000,
                       queryFn: () => counselApi.discrepancies(matter.id) });
  if (q.isError) return refused(q.error) ? null : <Panel icon={<Scale aria-hidden />} title="Open inconsistencies"><Failed what="The inconsistencies" /></Panel>;
  const toChat = (ask: string, focus?: string) => { setMatterId(matter.id); navigate('/chat', { state: { ask, focus } }); };
  const items = q.data?.discrepancies ?? [];
  return (
    <Panel icon={<Scale aria-hidden />} title="Open inconsistencies">
      {q.isPending ? <p className="qc-meta" style={{ margin: 0 }}>Loading…</p> : items.length === 0 ? (
        <div className="qc-glance-empty">
          <p className="qc-meta" style={{ margin: 0 }}>None open. Ask the chat to compare the documents' dates and amounts.</p>
          <button type="button" className="qc-btn" onClick={() => toChat('Find conflicting dates or amounts')}>Find conflicting dates or amounts</button>
        </div>
      ) : (
        <ul className="qc-glance-list">
          {items.slice(0, 4).map((d) => (
            <li key={d.id}>
              <button type="button" className="qc-glance-row" onClick={() => toChat('Show this potential inconsistency', d.id)}>
                <span className="qc-glance-main" title={d.title}>{shortTitle(d.title)}</span>
                <span className="qc-status-chip" data-status={d.status}>{ISSUE_STATUS[d.status]}</span>
                <ChevronRight aria-hidden />
              </button>
            </li>
          ))}
          {items.length > 4 && <li className="qc-meta">and {items.length - 4} more</li>}
        </ul>
      )}
    </Panel>
  );
}

function Deadlines({ matter }: { matter: Matter }) {
  const q = useObligations(matter.id);
  const base = `/m/${encodeURIComponent(matter.number)}/obligations`;
  if (q.isError) return refused(q.error) ? null : <Panel icon={<CalendarClock aria-hidden />} title="Next deadlines" to={base} more="All obligations"><Failed what="The deadlines" /></Panel>;
  const dated = (q.data?.obligations ?? []).filter((o) => o.due_date && o.status !== 'dismissed' && o.status !== 'done')
    .sort((a, b) => Number(b.overdue) - Number(a.overdue) || (a.due_date ?? '').localeCompare(b.due_date ?? ''));
  return (
    <Panel icon={<CalendarClock aria-hidden />} title="Next deadlines" to={base} more="All obligations">
      {q.isPending ? <p className="qc-meta" style={{ margin: 0 }}>Loading…</p> : dated.length === 0 ? (
        <p className="qc-meta" style={{ margin: 0 }}>No dated deadline is recorded. Obligations are read from the documents on the Obligations page.</p>
      ) : (
        <ul className="qc-glance-list">
          {dated.slice(0, 4).map((o) => (
            <li key={o.id}>
              <Link className="qc-glance-row" to={base}>
                <span className="qc-glance-date" data-overdue={o.overdue ? 'true' : undefined}>{day(o.due_date!)}</span>
                <span className="qc-glance-main">{o.action}</span>
                <span className="qc-tag" data-tag={o.overdue ? 'Deadline' : o.status === 'confirmed' ? 'Confirmed by a lawyer' : 'Uncertain'}>
                  {o.overdue ? 'Overdue' : DUE_STATE[o.status]}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function LatestDocuments({ matter }: { matter: Matter }) {
  const q = useDocuments(matter.id);
  const base = `/m/${encodeURIComponent(matter.number)}/documents`;
  if (q.isError) return refused(q.error) ? null : <Panel icon={<Files aria-hidden />} title="Documents" to={base} more="All documents"><Failed what="The documents" /></Panel>;
  const docs = [...(q.data ?? [])].sort((a, b) => (b.updated_at ?? b.created_at).localeCompare(a.updated_at ?? a.created_at));
  return (
    <Panel icon={<Files aria-hidden />} title={q.data ? `Documents (${q.data.length})` : 'Documents'} to={base} more="All documents">
      {q.isPending ? <p className="qc-meta" style={{ margin: 0 }}>Loading…</p> : docs.length === 0 ? (
        <p className="qc-meta" style={{ margin: 0 }}>No documents yet. Import them on the Documents page.</p>
      ) : (
        <ul className="qc-glance-list">
          {docs.slice(0, 4).map((d) => (
            <li key={d.id}>
              <Link className="qc-glance-row" to={`${base}/${encodeURIComponent(d.id)}/read`}>
                <span className="qc-glance-main">{d.title}</span>
                <span className="qc-meta" style={{ whiteSpace: 'nowrap', textAlign: 'right' }}>
                  {d.current_version && READING[d.current_version.index_status]
                    ? <span className="qc-tag" data-tag="Uncertain">{READING[d.current_version.index_status]}</span>
                    : <>{d.current_version ? `v${d.current_version.number} · ` : ''}{DOC_STATUS[d.status] ?? d.status}</>}
                  <br />{day(d.updated_at ?? d.created_at)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/** Drafts, notes and decisions on this matter waiting for the person (the reviews queue), shown only when there are some. */
function WaitingForYou({ matter }: { matter: Matter }) {
  const q = useReviews();
  const items = (q.data?.waiting ?? []).filter((i) => i.matter?.number === matter.number);
  if (!items.length) return null;
  return (
    <Panel icon={<ClipboardCheck aria-hidden />} title="Waiting for your decision" to="/drafts" more="All drafts">
      <ul className="qc-glance-list">
        {items.slice(0, 4).map((i) => (
          <li key={`${i.kind}-${i.id}`}>
            <Link className="qc-glance-row" to={i.path}>
              <span className="qc-glance-main">{i.title}</span>
              <span className="qc-meta" style={{ whiteSpace: 'nowrap' }}>{i.by ? `from ${i.by}` : ''}</span>
            </Link>
          </li>
        ))}
        {items.length > 4 && <li className="qc-meta">and {items.length - 4} more</li>}
      </ul>
    </Panel>
  );
}

export default function MatterGlance({ matter }: { matter: Matter }) {
  return (
    <section aria-label="At a glance" className="qc-glance-grid">
      <WaitingForYou matter={matter} />
      <Issues matter={matter} />
      <Deadlines matter={matter} />
      <LatestDocuments matter={matter} />
    </section>
  );
}
