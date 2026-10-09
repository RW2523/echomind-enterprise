import React, { useState } from 'react';
import { Download, ShieldCheck, ShieldAlert } from 'lucide-react';
import { ApiError } from '../api/client';
import { useAuditTrail, verifyAuditTrail, type AuditFilters } from '../api/queries';
import type { AuditVerification } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import AuditList from '../ui/AuditList';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const ACTIVITY: [string, string][] = [
  ['', 'All activity'], ['auth', 'Sign-in'], ['matter', 'Matters'], ['membership', 'Members'], ['wall', 'Ethical walls'],
  ['breakglass', 'Emergency access'], ['user', 'Accounts'], ['roles', 'Firm roles'], ['audit', 'Audit trail'], ['api', 'Other requests'],
];
const PROBLEMS: Record<string, string> = {
  event_altered: 'An event was changed after it was written.',
  link_broken: 'An event was removed or the order changed.',
  anchor_mismatch: 'The trail no longer matches a daily anchor: it was rewritten.',
  events_missing_after_anchor: 'Events recorded in a daily anchor are missing.',
  anchor_unreadable: 'A daily anchor file could not be read.',
};

/** The firm-wide audit trail (managing partners, records administrators, firm administrators). */
export default function AuditTrail() {
  const me = useSession();
  const allowed = hasFirmRole(me, 'firm_admin', 'managing_partner', 'records_admin');
  const [filters, setFilters] = useState<AuditFilters>({});
  const [person, setPerson] = useState('');
  const trail = useAuditTrail(filters, allowed);
  if (!allowed) return <NotAvailable />;
  const events = trail.data?.pages.flatMap((p) => p.events) ?? [];
  const showsNumbers = trail.data?.pages[0]?.shows_matter_numbers;
  const set = (k: keyof AuditFilters, v: string) => setFilters((f) => ({ ...f, [k]: v || undefined }));
  return (
    <>
      <Toolbar title="Audit trail">
        <a className="lw-btn lw-btn-sm lw-btn-ghost" href="/api/legal/audit/export?format=jsonl" download><Download />Verifiable export</a>
        <a className="lw-btn lw-btn-sm lw-btn-ghost" href="/api/legal/audit/export?format=csv" download>CSV</a>
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <Integrity />
        <form className="lw-filters" role="search" aria-label="Filter the trail" onSubmit={(e) => { e.preventDefault(); set('actor', person.trim()); }}>
          <Field label="Person">{(id) => <input id={id} className="lw-input" placeholder="Username" value={person}
            onChange={(e) => setPerson(e.target.value)} onBlur={() => set('actor', person.trim())} />}</Field>
          <Field label="Activity">{(id) => <select id={id} className="lw-input" value={filters.action ?? ''} onChange={(e) => set('action', e.target.value)}>
            {ACTIVITY.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          <Field label="Outcome">{(id) => <select id={id} className="lw-input" value={filters.outcome ?? ''} onChange={(e) => set('outcome', e.target.value)}>
            <option value="">Any outcome</option><option value="allow">Allowed</option><option value="deny">Refused</option><option value="error">Failed</option></select>}</Field>
        </form>
        {showsNumbers === false && <p className="lw-meta" style={{ margin: 0 }}>Matter numbers and titles are not shown to firm administrators; each matter appears as an opaque reference.</p>}
        {trail.isError ? <ErrorNote>{trail.error instanceof ApiError ? trail.error.message : 'The trail could not be loaded.'}</ErrorNote>
          : trail.isPending ? <div className="lw-paper lw-sheet" aria-busy="true"><Skeleton /><Skeleton /><Skeleton /></div>
          : events.length === 0 ? <EmptyState title="No matching events">Try a different person, activity or outcome.</EmptyState>
          : (
            <section className="lw-paper" aria-label="Events">
              <AuditList events={events} />
              {trail.hasNextPage && (
                <div style={{ padding: '12px 20px', borderTop: '1px solid var(--rule)' }}>
                  <button type="button" className="lw-btn lw-btn-sm" disabled={trail.isFetchingNextPage} onClick={() => trail.fetchNextPage()}>Show older events</button>
                </div>
              )}
            </section>
          )}
      </div>
    </>
  );
}

/** Integrity check: recompute the chain on the server and compare it with the daily anchors. */
function Integrity() {
  const [result, setResult] = useState<AuditVerification | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    setBusy(true); setError('');
    try { setResult(await verifyAuditTrail()); } catch (e) { setError(e instanceof ApiError ? e.message : 'The check could not be run.'); }
    finally { setBusy(false); }
  };
  const ok = result?.ok;
  return (
    <section className="lw-paper lw-sheet" aria-label="Integrity" style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
      {result && !ok ? <ShieldAlert size={28} style={{ color: 'var(--conflict)', flex: 'none' }} /> : <ShieldCheck size={28} style={{ color: ok ? 'var(--verified)' : 'var(--ink-3)', flex: 'none' }} />}
      <div style={{ flex: 1, minWidth: 0 }} aria-live="polite">
        <h2 className="lw-h2" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>Integrity
          {result && (ok ? <StatusPill tone="verified">Intact</StatusPill> : <StatusPill tone="conflict">Broken</StatusPill>)}</h2>
        <p className="lw-meta" style={{ margin: '4px 0 0', fontSize: 13.5 }}>
          {!result && 'Each event is sealed to the one before it, and the latest seal is copied to a daily anchor. Check that nothing was changed or removed.'}
          {result && ok && `${result.events.toLocaleString()} events checked against ${result.anchors_checked} daily anchor${result.anchors_checked === 1 ? '' : 's'} at ${new Date(result.verified_at).toLocaleTimeString([], { timeStyle: 'short' })}.`}
          {result && !ok && `${PROBLEMS[result.problem ?? ''] ?? 'The trail does not verify.'}${result.broken_at ? ` First affected event: #${result.broken_at}.` : ''} Preserve the database and its backups, and tell your security contact.`}
        </p>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
      <button type="button" className="lw-btn lw-btn-sm" onClick={run} disabled={busy}>{busy ? 'Checking…' : result ? 'Check again' : 'Check now'}</button>
    </section>
  );
}
