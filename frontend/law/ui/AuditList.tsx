import React from 'react';
import type { AuditEvent } from '../api/types';
import { actorName, describe, reasonLabel } from './auditText';
import { StatusPill } from './primitives';

const dayOf = (iso: string) => new Date(iso).toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
const timeOf = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** A ledger of events, newest first, grouped by day. Refusals carry a word, not just a colour. */
export default function AuditList({ events }: { events: AuditEvent[] }) {
  const days: [string, AuditEvent[]][] = [];
  for (const e of events) {
    const d = dayOf(e.ts);
    if (days.length && days[days.length - 1][0] === d) days[days.length - 1][1].push(e);
    else days.push([d, [e]]);
  }
  return (
    <div className="lw-ledger">
      {days.map(([day, list]) => (
        <section key={day} aria-label={day}>
          <h2 className="lw-ledger-day">{day}</h2>
          <ol className="lw-ledger-list">
            {list.map((e) => <AuditRow key={e.seq} e={e} />)}
          </ol>
        </section>
      ))}
    </div>
  );
}

function AuditRow({ e }: { e: AuditEvent }) {
  const reason = reasonLabel(e.reason);
  return (
    <li className="lw-ledger-row" data-outcome={e.outcome}>
      <time dateTime={e.ts} className="lw-tnum lw-meta">{timeOf(e.ts)}</time>
      <div style={{ minWidth: 0 }}>
        <div><b>{actorName(e)}</b> {describe(e)}</div>
        <details className="lw-ledger-more">
          <summary className="lw-meta">Details</summary>
          <dl className="lw-meta">
            <dt>Action</dt><dd className="lw-mono">{e.action}</dd>
            {e.matter?.title && <><dt>Matter</dt><dd>{e.matter.number} · {e.matter.title}</dd></>}
            {e.matter && !e.matter.number && <><dt>Matter</dt><dd className="lw-mono">ref {e.matter.id.slice(0, 8)}</dd></>}
            {e.actor?.roles?.length ? <><dt>Firm roles</dt><dd>{e.actor.roles.join(', ').replace(/_/g, ' ')}</dd></> : null}
            {e.auth && <><dt>Signed in with</dt><dd>{e.auth === 'obo' ? 'voice, on their behalf' : e.auth}</dd></>}
            {e.session_ref && <><dt>Session</dt><dd className="lw-mono">{e.session_ref}</dd></>}
            {e.ip && <><dt>Address</dt><dd className="lw-mono">{e.ip}</dd></>}
            <dt>Event</dt><dd className="lw-mono">#{e.seq}</dd>
          </dl>
        </details>
      </div>
      <div style={{ textAlign: 'right' }}>
        {e.outcome === 'deny' && <StatusPill tone="conflict">Refused</StatusPill>}
        {e.outcome === 'error' && <StatusPill tone="attention">Failed</StatusPill>}
        {e.outcome === 'allow' && e.reason === 'break_glass' && <StatusPill tone="conflict">Emergency access</StatusPill>}
        {reason && e.outcome !== 'allow' && <div className="lw-meta">{reason}</div>}
      </div>
    </li>
  );
}
