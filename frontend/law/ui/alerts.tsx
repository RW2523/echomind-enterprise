import React, { useState } from 'react';
import { Link } from 'react-router';
import { Check, X } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, meetingsPath } from '../api/queries';
import type { AssistAlert, AssistState, AssistTelemetry } from '../api/types';
import { useMatter } from '../app/matter';
import { ErrorNote, StatusPill, type Tone } from './primitives';

const VERDICT: Record<AssistAlert['verdict'], Tone> = { differs: 'conflict', matches: 'verified', needs_review: 'attention' };
const STATUS: Record<string, string> = { draft: 'Draft', final: 'Final', executed: 'Signed', received: 'Received' };

/** One alert: what was said, what it was checked against, and Confirm or Dismiss (D-015: never "true" or "false"). */
export function AlertCard({ a, matterId, meetingId, canDecide, onDecided }: {
  a: AssistAlert; matterId: string; meetingId: string; canDecide: boolean; onDecided?: (a: AssistAlert) => void;
}) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [dismissing, setDismissing] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const src = a.source && !a.source.removed ? a.source : null;
  const decide = async (decision: 'confirm' | 'dismiss') => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ alert: AssistAlert }>(`${meetingsPath(matterId)}/${encodeURIComponent(meetingId)}/alerts/${encodeURIComponent(a.id)}/${decision}`,
        decision === 'dismiss' ? { reason: reason.trim() } : {});
      onDecided?.(r.alert);
      setDismissing(false);
      await qc.invalidateQueries({ queryKey: [...keys.meeting(matterId, meetingId), 'alerts'] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The decision could not be saved.');
    } finally { setBusy(false); }
  };
  return (
    <li className="lw-alert" data-verdict={a.verdict} data-severity={a.severity} data-status={a.status} aria-label={a.title}>
      <div className="lw-alert-h">
        <StatusPill tone={a.rule === 'action' ? 'neutral' : VERDICT[a.verdict]}>
          {a.rule === 'action' ? 'Action' : a.verdict === 'differs' ? 'Differs' : a.verdict === 'matches' ? 'Matches' : 'Needs review'}</StatusPill>
        <b>{a.title.replace(/^Needs review: /, '').replace(/^./, (c) => c.toUpperCase())}</b>
      </div>
      <p className="lw-serif" style={{ margin: 0 }}><span className="lw-meta">Said: </span>“{a.said.quote}”</p>
      {src?.quote && (
        <p className="lw-serif" style={{ margin: 0 }}>
          <span className="lw-meta">{src.document_title ? `${src.document_title}${src.version_number ? ` v${src.version_number}` : ''}${src.version_status ? ` · ${STATUS[src.version_status] ?? src.version_status}` : ''}: ` : 'Earlier in this meeting: '}</span>
          “{src.quote}”
        </p>
      )}
      {a.source?.removed && <p className="lw-meta" style={{ margin: 0 }}>The document this was checked against was deleted.</p>}
      {a.explanation && <p className="lw-meta" style={{ margin: 0 }}>{a.explanation.replace(/\s*\[\d\]/g, '')}</p>}
      <div className="lw-alert-f">
        <span className="lw-meta">{`${a.method === 'model' ? 'Checked in context' : 'Found by rule'} · confidence ${Math.round(a.confidence * 100)}% (threshold ${Math.round(a.threshold * 100)}%)`}</span>
        {src?.document_id && src.version_id && src.start != null && (
          <Link className="lw-linklike lw-meta" to={`/m/${matter.number}/documents/${encodeURIComponent(src.document_id)}/read?v=${encodeURIComponent(src.version_id)}&at=${src.start}-${src.end}`}
            aria-label={`Open the passage for: ${a.title}`}>Open the passage</Link>
        )}
      </div>
      {a.status !== 'open' ? (
        <p className="lw-meta" style={{ margin: 0 }}>{`${a.status === 'confirmed' ? 'Confirmed' : 'Dismissed'} by ${a.decided_by ?? 'someone'}${a.decision_reason ? ` — ${a.decision_reason}` : ''}`}</p>
      ) : canDecide && (
        dismissing ? (
          <form className="lw-alert-dismiss" onSubmit={(e) => { e.preventDefault(); void decide('dismiss'); }}>
            <label className="lw-sr" htmlFor={`why-${a.id}`}>Why dismiss this alert?</label>
            <input id={`why-${a.id}`} className="lw-input lw-input-sm" placeholder="Why? (required)" value={reason} onChange={(e) => setReason(e.target.value)} />
            <button type="submit" className="lw-btn lw-btn-sm" disabled={busy || reason.trim().length < 3}>Dismiss</button>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setDismissing(false)}>Cancel</button>
          </form>
        ) : (
          <div style={{ display: 'flex', gap: 6 }}>
            <button type="button" className="lw-btn lw-btn-sm" disabled={busy} onClick={() => decide('confirm')}><Check />Confirm</button>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setDismissing(true)}><X />Dismiss…</button>
          </div>
        )
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </li>
  );
}

/** The meeting's alerts with the pack in use and how they fared. */
export function AlertsPanel({ alerts, assist, telemetry, matterId, meetingId, canDecide, live }: {
  alerts: AssistAlert[]; assist?: AssistState | null; telemetry?: AssistTelemetry | null; matterId: string; meetingId: string;
  canDecide: boolean; live?: boolean;
}) {
  const critical = alerts.filter((a) => a.severity === 'critical');
  return (
    <section className="lw-paper lw-sheet lw-alerts" aria-label="Silent Assist">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', alignItems: 'baseline' }}>
        <h2 className="lw-h2">Silent Assist</h2>
        <span className="lw-meta">{assist?.pack ? `${assist.label} pack` : 'Off for this meeting'}</span>
      </div>
      {telemetry && !telemetry.thresholds_approved && (
        <p className="lw-meta" style={{ margin: '4px 0 8px' }}>Thresholds are engineering defaults awaiting a lawyer’s sign-off. Alerts compare what was said with this matter’s documents; they are not legal conclusions.</p>
      )}
      {alerts.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>{live ? 'Listening. Alerts appear here.' : 'No alerts.'}</p> : (
        <ol className="lw-alert-list" aria-live={live ? 'polite' : undefined}>
          {[...critical, ...alerts.filter((a) => a.severity !== 'critical')].map((a) => (
            <AlertCard key={a.id} a={a} matterId={matterId} meetingId={meetingId} canDecide={canDecide} />
          ))}
        </ol>
      )}
      {telemetry && telemetry.alerts > 0 && !live && (
        <p className="lw-meta" style={{ margin: '10px 0 0' }}>
          {[`${telemetry.alerts} alert${telemetry.alerts === 1 ? '' : 's'}`, telemetry.per_hour != null ? `${telemetry.per_hour} an hour` : '',
            `${telemetry.confirmed} confirmed`, `${telemetry.dismissed} dismissed`,
            telemetry.latency_p50_s != null ? `median ${telemetry.latency_p50_s} s after the words` : ''].filter(Boolean).join(' · ')}
        </p>
      )}
    </section>
  );
}
