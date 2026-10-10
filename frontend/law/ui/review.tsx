import React, { useState } from 'react';
import { Check, Undo2 } from 'lucide-react';
import { api, ApiError } from '../api/client';
import type { ArtifactVersion, ReviewDecision } from '../api/types';
import { Dialog, ErrorNote, StatusPill, type Tone } from './primitives';

/** Review of AI drafts (S2.7, D-043): one way to show where a draft stands and to decide on it,
 *  for minutes, research notes and whatever is drafted next. */

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const short = (sha: string) => `${sha.slice(0, 12)}…`;

export function reviewState(v: ArtifactVersion): { label: string; tone: Tone } {
  const last = v.decisions[v.decisions.length - 1];
  if (v.status === 'approved') return { label: 'Approved', tone: 'verified' };
  if (v.status === 'superseded') return { label: 'Replaced by a later version', tone: 'neutral' };
  if (v.status === 'withdrawn') return { label: 'Withdrawn', tone: 'neutral' };
  if (last?.decision === 'changes_requested') return { label: 'Changes requested', tone: 'conflict' };
  return { label: 'Draft — not approved', tone: 'attention' };
}

export function ReviewPill({ v }: { v: ArtifactVersion }) {
  const s = reviewState(v);
  return <StatusPill tone={s.tone}>{s.label}</StatusPill>;
}

/** Where a draft stands, in one line under its title: who made it, and who approved or sent it back. */
export function ReviewBanner({ v, made }: { v: ArtifactVersion; made: string }) {
  const last = v.decisions[v.decisions.length - 1];
  const approval = [...v.decisions].reverse().find((d) => d.decision === 'approved');
  let line = `${made} · not final until a second lawyer approves it.`;
  if (v.status === 'approved' && approval) line = `Approved by ${approval.by ?? 'someone'} on ${day(approval.at)} · binds content ${short(approval.content_sha256)}`;
  else if (v.status === 'superseded') line = 'A later version replaced this one.';
  else if (last?.decision === 'changes_requested') line = `Sent back by ${last.by ?? 'someone'}: “${last.reason ?? ''}”`;
  return (
    <div className="lw-review-banner" data-state={reviewState(v).tone}>
      <ReviewPill v={v} />
      <span className="lw-meta">{line}</span>
    </div>
  );
}

export function ReviewDecisions({ decisions }: { decisions: ReviewDecision[] }) {
  if (decisions.length === 0) return null;
  return (
    <ol className="lw-review-decisions" aria-label="Review decisions">
      {decisions.map((d, i) => (
        <li key={i} className="lw-meta">{`${d.decision === 'approved' ? 'Approved' : 'Changes requested'} by ${d.by ?? 'someone'}, ${day(d.at)}`}
          {d.reason ? ` — ${d.reason}` : ''}{d.decision === 'approved' ? ` · content ${short(d.content_sha256)}` : ''}</li>
      ))}
    </ol>
  );
}

/** The reviewer's decision on exactly the version they read. Approving asks once more, showing the
 *  content hash it binds; asking for changes needs a reason. Never by the version's author. */
export function ApprovalDialog({ v, what, endpoint, canApprove, isAuthor, onDone }: {
  v: ArtifactVersion; what: string; endpoint: string; canApprove: boolean; isAuthor: boolean; onDone: () => void;
}) {
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [read, setRead] = useState(false);
  if (v.status !== 'draft') return null;
  if (!canApprove) return <p className="lw-meta" style={{ margin: 0 }}>Waiting for a lead, attorney or reviewer on this matter to approve.</p>;
  if (isAuthor) return <p className="lw-meta" style={{ margin: 0 }}>You made this version, so another lawyer approves it.</p>;
  const decide = async (decision: 'approved' | 'changes_requested') => {
    setBusy(true); setError('');
    try {
      await api.post(endpoint, { decision, content_sha256: v.content_sha256, reason: decision === 'approved' ? null : reason.trim() });
      setReason(''); setConfirming(false); setRead(false); onDone();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The decision could not be saved.');
      setConfirming(false);
    } finally { setBusy(false); }
  };
  return (
    <div className="lw-review" role="group" aria-label={`Review ${what}`}>
      <p className="lw-meta" style={{ margin: 0 }}>{`Approving binds exactly this version (content ${short(v.content_sha256)}). Any later change needs approval again.`}</p>
      <label className="lw-sr" htmlFor={`reason-${v.id}`}>What should change?</label>
      <input id={`reason-${v.id}`} className="lw-input lw-input-sm" placeholder="What should change? (to send it back)" value={reason}
        onChange={(e) => setReason(e.target.value)} />
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={() => setConfirming(true)}><Check />{`Approve ${what}`}</button>
        <button type="button" className="lw-btn lw-btn-sm" disabled={busy || reason.trim().length < 3} onClick={() => decide('changes_requested')}><Undo2 />Request changes</button>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {confirming && (
        <Dialog title={`Approve version ${v.version}?`} onClose={() => setConfirming(false)}
          footer={<>
            <button type="button" className="lw-btn" onClick={() => setConfirming(false)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!read || busy} onClick={() => decide('approved')}><Check />{`Approve version ${v.version}`}</button>
          </>}>
          <div style={{ display: 'grid', gap: 10 }}>
            <p style={{ margin: 0 }}>{`You are approving version ${v.version}, ${v.origin === 'model' ? 'drafted by the model for' : 'edited by'} ${v.created_by_name ?? 'another lawyer'}, exactly as you read it.`}</p>
            <p className="lw-meta" style={{ margin: 0 }}>Content <code className="lw-hash">{v.content_sha256}</code></p>
            <p className="lw-meta" style={{ margin: 0 }}>It can then be filed as a matter document carrying your name and this hash. Any later change is a new version that needs approval again.</p>
            <label className="lw-check"><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} /> I have read this version and checked its sources</label>
          </div>
        </Dialog>
      )}
    </div>
  );
}
