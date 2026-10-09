import React, { useEffect, useState } from 'react';
import { Lock, ShieldAlert, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, labelsPath, useLabels } from '../api/queries';
import type { Deletion, DeletionReport, PrivilegeLabel } from '../api/types';
import { Dialog, ErrorNote, Field, Skeleton, StatusPill, type Tone } from './primitives';

/** P-004 safe defaults: advisory labels (they never change who can see a document). */
export const LABELS: [PrivilegeLabel, string, Tone][] = [
  ['privileged', 'Privileged', 'privileged'], ['work_product', 'Work product', 'privileged'], ['confidential', 'Confidential', 'attention'],
  ['not_privileged', 'Not privileged', 'neutral'], ['unreviewed', 'Unreviewed', 'neutral'],
];
const MARKERS: Record<string, string> = {
  attorney_client: '“attorney-client privilege”', privileged_confidential: '“privileged and confidential”',
  legal_privilege: '“legally privileged”', without_prejudice: '“without prejudice”', work_product: '“work product”',
};
export const labelText = (l?: string | null) => LABELS.find(([v]) => v === l)?.[1] ?? 'Unreviewed';

export function LabelPill({ label, suggested }: { label?: PrivilegeLabel | null; suggested?: boolean }) {
  if (!label) return suggested ? <StatusPill tone="neutral">Label suggested</StatusPill> : null;
  const entry = LABELS.find(([v]) => v === label);
  return <StatusPill tone={entry?.[2] ?? 'neutral'} icon={label === 'privileged' || label === 'work_product' ? <Lock /> : undefined}>{entry?.[1] ?? label}</StatusPill>;
}

/** The label of one document or meeting. Leads, attorneys and reviewers set it; a paralegal's
 *  choice is recorded as a suggestion. */
export function LabelControl({ matterId, type, objectId, canSet, canSuggest, onChange }: {
  matterId: string; type: 'document' | 'meeting'; objectId: string; canSet: boolean; canSuggest: boolean; onChange?: () => void;
}) {
  const qc = useQueryClient();
  const st = useLabels(matterId, type, objectId);
  const [error, setError] = useState('');
  const choose = async (label: PrivilegeLabel) => {
    setError('');
    try {
      const next = await api.post(labelsPath(matterId, type, objectId), { label });
      qc.setQueryData(keys.labels(matterId, type, objectId), next);
      onChange?.();
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The label could not be saved.'); }
  };
  if (st.isPending) return <Skeleton />;
  if (st.isError) return null;
  const { current, suggestions } = st.data;
  return (
    <div className="lw-label-control" aria-label="Privilege label" role="group">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        {current ? <LabelPill label={current.label} /> : <StatusPill tone="neutral">Unreviewed</StatusPill>}
        {current && <span className="lw-meta">set by {current.by} · {new Date(current.at).toLocaleDateString()}</span>}
        {(canSet || canSuggest) && (
          <select aria-label={canSet ? 'Set label' : 'Suggest label'} className="lw-input lw-input-sm" value=""
            onChange={(e) => e.target.value && choose(e.target.value as PrivilegeLabel)}>
            <option value="">{canSet ? 'Change…' : 'Suggest…'}</option>
            {LABELS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        )}
      </div>
      {suggestions.map((s) => (
        <div key={`${s.label}-${s.at}`} className="lw-suggestion">
          <span>Suggested: <b>{labelText(s.label)}</b>{' '}
            <span className="lw-meta">{s.source === 'detector' ? `— the text contains ${MARKERS[s.reason_code ?? ''] ?? 'a privilege marker'}` : `— by ${s.by}`}</span></span>
          {canSet && <button type="button" className="lw-btn lw-btn-sm" onClick={() => choose(s.label)}>Accept</button>}
        </div>
      ))}
      <p className="lw-meta" style={{ margin: 0 }}>Labels advise; they do not change who can see this.</p>
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}

const n = (v?: number) => v ?? 0;
const plural = (count: number, one: string, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

export function ReportList({ report }: { report: DeletionReport }) {
  const removed = report.object === 'document'
    ? [plural(n(report.versions), 'version'), plural(n(report.passages), 'searchable passage'), plural(n(report.files), 'stored file'),
       `${plural(n(report.index_vectors), 'entry', 'entries')} in the matter’s search index`,
       n(report.findings) ? plural(n(report.findings), 'personal-data finding') : '',
       n(report.labels) ? plural(n(report.labels), 'label') : '',
       n(report.research_citations) ? `${plural(n(report.research_citations), 'research answer')} quoting it (the quotes are removed)` : '']
      .filter(Boolean)
    : [plural(n(report.segments), 'transcript segment')];
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <div><div className="lw-eyebrow">Will be removed</div><ul className="lw-report">{removed.map((r) => <li key={r}>{r}</li>)}</ul></div>
      <div><div className="lw-eyebrow">Kept</div><ul className="lw-report">{report.kept.map((r) => <li key={r}>{r}</li>)}</ul></div>
    </div>
  );
}

/** Request a deletion: shows the dry-run report, asks for a reason. The item is hidden at once and
 *  destroyed only after a records administrator or managing partner approves. */
export function DeleteDialog({ title, previewUrl, deleteUrl, onClose, onDone }: {
  title: string; previewUrl: string; deleteUrl: string; onClose: () => void; onDone: (d: Deletion) => void;
}) {
  const [report, setReport] = useState<DeletionReport | null>(null);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api.get<{ report: DeletionReport }>(previewUrl).then((r) => setReport(r.report))
      .catch((e) => setError(e instanceof ApiError ? e.message : 'The deletion preview could not be loaded.'));
  }, [previewUrl]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    try { onDone((await api.post<{ deletion: Deletion }>(deleteUrl, { reason: reason.trim() })).deletion); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The deletion could not be requested.'); }
    finally { setBusy(false); }
  };
  return (
    <Dialog title={title} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <p className="lw-meta" style={{ margin: 0, fontSize: 13.5 }}>It disappears from this matter at once and can be restored until a records administrator or managing partner approves; then it is destroyed and the result is verified.</p>
        {report ? <ReportList report={report} /> : !error && <Skeleton />}
        <Field label="Reason" hint="Seen by the approver and kept with the deletion record.">
          {(id) => <textarea id={id} rows={2} className="lw-input" value={reason} onChange={(e) => setReason(e.target.value)} />}
        </Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-danger-solid" disabled={!reason.trim() || !report || busy}><Trash2 />Request deletion</button>
        </div>
      </form>
    </Dialog>
  );
}

export function HoldNotice() {
  return (
    <div className="lw-note" role="note" style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
      <ShieldAlert size={18} style={{ color: 'var(--hold)', flex: 'none', marginTop: 1 }} />
      <span><b>Legal hold.</b> Nothing in this matter can be deleted while the hold is in force. A records administrator can tell you more.</span>
    </div>
  );
}
