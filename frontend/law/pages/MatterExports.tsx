import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { CheckCircle2, Download, FileOutput, Plus, XCircle } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { exportsPath, keys, useExport, useExportPreview, useExports } from '../api/queries';
import type { LawExport, PlanItem } from '../api/types';
import { useMatter } from '../app/matter';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import NotAvailable from './NotAvailable';

export const EXPORT_STATUS: Record<LawExport['status'], [string, Tone]> = {
  draft: ['Draft', 'neutral'], pending_approval: ['Awaiting approval', 'attention'], ready: ['Ready to download', 'verified'],
  failed: ['Not released — failed checks', 'conflict'], rejected: ['Rejected', 'neutral'], expired: ['Expired', 'neutral'],
};
const KIND: Record<string, string> = {
  email: 'Email addresses', phone: 'Phone numbers', long_number: 'Account and reference numbers', card: 'Card numbers', ssn: 'Social security numbers',
  id_number: 'Identity numbers', iban: 'Bank accounts', credential: 'Credentials and keys', matter_number: 'Matter references',
  party: 'Party names', privilege_marker: 'Privilege markers', manual: 'Added by hand',
};
const CHECKS: Record<string, string> = {
  redactions_absent: 'Redacted text is gone from text, metadata and every part of the file',
  metadata_clean: 'No author, title or other properties of the original',
  structure_clean: 'No comments, tracked changes, hidden text, annotations or attachments',
  watermark: 'Watermark on every page',
};
const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');
const canAsk = (role?: string | null) => ['lead', 'attorney', 'paralegal'].includes(role ?? '');

/** Start an export of a document or transcript, then continue on its page. */
export function ExportDialog({ matterId, number, objectType, objectId, title, onClose }: {
  matterId: string; number: string; objectType: 'document' | 'meeting_transcript'; objectId: string; title: string; onClose: () => void;
}) {
  const navigate = useNavigate();
  const [format, setFormat] = useState<'pdf' | 'docx'>('pdf');
  const [purpose, setPurpose] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const r = await api.post<{ export: LawExport }>(exportsPath(matterId), { object_type: objectType, object_id: objectId, format, purpose: purpose.trim() });
      navigate(`/m/${encodeURIComponent(number)}/exports/${r.export.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? (err.status === 403 ? 'You do not have the export right on this matter. Ask its lead.' : err.message) : 'The export could not be started.');
    } finally { setBusy(false); }
  };
  return (
    <Dialog title={`Export “${title}”`} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <p className="lw-meta" style={{ margin: 0, fontSize: 13.5 }}>You review the suggested redactions; a second lawyer approves; the file is then checked, watermarked and released to you for 24 hours.</p>
        <Field label="Format">{(id) => <select id={id} className="lw-input" value={format} onChange={(e) => setFormat(e.target.value as 'pdf' | 'docx')}>
          <option value="pdf">PDF</option><option value="docx">Word (.docx), rebuilt clean</option></select>}</Field>
        <Field label="Purpose" hint="Who it is for and why. Seen by the approver.">{(id) => <textarea id={id} rows={2} className="lw-input" value={purpose} onChange={(e) => setPurpose(e.target.value)} />}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!purpose.trim() || busy}><FileOutput />Prepare export</button>
        </div>
      </form>
    </Dialog>
  );
}

/** The matter's exports. */
export default function MatterExports() {
  const matter = useMatter();
  const list = useExports(matter.id);
  if (!canAsk(matter.my_role)) return <NotAvailable />;
  return (
    <>
      <Toolbar crumb={matter.number} title="Exports" />
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>Files leave EchoMind Law only through here: redactions reviewed, approved by a second lawyer, checked, watermarked, and available for 24 hours. Start one from a document or a meeting transcript.</p>
        {list.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : list.isError ? <ErrorNote>The exports could not be loaded.</ErrorNote>
          : list.data.length === 0 ? <EmptyState title="No exports yet" />
          : (
            <section className="lw-paper" aria-label="Exports">
              <table className="lw-table">
                <thead><tr><th>Export</th><th>Status</th><th>Requested</th><th>Decided</th></tr></thead>
                <tbody>{list.data.map((e) => (
                  <tr key={e.id}>
                    <td><Link className="lw-linklike" to={e.id}>{e.title}</Link><div className="lw-meta">{e.format.toUpperCase()} · {e.purpose}</div></td>
                    <td><StatusPill tone={EXPORT_STATUS[e.status][1]}>{EXPORT_STATUS[e.status][0]}</StatusPill></td>
                    <td className="lw-meta">{e.requested_by} · {when(e.requested_at)}</td>
                    <td className="lw-meta">{e.decided_by ? `${e.decided_by} · ${when(e.decided_at)}` : '—'}</td>
                  </tr>))}</tbody>
              </table>
            </section>
          )}
      </div>
    </>
  );
}

/** One export: preview with the redactions, the plan, approval, and the released file. */
export function MatterExport() {
  const matter = useMatter();
  const session = useSession();
  const { exportId = '' } = useParams();
  const qc = useQueryClient();
  const ex = useExport(matter.id, exportId);
  const [draft, setDraft] = useState<Record<string, boolean>>({});
  const [manual, setManual] = useState('');
  const [ack, setAck] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const e = ex.data;
  useEffect(() => { setDraft({}); }, [e?.id, e?.status]);
  const planKey = useMemo(() => (e ? e.plan.items.filter((i) => i.redact).map((i) => i.id).sort().join(',') : ''), [e]);
  const preview = useExportPreview(matter.id, exportId, planKey);
  if (!canAsk(matter.my_role)) return <NotAvailable />;
  if (ex.isPending) return <div className="lw-content"><Skeleton height={48} /><Skeleton height={240} /></div>;
  if (ex.isError || !e) return <NotAvailable />;

  const base = `${exportsPath(matter.id)}/${encodeURIComponent(e.id)}`;
  const mine = e.requested_by === session.user.username;
  const canApprove = !mine && ['lead', 'attorney'].includes(matter.my_role ?? '');
  const editable = e.status === 'draft' && mine;
  const changed = Object.keys(draft).length > 0;
  const call = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await fn(); await Promise.all([qc.invalidateQueries({ queryKey: keys.export(matter.id, e.id) }), qc.invalidateQueries({ queryKey: keys.exports(matter.id) })]); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'That did not work.'); }
    finally { setBusy(false); }
  };
  const save = () => call(async () => {
    await api.put(`${base}/plan`, { items: Object.entries(draft).map(([id, redact]) => ({ id, redact })), add: manual.trim() ? [manual.trim()] : [] });
    setDraft({}); setManual('');
  });
  const groups = e.plan.items.reduce<Record<string, PlanItem[]>>((acc, i) => ((acc[i.kind] ??= []).push(i), acc), {});
  const protectedLabel = e.label === 'privileged' || e.label === 'work_product';
  const [label, tone] = EXPORT_STATUS[e.status];
  return (
    <>
      <Toolbar crumb={`${matter.number} · Exports`} title={e.title} />
      <div className="lw-content lw-export">
        <section className="lw-paper lw-sheet" aria-label="Status" style={{ gridColumn: '1 / -1', display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <StatusPill tone={tone}>{label}</StatusPill>
          <span className="lw-meta">{e.format.toUpperCase()} · requested by {e.requested_by} · {when(e.requested_at)} — “{e.purpose}”</span>
          {protectedLabel && <StatusPill tone="privileged">Labelled {e.label === 'privileged' ? 'privileged' : 'work product'}</StatusPill>}
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            {e.status === 'draft' && mine && <button type="button" className="lw-btn lw-btn-primary" disabled={busy || changed} title={changed ? 'Save your changes first' : ''}
              onClick={() => call(() => api.post(`${base}/submit`))}>Submit for approval</button>}
            {e.status === 'ready' && <a className="lw-btn lw-btn-primary" href={`${base}/file`} download><Download />Download</a>}
          </span>
        </section>
        <section className="lw-paper lw-sheet" aria-label="Preview">
          <h2 className="lw-h2" style={{ marginBottom: 10 }}>Preview</h2>
          {e.plan.warnings.image_pages.length > 0 && <p className="lw-note" style={{ marginTop: 0 }}>Pages {e.plan.warnings.image_pages.join(', ')} are mostly images. Text inside images cannot be found or redacted until text recognition arrives.</p>}
          {e.format === 'docx' && <p className="lw-meta" style={{ marginTop: 0 }}>The Word file is rebuilt clean; this preview shows its content laid out as PDF.</p>}
          {preview.isPending ? <Skeleton height={300} /> : preview.isError ? <ErrorNote>The preview could not be made.</ErrorNote> : (
            <div className="lw-pages">
              {Array.from({ length: preview.data.pages }, (_, i) => (
                <img key={`${planKey}-${i}`} src={`${base}/preview/${i + 1}?v=${encodeURIComponent(planKey.slice(0, 40))}`} alt={`Page ${i + 1} as it will be released`} loading="lazy" />
              ))}
            </div>
          )}
        </section>
        <section className="lw-paper lw-sheet" aria-label="Redactions" style={{ display: 'grid', gap: 12, alignContent: 'start' }}>
          <h2 className="lw-h2">Redactions</h2>
          {e.plan.items.length === 0 && <p className="lw-meta" style={{ margin: 0 }}>Nothing was found to suggest. Add text below if something must be removed.</p>}
          {Object.entries(groups).map(([kind, items]) => (
            <fieldset key={kind} className="lw-plan-group">
              <legend className="lw-eyebrow">{KIND[kind] ?? kind}</legend>
              {items.map((i) => {
                const on = draft[i.id] ?? i.redact;
                return (
                  <label key={i.id} className="lw-plan-item" data-on={on || undefined}>
                    <input type="checkbox" checked={on} disabled={!editable} onChange={(ev) => setDraft({ ...draft, [i.id]: ev.target.checked })} />
                    <span style={{ minWidth: 0 }}><span className="lw-mono lw-plan-text">{i.text}</span>
                      <span className="lw-meta"> · {i.reason}{i.count > 1 ? ` · ${i.count}×` : ''}</span></span>
                  </label>
                );
              })}
            </fieldset>
          ))}
          {editable && (
            <form onSubmit={(ev) => { ev.preventDefault(); void save(); }} style={{ display: 'grid', gap: 8 }}>
              <Field label="Also remove" hint="Exact text; every occurrence is removed.">{(id) => <input id={id} className="lw-input" value={manual} onChange={(ev) => setManual(ev.target.value)} />}</Field>
              <div><button type="submit" className="lw-btn" disabled={busy || (!changed && !manual.trim())}><Plus />Save redactions</button></div>
            </form>
          )}
          {e.status === 'pending_approval' && !canApprove && (
            <p className="lw-meta" style={{ margin: 0 }}>{mine ? 'Waiting for another lead or attorney on the matter to approve.' : 'A lead or attorney on the matter approves exports.'}</p>
          )}
          {e.status === 'pending_approval' && canApprove && (
            <div className="lw-approve">
              <p className="lw-meta" style={{ margin: 0 }}>Approving renders the file, removes the redacted text, checks the result and releases it to {e.requested_by}. The person who asked cannot approve.</p>
              {protectedLabel && <label style={{ display: 'flex', gap: 8, fontSize: 14 }}><input type="checkbox" checked={ack} onChange={(ev) => setAck(ev.target.checked)} />
                <span>I confirm this {e.label === 'privileged' ? 'privileged' : 'work-product'} document is meant to leave the firm.</span></label>}
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="button" className="lw-btn" disabled={busy} onClick={() => call(() => api.post(`${base}/reject`))}>Reject</button>
                <button type="button" className="lw-btn lw-btn-primary" disabled={busy || (protectedLabel && !ack)}
                  onClick={() => call(() => api.post(`${base}/approve`, { acknowledge_privilege: ack }))}>{busy ? 'Checking…' : 'Approve and release'}</button>
              </div>
            </div>
          )}
          {e.verification && (
            <div aria-label="Checks" role="group">
              <h3 className="lw-eyebrow" style={{ margin: '4px 0 6px' }}>Checks on the released file</h3>
              <ul className="lw-checks">{Object.entries(e.verification.checks).map(([k, ok]) => (
                <li key={k}>{ok ? <CheckCircle2 size={16} style={{ color: 'var(--verified)' }} /> : <XCircle size={16} style={{ color: 'var(--conflict)' }} />}{CHECKS[k] ?? k}</li>))}</ul>
              {e.watermark && <p className="lw-meta" style={{ margin: '6px 0 0' }}>Watermark: <span className="lw-mono">{e.watermark}</span></p>}
              {e.status === 'ready' && e.expires_at && <p className="lw-meta" style={{ margin: '4px 0 0' }}>Available until {when(e.expires_at)} · approved by {e.decided_by}</p>}
            </div>
          )}
          {error && <ErrorNote>{error}</ErrorNote>}
        </section>
      </div>
    </>
  );
}
