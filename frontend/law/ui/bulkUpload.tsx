/**
 * Importing many documents at once (case management, 2026-11-07): several files, a whole folder, or a zip. Each file goes
 * through the same import as a single one (checked, stored encrypted, read into the matter's own search); a zip is unpacked
 * by the server, which says why any file in it was skipped. Files are sent one at a time, each with its own outcome.
 */
import React, { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, FileUp, FolderUp, Loader2, TriangleAlert } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys } from '../api/queries';
import type { LawDocument } from '../api/types';
import { Field } from './primitives';

export const ACCEPT = '.pdf,.docx,.txt,.md,.zip';
const SUPPORTED = /\.(pdf|docx|txt|md)$/i;
const SYSTEM = /(^|\/)(\.|__MACOSX\/|thumbs\.db$|desktop\.ini$)/i;
type Kind = 'contract' | 'correspondence' | 'pleading' | 'evidence' | 'memo' | 'note' | 'other';
type Status = 'draft' | 'final' | 'executed' | 'received';
const KINDS: [Kind, string][] = [['other', 'Mixed / other'], ['contract', 'Contracts'], ['correspondence', 'Correspondence'],
  ['pleading', 'Pleadings'], ['evidence', 'Evidence'], ['memo', 'Memos'], ['note', 'Notes']];
const STATUSES: [Status, string][] = [['received', 'Received'], ['final', 'Final'], ['executed', 'Signed (executed)'], ['draft', 'Draft']];

export interface UploadRow { name: string; state: 'waiting' | 'sending' | 'done' | 'same' | 'skipped' | 'failed'; note?: string }

/** The files to send: supported documents and zips, without system files; the rest are listed as skipped. */
export function sortFiles(files: File[]): { send: File[]; skipped: UploadRow[] } {
  const send: File[] = [];
  const skipped: UploadRow[] = [];
  for (const f of files) {
    const path = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    if (SYSTEM.test(path)) continue;
    if (SUPPORTED.test(f.name) || /\.zip$/i.test(f.name)) send.push(f);
    else skipped.push({ name: f.name, state: 'skipped', note: 'Only PDF, Word (.docx), text, Markdown and zip files can be imported.' });
  }
  return { send, skipped };
}

export async function sendFiles(matterId: string, files: File[], kind: Kind, status: Status, onRow: (rows: UploadRow[]) => void): Promise<UploadRow[]> {
  const base = `/api/legal/matters/${encodeURIComponent(matterId)}/documents`;
  const { send, skipped } = sortFiles(files);
  let rows: UploadRow[] = [...send.map((f) => ({ name: f.name, state: 'waiting' as const })), ...skipped];
  const set = (i: number, r: Partial<UploadRow>) => { rows = rows.map((x, j) => (j === i ? { ...x, ...r } : x)); onRow(rows); };
  onRow(rows);
  for (let i = 0; i < send.length; i++) {
    const f = send[i];
    set(i, { state: 'sending' });
    const body = new FormData();
    body.append('file', f);
    body.append('kind', kind);
    body.append('status', status);
    try {
      if (/\.zip$/i.test(f.name)) {
        const r = await api.upload<{ imported: { name: string }[]; skipped: { name: string; reason: string }[] }>(`${base}/import-zip`, body);
        set(i, { state: 'done', note: `${r.imported.length} file${r.imported.length === 1 ? '' : 's'} imported from the zip` +
          (r.skipped.length ? `, ${r.skipped.length} skipped` : '') });
        rows = [...rows, ...r.skipped.map((s) => ({ name: `${f.name} › ${s.name}`, state: 'skipped' as const, note: s.reason }))];
        onRow(rows);
      } else {
        body.append('title', f.name.replace(/\.[^.]+$/, ''));
        const r = await api.upload<{ document: LawDocument; same_content_as: string | null }>(`${base}/import`, body);
        set(i, r.same_content_as ? { state: 'same', note: `Same content as “${r.same_content_as}”` } : { state: 'done' });
      }
    } catch (e) {
      set(i, { state: 'failed', note: e instanceof ApiError ? e.message : 'It could not be imported.' });
    }
  }
  return rows;
}

const ICON: Record<UploadRow['state'], React.ReactNode> = {
  waiting: <span className="lw-meta" aria-hidden>·</span>, sending: <Loader2 aria-hidden className="lw-spin" />,
  done: <CheckCircle2 aria-hidden style={{ color: 'var(--verified, #2f7d4f)' }} />, same: <CheckCircle2 aria-hidden />,
  skipped: <TriangleAlert aria-hidden style={{ color: 'var(--attention, #a86b00)' }} />, failed: <TriangleAlert aria-hidden style={{ color: 'var(--conflict, #b3261e)' }} />,
};
const STATE: Record<UploadRow['state'], string> = { waiting: 'Waiting', sending: 'Importing', done: 'Imported', same: 'Imported', skipped: 'Skipped', failed: 'Not imported' };

export function UploadList({ rows }: { rows: UploadRow[] }) {
  if (!rows.length) return null;
  return (
    <ul className="lw-upload-list" aria-label="Files" aria-live="polite">
      {rows.map((r, i) => (
        <li key={`${r.name}-${i}`} data-state={r.state}>
          {ICON[r.state]}
          <span className="lw-upload-name" title={r.name}>{r.name}</span>
          <span className="lw-meta">{STATE[r.state]}{r.note ? `: ${r.note}` : ''}</span>
        </li>
      ))}
    </ul>
  );
}

/** Choose or drop files, a folder or a zip, then import them all into the matter. */
export default function BulkUpload({ matterId, onDone, compact = false }: { matterId: string; onDone?: (rows: UploadRow[]) => void; compact?: boolean }) {
  const qc = useQueryClient();
  const files = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState<UploadRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [kind, setKind] = useState<Kind>('other');
  const [status, setStatus] = useState<Status>('received');
  const run = async (list: File[]) => {
    if (!list.length || busy) return;
    setBusy(true);
    try {
      const done = await sendFiles(matterId, list, kind, status, setRows);
      await qc.invalidateQueries({ queryKey: keys.documents(matterId) });
      onDone?.(done);
    } finally { setBusy(false); }
  };
  const counts = rows.reduce((a, r) => ({ ...a, [r.state]: (a[r.state] ?? 0) + 1 }), {} as Record<string, number>);
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {!compact && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="The documents are">{(id) => <select id={id} className="lw-input" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {KINDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          <Field label="Status">{(id) => <select id={id} className="lw-input" value={status} onChange={(e) => setStatus(e.target.value as Status)}>
            {STATUSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
        </div>
      )}
      <div className="lw-dropzone" data-over={over ? 'true' : undefined}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); void run(Array.from(e.dataTransfer.files)); }}>
        <FileUp aria-hidden />
        <p style={{ margin: 0 }}><b>Drop files, a folder's files or a zip here</b></p>
        <p className="lw-meta" style={{ margin: 0 }}>PDF, Word (.docx), text and Markdown; a zip is unpacked and each file checked. Up to 200 files a zip.</p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
          <button type="button" className="lw-btn" disabled={busy} onClick={() => files.current?.click()}><FileUp />Choose files</button>
          <button type="button" className="lw-btn" disabled={busy} onClick={() => folder.current?.click()}><FolderUp />Choose a folder</button>
        </div>
        <input ref={files} type="file" multiple accept={ACCEPT} hidden aria-label="Choose files"
          onChange={(e) => { void run(Array.from(e.target.files ?? [])); e.target.value = ''; }} />
        <input ref={folder} type="file" multiple hidden aria-label="Choose a folder"
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
          onChange={(e) => { void run(Array.from(e.target.files ?? [])); e.target.value = ''; }} />
      </div>
      {rows.length > 0 && (
        <p role="status" className="lw-meta" style={{ margin: 0 }}>
          {busy ? 'Importing… ' : ''}{(counts.done ?? 0) + (counts.same ?? 0)} imported{counts.skipped ? `, ${counts.skipped} skipped` : ''}
          {counts.failed ? `, ${counts.failed} not imported` : ''}. Each document is read into the matter's search in the background.
        </p>
      )}
      <UploadList rows={rows} />
    </div>
  );
}
