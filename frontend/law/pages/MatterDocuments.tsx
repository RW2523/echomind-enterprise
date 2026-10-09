import React, { useRef, useState } from 'react';
import { Download, FileUp, RotateCcw, Search, Trash2, X } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { documentsPath, keys, useDocument, useDocuments, useMatterDeletions } from '../api/queries';
import type { Deletion, DocKind, DocStatus, DocVersion, LawDocument, SearchResult } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import { DeleteDialog, HoldNotice, LabelControl, LabelPill } from '../ui/records';

const KINDS: [DocKind, string][] = [['contract', 'Contract'], ['correspondence', 'Correspondence'], ['pleading', 'Pleading'],
  ['evidence', 'Evidence'], ['memo', 'Memo'], ['note', 'Note'], ['other', 'Other']];
const STATUSES: [DocStatus, string][] = [['draft', 'Draft'], ['final', 'Final'], ['executed', 'Executed'], ['received', 'Received']];
const kindLabel = (k: string) => KINDS.find(([v]) => v === k)?.[1] ?? k;
const statusLabel = (s: string) => STATUSES.find(([v]) => v === s)?.[1] ?? s;

const INDEX: Record<DocVersion['index_status'], [string, Tone]> = {
  queued: ['Waiting', 'neutral'], indexing: ['Reading…', 'accent'], indexed: ['Searchable', 'verified'],
  failed: ['Not searchable', 'conflict'], superseded: ['Replaced', 'neutral'],
};
const INDEX_ERRORS: Record<string, string> = {
  rights_revoked: 'The person who imported it lost access before it was read.',
  no_text: 'No text layer — a scan. Text recognition arrives in a later release.',
  integrity: 'The stored file failed its integrity check.',
  error: 'Reading failed. Try again, or tell your administrator.',
};
const FINDINGS: Record<string, [string, string]> = {
  email: ['email address', 'email addresses'], phone: ['phone number', 'phone numbers'], long_number: ['long number', 'long numbers'],
  card: ['card number', 'card numbers'], ssn: ['social security number', 'social security numbers'], id_number: ['ID number', 'ID numbers'],
};
const size = (n: number) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);
const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');
const pages = (a?: number | null, b?: number | null) => (a ? (b && b !== a ? `pp. ${a}–${b}` : `p. ${a}`) : '');

/** The matter's documents: library, import, a quiet inspector, and search over this matter only. */
export default function MatterDocuments() {
  const matter = useMatter();
  const docs = useDocuments(matter.id);
  const [selected, setSelected] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const role = matter.my_role ?? '';
  const canImport = ['lead', 'attorney', 'paralegal'].includes(role) && matter.status === 'open';
  const mayDelete = ['lead', 'attorney'].includes(role);
  const perms = { canSetLabel: ['lead', 'attorney', 'reviewer'].includes(role) && matter.status === 'open',
    canSuggestLabel: role === 'paralegal' && matter.status === 'open', canDelete: mayDelete && !matter.on_hold, onHold: !!matter.on_hold && mayDelete };
  return (
    <>
      <Toolbar crumb={matter.number} title="Documents">
        {canImport && <button type="button" className="lw-btn lw-btn-primary" onClick={() => setImporting(true)}><FileUp />Import</button>}
      </Toolbar>
      <div className="lw-content lw-docs" data-inspecting={selected ? 'true' : undefined}>
        <div style={{ display: 'grid', gap: 16, alignContent: 'start', minWidth: 0 }}>
          <MatterSearch matterId={matter.id} onOpen={setSelected} />
          {mayDelete && <PendingDeletions matterId={matter.id} />}
          {docs.isPending ? <div className="lw-paper lw-sheet" aria-busy="true"><Skeleton /><Skeleton /></div>
            : docs.isError ? <ErrorNote>{docs.error instanceof ApiError ? docs.error.message : 'The documents could not be loaded.'}</ErrorNote>
            : docs.data.length === 0 ? (
              <EmptyState title="No documents yet" action={canImport ? <button type="button" className="lw-btn lw-btn-primary" onClick={() => setImporting(true)}><FileUp />Import a document</button> : undefined}>
                Documents imported here are stored encrypted and can be searched only from within this matter.
              </EmptyState>
            ) : (
              <section className="lw-paper" aria-label="Library">
                <table className="lw-table">
                  <thead><tr><th>Document</th><th>Kind</th><th>Version</th><th>Search</th><th>Updated</th></tr></thead>
                  <tbody>
                    {docs.data.map((d) => {
                      const v = d.current_version;
                      const [label, tone] = v ? INDEX[v.index_status] : ['—', 'neutral' as Tone];
                      return (
                        <tr key={d.id} aria-selected={selected === d.id} className={selected === d.id ? 'lw-row-selected' : undefined}>
                          <td style={{ minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                              <button type="button" className="lw-linklike" onClick={() => setSelected(d.id)}>{d.title}</button>
                              <LabelPill label={d.label} suggested={d.label_suggested} />
                            </div>
                            <div className="lw-meta lw-ellipsis">{v?.filename}</div>
                          </td>
                          <td>{kindLabel(d.kind)}<div className="lw-meta">{statusLabel(d.status)}</div></td>
                          <td className="lw-tnum">v{v?.number ?? '—'}</td>
                          <td><StatusPill tone={tone}>{label}</StatusPill></td>
                          <td className="lw-meta">{when(d.updated_at ?? d.created_at)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </section>
            )}
        </div>
        {selected && <Inspector matterId={matter.id} docId={selected} canEdit={canImport} perms={perms} onClose={() => setSelected(null)} />}
      </div>
      {importing && <ImportDialog matterId={matter.id} documents={docs.data ?? []} onClose={() => setImporting(false)} />}
    </>
  );
}

// ── Search ─────────────────────────────────────────────────────────────────────
function highlight(raw: string, query: string): React.ReactNode {
  const text = raw.replace(/(^|\s)#{1,6}\s+/g, '$1');      // Markdown heading markers are syntax, not words
  const terms = Array.from(new Set(query.toLowerCase().match(/[\p{L}\p{N}@._-]{3,}/gu) ?? [])).sort((a, b) => b.length - a.length);
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : <React.Fragment key={i}>{part}</React.Fragment>));
}

function MatterSearch({ matterId, onOpen }: { matterId: string; onOpen: (docId: string) => void }) {
  const [q, setQ] = useState('');
  const [asked, setAsked] = useState('');
  const [result, setResult] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (!query) return;
    setBusy(true); setError('');
    try { setResult(await api.post<SearchResult>(`${documentsPath(matterId)}/search`, { query })); setAsked(query); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The search could not be run.'); }
    finally { setBusy(false); }
  };
  return (
    <section aria-label="Search this matter" style={{ display: 'grid', gap: 10 }}>
      <form role="search" onSubmit={run} className="lw-searchbar">
        <Search size={16} aria-hidden="true" />
        <label htmlFor="matter-search" className="sr-only">Search this matter’s documents</label>
        <input id="matter-search" className="lw-searchbar-input" placeholder="Search this matter’s documents — a clause, a name, a reference"
          value={q} onChange={(e) => setQ(e.target.value)} />
        {result && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label="Clear search" onClick={() => { setResult(null); setQ(''); }}><X /></button>}
        <button type="submit" className="lw-btn lw-btn-sm" disabled={busy || !q.trim()}>{busy ? 'Searching…' : 'Search'}</button>
      </form>
      {error && <ErrorNote>{error}</ErrorNote>}
      {result && (
        <div aria-live="polite" style={{ display: 'grid', gap: 8 }}>
          {result.abstained || result.passages.length === 0
            ? <p className="lw-meta" style={{ margin: 0, fontSize: 13.5 }}>Nothing in this matter’s documents closely matches “{asked}”.</p>
            : <>
              <p className="lw-meta" style={{ margin: 0 }}>{result.passages.length} passage{result.passages.length === 1 ? '' : 's'} from this matter</p>
              <ol className="lw-passages">
                {result.passages.map((p) => (
                  <li key={p.chunk_id} className="lw-paper lw-passage">
                    <blockquote className="lw-serif">{highlight(p.text, asked)}</blockquote>
                    <button type="button" className="lw-linklike lw-meta" onClick={() => onOpen(p.citation.document_id)}>
                      {p.citation.document_title} · v{p.citation.version_number}{pages(p.citation.page_start, p.citation.page_end) ? ` · ${pages(p.citation.page_start, p.citation.page_end)}` : ''}
                    </button>
                  </li>
                ))}
              </ol>
            </>}
        </div>
      )}
    </section>
  );
}

// ── Inspector ──────────────────────────────────────────────────────────────────
type Perms = { canSetLabel: boolean; canSuggestLabel: boolean; canDelete: boolean; onHold: boolean };

function Inspector({ matterId, docId, canEdit, perms, onClose }: { matterId: string; docId: string; canEdit: boolean; perms: Perms; onClose: () => void }) {
  const doc = useDocument(matterId, docId);
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const base = `${documentsPath(matterId)}/${encodeURIComponent(docId)}`;
  const reindex = async () => {
    setError('');
    try { await api.post(`${base}/reindex`); await qc.invalidateQueries({ queryKey: keys.documents(matterId) }); await doc.refetch(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'It could not be queued again.'); }
  };
  const d = doc.data;
  const v = d?.current_version;
  const findings = d ? Object.entries(d.findings).filter(([, n]) => n > 0) : [];
  return (
    <aside className="lw-paper lw-inspector" aria-label="Document details">
      <div className="lw-card-h">
        <h2 className="lw-h2">Details</h2>
        <button type="button" className="lw-btn lw-btn-sm" aria-label="Close details" onClick={onClose}><X /></button>
      </div>
      {doc.isPending ? <div style={{ padding: 20 }}><Skeleton /><Skeleton /></div>
        : doc.isError || !d ? <div style={{ padding: 20 }}><ErrorNote>This document is not available.</ErrorNote></div>
        : (
          <div style={{ padding: '16px 20px 20px', display: 'grid', gap: 14 }}>
            <div>
              <div className="lw-eyebrow">{kindLabel(d.kind)} · {statusLabel(d.status)}</div>
              <h3 className="lw-serif" style={{ fontSize: 19, lineHeight: '25px', margin: '4px 0 0', fontWeight: 600 }}>{d.title}</h3>
            </div>
            {v && (
              <dl className="lw-facts">
                <dt>File</dt><dd className="lw-ellipsis" title={v.filename}>{v.filename}</dd>
                <dt>Size</dt><dd>{size(v.size_bytes)}{v.pages ? ` · ${v.pages} page${v.pages === 1 ? '' : 's'}` : ''}</dd>
                <dt>Search</dt><dd><StatusPill tone={INDEX[v.index_status][1]}>{INDEX[v.index_status][0]}</StatusPill>
                  {v.index_status === 'indexed' && <span className="lw-meta"> · {v.chunk_count} passage{v.chunk_count === 1 ? '' : 's'}</span>}
                  {v.index_status === 'failed' && <div className="lw-meta">{INDEX_ERRORS[v.index_error ?? 'error'] ?? INDEX_ERRORS.error}</div>}</dd>
                <dt>Imported</dt><dd>{v.uploaded_by} · {when(v.uploaded_at)}</dd>
                <dt>Fingerprint</dt><dd className="lw-mono" title={`SHA-256 ${v.sha256}`}>{v.sha256.slice(0, 16)}…</dd>
              </dl>
            )}
            {findings.length > 0 && (
              <div className="lw-note">
                <b>Personal data found:</b> {findings.map(([k, n]) => `${n} ${(FINDINGS[k] ?? [k, k])[n === 1 ? 0 : 1]}`).join(' · ')}.
                <div className="lw-meta">It stays searchable inside this matter only.</div>
              </div>
            )}
            <div>
              <h3 className="lw-eyebrow" style={{ margin: '0 0 6px' }}>Privilege label</h3>
              <LabelControl matterId={matterId} type="document" objectId={d.id} canSet={perms.canSetLabel} canSuggest={perms.canSuggestLabel}
                onChange={() => qc.invalidateQueries({ queryKey: keys.documents(matterId) })} />
            </div>
            {perms.onHold && <HoldNotice />}
            {error && <ErrorNote>{error}</ErrorNote>}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {v && <a className="lw-btn lw-btn-sm" href={`${base}/versions/${encodeURIComponent(v.id)}/file`} download><Download />Download</a>}
              {canEdit && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setEditing(true)}>Edit details</button>}
              {canEdit && v?.index_status === 'failed' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={reindex}><RotateCcw />Try again</button>}
              {perms.canDelete && <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => setDeleting(true)}><Trash2 />Delete…</button>}
            </div>
            <div>
              <h3 className="lw-eyebrow" style={{ margin: '4px 0 6px' }}>Versions</h3>
              <ol className="lw-versions">
                {d.versions.map((ver) => (
                  <li key={ver.id}>
                    <span className="lw-tnum" style={{ fontWeight: 600 }}>v{ver.number}</span>
                    <span className="lw-meta lw-ellipsis" style={{ flex: 1 }}>{ver.uploaded_by} · {when(ver.uploaded_at)}</span>
                    <a className="lw-linklike lw-meta" href={`${base}/versions/${encodeURIComponent(ver.id)}/file`} download aria-label={`Download version ${ver.number}`}>Download</a>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
      {editing && d && <EditDialog matterId={matterId} doc={d} onClose={() => setEditing(false)} onSaved={() => doc.refetch()} />}
      {deleting && d && (
        <DeleteDialog title={`Delete “${d.title}”`} previewUrl={`${base}/deletion-preview`} deleteUrl={`${base}/delete`}
          onClose={() => setDeleting(false)}
          onDone={async () => {
            setDeleting(false); onClose();
            await Promise.all([qc.invalidateQueries({ queryKey: keys.documents(matterId) }), qc.invalidateQueries({ queryKey: keys.matterDeletions(matterId) })]);
          }} />
      )}
    </aside>
  );
}

function EditDialog({ matterId, doc, onClose, onSaved }: { matterId: string; doc: LawDocument; onClose: () => void; onSaved: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ title: doc.title, kind: doc.kind, status: doc.status });
  const [error, setError] = useState('');
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try {
      await api.patch(`${documentsPath(matterId)}/${encodeURIComponent(doc.id)}`, { ...form, version: doc.version });
      await qc.invalidateQueries({ queryKey: keys.documents(matterId) });
      onSaved(); onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The changes could not be saved.'); }
  };
  return (
    <Dialog title="Edit document details" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Title">{(id) => <input id={id} className="lw-input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />}</Field>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Kind">{(id) => <select id={id} className="lw-input" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as DocKind })}>
            {KINDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          <Field label="Status">{(id) => <select id={id} className="lw-input" value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value as DocStatus })}>
            {STATUSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
        </div>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.title.trim()}>Save</button>
        </div>
      </form>
    </Dialog>
  );
}

// ── Import ─────────────────────────────────────────────────────────────────────
function ImportDialog({ matterId, documents, onClose }: { matterId: string; documents: LawDocument[]; onClose: () => void }) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [form, setForm] = useState({ title: '', kind: 'contract' as DocKind, status: 'draft' as DocStatus, version_of: '' });
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true); setError('');
    const body = new FormData();
    body.append('file', file);
    if (form.version_of) body.append('version_of', form.version_of);
    else { body.append('title', form.title.trim() || file.name); body.append('kind', form.kind); body.append('status', form.status); }
    try {
      const r = await api.upload<{ document: LawDocument; same_content_as: string | null }>(`${documentsPath(matterId)}/import`, body);
      await qc.invalidateQueries({ queryKey: keys.documents(matterId) });
      if (r.same_content_as) { setNotice(`Imported. It has the same content as “${r.same_content_as}”.`); setFile(null); }
      else onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The file could not be imported.'); }
    finally { setBusy(false); }
  };
  return (
    <Dialog title="Import a document" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="File" hint="PDF, Word (.docx), text or Markdown. Stored encrypted; searchable only within this matter.">
          {(id) => <input id={id} ref={fileRef} type="file" className="lw-input" accept=".pdf,.docx,.txt,.md"
            onChange={(e) => { const f = e.target.files?.[0] ?? null; setFile(f); if (f && !form.title) setForm((s) => ({ ...s, title: f.name.replace(/\.[^.]+$/, '') })); }} />}
        </Field>
        {documents.length > 0 && (
          <Field label="This is">{(id) => <select id={id} className="lw-input" value={form.version_of} onChange={(e) => setForm({ ...form, version_of: e.target.value })}>
            <option value="">A new document</option>
            {documents.map((d) => <option key={d.id} value={d.id}>A new version of “{d.title}”</option>)}</select>}</Field>
        )}
        {!form.version_of && <>
          <Field label="Title">{(id) => <input id={id} className="lw-input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />}</Field>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <Field label="Kind">{(id) => <select id={id} className="lw-input" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value as DocKind })}>
              {KINDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
            <Field label="Status">{(id) => <select id={id} className="lw-input" value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value as DocStatus })}>
              {STATUSES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          </div>
        </>}
        {error && <ErrorNote>{error}</ErrorNote>}
        {notice && <p role="status" className="lw-meta" style={{ margin: 0, fontSize: 13.5 }}>{notice}</p>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>{notice ? 'Done' : 'Cancel'}</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!file || busy}><FileUp />{busy ? 'Importing…' : 'Import'}</button>
        </div>
      </form>
    </Dialog>
  );
}

/** Deletions requested in this matter that a records administrator or managing partner has not
 *  approved yet: still restorable. */
function PendingDeletions({ matterId }: { matterId: string }) {
  const qc = useQueryClient();
  const pending = useMatterDeletions(matterId, true);
  const [error, setError] = useState('');
  if (!pending.data?.length) return null;
  const restore = async (d: Deletion) => {
    setError('');
    try {
      await api.post(`/api/legal/deletions/${encodeURIComponent(d.id)}/restore`);
      await Promise.all([qc.invalidateQueries({ queryKey: keys.matterDeletions(matterId) }), qc.invalidateQueries({ queryKey: keys.documents(matterId) })]);
    } catch (e) { setError(e instanceof ApiError ? e.message : 'It could not be restored.'); }
  };
  return (
    <section className="lw-paper" aria-label="Waiting for deletion approval">
      <div className="lw-card-h"><h2 className="lw-h2">Waiting for deletion approval</h2></div>
      <ul className="lw-pending">
        {pending.data.map((d) => (
          <li key={d.id}>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ fontWeight: 600 }} className="lw-ellipsis">{d.object_label}</div>
              <div className="lw-meta">Requested by {d.requested_by} · {when(d.requested_at)} — “{d.reason}”</div>
            </div>
            {d.status === 'blocked' && <StatusPill tone="hold">Blocked by legal hold</StatusPill>}
            <button type="button" className="lw-btn lw-btn-sm" onClick={() => restore(d)}><RotateCcw />Restore</button>
          </li>
        ))}
      </ul>
      {error && <div style={{ padding: '0 20px 14px' }}><ErrorNote>{error}</ErrorNote></div>}
    </section>
  );
}
