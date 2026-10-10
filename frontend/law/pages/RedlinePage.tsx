import React, { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { ArrowLeft, Download, FileOutput, PenLine, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, useArtifact } from '../api/queries';
import type { RedlineChange, RedlineContent } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Skeleton } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';

/** The redline as the eye reads it: what is struck and what is added, word by word. */
export function Redline({ ops }: { ops: RedlineChange['ops'] }) {
  return (
    <p className="lw-serif lw-redline">
      {ops.map(([op, text], i) => (op === 'del' ? <del key={i}>{text}</del> : op === 'ins' ? <ins key={i}>{text}</ins> : <span key={i}>{text}</span>))}
    </p>
  );
}

/** A redline (S3.3, D-047): the clauses found outside the playbook, revised to the firm's positions;
 *  edited as new versions, approved by a second lawyer, filed as the document's next version in Word
 *  with tracked changes. */
export default function RedlinePage() {
  const { artifactId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const view = useArtifact<RedlineContent>(matter.id, artifactId);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  if (view.isPending) return <><Toolbar crumb={`${matter.number} · Documents`} title="Redline" /><div className="lw-content"><div className="lw-paper lw-sheet"><Skeleton /></div></div></>;
  if (view.isError) return <><Toolbar crumb={`${matter.number} · Documents`} title="Redline" /><div className="lw-content"><ErrorNote>This redline is not available.</ErrorNote></div></>;
  const { artifact: v, can, versions } = view.data;
  const c = v.content;
  const latest = versions[0]?.id === v.id;
  const canDraft = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'artifact'] }), qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const file = async () => {
    setBusy(true); setError('');
    try { await api.post(`${artifactPath(matter.id, v.id)}/file`); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The redline could not be filed.'); }
    finally { setBusy(false); }
  };
  const docLink = `/m/${matter.number}/documents/${encodeURIComponent(c.document.id)}/read`;
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={c.title}>
        <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents/${encodeURIComponent(c.document.id)}/playbook`}><ArrowLeft />Playbook check</Link>
        <a className="lw-btn lw-btn-sm" href={`/api/legal/matters/${encodeURIComponent(matter.id)}/redlines/${encodeURIComponent(v.id)}/word`} download><Download />Word, with tracked changes</a>
      </Toolbar>
      <div className="lw-content lw-draft-grid">
        <article className="lw-paper lw-sheet" aria-label="Redline">
          <ReviewBanner v={v} made={`Version ${v.version} · ${v.origin === 'model' ? 'proposed from the playbook check' : `edited by ${v.created_by_name}`} · ${c.playbook.name ?? 'playbook'} · document version ${c.document.version_number}`} />
          {!latest && <p className="lw-meta" style={{ margin: '0 0 10px' }}>A later version of this redline exists.</p>}
          {editing ? <RedlineEditor v={v} onClose={() => setEditing(false)} onSaved={async (id) => { setEditing(false); await refresh(); navigate(`/m/${matter.number}/redlines/${encodeURIComponent(id)}`); }} />
            : (
              <ol className="lw-redline-changes" aria-label="Changes">
                {c.changes.map((ch) => (
                  <li key={ch.id} aria-label={`${ch.title}, clause ${ch.clause.number ?? ''}`}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
                      <b>{ch.title}</b><span className="lw-meta">{`Clause ${ch.clause.number ?? '—'} · to the firm's ${ch.from.kind === 'standard' ? 'standard' : `fallback ${ch.from.n}`} (position version ${ch.from.position_version})`}</span>
                    </div>
                    <Redline ops={ch.ops} />
                    {ch.reason && <p className="lw-meta" style={{ margin: 0 }}>{`Why: ${ch.reason}`}</p>}
                    <p className="lw-meta" style={{ margin: 0 }}>{ch.fitted ? "Fitted to this contract's own terms; the firm's figures and obligations kept." : ch.fit_note ?? "The firm's wording as approved."}
                      {ch.edited_by ? ` · Edited by ${ch.edited_by}` : ''}</p>
                  </li>
                ))}
              </ol>
            )}
        </article>
        <aside className="lw-paper lw-sheet" aria-label="Review">
          <p className="lw-meta" style={{ marginTop: 0 }}>The Word file carries real tracked changes (author, date, and a comment with each reason), checked before it is sent: accepting every change gives this revision, rejecting gives the original. Not yet verified by a person in Microsoft Word.</p>
          {!editing && canDraft && latest && v.status !== 'superseded' && v.status !== 'approved' && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing(true)}><PenLine />Edit the revisions</button>}
          <div style={{ marginTop: 12 }}>
            <ApprovalDialog v={v} what="this redline" endpoint={`${artifactPath(matter.id, v.id)}/review`} canApprove={can.approve || can.author} isAuthor={can.author} onDone={refresh} />
            {v.status === 'approved' && (v.document_id
              ? <Link className="lw-btn lw-btn-sm" style={{ marginTop: 10 }} to={docLink}><FileOutput />Open the document</Link>
              : can.file && <button type="button" className="lw-btn lw-btn-sm" style={{ marginTop: 10 }} disabled={busy} onClick={file}><FileOutput />{busy ? 'Filing…' : 'File as the next version'}</button>)}
            {error && <ErrorNote>{error}</ErrorNote>}
            <ReviewDecisions decisions={v.decisions} />
          </div>
        </aside>
      </div>
    </>
  );
}

function RedlineEditor({ v, onClose, onSaved }: { v: { id: string; content: RedlineContent; content_sha256: string; version: number }; onClose: () => void; onSaved: (id: string) => void }) {
  const matter = useMatter();
  const [texts, setTexts] = useState<Record<string, string>>(Object.fromEntries(v.content.changes.map((c) => [c.id, c.revised])));
  const [removed, setRemoved] = useState<string[]>([]);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const save = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    const changes = v.content.changes.map((c) => (removed.includes(c.id) ? { id: c.id, remove: true }
      : texts[c.id] !== c.revised ? { id: c.id, revised: texts[c.id] } : null)).filter(Boolean);
    try {
      const r = await api.post<{ redline: { id: string } }>(`/api/legal/matters/${encodeURIComponent(matter.id)}/redlines/${encodeURIComponent(v.id)}/edit`,
        { content_sha256: v.content_sha256, changes, reason: reason.trim() || null });
      onSaved(r.redline.id);
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The changes could not be saved.'); setBusy(false); }
  };
  return (
    <form onSubmit={save} aria-label="Edit the revisions" style={{ display: 'grid', gap: 12 }}>
      {v.content.changes.map((c) => (
        <fieldset key={c.id} className="lw-minutes-edit" data-removed={removed.includes(c.id) || undefined}>
          <legend className="lw-meta">{`${c.title} · clause ${c.clause.number ?? '—'}`}</legend>
          <label className="lw-label">New wording<textarea className="lw-input lw-serif" rows={3} value={texts[c.id]} disabled={removed.includes(c.id)}
            onChange={(e) => setTexts((x) => ({ ...x, [c.id]: e.target.value }))} /></label>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setRemoved((r) => (r.includes(c.id) ? r.filter((x) => x !== c.id) : [...r, c.id]))}>
            <Trash2 />{removed.includes(c.id) ? 'Keep this change' : 'Drop this change'}</button>
        </fieldset>
      ))}
      <label className="lw-label">Why (optional)<input className="lw-input lw-input-sm" value={reason} onChange={(e) => setReason(e.target.value)} /></label>
      <div style={{ display: 'flex', gap: 6 }}>
        <button type="submit" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy}>{busy ? 'Saving…' : `Save as version ${v.version + 1}`}</button>
        <button type="button" className="lw-btn lw-btn-sm" onClick={onClose}>Cancel</button>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
    </form>
  );
}
