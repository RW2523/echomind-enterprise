import React, { useState } from 'react';
import { Link, useParams } from 'react-router';
import { ArrowLeft, FileOutput } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, useArtifact } from '../api/queries';
import type { ResearchEvidence, ResearchNoteContent } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const STATUS: Record<ResearchNoteContent['status'], [string, Tone]> = {
  answered: ['Answered from the documents', 'verified'], partial: ['Partly supported', 'attention'],
  unverified: ['Not verified', 'conflict'], abstained: ['Not answered by the documents', 'neutral'],
};
const FOUND: Record<string, string> = {
  verbatim: 'Found as written', normalized: 'Found (spacing or quotes differ)', not_quoted: 'Cited without a quote', failed: 'Not found in the source',
};

/** A research answer sent for review (S2.7, D-043): fixed as it was given, with its evidence; approved
 *  by a second lawyer for exactly this version, then filed as a matter document. */
export default function ResearchNote() {
  const { artifactId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const view = useArtifact<ResearchNoteContent>(matter.id, artifactId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'artifact', artifactId] }),
    qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/research`}><ArrowLeft />Research</Link>;

  if (view.isPending) return <><Toolbar crumb={`${matter.number} · Research`} title="Research note">{back}</Toolbar>
    <div className="lw-content"><div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div></div></>;
  if (view.isError) return <><Toolbar crumb={`${matter.number} · Research`} title="Research note">{back}</Toolbar>
    <div className="lw-content"><ErrorNote>{view.error instanceof ApiError && view.error.status !== 404 ? view.error.message : 'This note is not available.'}</ErrorNote></div></>;

  const { artifact: v, can, versions } = view.data;
  const c = v.content;
  const [label, tone] = STATUS[c.status];
  const file = async () => {
    setBusy(true); setError('');
    try { await api.post(`${artifactPath(matter.id, v.id)}/file`); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The note could not be filed.'); }
    finally { setBusy(false); }
  };
  return (
    <>
      <Toolbar crumb={`${matter.number} · Research`} title="Research note">{back}</Toolbar>
      <div className="lw-content">
        <article className="lw-paper lw-sheet" aria-label="Research note" aria-busy={busy}>
          <p className="lw-eyebrow" style={{ margin: 0 }}>Question</p>
          <h2 className="lw-h2 lw-serif" style={{ margin: '2px 0 6px' }}>{c.question}</h2>
          <ReviewBanner v={v} made={`Version ${v.version} · sent for review by ${c.asked_by}, ${day(v.created_at)}`} />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
            <StatusPill tone={tone}>{label}</StatusPill>
            <span className="lw-meta">{c.versions === 'signed' ? 'From signed versions' : 'From the latest drafts'} · asked {day(c.asked_at)}</span>
          </div>
          <p className="lw-note-answer lw-serif">{c.answer || 'The documents in this matter do not answer that.'}</p>
          {c.unsupported.length > 0 && <p className="lw-meta" style={{ margin: '8px 0 0' }}>{`${c.unsupported.length} statement(s) cite no source — check them before relying on the answer.`}</p>}

          {c.evidence.length > 0 && (
            <section aria-label="Evidence" style={{ marginTop: 16 }}>
              <h3 className="lw-h3">Evidence</h3>
              <ol className="lw-note-evidence">{c.evidence.map((e, i) => <EvidenceLine key={i} e={e} number={matter.number} />)}</ol>
            </section>
          )}
          {c.conflicts.length > 0 && (
            <section aria-label="Versions that read differently" style={{ marginTop: 16 }}>
              <h3 className="lw-h3">Versions that read differently</h3>
              <ul className="lw-note-evidence">{c.conflicts.map((x, i) => (
                <li key={i} className="lw-meta">{`${x.document_title ?? 'Document'}${x.clause ? `, clause ${x.clause}` : ''}: version ${x.other.number} (${x.other.status}) reads “${x.other.text.slice(0, 200)}”`}</li>
              ))}</ul>
            </section>
          )}

          {error && <ErrorNote>{error}</ErrorNote>}
          <div className="lw-minutes-actions">
            {v.status === 'approved' && (v.document_id
              ? <Link className="lw-btn lw-btn-sm" to={`/m/${matter.number}/documents/${encodeURIComponent(v.document_id)}/read`}><FileOutput />Open the filed note</Link>
              : can.file && <button type="button" className="lw-btn lw-btn-sm" disabled={busy} onClick={file}><FileOutput />{busy ? 'Filing…' : 'File as a matter document'}</button>)}
          </div>
          <ApprovalDialog v={v} what="this note" endpoint={`${artifactPath(matter.id, v.id)}/review`} canApprove={can.approve || can.author}
            isAuthor={can.author} onDone={refresh} />
          <ReviewDecisions decisions={v.decisions} />
          {versions.length > 1 && (
            <p className="lw-meta" style={{ marginTop: 10 }}>{`Versions: ${versions.map((x) => `${x.version} (${x.status})`).join(' · ')}`}</p>
          )}
        </article>
      </div>
    </>
  );
}

function EvidenceLine({ e, number }: { e: ResearchEvidence & { removed?: boolean }; number: string }) {
  const where = [`${e.document_title || 'Document'}, version ${e.version_number}${e.version_status ? ` (${e.version_status})` : ''}`,
    e.clause ? `clause ${e.clause}` : '', e.page_start ? `page ${e.page_start}` : '', e.line ? `line ${e.line}` : ''].filter(Boolean).join(', ');
  const href = e.char_start != null && e.char_end != null && e.document_id
    ? `/m/${number}/documents/${encodeURIComponent(e.document_id)}/read?v=${encodeURIComponent(e.version_id)}&at=${e.char_start}-${e.char_end}` : null;
  return (
    <li data-verification={e.verification}>
      <span className="lw-serif">{`[${e.n}] “${e.quote}”`}</span>
      <div className="lw-meta">{`${where} · ${FOUND[e.verification] ?? e.verification}`}
        {href && <> · <Link className="lw-linklike" to={href}>Open at the quote</Link></>}</div>
    </li>
  );
}
