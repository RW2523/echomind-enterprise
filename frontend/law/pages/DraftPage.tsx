import React, { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { ArrowLeft, Download, FileOutput, PenLine } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, draftsPath, useArtifact } from '../api/queries';
import type { DraftBlock, DraftContent, DraftField, DraftSource, LegalDraft } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';

const GAP = /(\[\[NEEDS INPUT:[^\]]*\]\])/;
const STATUS: Record<DraftField['status'], [string, Tone]> = {
  sourced: ['From the documents', 'verified'], matter: ['From the matter record', 'neutral'], system: ["Today's date", 'neutral'],
  entered: ['Entered by a lawyer', 'accent'], needs_input: ['Needs input', 'attention'],
};

function Text({ text }: { text: string }) {
  return <>{text.split('\n').map((line, i) => (
    <React.Fragment key={i}>{i > 0 && <br />}{line.split(GAP).map((piece, j) => (GAP.test(piece)
      ? <mark key={j} className="lw-gap">{piece.replace(/^\[\[NEEDS INPUT:\s*/, '').replace(/\]\]$/, '')}</mark> : <React.Fragment key={j}>{piece}</React.Fragment>))}
    </React.Fragment>))}</>;
}

function Quote({ s }: { s: DraftSource }) {
  const matter = useMatter();
  const href = s.char_start != null && s.char_end != null
    ? `/m/${matter.number}/documents/${encodeURIComponent(s.document_id)}/read?v=${encodeURIComponent(s.version_id)}&at=${s.char_start}-${s.char_end}` : null;
  return (
    <span className="lw-meta">{`“${s.quote}” — ${s.document_title}, version ${s.version_number}${s.version_status ? ` (${s.version_status})` : ''}${s.clause ? `, clause ${s.clause}` : ''}`}
      {href && <> · <Link className="lw-linklike" to={href}>Open at the quote</Link></>}</span>
  );
}

/** One draft (S3.1, D-045): the document as it will be filed, its fields with where each came from,
 *  and the review. A lawyer's change is a new version; a second lawyer approves once no gap is left. */
export default function DraftPage() {
  const { artifactId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const view = useArtifact<DraftContent>(matter.id, artifactId);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/drafting`}><ArrowLeft />Drafting</Link>;
  if (view.isPending) return <><Toolbar crumb={`${matter.number} · Drafting`} title="Draft">{back}</Toolbar><div className="lw-content"><div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div></div></>;
  if (view.isError) return <><Toolbar crumb={`${matter.number} · Drafting`} title="Draft">{back}</Toolbar><div className="lw-content"><ErrorNote>This draft is not available.</ErrorNote></div></>;
  const { artifact: v, can, versions } = view.data;
  const c = v.content;
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'artifact'] }),
    qc.invalidateQueries({ queryKey: ['matter', matter.id, 'drafts'] }), qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const canDraft = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const latest = versions[0]?.id === v.id;
  const file = async () => {
    setBusy(true); setError('');
    try { await api.post(`${artifactPath(matter.id, v.id)}/file`); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The draft could not be filed.'); }
    finally { setBusy(false); }
  };
  const made = `Version ${v.version} · ${v.origin === 'model' ? 'drafted from the documents' : `edited by ${v.created_by_name}`} · ${c.template.name} v${c.template.version}`;
  return (
    <>
      <Toolbar crumb={`${matter.number} · Drafting`} title={c.title}>
        {back}
        <a className="lw-btn lw-btn-sm lw-btn-ghost" href={`${draftsPath(matter.id)}/${encodeURIComponent(v.id)}/preview.pdf`} download><Download />PDF</a>
        <a className="lw-btn lw-btn-sm lw-btn-ghost" href={`${draftsPath(matter.id)}/${encodeURIComponent(v.id)}/preview.docx`} download><Download />Word</a>
      </Toolbar>
      <div className="lw-content lw-draft-grid">
        <article className="lw-paper lw-sheet lw-draft" aria-label="Draft document">
          <ReviewBanner v={v} made={made} />
          {c.gaps > 0 && <p className="lw-meta" role="status" style={{ margin: '0 0 10px' }}>{`${c.gaps} ${c.gaps === 1 ? 'gap' : 'gaps'} to fill, highlighted below, before a second lawyer can approve.`}</p>}
          {!latest && <p className="lw-meta" style={{ margin: '0 0 10px' }}>A later version of this draft exists.</p>}
          {c.blocks.map((b, i) => <Block key={i} b={b} />)}
        </article>
        <aside className="lw-paper lw-sheet" aria-label="Fields and review">
          {editing ? <DraftEditor v={v} onClose={() => setEditing(false)}
            onSaved={async (id) => { setEditing(false); await refresh(); navigate(`/m/${matter.number}/drafting/${encodeURIComponent(id)}`); }} />
            : <FieldList fields={c.fields} />}
          {!editing && canDraft && latest && v.status !== 'superseded' && (
            <button type="button" className="lw-btn lw-btn-sm" style={{ marginTop: 10 }} onClick={() => setEditing(true)}><PenLine />Fill in or correct</button>)}
          {error && <ErrorNote>{error}</ErrorNote>}
          <div style={{ marginTop: 14 }}>
            {c.gaps > 0 && v.status === 'draft'
              ? <p className="lw-meta" style={{ margin: 0 }}>{`Approval opens once the ${c.gaps} ${c.gaps === 1 ? 'gap is' : 'gaps are'} filled.`}</p>
              : <ApprovalDialog v={v} what="this draft" endpoint={`${artifactPath(matter.id, v.id)}/review`} canApprove={can.approve || can.author}
                isAuthor={can.author} onDone={refresh} />}
            {v.status === 'approved' && (v.document_id
              ? <Link className="lw-btn lw-btn-sm" style={{ marginTop: 10 }} to={`/m/${matter.number}/documents/${encodeURIComponent(v.document_id)}/read`}><FileOutput />Open the filed document</Link>
              : can.file && <button type="button" className="lw-btn lw-btn-sm" style={{ marginTop: 10 }} disabled={busy} onClick={file}><FileOutput />{busy ? 'Filing…' : 'File as a Word document'}</button>)}
            <ReviewDecisions decisions={v.decisions} />
          </div>
        </aside>
      </div>
    </>
  );
}

function Block({ b }: { b: DraftBlock }) {
  if (b.type === 'heading') {
    const H = (b.level ?? 1) <= 1 ? 'h2' : 'h3';
    return <H className={`lw-serif lw-draft-h${b.level ?? 1}`}><Text text={b.text} /></H>;
  }
  if (b.origin === 'template') return <p className="lw-serif lw-draft-p"><Text text={b.text} /></p>;
  return (
    <div className="lw-draft-written" data-origin={b.origin}>
      <p className="lw-serif lw-draft-p"><Text text={b.text} /></p>
      <details>
        <summary className="lw-meta">{b.origin === 'edited' ? `Written from the documents, then edited by ${b.edited_by}` : 'Written from the documents'}
          {` · ${b.sources?.length ?? 0} quoted ${b.sources?.length === 1 ? 'source' : 'sources'}`}</summary>
        <ul>{(b.sources ?? []).map((s, i) => <li key={i}><Quote s={s} /></li>)}</ul>
      </details>
      {(b.unsupported?.length ?? 0) > 0 && <p className="lw-meta lw-minutes-flag" style={{ margin: '2px 0 0' }}>{`Not supported by a quote: ${b.unsupported!.map((u) => `“${u}”`).join(' ')}`}</p>}
    </div>
  );
}

function FieldList({ fields }: { fields: DraftField[] }) {
  return (
    <>
      <h2 className="lw-h2" style={{ marginTop: 0 }}>Fields</h2>
      <ul className="lw-draft-fields" aria-label="Fields">
        {fields.map((f) => {
          const [label, tone] = STATUS[f.status];
          return (
            <li key={f.key} aria-label={`${f.label}: ${f.value ?? 'needs input'}`}>
              <div style={{ display: 'flex', gap: 6, alignItems: 'baseline', flexWrap: 'wrap' }}>
                <b>{f.label}</b><StatusPill tone={tone}>{f.status === 'entered' && f.edited_by ? `Entered by ${f.edited_by}` : label}</StatusPill>
              </div>
              <div className="lw-serif">{f.value ?? '—'}</div>
              {f.sources.map((s, i) => <div key={i}><Quote s={s} /></div>)}
              {f.suggestion && <div className="lw-meta lw-minutes-flag">{`The model suggested “${f.suggestion.value}”, not used: ${f.suggestion.reason}.`}</div>}
            </li>
          );
        })}
      </ul>
    </>
  );
}

function DraftEditor({ v, onClose, onSaved }: { v: LegalDraft; onClose: () => void; onSaved: (id: string) => void }) {
  const matter = useMatter();
  const [values, setValues] = useState<Record<string, string>>(Object.fromEntries(v.content.fields.map((f) => [f.key, f.value ?? ''])));
  const written = v.content.blocks.map((b, i) => [b, i] as const).filter(([b]) => b.origin !== 'template');
  const [texts, setTexts] = useState<Record<number, string>>(Object.fromEntries(written.map(([b, i]) => [i, b.text])));
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const save = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true); setError('');
    const fields = v.content.fields.filter((f) => (values[f.key] ?? '') !== (f.value ?? '')).map((f) => ({ key: f.key, value: values[f.key] || null }));
    const blocks = written.filter(([b, i]) => texts[i] !== b.text).map(([, i]) => ({ index: i, text: texts[i] }));
    try {
      const r = await api.post<{ draft: { id: string } }>(`${draftsPath(matter.id)}/${encodeURIComponent(v.id)}/edit`,
        { content_sha256: v.content_sha256, fields, blocks, reason: reason.trim() || null });
      onSaved(r.draft.id);                             // a change is a new version, with its own id
    }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The changes could not be saved.'); setBusy(false); }
  };
  return (
    <form onSubmit={save} aria-label="Fill in or correct the draft" style={{ display: 'grid', gap: 10 }}>
      <h2 className="lw-h2" style={{ margin: 0 }}>Fill in or correct</h2>
      {v.content.fields.map((f) => (
        <label key={f.key} className="lw-label">{f.label}{f.status === 'needs_input' ? ' (needs input)' : ''}
          <input className="lw-input lw-input-sm" value={values[f.key] ?? ''} placeholder={f.suggestion ? `Suggested: ${f.suggestion.value}` : ''}
            onChange={(e) => setValues((x) => ({ ...x, [f.key]: e.target.value }))} /></label>
      ))}
      {written.map(([b, i]) => (
        <label key={i} className="lw-label">{`Written passage: ${b.instruction ?? ''}`}
          <textarea className="lw-input lw-serif" rows={4} value={texts[i]} onChange={(e) => setTexts((x) => ({ ...x, [i]: e.target.value }))} /></label>
      ))}
      <label className="lw-label">Why (optional)<input className="lw-input lw-input-sm" value={reason} onChange={(e) => setReason(e.target.value)} /></label>
      <p className="lw-meta" style={{ margin: 0 }}>Values you enter are marked as yours. The template’s own wording is the firm’s and changes only in a new template version.</p>
      <div style={{ display: 'flex', gap: 6 }}>
        <button type="submit" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy}>{busy ? 'Saving…' : `Save as version ${v.version + 1}`}</button>
        <button type="button" className="lw-btn lw-btn-sm" onClick={onClose}>Cancel</button>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
    </form>
  );
}
