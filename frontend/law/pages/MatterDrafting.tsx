import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { FilePen, Plus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { draftsPath, useDocuments, useDrafts, useTemplates } from '../api/queries';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { KIND_LABEL } from './Templates';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** Drafting (S3.1, D-045): documents drafted from the firm's approved templates, every fact from
 *  the matter record, a checked quote or a lawyer; reviewed by a second lawyer; filed as Word. */
export default function MatterDrafting() {
  const matter = useMatter();
  const drafts = useDrafts(matter.id);
  const [creating, setCreating] = useState(false);
  const canDraft = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  return (
    <>
      <Toolbar crumb={matter.number} title="Drafting">
        {canDraft && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setCreating(true)}><Plus />New draft</button>}
      </Toolbar>
      <div className="lw-content">
        <section className="lw-paper" aria-labelledby="drafts-h">
          <div className="lw-card-h"><h2 className="lw-h2" id="drafts-h">Drafts</h2></div>
          {drafts.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /></div>
            : drafts.isError ? <div style={{ padding: 20 }}><ErrorNote>The drafts could not be loaded.</ErrorNote></div>
              : drafts.data.length === 0
                ? <div style={{ padding: 20 }}><EmptyState title="No drafts yet">{canDraft ? 'Start one from an approved firm template.' : 'Drafts made on this matter appear here.'}</EmptyState></div>
                : (
                  <table className="lw-table" aria-label="Drafts">
                    <thead><tr><th scope="col">Draft</th><th scope="col">Template</th><th scope="col">State</th><th scope="col">Last version</th></tr></thead>
                    <tbody>{drafts.data.map((d) => (
                      <tr key={d.id}>
                        <td><Link to={`/m/${matter.number}/drafting/${encodeURIComponent(d.id)}`} style={{ fontWeight: 600 }}>{d.title}</Link></td>
                        <td className="lw-meta">{`${d.template.name} v${d.template.version}`}</td>
                        <td>{d.status === 'approved' ? <StatusPill tone="verified">{d.document_id ? 'Approved · filed' : 'Approved'}</StatusPill>
                          : d.sent_back ? <StatusPill tone="conflict">Changes requested</StatusPill>
                            : d.gaps ? <StatusPill tone="attention">{`${d.gaps} ${d.gaps === 1 ? 'gap' : 'gaps'} to fill`}</StatusPill>
                              : <StatusPill tone="attention">Draft — not approved</StatusPill>}</td>
                        <td className="lw-meta">{`v${d.version} · ${d.created_by_name ?? ''}, ${day(d.created_at)}`}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                )}
        </section>
      </div>
      {creating && <NewDraft onClose={() => setCreating(false)} />}
    </>
  );
}

function NewDraft({ onClose }: { onClose: () => void }) {
  const matter = useMatter();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const templates = useTemplates();
  const docs = useDocuments(matter.id);
  const usable = (templates.data?.templates ?? []).filter((t) => t.versions.some((v) => v.status === 'approved'));
  const [template, setTemplate] = useState('');
  const [title, setTitle] = useState('');
  const [chosen, setChosen] = useState<string[]>([]);
  const [noteText, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const go = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.post<{ draft: { id: string } }>(draftsPath(matter.id), {
        template_id: template, title: title.trim(), documents: chosen.length ? chosen : null, note: noteText.trim() || null });
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'drafts'] });
      navigate(`/m/${matter.number}/drafting/${encodeURIComponent(r.draft.id)}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The draft could not be made.');
      setBusy(false);
    }
  };
  return (
    <Dialog title="New draft" onClose={onClose}
      footer={<><button type="button" className="lw-btn" onClick={onClose}>Cancel</button>
        <button type="button" className="lw-btn lw-btn-primary" disabled={busy || !template || !title.trim()} onClick={go}>
          <FilePen />{busy ? 'Drafting from the documents…' : 'Draft'}</button></>}>
      <div style={{ display: 'grid', gap: 10, minWidth: 'min(560px, 80vw)' }} aria-busy={busy}>
        <Field label="Template">{(id) => (
          <select id={id} className="lw-input" value={template} onChange={(e) => {
            setTemplate(e.target.value);
            const t = usable.find((x) => x.id === e.target.value);
            if (t && !title) setTitle(`${t.name} — ${matter.title}`);
          }}>
            <option value="">Choose an approved template</option>
            {usable.map((t) => <option key={t.id} value={t.id}>{`${t.name} (${KIND_LABEL[t.kind]}, version ${t.versions.find((v) => v.status === 'approved')!.version})`}</option>)}
          </select>)}</Field>
        {templates.data && usable.length === 0 && <p className="lw-meta" style={{ margin: 0 }}>No template is approved yet. A managing partner approves the firm’s templates under Templates.</p>}
        <Field label="Title">{(id) => <input id={id} className="lw-input" value={title} onChange={(e) => setTitle(e.target.value)} />}</Field>
        {(docs.data?.length ?? 0) > 0 && (
          <fieldset className="lw-agenda-docs">
            <legend className="lw-meta">Draft from these documents only (otherwise every document you can see, signed versions first)</legend>
            {docs.data!.map((d) => (
              <label key={d.id} className="lw-meta">
                <input type="checkbox" checked={chosen.includes(d.id)} onChange={(e) => setChosen((c) => (e.target.checked ? [...c, d.id] : c.filter((x) => x !== d.id)))} /> {d.title}
              </label>
            ))}
          </fieldset>
        )}
        <Field label="Note for the written passages (optional)">{(id) => <textarea id={id} className="lw-input" rows={2} value={noteText} onChange={(e) => setNote(e.target.value)} />}</Field>
        <p className="lw-meta" style={{ margin: 0 }}>Facts come only from the matter record, from quotes the documents contain, or from you. Anything missing is left as a gap to fill.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    </Dialog>
  );
}
