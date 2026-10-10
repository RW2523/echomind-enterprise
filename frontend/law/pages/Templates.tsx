import React, { useState } from 'react';
import { Check, FilePlus2, FileUp, Plus, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { useTemplates } from '../api/queries';
import type { FirmTemplate, TemplateField, TemplateKind, TemplateVersion } from '../api/types';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';

export const KIND_LABEL: Record<TemplateKind, string> = {
  engagement_letter: 'Engagement letter', nda: 'Non-disclosure agreement', demand_letter: 'Letter before claim', memo: 'Memorandum',
  contract_review: 'Contract review report', board_minutes: 'Board minutes', custom: 'Other',
};
const SOURCE_LABEL: Record<string, string> = { documents: 'From the documents', matter: 'From the matter record', person: 'Entered by a lawyer', today: "Today's date" };
const day = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString([], { dateStyle: 'medium' }) : '');

/** Firm templates (S3.1, D-045): the firm's precedents, versioned; a managing partner other than the
 *  author approves each version before anyone drafts from it. */
export default function Templates() {
  const q = useTemplates();
  const qc = useQueryClient();
  const [editing, setEditing] = useState<{ template?: FirmTemplate } | null>(null);
  const [error, setError] = useState('');
  const refresh = () => qc.invalidateQueries({ queryKey: ['templates'] });
  const presets = async () => {
    setError('');
    try { await api.post('/api/legal/templates/presets'); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The presets could not be installed.'); }
  };
  const can = q.data?.can;
  return (
    <>
      <Toolbar title="Templates">
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={presets}><FilePlus2 />Install the standard presets</button>}
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing({})}><Plus />New template</button>}
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>The firm's precedents for drafting. Each fact in a draft comes from the matter record, from a quote in the matter's documents, or from a lawyer — never from the model alone. A version is used once a managing partner other than its author approves it.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>The templates could not be loaded.</ErrorNote>
            : q.data.templates.length === 0
              ? <EmptyState title="No templates yet">{can?.manage ? 'Install the standard presets, or write the firm’s own.' : 'A lawyer who manages templates adds them here.'}</EmptyState>
              : q.data.templates.map((t) => <TemplateCard key={t.id} t={t} canManage={!!can?.manage} canApprove={!!can?.approve} onChange={refresh}
                onNewVersion={() => setEditing({ template: t })} />)}
      </div>
      {editing && <TemplateEditor template={editing.template} onClose={() => setEditing(null)} onSaved={async () => { setEditing(null); await refresh(); }} />}
    </>
  );
}

function TemplateCard({ t, canManage, canApprove, onChange, onNewVersion }: {
  t: FirmTemplate; canManage: boolean; canApprove: boolean; onChange: () => void; onNewVersion: () => void;
}) {
  const current = t.versions.find((v) => v.status === 'approved');
  return (
    <section className="lw-paper lw-sheet" aria-label={t.name}>
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{t.name}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{KIND_LABEL[t.kind]}{t.preset ? ' · standard preset' : ''}</p>
        </div>
        {current ? <StatusPill tone="verified">{`In use: version ${current.version}`}</StatusPill> : <StatusPill tone="attention">No approved version</StatusPill>}
      </div>
      <ol className="lw-template-versions" aria-label={`Versions of ${t.name}`}>
        {t.versions.map((v) => <VersionRow key={v.id} t={t} v={v} canManage={canManage} canApprove={canApprove} onChange={onChange} />)}
      </ol>
      {canManage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={onNewVersion}><Plus />New version</button>}
    </section>
  );
}

function VersionRow({ t, v, canManage, canApprove, onChange }: { t: FirmTemplate; v: TemplateVersion; canManage: boolean; canApprove: boolean; onChange: () => void }) {
  const me = useSession();
  const [error, setError] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [read, setRead] = useState(false);
  const base = `/api/legal/templates/${encodeURIComponent(t.id)}/versions/${encodeURIComponent(v.id)}`;
  const run = async (fn: () => Promise<unknown>) => {
    setError('');
    try { await fn(); onChange(); } catch (err) { setError(err instanceof ApiError ? err.message : 'That could not be done.'); }
  };
  const styles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    run(() => api.post(`${base}/styles`, form));
  };
  const tone = v.status === 'approved' ? 'verified' : v.status === 'draft' ? 'attention' : 'neutral';
  const mine = v.created_by === me.user.username;
  return (
    <li>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <b className="lw-tnum">{`Version ${v.version}`}</b>
        <StatusPill tone={tone}>{v.status === 'approved' ? 'Approved' : v.status === 'draft' ? 'Draft — not approved' : 'Retired'}</StatusPill>
        <span className="lw-meta">{`by ${v.created_by ?? 'someone'}, ${day(v.created_at)}`}{v.approved_by ? ` · approved by ${v.approved_by}, ${day(v.approved_at)}` : ''}
          {v.styles ? ` · firm styles: ${v.styles}` : ''} · <span className="lw-mono">{`${v.content_sha256.slice(0, 12)}…`}</span></span>
      </div>
      <details className="lw-template-body">
        <summary className="lw-meta">Wording and fields</summary>
        <pre className="lw-template-pre">{v.body}</pre>
        <ul className="lw-meta" style={{ margin: 0 }}>{v.fields.map((f) => <li key={f.key}><span className="lw-mono">{`{{${f.key}}}`}</span>{` ${f.label} — ${SOURCE_LABEL[f.source]}${f.question ? `: “${f.question}”` : ''}`}</li>)}</ul>
      </details>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
        {v.status === 'draft' && canManage && (
          <label className="lw-btn lw-btn-sm lw-btn-ghost"><FileUp />Attach firm styles (.docx)
            <input type="file" accept=".docx" className="lw-sr" onChange={styles} /></label>
        )}
        {v.status === 'draft' && canApprove && !mine && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => setConfirming(true)}><Check />{`Approve version ${v.version}`}</button>}
        {v.status === 'draft' && canApprove && mine && <span className="lw-meta">You wrote this version, so another managing partner approves it.</span>}
        {v.status === 'approved' && canApprove && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => run(() => api.post(`${base}/retire`))}><Trash2 />Retire</button>}
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {confirming && (
        <Dialog title={`Approve version ${v.version} of ${t.name}?`} onClose={() => setConfirming(false)}
          footer={<>
            <button type="button" className="lw-btn" onClick={() => setConfirming(false)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!read} onClick={() => { setConfirming(false); run(() => api.post(`${base}/approve`, { content_sha256: v.content_sha256 })); }}>
              <Check />{`Approve version ${v.version}`}</button>
          </>}>
          <div style={{ display: 'grid', gap: 10 }}>
            <p style={{ margin: 0 }}>Lawyers will draft from exactly this wording and these fields. The version approved before it is retired.</p>
            <p className="lw-meta" style={{ margin: 0 }}>Content <code className="lw-hash">{v.content_sha256}</code></p>
            <label className="lw-check"><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} /> I have reviewed this wording and its fields</label>
          </div>
        </Dialog>
      )}
    </li>
  );
}

const KINDS: TemplateKind[] = ['engagement_letter', 'nda', 'demand_letter', 'memo', 'contract_review', 'board_minutes', 'custom'];
const FIELD_KINDS = ['text', 'party', 'date', 'money', 'period', 'percent', 'address', 'agreement'];
const blankField = (key = ''): TemplateField => ({ key, label: '', kind: 'text', source: 'documents', question: '' });

function TemplateEditor({ template, onClose, onSaved }: { template?: FirmTemplate; onClose: () => void; onSaved: () => void }) {
  const latest = template?.versions[0];
  const [name, setName] = useState(template?.name ?? '');
  const [kind, setKind] = useState<TemplateKind>(template?.kind ?? 'custom');
  const [body, setBody] = useState(latest?.body ?? '# Title\n\nDear {{client_contact}},\n\n[[write: What the model should write from the documents.]]');
  const [fields, setFields] = useState<TemplateField[]>(latest?.fields ?? [blankField('client_contact')]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const detect = () => {
    const keys = Array.from(new Set(Array.from(body.matchAll(/\{\{\s*([a-z][a-z0-9_]*)\s*\}\}/g)).map((m) => m[1])));
    setFields((fs) => [...fs.filter((f) => keys.includes(f.key)), ...keys.filter((k) => !fs.some((f) => f.key === k)).map((k) => blankField(k))]);
  };
  const set = (i: number, patch: Partial<TemplateField>) => setFields((fs) => fs.map((f, j) => (j === i ? { ...f, ...patch } : f)));
  const save = async () => {
    setBusy(true); setError('');
    const payload = { body, fields: fields.map((f) => ({ ...f, question: f.source === 'documents' ? f.question : undefined })) };
    try {
      if (template) await api.post(`/api/legal/templates/${encodeURIComponent(template.id)}/versions`, payload);
      else await api.post('/api/legal/templates', { name: name.trim(), kind, ...payload });
      onSaved();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The template could not be saved.'); }
    finally { setBusy(false); }
  };
  return (
    <Dialog title={template ? `New version of ${template.name}` : 'New template'} onClose={onClose}
      footer={<><button type="button" className="lw-btn" onClick={onClose}>Cancel</button>
        <button type="button" className="lw-btn lw-btn-primary" disabled={busy || (!template && !name.trim())} onClick={save}>{busy ? 'Saving…' : 'Save as a draft version'}</button></>}>
      <div style={{ display: 'grid', gap: 10, minWidth: 'min(720px, 80vw)' }}>
        {!template && (
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 8 }}>
            <Field label="Name">{(id) => <input id={id} className="lw-input" value={name} onChange={(e) => setName(e.target.value)} />}</Field>
            <Field label="Kind">{(id) => <select id={id} className="lw-input" value={kind} onChange={(e) => setKind(e.target.value as TemplateKind)}>
              {KINDS.map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}</select>}</Field>
          </div>
        )}
        <Field label="Wording" hint="# Heading · {{field}} · [[write: an instruction]] for a passage written from the documents, citing them">
          {(id) => <textarea id={id} className="lw-input lw-mono" rows={12} value={body} onChange={(e) => setBody(e.target.value)} />}</Field>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="lw-eyebrow" style={{ margin: 0 }}>Fields</h3>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={detect}>Match the fields to the wording</button>
        </div>
        <table className="lw-table" aria-label="Fields">
          <thead><tr><th scope="col">Field</th><th scope="col">Label</th><th scope="col">Kind</th><th scope="col">Comes from</th><th scope="col">Question to the documents</th><th /></tr></thead>
          <tbody>{fields.map((f, i) => (
            <tr key={i}>
              <td><input aria-label={`Key of field ${i + 1}`} className="lw-input lw-input-sm lw-mono" value={f.key} onChange={(e) => set(i, { key: e.target.value })} /></td>
              <td><input aria-label={`Label of ${f.key || `field ${i + 1}`}`} className="lw-input lw-input-sm" value={f.label} onChange={(e) => set(i, { label: e.target.value })} /></td>
              <td><select aria-label={`Kind of ${f.key || `field ${i + 1}`}`} className="lw-input lw-input-sm" value={f.kind} onChange={(e) => set(i, { kind: e.target.value })}>
                {FIELD_KINDS.map((k) => <option key={k}>{k}</option>)}</select></td>
              <td><select aria-label={`Source of ${f.key || `field ${i + 1}`}`} className="lw-input lw-input-sm" value={f.source} onChange={(e) => set(i, { source: e.target.value as TemplateField['source'] })}>
                {Object.entries(SOURCE_LABEL).map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></td>
              <td><input aria-label={`Question for ${f.key || `field ${i + 1}`}`} className="lw-input lw-input-sm" disabled={f.source !== 'documents'} value={f.question ?? ''} onChange={(e) => set(i, { question: e.target.value })} /></td>
              <td><button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Remove ${f.key || `field ${i + 1}`}`} onClick={() => setFields((fs) => fs.filter((_, j) => j !== i))}><Trash2 /></button></td>
            </tr>
          ))}</tbody>
        </table>
        <div><button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setFields((fs) => [...fs, blankField()])}><Plus />Add a field</button></div>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    </Dialog>
  );
}
