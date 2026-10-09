import React, { useState } from 'react';
import { Link } from 'react-router';
import { api, ApiError } from '../api/client';
import { keys, useApiMutation } from '../api/queries';
import type { Matter, Sensitivity } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { Dialog, ErrorNote, Field, MatterStatusPill, roleLabel, SensitivityPill } from '../ui/primitives';
import { HoldNotice } from '../ui/records';

export default function MatterOverview() {
  const matter = useMatter();
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState('');
  const isLead = matter.my_role === 'lead';
  const setStatus = useApiMutation((status: string) =>
    api.patch(`/api/legal/matters/${encodeURIComponent(matter.id)}`, { status, version: matter.version }), [keys.matters]);
  const changeStatus = async (status: string) => {
    setError('');
    try { await setStatus.mutateAsync(status); } catch (e) { setError(e instanceof ApiError ? e.message : 'The matter could not be updated.'); }
  };
  const rows: [string, React.ReactNode][] = [
    ['Client', matter.client.name], ['Matter number', <span className="lw-tnum">{matter.number}</span>],
    ['Practice area', matter.practice_area || '—'], ['Jurisdiction', matter.jurisdiction || '—'],
    ['Status', <MatterStatusPill value={matter.status} />], ['Sensitivity', <SensitivityPill value={matter.sensitivity} />],
    ['Your role', roleLabel(matter.my_role)],
  ];
  return (
    <>
      <Toolbar crumb={matter.number} title="Overview">
        {isLead && matter.status === 'open' && <button type="button" className="lw-btn" onClick={() => setEditing(true)}>Edit details</button>}
        {isLead && (matter.status === 'open'
          ? <button type="button" className="lw-btn" onClick={() => changeStatus('closed')}>Close matter</button>
          : <button type="button" className="lw-btn lw-btn-primary" onClick={() => changeStatus('open')}>Reopen matter</button>)}
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 880 }}>
        <section className="lw-paper lw-sheet" aria-labelledby="matter-title" style={{ display: 'grid', gap: 18 }}>
          <div>
            <div className="lw-eyebrow">{matter.client.name}</div>
            <h2 className="lw-h1 lw-serif" id="matter-title" style={{ marginTop: 4 }}>{matter.title}</h2>
          </div>
          {matter.status !== 'open' && <p className="lw-meta" style={{ margin: 0 }}>This matter is {matter.status}. Its content can be read but not changed until it is reopened.</p>}
          {matter.on_hold && <HoldNotice />}
          {error && <ErrorNote>{error}</ErrorNote>}
          <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '10px 24px', margin: 0 }}>
            {rows.map(([k, v]) => (<React.Fragment key={k}><dt className="lw-meta" style={{ fontSize: 13 }}>{k}</dt><dd style={{ margin: 0 }}>{v}</dd></React.Fragment>))}
          </dl>
          <div style={{ display: 'flex', gap: 8 }}>
            <Link to="../documents" relative="path" className="lw-btn lw-btn-ghost lw-btn-sm">Documents</Link>
            <Link to="../research" relative="path" className="lw-btn lw-btn-ghost lw-btn-sm">Research</Link>
            <Link to="../members" relative="path" className="lw-btn lw-btn-ghost lw-btn-sm">Members</Link>
          </div>
        </section>
        <p className="lw-meta" style={{ margin: 0 }}>
          Drafting and live assistance for this matter arrive in later releases of EchoMind Law.
        </p>
      </div>
      {editing && <EditMatterDialog matter={matter} onClose={() => setEditing(false)} />}
    </>
  );
}

function EditMatterDialog({ matter, onClose }: { matter: Matter; onClose: () => void }) {
  const [form, setForm] = useState({ title: matter.title, practice_area: matter.practice_area ?? '', jurisdiction: matter.jurisdiction ?? '', sensitivity: matter.sensitivity });
  const [error, setError] = useState('');
  const save = useApiMutation(() => api.patch(`/api/legal/matters/${encodeURIComponent(matter.id)}`, { ...form, version: matter.version }), [keys.matters]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await save.mutateAsync(undefined); onClose(); } catch (err) { setError(err instanceof ApiError ? err.message : 'The changes could not be saved.'); }
  };
  return (
    <Dialog title="Edit matter details" onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <Field label="Title">{(id) => <input id={id} className="lw-input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />}</Field>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <Field label="Practice area">{(id) => <input id={id} className="lw-input" value={form.practice_area} onChange={(e) => setForm({ ...form, practice_area: e.target.value })} />}</Field>
          <Field label="Jurisdiction">{(id) => <input id={id} className="lw-input" value={form.jurisdiction} onChange={(e) => setForm({ ...form, jurisdiction: e.target.value })} />}</Field>
        </div>
        <Field label="Sensitivity">{(id) => (
          <select id={id} className="lw-input" value={form.sensitivity} onChange={(e) => setForm({ ...form, sensitivity: e.target.value as Sensitivity })}>
            <option value="standard">Standard</option><option value="restricted">Restricted</option><option value="highly_restricted">Highly restricted</option>
          </select>)}
        </Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!form.title || save.isPending}>Save</button>
        </div>
      </form>
    </Dialog>
  );
}
