import React, { useState } from 'react';
import { Check, FileUp, Plus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { usePlaybooks } from '../api/queries';
import type { Playbook, PlaybookPosition, PlaybookProposal } from '../api/types';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';

const day = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString([], { dateStyle: 'medium' }) : '');

/** Playbooks (S3.2, D-046): the firm's positions by topic — the standard wording and the fallbacks it
 *  accepts — versioned, and approved by a managing partner other than the author. */
export default function Playbooks() {
  const q = usePlaybooks();
  const qc = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState<{ playbook: Playbook; draft?: Partial<PositionDraft> } | null>(null);
  const [importing, setImporting] = useState<Playbook | null>(null);
  const refresh = () => qc.invalidateQueries({ queryKey: ['playbooks'] });
  const can = q.data?.can;
  return (
    <>
      <Toolbar title="Playbooks">
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setCreating(true)}><Plus />New playbook</button>}
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>The firm's positions for negotiating contracts: for each topic, the standard wording and the fallbacks the firm accepts. A matter document can be checked against an approved playbook; the result is advisory, for a lawyer to confirm.</p>
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>The playbooks could not be loaded.</ErrorNote>
            : q.data.playbooks.length === 0 ? <EmptyState title="No playbooks yet">{can?.manage ? 'Create one, then add positions or read them from a precedent.' : 'A lawyer who manages playbooks adds them here.'}</EmptyState>
              : q.data.playbooks.map((pb) => (
                <PlaybookCard key={pb.id} pb={pb} canManage={!!can?.manage} canApprove={!!can?.approve} onChange={refresh}
                  onAdd={(draft) => setEditing({ playbook: pb, draft })} onImport={() => setImporting(pb)} />))}
      </div>
      {creating && <NewPlaybook onClose={() => setCreating(false)} onSaved={async () => { setCreating(false); await refresh(); }} />}
      {editing && <PositionEditor pb={editing.playbook} topics={q.data?.topics ?? {}} initial={editing.draft} onClose={() => setEditing(null)}
        onSaved={async () => { setEditing(null); await refresh(); }} />}
      {importing && <ImportPrecedent pb={importing} topics={q.data?.topics ?? {}} onClose={() => setImporting(null)}
        onUse={(draft) => { setImporting(null); setEditing({ playbook: importing, draft }); }} />}
    </>
  );
}

function PlaybookCard({ pb, canManage, canApprove, onChange, onAdd, onImport }: {
  pb: Playbook; canManage: boolean; canApprove: boolean; onChange: () => void; onAdd: (draft?: Partial<PositionDraft>) => void; onImport: () => void;
}) {
  const topics = Array.from(new Set(pb.positions.map((p) => p.topic)));
  return (
    <section className="lw-paper lw-sheet" aria-label={pb.name}>
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{pb.name}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{[pb.jurisdiction, pb.description].filter(Boolean).join(' · ') || ' '}</p>
        </div>
        {canManage && <div style={{ display: 'flex', gap: 6 }}>
          <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={onImport}><FileUp />Read a precedent</button>
          <button type="button" className="lw-btn lw-btn-sm" onClick={() => onAdd()}><Plus />Add a position</button>
        </div>}
      </div>
      {topics.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>No positions yet.</p> : (
        <ul className="lw-template-versions" aria-label={`Positions in ${pb.name}`}>
          {topics.map((t) => <TopicRow key={t} pb={pb} versions={pb.positions.filter((p) => p.topic === t)} canManage={canManage} canApprove={canApprove}
            onChange={onChange} onNewVersion={(p) => onAdd({ topic: p.topic, title: p.title, standard: p.standard, fallbacks: p.fallbacks.map((f) => ({ text: f.text, note: f.note ?? '' })), guidance: p.guidance ?? '' })} />)}
        </ul>
      )}
    </section>
  );
}

function TopicRow({ pb, versions, canManage, canApprove, onChange, onNewVersion }: {
  pb: Playbook; versions: PlaybookPosition[]; canManage: boolean; canApprove: boolean; onChange: () => void; onNewVersion: (p: PlaybookPosition) => void;
}) {
  const me = useSession();
  const [confirming, setConfirming] = useState<PlaybookPosition | null>(null);
  const [read, setRead] = useState(false);
  const [error, setError] = useState('');
  const current = versions.find((v) => v.status === 'approved');
  const pending = versions.find((v) => v.status === 'draft');
  const shown = current ?? pending ?? versions[0];
  const approve = async (v: PlaybookPosition) => {
    setError(''); setConfirming(null);
    try { await api.post(`/api/legal/playbooks/${encodeURIComponent(pb.id)}/positions/${encodeURIComponent(v.id)}/approve`, { content_sha256: v.content_sha256 }); onChange(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The position could not be approved.'); }
  };
  return (
    <li aria-label={`${shown.topic_label}: ${shown.title}`}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <b>{shown.title}</b><span className="lw-meta">{shown.topic_label}</span>
        {current && <StatusPill tone="verified">{`In use: version ${current.version}`}</StatusPill>}
        {pending && <StatusPill tone="attention">{`Version ${pending.version} waiting for approval`}</StatusPill>}
      </div>
      <dl className="lw-position">
        <dt>Standard</dt><dd className="lw-serif">{shown.standard}</dd>
        {shown.fallbacks.map((f, i) => <React.Fragment key={i}><dt>{`Fallback ${i + 1}`}</dt><dd className="lw-serif">{f.text}{f.note ? <span className="lw-meta">{` — ${f.note}`}</span> : null}</dd></React.Fragment>)}
        {shown.guidance && <><dt>Guidance</dt><dd>{shown.guidance}</dd></>}
      </dl>
      <p className="lw-meta" style={{ margin: '4px 0 0' }}>{`Version ${shown.version} by ${shown.created_by ?? 'someone'}${shown.approved_by ? ` · approved by ${shown.approved_by}, ${day(shown.approved_at)}` : ''}`}</p>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
        {pending && canApprove && pending.created_by !== me.user.username && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => { setRead(false); setConfirming(pending); }}><Check />{`Approve version ${pending.version}`}</button>}
        {pending && canApprove && pending.created_by === me.user.username && <span className="lw-meta">You wrote this version, so another managing partner approves it.</span>}
        {canManage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => onNewVersion(shown)}>New version</button>}
      </div>
      {pending && pending !== shown && <p className="lw-meta" style={{ margin: '6px 0 0' }}>{`Waiting: “${pending.standard}”`}</p>}
      {error && <ErrorNote>{error}</ErrorNote>}
      {confirming && (
        <Dialog title={`Approve ${confirming.title}, version ${confirming.version}?`} onClose={() => setConfirming(null)}
          footer={<><button type="button" className="lw-btn" onClick={() => setConfirming(null)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!read} onClick={() => approve(confirming)}><Check />{`Approve version ${confirming.version}`}</button></>}>
          <div style={{ display: 'grid', gap: 10 }}>
            <p style={{ margin: 0 }}>Documents will be checked against exactly this standard and these fallbacks. The version approved before it is retired.</p>
            <p className="lw-meta" style={{ margin: 0 }}>Content <code className="lw-hash">{confirming.content_sha256}</code></p>
            <label className="lw-check"><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} /> I have reviewed this position and its fallbacks</label>
          </div>
        </Dialog>
      )}
    </li>
  );
}

function NewPlaybook({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState('');
  const [jurisdiction, setJurisdiction] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');
  const save = async () => {
    setError('');
    try { await api.post('/api/legal/playbooks', { name: name.trim(), jurisdiction: jurisdiction.trim() || null, description: description.trim() || null }); onSaved(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The playbook could not be created.'); }
  };
  return (
    <Dialog width={520} title="New playbook" onClose={onClose}
      footer={<><button type="button" className="lw-btn" onClick={onClose}>Cancel</button><button type="button" className="lw-btn lw-btn-primary" disabled={!name.trim()} onClick={save}>Create</button></>}>
      <div style={{ display: 'grid', gap: 10 }}>
        <Field label="Name" hint="The kind of contract and side, e.g. Commercial services (supplier)">{(id) => <input id={id} className="lw-input" value={name} onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="Jurisdiction">{(id) => <input id={id} className="lw-input" value={jurisdiction} onChange={(e) => setJurisdiction(e.target.value)} />}</Field>
        <Field label="Description">{(id) => <textarea id={id} className="lw-input" rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    </Dialog>
  );
}

interface PositionDraft { topic: string; title: string; standard: string; fallbacks: { text: string; note: string }[]; guidance: string }

function PositionEditor({ pb, topics, initial, onClose, onSaved }: { pb: Playbook; topics: Record<string, string>; initial?: Partial<PositionDraft>; onClose: () => void; onSaved: () => void }) {
  const [d, setD] = useState<PositionDraft>({ topic: initial?.topic ?? '', title: initial?.title ?? '', standard: initial?.standard ?? '',
    fallbacks: initial?.fallbacks ?? [], guidance: initial?.guidance ?? '' });
  const [error, setError] = useState('');
  const set = (patch: Partial<PositionDraft>) => setD((x) => ({ ...x, ...patch }));
  const save = async () => {
    setError('');
    try {
      await api.post(`/api/legal/playbooks/${encodeURIComponent(pb.id)}/positions`, {
        topic: d.topic, title: d.title.trim() || null, standard: d.standard, guidance: d.guidance.trim() || null,
        fallbacks: d.fallbacks.filter((f) => f.text.trim()).map((f) => ({ text: f.text, note: f.note.trim() || null })) });
      onSaved();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The position could not be saved.'); }
  };
  return (
    <Dialog width={640} title={`Position in ${pb.name}`} onClose={onClose}
      footer={<><button type="button" className="lw-btn" onClick={onClose}>Cancel</button>
        <button type="button" className="lw-btn lw-btn-primary" disabled={!d.topic || !d.standard.trim()} onClick={save}>Save as a draft version</button></>}>
      <div style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <Field label="Topic">{(id) => <select id={id} className="lw-input" value={d.topic} onChange={(e) => set({ topic: e.target.value, title: d.title || topics[e.target.value] || '' })}>
            <option value="">Choose a topic</option>{Object.entries(topics).map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>}</Field>
          <Field label="Title">{(id) => <input id={id} className="lw-input" value={d.title} onChange={(e) => set({ title: e.target.value })} />}</Field>
        </div>
        <Field label="Standard position">{(id) => <textarea id={id} className="lw-input lw-serif" rows={3} value={d.standard} onChange={(e) => set({ standard: e.target.value })} />}</Field>
        {d.fallbacks.map((f, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 8 }}>
            <Field label={`Fallback ${i + 1}`}>{(id) => <textarea id={id} className="lw-input lw-serif" rows={2} value={f.text}
              onChange={(e) => set({ fallbacks: d.fallbacks.map((x, j) => (j === i ? { ...x, text: e.target.value } : x)) })} />}</Field>
            <Field label={`Note for fallback ${i + 1}`} hint="e.g. who must agree">{(id) => <input id={id} className="lw-input" value={f.note}
              onChange={(e) => set({ fallbacks: d.fallbacks.map((x, j) => (j === i ? { ...x, note: e.target.value } : x)) })} />}</Field>
          </div>
        ))}
        {d.fallbacks.length < 5 && <div><button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => set({ fallbacks: [...d.fallbacks, { text: '', note: '' }] })}><Plus />Add a fallback</button></div>}
        <Field label="Guidance (optional)">{(id) => <textarea id={id} className="lw-input" rows={2} value={d.guidance} onChange={(e) => set({ guidance: e.target.value })} />}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    </Dialog>
  );
}

function ImportPrecedent({ pb, topics, onClose, onUse }: { pb: Playbook; topics: Record<string, string>; onClose: () => void; onUse: (d: Partial<PositionDraft>) => void }) {
  const [found, setFound] = useState<PlaybookProposal[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const read = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true); setError('');
    const form = new FormData();
    form.append('file', file);
    try { setFound((await api.post<{ proposals: PlaybookProposal[] }>(`/api/legal/playbooks/${encodeURIComponent(pb.id)}/import`, form)).proposals); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The precedent could not be read.'); }
    finally { setBusy(false); }
  };
  return (
    <Dialog width={640} title={`Read a precedent into ${pb.name}`} onClose={onClose}>
      <div style={{ display: 'grid', gap: 10 }} aria-busy={busy}>
        <p className="lw-meta" style={{ margin: 0 }}>The firm's precedent is read clause by clause; each clause is proposed under the topic its words suggest. Nothing is saved until you add it as a position, which then waits for approval.</p>
        <Field label="Precedent (Word, PDF or text)">{(id) => <input id={id} type="file" accept=".docx,.pdf,.txt,.md" onChange={read} />}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
        {found && (found.length === 0 ? <p className="lw-meta">No numbered clauses were found in this file.</p> : (
          <ul className="lw-proposals" aria-label="Proposed positions">
            {found.map((p, i) => (
              <li key={i}>
                <div className="lw-meta">{`${p.clause}${p.title ? ` · ${p.title}` : ''} · ${p.topic ? topics[p.topic] : 'no topic suggested'}`}</div>
                <div className="lw-serif">{p.text}</div>
                <button type="button" className="lw-btn lw-btn-sm" onClick={() => onUse({ topic: p.topic ?? '', title: p.title ?? '', standard: p.text.replace(/^\s*[\d.]+\s+/, ''), fallbacks: [] })}>Use as the standard position</button>
              </li>
            ))}
          </ul>
        ))}
      </div>
    </Dialog>
  );
}
