import React, { useState } from 'react';
import { Check, Copy, Plus, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { useChecklists } from '../api/queries';
import type { DdChecklist, DdGovernance, DdItem, DdQuestion } from '../api/types';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

const QUESTION: Record<DdQuestion, string> = {
  change_of_control: 'Change of control', assignment: 'Assignment', termination: 'Termination', governing_law: 'Governing law',
  liability: 'Liability', renewal: 'Renewal and expiry',
};
const GOVERNANCE: Record<DdGovernance, string> = {
  share_capital: 'Shares in the register against approvals and the articles', authority: 'Contracts against the delegation of authority',
  constitution: 'Policies against the articles', ip_chain: 'Founders against IP assignments',
};
const day = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString([], { dateStyle: 'medium' }) : '');

/** Due-diligence checklists (S3.7, D-051): the documents a deal needs, by section, and the questions asked of
 *  them. Versions are fixed; a managing partner other than the author approves exactly the version read. */
export default function Checklists() {
  const q = useChecklists();
  const qc = useQueryClient();
  const [editing, setEditing] = useState<{ from: DdChecklist | null } | null>(null);
  const refresh = () => qc.invalidateQueries({ queryKey: ['dd-checklists'] });
  const can = q.data?.can;
  const families = new Map<string, DdChecklist[]>();
  for (const c of q.data?.checklists ?? []) families.set(c.family_id, [...(families.get(c.family_id) ?? []), c]);
  return (
    <>
      <Toolbar title="Checklists">
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing({ from: null })}><Plus />New checklist</button>}
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>The documents a transaction needs and the questions asked of them. A matter's data room is checked against an approved checklist: what is missing, and what each document says on change of control, assignment, termination, law, liability and renewal. Every finding is preliminary until a reviewer decides.</p>
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>The checklists could not be loaded.</ErrorNote>
            : families.size === 0 ? <EmptyState title="No checklists yet">{can?.manage ? 'Create one: its items are the documents a deal needs.' : 'A lawyer who manages checklists adds them here.'}</EmptyState>
              : Array.from(families.values()).map((versions) => (
                <Family key={versions[0].family_id} versions={versions} canManage={!!can?.manage} canApprove={!!can?.approve} onChange={refresh}
                  onNewVersion={(c) => setEditing({ from: c })} />))}
      </div>
      {editing && <Editor from={editing.from} onClose={() => setEditing(null)} onSaved={async () => { setEditing(null); await refresh(); }} />}
    </>
  );
}

function Family({ versions, canManage, canApprove, onChange, onNewVersion }: {
  versions: DdChecklist[]; canManage: boolean; canApprove: boolean; onChange: () => void; onNewVersion: (c: DdChecklist) => void;
}) {
  const me = useSession();
  const approved = versions.find((v) => v.status === 'approved');
  const pending = versions.find((v) => v.status === 'draft');
  const shown = pending && (canManage || canApprove) ? pending : approved ?? versions[0];
  const [confirming, setConfirming] = useState(false);
  const [read, setRead] = useState(false);
  const [error, setError] = useState('');
  const sections = Array.from(new Set(shown.items.map((i) => i.section)));
  const approve = async () => {
    setError('');
    try { await api.post(`/api/legal/diligence/checklists/${encodeURIComponent(shown.id)}/approve`, { content_sha256: shown.content_sha256 }); setConfirming(false); onChange(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The checklist could not be approved.'); }
  };
  return (
    <section className="lw-paper lw-sheet" aria-label={shown.name}>
      <div className="lw-minutes-h">
        <div>
          <h2 className="lw-h2" style={{ margin: 0 }}>{shown.name}</h2>
          <p className="lw-meta" style={{ margin: '2px 0 0' }}>{[`Version ${shown.version} by ${shown.created_by ?? 'someone'}`, shown.approved_by && `approved by ${shown.approved_by}, ${day(shown.approved_at)}`,
            shown.expected_law && `expected law: ${shown.expected_law}`].filter(Boolean).join(' · ')}</p>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <StatusPill tone={shown.status === 'approved' ? 'verified' : 'attention'}>{shown.status === 'approved' ? 'In use' : 'Draft — not yet approved'}</StatusPill>
          {canManage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => onNewVersion(shown)}><Copy />New version</button>}
          {shown.status === 'draft' && canApprove && shown.created_by !== me.user.username && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setConfirming(true)}><Check />Approve</button>}
        </div>
      </div>
      {shown.status === 'draft' && canApprove && shown.created_by === me.user.username && <p className="lw-meta" style={{ margin: '0 0 6px' }}>You wrote this version, so another managing partner approves it.</p>}
      {(shown.governance ?? []).length > 0 && (
        <p className="lw-meta" style={{ margin: '8px 0 0' }}>{`Governance checks: ${(shown.governance ?? []).map((g) => GOVERNANCE[g]).join('; ')}.`}</p>
      )}
      {sections.map((sec) => (
        <div key={sec}>
          <h3 className="lw-h3" style={{ margin: '12px 0 2px' }}>{sec}</h3>
          <ul className="lw-check-lines">
            {shown.items.filter((i) => i.section === sec).map((i) => (
              <li key={i.id} aria-label={i.title}>
                <b>{i.title}</b>
                <span className="lw-meta">{`Found by: ${i.expect.join(', ')}`}{i.questions.length ? ` · asks: ${i.questions.map((x) => QUESTION[x]).join(', ')}` : ''}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
      {error && <ErrorNote>{error}</ErrorNote>}
      {confirming && (
        <Dialog title={`Approve ${shown.name}, version ${shown.version}?`} onClose={() => setConfirming(false)}
          footer={<><button type="button" className="lw-btn" onClick={() => setConfirming(false)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!read} onClick={approve}><Check />{`Approve version ${shown.version}`}</button></>}>
          <p style={{ margin: 0 }}>Data rooms will be checked against exactly these items and questions. The version approved before it is retired.</p>
          <label className="lw-meta" style={{ display: 'flex', gap: 8, marginTop: 10 }}><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} />I have read every item of this version</label>
        </Dialog>
      )}
    </section>
  );
}

type Row = { id: string; section: string; title: string; expect: string; questions: DdQuestion[] };
const toRow = (i: DdItem): Row => ({ id: i.id, section: i.section, title: i.title, expect: i.expect.join(', '), questions: i.questions });

function Editor({ from, onClose, onSaved }: { from: DdChecklist | null; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(from?.name ?? '');
  const [law, setLaw] = useState(from?.expected_law ?? '');
  const [gov, setGov] = useState<DdGovernance[]>(from?.governance ?? []);
  const [rows, setRows] = useState<Row[]>(from ? from.items.map(toRow) : [{ id: 'item-1', section: 'Corporate', title: '', expect: '', questions: [] }]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (k: number, patch: Partial<Row>) => setRows((r) => r.map((x, i) => (i === k ? { ...x, ...patch } : x)));
  const save = async () => {
    setBusy(true); setError('');
    try {
      await api.post('/api/legal/diligence/checklists', {
        name, expected_law: law.trim() || null, family_id: from?.family_id ?? null, governance: gov,
        items: rows.map((r) => ({ id: r.id.trim(), section: r.section.trim(), title: r.title.trim(), expect: r.expect.split(',').map((x) => x.trim()).filter(Boolean), questions: r.questions })),
      });
      onSaved();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The checklist could not be saved.'); setBusy(false); }
  };
  return (
    <Dialog title={from ? `New version of ${from.name}` : 'New checklist'} onClose={onClose}
      footer={<><button type="button" className="lw-btn" onClick={onClose}>Cancel</button>
        <button type="button" className="lw-btn lw-btn-primary" disabled={busy || !name.trim()} onClick={save}>{busy ? 'Saving…' : 'Save as a draft'}</button></>}>
      <div style={{ display: 'grid', gap: 8 }}>
        <label className="lw-label">Name<input className="lw-input" value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label className="lw-label">Governing law expected (optional)<input className="lw-input" value={law} onChange={(e) => setLaw(e.target.value)} placeholder="England and Wales" /></label>
        <div className="lw-meta" role="group" aria-label="Governance checks" style={{ display: 'grid', gap: 4 }}>
          <span>Governance checks across the company's documents</span>
          {(Object.keys(GOVERNANCE) as DdGovernance[]).map((g) => (
            <label key={g} style={{ display: 'flex', gap: 6, alignItems: 'center' }}><input type="checkbox" checked={gov.includes(g)}
              onChange={(e) => setGov((x) => (e.target.checked ? [...x, g] : x.filter((y) => y !== g)))} />{GOVERNANCE[g]}</label>
          ))}
        </div>
        {rows.map((r, k) => (
          <fieldset key={k} className="lw-minutes-edit" aria-label={`Item ${k + 1}`}>
            <legend className="lw-meta">{`Item ${k + 1}`}</legend>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <label className="lw-label">Section<input className="lw-input lw-input-sm" value={r.section} onChange={(e) => set(k, { section: e.target.value })} /></label>
              <label className="lw-label">Document<input className="lw-input lw-input-sm" value={r.title} onChange={(e) => set(k, { title: e.target.value })} /></label>
              <label className="lw-label">Id<input className="lw-input lw-input-sm" value={r.id} onChange={(e) => set(k, { id: e.target.value })} /></label>
            </div>
            <label className="lw-label">Found by these words in a document's title (comma-separated)<input className="lw-input lw-input-sm" value={r.expect} onChange={(e) => set(k, { expect: e.target.value })} /></label>
            <div className="lw-meta" role="group" aria-label={`Questions for item ${k + 1}`} style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {(Object.keys(QUESTION) as DdQuestion[]).map((qk) => (
                <label key={qk} style={{ display: 'flex', gap: 4, alignItems: 'center' }}><input type="checkbox" checked={r.questions.includes(qk)}
                  onChange={(e) => set(k, { questions: e.target.checked ? [...r.questions, qk] : r.questions.filter((x) => x !== qk) })} />{QUESTION[qk]}</label>
              ))}
            </div>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setRows((x) => x.filter((_, i) => i !== k))}><Trash2 />Remove the item</button>
          </fieldset>
        ))}
        <div><button type="button" className="lw-btn lw-btn-sm" onClick={() => setRows((x) => [...x, { id: `item-${x.length + 1}`, section: x.at(-1)?.section ?? 'General', title: '', expect: '', questions: [] }])}><Plus />Add an item</button></div>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    </Dialog>
  );
}
